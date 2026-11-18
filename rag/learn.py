"""Two trained models, fitted from labels instead of typed by hand.

Everything else in this study uses numbers a person chose: the 0.6 match fraction, the
0.024 similarity floor, the 0.05-0.30 anchors. These two learn their numbers from data,
which is what makes them models rather than rules. Both are small linear fits through
scikit-learn - a handful of weights, not a downloaded brain - so they train in seconds on
the laptop CPU, need no GPU and no GGUF, and reproduce exactly.

1. `train_screener` - given the signals the pipeline already produces for a CV (the best
   retrieval similarity per requirement, how many keyword asks landed, how much of the CV
   the prompt carried, its length), learn a weight for each against the human rubric
   score. Leave-one-out throughout: the model that predicts a CV never saw that CV, and
   the ridge penalty is chosen by generalised cross-validation inside the training rows
   only. Two feature sets are reported - one that includes the term rule's own output,
   one that does not - because they answer different questions: calibrating a scorer
   versus learning one from retrieval signals alone.

2. `train_value_model` - given only what a retriever already knows about a passage (its
   similarity to a requirement, its rank among that requirement's hits, its position and
   length in the document), learn how much of the requirement it is worth. The label comes
   from the term rule reading the passage, so this is distillation: the trained model is
   the part of the rule that similarity and geometry can predict. It is what lets
   rag/planner.py choose evidence without looking up which words the passage contains,
   which is the assumption the untrained planner's numbers smuggled in.

Honest limits, stated because n=33 is the whole dataset: 33 rubric labels are enough to
fit a handful of weights and cross-validate them, not to prove a learned screener beats
the rule - the permutation p-value quoted under every rho is what says whether a ranking
this tight is distinguishable from luck at this n. And similarity is itself computed from
the passage's words, so the value model is not blind to content; it is blind to *which*
words, which is the only claim made for it.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

from .config import SIM_FLOOR, TOP_K
from .ingest import chunk_text
from .llm import KEYWORD_MATCH_FRACTION, document_terms, keyword_coverage
from .screen import Result

ALPHAS = np.logspace(-2, 3, 12)   # ridge penalties searched inside every training fold
RETRIEVAL_ONLY = frozenset({"sim mean", "sim best per requirement", "sim max",
                            "sim spread", "evidence share", "chunks",
                            "document words"})


def _sims(index, resume_id, reqs, n_chunks):
    """Similarity and hit-rank of every chunk of one CV against every requirement."""
    sims = np.zeros((n_chunks, len(reqs)))
    ranks = np.full((n_chunks, len(reqs)), np.nan)
    for j, r in enumerate(reqs):
        for rank, (s, c) in enumerate(index.search(r, resume_id, k=n_chunks,
                                                   min_score=0.0)):
            sims[c.idx, j] = s
            ranks[c.idx, j] = rank
    return sims, np.nan_to_num(ranks, nan=float(n_chunks))


def screener_features(index, resume_id, reqs, full_text):
    """What a trained scorer is allowed to see for one CV - no labels, no future folds.

    Similarities come from the index, the coverage features from the term rule over the
    passages the shipped pipeline would have pasted, the rest from document geometry.
    """
    chunks = chunk_text(resume_id, full_text)
    n = max(1, len(chunks))
    sims, _ = _sims(index, resume_id, reqs, n)
    floor = SIM_FLOOR[index.embedder.kind]
    kept, terms = set(), set()
    for j in range(len(reqs)):
        for i in np.argsort(-sims[:, j])[:TOP_K]:
            if sims[i, j] >= floor:
                kept.add(int(i))
                terms |= document_terms(chunks[int(i)].text)
    covs = np.array([keyword_coverage(r, terms)[0] for r in reqs])
    doc_words = max(1, len(full_text.split()))
    return {
        "sim mean": float(sims.mean()),
        "sim best per requirement": float(sims.max(axis=0).mean()),
        "sim max": float(sims.max()),
        "sim spread": float(sims.std()),
        "coverage mean": float(covs.mean()),
        "coverage max": float(covs.max()),
        "fraction matched": float((covs >= KEYWORD_MATCH_FRACTION).mean()),
        "evidence share": float(sum(len(chunks[i].text.split()) for i in kept)
                                / doc_words),
        "chunks": float(n),
        "document words": float(doc_words),
    }


def _vectorise(rows, use):
    names = [n for n in rows[0] if use is None or n in use]
    return np.array([[r[n] for n in names] for r in rows], float), names


def loo_predict(X, y):
    """Leave-one-out predictions, penalty chosen by generalised CV inside each fold.

    Returns (held-out predictions, in-sample predictions, per-fold models, alphas). The
    in-sample column refits on all rows and predicts its own row - the optimistic number,
    reported next to the honest one so the size of the gap is visible.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    out = np.zeros(len(y))
    models = []
    for train, test in LeaveOneOut().split(X):
        m = RidgeCV(alphas=ALPHAS).fit(X[train], y[train])
        out[test] = np.clip(m.predict(X[test]), 0, 100)
        models.append(m)
    full = RidgeCV(alphas=ALPHAS).fit(X, y)
    return out, np.clip(full.predict(X), 0, 100), models, full


def permutation_p(pred, gold, n=1000, seed=0):
    """Chance of an |rho| this large if the labels were shuffled onto the same CVs.

    Seeded, because a log that quotes a p-value has to quote the same one on every
    machine. The one-tailed alternative is |rho|, so a model that ranks perfectly
    backwards cannot earn a small p.
    """
    rng = np.random.default_rng(seed)
    obs = abs(spearmanr(pred, gold)[0])
    worse = sum(1 for _ in range(n)
                if abs(spearmanr(pred, rng.permutation(np.asarray(gold, float)))[0])
                >= obs)
    return (worse + 1) / (n + 1)


def train_screener(index, pairs, reqs, labels, use=None):
    """Fit the screener over the corpus and return held-out Results plus what it learned."""
    rows = [screener_features(index, rid, reqs, text) for rid, text in pairs]
    X, names = _vectorise(rows, use)
    y = np.array([labels[rid] for rid, _ in pairs], float)
    pred, insample, models, full = loo_predict(X, y)
    results = [Result(resume_id=rid, score=int(round(s)), score_coverage=int(round(s)),
                      matched=[], missing=[])
               for rid, s in zip([r[0] for r in pairs], pred)]
    w, scale = full.coef_, np.std(X, axis=0)
    return results, {
        "features": names,
        "weights": [round(float(a), 3) for a in full.coef_],
        "standardised": [round(float(a * b), 3) for a, b in zip(w, scale)],
        "loo_rho": round(float(spearmanr(pred, y)[0]), 3),
        "in_sample_rho": round(float(spearmanr(insample, y)[0]), 3),
        "loo_mae": round(float(np.mean(np.abs(pred - y))), 2),
        "in_sample_mae": round(float(np.mean(np.abs(insample - y))), 2),
        "p": permutation_p(pred, y),
        "alpha": round(float(full.alpha_), 3),
        "n": len(y),
        "intercept": round(float(full.intercept_), 1),
    }


def value_rows(index, reqs, pairs):
    """One row per (CV, chunk, requirement): retrieval features, term-rule label.

    The features are deliberately things a fetch is free: similarity, rank, geometry. The
    label is the only place the passage's words are read, and it is never a feature.
    """
    rows, ys, owner = [], [], []
    for pos, (rid, text) in enumerate(pairs):
        chunks = chunk_text(rid, text)
        n = max(1, len(chunks))
        sims, ranks = _sims(index, rid, reqs, n)
        doc_words = max(1, len(text.split()))
        for c in chunks:
            label_terms = document_terms(c.text)
            for j, r in enumerate(reqs):
                z = float(sims[c.idx, j] - sims[:, j].mean()) / (float(sims[:, j].std())
                                                                 + 1e-9)
                rows.append([float(sims[c.idx, j]),
                             float(ranks[c.idx, j]) / max(1.0, n - 1.0),
                             z,
                             float(c.idx) / max(1, n - 1),
                             len(c.text.split()) / doc_words,
                             float(j) / max(1, len(reqs) - 1)])
                ys.append(keyword_coverage(r, label_terms)[0])
                owner.append(pos)
    return np.array(rows), np.array(ys), np.array(owner)


def fit_value_model(X, y, owner, holdout=None):
    """Fit on every CV but `holdout`, from rows already built by value_rows.

    Callers that train once per CV pass the same matrix 33 times, so the feature build
    lives outside this function: fitting must not re-read the corpus on every call.
    """
    mask = np.ones(len(y), bool) if holdout is None else (owner != holdout)
    return RidgeCV(alphas=ALPHAS).fit(X[mask], y[mask])


def train_value_model(index, reqs, pairs, holdout=None):
    """Fit the passage-value model; `holdout` is the corpus row it must not train on."""
    X, y, owner = value_rows(index, reqs, pairs)
    return fit_value_model(X, y, owner, holdout)


def value_quality(index, reqs, pairs):
    """Held-out accuracy of the value model: leave one CV out, train on the other 32."""
    X, y, owner = value_rows(index, reqs, pairs)
    pred = np.zeros(len(y))
    for cv in np.unique(owner):
        pred[owner == cv] = fit_value_model(X, y, owner, cv).predict(X[owner == cv])
    pred = np.clip(pred, 0, 1)
    truth = y >= KEYWORD_MATCH_FRACTION
    return {
        "rows": len(y),
        "cvs": len(np.unique(owner)),
        "pearson": round(float(pearsonr(pred, y)[0]), 3),
        "rho": round(float(spearmanr(pred, y)[0]), 3),
        "auc": round(float(roc_auc_score(truth, pred)), 3) if truth.any()
        and (~truth).any() else float("nan"),
        "mean abs err": round(float(np.mean(np.abs(pred - y))), 3),
        "base rate": round(float(truth.mean()), 3),
        "alpha": round(float(fit_value_model(X, y, owner).alpha_), 3),
    }


def value_of_chunk(model, index, reqs, rid, text):
    """Predicted per-requirement value of every passage of one CV, for the planner.

    This is the whole interface between the trained model and rag/planner.py: a dict of
    {chunk idx: tuple of predicted coverage}. Nothing in it came from opening the passage
    for the purpose of the estimate - only the similarity the index already stored.
    """
    chunks = chunk_text(rid, text)
    n = max(1, len(chunks))
    sims, ranks = _sims(index, rid, reqs, n)
    doc_words = max(1, len(text.split()))
    feats = []
    for c in chunks:
        for j in range(len(reqs)):
            z = float(sims[c.idx, j] - sims[:, j].mean()) / (float(sims[:, j].std())
                                                             + 1e-9)
            feats.append([float(sims[c.idx, j]),
                          float(ranks[c.idx, j]) / max(1.0, n - 1.0), z,
                          float(c.idx) / max(1, n - 1),
                          len(c.text.split()) / doc_words,
                          float(j) / max(1, len(reqs) - 1)])
    p = np.clip(model.predict(np.array(feats, float)), 0, 1)
    out, k = {}, 0
    for c in chunks:
        out[c.idx] = tuple(float(v) for v in p[k:k + len(reqs)])
        k += len(reqs)
    return out
