"""Are the ranking numbers significant, or are they 33 candidates?

`README.md` quotes Spearman rho for each embedder and compares the two by eyeballing whether
their Fisher-z intervals overlap. This replaces that with three measurements, none of which
need the LLM: a permutation p-value, a bootstrap interval (Fisher-z assumes continuous,
normally-approximate rho, and these scorers emit between 4 and 24 distinct values across 33
candidates), and a leave-one-out range, because on a near-constant scorer one CV can carry the
whole correlation.

Paired comparisons resample *candidates*, not runs: embedder A and embedder B scored the same
33 resumes, so the difference is a paired quantity and the two single-run intervals were never
the right test for it.

    python significance.py                  # the committed artifacts, ~15 s
    python significance.py --draws 20000    # default
    python significance.py --data dir       # a reproduction's outputs, under these filenames
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

from evaluate import rank_metrics

SEED = 7  # same seed the dataset generator uses


def _rho_rows(x, y):
    """Row-wise Spearman on (draws, n) rank matrices; ties get average ranks."""
    rx = rankdata(x, axis=1)
    ry = rankdata(y, axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    out = np.full(len(x), np.nan)
    ok = denom > 0
    out[ok] = (rx * ry).sum(axis=1)[ok] / denom[ok]
    return out


def permutation_p(score, gold, draws, rng):
    """Two-sided: how often a label permutation ranks at least this well.

    Permuting the candidate labels breaks any real association while preserving both
    marginals, which is the null a screening tool is actually being asked about.
    """
    obs = abs(spearmanr(score, gold)[0])
    keys = rng.random(size=(draws, len(gold)))
    perm = _rho_rows(np.tile(score, (draws, 1)), gold[np.argsort(keys, axis=1)])
    return float((1 + np.sum(np.abs(perm) >= obs - 1e-12)) / (draws + 1))


def bootstrap_ci(score, gold, draws, rng):
    """Percentile interval over candidate resamples; degenerate resamples have no rank."""
    idx = rng.integers(0, len(score), size=(draws, len(score)))
    r = _rho_rows(score[idx], gold[idx])
    nans = int(np.isnan(r).sum())
    r = r[~np.isnan(r)]
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))), nans


def loo_range(score, gold):
    r = [rank_metrics(np.delete(score, i), np.delete(gold, i))[0] for i in range(len(score))]
    r = np.array([x for x in r if not np.isnan(x)])
    return float(r.min()), float(r.max())


def fisher_z_ci(rho, n):
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def paired_diff(a, b, gold, draws, rng):
    """Resample candidates and re-rank both scorers on the same resample."""
    idx = rng.integers(0, len(gold), size=(draws, len(gold)))
    d = _rho_rows(a[idx], gold[idx]) - _rho_rows(b[idx], gold[idx])
    nans = int(np.isnan(d).sum())
    d = d[~np.isnan(d)]
    return {
        "obs": float(rank_metrics(a, gold)[0] - rank_metrics(b, gold)[0]),
        "ci": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
        "p_greater": float(np.mean(d > 0)),
        "p_two_sided": float(2 * min(np.mean(d > 0), np.mean(d < 0))),
        "degenerate": nans,
    }


def fmt_p(p, draws, null_word="no permutation beat it"):
    """A resampling p cannot go below 1/(draws+1); say so instead of printing 0.0000."""
    floor = 1.0 / (draws + 1)
    if p <= floor + 1e-12:
        return f"p<{2 * floor:.1e} ({null_word} in {draws})"
    return f"p={p:.4f}" if p >= 0.001 else f"p={p:.1e}"


def describe(label, score, gold, draws, rng):
    rho, tau = rank_metrics(score, gold)
    if np.isnan(rho):
        print(f"{label:<34} rho undefined - {len(set(score.tolist()))} distinct score(s) "
              f"across {len(score)} candidates, no rank to test")
        return None
    boot, nans = bootstrap_ci(score, gold, draws, rng)
    p = permutation_p(score, gold, draws, rng)
    lo, hi = loo_range(score, gold)
    fz = fisher_z_ci(rho, len(gold))
    print(f"{label:<34} rho {rho:+.3f} tau {tau:+.3f} | permutation {fmt_p(p, draws)}")
    print(f"{'':<34} bootstrap 95% [{boot[0]:+.3f}, {boot[1]:+.3f}]  "
          f"(Fisher-z [{fz[0]:+.3f}, {fz[1]:+.3f}]; {nans}/{draws} resamples had no rank)")
    print(f"{'':<34} leave-one-out rho spans [{lo:+.3f}, {hi:+.3f}] "
          f"-> dropping one CV moves it by up to {max(abs(rho - lo), abs(rho - hi)):.3f}")
    return rho


def load(path, mode):
    d = pd.read_csv(path)
    d = d[d["mode"] == mode].sort_values("resume_id")
    return d


def selfcheck(draws=200, seed=SEED):
    """The vectorised tie-aware rho has to equal scipy's, or every number here is bogus."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    checked = 0
    for _ in range(draws):
        x = rng.integers(0, 6, 33).astype(float)   # heavy ties, like a 4-value scorer
        y = rng.integers(0, 9, 33).astype(float)
        if len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
            continue
        checked += 1
        worst = max(worst, abs(_rho_rows(x[None, :], y[None, :])[0] - spearmanr(x, y)[0]))
    if worst > 1e-12:
        raise SystemExit(f"rho implementation disagrees with scipy by {worst:.2e}")
    print(f"self-check: vectorised rho == scipy spearmanr on {checked} tie-heavy "
          f"vectors (max abs error {worst:.1e})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--data", default=".")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    root = Path(args.data)
    draws = args.draws

    runs = {
        "rag, tfidf (LLM)": root / "per_resume_v2.csv",
        "rag, dense (LLM)": root / "per_resume_v2_dense.csv",
        "rag, tfidf (no LLM)": root / "per_resume_v2_fallback.csv",
        "rag, dense (no LLM)": root / "per_resume_v2_fallback_dense.csv",
        "full, tfidf (LLM)": root / "per_resume_v2.csv",
    }
    print(f"seed={SEED} draws={draws} n=33 candidates per run | "
          f"Spearman recomputed with average ties, cross-checked against scipy")
    selfcheck()
    print(f"correlation with the rubric label (human_score)\n")

    scored = {}
    for label, path in runs.items():
        d = load(path, label.split(",")[0])
        gold = d["human"].to_numpy(float)
        rho = describe(f"{label:<20} score", d["score"].to_numpy(float), gold, draws, rng)
        if rho is not None:
            scored[label] = (d, rho)
        cov = d["score_coverage"].to_numpy(float)
        print()
        if len(set(cov.tolist())) > 1:
            describe(f"{label:<20} coverage", cov, gold, draws, rng)
        print()

    print("\npaired comparisons (same 33 candidates, resampled together)")
    print("a 'no LLM' row scores requirements by anchored max similarity and never calls a"
          "\nmodel; its embedder is named in its own label, so mixed pairs are labelled too")
    ids = {k: set(v[0]["resume_id"]) for k, v in scored.items()}
    pairs = [("rag, tfidf (LLM)", "rag, dense (LLM)", "same scorer, other retriever"),
             ("rag, tfidf (LLM)", "rag, tfidf (no LLM)", "LLM vs similarity, tfidf"),
             ("rag, dense (LLM)", "rag, dense (no LLM)", "LLM vs similarity, dense"),
             ("rag, dense (LLM)", "rag, tfidf (no LLM)", "MIXED retriever, quoted with care")]
    for a_name, b_name, why in pairs:
        if ids[a_name] != ids[b_name]:
            print(f"{a_name} vs {b_name}: different candidate sets, skipped")
            continue
        a = scored[a_name][0].sort_values("resume_id")
        b = scored[b_name][0].sort_values("resume_id")
        res = paired_diff(a["score"].to_numpy(float), b["score"].to_numpy(float),
                          a["human"].to_numpy(float), draws, rng)
        print(f"\n{a_name} - {b_name}  [{why}]")
        print(f"  drho {res['obs']:+.3f}, bootstrap 95% [{res['ci'][0]:+.3f}, "
              f"{res['ci'][1]:+.3f}], P(first ranks better)={res['p_greater']:.3f}, "
              f"bootstrap two-sided {fmt_p(res['p_two_sided'], draws, 'no resample reversed the sign')}")
        ea = np.abs(a["score"].to_numpy(float) - a["human"].to_numpy(float))
        eb = np.abs(b["score"].to_numpy(float) - b["human"].to_numpy(float))
        w = wilcoxon(ea, eb)
        wp = f"{w.pvalue:.4f}" if w.pvalue >= 1e-4 else f"{w.pvalue:.1e}"
        print(f"  absolute error: mean {ea.mean():.2f} vs {eb.mean():.2f}, first lower on "
              f"{int((ea < eb).sum())} candidates / second on {int((eb < ea).sum())} / "
              f"{int((ea == eb).sum())} equal, Wilcoxon p={wp} (stat {w.statistic:.1f})")

    print("What this does and does not license:")
    print(" - A permutation p-value below 0.05 means the ordering is unlikely to be a lucky "
          "shuffle of these labels; it says nothing about real hiring outcomes.")
    print(" - P(first ranks better) near 0.5 with an interval across zero is the honest form "
          "of 'the embedders are indistinguishable here'.")
    print(" - The paired two-sided p is a bootstrap tail-area (2 x the smaller tail), not a "
          "randomisation test of the difference; the interval is the number to read.")
    print(" - Wilcoxon compares the two scorers candidate by candidate on absolute error. It "
          "asks a different question from rho - accuracy, not ordering - and can disagree.")
    print(" - The 'no LLM' rows are not starved inputs: RetrievalOnlyLLM scores every "
          "requirement by its best-matching chunk, while the RAG prompt carries only the "
          "budgeted passages. Where the heuristic matches the LLM, that is the result.")
    print(" - Repeating a run at temperature=0 is not a resample: the pairs in ARTIFACTS.md "
          "agree exactly, so they add no width and no significance to anything below.")


if __name__ == "__main__":
    main()
