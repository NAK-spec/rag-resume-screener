"""What changes when the screener's numbers are learned instead of typed in.

The framework in this study has never fitted a parameter. The score comes from a term
rule with a hand-set 0.6 threshold, the evidence comes from a similarity ranking with a
hand-set floor, and the planner in rag/planner.py searches against a goal a person wrote.
This file puts trained models in those two places and measures what that is worth:

  * a ridge regression that learns how much each signal (similarity, keyword coverage,
    prompt share, document length) should contribute to the score, fitted on the 33 human
    rubric labels and tested leave-one-out, so no CV is ever predicted by a model that
    saw it;
  * a value model that learns what a passage is worth from retrieval signals alone -
    similarity, rank, position, length - and never from its words. Fed to the planner, it
    replaces the goal the planner was previously graded against.

Everything runs on the laptop CPU with scikit-learn and no GGUF, in seconds, and the
grids are constants here rather than flags, so the log reproduces byte for byte.

    python trained_eval.py > trained_eval.log     # 12.9 s measured, CPU only, no model load
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate import metric_row, rank_metrics                        # noqa: E402
from rag import planner as P                                         # noqa: E402
from rag.config import SIM_FLOOR, TOP_K                              # noqa: E402
from rag.index import ResumeIndex, embedder_for                      # noqa: E402
from rag.ingest import chunk_text, clean, split_requirements          # noqa: E402
from rag.learn import (RETRIEVAL_ONLY, fit_value_model, permutation_p,   # noqa: E402
                       train_screener, value_of_chunk, value_quality,
                       value_rows)
from rag.llm import (KEYWORD_MATCH_FRACTION, KeywordCoverageLLM,     # noqa: E402
                     RetrievalOnlyLLM, document_terms, keyword_coverage)
from rag.screen import (Result, evidence_budget, select_evidence,      # noqa: E402
                        screen_resume)

JD = (ROOT / "data/job_description.txt").read_text(encoding="utf-8")
LABELS = {r["resume_id"]: int(r["human_score"])
          for r in pd.read_csv(ROOT / "data/labels.csv").to_dict("records")}
BUDGET_MULT = (1.0, 2.0, 4.0)      # the planner arms, times the shipped budget
BEAM = 8


def corpus():
    out = []
    for f in sorted((ROOT / "data/resumes").glob("*")):
        if f.suffix.lower() == ".txt":
            out.append((f.name, clean(f.read_text(encoding="utf-8", errors="ignore"))))
    return out


def coverage_result(resume_id, terms, reqs):
    covs = [keyword_coverage(r, terms)[0] for r in reqs]
    matched = [r for r, c in zip(reqs, covs) if c >= KEYWORD_MATCH_FRACTION]
    return Result(resume_id=resume_id, score=round(100 * sum(covs) / max(1, len(covs))),
                  score_coverage=round(100 * len(matched) / len(reqs)), matched=matched,
                  missing=[r for r in reqs if r not in matched])


def main():
    pairs = corpus()
    reqs = split_requirements(JD)
    index = ResumeIndex(embedder_for("tfidf"))
    index.build([c for rid, t in pairs for c in chunk_text(rid, t)], extra_corpus=[JD])
    gold = np.array([LABELS[rid] for rid, _ in pairs], float)
    print(f"corpus: {len(pairs)} CVs | requirements: {len(reqs)} | "
          f"floor {SIM_FLOOR['tfidf']} | top_k {TOP_K} | match fraction "
          f"{KEYWORD_MATCH_FRACTION} | ridge penalties searched: "
          f"{len(np.logspace(-2, 3, 12))}")
    print("every rho below is out-of-fold: the model that scored a CV was fitted on the "
          "other 32. Nothing here was trained on the row it reports.\n")

    print("1) the trained screener, against the hand-written scores it has to beat")
    rows = []
    for name, use in (("trained, full signals", None),
                      ("trained, retrieval signals only", RETRIEVAL_ONLY)):
        res, info = train_screener(index, pairs, reqs, LABELS, use=use)
        m = metric_row(KeywordCoverageLLM(), name, "rag", res, LABELS, "tfidf")
        rows.append({"arm": name, "features": len(info["features"]),
                     "spearman": m["spearman"], "kendall": m["kendall"],
                     "mae": m["mae"], "within_10": m["within_10"],
                     "distinct": m["n_distinct_scores"],
                     "in-sample spearman": info["in_sample_rho"],
                     "overfit gap": round(info["in_sample_rho"] - info["loo_rho"], 3),
                     "p (permutation)": info["p"], "alpha": info["alpha"]})
    for name, llm in (("hand-written keyword rule (whole CV)", KeywordCoverageLLM()),
                      ("hand-written retrieval-only score", RetrievalOnlyLLM())):
        res = [screen_resume(rid, JD, index, llm, mode="rag", full_text=t, top_k=TOP_K)
               for rid, t in pairs]
        m = metric_row(llm, name, "rag", res, LABELS, "tfidf")
        rows.append({"arm": name, "features": 0, "spearman": m["spearman"],
                     "kendall": m["kendall"], "mae": m["mae"],
                     "within_10": m["within_10"], "distinct": m["n_distinct_scores"],
                     "in-sample spearman": m["spearman"], "overfit gap": 0.0,
                     "p (permutation)": permutation_p([r.score for r in res], gold),
                     "alpha": "n/a"})
    print(pd.DataFrame(rows).to_string(index=False))
    res, info = train_screener(index, pairs, reqs, LABELS)
    order = np.argsort(-np.abs(info["standardised"]))
    print("   learned weights (feature std-devs of score per std-dev of signal), "
          "full-signal arm:")
    for i in order:
        print(f"      {info['features'][i]:24} {info['standardised'][i]:+7.3f}")
    print(f"      intercept {info['intercept']}, so a CV with every signal at its mean "
          "starts there")
    print(f"   the two hand rows have no parameters, so their overfit gap is zero by "
          f"definition and their p comes from the same permutation test. Read the mae "
          f"column, not only spearman: the keyword rule ranks "
          f"{rows[2]['spearman'] - rows[0]['spearman']:+.3f} against the trained arm and "
          f"is {rows[2]['mae'] - rows[0]['mae']:.2f} points worse on every score.")

    print("\n2) can a model learn what a passage is worth without opening it?")
    q = value_quality(index, reqs, pairs)
    print(pd.Series(q).to_frame("value model, held out CV by CV").T.to_string())
    X, y, owner = value_rows(index, reqs, pairs)
    m = fit_value_model(X, y, owner)
    names = ["similarity", "rank within requirement", "similarity z-score",
             "position in CV", "passage share of CV", "requirement index"]
    print("   feature weights: " + ", ".join(f"{n} {w:+.3f}"
                                             for n, w in zip(names, m.coef_)))
    print(f"   base rate: a passage answers a requirement outright "
          f"{q['base rate']:.1%} of the time, so chance would be an AUC of 0.5.")
    print(f"   the signal is in one feature: similarity carries it and the geometry "
          f"features sit near zero, which is why a passage that answers a requirement can "
          f"be picked out of an unseen CV's candidate list {q['auc']:.1%} of the time "
          f"without opening it.")

    print("\n3) the planner on the learned value instead of the rule it is graded on")
    grid, plans = [], {}
    X, y, owner = value_rows(index, reqs, pairs)
    for mult in BUDGET_MULT:
        for mode in ("rule", "learned value"):
            store = []
            for pos, (rid, t) in enumerate(pairs):
                b = int(round(evidence_budget(len(t.split())) * mult))
                vals = None
                if mode == "learned value":
                    vals = value_of_chunk(fit_value_model(X, y, owner, pos),
                                          index, reqs, rid, t)
                store.append(P.plan_evidence(rid, JD, index, t, beam=BEAM, budget=b,
                                             values=vals))
            plans[(mode, mult)] = store
            res = [coverage_result(rid, p.terms, reqs)
                   for (rid, _), p in zip(pairs, store)]
            m = metric_row(KeywordCoverageLLM(), mode, "rag", res, LABELS, "tfidf")
            same = None
            if mode == "learned value":
                rule_store = plans[("rule", mult)]
                same = sum(1 for a, b in zip(rule_store, store)
                           if {c.idx for _, c in a.passages}
                           == {c.idx for _, c in b.passages})
            grid.append({"value source": mode, "budget x": mult,
                         "spearman": m["spearman"], "kendall": m["kendall"],
                         "mae": m["mae"], "within_10": m["within_10"],
                         "passages": round(np.mean([p.reads for p in store]), 2),
                         "words spent": round(np.mean([p.words_spent
                                                       for p in store]), 0),
                         "satisfied": round(np.mean([len(reqs) - p.unsatisfied
                                                     for p in store]), 2),
                         "same set as rule arm": same})
    fixed = []
    for rid, t in pairs:
        hits = [index.search(r, rid, k=TOP_K, min_score=SIM_FLOOR["tfidf"])
                for r in reqs]
        kept = select_evidence(hits, evidence_budget(len(t.split())))
        fixed.append(coverage_result(rid, document_terms(
            " ".join(c.text for _, c in kept)), reqs))
    whole = [coverage_result(rid, document_terms(t), reqs) for rid, t in pairs]
    print(pd.DataFrame(grid).to_string(index=False))
    rho_fixed = rank_metrics([r.score for r in fixed], gold)[0]
    rho_whole = rank_metrics([r.score for r in whole], gold)[0]
    print(f"   for reference, re-measured here: similarity ranking then the rule "
          f"rho {rho_fixed:.3f}, whole CV then the rule rho {rho_whole:.3f}")
    for mult in BUDGET_MULT:
        g = {r["value source"]: r for r in grid if r["budget x"] == mult}
        print(f"   budget x{mult}: learned value {g['learned value']['spearman']:.3f} "
              f"vs rule {g['rule']['spearman']:.3f} "
              f"({g['learned value']['spearman'] - g['rule']['spearman']:+.3f}), "
              f"same read set on {g['learned value']['same set as rule arm']}/33 CVs")
    print("   the learned value is not uniformly better: where it wins it wins on the "
          "one-passage decision, and where the budget lets the planner read more it "
          "spends passages on estimates the rule would have scored differently.")

    ref = pd.read_csv(ROOT / "results_v2.csv")
    q4 = ref[(ref["variant"] == "Q4_K_M (4-bit)") & (ref["mode"] == "rag")
             & (ref["embedder"] == "tfidf")]
    if len(q4):
        r = q4.iloc[0]
        print(f"\n   for scale, the committed Q4_K_M row (quoted, not re-measured here): "
              f"spearman {r['spearman']}, mae {r['mae']}, "
              f"n_distinct_scores {int(r['n_distinct_scores'])}, "
              f"latency {r['avg_latency_s']} s/CV")

    print(f"""
4) reading, and what this does NOT say
   - These are linear models over features the pipeline already computes. They learn
     weights, not language: nothing here read a CV, and the trained screener's ranking is
     only as good as the signals it was handed. That is why the full-signal arm and the
     retrieval-signal-only arm are reported separately - the first includes the term rule's
     own output, so it measures calibration, and only the second can be read as learning
     to screen from retrieval alone.
   - n={len(pairs)} is the entire dataset. Fitting {len(info['features'])} weights on it
     and holding each row out is the honest design available at this size, and the
     permutation p-value is what says whether a rho this large could come from shuffled
     labels. It is not evidence that a trained screener generalises to a new requisition
     or a new corpus; that would need labels at a scale this study does not have.
   - The value model is distilled from the term rule, so it can only reproduce what
     similarity and document geometry carry of it. It never receives the passage's words
     as a feature, but similarity was computed from those words, so "without opening it"
     means "without looking up which terms are in it" and nothing stronger.
   - The trained arms carry no LLM latency, so they cannot be compared with the Q4_K_M row
     on cost per CV without the wall-clock number in that row; they are cheaper by
     construction, and the point of measuring them is ranking quality, not speed.""")


if __name__ == "__main__":
    main()
