"""Does a planner pick better evidence than the retriever's ranking does?

The study's scoring arms so far differ only in *who reads the evidence*: a quantized LLM
(results_v2.csv), a chunk-similarity fallback, and a keyword-coverage rule
(keyword_coverage_eval.log). In all three, the passages themselves are chosen by
`select_evidence`: top-k per requirement by cosine similarity, de-duplicated, truncated to
a word budget. That choice has never been tested, because nothing in it depends on what
the passages say.

This file replaces it with a planner (rag/planner.py): an explicit state, an action model
with preconditions and effects, a goal, and a beam search that decides the order of its
own steps. It is scored with the *same* keyword-coverage rule as the keyword arm, on the
*same* 33 CVs and rubric labels, with no LLM and no torch, in seconds. Because the rule is
shared, every number here is a measure of one thing: which passages the budget went on.

    python planning_eval.py > planning_eval.log   # 17.5 s measured, no model, no GPU

The grids are fixed constants in this file, not command-line flags, so the log reproduces
byte for byte and tests/test_pipeline.py can re-play the planner and check its rows
against it. The beam widths (1 / 8 / 24) are the search ablation: width 1 is pure greedy.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate import metric_row, rank_metrics                       # noqa: E402
from rag import planner as P                                        # noqa: E402
from rag.config import SIM_FLOOR, TOP_K                             # noqa: E402
from rag.index import ResumeIndex, embedder_for                     # noqa: E402
from rag.ingest import chunk_text, clean, split_requirements         # noqa: E402
from rag.llm import (KEYWORD_MATCH_FRACTION, KeywordCoverageLLM,    # noqa: E402
                     RetrievalOnlyLLM, document_terms, keyword_coverage)
from rag.screen import (Result, evidence_budget, select_evidence,   # noqa: E402
                        screen_resume)

JD = (ROOT / "data/job_description.txt").read_text(encoding="utf-8")
LABELS = {r["resume_id"]: int(r["human_score"])
          for r in pd.read_csv(ROOT / "data/labels.csv").to_dict("records")}
BUDGET_MULT = (0.5, 1.0, 2.0, 4.0)   # times the shipped evidence budget
BEAMS = (1, 8, 24)                   # search ablation; width 1 is pure greedy


def corpus():
    out = []
    for f in sorted((ROOT / "data/resumes").glob("*")):
        if f.suffix.lower() == ".txt":
            out.append((f.name, clean(f.read_text(encoding="utf-8", errors="ignore"))))
    return out


def klass(resume_id):
    for k in ("strong", "medium", "weak", "hand"):
        if k in resume_id.lower():
            return k
    return "other"


def coverage_result(resume_id, terms, reqs):
    """Score a term set with the shipped rule and wrap it as the shared Result record,
    so metric_row and rank_metrics read a plan and a model run the same way."""
    covs = [keyword_coverage(r, terms)[0] for r in reqs]
    matched = [r for r, c in zip(reqs, covs) if c >= KEYWORD_MATCH_FRACTION]
    return Result(resume_id=resume_id, score=round(100 * sum(covs) / max(1, len(covs))),
                  score_coverage=round(100 * len(matched) / len(reqs)), matched=matched,
                  missing=[r for r in reqs if r not in matched])


def read_set(plan):
    return {c.idx for _, c in plan.passages}


def main():
    pairs = corpus()
    reqs = split_requirements(JD)
    index = ResumeIndex(embedder_for("tfidf"))
    chunks = [c for rid, t in pairs for c in chunk_text(rid, t)]
    index.build(chunks, extra_corpus=[JD])
    budgets = [evidence_budget(len(t.split())) for _, t in pairs]
    print(f"corpus: {len(pairs)} CVs | requirements: {len(reqs)} | chunks: {len(chunks)} "
          f"({len(chunks) / len(pairs):.1f} per CV) | floor {SIM_FLOOR['tfidf']} | "
          f"top_k {TOP_K} | match fraction {KEYWORD_MATCH_FRACTION}")
    print(f"budget: the shipped evidence_budget(resume words), mean {round(np.mean(budgets))}"
          f" of a mean {round(np.mean([len(t.split()) for _, t in pairs]))}-word CV, "
          f"range {min(budgets)}-{max(budgets)}\n")
    gold = [LABELS[rid] for rid, _ in pairs]

    print("1) the three policies that already exist, before any planning")
    rows = []
    for name, llm in (("Retrieval-only (chunk similarity)", RetrievalOnlyLLM()),
                      ("Keyword coverage (whole document)", KeywordCoverageLLM())):
        res = [screen_resume(rid, JD, index, llm, mode="rag", full_text=t, top_k=TOP_K)
               for rid, t in pairs]
        rows.append({"policy": name, **metric_row(llm, name, "rag", res, LABELS, "tfidf")})
    fixed = []
    for rid, t in pairs:
        hits = [index.search(r, rid, k=TOP_K, min_score=SIM_FLOOR["tfidf"]) for r in reqs]
        kept = select_evidence(hits, evidence_budget(len(t.split())))
        fixed.append(coverage_result(rid, document_terms(
            " ".join(c.text for _, c in kept)), reqs))
    whole = [coverage_result(rid, document_terms(t), reqs) for rid, t in pairs]
    rows.append({"policy": "Fixed top-k passages (select_evidence)",
                 **metric_row(KeywordCoverageLLM(), "fixed", "rag", fixed, LABELS, "tfidf")})
    cols = ["policy", "n", "spearman", "kendall", "n_distinct_scores", "modal_share",
            "mae", "within_10"]
    print(pd.DataFrame(rows)[cols].to_string(index=False))
    print("   no timing column here by design: no model ran, so it would measure Python "
          "rather than a method (the same rule keyword_coverage_eval.log follows).")
    print(f"   satisfied requirements: fixed top-k "
          f"{np.mean([len(r.matched) for r in fixed]):.2f} of {len(reqs)}, whole document "
          f"{np.mean([len(r.matched) for r in whole]):.2f}")

    print("\n2) the planner: budget x beam width, same coverage rule, same 33 CVs")
    plans, grid = {}, []
    for beam in BEAMS:
        for mult in BUDGET_MULT:
            store = []
            for rid, t in pairs:
                b = int(round(evidence_budget(len(t.split())) * mult))
                store.append(P.plan_evidence(rid, JD, index, t, beam=beam, budget=b))
            plans[(beam, mult)] = store
            res = [coverage_result(rid, p.terms, reqs)
                   for (rid, _), p in zip(pairs, store)]
            m = metric_row(KeywordCoverageLLM(), f"beam{beam}", "rag", res, LABELS, "tfidf")
            grid.append({
                "beam": beam, "budget x": mult, "spearman": m["spearman"],
                "kendall": m["kendall"], "mae": m["mae"], "within_10": m["within_10"],
                "distinct": m["n_distinct_scores"],
                "mean score": round(np.mean([p.score for p in store]), 1),
                "passages": round(np.mean([p.reads for p in store]), 2),
                "words spent": round(np.mean([p.words_spent for p in store]), 0),
                "budget": round(np.mean([p.budget for p in store]), 0),
                "satisfied": round(np.mean([len(reqs) - p.unsatisfied
                                            for p in store]), 2),
                "nodes": round(np.mean([p.nodes_expanded for p in store]), 0),
                "re-asks": round(np.mean([p.relax_rounds for p in store]), 2)})
    print(pd.DataFrame(grid).to_string(index=False))
    print("   satisfied = mean requirements at or above the match fraction; nodes = search "
          "states expanded per CV, the only cost the search itself incurs.")

    print("\n3) what search buys over greedy at the same budget")
    for mult in BUDGET_MULT:
        rho = {b: rank_metrics([p.score for p in plans[(b, mult)]], gold)[0]
               for b in BEAMS}
        d18 = sum(1 for a, b in zip(plans[(1, mult)], plans[(8, mult)])
                  if read_set(a) != read_set(b))
        d824 = sum(1 for a, b in zip(plans[(8, mult)], plans[(24, mult)])
                   if read_set(a) != read_set(b))
        print(f"   budget x{mult}: greedy rho {rho[1]:.3f} | beam 8 {rho[8]:.3f} "
              f"({d18}/{len(pairs)} different read sets) | beam 24 {rho[24]:.3f} "
              f"({d824}/{len(pairs)} different from beam 8)")

    print("\n3b) the choice that actually moves the numbers: rank states by the goal "
          "count or by the score")
    fixed_by_id = {r.resume_id: r.score for r in fixed}
    orders, by_order = [], {}
    for mult in BUDGET_MULT:
        for order in ("goal", "coverage"):
            store = [P.plan_evidence(rid, JD, index, t, beam=8, order=order,
                                     budget=int(round(evidence_budget(len(t.split()))
                                                      * mult)))
                     for rid, t in pairs]
            by_order.setdefault(order, {})[mult] = store
            orders.append({
                "budget x": mult, "state ranking": order,
                "spearman": round(rank_metrics([p.score for p in store], gold)[0], 3),
                "mae": round(np.mean([abs(p.score - LABELS[p.resume_id])
                                      for p in store]), 2),
                "satisfied": round(np.mean([len(reqs) - p.unsatisfied
                                            for p in store]), 2),
                "mean score": round(np.mean([p.score for p in store]), 1),
                "passages": round(np.mean([p.reads for p in store]), 2),
                "CVs scored below the ranking": sum(
                    1 for p in store if p.score < fixed_by_id[p.resume_id])})
    print(pd.DataFrame(orders).to_string(index=False))
    drops = {}
    for order in ("goal", "coverage"):
        seq = [by_order[order][m] for m in BUDGET_MULT]
        drops[order] = sum(1 for i in range(len(pairs))
                          for prev, nxt in zip(seq[:-1], seq[1:])
                          if nxt[i].score < prev[i].score)
    print(f"   more words can mean a lower score: across the "
          f"{len(pairs) * (len(BUDGET_MULT) - 1)} budget steps of this grid, the `goal` "
          f"ranking leaves a CV below the score it earned one step earlier "
          f"{drops['goal']} times and `coverage` {drops['coverage']} times. A bigger "
          "budget is supposed to be a floor under the score, so the goal predicate costs "
          "something a person would notice - and it is a search artefact, not a law of "
          "the problem: the search keeps the best node it saw at any depth, so the only "
          "way a wider budget can lose is by changing which nodes the beam keeps.")
    print("   `goal` is the classical one - a requirement is a predicate, either satisfied "
          "or not - and it can prefer one clean hit over more spread-out partial credit. "
          "`coverage` ranks on the graded mean, which is what the score column is made "
          "from, and the last column counts where that difference shows up.")

    print("\n4) against the two policies a planner has to beat")
    rho_fixed = rank_metrics([r.score for r in fixed], gold)[0]
    rho_whole = rank_metrics([r.score for r in whole], gold)[0]
    best_key = max(plans, key=lambda k: rank_metrics([p.score for p in plans[k]], gold)[0])
    best = plans[best_key]
    rho_best = rank_metrics([p.score for p in best], gold)[0]
    print(f"   fixed top-k passages (what select_evidence does)   rho {rho_fixed:.3f}  "
          f"at {np.mean([len(r.matched) for r in fixed]):.2f} of {len(reqs)} satisfied")
    print(f"   best planner arm (beam {best_key[0]}, budget x{best_key[1]})      "
          f"rho {rho_best:.3f}  at "
          f"{np.mean([len(reqs) - p.unsatisfied for p in best]):.2f} of {len(reqs)} satisfied")
    print(f"   whole document, no retrieval at all                rho {rho_whole:.3f}  "
          f"at {np.mean([len(r.matched) for r in whole]):.2f}")
    print("   the last row is the ceiling any budget-limited policy is measured against, "
          "and it is not a prompt: it reads every word of the CV.")

    print("\n5) the sensing step on its own (beam 8, shipped budget)")
    sensing = []
    for label, kwargs in (("explore once at the shipped floor, never re-ask",
                           {"relax_rounds": 0}),
                          ("shipped arm: re-ask unsatisfied requirements, floor halved",
                           {"relax_rounds": P.MAX_RELAX_ROUNDS}),
                          ("no floor: every top-k chunk is readable", {"floor": 0.0})):
        store = [P.plan_evidence(rid, JD, index, t, beam=8, **kwargs)
                 for rid, t in pairs]
        r = [coverage_result(rid, p.terms, reqs) for (rid, _), p in zip(pairs, store)]
        sensing.append({
            "sensing policy": label,
            "rho": round(rank_metrics([x.score for x in r], gold)[0], 3),
            "mae": round(np.mean([abs(x.score - LABELS[x.resume_id]) for x in r]), 2),
            "passages": round(np.mean([p.reads for p in store]), 2),
            "satisfied": round(np.mean([len(reqs) - p.unsatisfied
                                         for p in store]), 2),
            "CVs with no evidence": sum(1 for p in store if not p.reads),
            "CVs with a requirement no passage answers": sum(
                1 for p in store if any(c == 0 for c in p.coverages))})
    print(pd.DataFrame(sensing).to_string(index=False))
    print("   the floor is the reason a CV can be scored 9/100 with every tool named "
          "(keyword_coverage_eval.log section 1); this is what the planner does about it.")

    print("\n6) mean score by rubric class (planner at beam 8, shipped budget)")
    store = plans[(8, 1.0)]
    for k in ("strong", "medium", "weak", "hand"):
        idx = [i for i, (rid, _) in enumerate(pairs) if klass(rid) == k]
        if not idx:
            continue
        print(f"   {k:12} n={len(idx):2}  label "
              f"{np.mean([LABELS[pairs[i][0]] for i in idx]):5.1f}  planner "
              f"{np.mean([store[i].score for i in idx]):5.1f}  whole-doc "
              f"{np.mean([whole[i].score for i in idx]):5.1f}")

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
7) reading, and what this does NOT say
   - The planner optimises the same coverage function it is graded on, and rag/index.py
     holds every chunk's text, so a read action's effect is known before the action runs.
     This is therefore a budgeted maximum-coverage problem solved by forward search, and
     these rows are an upper bound on what evidence *selection* can do - not an unbiased
     estimate of a screener that had to learn what good evidence looks like, and not
     comprehension: no term rule here tells "built a churn model" from "trained on churn
     models", which is the thing the Q4_K_M row is in the loop for.
   - The budget, not the search, is what binds: at the shipped budget a plan reads
     {np.mean([p.reads for p in store]):.1f} passages, so most CVs are decided by one read
     and a wider beam has almost nothing to look ahead at. That is why the planner's gain
     here is a *selection* gain, not a sequencing gain: section 3 shows a wider beam
     changing the read set on a handful of CVs, and section 3b shows the objective the
     states are ranked by moving rho far more than the search ever does - including the
     one place where the classical goal ordering is actively worse, a bigger budget that
     returns a lower score. A planner that cannot be wrong about which words a passage
     holds has little to plan.
   - Which passages are even readable depends on the TF-IDF geometry (chunking 90/20,
     top_k {TOP_K}, floor {SIM_FLOOR['tfidf']}). The whole-document keyword arm is
     retriever-independent; a planner that must retrieve is not, so it inherits every
     finding in retrieval_quality.log.
   - The corpus is synthetic and generate_dataset.py wrote its skill bullets from the
     requisition's own vocabulary, which flatters any term-presence scorer - here and in
     keyword_coverage_eval.log.""")


if __name__ == "__main__":
    main()
