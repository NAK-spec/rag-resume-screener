"""Can a no-LLM screener rank at all? Two candidates, measured on the same 33 CVs.

The app's default scorer is Retrieval-only, which turns the best chunk similarity per
requirement into a score. That number answers "how much do these two blobs of text
overlap", and a single exact keyword inside a 90-word chunk scores ~0.06 with TF-IDF -
so a CV that names every tool in the requisition can come back 9/100 with 0 matched.
Keyword coverage is the other no-model option: does the document contain what each
requirement asks for - one ask per content word, with an example list or a run of names
counted as a single any-of ask.

Both are scored here against the committed rubric labels, with no GGUF and no torch, in
seconds. The committed Q4_K_M row is read from results_v2.csv and quoted for scale, not
re-measured.

    python keyword_coverage_eval.py > keyword_coverage_eval.log   # ~10 s
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate import metric_row, rank_metrics          # noqa: E402
from rag.config import SIM_FLOOR, TOP_K                # noqa: E402
from rag.index import ResumeIndex, embedder_for        # noqa: E402
from rag.ingest import chunk_text, clean, split_requirements  # noqa: E402
from rag.llm import (KEYWORD_MATCH_FRACTION, KeywordCoverageLLM, RetrievalOnlyLLM,  # noqa: E402
                     document_terms, keyword_coverage, keyword_terms, requirement_groups)
from rag.screen import screen_resume                   # noqa: E402

JD = (ROOT / "data/job_description.txt").read_text(encoding="utf-8")
LABELS = {r["resume_id"]: int(r["human_score"])
          for r in pd.read_csv(ROOT / "data/labels.csv").to_dict("records")}


def corpus():
    out = []
    for f in sorted((ROOT / "data/resumes").glob("*")):
        if f.suffix.lower() == ".txt":
            out.append((f.name, clean(f.read_text(encoding="utf-8", errors="ignore"))))
    return out


def klass(resume_id):
    for k in ("strong", "medium", "weak"):
        if k in resume_id:
            return k
    return "hand-written"


def main():
    pairs = corpus()
    reqs = split_requirements(JD)
    index = ResumeIndex(embedder_for("tfidf"))
    index.build([c for rid, t in pairs for c in chunk_text(rid, t)], extra_corpus=[JD])

    print(f"corpus: {len(pairs)} CVs | requirements: {len(reqs)} | "
          f"floor {SIM_FLOOR['tfidf']} | top_k {TOP_K} | "
          f"keyword match fraction {KEYWORD_MATCH_FRACTION}")
    print("the keyword scorer reads the whole document, so the chunking, floor, top-k and "
          "retriever choice cannot change its output; it is run once and reported as is.\n")

    rows, per_cv = [], []
    for name, llm in (("Retrieval-only (chunk similarity)", RetrievalOnlyLLM()),
                      ("Keyword coverage (whole document)", KeywordCoverageLLM())):
        results = [screen_resume(rid, JD, index, llm, mode="rag", full_text=t,
                                 top_k=TOP_K) for rid, t in pairs]
        rows.append({"scorer": name, **metric_row(llm, name, "rag", results, LABELS,
                                                 "tfidf")})
        per_cv.append(results)

    table = pd.DataFrame(rows)[["scorer", "n", "spearman", "kendall", "n_distinct_scores",
                                "modal_share", "mae", "within_10", "avg_latency_s"]]
    print("1) ranking quality against the rubric labels (no model in either row)")
    # avg_latency_s is dropped from this table on purpose: both rows are pure text
    # matching, so the column measures Python, not a method. Screening all 33 CVs with
    # the keyword scorer takes a couple of seconds end to end and no latency claim is
    # made from it - unlike results_v2.csv, where latency is the efficiency measurement.
    print(table.drop(columns=["avg_latency_s"]).to_string(index=False))

    ref = pd.read_csv(ROOT / "results_v2.csv")
    q4 = ref[(ref["variant"] == "Q4_K_M (4-bit)") & (ref["mode"] == "rag")
             & (ref["embedder"] == "tfidf")]
    if len(q4):
        r = q4.iloc[0]
        print(f"\n   for scale, the committed Q4_K_M row (not re-measured here): "
              f"spearman {r['spearman']}, mae {r['mae']}, "
              f"n_distinct_scores {int(r['n_distinct_scores'])}, "
              f"latency {r['avg_latency_s']} s/CV")

    print("\n2) mean score by rubric class (does keyword coverage inflate the weak CVs?)")
    for k in ("strong", "medium", "weak", "hand-written"):
        idx = [i for i, (rid, _) in enumerate(pairs) if klass(rid) == k]
        if not idx:
            continue
        gold = np.mean([LABELS[pairs[i][0]] for i in idx])
        sim = np.mean([per_cv[0][i].score for i in idx])
        key = np.mean([per_cv[1][i].score for i in idx])
        print(f"   {k:12} n={len(idx):2}  label {gold:5.1f}  "
              f"retrieval-only {sim:5.1f}  keyword {key:5.1f}")

    print(f"\n3) where the two disagree (matched-list length per CV)")
    zero_sim = sum(1 for r in per_cv[0] if not r.matched)
    zero_key = sum(1 for r in per_cv[1] if not r.matched)
    flip = [(b.resume_id, len(a.matched), len(b.matched))
            for a, b in zip(per_cv[0], per_cv[1]) if len(b.matched) > len(a.matched)]
    print(f"   CVs scored with 0 matched: retrieval-only {zero_sim}, keyword {zero_key}")
    print(f"   CVs the keyword scorer upgraded above retrieval-only: {len(flip)} of {len(pairs)}")
    for rid, a, b in flip[:8]:
        print(f"      {rid:22} {a} -> {b} matched")
    if len(flip) > 8:
        print(f"      ... {len(flip) - 8} more, same direction")

    gold = [LABELS[rid] for rid, _ in pairs]
    rho_s, _ = rank_metrics([r.score for r in per_cv[0]], gold)
    rho_k, _ = rank_metrics([r.score for r in per_cv[1]], gold)

    print("\n4) the same keyword rule, but reading only what RAG pasted")
    # This is the number the framework actually needs: if the evidence block carries the
    # same terms as the whole document, retrieval has cost the keyword scorer nothing.
    ev_results = []
    for rid, t in pairs:
        r = screen_resume(rid, JD, index, KeywordCoverageLLM(), mode="rag", full_text=t,
                          top_k=TOP_K)
        block = " ".join(s for passage in r.evidence.values() for s in passage)
        doc_terms = document_terms(block)
        covs = [keyword_coverage(q, doc_terms)[0] for q in reqs]
        ev_results.append((rid, round(100 * sum(covs) / len(covs)),
                           len([1 for c in covs if c >= KEYWORD_MATCH_FRACTION])))
    gold = [LABELS[rid] for rid, _, _ in ev_results]
    rho_ev, _ = rank_metrics([s for _, s, _ in ev_results], gold)
    print(f"   rho {rho_ev:.3f} when coverage is scored on the retrieved passages only "
          f"(whole document: {rho_k:.3f})")
    print(f"   mean evidenced requirements: "
          f"{np.mean([m for _, _, m in ev_results]):.2f} of {len(reqs)} (whole document "
          f"{np.mean([len(r.matched) for r in per_cv[1]]):.2f})")
    print("   the gap is what the word budget costs a term-presence scorer - and unlike the")
    print("   model, this scorer cannot reason past a term that is not on the page.")

    print("\n5) what keywords structurally cannot see")
    corpus_terms = document_terms(" ".join(t for _, t in pairs))
    for r in reqs:
        absent = keyword_coverage(r, corpus_terms)[1]
        print(f"   {len(requirement_groups(r))} asks, "
              f"{len(keyword_terms(r))} stems | {r[:52]:52} | in no CV of the study: "
              f"{', '.join(absent) or 'none'}")
    print("   coverage above is against the whole 33-CV corpus, so a word listed here is "
          "one no CV in the study contains - a keyword scorer can only ever call those "
          "missing, and that is the ceiling on this method, not a bug in it.")

    print(f"\n6) reading: rho {rho_s:.3f} (retrieval-only) -> {rho_k:.3f} (keyword).")
    print("   Three things this does NOT say:")
    print("   - it is not a RAG result. Coverage reads the whole document, so nothing here")
    print("     depends on retrieval; it prices the *scorer*, not the framework.")
    print("   - the corpus is synthetic and generate_dataset.py wrote the rubric's skill")
    print("     bullets using the requisition's own vocabulary. A keyword scorer is graded on")
    print("     the very tokens the generator was told to place, so this rho is the most")
    print("     generous measurement that method can ever get; on real CVs, which paraphrase,")
    print("     it will be lower. It is also gameable by keyword stuffing, which is why the")
    print("     study's headline stays the Q4_K_M row that reads the evidence.")
    print("   - the scale is compressed (see section 2): ranking improved, absolute scores did")
    print("     not track the labels, and mae is quoted next to rho for exactly that reason.")


if __name__ == "__main__":
    main()
