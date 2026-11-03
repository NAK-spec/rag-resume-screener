"""Sweep the retrieval similarity floor and report what each value costs.

Top-k retrieval always returns *something*, so without a floor every job requirement
looks evidenced - including for a candidate who has nothing to say about it - and the
LLM reports that noise as proof. The floor is the knob that stops it, and it is
embedder-specific: TF-IDF and dense cosine similarities live on different scales, so
SIM_FLOOR in rag/config.py must be swept per embedder, never copied across.

For each candidate floor this prints, over the whole corpus, the mean number of
requirements left with at least one surviving passage (split by score tier), the mean
de-duplicated passages a RAG prompt would carry, and the share of requirements that
return nothing.

Usage:  python sweep_floor.py [--data data] [--top-k 3] [--min-sim 0.05]
"""
import argparse
import statistics
from pathlib import Path

from rag.config import SIM_FLOOR, TOP_K
from rag.index import ResumeIndex, embedder_for
from rag.ingest import chunk_text, clean, extract_text, split_requirements
from rag.screen import evidence_budget, select_evidence
from evaluate import load_data


def tier_of(score: float) -> str:
    return "weak" if score < 40 else ("medium" if score < 75 else "strong")


def measure(index, resumes, labels, reqs, floor, top_k):
    """(tier -> mean requirements with evidence, mean passages, share of empty reqs)."""
    per_tier, passages, empty, total = {"weak": [], "medium": [], "strong": []}, [], 0, 0
    for rid, text in resumes.items():
        hits = [index.search(r, rid, k=top_k, min_score=floor) for r in reqs]
        kept = select_evidence(hits, evidence_budget(len(text.split())))
        per_tier[tier_of(labels[rid])].append(sum(1 for h in hits if h))
        passages.append(len(kept))
        empty += sum(1 for h in hits if not h)
        total += len(reqs)
    means = {t: (statistics.mean(v) if v else float("nan")) for t, v in per_tier.items()}
    return means, statistics.mean(passages), 100.0 * empty / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--min-sim", type=float, default=0.05,
                    help="largest floor value to try")
    ap.add_argument("--embedder", default="auto", choices=["auto", "tfidf", "dense"],
                    help="the floor scale is embedder-specific, so sweep each one "
                         "separately instead of trusting auto's fallback")
    args = ap.parse_args()

    jd, resumes, labels = load_data(Path(args.data))
    reqs = split_requirements(jd)
    index = ResumeIndex(embedder_for(args.embedder))
    index.build([c for rid, t in resumes.items() for c in chunk_text(rid, t)],
                extra_corpus=[jd])
    print(f"embedder: {index.embedder.kind} | resumes: {len(resumes)} | "
          f"chunks: {len(index.chunks)} | requirements: {len(reqs)}")
    if index.embedder.kind == "tfidf":
        print("[caveat] TF-IDF similarity scale; a dense run needs its own sweep.")

    header = (f"{'floor':>6} {'weak':>7} {'medium':>7} {'strong':>7} {'str-weak':>9} "
              f"{'passages':>9} {'no-evidence':>12}")
    print("\n" + header + "\n" + "-" * len(header))
    steps = [round(i * args.min_sim / 10, 4) for i in range(11)]
    if SIM_FLOOR.get(index.embedder.kind) not in steps:
        steps.append(SIM_FLOOR.get(index.embedder.kind, 0.0))
    for floor in sorted(steps):
        means, pass_mean, empty_pct = measure(index, resumes, labels, reqs, floor,
                                              args.top_k)
        mark = "*" if floor == SIM_FLOOR.get(index.embedder.kind) else " "
        print(f"{mark}{floor:>5.3f} {means['weak']:>7.2f} {means['medium']:>7.2f} "
              f"{means['strong']:>7.2f} {means['strong'] - means['weak']:>9.2f} "
              f"{pass_mean:>9.2f} {empty_pct:>11.1f}%")
    print("\n* = the value currently in rag/config.py SIM_FLOOR")
    print("Pick the floor with the widest strong-weak gap that still returns passages; "
          "past the peak, strong candidates lose evidence faster than weak ones.")


if __name__ == "__main__":
    main()
