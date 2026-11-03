"""Evaluation harness for the study.

Compares (a) RAG vs full-context prompting and (b) each quantization level on:
  - ranking agreement with human scores (Spearman, Kendall)
  - mean absolute error vs human scores, and +/-10-point accuracy
  - latency, tokens/sec, peak RSS memory, model file size, prompt length
  - prompt compression: how much shorter the RAG prompt is than the full resume

Variants whose GGUF file is missing are SKIPPED rather than reported, because they
silently fall back to retrieval-only scoring and the CSV then contains three
identical rows that look like three quantization results but are not. Use --include-
missing to force them, or --fallback to benchmark the retrieval-only baseline.

Usage:  python evaluate.py [--data data] [--out results.csv] [--top-k 3]
        python evaluate.py --include-missing     # also rows for absent GGUF files
        python evaluate.py --fallback            # add the no-LLM retrieval baseline
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from rag.config import MODELS, SIM_FLOOR, TOP_K
from rag.index import ResumeIndex, embedder_for
from rag.ingest import chunk_text, clean, extract_text
from rag.llm import RetrievalOnlyLLM, load_llm
from rag.screen import screen_resume

TOLERANCE = 10  # points: a score this close to the human label counts as accurate


def load_data(data_dir: Path):
    jd = (data_dir / "job_description.txt").read_text(encoding="utf-8")
    resumes = {}
    for p in sorted((data_dir / "resumes").iterdir()):
        text = clean(extract_text(p.name, p.read_bytes()))
        if text:
            resumes[p.name] = text
    labels = pd.read_csv(data_dir / "labels.csv").set_index("resume_id")["human_score"]
    missing = [r for r in resumes if r not in labels.index]
    if missing:
        raise SystemExit(f"labels.csv has no human_score for: {', '.join(missing)}")
    return jd, resumes, labels


def rank_metrics(pred, gold):
    """Correlation needs spread in the predictions; a constant scorer has no rank.

    A near-constant scorer is the failure mode this catches: 32 of 33 candidates at the
    same score still yields a computable rho, but that rho is carried by one or two
    outliers and says nothing about the method. `n_distinct_scores` is reported with it.
    """
    if len(pred) < 3 or len(set(pred)) < 2 or len(set(gold)) < 2:
        return float("nan"), float("nan")
    return float(spearmanr(pred, gold)[0]), float(kendalltau(pred, gold)[0])


def metric_row(llm, variant, mode, results, labels, embedder):
    pred = np.array([r.score for r in results], dtype=float)
    gold = np.array([labels[r.resume_id] for r in results], dtype=float)
    rho, tau = rank_metrics(pred, gold)
    cov = np.array([r.score_coverage for r in results], dtype=float)
    cov_rho, _ = rank_metrics(cov, gold)
    return {
        "model": getattr(llm, "name", variant), "variant": variant, "mode": mode,
        "embedder": embedder, "n": len(pred),
        "spearman": round(rho, 3), "kendall": round(tau, 3),
        "n_distinct_scores": int(len(set(pred))),
        "modal_share": round(float(np.max(np.unique(pred, return_counts=True)[1])) / len(pred), 3),
        "mae": round(float(np.mean(np.abs(pred - gold))), 2),
        "within_10": round(float(np.mean(np.abs(pred - gold) <= TOLERANCE)), 3),
        "cov_spearman": round(cov_rho, 3),
        "cov_mae": round(float(np.mean(np.abs(cov - gold))), 2),
        "cov_within_10": round(float(np.mean(np.abs(cov - gold) <= TOLERANCE)), 3),
        "avg_latency_s": round(float(np.mean([r.latency_s for r in results])), 3),
        # A retrieval-only score is produced in microseconds, so its tokens/second figure is
        # a division artefact (it reported 1.1 million tok/s) rather than a throughput
        # measurement. Only rows that actually generated are averaged.
        "avg_tok_per_s": (lambda v: round(float(np.mean(v)), 1) if v else float("nan"))(
            [r.tok_per_s for r in results if r.latency_s >= 0.01]),
        "model_load_s": round(getattr(llm, "load_s", 0.0), 2),
        "peak_rss_mb": round(max(r.rss_mb for r in results), 0),
        "model_size_mb": round(getattr(llm, "size_mb", 0.0), 0),
        "avg_prompt_words": round(float(np.mean([r.prompt_words for r in results])), 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="auto", choices=["auto", "tfidf", "dense"],
                    help="pin the retrieval backend; a dense-vs-tfidf comparison must "
                         "pin it, because auto silently falls back and the similarity "
                         "floor differs per embedder")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--dump", default="",
                    help="also write one row per (variant, mode, resume) for auditing "
                         "individual scores against their evidence")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--include-missing", action="store_true",
                    help="also report variants whose GGUF file is absent (they run "
                         "the retrieval-only fallback, not a real quantization)")
    ap.add_argument("--fallback", action="store_true",
                    help="add a retrieval-only (no LLM) baseline row for both modes. Note "
                         "this does NOT skip the GGUF variants: once a model is in ./models "
                         "the full LLM eval runs too. For the baseline alone, add "
                         "--variant 'Retrieval-only (no LLM)'")
    ap.add_argument("--variant", default="",
                    help="benchmark only this variant, in its own process, for clean "
                         "per-variant memory numbers")
    ap.add_argument("--floor", type=float, default=None,
                    help="override SIM_FLOOR for the embedder that actually ran, for an "
                         "A/B of the relevance floor. The value is echoed in the run's "
                         "first log line; editing rag/config.py instead leaves no trace.")
    args = ap.parse_args()

    jd, resumes, labels = load_data(Path(args.data))
    chunks = [c for rid, t in resumes.items() for c in chunk_text(rid, t)]
    index = ResumeIndex(embedder_for(args.embedder))
    index.build(chunks, extra_corpus=[jd])
    if args.floor is not None:
        SIM_FLOOR[index.embedder.kind] = args.floor
    words = [len(t.split()) for t in resumes.values()]
    print(f"embedder: {index.embedder.kind} | sim_floor: {SIM_FLOOR[index.embedder.kind]} "
          f"| resumes: {len(resumes)} | chunks: {len(chunks)}")
    if index.embedder.kind == "tfidf":
        print("[caveat] TF-IDF retrieval, not the dense embedder - the similarity floor "
              f"in rag/config.py was swept for tfidf only. Dense load error: "
              f"{index.embedder.dense_error}")
    print(f"resume length: min {min(words)} / mean {round(np.mean(words))} / "
          f"max {max(words)} words")

    variants = dict(MODELS)
    absent = [k for k, p in variants.items() if not Path(p).exists()]
    if absent and not args.include_missing:
        print(f"skipping (no GGUF in ./models): {', '.join(absent)}")
    selected = {k: v for k, v in variants.items()
                if args.include_missing or Path(v).exists()}
    if args.fallback:
        selected["Retrieval-only (no LLM)"] = None
    if args.variant:
        if args.variant not in selected:
            raise SystemExit(f"--variant {args.variant!r} is not available; "
                             f"options: {', '.join(selected) or 'none'}")
        selected = {args.variant: selected[args.variant]}

    if not selected:
        raise SystemExit("No GGUF model found in ./models and --fallback not given. "
                         "Download one (see README) or pass --fallback.")

    rows, per_resume = [], {}
    per_resume_rows = []
    for name, path in selected.items():
        llm = RetrievalOnlyLLM() if path is None else load_llm(path)
        if isinstance(llm, RetrievalOnlyLLM) and path is not None:
            print(f"[warn] {name}: failed to load {path}, fell back to retrieval-only")
        for mode in ("rag", "full"):
            results = [screen_resume(rid, jd, index, llm, mode=mode,
                                     full_text=txt, top_k=args.top_k)
                       for rid, txt in resumes.items()]
            rows.append(metric_row(llm, name, mode, results, labels,
                                   index.embedder.kind))
            for r in results:
                per_resume[(name, r.resume_id, mode)] = r.prompt_words
                per_resume_rows.append({
                    "variant": name, "mode": mode, "embedder": index.embedder.kind,
                    "resume_id": r.resume_id, "human": labels[r.resume_id],
                    "score": r.score, "score_coverage": r.score_coverage,
                    "matched": len(r.matched), "missing": len(r.missing),
                    "passages": len({x for v in r.evidence_refs.values() for x in v}),
                    "prompt_words": r.prompt_words,
                    # outputs near MAX_NEW_TOKENS risk truncated JSON; worth a column
                    "output_words": len(r.raw.split()),
                    "latency_s": round(r.latency_s, 3),
                })
        del llm

    # prompt compression: same resumes, same model, RAG against full context
    for row in rows:
        if row["mode"] != "rag":
            continue
        rag_w = [per_resume[(row["variant"], rid, "rag")] for rid in resumes]
        full_w = [per_resume[(row["variant"], rid, "full")] for rid in resumes]
        row["rag_words_vs_full"] = round(float(np.mean(rag_w) / np.mean(full_w)), 3)
        row["median_rag_words"] = round(float(np.median(rag_w)), 0)
        row["median_full_words"] = round(float(np.median(full_w)), 0)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    if args.dump:
        detail = pd.DataFrame(per_resume_rows)
        detail.to_csv(args.dump, index=False)
        print(f"\nper-resume detail: {len(detail)} rows -> {args.dump}")
    print("\n" + out.drop(columns=["model"]).to_string(index=False))
    degenerate = out[out["modal_share"] > 0.5]
    if not degenerate.empty:
        print("\n[degenerate scores] these rows put more than half of all candidates on ONE "
              "score, so their rho/tau is carried by a few outliers and must not be quoted "
              "as a ranking result: "
              + ", ".join(f"{r.variant}/{r.mode} (modal {r.modal_share:.0%}, "
                          f"{r.n_distinct_scores} distinct of {r.n})"
                          for r in degenerate.itertuples()))
    print("\npeak_rss_mb is process-wide: to compare memory across quantization levels, "
          "run one variant per process, e.g. evaluate.py --variant 'Q4_K_M (4-bit)'.")

    # headline efficiency claim, per variant
    print("\nPrompt compression (RAG vs full context, mean words):")
    for name in selected:
        rag = [per_resume[(name, rid, "rag")] for rid in resumes]
        full = [per_resume[(name, rid, "full")] for rid in resumes]
        if not rag or not full:
            continue
        reduction = 100 * (1 - np.mean(rag) / np.mean(full))
        shorter = sum(1 for a, b in zip(rag, full) if a < b)
        print(f"  {name:22s} {np.mean(full):6.0f} -> {np.mean(rag):6.0f} words "
              f"({reduction:+.1f}%), RAG shorter on {shorter}/{len(resumes)} resumes")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
