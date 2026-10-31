"""Evaluation harness for the study.

Compares (a) RAG vs full-context prompting and (b) each quantization level on:
  - ranking agreement with human scores (Spearman, Kendall)
  - mean absolute error vs human scores
  - latency, tokens/sec, peak RSS memory, model file size, prompt length

Usage:  python evaluate.py [--data data] [--out results.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from rag.config import MODELS
from rag.index import Embedder, ResumeIndex
from rag.ingest import chunk_text, clean, extract_text
from rag.llm import load_llm
from rag.screen import screen_resume


def load_data(data_dir: Path):
    jd = (data_dir / "job_description.txt").read_text(encoding="utf-8")
    resumes = {}
    for p in sorted((data_dir / "resumes").iterdir()):
        resumes[p.name] = clean(extract_text(p.name, p.read_bytes()))
    labels = pd.read_csv(data_dir / "labels.csv").set_index("resume_id")["human_score"]
    return jd, resumes, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    jd, resumes, labels = load_data(Path(args.data))
    chunks = [c for rid, t in resumes.items() for c in chunk_text(rid, t)]
    index = ResumeIndex(Embedder())
    index.build(chunks, extra_corpus=[jd])
    print(f"Embedder: {index.embedder.kind} | resumes: {len(resumes)} | chunks: {len(chunks)}")

    rows = []
    for name, path in MODELS.items():
        llm = load_llm(path)
        for mode in ("rag", "full"):
            res = [screen_resume(rid, jd, index, llm, mode=mode, full_text=txt)
                   for rid, txt in resumes.items()]
            pred = np.array([r.score for r in res], dtype=float)
            gold = np.array([labels[r.resume_id] for r in res], dtype=float)
            rho = spearmanr(pred, gold)[0] if len(set(pred)) > 1 else float("nan")
            tau = kendalltau(pred, gold)[0] if len(set(pred)) > 1 else float("nan")
            rows.append({
                "model": getattr(llm, "name", name), "variant": name, "mode": mode,
                "spearman": round(rho, 3), "kendall": round(tau, 3),
                "mae": round(float(np.mean(np.abs(pred - gold))), 2),
                "avg_latency_s": round(float(np.mean([r.latency_s for r in res])), 3),
                "avg_tok_per_s": round(float(np.mean([r.tok_per_s for r in res])), 1),
                "peak_rss_mb": round(max(r.rss_mb for r in res), 0),
                "model_size_mb": round(getattr(llm, "size_mb", 0.0), 0),
                "avg_prompt_words": round(float(np.mean([r.prompt_words for r in res])), 0),
            })
        del llm

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
