"""Quantized LLM wrapper (llama.cpp / GGUF) with a retrieval-only fallback."""
import json
import os
import time
from dataclasses import dataclass

import psutil

from .config import MAX_NEW_TOKENS, N_CTX, N_THREADS


@dataclass
class GenResult:
    text: str
    latency_s: float
    new_tokens: int
    rss_mb: float

    @property
    def tok_per_s(self) -> float:
        return self.new_tokens / self.latency_s if self.latency_s > 0 else 0.0


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


class QuantizedLLM:
    """Loads a GGUF model (any quantization level) with llama-cpp-python."""

    def __init__(self, path: str, name: str = ""):
        from llama_cpp import Llama
        self.name = name or os.path.basename(str(path))
        self.size_mb = os.path.getsize(path) / 1e6
        t0 = time.perf_counter()
        self.llm = Llama(model_path=str(path), n_ctx=N_CTX,
                         n_threads=N_THREADS, verbose=False)
        self.load_s = time.perf_counter() - t0

    def generate(self, system: str, user: str, meta=None) -> GenResult:
        t0 = time.perf_counter()
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=MAX_NEW_TOKENS, temperature=0.0)
        dt = time.perf_counter() - t0
        text = out["choices"][0]["message"]["content"]
        ntok = out.get("usage", {}).get("completion_tokens", len(text.split()))
        return GenResult(text, dt, ntok, _rss_mb())


class RetrievalOnlyLLM:
    """Fallback used when no GGUF model is available (demo / tests).

    Produces the same JSON schema as the real LLM, but derives the score from
    the retrieval similarities passed in `meta`. Lets the whole app and the
    evaluation harness run end to end without a model download.
    """
    name = "Retrieval-only (no LLM)"
    size_mb = 0.0
    load_s = 0.0

    def generate(self, system: str, user: str, meta=None) -> GenResult:
        t0 = time.perf_counter()
        meta = meta or {}
        reqs = meta.get("requirements", [])
        sims = meta.get("req_scores", [])
        matched = [r for r, s in zip(reqs, sims) if s >= 0.28]
        missing = [r for r, s in zip(reqs, sims) if s < 0.28]
        score = round(100 * sum(min(1.0, s / 0.5) for s in sims) / max(1, len(sims)))
        text = json.dumps({
            "score": int(max(0, min(100, score))),
            "matched": matched, "missing": missing,
            "rationale": "Score derived from semantic similarity between job "
                         "requirements and the most relevant resume passages."})
        return GenResult(text, time.perf_counter() - t0, len(text.split()), _rss_mb())


def load_llm(path):
    """Return a QuantizedLLM if the file + runtime exist, else the fallback."""
    try:
        if path and os.path.exists(str(path)):
            return QuantizedLLM(str(path))
    except Exception as e:  # missing llama_cpp, bad file, etc.
        print(f"[llm] could not load {path}: {e}")
    return RetrievalOnlyLLM()
