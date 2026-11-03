"""Re-derive the pre-dedup prompt inflation, on the current corpus, with no LLM involved.

Why this file exists: the README quotes "304 -> 416 words" for the prompt-bloat bug, but
that figure was measured live during the fix and never written to a committed artifact, so
it was the one number in the results section that a reader could not check. This script
rebuilds the historical prompt exactly as `rag/screen.py` stood at commit 9d2e10a
(`TOP_K = 4`, no relevance floor, no word budget, one copy of each chunk per requirement)
and prints both lengths side by side, so the claim becomes reproducible from the repo.

Run: python prompt_inflation.py            (seconds, TF-IDF only, no model load)
"""
import statistics
from pathlib import Path

from rag.config import SIM_FLOOR, TOP_K
from rag.index import ResumeIndex, embedder_for
from rag.ingest import chunk_text, split_requirements
from rag.screen import SCHEMA, build_prompt, evidence_budget, select_evidence
from evaluate import load_data

OLD_TOP_K = 4          # rag/config.py at commit 9d2e10a, before TOP_K became 3


def old_build_prompt(jd, requirements, evidence):
    """Verbatim from rag/screen.py at 9d2e10a - each requirement prints its own copy of the
    evidence text, so a chunk matching several requirements is pasted in several times."""
    ev = []
    for i, r in enumerate(requirements, 1):
        snippets = " ... ".join(evidence[r])
        ev.append(f"[{i}] Requirement: {r}\n    Evidence: {snippets}")
    return (f"JOB DESCRIPTION:\n{jd}\n\nRETRIEVED RESUME EVIDENCE:\n"
            + "\n".join(ev) + f"\n\nReturn JSON in this format:\n{SCHEMA}")


def main():
    jd, resumes, _ = load_data(Path("data"))
    reqs = split_requirements(jd)
    index = ResumeIndex(embedder_for("tfidf"))
    index.build([c for rid, t in resumes.items() for c in chunk_text(rid, t)],
                extra_corpus=[jd])

    old_words, new_words, full_words = [], [], []
    evidence_words, kept_words, doc_words = [], [], []
    for rid, text in resumes.items():
        # historical path: TOP_K=4, no floor (index.search had no min_score argument)
        evidence = {}
        for r in reqs:
            hits = index.search(r, rid, k=OLD_TOP_K, min_score=0.0)
            evidence[r] = [c.text for _, c in hits]
        old_words.append(len(old_build_prompt(jd, reqs, evidence).split()))
        evidence_words.append(sum(len(t.split()) for v in evidence.values() for t in v))
        doc_words.append(len(text.split()))

        # current path: TOP_K=3, per-embedder floor, dedup, budget
        hits_per_req = [index.search(r, rid, k=TOP_K, min_score=SIM_FLOOR["tfidf"])
                        for r in reqs]
        passages = select_evidence(hits_per_req, evidence_budget(len(text.split())))
        kept_words.append(sum(len(c.text.split()) for _, c in passages))
        new_words.append(len(build_prompt(jd, reqs, hits_per_req, passages).split()))
        full_words.append(len((f"JOB DESCRIPTION:\n{jd}\n\nFULL RESUME:\n{text}\n\n"
                               f"Return JSON in this format:\n{SCHEMA}").split()))

    def line(label, xs):
        print(f"{label:<34} mean {round(statistics.mean(xs)):>4} words | "
              f"median {int(statistics.median(xs)):>4} | min {min(xs):>4} | max {max(xs):>4}")

    print(f"embedder: {index.embedder.kind} | resumes: {len(resumes)} | "
          f"chunks: {len(index.chunks)} | requirements: {len(reqs)}")
    print(f"old path: TOP_K={OLD_TOP_K}, no floor, no dedup, no budget "
          f"(rag/screen.py at 9d2e10a)\n")
    line("prompt: pre-dedup (old)", old_words)
    line("prompt: full-context baseline", full_words)
    line("prompt: dedup + floor (current)", new_words)
    print("Note: the old prompt also pasted the entire job description, while the current "
          "one\nprints only the role line plus the requirement list. So the totals above "
          "conflate\ntwo fixes; the evidence-block figure at the end isolates the dedup "
          "and budget.")
    inflation = statistics.mean(old_words) - statistics.mean(full_words)
    print(f"\nThe old RAG prompt was {round(inflation)} words LONGER than the baseline it "
          f"was meant to compress ({round(statistics.mean(old_words))} vs "
          f"{round(statistics.mean(full_words))}); the current one is "
          f"{round(statistics.mean(full_words) - statistics.mean(new_words))} shorter "
          f"({round(statistics.mean(new_words))} vs {round(statistics.mean(full_words))}).")

    # The prompt totals above also carry the JD text (old path) or the requirement list and
    # labels (new path), so the cleanest statement of the bug is the evidence block alone:
    # the old path pasted 6 requirements x top-4 chunks with no dedup and no budget.
    ratio = [e / max(1, d) for e, d in zip(evidence_words, doc_words)]
    kept_ratio = [k / max(1, d) for k, d in zip(kept_words, doc_words)]
    print(f"\nEvidence block only, mean pasted words per resume: old "
          f"{round(statistics.mean(evidence_words))} vs current "
          f"{round(statistics.mean(kept_words))}, against a mean document of "
          f"{round(statistics.mean(doc_words))} words. The old path reproduced "
          f"{round(statistics.mean(ratio), 1)}x the document in evidence text; the current "
          f"path carries {round(statistics.mean(kept_ratio), 2)}x, capped by the budget.")


if __name__ == "__main__":
    main()
