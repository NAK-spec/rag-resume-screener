"""Central configuration. Edit MODELS to point at the GGUF files you download."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

# Quantization variants compared in the study (same base model, different precision).
# Download the GGUF files into ./models (see README).
MODELS = {
    "Q4_K_M (4-bit)": MODEL_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Q8_0 (8-bit)": MODEL_DIR / "qwen2.5-1.5b-instruct-q8_0.gguf",
    "F16 (baseline)": MODEL_DIR / "qwen2.5-1.5b-instruct-fp16.gguf",
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 90        # words per resume chunk
CHUNK_OVERLAP = 20      # overlapping words between chunks
TOP_K = 3               # chunks retrieved per job requirement
# RAG only beats full-context prompting if the evidence budget is a FRACTION of the
# resume. A flat cap never binds on a 200-word CV, so the prompt stayed longer than
# the document it was supposed to compress.
RAG_EVIDENCE_FRACTION = 0.4   # resume words allowed into the prompt as evidence
RAG_MIN_EVIDENCE_WORDS = 120  # floor: short resumes still get their whole text
MAX_EVIDENCE_WORDS = 400      # ceiling: long resumes are still truncated
# Similarity floor for retrieved evidence. Top-k always returns *some* chunk, so
# without a floor a hospitality CV's "managed rosters" bullet is handed to the LLM as
# evidence for "machine learning" and it scores that candidate 100/100 (rubric label 10).
# Both floors come from sweep_floor.py, which maximises the strong-weak gap in
# requirements-left-with-evidence; the two embedders peak at very different filter
# strengths, so the numbers are not interchangeable. Reproduced in sweep_tfidf.log and
# sweep_dense.log (strong-weak gap in evidenced requirements):
#   tfidf: 0.016 -> 2.50, 0.024 -> 2.75 (peak), 0.032 -> 1.94, 0.048 -> 0.50
#   dense: 0.180 -> 1.26, 0.240 -> 2.31 (peak), 0.270 -> 2.24, 0.360 -> 1.71, 0.450 -> 0.35
# Past the peak the floor removes more of the good candidates' real evidence than the weak
# candidates' noise: by 0.405 dense leaves a strong candidate 1.40 evidenced requirements
# against a weak candidate's 0.73, so the gap has fallen from 2.31 to 0.67 without ever
# inverting. Applying either floor to the other embedder is a silent mis-tune: 0.024
# against dense similarities filters almost nothing (gap 0.09), and 0.24 against TF-IDF
# leaves nearly every requirement with no evidence.
SIM_FLOOR = {"tfidf": 0.024, "dense": 0.24}
N_CTX = 4096
MAX_NEW_TOKENS = 220
N_THREADS = 4
