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
TOP_K = 4               # chunks retrieved per job requirement
N_CTX = 4096
MAX_NEW_TOKENS = 220
N_THREADS = 4
