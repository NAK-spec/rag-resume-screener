"""Vector index with dense embeddings (sentence-transformers) or TF-IDF fallback."""
import numpy as np

from .config import EMBED_MODEL
from .ingest import Chunk


class Embedder:
    def __init__(self, prefer_dense: bool = True):
        self.kind = "tfidf"
        self.model = None
        self.tfidf = None
        if prefer_dense:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(EMBED_MODEL)
                self.kind = "dense"
            except Exception:
                self.model = None

    def fit(self, corpus: list[str]):
        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                         stop_words="english")
            self.tfidf.fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.kind == "dense":
            v = self.model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)
        m = self.tfidf.transform(texts).toarray().astype(np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1
        return m / n


class ResumeIndex:
    """In-memory cosine-similarity index over resume chunks."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk], extra_corpus: list[str] | None = None):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embedder.fit(texts + (extra_corpus or []))
        self.matrix = self.embedder.encode(texts)

    def search(self, query: str, resume_id: str, k: int = 4):
        """Top-k chunks of ONE resume for a query. Returns [(score, Chunk)]."""
        idx = [i for i, c in enumerate(self.chunks) if c.resume_id == resume_id]
        if not idx:
            return []
        q = self.embedder.encode([query])[0]
        sims = self.matrix[idx] @ q
        order = np.argsort(-sims)[:k]
        return [(float(sims[o]), self.chunks[idx[o]]) for o in order]
