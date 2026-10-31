"""Resume ingestion: file -> text -> overlapping chunks."""
import io
import re
from dataclasses import dataclass

from .config import CHUNK_OVERLAP, CHUNK_WORDS


def extract_text(name: str, data: bytes) -> str:
    name = name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    return data.decode("utf-8", errors="ignore")


def clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Chunk:
    resume_id: str
    idx: int
    text: str


def chunk_text(resume_id: str, text: str,
               size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        piece = words[start:start + size]
        if piece:
            chunks.append(Chunk(resume_id, i, " ".join(piece)))
        if start + size >= len(words):
            break
    return chunks


def split_requirements(jd: str) -> list[str]:
    """Break a job description into individual requirement lines (bullets/sentences)."""
    lines = []
    for raw in re.split(r"[\n;]|(?<=[.!?])\s+", jd):
        s = re.sub(r"^[\-\*•\d\.\)\s]+", "", raw).strip()
        if len(s.split()) >= 2:
            lines.append(s)
    return lines
