"""RAG screening pipeline: requirements -> retrieve evidence -> LLM scoring."""
import json
import re
from dataclasses import dataclass, field

from .config import (MAX_EVIDENCE_WORDS, RAG_EVIDENCE_FRACTION, RAG_MIN_EVIDENCE_WORDS,
                     SIM_FLOOR, TOP_K)
from .index import ResumeIndex
from .ingest import BULLET_RE, split_requirements

SYSTEM = (
    "You are an objective resume screening assistant. Judge ONLY from the "
    "evidence provided. Ignore name, gender, age, nationality and photo. "
    "Count a requirement as matched only if some passage shows it in actual work "
    "or a project; a skill merely listed in a Skills line is not proof, and a "
    "requirement with no passage goes in missing. Score the fit, do not assume it. "
    "Reply with a single JSON object and nothing else."
)

SCHEMA = ('{"score": <integer 0-100 overall fit>, '
          '"matched": [<requirements clearly supported by the evidence>], '
          '"missing": [<requirements with no support>], '
          '"rationale": "<2 short sentences citing the evidence>"}')


@dataclass
class Result:
    resume_id: str
    score: int
    score_coverage: int = -1   # fit implied by the matched list, not the model's guess
    matched: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    rationale: str = ""
    evidence: dict = field(default_factory=dict)   # requirement -> [chunk texts]
    evidence_refs: dict = field(default_factory=dict)  # requirement -> [passage labels]
    latency_s: float = 0.0
    tok_per_s: float = 0.0
    rss_mb: float = 0.0
    prompt_words: int = 0
    raw: str = ""


def parse_json(text: str) -> dict:
    """Model output is usually the requested JSON object, but a small quantized model
    can emit prose; a bare `score: 62` still has to be usable."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    sc = re.search(r'"?score"?\s*[:=]\s*(\d{1,3})', text)
    return {"score": int(sc.group(1))} if sc else {}


def _key(chunk):
    return (chunk.resume_id, chunk.idx)


def evidence_budget(resume_words: int) -> int:
    """How much resume text a RAG prompt may carry, as a fraction of the document.

    Never more than the document itself: on a resume shorter than the minimum, "all the
    evidence" is the whole resume, and a RAG prompt longer than the full-resume baseline
    it is meant to compress would be a reporting error, not an efficiency result.
    """
    if resume_words <= 0:
        return MAX_EVIDENCE_WORDS
    return int(min(MAX_EVIDENCE_WORDS, resume_words,
                   max(RAG_MIN_EVIDENCE_WORDS, RAG_EVIDENCE_FRACTION * resume_words)))


def select_evidence(hits_per_req, max_words: int = MAX_EVIDENCE_WORDS):
    """Merge the per-requirement hits into one de-duplicated passage set.

    top-k per requirement repeats the same resume text across requirements, which
    made the RAG prompt longer than the full resume it was meant to compress. A
    chunk is kept once, ranked by its best similarity over all requirements, until
    the word budget is spent; the first chunk is always kept so there is evidence.
    Returns [(label, Chunk)] in resume order.
    """
    best: dict = {}
    for hits in hits_per_req:
        for score, c in hits:
            k = _key(c)
            if k not in best or score > best[k][0]:
                best[k] = (score, c)
    ranked = sorted(best.values(), key=lambda sc: -sc[0])
    kept, words = [], 0
    for _, c in ranked:
        w = len(c.text.split())
        if kept and words + w > max_words:
            continue
        kept.append(c)
        words += w
    kept.sort(key=lambda c: c.idx)
    return [(f"E{i + 1}", c) for i, c in enumerate(kept)]


def jd_context(jd: str, reqs: list[str]) -> str:
    """The parts of the JD that are not requirements (role title, team, location)."""
    keep = [s for l in jd.splitlines() if (s := l.strip())
            and re.sub(BULLET_RE.pattern, "", s).rstrip(".;") not in reqs]
    return " ".join(keep)[:200]


def build_prompt(jd: str, requirements, hits_per_req, passages) -> str:
    """passages: [(label, Chunk)] from select_evidence. Each requirement and each
    passage appears exactly once: the requirement is printed numbered, and the
    per-requirement evidence is a list of passage labels rather than the JD text
    repeated a second time."""
    labels_by_key = {_key(c): label for label, c in passages}
    body = "\n".join(f"[{label}] {c.text}" for label, c in passages)
    reqs = "\n".join(f"[{i}] {r}" for i, r in enumerate(requirements, 1))
    refs = []
    for i, hits in enumerate(hits_per_req, 1):
        seen = list(dict.fromkeys(labels_by_key[_key(c)] for _, c in hits
                                 if _key(c) in labels_by_key))
        refs.append(f"[{i}] " + (", ".join(seen) if seen else "none"))
    context = jd_context(jd, requirements)
    head = f"ROLE: {context}\n\n" if context else ""
    return (f"{head}JOB REQUIREMENTS:\n{reqs}\n\n"
            f"RESUME EVIDENCE ({len(passages)} retrieved passages):\n{body}\n\n"
            "SUPPORTING PASSAGES PER REQUIREMENT:\n"
            + "\n".join(refs) + f"\n\nReturn JSON in this format:\n{SCHEMA}")


def normalize_requirements(items, reqs):
    """The prompt numbers the requirements, and a small model echoes that numbering
    into matched/missing ("[1] Proficiency in Python..."). Restoring the canonical
    requirement text keeps the lists joinable and the UI tables readable."""
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, str):
            continue
        m = re.match(r"^\s*\[(\d{1,2})\]\s*", it)
        if m and 1 <= int(m.group(1)) <= len(reqs):
            out.append(reqs[int(m.group(1)) - 1])
        else:
            out.append(it.strip())
    return out


def screen_resume(resume_id: str, jd: str, index: ResumeIndex, llm,
                  mode: str = "rag", full_text: str = "", top_k: int = TOP_K) -> Result:
    """mode='rag'  -> retrieve top-k chunks per requirement (efficient)
       mode='full' -> stuff the whole resume into the prompt (baseline)."""
    reqs = split_requirements(jd) or [jd]
    floor = SIM_FLOOR.get(index.embedder.kind, 0.0)
    hits_per_req = [index.search(r, resume_id, k=top_k, min_score=floor) for r in reqs]
    req_scores = [max((s for s, _ in h), default=0.0) for h in hits_per_req]

    if mode == "rag":
        passages = select_evidence(hits_per_req, evidence_budget(len(full_text.split())))
        prompt = build_prompt(jd, reqs, hits_per_req, passages)
    else:
        passages = []
        prompt = (f"JOB DESCRIPTION:\n{jd}\n\nFULL RESUME:\n{full_text}\n\n"
                  f"Return JSON in this format:\n{SCHEMA}")

    kept = {_key(c): label for label, c in passages}
    evidence, evidence_refs = {}, {}
    for r, hits in zip(reqs, hits_per_req):
        refs = [kept[_key(c)] for _, c in hits if _key(c) in kept]
        evidence_refs[r] = list(dict.fromkeys(refs))
        evidence[r] = [c.text for _, c in hits if _key(c) in kept]

    g = llm.generate(SYSTEM, prompt,
                     meta={"requirements": reqs, "req_scores": req_scores,
                           "embedder": index.embedder.kind, "full_text": full_text})
    d = parse_json(g.text)
    score = int(max(0, min(100, d.get("score", 0))))
    matched = normalize_requirements(d.get("matched", []), reqs)
    # A requirement cannot be both evidenced and absent, and a small model does put the
    # same line in both lists; matched wins, because it is the one with a citation.
    missing = [m for m in normalize_requirements(d.get("missing", []), reqs)
               if m not in set(matched)]
    # The model's single 0-100 guess can contradict its own matched list, so record the
    # coverage implied by that list and let the evaluation compare the two columns.
    coverage = round(100 * len({m for m in matched if m in set(reqs)}) / max(1, len(reqs)))
    return Result(resume_id=resume_id, score=score, score_coverage=coverage,
                  matched=matched, missing=missing,
                  rationale=d.get("rationale", ""), evidence=evidence,
                  evidence_refs=evidence_refs,
                  latency_s=g.latency_s, tok_per_s=g.tok_per_s, rss_mb=g.rss_mb,
                  prompt_words=len(prompt.split()), raw=g.text)
