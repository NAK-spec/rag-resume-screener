"""RAG screening pipeline: requirements -> retrieve evidence -> LLM scoring."""
import json
import re
from dataclasses import dataclass, field

from .config import TOP_K
from .index import ResumeIndex
from .ingest import split_requirements

SYSTEM = (
    "You are an objective resume screening assistant. Judge ONLY from the "
    "evidence provided. Ignore name, gender, age, nationality and photo. "
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
    matched: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    rationale: str = ""
    evidence: dict = field(default_factory=dict)   # requirement -> [chunk texts]
    latency_s: float = 0.0
    tok_per_s: float = 0.0
    rss_mb: float = 0.0
    prompt_words: int = 0
    raw: str = ""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        sc = re.search(r'"?score"?\s*[:=]\s*(\d{1,3})', text)
        return {"score": int(sc.group(1))} if sc else {}


def build_prompt(jd: str, requirements, evidence: dict) -> str:
    ev = []
    for i, r in enumerate(requirements, 1):
        snippets = " ... ".join(evidence[r])
        ev.append(f"[{i}] Requirement: {r}\n    Evidence: {snippets}")
    return (f"JOB DESCRIPTION:\n{jd}\n\nRETRIEVED RESUME EVIDENCE:\n"
            + "\n".join(ev) + f"\n\nReturn JSON in this format:\n{SCHEMA}")


def screen_resume(resume_id: str, jd: str, index: ResumeIndex, llm,
                  mode: str = "rag", full_text: str = "", top_k: int = TOP_K) -> Result:
    """mode='rag'  -> retrieve top-k chunks per requirement (efficient)
       mode='full' -> stuff the whole resume into the prompt (baseline)."""
    reqs = split_requirements(jd) or [jd]
    evidence, req_scores = {}, []
    if mode == "rag":
        for r in reqs:
            hits = index.search(r, resume_id, k=top_k)
            evidence[r] = [c.text for _, c in hits]
            req_scores.append(max((s for s, _ in hits), default=0.0))
        prompt = build_prompt(jd, reqs, evidence)
    else:
        prompt = (f"JOB DESCRIPTION:\n{jd}\n\nFULL RESUME:\n{full_text}\n\n"
                  f"Return JSON in this format:\n{SCHEMA}")
        for r in reqs:
            hits = index.search(r, resume_id, k=1)
            req_scores.append(hits[0][0] if hits else 0.0)

    g = llm.generate(SYSTEM, prompt,
                     meta={"requirements": reqs, "req_scores": req_scores})
    d = parse_json(g.text)
    score = int(max(0, min(100, d.get("score", 0))))
    return Result(resume_id=resume_id, score=score,
                  matched=d.get("matched", []), missing=d.get("missing", []),
                  rationale=d.get("rationale", ""), evidence=evidence,
                  latency_s=g.latency_s, tok_per_s=g.tok_per_s, rss_mb=g.rss_mb,
                  prompt_words=len(prompt.split()), raw=g.text)
