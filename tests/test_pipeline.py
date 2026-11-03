"""Regression tests for the screening pipeline.

Run with pytest (`pip install pytest`, then `pytest tests -q`) or directly with no
extra dependency:  python tests/test_pipeline.py
"""
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import auth
from rag.index import Embedder, ResumeIndex
from rag.ingest import BULLET_RE, chunk_text, clean, extract_text, split_requirements
from rag.llm import GenResult, RetrievalOnlyLLM
from rag.screen import (Result, build_prompt, evidence_budget, jd_context,
                        normalize_requirements, parse_json, screen_resume,
                        select_evidence)

JD = ("Junior Data Scientist\n"
      "- Proficiency in Python and pandas for data analysis\n"
      "- Experience building machine learning models with scikit-learn\n"
      "- Knowledge of SQL and relational databases\n"
      "- Strong written and verbal communication skills")


def _index(resumes: dict):
    chunks = [c for rid, t in resumes.items() for c in chunk_text(rid, t)]
    idx = ResumeIndex(Embedder(prefer_dense=False))  # TF-IDF: no model download in tests
    idx.build(chunks, extra_corpus=[JD])
    return idx


LONG = "Python pandas scikit-learn SQL. " + " ".join(
    f"bullet {i} built {'predictive models with scikit-learn and Python dataframes' if i % 3 == 0 else ('dashboards in Tableau for stakeholder reviews' if i % 3 == 1 else 'reception rosters and petty cash paperwork')}"
    for i in range(90))


def test_title_is_not_a_requirement():
    reqs = split_requirements(JD)
    assert len(reqs) == 4, reqs
    assert not any("Junior Data Scientist" in r for r in reqs)


def test_numbered_list_keeps_text():
    reqs = split_requirements("Role: Analyst\n1. Python and pandas\n2. SQL reporting")
    assert reqs == ["Python and pandas", "SQL reporting"], reqs


def test_prose_jd_splits_on_sentences():
    reqs = split_requirements("Must build machine learning models. SQL fluency is required.")
    assert len(reqs) == 2, reqs


def test_chunk_overlap():
    words = [f"w{i}" for i in range(200)]
    cs = chunk_text("r", " ".join(words), size=90, overlap=20)
    assert len(cs) >= 3
    # the tail of one chunk is repeated as the head of the next
    assert cs[0].text.split()[-20:] == cs[1].text.split()[:20]
    assert cs[0].text.split()[0] == "w0" and cs[1].text.split()[0] == "w70"


def test_search_is_scoped_to_one_resume():
    idx = _index({"a.txt": "python pandas machine learning", "b.txt": "hotel reception"})
    hits = idx.search("python pandas", "b.txt", k=3)
    assert all(c.resume_id == "b.txt" for _, c in hits)
    assert len(hits) <= 3


def test_evidence_is_deduplicated_across_requirements():
    idx = _index({"a.txt": LONG})
    reqs = split_requirements(JD)[1:]
    hits = [idx.search(r, "a.txt", k=3) for r in reqs]
    raw = sum(len(h) for h in hits)
    passages = select_evidence(hits, max_words=10**6)
    assert len(passages) < raw, "top-k per requirement must not repeat chunks"
    assert len({_key(c) for _, c in passages}) == len(passages)


def _key(c):
    return (c.resume_id, c.idx)


def test_budget_is_a_fraction_of_the_resume():
    assert evidence_budget(0) == 400
    assert evidence_budget(50) == 50           # never more text than the document has
    assert evidence_budget(150) == 120         # floor keeps mid-length CVs intact
    assert evidence_budget(1000) == 400        # ceiling caps long ones
    assert evidence_budget(500) == 200         # 40% in between


def test_evidence_never_exceeds_the_document():
    """What the budget does guarantee: the resume text a RAG prompt carries is never more
    than the document itself. The whole *prompt* can still exceed the full-context prompt
    on a very short CV, because the numbered requirement block and the passage map are a
    fixed cost the baseline pays only once - that is documented in the README, not a bug
    this test can assert away."""
    short = ("Candidate C. Certificate in Hospitality Management. Hotel front desk "
             "for three years. Managed bookings and trained new staff. Skills: "
             "customer service, scheduling, Microsoft Word.")
    idx = _index({"c.txt": clean(short)})
    reqs = split_requirements(JD)
    for floor in (0.0, 0.024, 0.03):
        hits = [idx.search(r, "c.txt", k=3, min_score=floor) for r in reqs]
        passages = select_evidence(hits, evidence_budget(len(short.split())))
        words = sum(len(c.text.split()) for _, c in passages)
        assert words <= len(short.split()), (floor, words, len(short.split()))


def test_requirement_in_both_lists_counts_as_matched():
    class Contradictory:
        def generate(self, system, user, meta=None):
            reqs = (meta or {}).get("requirements", [])
            both = reqs[0]
            text = ('{"score": 50, "matched": ["%s"], "missing": ["%s"], '
                    '"rationale": ""}' % (both, both))
            return GenResult(text, 0.0, 1, 0.0)

    idx = _index({"a.txt": clean(LONG)})
    r = screen_resume("a.txt", JD, idx, Contradictory(), mode="rag", full_text=LONG)
    assert r.matched, r.matched
    assert not (set(r.matched) & set(r.missing)), (r.matched, r.missing)


def test_rag_prompt_is_shorter_than_full_context():
    idx = _index({"a.txt": clean(LONG)})
    rag = screen_resume("a.txt", JD, idx, RetrievalOnlyLLM(), mode="rag", full_text=LONG)
    full = screen_resume("a.txt", JD, idx, RetrievalOnlyLLM(), mode="full", full_text=LONG)
    assert rag.prompt_words < full.prompt_words, (rag.prompt_words, full.prompt_words)


def test_prompt_prints_each_requirement_once():
    idx = _index({"a.txt": LONG})
    hits = [idx.search(r, "a.txt", k=2) for r in split_requirements(JD)[1:]]
    p = build_prompt(JD, split_requirements(JD)[1:], hits, select_evidence(hits, 300))
    assert p.count("Proficiency in Python and pandas") == 1
    assert re.search(r"SUPPORTING PASSAGES PER REQUIREMENT:\n\[1\] ", p)


def test_jd_context_extracts_the_role_line():
    assert jd_context(JD, split_requirements(JD)) == "Junior Data Scientist"


def test_parse_json_variants():
    assert parse_json('{"score": 71}')["score"] == 71
    assert parse_json('noise {"score": 12, "matched": []} tail')["score"] == 12
    assert parse_json('score = 44 prose only')["score"] == 44
    assert parse_json("no json here") == {}


def test_fallback_score_is_in_range():
    idx = _index({"a.txt": LONG, "b.txt": "hotel front desk reception customer service"})
    for rid in ("a.txt", "b.txt"):
        r = screen_resume(rid, JD, idx, RetrievalOnlyLLM(), mode="rag", full_text=LONG)
        assert isinstance(r, Result) and 0 <= r.score <= 100


def test_text_extraction():
    assert extract_text("x.txt", "plain — text".encode("utf-8")) == "plain — text"
    assert isinstance(extract_text("x.txt", b"\xff\xfe broken"), str)  # never raises

    import io
    import docx
    d = docx.Document()
    d.add_paragraph("Built churn models with scikit-learn")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "SQL"
    t.rows[0].cells[1].text = "Tableau"
    buf = io.BytesIO()
    d.save(buf)
    out = extract_text("r.docx", buf.getvalue())
    assert "churn models with scikit-learn" in out
    assert "SQL | Tableau" in out  # table rows are flattened, not dropped


def test_auth_roundtrip_and_password_rules():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "users.db"
        auth.init_db(db)
        assert auth.create_user("alice", "shortie", db)[0] is False       # < 8 chars
        assert auth.create_user("al!ce", "password1", db)[0] is False     # bad chars
        ok, _ = auth.create_user("alice", "password1", db)
        assert ok and auth.create_user("alice", "password1", db)[1] == "That username is already taken."
        assert auth.verify_user("alice", "password1", db) == "alice"
        assert auth.verify_user("alice", "wrongpass", db) is None
        assert auth.verify_user("nobody", "password1", db) is None


def test_normalize_strips_echoed_numbering():
    reqs = split_requirements(JD)
    got = normalize_requirements(
        ["[1] Proficiency in Python and pandas for data analysis",
         "[4] Strong written and verbal communication skills",
         "Some free-form skill the model invented",
         "[9] out of range", 17], reqs)
    assert got == [reqs[0], reqs[3], "Some free-form skill the model invented",
                   "[9] out of range"], got


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
