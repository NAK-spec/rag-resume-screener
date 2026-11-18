"""Regression tests for the screening pipeline.

Run with pytest (`pip install pytest`, then `pytest tests -q`) or directly with no
extra dependency:  python tests/test_pipeline.py

Every test here is TF-IDF-only and loads no GGUF. 57 tests take about 27 s on the
CPU-only laptop the study ran on, and the planning arm accounts for most of it: re-playing
planning_eval.log means 17 forward-search passes over the 33 committed CVs, cached in one
process because eight tests read the same arms. The trained arms add a leave-one-out ridge
fit, a distillation fit and three learned-value planner passes. The dense path is covered by
test_embedder_for_refuses_a_silent_scale_swap, which forces the import to fail rather than
downloading a model.
"""
import math
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import metric_row, rank_metrics
from rag import auth
from rag import planner as P
from rag.config import SIM_FLOOR
from rag.index import Embedder, ResumeIndex, embedder_for
from rag.ingest import BULLET_RE, chunk_text, clean, extract_text, split_requirements
from rag.llm import (KEYWORD_MATCH_FRACTION, GenResult, KeywordCoverageLLM,
                     RetrievalOnlyLLM, TOKEN_RE, document_terms, keyword_coverage,
                     keyword_terms, requirement_groups, spelling_key)
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


def test_keyword_coverage_finds_what_chunk_cosine_cannot():
    """The claim the new scorer exists for: an exact keyword must not need a similar
    paragraph. TF-IDF cosine of one requirement against a 90-word chunk that contains
    it verbatim lands near 0.06, under the 0.175 the retrieval-only scorer calls a
    match - so a CV naming every tool in the requisition scores ~0."""
    cv = ("Data Scientist. Analysis of data with Python and pandas, machine learning "
          "models built with scikit-learn, SQL queries against relational databases, "
          "dashboards for stakeholders. " + " ".join(f"bullet {i}" for i in range(120)))
    idx = _index({"a.txt": cv})
    sim = screen_resume("a.txt", JD, idx, RetrievalOnlyLLM(), mode="rag", full_text=cv)
    key = screen_resume("a.txt", JD, idx, KeywordCoverageLLM(), mode="rag", full_text=cv)
    assert len(sim.matched) == 0, sim.matched
    assert {"Proficiency in Python and pandas for data analysis",
            "Knowledge of SQL and relational databases"} <= set(key.matched), key.matched
    assert key.score > sim.score


def test_keyword_coverage_folds_spelling_variants():
    """One letter apart in the corpus, the same requirement to a reader."""
    cv = "Trained data visualisation pipelines and published dashboards for review."
    req = "Experience with data visualization and dashboards"
    assert spelling_key("visualisation") == "visualization"
    assert keyword_coverage(req, {spelling_key(t) for t in TOKEN_RE.findall(cv)})[0] >= 0.6


def test_boilerplate_alone_never_counts_as_a_requirement():
    """A requirement with no content words has nothing to look for, and coverage of a
    empty term list is 0 - not 1. Treating it as matched would hand every CV a free
    point for the JD's own filler."""
    assert keyword_terms("Strong experience and knowledge required") == []
    assert keyword_coverage("Strong experience and knowledge required",
                            {"python", "sql"}) == (0.0, [])


def test_a_keyword_hit_is_a_word_not_a_substring():
    """PostgreSQL contains SQL, but a CV that only says PostgreSQL has not said SQL,
    and the scorer should not invent that evidence for itself."""
    terms = {spelling_key(t) for t in TOKEN_RE.findall("queries against PostgreSQL data")}
    assert "sql" not in terms
    assert keyword_coverage("Knowledge of SQL databases", terms)[0] < 0.6


def test_an_examples_list_is_one_ask_not_four():
    """The tool list in parentheses is a gloss on the skill, not four more things the CV
    has to repeat. Scored word by word, "(e.g. Power BI, Tableau, Plotly, Matplotlib)"
    put four names in the denominator, and someone who built dashboards without naming a
    tool fell under the bar for want of a brand."""
    req = ("Data visualisation and dashboards (e.g. Power BI, Tableau, Plotly, "
           "Matplotlib)")
    assert len(requirement_groups(req)) == 4          # data, visualisation, dashboards, list
    cov, absent = keyword_coverage(req, document_terms(
        "Reporting and data visualisation for dashboards, weekly."))
    assert cov == 0.75, (cov, absent)
    assert absent == ["Tableau"], absent               # the closest name, not all four


def test_a_list_of_names_is_one_ask_and_the_thing_above_it_is_still_an_ask():
    """Six acceptable degrees are one ask. The qualification itself stays a separate
    ask, so a CV that names a field but never says it holds the degree is not handed
    the requirement."""
    req = ("Bachelor's degree in Computer Science, Data Science, Statistics, "
           "Mathematics, Information Technology or Software Engineering")
    assert len(requirement_groups(req)) == 3           # bachelor, degree, one any-of list
    assert keyword_coverage(req, document_terms(
        "BSc Data Science, Makerere University."))[0] == 1 / 3
    assert keyword_coverage(req, document_terms(
        "Bachelor of Science in Computer Science."))[0] == 2 / 3


def test_an_inflected_word_still_answers_the_requirement():
    """A CV writes what it did, the requisition writes what it asks for. The same fold
    runs on both sides, so it can merge "training" with "trained" and nothing else."""
    for a, b in (("training", "trained"), ("querying", "queries"),
                 ("databases", "database"), ("tuning", "tuned"),
                 ("deploying", "deployed"), ("studies", "study"),
                 ("analysed", "analyzed"), ("models", "model"),
                 ("running", "run"), ("planning", "plan"), ("years", "year"),
                 ("visualisation", "visualization")):
        assert spelling_key(a) == spelling_key(b), (a, b)
    assert keyword_coverage(
        "Hands-on machine learning: training, tuning and evaluating models",
        document_terms("Built machine learning models, trained and tuned them, "
                       "evaluated them monthly."))[0] >= 0.6


def test_a_hyphen_and_an_apostrophe_are_not_part_of_a_word():
    """Two ways to write one phrase, and two ways to write one noun: both have to land
    on the same stems, or the scorer is matching spelling instead of skill."""
    assert document_terms("hands-on work") == document_terms("hands on work")
    assert document_terms("the candidate's model") == document_terms("the candidates model")


def test_a_jd_field_label_is_not_a_requirement():
    """The requisition's own metadata - the job title it is advertising, where the seat
    is - asks nothing of a candidate. Scoring it reported every candidate as missing the
    job's title, the way a bare title line used to be scored as a 7th requirement."""
    assert split_requirements(
        "Job Title: Junior Data Scientist\nLocation: Kampala, remote\n"
        "Hands-on machine learning: training, tuning and evaluating models") == [
        "Hands-on machine learning: training, tuning and evaluating models"]


def test_eg_does_not_cut_a_requirement_in_two():
    """Sentence-splitting a JD cut "dashboards (e.g." at its abbreviation and left the
    tool list standing as a requirement of its own, which then counted against the
    denominator of both halves."""
    reqs = split_requirements(
        "Data visualisation and dashboards (e.g. Power BI, Tableau, Plotly, "
        "Matplotlib)\nSQL for querying relational databases")
    assert len(reqs) == 2, reqs
    assert reqs[0].endswith("Matplotlib)"), reqs[0]


def test_keyword_coverage_is_evidence_independent():
    """The claim behind keyword_coverage_eval.log: this scorer reads the document, not
    the evidence block. In full mode nothing is retrieved at all and the score is
    unchanged - which is exactly why its rho is not a RAG result."""
    cv = LONG
    idx = _index({"a.txt": cv})
    rag = screen_resume("a.txt", JD, idx, KeywordCoverageLLM(), mode="rag", full_text=cv)
    full = screen_resume("a.txt", JD, idx, KeywordCoverageLLM(), mode="full", full_text=cv)
    assert rag.score == full.score and rag.matched == full.matched
    assert not any(full.evidence.values()) and any(rag.evidence.values())


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


DATA = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def _study_index():
    """The index exactly as evaluate.py builds it: every committed resume chunk, with the
    job description in the IDF corpus. Cached because three tests want the same one."""
    jd = (DATA / "job_description.txt").read_text(encoding="utf-8")
    resumes = {p.name: clean(extract_text(p.name, p.read_bytes()))
               for p in sorted((DATA / "resumes").iterdir())}
    chunks = [c for rid, t in resumes.items() for c in chunk_text(rid, t)]
    idx = ResumeIndex(Embedder(prefer_dense=False))
    idx.build(chunks, extra_corpus=[jd])
    return jd, resumes, idx


def test_floor_rejects_the_spurious_evidence_that_broke_rag():
    """The mechanism the README credits for making RAG usable, pinned to the dataset.

    weak_candidate.txt is the hospitality CV (human label 10). Top-k always returns
    *some* chunk, so with no floor all six requirements are handed a passage - including
    "machine learning" at similarity 0.005 - and the scorer is given nothing to reject.
    At the swept TF-IDF floor one requirement survives. Without this test the floor
    looks like a free tuning knob; with it, the knob that stopped a label-10 CV being
    described as a six-for-six match is fixed.
    """
    jd, _, index = _study_index()
    reqs = split_requirements(jd)
    rid = "weak_candidate.txt"
    assert len(reqs) == 6, reqs

    def evidenced(floor):
        return sum(1 for r in reqs if index.search(r, rid, k=3, min_score=floor))

    assert evidenced(0.0) == 6, "unfiltered top-k evidences every requirement"
    assert evidenced(SIM_FLOOR["tfidf"]) == 1, "only genuine lexical support should survive"
    ml = next(r for r in reqs if "machine learning" in r)
    best = max(s for s, _ in index.search(ml, rid, k=3, min_score=0.0))
    assert 0.0 < best < SIM_FLOOR["tfidf"], (best, SIM_FLOOR["tfidf"])


def test_a_requirement_the_floor_rejected_is_never_reported_matched():
    jd, resumes, index = _study_index()
    rid = "weak_candidate.txt"
    r = screen_resume(rid, jd, index, RetrievalOnlyLLM(), mode="rag",
                      full_text=resumes[rid], top_k=3)
    unevidenced = [q for q, refs in r.evidence_refs.items() if not refs]
    assert unevidenced, "most of this CV should come back with no supporting passage"
    assert not (set(unevidenced) & set(r.matched)), (unevidenced, r.matched)
    assert set(unevidenced) <= set(r.missing), (unevidenced, r.missing)
    # the coverage column is the matched list as a percentage, not the model's guess
    assert r.score_coverage == round(100 * len(r.matched) / len(r.evidence_refs))


def test_embedder_for_refuses_a_silent_scale_swap():
    """The app may fall back quietly; a study run must not.

    A run that meant to be dense and got TF-IDF would score its evidence against 0.24 on
    a 0.024 scale, which filters almost nothing, and the CSV would still read as a dense
    row. embedder_for makes that combination raise instead.
    """
    assert embedder_for("tfidf").kind == "tfidf"
    saved = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = None      # blocks the import inside Embedder
    try:
        assert Embedder(prefer_dense=True).kind == "tfidf"   # app: silent fallback
        assert embedder_for("auto").kind == "tfidf"          # auto: silent fallback
        try:
            embedder_for("dense")
        except RuntimeError as e:
            assert "dense requested" in str(e) and "tfidf was built" in str(e), e
        else:
            raise AssertionError("embedder_for('dense') must not downgrade silently")
    finally:
        if saved is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = saved


def test_the_two_floors_are_on_different_scales():
    """README: the floor values are not interchangeable. TF-IDF unit-vector cosine and
    MiniLM cosine do not share a range, so one global threshold cannot serve both, and
    copying either value into the other slot is a silent mis-tune rather than an error."""
    assert set(SIM_FLOOR) == {"tfidf", "dense"}, SIM_FLOOR
    assert 0 < SIM_FLOOR["tfidf"] < SIM_FLOOR["dense"] < 1
    assert SIM_FLOOR["dense"] / SIM_FLOOR["tfidf"] >= 5


def test_the_significance_rho_is_spearmans_rho():
    """Every p-value and interval in significance.log comes from the script's own
    vectorised Spearman, not scipy's. These scorers emit between 4 and 24 distinct values
    over 33 candidates, so tie handling is not a detail - if it drifted from scipy's
    average-rank definition the log would still print plausible numbers."""
    import numpy as np
    from scipy.stats import spearmanr

    from significance import _rho_rows, loo_range
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(60):
        x = rng.integers(0, 5, 33).astype(float)
        y = rng.integers(0, 8, 33).astype(float)
        if len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
            continue
        assert abs(_rho_rows(x[None, :], y[None, :])[0] - spearmanr(x, y)[0]) < 1e-12
        checked += 1
    assert checked > 30, f"the tie-heavy draw was degenerate too often: {checked}"

    gold = np.arange(33, dtype=float)
    assert _rho_rows(gold[None, :], (2 * gold)[None, :])[0] == 1.0
    assert np.isnan(_rho_rows(np.zeros((1, 33)), gold[None, :])[0]), "a constant scorer has no rank"
    lo, hi = loo_range(gold, gold)
    assert lo == hi == 1.0, (lo, hi)


def test_the_power_simulator_is_calibrated():
    """power_analysis.py's claims rest on three generator properties, none of which numpy
    checks on its own. An asymmetric correlation matrix is factorised from its lower triangle
    alone, so the paired generator can silently produce a different dependence than the one
    asked for; a damaged marginal would quietly change the tie structure being simulated."""
    import numpy as np

    from power_analysis import at, curve, paired_delta, assign

    rng = np.random.default_rng(7)
    pool = np.sort(np.array([80.0] * 19 + [60.0] * 10 + [40.0, 20.0, 10.0, 5.0]))
    labels = np.sort(np.arange(33.0) * 3)

    rows = assign(rng.standard_normal((25, len(pool))), pool, rng)
    assert all(sorted(r) == sorted(pool) for r in rows), "the simulated scorer lost its ties"

    null_d, clipped = paired_delta(0.6, 0.6, 0.5, pool, pool, labels, 400, rng)
    assert abs(float(np.nanmean(null_d))) < 0.05, f"H0 delta drifted to {np.nanmean(null_d):+.3f}"

    c = curve(pool, labels, 400, rng)
    assert c["mean"][0] < 0.05, "independent latents correlated"
    assert c["mean"][1] < c["ceiling"], "the calibration grid is not monotone"
    assert at(c, c["ceiling"] + 0.5)["clamped"], "an impossible target was accepted silently"


def test_the_provenance_replay_is_the_ground_truth_of_retrieval_quality():
    """retrieval_quality.py measures delivery against bullets the generator says it wrote, so
    two properties have to hold before any recall number means anything: every bullet must be
    findable in its own document (otherwise the replay and the corpus are different text), and
    pasting the whole document must reproduce the rubric score that labels.csv holds for that
    CV (otherwise the span arithmetic or the rubric link is wrong). Neither needs an embedder,
    so this runs in about a second."""
    from retrieval_quality import Ground, ceiling_of, load_generator, measure

    gd = load_generator()
    rows = gd.build(30, 7)
    assert len(rows) == 30
    exact, labels_ok = 0, 0
    for name, _tier, credits, text, prov in rows:
        g = Ground(name, clean(text), credits, prov, gd.TYPOS)
        assert not g.missing, f"{name}: {g.missing[0][1][:50]!r} is not in its own document"
        # every bullet lives on exactly one line, which is what makes a span a span
        for (s, e), raw in g.bullets + [(f, f) for f in g.filler]:
            assert g.words[s] == "-", f"{name}: bullet does not start at a list marker"
        whole_doc = [(0, g.n_words)]
        ceiling = ceiling_of(g, measure(g, whole_doc), gd.rubric_score)
        labels_ok += int(ceiling == gd.rubric_score(credits))
        exact += int(all(any(a <= s and e <= b for a, b in whole_doc)
                         for (s, e), _ in g.bullets))
    assert exact == 30 and labels_ok == 30, (exact, labels_ok)

    # the span helpers the word counts are built from
    from retrieval_quality import covered, merge

    assert merge([(0, 90), (70, 160), (140, 185)]) == [(0, 185)]
    assert merge([(0, 10), (20, 30)]) == [(0, 10), (20, 30)]
    assert covered((5, 15), [(0, 10), (20, 30)]) == 5
    assert covered((0, 100), [(0, 185)]) == 100, "the reference case must never lose a word"


def _metric_row(scores, latencies=None):
    latencies = latencies or [0.5] * len(scores)
    results = [SimpleNamespace(resume_id=f"r{i}.txt", score=s, score_coverage=float(s),
                               latency_s=latencies[i], tok_per_s=12.0,
                               rss_mb=1500.0, prompt_words=170)
               for i, s in enumerate(scores)]
    labels = {f"r{i}.txt": float(10 * (i + 1)) for i in range(len(scores))}
    return metric_row(RetrievalOnlyLLM(), "Retrieval-only (no LLM)", "rag",
                      results, labels, "tfidf")


def test_a_constant_scorer_has_no_rank():
    gold = [10.0, 20.0, 30.0, 40.0, 50.0]
    rho, tau = rank_metrics([85] * 5, gold)
    assert math.isnan(rho) and math.isnan(tau)
    assert abs(rank_metrics(gold, gold)[0] - 1.0) < 1e-9


def test_degeneracy_is_reported_not_hidden():
    """32 of 33 candidates on one score is the failure mode this study has to show.

    rho from such a run is still *computable* - it is carried by the one outlier - so the
    guard is the published modal_share, not a suppressed correlation.
    """
    spike = _metric_row([85] * 32 + [20])
    assert spike["n_distinct_scores"] == 2 and spike["modal_share"] > 0.5
    assert not math.isnan(spike["spearman"]), "computable is not the same as trustworthy"
    flat = _metric_row([85] * 33)
    assert math.isnan(flat["spearman"]) and flat["modal_share"] == 1.0
    spread = _metric_row(list(range(10, 100, 10)))
    assert spread["n_distinct_scores"] == 9 and spread["modal_share"] == 0.111


def test_retrieval_only_rows_do_not_report_throughput():
    """A fallback score is produced in microseconds, so tokens/second is a division
    artefact: an earlier run of the harness published 1.1 million tok/s for rows that
    never generated anything."""
    fast = _metric_row([50, 60, 70], latencies=[0.0005, 0.0008, 0.0009])
    assert math.isnan(fast["avg_tok_per_s"]), fast["avg_tok_per_s"]
    slow = _metric_row([50, 60, 70], latencies=[1.0, 2.0, 3.0])
    assert slow["avg_tok_per_s"] == 12.0


def test_role_year_ranges_never_run_backwards():
    """data/generate_dataset.py walks a candidate's roles back from PRESENT. The clamp it
    used to guard graduation year could return a start later than its own end once the
    cursor stepped below it, which is how four shipped CVs came to print "2024-2023"."""
    sys.path.insert(0, str(DATA))
    try:
        from generate_dataset import PRESENT, role_years
    finally:
        sys.path.pop(0)
    assert role_years(2023, 2024, 1) == (2023, 2023), "cursor below graduation year"
    for grad in (2016, 2021, 2024, 2025):
        for span in (1, 2, 3, 4):
            cursor, previous_start = PRESENT, None
            for _ in range(8):
                start, end = role_years(cursor, grad, span)
                assert start <= end, (grad, span, cursor, start, end)
                assert end <= cursor
                if previous_start is not None:
                    assert end < previous_start, "roles must not overlap or repeat a year"
                cursor, previous_start = start - 1, start


def test_the_geometry_sweep_reads_a_chunks_position_from_the_same_step_as_chunk_text():
    """geometry_sweep.py does not re-find a chunk inside the document: it derives the chunk's
    word span as `idx * (size - overlap)`, and every recall figure in `geometry_sweep.log` is
    computed from those spans. The derivation is only honest while `chunk_text` uses the same
    step, starts chunk i at i*step, and leaves no gap between consecutive windows - so pin all
    three over the geometries that sweep walks, including the ones whose overlap is larger than
    half the window and the one-chunk-per-CV reference row."""
    import geometry_sweep as gs

    docs = {"a": clean(JD + " " + " ".join(f"w{i}" for i in range(200))),
            "b": clean("one two three"),
            "c": clean(" ".join("xyzw" for _ in range(90)))}
    gs.KINDS["tfidf"] = {"embedder": Embedder(prefer_dense=False), "jd": JD,
                         "reqs": split_requirements(JD)}
    for label, size, overlap in gs.GEOMETRIES:
        geo = gs.build_geometry("tfidf", docs, size, overlap)
        for rid, text in docs.items():
            words = text.split()
            spans = sorted(geo.spans[rid].items())
            assert spans, f"{label}: {rid} chunked to nothing"
            for idx, (lo, hi) in spans:
                mine = [c for c in geo.index.chunks if c.resume_id == rid and c.idx == idx]
                assert len(mine) == 1, f"{label}: {rid}#{idx} is not one chunk"
                assert " ".join(words[lo:hi]) == mine[0].text, (label, rid, idx)
            windows = sorted(geo.spans[rid].values())
            assert windows[0][0] == 0 and windows[-1][1] == len(words), (label, rid, windows)
            assert all(b[0] <= a[1] for a, b in zip(windows, windows[1:])), \
                f"{label}: {rid} has a gap"


def test_the_floor_ladder_carries_the_committed_floor_and_the_rho_rule_is_checkable():
    """`geometry_sweep.py` sections 8-9 re-fit SIM_FLOOR per chunking with `best_floor`, so two
    things have to hold by construction. The ladder must contain the committed value - otherwise
    the shipped row is not one of the candidates and the re-fit cannot be checked against it -
    and it must stop at that embedder's own SIM_ANCHORS ceiling, the score the fallback mapping
    calls a perfect match: a floor above it would filter away the evidence the rubric itself
    scores 100%. Then the ranking rule: `sweep_floor.py` picked the floor by the strongest
    evidence gap, the ladder ranks by the ceiling rubric applied to the delivered evidence - the
    same quantity one stage earlier - and a tie has to go to the shorter prompt, never to a NaN
    ceiling."""
    import geometry_sweep as gs
    from rag.llm import SIM_ANCHORS

    for kind, ladder in gs.LADDER.items():
        assert ladder[0] == 0.0, f"{kind}: no floor-0 rung, so the un-floored row is unmeasured"
        assert SIM_FLOOR[kind] in ladder, f"{kind}: {SIM_FLOOR[kind]} is not a candidate"
        assert ladder == sorted(set(ladder)), f"{kind}: ladder is not strictly increasing"
        assert ladder[-1] <= SIM_ANCHORS[kind][1], (kind, ladder[-1], SIM_ANCHORS[kind])

    def cell(rho, prompt):
        return {"rho": rho, "prompt": prompt}

    lad = {("g", 0.01): cell(0.5, 200.0), ("g", 0.02): cell(0.5, 180.0),
           ("g", 0.03): cell(float("nan"), 10.0), ("g", 0.04): cell(0.7, 400.0)}
    fl, chosen = gs.best_floor(lad, "g", [0.01, 0.02, 0.03, 0.04])
    assert (fl, chosen["prompt"]) == (0.04, 400.0), "a NaN ceiling must never be the best"
    lad2 = {k: v for k, v in lad.items() if k[1] != 0.04}
    fl2, chosen2 = gs.best_floor(lad2, "g", [0.01, 0.02, 0.03])
    assert (fl2, chosen2["prompt"]) == (0.02, 180.0), "a tie has to go to the shorter prompt"


def test_the_runner_gate_expectations_are_the_sweeps_own_tfidf_rows():
    """.github/workflows/score-geometry.yml will not load a model until a retrieval-only pass at
    its own chunking reproduces the numbers written in `geometry_sweep.log`, so the workflow
    carries a copy of that log: mean passages, mean prompt words, chunk count per matrix row. A
    copy is the thing that drifts. Re-run the sweep and edit one gate figure to whatever the new
    run prints, and the gate becomes a check that the pipeline agrees with itself - it would pass
    on a corpus that no longer has the geometry it claims. So read the log the way the workflow
    reads its own dump and require every row to be that log's tfidf section-2 row: the chunking
    must still produce the chunk count `chunk_text` produces on the committed resumes, and the
    control job must still be the shipped 90/20 the rest of the study runs at. The weights pin is
    checked against the GGUF that produced results_v2.csv when that file is on disk, because a run
    whose model bytes differ is a different model wearing the same name."""
    import re
    import yaml

    root = DATA.parent
    wf = yaml.safe_load((root / ".github" / "workflows" / "score-geometry.yml")
                        .read_text(encoding="utf-8"))
    matrix = wf["jobs"]["score"]["strategy"]["matrix"]["include"]
    log = (root / "geometry_sweep.log").read_text(encoding="utf-8-sig")
    # section 2 repeats once per embedder; the gate is a tfidf run, so only the tfidf half counts
    tfidf = log.split("embedder: dense")[0]
    rows = {}
    for line in tfidf.splitlines():
        m = re.match(r"\s+chunk (\d+)/(\d+)(?: \(shipped\))?\s+(\S+)\s+(\S+)\s+(\S+)\s", line)
        if m:
            size, overlap, passages, _pasted, prompt = m.groups()
            assert f"{size}/{overlap}" not in rows, f"section 2 prints {size}/{overlap} twice"
            rows[f"{size}/{overlap}"] = (float(passages), float(prompt))
    assert set(rows) >= {"90/20", "45/10"}, sorted(rows)

    _jd, resumes, _idx = _study_index()
    seen = []
    for cell in matrix:
        geo = f"{cell['chunk_words']}/{cell['chunk_overlap']}"
        seen.append(geo)
        assert geo in rows, f"{cell['label']}: {geo} is not a row of the sweep's section 2"
        passages, prompt = rows[geo]
        assert float(cell["expect_passages"]) == passages, (cell["label"], geo, passages)
        assert float(cell["expect_prompt_words"]) == prompt, (cell["label"], geo, prompt)
        n = sum(len(chunk_text(rid, t, cell["chunk_words"], cell["chunk_overlap"]))
                for rid, t in resumes.items())
        assert int(cell["expect_chunks"]) == n, (cell["label"], geo, n)

    assert seen.count("45/10") == 2 and seen.count("90/20") == 1, seen
    gguf = Path(wf["env"]["GGUF_FILE"])
    if (root / "models" / gguf).exists():
        import hashlib
        h = hashlib.sha256()
        size = 0
        with (root / "models" / gguf).open("rb") as fh:      # streamed: 1.1 GB is not a read_bytes
            for block in iter(lambda: fh.read(1 << 22), b""):
                size += len(block)
                h.update(block)
        assert size == int(wf["env"]["GGUF_BYTES"]), gguf
        assert h.hexdigest() == wf["env"]["GGUF_SHA256"], \
            f"{gguf}: the workflow pins different weights than the model on disk"


def test_the_keyword_coverage_log_is_what_the_scorers_still_produce():
    """README and ARTIFACTS quote rho +0.787 for keyword coverage and +0.306 for
    retrieval-only. Neither is a model run, so both are properties of rag/llm.py: an edit
    to the ask rules, the filler list or the stem folds silently invalidates the quoted
    number. Re-play both no-model scorers over the committed 33 CVs and require the log's
    own two rows to still be what the code produces."""
    sys.path.insert(0, str(DATA.parent))
    import keyword_coverage_eval as kw

    log = (DATA.parent / "keyword_coverage_eval.log").read_text(encoding="utf-8-sig")
    pairs = kw.corpus()
    index = ResumeIndex(embedder_for("tfidf"))
    index.build([c for rid, t in pairs for c in chunk_text(rid, t)], extra_corpus=[kw.JD])
    gold = [kw.LABELS[rid] for rid, _ in pairs]
    for name, llm in (("Retrieval-only (chunk similarity)", RetrievalOnlyLLM()),
                      ("Keyword coverage (whole document)", KeywordCoverageLLM())):
        results = [screen_resume(rid, kw.JD, index, llm, mode="rag", full_text=t,
                                 top_k=kw.TOP_K) for rid, t in pairs]
        rho, _ = rank_metrics([r.score for r in results], gold)
        m = re.search(re.escape(name) + r"\s+\d+\s+(-?\d\.\d{3})", log)
        assert m, f"{name} is not a row of keyword_coverage_eval.log"
        assert abs(rho - float(m.group(1))) < 5e-4, (name, rho, m.group(1))


# ---------------------------------------------------------------------------
# the planning arm (rag/planner.py) and the trained arms (rag/learn.py)
# ---------------------------------------------------------------------------

def _log_table(log, after, before):
    """The rows of the whitespace-aligned table printed between two markers, header
    dropped, each row split into tokens.

    Column names are not recovered, because pandas pads a multi-word header into more
    tokens than it has columns. Every caller below counts its own columns from the right
    and asserts the row count, so a reordered or added column shows up as a wrong number
    rather than a silently skipped row.
    """
    body = log.split(after, 1)[1].split("\n", 1)[1].split(before)[0]
    lines = [l for l in body.splitlines() if l.strip()]
    rows = [l.split() for l in lines[1:]]
    assert rows, f"no table printed between {after!r} and {before!r}"
    return rows


def _matches(token, value):
    """Does a freshly measured number equal the log's cell at the precision the log prints?

    Reading the decimals from the token means the test follows the log's own rounding
    instead of inventing a tolerance of its own.
    """
    dp = len(token.split(".")[1]) if "." in token else 0
    return round(float(value), dp) == float(token)


@lru_cache(maxsize=1)
def _planning_study():
    """The planner exactly as planning_eval.py runs it: same corpus order, same budgets,
    same beams, same scoring rule. Cached, because eight tests want these arms and one
    pass over the 33 CVs costs about half a second."""
    sys.path.insert(0, str(DATA.parent))
    import planning_eval as PE
    from rag.config import TOP_K

    jd = (DATA / "job_description.txt").read_text(encoding="utf-8")
    pairs = tuple(PE.corpus())
    reqs = split_requirements(jd)
    index = ResumeIndex(embedder_for("tfidf"))
    index.build([c for rid, t in pairs for c in chunk_text(rid, t)], extra_corpus=[jd])
    labels = PE.LABELS

    def planned(beam, mult, order=None, **kw):
        if order is None:
            return tuple(P.plan_evidence(rid, jd, index, t, beam=beam,
                                         budget=int(round(evidence_budget(len(t.split()))
                                                          * mult)), **kw)
                         for rid, t in pairs)
        return tuple(P.plan_evidence(rid, jd, index, t, beam=beam, order=order,
                                     budget=int(round(evidence_budget(len(t.split()))
                                                      * mult)), **kw)
                     for rid, t in pairs)

    # `beam{b}x{m}` follows the module default, exactly as the log's section-2 grid does, so
    # flipping planner.DEFAULT_ORDER moves the test and the log together. The two named
    # orderings are section 3b's, which prints both whatever the default is.
    arms = {f"beam{beam}x{mult}": planned(beam, mult)
            for beam in PE.BEAMS for mult in PE.BUDGET_MULT}
    other = {"goal": "coverage", "coverage": "goal"}[P.DEFAULT_ORDER]
    for mult in PE.BUDGET_MULT:
        arms[f"{P.DEFAULT_ORDER}_x{mult}"] = arms[f"beam8x{mult}"]
        arms[f"{other}_x{mult}"] = planned(8, mult, order=other)
    arms["never_reask"] = planned(8, 1.0, relax_rounds=0)

    fixed = []
    for rid, t in pairs:
        hits = [index.search(r, rid, k=TOP_K, min_score=SIM_FLOOR["tfidf"]) for r in reqs]
        kept = select_evidence(hits, evidence_budget(len(t.split())))
        fixed.append(PE.coverage_result(rid, document_terms(
            " ".join(c.text for _, c in kept)), reqs))
    return SimpleNamespace(PE=PE, jd=jd, pairs=pairs, reqs=reqs, index=index,
                           labels=labels, gold=[labels[rid] for rid, _ in pairs],
                           arms=arms, fixed=fixed,
                           whole=[PE.coverage_result(rid, document_terms(t), reqs)
                                  for rid, t in pairs])


@lru_cache(maxsize=1)
def _trained_study():
    """Both trained arms as trained_eval.py runs them, built once: the feature matrices,
    the leave-one-out screener, the value model's held-out quality and the planner driven
    by learned values. Cached because the poison tests need several fits from one feature
    build and the log-agreement test needs the same numbers again."""
    import numpy as np

    sys.path.insert(0, str(DATA.parent))
    import trained_eval as TE
    from rag import learn as L

    st = _planning_study()
    labels = st.labels
    y = np.array([labels[rid] for rid, _ in st.pairs], float)
    rows = tuple(L.screener_features(st.index, rid, st.reqs, t) for rid, t in st.pairs)
    X, names = L._vectorise(rows, None)
    Xr, names_r = L._vectorise(rows, L.RETRIEVAL_ONLY)
    pred, insample, _models, _full = L.loo_predict(X, y)
    Xv, yv, owner = L.value_rows(st.index, st.reqs, st.pairs)
    learned = {}
    for mult in TE.BUDGET_MULT:
        store = []
        for pos, (rid, t) in enumerate(st.pairs):
            vals = L.value_of_chunk(L.fit_value_model(Xv, yv, owner, pos),
                                    st.index, st.reqs, rid, t)
            store.append(P.plan_evidence(rid, st.jd, st.index, t, beam=TE.BEAM,
                                         budget=int(round(evidence_budget(len(t.split()))
                                                          * mult)), values=vals))
        learned[mult] = tuple(store)
    return SimpleNamespace(TE=TE, L=L, st=st, pairs=st.pairs, y=y, rows=rows,
                           X=X, names=names, Xr=Xr, names_r=names_r, pred=pred,
                           insample=insample, Xv=Xv, yv=yv, owner=owner,
                           learned=learned)


def test_a_planned_prompt_spends_only_the_budget_it_was_given():
    """The `read` action's precondition is the same word budget select_evidence gets, so
    the planner and the shipped policy are handed identical resources. The one documented
    exception is the first read, exempt so a plan is never empty - and when it triggers,
    the passage that got in has to be the cheapest readable one, which is what planner.py's
    exemption condition actually guarantees. Readable means `plan.pool`, the set the
    winning plan searched: a later `relax` round can widen the asking without changing the
    plan. The second half checks the budget binds at all:
    too small to fit any passage, the plan is still exactly one passage long."""
    st = _planning_study()
    for key, store in st.arms.items():
        for p in store:
            if p.words_spent <= p.budget:
                continue
            assert p.reads == 1, (key, p.resume_id, p.words_spent, p.budget, p.reads)
            assert p.words_spent == min(len(c.text.split()) for c in p.pool), \
                (key, p.resume_id, "an exempted first read must be the cheapest passage")
    for p in st.arms["beam8x1.0"]:
        text = dict(st.pairs)[p.resume_id]
        assert p.budget == evidence_budget(len(text.split())), p.resume_id

    idx = _index({"tiny": LONG})
    tight = P.plan_evidence("tiny", JD, idx, LONG, beam=8, budget=10)
    assert tight.reads == 1 and tight.words_spent > tight.budget, \
        (tight.reads, tight.words_spent, tight.budget)
    assert tight.stop_reason.startswith("budget spent"), tight.stop_reason


def test_a_plan_can_be_replayed_action_by_action():
    """Plan.steps is advertised as replayable from the empty state, and the log's per-step
    numbers come from that replay. Re-run the transitions here, without the search, and
    require every recorded word count, score and unsatisfied count to be what the same
    actions produce again - and the replayed term set to be the plan's own."""
    st = _planning_study()
    for p in st.arms["beam8x1.0"]:
        kinds = [s.kind for s in p.steps]
        assert kinds[:len(st.reqs)] == ["explore"] * len(st.reqs), kinds
        assert kinds[len(kinds) - p.reads:] == ["read"] * p.reads, kinds
        assert kinds.count("relax") == p.relax_rounds, (p.resume_id, kinds)
        assert sum(1 for s in p.steps if s.kind == "read") == p.reads == len(p.path)
        label_of = {c.idx: lab for lab, c in p.passages}
        terms, words = set(), 0
        for s in [x for x in p.steps if x.kind == "read"]:
            cid = [i for i, lab in label_of.items() if lab == s.target]
            assert len(cid) == 1, (p.resume_id, s.target, label_of)
            chunk = [c for _, c in p.passages if c.idx == cid[0]][0]
            terms |= document_terms(chunk.text)
            words += len(chunk.text.split())
            covs = [keyword_coverage(r, terms)[0] for r in st.reqs]
            assert s.words_spent == words, (p.resume_id, s.target, words)
            assert s.score_after == round(100 * sum(covs) / len(covs)), \
                (p.resume_id, s.target, covs)
            assert s.unsatisfied_after == sum(1 for c in covs
                                              if c < KEYWORD_MATCH_FRACTION)
        assert frozenset(terms) == p.terms, p.resume_id
        if p.reads:
            assert p.steps[-1].score_after == p.score, (p.resume_id, p.steps[-1].score_after,
                                                        p.score)


def test_the_same_cv_gets_the_same_plan_twice():
    """A plan is quoted in a log and defended in a viva, so it has to be the plan the same
    code returns on a machine that has never seen this one. Nothing in the search is timed,
    seeded or hashed-order dependent: the ranking key ends in the sorted passage ids for
    exactly this reason."""
    st = _planning_study()
    for (rid, t), p in zip(st.pairs, st.arms["beam8x1.0"]):
        q = P.plan_evidence(rid, st.jd, st.index, t, beam=8, budget=p.budget)
        assert (q.path, q.score, q.words_spent, q.stop_reason) == \
            (p.path, p.score, p.words_spent, p.stop_reason), rid
        assert [s.detail for s in q.steps] == [s.detail for s in p.steps], rid


def test_only_the_goal_ranking_can_make_more_words_mean_a_lower_score():
    """A bigger budget widens the set of feasible plans, so no plan should score below what
    the same CV scored with fewer words. The shipped `coverage` ordering never does; the
    classical `goal` ordering does, three times in this grid, because ranking states by the
    satisfied-predicate count first lets a wider beam keep a passage that satisfies one
    requirement outright over one that spreads more partial credit. That number is the
    sharpest single cost of the classical choice in planning_eval.log section 3b, and it is
    why DEFAULT_ORDER is `coverage`, so it is pinned to the footnote that prints it - flip
    the default ordering and this test, and the log, both have to be re-read."""
    st = _planning_study()
    mults = (0.5, 1.0, 2.0, 4.0)

    def drops(arms):
        seq = [arms[m] for m in mults]
        return sum(1 for i in range(len(st.pairs))
                   for prev, nxt in zip(seq[:-1], seq[1:])
                   if nxt[i].score < prev[i].score)

    goal = drops({m: st.arms[f"goal_x{m}"] for m in mults})
    cover = drops({m: st.arms[f"coverage_x{m}"] for m in mults})
    assert cover == 0, "a budget is a floor under the score; the coverage ranking broke it"
    log = (DATA.parent / "planning_eval.log").read_text(encoding="utf-8-sig")
    m = re.search(r"earlier (\d+) times and `coverage` (\d+) times", log)
    assert m, "planning_eval.log no longer prints its budget-monotonicity footnote"
    assert (goal, cover) == (int(m.group(1)), int(m.group(2))), (goal, cover, m.groups())
    assert goal > 0, ("the goal ordering is no longer non-monotone anywhere: section 3b of "
                      "the log has lost its point, re-run it")


def test_re_asking_a_blind_floor_is_what_leaves_those_cvs_with_evidence():
    """The similarity floor is why a CV can be scored 9/100 with every tool named: top-k
    returns nothing at all for a requirement whose passages all sit under it, and a policy
    that cannot re-ask then plans over an empty pool. planner.py's `relax` action is the
    answer, and the two CVs it saves are named here, because the fix costs ranking
    (0.781 -> 0.777 in the log's section 5) and a future edit that drops it to pay that
    price back has to notice the two CVs coming back."""
    st = _planning_study()
    blind = [p.resume_id for p in st.arms["never_reask"] if not p.reads]
    assert blind == ["r04_weak.txt", "r06_weak.txt"], blind
    for p in st.arms["never_reask"]:
        if not p.reads:
            assert "similarity floor" in p.stop_reason, (p.resume_id, p.stop_reason)
    shipped = {p.resume_id: p for p in st.arms["beam8x1.0"]}
    for rid in blind:
        p = shipped[rid]
        assert p.relax_rounds >= 1 and p.reads == 1, \
            (rid, p.relax_rounds, p.reads, p.stop_reason)
        assert p.words_spent <= p.budget or p.reads == 1, (rid, p.words_spent, p.budget)


def test_noisy_or_never_undoes_evidence_already_gathered():
    """The learned-value search combines a passage's predicted per-requirement coverage
    with what it already believes using noisy-OR, because a second passage can only add
    evidence for an ask. If it could subtract, the planner's belief would be able to go
    backwards and the estimates would stop meaning anything."""
    cases = [((0.0, 0.5, 1.0, 0.25), (0.3, 0.0, 0.2, 0.9)),
             ((0.0, 0.0, 0.0, 0.0), (0.7, 0.7, 0.7, 0.7)),
             ((0.9, 0.9, 0.1, 0.7), (0.0, 0.0, 0.0, 0.0)),
             ((0.4, 0.6, 0.8, 0.0), (1.0, 1.0, 1.0, 1.0))]
    for a, b in cases:
        got = P._noisy_or(a, b)
        # + 1e-12 because the combine is float arithmetic: 1 - 0.9*1.0 is 0.09999999999999998
        assert all(x <= y + 1e-12 and y <= 1.0 for x, y in zip(a, got)), (a, b, got)
    assert all(math.isclose(x, y) for x, y in
               zip(P._noisy_or((0.4, 0.6), (0.0, 0.0)), (0.4, 0.6)))
    assert all(math.isclose(v, 1.0) for v in P._noisy_or((0.5, 0.5), (1.0, 1.0)))


def test_a_planned_result_is_the_record_the_app_and_the_metrics_already_read():
    """screen_resume_planned has to hand back the same Result the LLM run hands back, or
    the app and metric_row read a plan as a different kind of measurement. The score is the
    plan's, score_coverage is the matched count, matched and missing partition the
    requirements, and every prompt actually carries words."""
    st = _planning_study()
    res = [P.screen_resume_planned(rid, st.jd, st.index, t) for rid, t in st.pairs]
    for r, p in zip(res, st.arms["beam8x1.0"]):
        assert r.score == p.score, r.resume_id
        assert r.score_coverage == round(100 * len(p.matched) / len(st.reqs)), r.resume_id
        assert sorted(r.matched + r.missing) == sorted(st.reqs), r.resume_id
        assert not (set(r.matched) & set(r.missing)), r.resume_id
        assert r.prompt_words > 0 and r.raw.startswith("{"), r.resume_id
    rho = rank_metrics([r.score for r in res], st.gold)[0]
    row = metric_row(KeywordCoverageLLM(), "planned", "rag", res, st.labels, "tfidf")
    assert abs(row["spearman"] - rho) < 5e-4, (row["spearman"], rho)
    assert row["n"] == len(st.pairs) and row["avg_latency_s"] == 0.0, row


def test_a_plan_built_on_estimates_is_still_graded_by_the_term_rule():
    """The unbiased arm of the planner searches on a model's estimate of what a passage is
    worth and never consults the rule while choosing. That only measures anything if the
    passages it chose are then scored the same way every other arm is: plan.coverages is the
    rule over the terms actually read, plan.search_covs is what the model believed, and the
    two must differ somewhere or the arm quietly became the trained rule."""
    st = _trained_study()
    base, learned = st.st.arms["beam8x1.0"], st.learned[1.0]
    belief_changed, belief_differs = 0, 0
    for p, q in zip(base, learned):
        covs = [keyword_coverage(r, q.terms)[0] for r in st.st.reqs]
        assert q.coverages == covs, (q.resume_id, q.coverages, covs)
        if q.search_covs != q.coverages:
            belief_differs += 1
        if {c.idx for _, c in p.passages} != {c.idx for _, c in q.passages}:
            belief_changed += 1
    assert belief_differs, "the model's belief equals the rule everywhere: not a model run"
    assert belief_changed, "the learned value changed no passage choice: the arm is a no-op"


def test_the_ridge_screener_cannot_see_the_label_of_the_cv_it_scores():
    """Every rho in trained_eval.log is out-of-fold, which is the study's only defence
    against a scorer that memorised its 33 rows. Prove it the way a reviewer would ask:
    replace one CV's human label with its opposite and require that CV's prediction not to
    move by a bit, while the other 32 do move - so the poison really landed and really
    stayed outside the fold that scored its own row."""
    import numpy as np

    st = _trained_study()
    L = st.L
    pred, _ins, _m, _f = L.loo_predict(st.X, st.y)
    assert np.array_equal(pred, st.pred)
    for i in (0, len(st.pairs) // 2, len(st.pairs) - 1):
        poisoned = st.y.copy()
        poisoned[i] = 0.0 if st.y[i] > 50 else 100.0
        p2, *_ = L.loo_predict(st.X, poisoned)
        assert p2[i] == pred[i], (st.st.pairs[i][0], p2[i], pred[i])
        others = [abs(p2[j] - pred[j]) for j in range(len(pred)) if j != i]
        assert max(others) > 1e-6, "the poisoned label changed nothing anywhere"


def test_the_value_model_cannot_see_the_cv_it_is_fitted_on():
    """The value model is what removes planner.py's smuggled assumption, and trained_eval.log
    section 2 holds a CV out of every fit. Same test as the screener's, at the row the
    planner actually consumes: shift every label of the held-out CV's passages and require
    its predictions to be bit-identical, while a model allowed to train on them moves."""
    import numpy as np

    st = _trained_study()
    L, Xv, yv, owner = st.L, st.Xv, st.yv, st.owner
    for cv in (0, len(st.pairs) - 1):
        sel = owner == cv
        poisoned = yv.copy()
        poisoned[sel] = np.clip(yv[sel] + 0.5, 0.0, 1.0)
        assert not np.array_equal(poisoned[sel], yv[sel]), "the poison did not land"
        clean = L.fit_value_model(Xv, yv, owner, cv).predict(Xv[sel])
        swapped = L.fit_value_model(Xv, poisoned, owner, cv).predict(Xv[sel])
        assert np.array_equal(clean, swapped), cv
        inside = L.fit_value_model(Xv, poisoned, owner).predict(Xv[sel])
        assert not np.array_equal(clean, inside), \
            "a model trained on the changed labels did not change: the fit is not reading y"


def test_the_retrieval_signal_arm_excludes_the_term_rules_own_output():
    """trained_eval.log reports two trained screeners, and only the retrieval-signals arm can
    be read as learning to screen from retrieval alone: the full arm is given the term rule's
    own coverage, so it measures calibration. That split lives in one frozenset, so pin it -
    and pin the feature order the log's weight list is printed in."""
    st = _trained_study()
    L = st.L
    assert set(st.names) - set(st.names_r) == {"coverage mean", "coverage max",
                                               "fraction matched"}, (st.names, st.names_r)
    assert set(st.names_r) == set(L.RETRIEVAL_ONLY), st.names_r
    assert len(st.names) == len(st.rows[0]) == 10, st.names
    assert st.X.shape == (len(st.rows), len(st.names)), st.X.shape
    assert st.Xr.shape == (len(st.rows), len(st.names_r)), st.Xr.shape


def test_the_permutation_p_is_seeded_and_tightens_with_the_ranking():
    """n=33 is the whole dataset, so every rho in the trained log is quoted with a
    permutation p-value - a number a reader could otherwise not tell from a coin flip. It
    has to be the same number on every machine, so it is seeded, and it has to behave: a
    perfect ranking cannot have a looser p than a mis-ranked one, and p must stay on the
    (w+1)/(n+1) grid the function documents."""
    import numpy as np

    st = _trained_study()
    L, gold = st.L, st.y
    p1 = L.permutation_p(st.pred, gold, n=100)
    assert p1 == L.permutation_p(st.pred, gold, n=100), "the seed does not fix the draw"
    assert 0.0 < p1 <= 1.0 and abs(p1 * 101 - round(p1 * 101)) < 1e-9, p1
    loose = gold.copy()
    loose[0], loose[-1] = loose[-1], loose[0]
    perfect, ranked = L.permutation_p(gold, gold, n=100), L.permutation_p(loose, gold, n=100)
    assert perfect <= ranked, (perfect, ranked)
    assert abs(ranked - 1 / 101) < 1e-9 or ranked > perfect, (perfect, ranked)


def test_the_planning_log_is_what_the_planner_still_produces():
    """planning_eval.log is the artifact the planning arm is quoted from and no model wrote
    any row of it: each is rag/planner.py plus the shared term rule over the committed 33
    CVs. A change to the beam default, the budget formula, the relax rounds or the stem
    folds invalidates the quoted numbers silently, so re-play the policies, the budget x
    beam grid, the ranking ablation and the sensing rows against the log's own cells."""
    st = _planning_study()
    log = (DATA.parent / "planning_eval.log").read_text(encoding="utf-8-sig")

    rows = _log_table(log, "1) the three policies", "   no timing column")
    assert len(rows) == 3, rows
    replay = {"Retrieval-only (chunk similarity)":
              [screen_resume(rid, st.jd, st.index, RetrievalOnlyLLM(), mode="rag",
                             full_text=t, top_k=st.PE.TOP_K) for rid, t in st.pairs],
              "Keyword coverage (whole document)":
              [screen_resume(rid, st.jd, st.index, KeywordCoverageLLM(), mode="rag",
                             full_text=t, top_k=st.PE.TOP_K) for rid, t in st.pairs],
              "Fixed top-k passages (select_evidence)": st.fixed}
    # 7 numeric columns: n spearman kendall distinct modal_share mae within_10
    for tok in rows:
        name = " ".join(tok[:-7])
        res = replay[name]
        m = metric_row(KeywordCoverageLLM(), name, "rag", res, st.labels, "tfidf")
        assert _matches(tok[-6], m["spearman"]), (name, tok[-6], m["spearman"])
        assert _matches(tok[-2], m["mae"]), (name, tok[-2], m["mae"])

    rows = _log_table(log, "2) the planner: budget x beam width", "   satisfied = mean")
    assert len(rows) == len(st.PE.BEAMS) * len(st.PE.BUDGET_MULT), rows
    for tok in rows:
        assert len(tok) == 14, tok
        beam, mult = int(tok[0]), float(tok[1])
        store = st.arms[f"beam{beam}x{mult}"]
        res = [st.PE.coverage_result(p.resume_id, p.terms, st.reqs) for p in store]
        m = metric_row(KeywordCoverageLLM(), "planner", "rag", res, st.labels, "tfidf")
        assert _matches(tok[-12], m["spearman"]), tok[:2]
        assert _matches(tok[-10], m["mae"]), tok[:2]
        assert _matches(tok[-6], sum(p.reads for p in store) / len(store)), tok[:2]

    rows = _log_table(log, "3b) the choice that actually moves", "   more words can mean")
    assert len(rows) == 2 * len(st.PE.BUDGET_MULT), rows
    for tok in rows:
        assert len(tok) == 8, tok
        mult, order = float(tok[0]), tok[1]
        store = st.arms[f"{order}_x{mult}"]
        res = [st.PE.coverage_result(p.resume_id, p.terms, st.reqs) for p in store]
        m = metric_row(KeywordCoverageLLM(), order, "rag", res, st.labels, "tfidf")
        assert _matches(tok[2], m["spearman"]), tok[:2]
        assert _matches(tok[3], m["mae"]), tok[:2]

    rows = _log_table(log, "5) the sensing step on its own", "   the floor is the reason")
    assert len(rows) == 3, rows
    for tok in rows:
        label = " ".join(tok[:-6])
        if "never re-ask" in label:
            store = st.arms["never_reask"]
        elif "shipped arm" in label:
            store = st.arms["beam8x1.0"]
        else:
            store = tuple(P.plan_evidence(rid, st.jd, st.index, t, beam=8, floor=0.0)
                          for rid, t in st.pairs)
        res = [st.PE.coverage_result(p.resume_id, p.terms, st.reqs) for p in store]
        rho = rank_metrics([r.score for r in res], st.gold)[0]
        assert _matches(tok[-6], rho), label
        assert int(tok[-2]) == sum(1 for p in store if not p.reads), label


def test_the_trained_log_is_what_the_models_still_produce():
    """trained_eval.log quotes a leave-one-out rho, a held-out AUC and a learned-value
    planner arm, none of which came from a model download - all three are properties of
    rag/learn.py, rag/planner.py and the committed labels. Refitting them here and requiring
    every cell to match is what stops a silent change of feature set, penalty grid or label
    column from surviving into the defense."""
    st = _trained_study()
    L, TE, base = st.L, st.TE, st.st
    log = (DATA.parent / "trained_eval.log").read_text(encoding="utf-8-sig")

    rows = _log_table(log, "1) the trained screener", "   learned weights")
    assert len(rows) == 4, rows
    # 10 numeric columns: features spearman kendall mae within_10 distinct in-sample
    # overfit gap p alpha
    for tok, use in zip(rows[:2], (None, L.RETRIEVAL_ONLY)):
        res, info = L.train_screener(base.index, base.pairs, base.reqs, base.labels, use=use)
        m = metric_row(KeywordCoverageLLM(), "trained", "rag", res, base.labels, "tfidf")
        name = " ".join(tok[:-10])
        for back, value in ((10, len(info["features"])), (9, m["spearman"]),
                            (7, m["mae"]), (6, m["within_10"]),
                            (5, m["n_distinct_scores"]), (4, info["in_sample_rho"]),
                            (3, info["in_sample_rho"] - info["loo_rho"]), (2, info["p"])):
            assert _matches(tok[-back], value), (name, back, tok[-back], value)

    rows = _log_table(log, "2) can a model learn", "   feature weights")
    assert len(rows) == 1, rows
    q = L.value_quality(base.index, base.reqs, base.pairs)
    tok = rows[0][-8:]
    for cell, key in zip(tok, ("rows", "cvs", "pearson", "rho", "auc",
                               "mean abs err", "base rate", "alpha")):
        assert _matches(cell, q[key]), (key, cell, q[key])

    rows = _log_table(log, "3) the planner on the learned value", "   for reference")
    assert len(rows) == 2 * len(TE.BUDGET_MULT), rows
    same_set = {}
    for mult in TE.BUDGET_MULT:
        learned, rule = st.learned[mult], base.arms[f"beam8x{mult}"]
        same_set[mult] = sum(1 for a, b in zip(rule, learned)
                             if {c.idx for _, c in a.passages}
                             == {c.idx for _, c in b.passages})
    for tok in rows:
        mode, mult = " ".join(tok[:-9]), float(tok[-9])
        store = (st.learned[mult] if mode == "learned value"
                 else base.arms[f"beam8x{mult}"])
        res = [TE.coverage_result(p.resume_id, p.terms, base.reqs) for p in store]
        m = metric_row(KeywordCoverageLLM(), mode, "rag", res, base.labels, "tfidf")
        # 9 numeric columns: budget x spearman kendall mae within_10 passages
        # words spent satisfied same set as rule arm
        assert _matches(tok[-8], m["spearman"]), tok[:2]
        assert _matches(tok[-6], m["mae"]), tok[:2]
        if mode == "learned value":
            assert _matches(tok[-1], same_set[mult]), (mult, tok[-1], same_set[mult])


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
