"""Regression tests for the screening pipeline.

Run with pytest (`pip install pytest`, then `pytest tests -q`) or directly with no
extra dependency:  python tests/test_pipeline.py

Every test here is TF-IDF-only and loads no GGUF, so the file runs in a couple of
seconds on a CPU-only laptop. The dense path is covered by
test_embedder_for_refuses_a_silent_scale_swap, which forces the import to fail
rather than downloading a model.
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
from rag.config import SIM_FLOOR
from rag.index import Embedder, ResumeIndex, embedder_for
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
