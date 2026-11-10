"""The retrieval geometry as a measured trade-off, with no LLM in the loop.

`retrieval_quality.py` established that the shipped evidence block is about one passage and a
quarter of the CV's proof, and that lifting the word cap lifts the score ceiling to 0.975 /
1.000 - but it measured two points on a curve and then recommended the geometry change anyway.
This script measures the curve: chunk length, chunk overlap, top-k per requirement and the word
budget each move the same three quantities in opposite directions, so the interesting fact is
where the knee is rather than that a trade-off exists.

Three things are reported per cell, all in the currency the thesis already spends:

  cost      - pasted evidence words per CV, and the **real prompt length** from `build_prompt`,
              against the committed `per_resume_v2.csv` figures for the shipped cell
  delivery  - the fraction of the generator's own proof bullets that arrive inside one passage,
              the proven skill-instances reached, and the requirements the prompt marks `none`
  ceiling   - the rubric re-applied to the delivered evidence: the best score *any* reader of
              that block could have produced, and its rank correlation with labels.csv

It is a measurement of the retrieval stage, so every number is a bound on a scorer rather than
a score: no model was run at any of these geometries. The shipped floors were swept at chunk 90
only (`sweep_floor.py`), so each cell also appears at floor 0 (sections 4 and 6) and the floor
itself is re-swept under every chunking (sections 8 and 9).

    python geometry_sweep.py                 # both embedders, ~2 min (dense loads torch)
    python geometry_sweep.py --tfidf-only    # ~35 s, no torch

Sections 1-4 sweep the four dials per embedder, 5-7 read the frontier and its units, 8-9
re-sweep the similarity floor under every chunking and re-read the frontier there.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate import load_data, rank_metrics                   # noqa: E402
from rag.config import (CHUNK_OVERLAP, CHUNK_WORDS, MAX_EVIDENCE_WORDS,  # noqa: E402
                        SIM_FLOOR, TOP_K)
from rag.index import ResumeIndex, embedder_for                # noqa: E402
from rag.ingest import chunk_text, clean, split_requirements   # noqa: E402
from rag.screen import build_prompt, evidence_budget, select_evidence  # noqa: E402
from retrieval_quality import (Ground, ceiling_of, load_generator,  # noqa: E402
                               measure, merge)

GEN = tuple(f"r{i:02d}_" for i in range(1, 31))   # the 30 CVs that have provenance

# The committed per-resume dumps, one per embedder: this sweep's shipped cell has to reproduce
# the prompt the study actually ran, and each embedder retrieved its own passages.
PER_RESUME = {"tfidf": "per_resume_v2.csv", "dense": "per_resume_v2_dense.csv"}


def _rag_dump(fname):
    """The RAG rows of one dump, restricted to the 30 CVs that have provenance."""
    d = pd.read_csv(ROOT / fname)
    return d[(d["mode"] == "rag") & d["resume_id"].str.startswith(GEN)].set_index("resume_id")

# (label, size, overlap) - the shipped pair is first in every table, and CHUNK_WORDS /
# CHUNK_OVERLAP are the values the committed runs were produced with.
GEOMETRIES = [("90/20 (shipped)", CHUNK_WORDS, CHUNK_OVERLAP),
              ("45/10", 45, 10),
              ("150/30", 150, 30),
              ("240/0", 240, 0),
              ("400/0", 400, 0),
              ("one chunk per CV", 10_000, 0)]
TOP_KS = [1, 2, TOP_K, 5, 8]
BUDGETS = [("fn (shipped)", None), ("60", 60), ("120", 120), ("180", 180),
           ("240", 240), (str(MAX_EVIDENCE_WORDS), MAX_EVIDENCE_WORDS), ("no cap", 10 ** 6)]

# `sweep_floor.py` fitted SIM_FLOOR at the shipped chunking, so every geometry row above is a
# geometry compared against a threshold it was never tuned to. This ladder re-sweeps the floor
# under each of the six chunkings, in that embedder's own scale and never past its SIM_ANCHORS
# ceiling (rag/llm.py: the score the fallback mapping calls a perfect match, so a floor above it
# would filter the evidence the rubric itself treats as 100%), with the committed value included
# so the shipped row is one of the candidates and the re-fitting is checkable rather than
# assumed.
LADDER = {"tfidf": [0.0, 0.008, 0.016, 0.024, 0.032, 0.040, 0.056, 0.072, 0.080],
          "dense": [0.0, 0.05, 0.10, 0.20, 0.24, 0.30, 0.36, 0.45, 0.55]}


def cap_of(spec, n_words):
    return evidence_budget(n_words) if spec is None else spec


def best_floor(ladder, label, floors):
    """The floor with the highest ceiling rho under one chunking; ties go to the shorter prompt.

    `sweep_floor.py` picked SIM_FLOOR the same way - widest strong-weak gap that still returns
    passages - but only at the shipped chunking, so this is the same rule applied per geometry.
    """
    ranked = sorted(((s["rho"] if not np.isnan(s["rho"]) else float("-inf"), -s["prompt"], fl)
                     for fl in floors for s in [ladder[(label, fl)]]), reverse=True)
    chosen = ranked[0][2]
    return chosen, ladder[(label, chosen)]


class Geometry:
    """One chunking, indexed - the hits per requirement are cached per (resume, top-k)."""

    def __init__(self, index, spans_by_resume):
        self.index = index
        self.spans = spans_by_resume
        self._hits = {}

    def hits(self, rid, reqs, k):
        if (rid, k) not in self._hits:
            # top-k first, floor applied afterwards: `index.search` sorts and truncates before
            # it filters, so filtering the un-floored top-k is the same set, and one search can
            # serve both floors. Asserted against the real call in the self-check.
            self._hits[(rid, k)] = [self.index.search(r, rid, k=k, min_score=0.0) for r in reqs]
        return self._hits[(rid, k)]


def build_geometry(kind, resumes, size, overlap):
    chunks = [c for rid, t in resumes.items() for c in chunk_text(rid, t, size, overlap)]
    index = ResumeIndex(KINDS[kind]["embedder"])
    index.build(chunks, extra_corpus=[KINDS[kind]["jd"]])
    spans, by_resume = {}, {}
    for c in chunks:
        by_resume.setdefault(c.resume_id, []).append(c)
    for rid, cs in by_resume.items():
        step = max(1, size - overlap)
        n = len(resumes[rid].split())
        spans[rid] = {c.idx: (c.idx * step, min(c.idx * step + size, n)) for c in cs}
    return Geometry(index, spans)


KINDS = {}          # kind -> {"embedder", "jd", "reqs"}, filled in main()


def evaluate_cell(geo, grounds, reqs, k, budget, floor, kind):
    rows = []
    for g in grounds:
        cap = cap_of(budget, g.n_words)
        per_req = [[h for h in hh if h[0] >= floor] for hh in geo.hits(g.name, reqs, k)]
        passages = select_evidence(per_req, cap)
        kept = {(c.resume_id, c.idx) for _, c in passages}
        ranges = merge([geo.spans[g.name][c.idx] for _, c in passages])
        starved = sum(1 for hh in per_req
                      if not any((c.resume_id, c.idx) in kept for _, c in hh))
        m = measure(g, ranges)
        m.update(name=g.name, starved=starved, passages=len(passages),
                 prompt_words=len(build_prompt(KINDS[kind]["jd"], reqs, per_req,
                                               passages).split()),
                 score=ceiling_of(g, m, KINDS[kind]["rubric_score"]))
        rows.append(m)
    return rows


def summarize(rows, gold):
    b = sum(r["n_bullets"] for r in rows)
    ceil = np.array([r["score"] for r in rows], dtype=float)
    lab = np.array([gold[r["name"]] for r in rows], dtype=float)
    rho, tau = rank_metrics(ceil, lab)
    return {"bullets": b, "whole": sum(r["whole"] for r in rows) / b,
            "any": sum(r["whole"] + r["partial"] for r in rows) / b,
            "pairs": sum(len(r["reached"]) for r in rows),
            "pasted": np.mean([r["pasted"] for r in rows]),
            "prompt": np.mean([r["prompt_words"] for r in rows]),
            "of_doc": sum(r["pasted"] for r in rows) / sum(r["doc_words"] for r in rows),
            "passages": np.mean([r["passages"] for r in rows]),
            "none": np.mean([r["starved"] for r in rows]),
            "rho": rho, "tau": tau, "mae": np.abs(ceil - lab).mean(),
            "distinct": len(set(ceil.tolist())), "rows": rows}


HEADER = (f"   {'cell':<26} {'pass':>5} {'pasted':>7} {'prompt':>7} {'of doc':>7} "
          f"{'whole':>6} {'reqs':>5} {'none':>5} {'ceil rho':>8} {'tau':>6} {'MAE':>6} {'d':>3}")


def print_cell(label, s, n_pairs):
    print(f"   {label:<26} {s['passages']:>5.2f} {s['pasted']:>7.1f} {s['prompt']:>7.1f} "
          f"{s['of_doc']:>6.2f} {s['whole']:>6.1%} {s['pairs']:>3}/{n_pairs} "
          f"{s['none']:>5.2f} {s['rho']:>+8.3f} {s['tau']:>+6.3f} {s['mae']:>6.2f} "
          f"{s['distinct']:>3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf-only", action="store_true", help="skip the dense side (no torch)")
    args = ap.parse_args()

    data_dir = ROOT / "data"
    jd, resumes_all, labels = load_data(data_dir)
    reqs = split_requirements(jd)
    resumes = {k: v for k, v in resumes_all.items() if k.startswith(GEN)}
    gd = load_generator()

    rows = gd.build(30, 7)
    if set(n for n, *_ in rows) != set(resumes):
        raise SystemExit("the generator no longer covers exactly the 30 CVs measured here")
    grounds = []
    for name, _tier, credits, text, prov in rows:
        g = Ground(name, clean(text), credits, prov, gd.TYPOS)
        if g.missing:
            raise SystemExit(f"{name}: provenance bullet not in its own text: {g.missing[0]}")
        grounds.append(g)
    n_pairs = sum(len(g.evidence) for g in grounds)
    gold = {name: float(labels[name]) for name in resumes}

    # ---- self-check 1: the shipped cell must be the committed artifact's shipped row ------
    shipped_log = (ROOT / "retrieval_quality.log").read_text(encoding="utf-8-sig").splitlines()
    ref = {}
    # the delivery row, not the ceiling row that repeats the same label: bullets | whole | any |
    # reqs | none | pasted | passages | rubric | known | of doc
    shipped_row = re.compile(
        r"^\s+rag, floor as shipped\s+(?P<bullets>\d+)\s+(?P<whole>[\d.]+)%\s+"
        r"(?P<any>[\d.]+)%\s+(?P<pairs>\d+)\s+(?P<none>[\d.]+)\s+(?P<pasted>\d+)\s+")
    for line in shipped_log:
        m = shipped_row.match(line)
        if m:
            ref["tfidf" if not ref else "dense"] = {"bullets": int(m["bullets"]),
                                                    "whole": float(m["whole"]) / 100,
                                                    "pairs": int(m["pairs"]),
                                                    "none": float(m["none"]),
                                                    "pasted": int(m["pasted"])}
    if set(ref) != {"tfidf", "dense"}:
        raise SystemExit(f"could not read both shipped rows from retrieval_quality.log: {ref}")

    # ---- self-check 2: the prompt this sweep builds must be the committed run's prompt ----
    if args.tfidf_only:
        pers, kinds = {"tfidf": _rag_dump(PER_RESUME["tfidf"])}, ["tfidf"]
    else:
        pers = {kind: _rag_dump(fname) for kind, fname in PER_RESUME.items()}
        kinds = list(PER_RESUME)
    all_cells = {}
    all_ladder = {}
    shipped_cells = {}
    for kind in kinds:
        embedder = embedder_for(kind)
        if embedder.kind != kind:
            raise SystemExit(f"asked for {kind}, got {embedder.kind}")
        KINDS[kind] = {"embedder": embedder, "jd": jd, "reqs": reqs,
                       "rubric_score": gd.rubric_score}
        floor = SIM_FLOOR[kind]
        print(f"\nembedder: {kind} | shipped: chunk {CHUNK_WORDS}/{CHUNK_OVERLAP}, top_k "
              f"{TOP_K}, floor {floor}, budget = evidence_budget(doc words)")

        # One index per chunking, built over ALL 33 CVs - which is what the committed runs
        # indexed, and what decides the TF-IDF vocabulary. It also means each geometry's cells
        # have to be computed before the next geometry is built: `Embedder.fit` re-fits the
        # shared vectorizer, which silently invalidates the previous index's matrix.
        base = build_geometry(kind, resumes_all, CHUNK_WORDS, CHUNK_OVERLAP)

        # filtering the un-floored top-k must equal what index.search(min_score=floor) returns
        checked = mismatch = 0
        for g in grounds[:6]:
            for i, r in enumerate(reqs):
                direct = base.index.search(r, g.name, k=TOP_K, min_score=floor)
                filtered = [h for h in base.hits(g.name, reqs, TOP_K)[i] if h[0] >= floor]
                checked += 1
                mismatch += int([(s, c.idx) for s, c in direct] !=
                                [(s, c.idx) for s, c in filtered])
        if mismatch:
            raise SystemExit(f"filtering top-k after the fact is not the pipeline's search: "
                             f"{mismatch} of {checked} queries differ")
        print(f"self-check: filtering the un-floored top-k at the floor equals index.search on "
              f"{checked}/{checked} requirement queries, and the index holds "
              f"{len(base.index.chunks)} chunks over {len(resumes_all)} CVs")

        cells = {}
        for label, cap in BUDGETS:
            cells[("budget", label, floor)] = summarize(
                evaluate_cell(base, grounds, reqs, TOP_K, cap, floor, kind), gold)
        for k in TOP_KS:
            cells[("top_k", str(k), floor)] = summarize(
                evaluate_cell(base, grounds, reqs, k, None, floor, kind), gold)
        floor_ladder = {}
        for label, size, overlap in GEOMETRIES:
            geo = (base if (size, overlap) == (CHUNK_WORDS, CHUNK_OVERLAP)
                   else build_geometry(kind, resumes_all, size, overlap))
            for fl in (floor, 0.0):
                cells[("chunk", label, fl)] = summarize(
                    evaluate_cell(geo, grounds, reqs, TOP_K, None, fl, kind), gold)
            # the same geo must be measured before the next one is built: `Embedder.fit`
            # re-fits the shared vectorizer in place and invalidates this index's matrix
            for fl in LADDER[kind]:
                floor_ladder[(label, fl)] = cells.get(("chunk", label, fl)) or summarize(
                    evaluate_cell(geo, grounds, reqs, TOP_K, None, fl, kind), gold)
        for label, _s, _o in GEOMETRIES:
            if ("chunk", label, floor) not in cells:
                raise SystemExit(f"the ladder skipped {label}, so its floor is unmeasured")
        all_ladder[kind] = floor_ladder

        # the geometry-independent invariant: one chunk per CV, no floor, is the whole document
        whole_cv = cells[("chunk", GEOMETRIES[-1][0], 0.0)]
        if whole_cv["whole"] < 1.0 or abs(whole_cv["of_doc"] - 1.0) > 1e-9:
            raise SystemExit(
                f"one chunk per CV at floor 0 delivered {whole_cv['whole']:.1%} of the bullets "
                f"and {whole_cv['of_doc']:.2f} of the document words, so the span arithmetic is "
                f"wrong at chunking {GEOMETRIES[-1][1:]}")

        # --- 1) evidence budget, at the shipped chunking ---------------------------------
        print(f"\n1) word budget at the shipped chunking, floor {floor} (the shipped row is the "
              f"one labelled 'fn (shipped)')")
        print(HEADER)
        for label, _cap in BUDGETS:
            s = cells[("budget", label, floor)]
            print_cell(f"budget {label}", s, n_pairs)
            all_cells[(kind, "budget", label, floor)] = s
        print(f"   'fn (shipped)' is min({MAX_EVIDENCE_WORDS}, doc, max(120, 0.4*doc)); "
              f"'no cap' is select_evidence with no word limit")
        mean_doc = np.mean([g.n_words for g in grounds])
        ladder = sorted((cap, cells[("budget", label, floor)]["whole"])
                        for label, cap in BUDGETS if cap is not None)
        back = [f"{a[0]} -> {b[0]}" for a, b in zip(ladder, ladder[1:])
                if b[1] < a[1] - 1e-9]
        binds = ladder[-1][1] != ladder[-2][1]
        cap_note = ("binds: the 400-word cell delivers less proof than the uncapped one"
                    if binds else
                    "buys nothing: the 400-word cell and the uncapped one are identical here")
        print(f"   the CVs here run to {max(g.n_words for g in grounds)} words (mean "
              f"{mean_doc:.1f}), so the {MAX_EVIDENCE_WORDS}-word cap {cap_note}")
        print(f"   going up the budget ladder loses proof in {len(back)} of "
              f"{len(ladder) - 1} steps" + (f" ({', '.join(back)})" if back else ""))

        # --- 2) chunk geometry, at the shipped budget ------------------------------------
        print("\n2) chunk size / overlap at the shipped budget function")
        print(HEADER)
        for label, _size, _overlap in GEOMETRIES:
            s = cells[("chunk", label, floor)]
            print_cell(f"chunk {label}", s, n_pairs)
            all_cells[(kind, "chunk", label, floor)] = s
            all_cells[(kind, "chunk", label, 0.0)] = cells[("chunk", label, 0.0)]
        print("   the last row is the reference: one chunk per CV at floor 0 is the whole "
              "document, so it must recall everything")

        # --- 3) top-k per requirement -----------------------------------------------------
        print("\n3) top-k per requirement at the shipped chunking and budget")
        print(HEADER)
        for k in TOP_KS:
            s = cells[("top_k", str(k), floor)]
            print_cell(f"top_k {k}" + (" (shipped)" if k == TOP_K else ""), s, n_pairs)
            all_cells[(kind, "top_k", str(k), floor)] = s

        # --- 4) floor sensitivity across the geometries -----------------------------------
        print(f"\n4) the shipped floor {floor} against no floor at all, per chunking, so a "
              f"threshold swept at {CHUNK_WORDS}/{CHUNK_OVERLAP} cannot be the hidden cause "
              f"of a geometry row")
        print(f"   {'chunking':<26} {'whole':>13} {'pasted':>13} {'none':>12} "
              f"{'ceil rho':>15}")
        for label, _size, _overlap in GEOMETRIES:
            a = all_cells[(kind, "chunk", label, floor)]
            b = all_cells[(kind, "chunk", label, 0.0)]
            print(f"   {label:<26} {a['whole']:>6.1%} ->{b['whole']:>6.1%} "
                  f"{a['pasted']:>6.1f} ->{b['pasted']:>6.1f} "
                  f"{a['none']:>5.2f} ->{b['none']:>5.2f} "
                  f"{a['rho']:>+7.3f} ->{b['rho']:>+7.3f}")

        # --- self-check on this embedder's shipped row ------------------------------------
        shipped = all_cells[(kind, "budget", "fn (shipped)", floor)]
        want = ref[kind]
        drift = []
        if shipped["bullets"] != want["bullets"]:
            drift.append(f"bullets {shipped['bullets']} vs {want['bullets']}")
        if abs(shipped["whole"] - want["whole"]) > 5e-4:
            drift.append(f"whole {shipped['whole']:.3%} vs {want['whole']:.3%}")
        if shipped["pairs"] != want["pairs"]:
            drift.append(f"reqs {shipped['pairs']} vs {want['pairs']}")
        if abs(shipped["none"] - want["none"]) > 5e-3:
            drift.append(f"none {shipped['none']:.2f} vs {want['none']:.2f}")
        if abs(shipped["pasted"] * 30 - want["pasted"]) > 1:
            drift.append(f"pasted {shipped['pasted'] * 30:.0f} vs {want['pasted']}")
        dump = pers[kind]
        pw = sum(int(round(r["prompt_words"]) != round(dump.loc[r["name"], "prompt_words"]))
                 for r in shipped["rows"])
        np_ = sum(int(r["passages"] != dump.loc[r["name"], "passages"])
                  for r in shipped["rows"])
        if pw or np_:
            drift.append(f"prompt words differ on {pw} of 30, passages on {np_} of 30")
        if drift:
            raise SystemExit(f"the shipped cell does not reproduce the committed artifacts: "
                             f"{'; '.join(drift)}")
        print(f"\nself-check: the shipped cell reproduces retrieval_quality.log exactly "
              f"({want['whole']:.1%} whole, {want['pairs']}/{n_pairs} reqs, "
              f"{want['none']:.2f} none, {want['pasted']} pasted words) and "
              f"{PER_RESUME[kind]}'s `prompt_words` and `passages` on 30/30 CVs")
        shipped_cells[kind] = shipped

    # --- 5) the frontier -------------------------------------------------------------------
    print("\n5) the frontier: every cell sorted by prompt length, with what it buys")
    for kind in kinds:
        floor = SIM_FLOOR[kind]
        cells = sorted((s["prompt"], axis, name, s) for (k, axis, name, fl), s
                       in all_cells.items() if k == kind and fl == floor)
        print(f"\n   {kind} (floor {floor}):")
        print(HEADER)
        for _prompt, axis, name, s in cells:
            print_cell(f"{axis} {name}", s, n_pairs)
        sp = shipped_cells[kind]
        for target in (0.90, 0.95):
            reach = [c for c in cells if not np.isnan(c[3]["rho"]) and c[3]["rho"] >= target]
            if not reach:
                print(f"   ceiling rho {target:.2f}: never reached by any cell measured here")
                continue
            p, axis, name, s = min(reach)
            print(f"   cheapest cell with ceiling rho >= {target:.2f}: {axis} {name} at "
                  f"{p:.1f} prompt words ({s['whole']:.1%} of proof, {s['of_doc']:.2f} of the "
                  f"document) against the shipped {sp['prompt']:.1f} words at rho "
                  f"{sp['rho']:+.3f} / {sp['whole']:.1%} of proof")

    print("\n6) the same frontier with the floor off, where a floor mis-tuned for the chunking "
          "cannot be the reason")
    for kind in kinds:
        front = sorted((s["prompt"], axis, name, s) for (k, axis, name, fl), s
                       in all_cells.items() if k == kind and fl == 0.0)
        print(f"\n   {kind} (floor 0.0):")
        print(HEADER)
        for _prompt, axis, name, s in front:
            print_cell(f"{axis} {name}", s, n_pairs)

    print("\n7) what the frontier says, in the units the claims around it are made in")
    for kind in kinds:
        floor = SIM_FLOOR[kind]
        ship = shipped_cells[kind]
        tk = [all_cells[(kind, "top_k", str(k), floor)] for k in TOP_KS]
        fine = all_cells[(kind, "chunk", GEOMETRIES[1][0], floor)]
        floored = [(all_cells[(kind, "chunk", label, floor)]["whole"]
                    - all_cells[(kind, "chunk", label, 0.0)]["whole"],
                    all_cells[(kind, "chunk", label, floor)]["none"]
                    - all_cells[(kind, "chunk", label, 0.0)]["none"])
                   for label, _s, _o in GEOMETRIES]
        knees = {}
        for axis in ("budget", "chunk", "top_k"):
            reach = sorted((s["prompt"], name, s) for (k, a, name, fl), s in all_cells.items()
                           if k == kind and a == axis and fl == floor
                           and not np.isnan(s["rho"]) and s["rho"] >= 0.90)
            knees[axis] = reach[0] if reach else None
        reached = [(axis, cell[1], cell[0]) for axis, cell in knees.items() if cell]
        knee_txt = ", ".join(f"{axis} {name} at {p:.0f} words"
                             for axis, name, p in reached) or "none of the three axes"
        print(f"\n   {kind}:")
        print(f"   - top_k is inert while the budget keeps one passage: {TOP_KS[0]} -> "
              f"{TOP_KS[-1]} moves the ceiling by at most "
              f"{max(abs(t['rho'] - tk[0]['rho']) for t in tk):.3f} of rho and the prompt by "
              f"{max(t['prompt'] for t in tk) - min(t['prompt'] for t in tk):.1f} words, "
              f"because {min(t['passages'] for t in tk):.2f}-{max(t['passages'] for t in tk):.2f} "
              f"passages survive whichever way the candidates were ranked")
        print(f"   - the finer chunking buys ceiling per prompt word most cheaply: "
              f"{GEOMETRIES[1][0]} keeps "
              f"{fine['passages']:.2f} passages in {fine['prompt']:.1f} prompt words and reaches "
              f"ceiling rho {fine['rho']:+.3f} against the shipped rho {ship['rho']:+.3f} at "
              f"{ship['prompt']:.1f} words, i.e. {fine['rho'] - ship['rho']:+.3f} of ceiling for "
              f"{fine['prompt'] - ship['prompt']:+.1f} prompt words "
              f"({100 * (fine['prompt'] / ship['prompt'] - 1):+.1f}%)")
        print(f"   - reaching ceiling 0.90 by widening the budget instead costs "
              f"{knees['budget'][0]:.1f} words" if knees["budget"] else
              "   - no budget on its own reaches ceiling 0.90 here")
        print(f"   - cheapest cell at ceiling rho >= 0.90 per axis: {knee_txt}"
              + ("" if len(reached) == 3 else
                 f"; {', '.join(a for a in ('budget', 'chunk', 'top_k') if knees[a] is None)} "
                 f"never reaches it at any value swept here"))
        proof_loss = [-a for a, _b in floored]
        none_gain = [b for _a, b in floored]
        worst_i = int(np.argmax(proof_loss))
        worst_cell = all_cells[(kind, "chunk", GEOMETRIES[worst_i][0], floor)]
        empty = sum(1 for r in worst_cell["rows"] if r["passages"] == 0)
        if proof_loss[worst_i] < 1e-9:
            print(f"   - across all {len(floored)} chunkings the floor costs no delivered proof "
                  f"at all (largest loss {abs(max(proof_loss)) * 100:.1f} points), and "
                  f"raises the "
                  f"requirements marked `none` by {min(none_gain):.2f} to "
                  f"{max(none_gain):.2f} per CV - claim discipline only, at every geometry")
        else:
            print(f"   - across all {len(floored)} chunkings the floor costs at most "
                  f"{proof_loss[worst_i] * 100:.1f} points of delivered proof "
                  f"({GEOMETRIES[worst_i][0]}, where one coarse chunk can fail every "
                  f"requirement at once and {empty} of {len(grounds)} CVs are left with no "
                  f"passage at all) and {abs(proof_loss[0]) * 100:.1f} at the shipped "
                  f"one, while it raises "
                  f"the requirements marked `none` by {min(none_gain):.2f} to "
                  f"{max(none_gain):.2f} per CV - so at every geometry it buys claim "
              f"discipline rather than evidence")

    # --- 8) the floor re-swept under each chunking -----------------------------------------
    print("\n8) the floor re-swept under every chunking: SIM_FLOOR was fitted at "
          f"{CHUNK_WORDS}/{CHUNK_OVERLAP} only (`sweep_floor.py`), so the other five chunkings "
          "in sections 1-7 are each measured against a threshold tuned for a geometry they are "
          "not. Same rule as that sweep - the floor with the best ceiling rho, ties to the "
          "shorter prompt - applied per chunking.")
    for kind in kinds:
        ladder, committed = LADDER[kind], SIM_FLOOR[kind]
        best = {label: best_floor(all_ladder[kind], label, ladder) for label, *_ in GEOMETRIES}
        print(f"\n   {kind}: floor ladder "
              + " ".join(f"{fl:.3f}" for fl in ladder))
        for title, get in (("ceiling rho", lambda s: f"{s['rho']:>+6.3f}"),
                           ("prompt words", lambda s: f"{s['prompt']:>6.1f}")):
            print(f"\n   {title} at each floor (`*` = the committed {committed}, "
                  "`^` = this row's best ceiling rho):")
            print(f"   {'chunking':<26}"
                  + "".join(f"{fl:>6.3f}{'*' if fl == committed else ' '} " for fl in ladder))
            for label, *_ in GEOMETRIES:
                print(f"   {label:<26}" + "".join(
                    f"{get(all_ladder[kind][(label, fl)])}"
                    f"{'^' if fl == best[label][0] else ' '} " for fl in ladder))
        n_opt = sum(1 for label, *_ in GEOMETRIES if best[label][0] == committed)
        print(f"\n   the committed floor is the rho-best at {n_opt} of {len(GEOMETRIES)} "
              f"chunkings; what moving it to this chunking's own best would change:")
        for label, _s, _o in GEOMETRIES:
            bfl, b = best[label]
            c = all_ladder[kind][(label, committed)]
            if bfl == committed:
                print(f"   {label:<26} unchanged: rho {c['rho']:+.3f} at {c['prompt']:.1f} "
                      f"prompt words, {c['whole']:.1%} of the generator's proof delivered")
            else:
                gain = b["rho"] - c["rho"]
                say = (f"rho {c['rho']:+.3f} -> {b['rho']:+.3f}" if abs(gain) >= 5e-4 else
                       f"rho {c['rho']:+.3f} -> {b['rho']:+.3f}, a tie at this rounding: "
                       f"the shorter prompt decided")
                print(f"   {label:<26} floor {committed:.3f} -> {bfl:.3f}: {say}, prompt "
                      f"{c['prompt']:.1f} -> {b['prompt']:.1f}, proof "
                      f"{c['whole']:.1%} -> {b['whole']:.1%}, `none` {c['none']:.2f} -> "
                      f"{b['none']:.2f}")
        gain = sorted(((best[l][1]["rho"] - all_ladder[kind][(l, committed)]["rho"], l)
                       for l, *_ in GEOMETRIES
                       if not (np.isnan(best[l][1]["rho"])
                               or np.isnan(all_ladder[kind][(l, committed)]["rho"]))),
                      reverse=True)
        near = sum(1 for g, _l in gain if abs(g) < 1e-3)
        print(f"\n   the committed floor is within 0.001 of what its own chunking achieves at "
              f"{near} of {len(GEOMETRIES)} chunkings; the largest gap is "
              f"{gain[0][0]:+.3f} of ceiling at {gain[0][1]}")
        collapse = []
        for label, (bfl, b) in best.items():
            worse = [(fl, b["rho"] - all_ladder[kind][(label, fl)]["rho"])
                     for fl in LADDER[kind] if fl > bfl]
            hit = next(((fl, d) for fl, d in worse if d > 0.05), None)
            collapse.append(f"{label} " + ("-" if hit is None else f"{hit[0]:.3f}"))
        print(f"   the lowest floor above the best that loses more than 0.05 of ceiling, per "
              f"chunking (ladder ends at {LADDER[kind][-1]:.3f}, '-' = nothing in the ladder "
              f"costs that much): " + ", ".join(collapse))

    # --- 9) the frontier at each chunking's own floor ---------------------------------------
    print("\n9) the frontier of section 7 re-read with every chunking at its own best floor, so "
          "a knee cannot be an artefact of a threshold tuned for a different geometry")
    verdicts = {}
    for kind in kinds:
        committed = SIM_FLOOR[kind]
        best = {label: best_floor(all_ladder[kind], label, LADDER[kind])
                for label, *_ in GEOMETRIES}
        print(f"\n   {kind} (shipped floor {committed:.3f}):")
        print(HEADER)
        for _p, label in sorted((best[label][1]["prompt"], label) for label, *_ in GEOMETRIES):
            bfl, b = best[label]
            print_cell(f"{label} @ {bfl:.3f}", b, n_pairs)
        for target in (0.90, 0.95):
            own = sorted((best[label][1]["prompt"], label, best[label][0], best[label][1])
                         for label, *_ in GEOMETRIES
                         if not np.isnan(best[label][1]["rho"])
                         and best[label][1]["rho"] >= target)
            com = sorted((all_cells[(kind, "chunk", label, committed)]["prompt"], label)
                         for label, *_ in GEOMETRIES
                         if not np.isnan(all_cells[(kind, "chunk", label,
                                                    committed)]["rho"])
                         and all_cells[(kind, "chunk", label, committed)]["rho"] >= target)
            if not own:
                print(f"   ceiling rho {target:.2f}: no chunking reaches it even at its own "
                      f"best floor" + (f" ({len(com)} reach it at {committed:.3f})"
                                       if com else ""))
                continue
            p, label, bfl, s = own[0]
            ref = com[0][1] if com else None
            use = [r for r in own if r[3]["of_doc"] < 0.95]
            say = ("no chunking reaches it at the committed floor, so the floor is what unlocks "
                   f"this row" if ref is None else
                   f"the same chunking ({ref}) wins at {committed:.3f}: the knee survives"
                   if ref == label else
                   f"at {committed:.3f} the cheapest was {ref} at {com[0][0]:.1f} words, so "
                   "recalibration MOVES the knee")
            print(f"   cheapest chunking at ceiling rho >= {target:.2f} on its own floor: "
                  f"{label} @ {bfl:.3f}, {p:.1f} prompt words, {s['of_doc']:.2f} of the CV; "
                  f"{say}"
                  + (" - that row is the whole document, so the ceiling is the bound, not a "
                     "usable geometry" if s["of_doc"] >= 0.95 else ""))
            print(f"   - among chunkings that deliver under 0.95 of the CV: "
                  + (f"{use[0][1]} @ {use[0][2]:.3f} at {use[0][0]:.1f} prompt words"
                     if use else f"none reaches rho {target:.2f} at any floor in the ladder"))
            if target == 0.90:
                verdicts[kind] = ref == label
        sp = shipped_cells[kind]
        own_ship = best[GEOMETRIES[0][0]]
        print(f"   the shipped chunking {GEOMETRIES[0][0]} sits at rho {sp['rho']:+.3f} on the "
              f"committed floor and {own_ship[1]['rho']:+.3f} on its own best "
              f"({own_ship[0]:.3f}), i.e. recalibrating it would buy "
              f"{own_ship[1]['rho'] - sp['rho']:+.3f} of ceiling")
    survived = [kind for kind, ok in verdicts.items() if ok]
    moved = [kind for kind in verdicts if kind not in survived]
    print(f"\n   the section 7 knee at ceiling rho 0.90 survives re-fitting the floor per "
          f"chunking for {', '.join(survived) or 'neither embedder'}"
          + (f"; it MOVES for {', '.join(moved)}" if moved else
             " - the geometry recommendation is not an artefact of the shipped threshold"))

    print("""
What this does and does not license:
 - No model was run at any geometry here. Every rho is the ceiling the delivered evidence
   allows, which bounds a perfect reader - it is not a score the pipeline got, and none of
   these cells may be reported as a result of Q4_K_M.
 - The floors were fitted at the shipped chunking only (`sweep_floor.py` swept 90/20), so the
   other chunkings in sections 1-6 are measured against a threshold tuned for a geometry they
   are not. Sections 8 and 9 close that loop: the floor is re-swept under every chunking, the
   frontier is re-read there, and the verdict on the section 7 knee is printed per embedder
   rather than assumed. What the re-fit cannot do is change a committed number: 0.024 and 0.24
   are what the model ran with, so a floor tuned for another chunking would have to go through
   Q4_K_M again to mean anything about this study.
 - Changing the chunking changes what the committed runs mean: SIM_FLOOR, the 172-word prompt
   (the 33-resume mean; the shipped cell measures 173.4 here, over the 30 CVs with provenance)
   and every rho in README belong to 90/20 with top_k 3 and the shipped budget function. None
   of these cells has been scored by the model, so none of them replaces a committed number.
 - Recall is against the generator's own bullets, so it measures presence, not evidence
   quality, and a bullet that proves a skill only in combination with another one is counted
   once for its own skill.
 - n = 30 generated CVs. The three hand-written ones have no provenance.
 - The ceiling can be high while the block is *less* discriminating in ways rho does not see:
   a cell that pastes the whole document reaches rho 1.000 by construction, and is exactly the
   baseline RAG is supposed to beat on prompt length.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
