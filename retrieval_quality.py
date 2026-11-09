"""What the retriever actually delivers to the prompt, measured without an LLM.

Every quality number in this study correlates a *score* with a label, so the repo can say how
well the pipeline ranks but not what the retriever handed the model - which is the load-bearing
claim of a RAG framework. The evidence block is 172 prompt words against a 304-word full-context
baseline, but nothing measured whether those 172 words are the words the rubric needs.

For the 30 generated resumes that question has an exact answer, because `generate_dataset.py`
knows which sentence it wrote to demonstrate which skill and now returns that alongside the
text. So per requirement: did the bullet that proves it survive chunking, the similarity floor,
top-k, de-duplication and the word budget into the text the scorer is shown? Three consequences
are reported below: (1) recall and precision of the evidence block per embedder and per floor,
(2) the **score ceiling the retrieval imposes** - re-run the rubric on only the evidence that
arrived, which bounds what *any* reader of this prompt could score - and (3) what the pasted
words are made of. The 3 hand-written CVs are excluded: their provenance was never generated.

    python retrieval_quality.py                # both embedders, ~25 s (dense loads torch)
    python retrieval_quality.py --tfidf-only   # ~4 s, no torch
"""
import argparse
import importlib.util
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate import load_data, rank_metrics                # noqa: E402
from rag.config import CHUNK_OVERLAP, CHUNK_WORDS, SIM_FLOOR, TOP_K  # noqa: E402
from rag.index import ResumeIndex, embedder_for             # noqa: E402
from rag.ingest import chunk_text, clean                    # noqa: E402
from rag.screen import evidence_budget, select_evidence     # noqa: E402

# The JD is a six-bullet list in the same order as the rubric's SKILLS, which is the only link
# between requirement text and qualification fact. Asserted, not assumed: if either file is
# edited so they stop lining up, this script stops rather than reporting a mapping that shifted
# by one.
REQ_KEYWORD = {"python_pandas": "pandas", "ml_sklearn": "scikit-learn", "sql": "sql",
               "visualization": "visualization", "nlp_llm": "nlp", "communication": "communication"}
CONDITIONS = [("rag, floor as shipped", "shipped", None),
              ("rag, floor 0.0 (no filter)", 0.0, None),
              ("rag, shipped floor, no cap", "shipped", 10 ** 6),
              ("full context (reference)", None, None)]


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_dataset",
                                                  ROOT / "data" / "generate_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def line_word_spans(text):
    """[(start, end)] word range of each non-empty line, in the same word list `chunk_text`
    splits, so a bullet's span and a chunk's span live in one coordinate system."""
    spans, pos = [], 0
    for line in text.split("\n"):
        n = len(line.split())
        if n:
            spans.append((pos, pos + n))
        pos += n
    return spans


def bullet_variants(raw, typos):
    """The bullet as written, plus the spellings the generator's typo pass could have left.

    `render()` misspells up to two words in the finished document, first occurrence only, so a
    given bullet may carry the correct or the misspelt form depending on what came earlier.
    Combinations up to two are covered; a variant matches exactly when the shipped text holds
    this sentence.
    """
    present = [w for w in typos if re.search(rf"\b{w}\b", raw, re.I)]
    out = [raw]
    for k in (1, 2):
        for combo in itertools.combinations(present, k):
            v = raw
            for w in combo:
                v = re.sub(rf"\b{w}\b", typos[w], v, flags=re.I)
            if v not in out:
                out.append(v)
    return out


def find_span(words, line_spans, raw, typos):
    """Word range of one bullet in its document, or None if the text does not contain it.

    Search is line-scoped because the CV writes `- <sentence>.` on its own line: the needle is
    that whole line's word list, so a bullet cannot match across a line break or merge with the
    role heading above it.
    """
    for variant in bullet_variants(raw, typos):
        needle = ["-"] + f"{variant}.".split()
        for start, end in line_spans:
            if end - start == len(needle) and words[start:end] == needle:
                return (start, end)
    return None


def merge(ranges):
    out = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def covered(span, ranges):
    s, e = span
    return sum(max(0, min(e, b) - max(s, a)) for a, b in ranges)


class Ground:
    """One resume's provenance, as word spans in the document it was written into."""

    def __init__(self, name, text, credits, provenance, typos):
        self.name = name
        self.credits = credits
        self.words = [w for line in text.split("\n") for w in line.split()]
        spans = line_word_spans(text)
        self.evidence = {}          # skill -> [(span, text)]
        self.filler = []            # [span]
        self.missing = []           # provenance the text does not contain: a hard failure
        for tag, raw in provenance:
            span = find_span(self.words, spans, raw, typos)
            if span is None:
                self.missing.append((tag, raw))
            elif tag == "filler":
                self.filler.append(span)
            else:
                self.evidence.setdefault(tag, []).append((span, raw))
        self.skills_line = next(((a, b) for a, b in spans if self.words[a] == "Skills:"), None)
        self.n_words = len(self.words)

    @property
    def bullets(self):
        return [(s, t) for group in self.evidence.values() for s, t in group]

    @property
    def n_proven_pairs(self):
        return len(self.evidence)


def pasted_ranges(index, resume_id, n_words):
    """Word span of every chunk of this resume - the full-context reference case.

    Merged, because consecutive chunks share CHUNK_OVERLAP words and every metric below is a
    word count: unmerged spans would bill the same 20 words twice and put this row above the
    document it is supposed to equal.
    """
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    n_chunks = len([c for c in index.chunks if c.resume_id == resume_id])
    return merge([(i * step, min(i * step + CHUNK_WORDS, n_words)) for i in range(n_chunks)])


def retrieve(index, rid, reqs, floor, cap, n_words):
    """The pipeline's own evidence selection for one condition.

    Returns the merged word spans plus the count of requirements the prompt will mark `none` -
    the same test `build_prompt` applies when it writes "SUPPORTING PASSAGES PER REQUIREMENT",
    which is the one place the model is told outright that it has no evidence for a line.

    `floor` None means full context: paste every chunk, which is the row the RAG rows are
    compared against and the check that the spans are right - it must recall every bullet.
    """
    if floor is None:
        return pasted_ranges(index, rid, n_words), None, None
    hits = [index.search(r, rid, k=TOP_K, min_score=floor) for r in reqs]
    passages = select_evidence(hits, cap if cap is not None else evidence_budget(n_words))
    kept = {(c.resume_id, c.idx) for _, c in passages}
    starved = sum(1 for h in hits if not any((c.resume_id, c.idx) in kept for _, c in h))
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    starts = {i: s for i, s in enumerate(range(0, n_words, step))}
    return (merge([(starts[c.idx], min(starts[c.idx] + CHUNK_WORDS, n_words))
                   for _, c in passages]), starved, len(passages))


def measure(ground, ranges):
    """Delivery and composition of one resume under one condition."""
    whole = partial = 0
    ev_words = 0
    for (s, e), _text in ground.bullets:
        inner = (s + 1, e)                      # drop the "-" list marker
        ev_words += covered(inner, ranges)
        if any(a <= inner[0] and inner[1] <= b for a, b in ranges):
            whole += 1
        elif covered(inner, ranges):
            partial += 1
    reached = {skill for skill, group in ground.evidence.items()
               if any(any(a <= s + 1 and e <= b for a, b in ranges) for (s, e), _ in group)}
    skills_in = bool(ground.skills_line) and covered(ground.skills_line, ranges) > 0
    return {
        "n_bullets": len(ground.bullets), "whole": whole, "partial": partial,
        "reached": reached, "skills_in": skills_in,
        "pasted": sum(e - s for s, e in ranges), "doc_words": ground.n_words,
        "ev_words": ev_words,
        "filler_words": sum(covered((s + 1, e), ranges) for s, e in ground.filler),
        "skills_words": covered(ground.skills_line, ranges) if ground.skills_line else 0,
    }


def ceiling_of(ground, m, rubric_score):
    """The rubric score reachable from this evidence block alone.

    A proven skill whose bullet never arrived is scored on what the prompt does show: the
    Skills-line claim if that line is in the block, otherwise nothing. That is the honest
    information set - the system prompt tells the reader a bare Skills-line mention is a claim,
    not proof.
    """
    credits = {}
    for skill, credit in ground.credits.items():
        if credit != "proven":
            credits[skill] = credit
        elif skill in m["reached"]:
            credits[skill] = "proven"
        else:
            credits[skill] = "claimed" if m["skills_in"] else "absent"
    return rubric_score(credits)


def print_condition(label, rows):
    b = sum(r["n_bullets"] for r in rows)
    w = sum(r["whole"] for r in rows)
    p = sum(r["partial"] for r in rows)
    pairs = sum(len(r["reached"]) for r in rows)
    pasted = sum(r["pasted"] for r in rows)
    ev = sum(r["ev_words"] for r in rows)
    known = ev + sum(r["filler_words"] + r["skills_words"] for r in rows)
    doc = sum(r["doc_words"] for r in rows)
    starved = [r["starved"] for r in rows if r["starved"] is not None]
    blank = f"{np.mean(starved):>6.2f}" if starved else "     -"
    passages = [r["passages"] for r in rows if r["passages"] is not None]
    npass = f"{np.mean(passages):>6.2f}" if passages else "     -"
    print(f"   {label:<28} {b:>5} {w / b:>7.1%} {(w + p) / b:>6.1%} {pairs:>7} "
          f"{blank} {pasted:>8} {npass} {100 * ev / pasted:>8.1f}% "
          f"{100 * known / pasted:>7.1f}% {pasted / doc:>7.2f}")
    return {"bullets": b, "whole": w / b, "any": (w + p) / b, "pairs": pairs,
            "pasted": pasted, "relevant": ev / pasted, "known": known / pasted,
            "fraction": pasted / doc, "resumes": len(rows),
            "blank": np.mean(starved) if starved else None,
            "passages": np.mean(passages) if passages else None}


def prompt_inflation_words():
    """The evidence-block word count from the committed `prompt_inflation.log`, if readable.

    That script counts words as they enter the prompt; this one counts the word spans the
    passages cover. Two independent routes to the same quantity, so the log is worth quoting by
    value rather than by reference - and returning None keeps this script working if the log is
    regenerated with different wording.
    """
    path = ROOT / "prompt_inflation.log"
    if not path.exists():
        return None
    raw = path.read_bytes()
    text = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8",
                      errors="replace")
    m = re.search(r"old \d+ vs current (\d+)", text)
    return int(m.group(1)) if m else None


def committed_stats(kind):
    """The Q4_K_M and retrieval-only rho/MAE, recomputed on the 30 provenance resumes only.

    README's headlines include the 3 hand-written CVs; the ceiling cannot be computed for them,
    so comparing ceiling with model needs both sides restricted or it is not a comparison.
    """
    files = {"tfidf": ("per_resume_v2.csv", "per_resume_v2_fallback.csv"),
             "dense": ("per_resume_v2_dense.csv", "per_resume_v2_fallback_dense.csv")}[kind]
    out = []
    for name in files:
        path = ROOT / name
        if not path.exists():
            return None, None
        d = pd.read_csv(path)
        d = d[d["mode"] == "rag"].sort_values("resume_id")
        gen = d[d["resume_id"].str.match(r"r\d\d_(strong|medium|weak)\.txt")]
        if len(gen) != 30:
            return None, None
        pred, gold = gen["score"].to_numpy(float), gen["human"].to_numpy(float)
        out.append((float(rank_metrics(pred, gold)[0]), float(np.abs(pred - gold).mean())))
    return out[0], out[1]


def over_claim(kind, per_cond, grounds, n_req):
    """Join the re-derived retrieval onto the committed run's own reported counts.

    Two checks, in order of how much they mean. First the validation: the passage count this
    script re-derives has to equal the `passages` column of the CSV the headline rho was
    computed from, because if the retrieval replay is not the run that produced the scores then
    nothing below compares like with like. Then the finding: `build_prompt` prints `[i] none`
    for every requirement the floor left without a passage, so a model that still reports that
    requirement as matched has contradicted an instruction it was given in plain text.
    """
    path = {"tfidf": ROOT / "per_resume_v2.csv",
            "dense": ROOT / "per_resume_v2_dense.csv"}[kind]
    if not path.exists():
        print(f"   {path.name} missing - skipped")
        return
    d = pd.read_csv(path)
    d = d[(d["mode"] == "rag") & (d["embedder"] == kind)].set_index("resume_id")
    key = CONDITIONS[0][0]
    rows = []
    for rid in sorted(grounds):
        if rid not in d.index:
            continue
        m = per_cond[key][rid]
        # `matched` is the raw list length and `normalize_requirements` keeps strings that are
        # not one of the six requirements, so the count of requirements actually claimed is the
        # one `score_coverage` was computed from. Using the raw length would overstate any
        # violation on a model that pads its list with paraphrases.
        canonical = int(round(float(d.loc[rid, "score_coverage"]) * n_req / 100.0))
        rows.append((rid, canonical, int(d.loc[rid, "matched"]), len(m["reached"]),
                     m["starved"], m["passages"], int(d.loc[rid, "passages"])))
    agree = sum(1 for r in rows if r[5] is not None and r[5] == r[6])
    pad = sum(r[2] - r[1] for r in rows)
    print(f"   replay check: the passages this script selects match {path.name}'s own "
          f"`passages` column on {agree}/{len(rows)} generated resumes "
          f"(mean {np.mean([r[5] for r in rows]):.2f} passages per CV)")
    print(f"   {pad} of {sum(r[2] for r in rows)} `matched` entries are not one of the six "
          f"requirement strings, so the counts\n   below use the requirement-scoped ones "
          f"(mean {np.mean([r[1] for r in rows]):.2f} claimed per CV)")
    over = [r for r in rows if r[1] > n_req - r[4]]
    print(f"   the prompt marked {np.mean([r[4] for r in rows]):.2f} of {n_req} requirements "
          f"per CV as having no passage at all, and on {len(over)} of "
          f"{len(rows)} CVs the model reported more requirements matched than that leaves:")
    for rid, canonical, _raw, _reached, starved, _p, _q in over[:8]:
        print(f"      {rid:<16} matched {canonical} > {n_req - starved} requirements with "
              f"evidence in the prompt ({starved} marked none)")
    strict = [r for r in rows if r[1] > r[3]]
    print(f"   on the stricter reading - a requirement counts only if its own evidence bullet is\n"
          f"   pasted whole - the model matched more than the prompt carried on "
          f"{len(strict)} of {len(rows)} CVs\n"
          f"   (mean claimed {np.mean([r[1] for r in rows]):.2f} vs mean whole-bullet "
          f"evidenced {np.mean([r[3] for r in rows]):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf-only", action="store_true",
                    help="skip the dense embedder, i.e. skip loading torch")
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

    gd = load_generator()
    jd, resumes, labels = load_data(Path(args.data))
    built = {name: (clean(text), credits, prov)
             for name, _tier, credits, text, prov in gd.build(30, 7)}

    # The invariant this analysis needs is the *word sequence*, not the bytes: the shipped
    # files are CRLF, `clean()` maps every \r\n to a blank line, and `chunk_text` splits on
    # whitespace - so two texts that chunk identically can differ as strings. `data/
    # check_dataset.py` owns the stricter byte-level drift check against the generator.
    def wseq(t):
        return t.split()

    drifted = [n for n, (t, _c, _p) in built.items()
               if n not in resumes or wseq(resumes[n]) != wseq(t)]
    if drifted:
        raise SystemExit(f"the generator in this checkout does not produce the shipped word "
                         f"sequence for {len(drifted)} resumes ({', '.join(drifted[:3])}). "
                         "There is no ground truth to measure against until the corpus is "
                         "regenerated and re-scored.")

    grounds = {}
    for name, (text, credits, prov) in built.items():
        grounds[name] = Ground(name, text, credits, prov, gd.TYPOS)
    unlocated = [(n, m) for n, g in grounds.items() for m in g.missing]
    if unlocated:
        raise SystemExit(f"{len(unlocated)} provenance bullets are absent from their own "
                         f"document, e.g. {unlocated[0][0]}: {unlocated[0][1][1][:60]!r}")
    label_mismatch = [n for n, (_t, c, _p) in built.items()
                      if gd.rubric_score(c) != int(labels[n])]
    if label_mismatch:
        raise SystemExit(f"labels.csv disagrees with rubric(credits) for {label_mismatch}")

    reqs = [s.rstrip(".") for s in
            (l.strip()[1:].strip() for l in jd.splitlines() if l.strip().startswith("-"))]
    skills = list(gd.SKILLS)
    if len(reqs) != len(skills):
        raise SystemExit(f"{len(reqs)} requirement bullets vs {len(skills)} rubric skills - "
                         "the index-based mapping is no longer valid")
    for req, skill in zip(reqs, skills):
        if REQ_KEYWORD[skill] not in req.lower():
            raise SystemExit(f"requirement {req!r} does not look like {skill!r}")

    n_bul = sum(len(g.bullets) for g in grounds.values())
    n_fill = sum(len(g.filler) for g in grounds.values())
    n_pairs = sum(g.n_proven_pairs for g in grounds.values())
    print(f"self-check: the replay reproduces {len(built)}/30 shipped resumes word-for-word, "
          f"all {n_bul + n_fill} provenance bullets are found in their own document, and "
          f"rubric(credits) == labels.csv for 30/30")
    print(f"requirement map: {len(reqs)} JD bullets <-> {', '.join(skills)}, keyword-asserted")
    print(f"ground truth: {n_bul} evidence bullets over {n_pairs} proven skill-instances, "
          f"{n_fill} filler bullets, mean {n_bul / len(grounds):.1f} evidence bullets per CV; "
          f"the {len(resumes) - len(grounds)} hand-written CVs have no provenance and are "
          "excluded\n")

    chunks = [c for rid, t in resumes.items() for c in chunk_text(rid, t)]
    for kind in ("tfidf", "dense"):
        if args.tfidf_only and kind == "dense":
            continue
        index = ResumeIndex(embedder_for(kind))
        index.build(chunks, extra_corpus=[jd])
        if index.embedder.kind != kind:
            raise SystemExit(f"asked for {kind}, got {index.embedder.kind}")
        floor = SIM_FLOOR[kind]
        print(f"embedder: {kind} | sim_floor: {floor} | top_k: {TOP_K} | chunks: "
              f"{len(index.chunks)} | budget: {evidence_budget(300)} of a 300-word CV")
        print(f"   {'condition':<28} {'bullets':>7} {'whole':>7} {'any':>7} {'reqs':>7} "
              f"{'blank':>6} {'pasted':>8} {'pass':>6} {'rubric':>8} {'known':>7} {'of doc':>7}")
        per_cond = {}
        stats = {}
        for label, fl, cap in CONDITIONS:
            floor_ = floor if fl == "shipped" else fl
            rows = {}
            for rid in sorted(grounds):
                ranges, starved, n_pass = retrieve(index, rid, reqs, floor_, cap,
                                                   grounds[rid].n_words)
                rows[rid] = measure(grounds[rid], ranges)
                rows[rid]["starved"] = starved
                rows[rid]["passages"] = n_pass
            per_cond[label] = rows
            stats[label] = print_condition(label, [rows[r] for r in sorted(grounds)])
        ref = stats[CONDITIONS[-1][0]]
        if ref["whole"] < 1.0 or abs(ref["fraction"] - 1.0) > 1e-9:
            raise SystemExit(f"the reference row does not behave: pasting every chunk of every "
                             f"resume delivered {ref['whole']:.1%} of the bullets and "
                             f"{ref['fraction']:.2f} of the document words. Bullet spans and "
                             "chunk spans are not in one coordinate system, so every recall "
                             "number above is wrong.")
        print("   whole = the bullet sits inside one pasted passage; any = some of it does; "
              "blank = requirements per CV\n   whose prompt line reads `none`; pass = "
              "passages kept per CV; rubric = share of pasted words that\n   are evidence "
              f"bullets; known = that plus filler and the Skills line; of doc = pasted / "
              f"document\n   words; reqs = the {n_pairs} proven skill-instances whose own "
              "bullet arrived whole")
        shipped, nofloor = stats[CONDITIONS[0][0]], stats[CONDITIONS[1][0]]
        if shipped["pasted"] == nofloor["pasted"]:
            print(f"   the floor changed no pasted text at all on this embedder "
                  f"({shipped['pasted']} words either way) and moved only the citation "
                  f"markers:\n   requirements left with no passage went "
                  f"{nofloor['blank']:.2f} -> {shipped['blank']:.2f} per CV. Whatever the "
                  f"floor's swept\n   optimum buys, it buys it in what the prompt claims to "
                  "be evidenced, not in what it contains.\n")
        else:
            print(f"   the floor removed {nofloor['pasted'] - shipped['pasted']} of "
                  f"{nofloor['pasted']} pasted words "
                  f"({100 * (1 - shipped['pasted'] / nofloor['pasted']):.1f}%) and left "
                  f"{shipped['blank']:.2f} vs {nofloor['blank']:.2f} requirements per CV "
                  "without a passage\n")
        per_cv = shipped["pasted"] / shipped["resumes"]
        inflation = prompt_inflation_words()
        print(f"   the budget asks for {evidence_budget(238)} words on a 238-word CV but a "
              f"chunk is {CHUNK_WORDS} and two\n   do not fit, so the block is one passage on "
              f"average ({shipped['passages']:.2f}): {per_cv:.1f} pasted words per CV. "
              f"`prompt_inflation.log`\n   counts "
              + (f"{inflation} the same way from the prompt side, on all 33 resumes"
                 if inflation is not None else "69 words there, on all 33 resumes") + ".\n")

        print("2) the ceiling this retrieval puts on any scorer, against what the study got")
        llm, base = committed_stats(kind)
        gold = np.array([int(labels[r]) for r in sorted(grounds)], dtype=float)
        ceil_rows = {}
        for label, _fl, _cap in CONDITIONS:
            ceil = np.array([ceiling_of(grounds[r], per_cond[label][r], gd.rubric_score)
                             for r in sorted(grounds)], dtype=float)
            ceil_rows[label] = ceil
            rho, tau = rank_metrics(ceil, gold)
            tail = (f" | Q4_K_M {llm[0]:+.3f} / MAE {llm[1]:.2f}, no-LLM {base[0]:+.3f} / "
                    f"MAE {base[1]:.2f} on these same 30" if llm is not None else "")
            print(f"   {label:<28} ceiling rho {rho:+.3f} tau {tau:+.3f} "
                  f"(distinct {len(set(ceil.tolist()))}, MAE "
                  f"{np.abs(ceil - gold).mean():.2f}){tail}")
        if llm is not None:
            ship = ceil_rows[CONDITIONS[0][0]]
            c_rho = float(rank_metrics(ship, gold)[0])
            print(f"   the ceiling is not a model result: it is the rubric applied to what the "
                  f"retriever\n   delivered, so it bounds any reader of that block. "
                  f"Splitting the shortfall on these 30:\n"
                  f"   {1 - c_rho:+.3f} rho is the evidence's (1.000 -> {c_rho:+.3f}) and "
                  f"{c_rho - llm[0]:+.3f} is the model's\n"
                  f"   ({c_rho:+.3f} -> {llm[0]:+.3f}). On MAE the model gives away more: "
                  f"{np.abs(ship - gold).mean():.2f} for a\n"
                  f"   faithful reader of the same block against {llm[1]:.2f} measured.")
        print()
        print("3) what the prompt told the model, against what the model reported")
        over_claim(kind, per_cond, grounds, len(reqs))
        print()

    print("What this does and does not license:")
    print(" - Provenance is the generator's own record of which sentence demonstrates which")
    print("   skill, so 'delivered' is a fact about this corpus, not a proxy for relevance.")
    print("   It cannot say whether an evidence bullet the candidate never wrote would have")
    print("   helped, and it cannot judge evidence quality - only presence.")
    print(" - The ceiling credits a skill when a *whole* bullet reached one passage. A reader")
    print("   can also infer from a bullet split across two passages, which is why 'any' sits")
    print("   next to 'whole', and from a bullet phrased for another requirement.")
    print(" - Recall counts every evidence bullet as relevant to its own skill. The retriever")
    print("   is not wrong to surface a strong candidate's SQL line for a visualisation")
    print("   requirement; that is why 'rubric share' is across all proven bullets.")
    print(" - n = 30 resumes here, not 33: the hand-written CVs are illustrations of the")
    print("   pipeline, and their provenance was never generated.")


if __name__ == "__main__":
    main()
