"""Streamlit UI in an ATS shell.

Three pages: Screen (configure a run and launch it), Candidates (ranked rows with a
per-candidate evidence drawer), Study config (the retrieval/model/runtime constants and
the artifacts behind each headline number).

Every number shown here comes either from the run that just happened or from a committed
artifact named next to it. There are no mock values in this file by design - a card that
would need invented data is a card that does not get built.
"""
import csv
import statistics
from pathlib import Path

import pandas as pd
import streamlit as st

from rag import auth
from rag.config import (CHUNK_OVERLAP, CHUNK_WORDS, EMBED_MODEL, MAX_EVIDENCE_WORDS,
                        MODELS, N_THREADS, RAG_EVIDENCE_FRACTION,
                        RAG_MIN_EVIDENCE_WORDS, SIM_FLOOR, TOP_K)
from rag.index import ResumeIndex, embedder_for
from rag.ingest import chunk_text, clean, extract_text, split_requirements
from rag.llm import RetrievalOnlyLLM, load_llm
from rag.screen import screen_resume

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_CHOICES = list(MODELS.keys()) + ["Retrieval-only (no LLM)"]
# Retrieval-only is the default on purpose: the demo policy is to avoid loading the
# 1.1 GB GGUF live, because thermal throttling makes the latency numbers on screen stop
# matching the ones being quoted.
DEFAULT_MODEL = MODEL_CHOICES.index("Retrieval-only (no LLM)")

THEME = """
<style>
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
h1, h2, h3 { letter-spacing: -0.015em; }
[data-testid="stVerticalBlockBorderWrapper"] {
  box-shadow: 0 2px 8px rgba(0, 0, 0, .35);
}
.hl-brand {
  width: 54px; height: 54px; border-radius: 14px;
  background: #34A98A; color: #08110D;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 1.02rem; letter-spacing: .1em;
  margin-bottom: 18px;
}
.hl-kicker {
  color: #7E8C99; font-size: .82rem; letter-spacing: .09em;
  text-transform: uppercase; margin-bottom: 6px;
}
</style>
"""

st.set_page_config(page_title="Resume Screener", layout="wide")
st.markdown(THEME, unsafe_allow_html=True)


def chips(pairs):
    """Header metadata as inline-code tags - the ATS idiom, styled by the theme tokens."""
    return " · ".join(f"`{k} {v}`" for k, v in pairs)


def req_chips(reqs):
    return " · ".join(f"`[{i}]` {r}" for i, r in enumerate(reqs, 1))


@st.cache_data(show_spinner=False)
def load_sample():
    """The study corpus: the CVs in data/resumes plus the human rubric score for each."""
    labels = {}
    if (DATA_DIR / "labels.csv").exists():
        with (DATA_DIR / "labels.csv").open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                labels[row["resume_id"]] = int(row["human_score"])
    pairs = []
    for f in sorted((DATA_DIR / "resumes").glob("*")):
        if f.suffix.lower() in {".txt", ".pdf", ".docx"}:
            pairs.append((f.name, clean(extract_text(f.name, f.read_bytes()))))
    return labels, pairs


def default_jd():
    p = DATA_DIR / "job_description.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def n_evidenced(result) -> int:
    return sum(1 for refs in result.evidence_refs.values() if refs)


auth.init_db()
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        st.markdown('<div class="hl-kicker">Local RAG screening - Q4_K_M - CPU</div>'
                    '<div class="hl-brand">RS</div>', unsafe_allow_html=True)
        st.title("Resume Screener")
        st.caption("Sign in to screen candidates.")
        with st.container(border=True):
            tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
            with tab_login:
                with st.form("login_form"):
                    u = st.text_input("Username")
                    p = st.text_input("Password", type="password")
                    if st.form_submit_button("Log in", type="primary"):
                        name = auth.verify_user(u, p)
                        if name:
                            st.session_state.user = name
                            st.rerun()
                        else:
                            st.error("Wrong username or password.")
            with tab_signup:
                with st.form("signup_form"):
                    u2 = st.text_input("Choose a username")
                    p2 = st.text_input("Choose a password (8+ characters)", type="password")
                    p3 = st.text_input("Repeat password", type="password")
                    if st.form_submit_button("Create account", type="primary"):
                        if p2 != p3:
                            st.error("Passwords do not match.")
                        else:
                            ok, msg = auth.create_user(u2, p2)
                            if ok:
                                # Signing in here instead of asking for the password again
                                # avoids the one place a browser autofill can make a new
                                # account look lost.
                                st.session_state.user = u2.strip()
                                st.rerun()
                            else:
                                st.error(msg)
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def get_llm(variant: str):
    if variant == "Retrieval-only (no LLM)":
        return RetrievalOnlyLLM()
    return load_llm(MODELS[variant])


@st.cache_resource(show_spinner="Loading embedder...")
def get_embedder(prefer_dense: bool):
    # embedder_for, not Embedder: a dense load that silently fell back to TF-IDF would
    # score evidence against the wrong similarity scale and the floor would be mis-tuned.
    return embedder_for("dense" if prefer_dense else "tfidf")


def run_screening(pairs, jd, mode, top_k, variant, kind):
    try:
        embedder = get_embedder(kind == "dense")
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    index = ResumeIndex(embedder)
    chunks = [c for rid, t in pairs for c in chunk_text(rid, t)]
    index.build(chunks, extra_corpus=[jd])
    llm = get_llm(variant)
    if variant != "Retrieval-only (no LLM)" and isinstance(llm, RetrievalOnlyLLM):
        st.warning(f"Model file for '{variant}' not found in ./models - using "
                   "retrieval-only scoring. See README to download it.")
    reqs = split_requirements(jd) or [jd]
    sample_labels, _ = load_sample()

    bar = st.progress(0.0, text=f"Screening {len(pairs)} candidates...")
    results = []
    for i, (rid, txt) in enumerate(pairs):
        results.append(screen_resume(rid, jd, index, llm, mode=mode,
                                     full_text=txt, top_k=top_k))
        bar.progress((i + 1) / len(pairs), text=f"Screening {i + 1} of {len(pairs)}: {rid}")
    bar.empty()
    results.sort(key=lambda r: -r.score)
    st.session_state["run"] = {
        "results": results, "reqs": reqs, "mode": mode, "top_k": top_k,
        "variant": variant, "kind": kind, "floor": SIM_FLOOR.get(index.embedder.kind, 0.0),
        "n_chunks": len(chunks),
        "labels": {rid: sample_labels[rid] for rid, _ in pairs if rid in sample_labels},
    }


@st.dialog("Candidate", width="large")
def candidate_detail(result, cfg):
    reqs, labels, mode = cfg["reqs"], cfg["labels"], cfg["mode"]
    st.markdown(f"### {result.resume_id}")
    st.caption(chips([("mode", mode), ("retriever", cfg["kind"]),
                      ("floor", cfg["floor"]), ("top-k", cfg["top_k"]),
                      ("prompt", f"{result.prompt_words} w")]))
    c1, c2, c3 = st.columns(3)
    c1.metric("Model score", f"{result.score}/100",
              help="The single 0-100 value the model returned.")
    if result.score_coverage >= 0:
        c2.metric("Matched-list coverage", f"{result.score_coverage}/100",
                  help="Score implied by the model's own matched list. The study compares "
                       "this against the number the model guessed.")
    if result.resume_id in labels:
        c3.metric("Rubric label", f"{labels[result.resume_id]}/100",
                  help="Synthetic label from data/labels.csv. Never shown to the model.")
    st.markdown(f"**Rationale**  \n{result.rationale or '_no rationale emitted_'}")
    m1, m2 = st.columns(2)
    m1.markdown("**Matched**")
    m1.markdown("\n".join(f"- {x}" for x in result.matched) or "- _none_")
    m2.markdown("**Missing**")
    m2.markdown("\n".join(f"- {x}" for x in result.missing) or "- _none_")
    st.divider()

    if mode == "full":
        st.caption("Full-context baseline: the whole resume was pasted into the prompt, so "
                   "there is no requirement-to-passage map to show. With the 4-bit model "
                   "this mode emitted 33/33 identical scores - it does not rank; in "
                   "Retrieval-only mode the heuristic still varies the score.")
        return
    if not result.evidence_refs:
        st.caption("No retrieval map was recorded for this run.")
        return

    matched_set = set(result.matched)
    rows, evidenced, over = [], 0, 0
    for i, req in enumerate(reqs, 1):
        refs = result.evidence_refs.get(req, [])
        claimed = req in matched_set
        if refs:
            evidenced += 1
        elif claimed:
            over += 1
        rows.append({"#": i, "Requirement": req, "Passages": ", ".join(refs) or "—",
                     "The prompt printed": "evidence" if refs else "none",
                     "Claimed matched": "yes" if claimed else "no"})
    st.markdown("**What the model was shown**")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config={
                     "#": st.column_config.NumberColumn(width="small"),
                     "Requirement": st.column_config.TextColumn(width="large"),
                     "The prompt printed": st.column_config.TextColumn(
                         help="'none' means the requirement reached the prompt with no "
                              "supporting passage - the [i] none line the retrieval floor "
                              "exists to make honest."),
                 })
    st.caption(f"{evidenced} of {len(rows)} requirements arrived with a passage; the prompt "
               f"printed `none` for the rest. The answer lists {len(matched_set)} "
               "requirements as matched.")
    if over:
        st.warning(f"{over} requirements the model called matched had no passage of their "
                   "own in its prompt. That is the over-claim the study measures (24 of 30 "
                   "CVs in the RAG condition) - a finding about the instruction follower, "
                   "not evidence about this candidate.")
    snips = list(dict.fromkeys(s for block in result.evidence.values() for s in block))
    if snips:
        with st.expander(f"The {len(snips)} passages pasted into the prompt"):
            for s in snips:
                st.markdown(f"> {s}")


def page_screen():
    st.title("Screen")
    st.caption("Retrieve only the resume passages that answer each requirement, then score "
               "and explain the match on a CPU-only laptop.")

    with st.container(border=True):
        st.subheader("Run configuration")
        c1, c2, c3, c4 = st.columns([1.15, 1.6, .75, 1.3])
        variant = c1.selectbox("Model / quantization", MODEL_CHOICES, index=DEFAULT_MODEL,
                               key="variant",
                               help="Only Q4_K_M is present on this machine, and only "
                                    "Q4_K_M was benchmarked - one point on the "
                                    "quantization curve.")
        mode = c2.segmented_control("Context strategy", ["rag", "full"], default="rag",
                                    key="mode",
                                    format_func=lambda m: "RAG (retrieved evidence)"
                                    if m == "rag" else "Full resume (baseline)") or "rag"
        top_k = c3.slider("Chunks per requirement", 1, 8, TOP_K, key="top_k")
        kind = c4.pills("Retriever", ["tfidf", "dense"], default="tfidf", key="kind",
                        format_func=lambda e: "TF-IDF (study headline)" if e == "tfidf"
                        else "Dense (MiniLM)") or "tfidf"
        st.caption(f"Similarity floor for `{kind}` is {SIM_FLOOR[kind]}. The two floors are "
                   "~10x apart and not interchangeable - applying one to the other embedder "
                   "is a silent mis-tune. See the Study config page.")

    with st.container(border=True):
        st.subheader("Requisition and candidates")
        jd = st.text_area("Job description (one requirement per line works best)",
                          value=default_jd(), height=170, key="jd")
        reqs = split_requirements(jd)
        if reqs:
            st.markdown(req_chips(reqs))
            st.caption(f"{len(reqs)} requirements parsed. A bare job title does not count "
                       "as one.")
        else:
            st.caption("No requirements parsed yet - write one per line.")
        source = st.radio("Candidate source",
                          ["Sample set (the study corpus)", "Upload my own"], key="source")
        pairs = []
        if source.startswith("Sample"):
            _, pairs = load_sample()
            st.caption(f"{len(pairs)} resumes from `data/resumes`: 3 hand-written CVs plus "
                       "30 generated ones with rubric labels.")
        else:
            files = st.file_uploader("Resumes (.pdf, .docx, .txt)",
                                     type=["pdf", "docx", "txt"],
                                     accept_multiple_files=True, key="uploads")
            pairs = [(f.name, clean(extract_text(f.name, f.read()))) for f in files or []]

        if st.button("Screen candidates", type="primary",
                     disabled=not (reqs and pairs)):
            run_screening(pairs, jd.strip(), mode, top_k, variant, kind)
            st.toast(f"Screened {len(pairs)} candidates.")
            st.rerun()

    with st.container(border=True):
        st.subheader("Last run")
        run = st.session_state.get("run")
        if not run:
            st.info("Nothing has been screened in this session yet.")
            return
        res, reqs = run["results"], run["reqs"]
        scores = [r.score for r in res]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates", len(res))
        m2.metric("Mean score", f"{statistics.mean(scores):.1f}/100")
        m3.metric("Requirements with evidence",
                  f"{statistics.mean([n_evidenced(r) for r in res]):.1f} / {len(reqs)}",
                  help="Average number of requirements that reached the prompt with at "
                       "least one passage. The retrieval floor exists to keep this honest.")
        m4.metric("Mean latency", f"{statistics.mean([r.latency_s for r in res]):.1f} s",
                  help="From this 4-thread CPU laptop, end-to-end with prefill, so lower "
                       "than decode-only numbers llama.cpp reports.")
        st.caption(chips([("model", run["variant"]), ("mode", run["mode"]),
                          ("retriever", run["kind"]), ("floor", run["floor"]),
                          ("top-k", run["top_k"]), ("chunks", run["n_chunks"])]))
        st.caption("Full ranking, per-candidate score bars and the evidence map are on the "
                   "Candidates page.")


def page_candidates():
    run = st.session_state.get("run")
    st.title("Candidates")
    if not run:
        st.info("Run a screening on the Screen page first - this page shows that run.")
        return
    res = run["results"]
    st.caption(chips([("mode", run["mode"]), ("retriever", run["kind"]),
                      ("top-k", run["top_k"]), ("model", run["variant"]),
                      ("n", len(res))]))
    with st.container(border=True):
        st.subheader("Ranking")
        for i, r in enumerate(res, 1):
            c1, c2, c3, c4, c5, c6 = st.columns([.3, 2.3, 1.5, 1.1, 1.1, .6])
            c1.markdown(f"**{i}**")
            none_ct = sum(1 for refs in r.evidence_refs.values() if not refs)
            sub = f"{len(r.matched)} matched · {len(r.missing)} missing"
            if r.evidence_refs:
                sub += f" · {none_ct} reqs had no passage"
            if r.resume_id in run["labels"]:
                sub += f" · rubric label {run['labels'][r.resume_id]}"
            c2.markdown(f"**{r.resume_id}**  \n{sub}")
            c3.progress(r.score / 100, text=f"{r.score}/100")
            c4.metric("Prompt", f"{r.prompt_words} w")
            c5.metric("Latency", f"{r.latency_s:.2f} s")
            if c6.button("Open", key=f"open_{i}_{r.resume_id}"):
                candidate_detail(r, run)
    st.info("Scores support, not replace, human review. Name and personal attributes are "
            "instructed to be ignored, but the fairness audit could not separate identity "
            "from its placebo (n_effective = 1) - that question is unresolved, not closed.")


def page_study_config():
    st.title("Study config")
    st.caption("The constants a run actually uses, and where each headline number comes "
               "from. Read-only.")

    with st.container(border=True):
        st.subheader("Retrieval")
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunking", f"{CHUNK_WORDS}/{CHUNK_OVERLAP}",
                  help="words per chunk / overlapping words. The geometry sweep prices "
                       "45/10 at +0.184 score ceiling for +13.1% prompt - priced, and not "
                       "yet scored with a model.")
        c2.metric("Default top-k", TOP_K)
        c3.metric("Evidence budget", f"{MAX_EVIDENCE_WORDS} w cap",
                  help=f"min({MAX_EVIDENCE_WORDS}, document, max({RAG_MIN_EVIDENCE_WORDS}, "
                       f"{RAG_EVIDENCE_FRACTION} x document)) - a budget that must fill is "
                       "a budget that fabricates.")
        st.dataframe(pd.DataFrame([
            {"Embedder": "tfidf", "Floor": SIM_FLOOR["tfidf"],
             "Strong-weak gap at peak": 2.75, "Swept in": "sweep_tfidf_v2.log"},
            {"Embedder": "dense (MiniLM)", "Floor": SIM_FLOOR["dense"],
             "Strong-weak gap at peak": 2.31, "Swept in": "sweep_dense_fine_v2.log"},
        ]), width="stretch", hide_index=True)
        st.caption(f"Dense embedder weights: `{EMBED_MODEL}`. Gap = requirements left with "
                   "evidence, strong candidate minus weak candidate, at that floor.")

    with st.container(border=True):
        st.subheader("Model files")
        rows = []
        for name, path in MODELS.items():
            present = path.exists()
            rows.append({"Variant": name, "File": path.name,
                         "On disk": "yes" if present else "no",
                         "Size (MB)": round(path.stat().st_size / 1e6, 1) if present else 0,
                         "Benchmarked here": "yes" if name == "Q4_K_M (4-bit)" else "no"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("One quantization level measured end-to-end: a point on the curve, not "
                   "the curve.")

    with st.container(border=True):
        st.subheader("Runtime")
        r1, r2, r3 = st.columns(3)
        r1.metric("Inference threads", N_THREADS)
        r2.metric("Context / max new tokens", "4096 / 220")
        r3.metric("Hardware", "CPU only",
                  help="4-thread laptop. Latency will not transfer; compare avg_latency_s, "
                       "not avg_tok_per_s.")

    with st.container(border=True):
        st.subheader("Provenance")
        for f, note in [
            ("results_v2.csv", "headline tfidf run: rho 0.753 against the rubric labels"),
            ("results_v2_repeat.csv", "second process: 0 of 66 rows differ from the first"),
            ("significance.log", "permutation p, bootstrap CI, paired tests"),
            ("power_analysis.log", "what n = 33 could have detected: MDE ~0.46, and 15.1% "
                                   "power on the embedder comparison"),
            ("retrieval_quality.log", "what the retriever delivers, and the 0.864 score "
                                      "granularity ceiling"),
            ("geometry_sweep.log", "the chunk/budget/top-k/floor frontier, measured with no "
                                   "model"),
            ("bias_q4.log", "identity-vs-placebo audit: no effect separable, n_effective = 1"),
            ("ARTIFACTS.md", "index of every committed measurement - read it before quoting "
                             "any results_*.csv; three configurations disagree"),
        ]:
            st.markdown(f"- `{f}` — {note}" if (ROOT / f).exists()
                        else f"- `{f}` — not in this checkout")
        st.caption("Every number on this page comes from a committed artifact or from "
                   "rag/config.py. Nothing here is a mock value.")


with st.sidebar:
    st.write(f"Signed in as **{st.session_state.user}**")
    if st.button("Log out"):
        st.session_state.user = None
        st.rerun()

nav = st.navigation([
    st.Page(page_screen, title="Screen", icon=":material/search:", default=True),
    st.Page(page_candidates, title="Candidates", icon=":material/groups:"),
    st.Page(page_study_config, title="Study config", icon=":material/settings:"),
])
nav.run()
