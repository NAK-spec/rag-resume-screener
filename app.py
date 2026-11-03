"""Streamlit UI: upload resumes + job description -> ranked, explained shortlist."""
import pandas as pd
import streamlit as st

from rag import auth
from rag.config import MODELS
from rag.index import Embedder, ResumeIndex
from rag.ingest import chunk_text, clean, extract_text
from rag.llm import RetrievalOnlyLLM, load_llm
from rag.screen import screen_resume

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("Resume Screener")
st.caption("Retrieves only the resume passages relevant to each job requirement, "
           "then a quantized local LLM scores and explains the match.")

auth.init_db()
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.subheader("Sign in to continue")
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
            if st.form_submit_button("Create account"):
                if p2 != p3:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = auth.create_user(u2, p2)
                    (st.success if ok else st.error)(msg)
    st.stop()

with st.sidebar:
    st.write(f"Signed in as **{st.session_state.user}**")
    if st.button("Log out"):
        st.session_state.user = None
        st.rerun()


@st.cache_resource(show_spinner="Loading model...")
def get_llm(variant: str):
    if variant == "Retrieval-only (no LLM)":
        return RetrievalOnlyLLM()
    return load_llm(MODELS[variant])


@st.cache_resource(show_spinner="Loading embedder...")
def get_embedder():
    return Embedder()


with st.sidebar:
    st.header("Settings")
    variant = st.selectbox("Model / quantization",
                           list(MODELS.keys()) + ["Retrieval-only (no LLM)"])
    mode = st.radio("Context strategy", ["rag", "full"],
                    format_func=lambda m: "RAG (retrieved evidence)" if m == "rag"
                    else "Full resume (baseline)")
    top_k = st.slider("Chunks retrieved per requirement", 1, 8, 3)

jd = st.text_area("Job description (one requirement per line works best)", height=180,
                  value="- Python and pandas\n- Machine learning with scikit-learn\n"
                        "- SQL databases\n- Data visualization\n- Strong communication")
files = st.file_uploader("Upload resumes (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"],
                         accept_multiple_files=True)

if st.button("Screen candidates", type="primary") and files and jd.strip():
    resumes = {f.name: clean(extract_text(f.name, f.read())) for f in files}
    chunks = [c for rid, t in resumes.items() for c in chunk_text(rid, t)]
    index = ResumeIndex(get_embedder())
    index.build(chunks, extra_corpus=[jd])
    llm = get_llm(variant)
    if variant != "Retrieval-only (no LLM)" and isinstance(llm, RetrievalOnlyLLM):
        st.warning(f"Model file for '{variant}' not found in ./models - using "
                   "retrieval-only scoring. See README to download it.")
    if index.embedder.kind == "tfidf":
        st.caption(f"Retrieval: TF-IDF (dense unavailable - "
                   f"{index.embedder.dense_error}). The similarity floor is tuned for this "
                   "embedder only.")

    results = []
    bar = st.progress(0.0)
    for i, (rid, txt) in enumerate(resumes.items()):
        results.append(screen_resume(rid, jd, index, llm, mode=mode,
                                     full_text=txt, top_k=top_k))
        bar.progress((i + 1) / len(resumes))
    results.sort(key=lambda r: -r.score)

    st.subheader("Ranking")
    st.dataframe(pd.DataFrame([{
        "Rank": i + 1, "Resume": r.resume_id, "Score": r.score,
        "Matched": len(r.matched), "Missing": len(r.missing),
        "Latency (s)": round(r.latency_s, 2), "Prompt words": r.prompt_words,
    } for i, r in enumerate(results)]), use_container_width=True, hide_index=True)

    st.subheader("Explanations")
    for r in results:
        with st.expander(f"{r.resume_id} - {r.score}/100"):
            st.write(r.rationale)
            c1, c2 = st.columns(2)
            c1.markdown("**Matched**\n" + "\n".join(f"- {m}" for m in r.matched))
            c2.markdown("**Missing**\n" + "\n".join(f"- {m}" for m in r.missing))
            if r.evidence_refs:
                st.markdown("**What the model was shown**")
                st.caption(f"{len({p for ps in r.evidence_refs.values() for p in ps})} "
                           f"unique passages, {r.prompt_words} prompt words "
                           f"(retrieved evidence only, not the whole resume)")
                for i, (req, refs) in enumerate(r.evidence_refs.items(), 1):
                    st.markdown(f"**[{i}]** {req} → " + (", ".join(refs) if refs
                                                         else "*no supporting passage*"))
                for s in dict.fromkeys(s for snips in r.evidence.values() for s in snips):
                    st.caption(s)
    st.info("Scores support, not replace, human review. Name and personal "
            "attributes are instructed to be ignored; audit for bias before real use.")
