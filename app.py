"""Streamlit UI: upload resumes + job description -> ranked, explained shortlist."""
import pandas as pd
import streamlit as st

from rag.config import MODELS
from rag.index import Embedder, ResumeIndex
from rag.ingest import chunk_text, clean, extract_text
from rag.llm import RetrievalOnlyLLM, load_llm
from rag.screen import screen_resume

st.set_page_config(page_title="RAG Resume Screener", layout="wide")
st.title("Efficient RAG + Quantized LLM Resume Screener")
st.caption("Retrieves only the resume passages relevant to each job requirement, "
           "then a quantized local LLM scores and explains the match.")


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
    top_k = st.slider("Chunks retrieved per requirement", 1, 8, 4)

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
            if r.evidence:
                st.markdown("**Retrieved evidence**")
                for req, snippets in r.evidence.items():
                    st.markdown(f"*{req}*")
                    st.caption(snippets[0] if snippets else "-")
    st.info("Scores support, not replace, human review. Name and personal "
            "attributes are instructed to be ignored; audit for bias before real use.")
