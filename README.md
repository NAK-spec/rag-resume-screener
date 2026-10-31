# RAG + Quantized LLM Resume Screener

Prototype for the project *"An Efficient Retrieval-Augmented Generation Framework with Quantized Large Language Models for Automated Resume Screening"*.

## Pipeline

```
Resume (.pdf/.docx/.txt) -> text -> overlapping chunks -> embeddings (vector index)
Job description -> individual requirements
For each requirement: retrieve top-k resume chunks (RAG)
Prompt = JD + retrieved evidence -> quantized GGUF LLM (llama.cpp, CPU) -> JSON
{score, matched, missing, rationale} -> ranked, explainable shortlist
```

Efficiency comes from two things you can measure: **RAG** (short prompts instead of whole resumes) and **quantization** (4-bit / 8-bit weights vs FP16).

## Setup (Windows, PowerShell)

```
cd resume_rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `llama-cpp-python` fails to build, install the prebuilt CPU wheel:

```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

## Download the models

Make a `models` folder and put the GGUF files of ONE base model at different quantization levels in it (default: Qwen2.5-1.5B-Instruct):

```
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf --local-dir models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-fp16.gguf --local-dir models
```

File names are set in `rag/config.py`. Any missing model automatically falls back to retrieval-only scoring, so the app still runs.

## Run

```
streamlit run app.py        # the app
python evaluate.py          # study results -> results.csv
```

## Evaluation

`evaluate.py` runs every quantization level in both RAG and full-context mode and reports Spearman/Kendall agreement with human scores, MAE, latency, tokens/s, memory, model size and prompt length. Replace `data/resumes`, `data/job_description.txt` and `data/labels.csv` with your real labelled set (the 3 samples are only a smoke test).

## Notes

- Add a bias audit: perturb names/universities/gaps and compare score deltas.
- Scores support, not replace, human decisions.
