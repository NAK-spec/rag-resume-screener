# RAG + Quantized LLM Resume Screener

Prototype for the project *"An Efficient Retrieval-Augmented Generation Framework with
Quantized Large Language Models for Automated Resume Screening"*.

## Pipeline

```
Resume (.pdf/.docx/.txt) -> text -> overlapping chunks -> embeddings (vector index)
Job description -> individual requirements
For each requirement: retrieve top-k resume chunks
Merge the hits into ONE de-duplicated passage set, capped at 40% of the resume
Prompt = numbered requirements + passages + requirement->passage map -> quantized GGUF LLM (llama.cpp, CPU)
{score, matched, missing, rationale} -> ranked, explainable shortlist
```

Efficiency comes from two measurable things: **RAG** (a bounded evidence prompt instead of
the whole resume) and **quantization** (4-bit weights instead of FP16). Only the first is
measured here - see [Results](#results-measured); one quantization level is present, so
there is no efficiency curve along the second axis.

The evidence step matters. Retrieving top-k chunks *per requirement* and concatenating
them repeats the same resume text five or six times over, which made the "efficient" RAG
prompt **longer** than the full-resume baseline it was meant to beat. `select_evidence()`
keeps every chunk once, ranked by its best similarity across all requirements, under a
budget that scales with the document (`rag/screen.py:54`). Measured on the 33-resume set,
that turns the RAG prompt from 304 -> 416 words (i.e. *longer* than the baseline it was
supposed to beat) into 304 -> 172 words.

## Setup (Windows, PowerShell)

```
cd resume_rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `llama-cpp-python` fails to build, install the prebuilt CPU wheel. PyPI only ships an
sdist for this package, so without a wheel pip tries to compile it and needs a C++ toolchain:

```
pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python
```

`--only-binary=:all:` forces the wheel rather than letting the resolver fall back to the
PyPI sdist; the CPU index carries `cp3XX` and `py3-none-win_amd64` wheels (0.3.36 exists for
Python 3.13 on win_amd64). If your Python version has no matching wheel tag there, the
source build is the only option and it will require CMake plus a compiler.

For the dense embedder, torch's PyPI wheel is the one that stalls on a slow link. The
verified five-command route via `download.pytorch.org/whl/cpu` (torch 2.15.0+cpu,
sentence-transformers 6.2.0 on Python 3.13) is in `requirements.txt`. Without it everything
still runs, on TF-IDF retrieval - which is why `evaluate.py`/`bias_audit.py`/`sweep_floor.py`
take `--embedder {auto,tfidf,dense}` and *fail loudly* when a pinned embedder is not the one
that got built. `auto` is fine for the app and useless for a comparison: a run that meant to
be dense and silently fell back to TF-IDF would be quoting a similarity floor tuned on the
wrong scale.

## Download the models

Put GGUF files of ONE base model at the quantization levels you want to compare in
`models/` (default: Qwen2.5-1.5B-Instruct). One level is enough to run the screening path:

```
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf --local-dir models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-fp16.gguf --local-dir models
```

File names are set in `rag/config.py`. A variant whose GGUF file is missing is *skipped*
by `evaluate.py` (it would silently run the retrieval-only fallback and put three identical
rows in `results.csv` that look like three quantization results); pass `--include-missing`
to force them or `--fallback` to benchmark the no-LLM baseline deliberately.

## Dataset

`data/resumes` holds 3 hand-written smoke-test CVs plus 30 generated ones, with
`data/labels.csv` giving the human score for each:

```
python data/generate_dataset.py            # seed 7, 30 resumes, overwrites the generated set
python data/generate_dataset.py --n 60 --seed 12
```

`generate_dataset.py` documents the rubric the labels encode: each job requirement is
`proven` (concrete evidence in a role or project), `claimed` (only in the Skills line) or
`absent`, weighted 2 for the three core skills and 1 for the rest. Resumes vary in length
(105-480 words for the generated set; the three hand-written CVs are 39, 56 and 85),
paraphrase skills instead of always naming the tool, carry irrelevant
work text for retrieval to drop, and include personal details that the rubric ignores.
Reproducible via the seed.

`python data/check_dataset.py` verifies the corpus rather than trusting it: labels and
files paired both ways, no duplicate texts, experience ranges ordered, and whether the
generator in this checkout still reproduces the shipped text. Its current output
(`data/check_dataset.log`) is 12 FAIL / 3 WARN, all of them the reversed experience
described in Limitations.

## Run

```
streamlit run app.py        # the app
python evaluate.py --embedder tfidf --out results.csv --dump per_resume.csv
python sweep_floor.py --embedder tfidf --min-sim 0.08   # what each floor costs (no LLM)
python bias_audit.py --limit 6 --embedder tfidf --model "Q4_K_M (4-bit)" --out bias_q4.csv
python data/check_dataset.py   # dataset integrity + generator drift (no model needed)
python tests/test_pipeline.py    # or: pytest tests -q
```

The app's sidebar has a **Retriever** control, `TF-IDF (study headline)` (default) or
`Dense (MiniLM)`, so a demo can reproduce the configuration behind the 0.753 rather than
always running dense. Pick **Retrieval-only (no LLM)** in the model select to browse the
ranking and the per-requirement evidence without loading the 1.1 GB GGUF; the GGUF is only
read when "Screen candidates" is pressed.

**Every committed measurement file - which embedder, which floor, which of them are
duplicates or superseded - is indexed in [ARTIFACTS.md](ARTIFACTS.md).** Read it before
quoting a number from any `results_*.csv`: four retrieval configurations are present and
three of them disagree.

## Evaluation

`evaluate.py` runs every available quantization level in both RAG and full-context mode and
reports Spearman/Kendall agreement with human scores, MAE, +/-10-point accuracy, latency,
tokens/s, peak RSS, model size, prompt length and the RAG-vs-full prompt compression ratio.
The dataset is synthetic and rubric-labelled, so treat the correlations as evidence the
pipeline ranks sensibly on controlled data, not as a claim about real hiring decisions;
swap in a real labelled set before drawing conclusions.

## Results (measured)

All figures below are from real runs on this machine (Windows, CPU-only, Python 3.13,
`temperature=0`, `TOP_K=3`, evidence budget 40% of the resume). Nothing here is simulated.
Each table names the file it came from; `ARTIFACTS.md` lists all of them, including the
superseded ones that still sit in this directory.

Setup actually measured: **Qwen2.5-1.5B-Instruct Q4_K_M** (1,117 MB GGUF, llama.cpp CPU),
run twice over the same 33 resumes - once with **TF-IDF retrieval** and once with the dense
**`all-MiniLM-L6-v2`** embedder (torch 2.15.0+cpu). Each embedder uses its own similarity
floor, swept with `sweep_floor.py` (`rag/config.py`: `{"tfidf": 0.024, "dense": 0.24}`), and
the runs are pinned with `--embedder` so a silent fallback cannot pass one run off as the
other. n = 33 resumes, 111 chunks, 6 requirements.

Artifacts: `results.csv` / `per_resume.csv` (TF-IDF LLM run - note `results.csv` is a
byte-identical copy of `results_floor024.csv`, i.e. **one** process, not two runs),
`results_dense.csv` / `per_resume_dense.csv` (dense LLM run), `results_floor024.csv` +
`results_floor030.csv` (the floor A/B), `results_tfidf_q4.csv` (pre-floor ablation, kept for
that comparison), `results_fallback_floor024.csv` (no-LLM baseline; `results_fallback.csv` is
the superseded pre-fix version), `sweep_tfidf.log` + `sweep_dense.log` (the two floor grids),
`bias_q4.csv` / `bias_q4.log`, `bias_retrieval_only_control.csv`.
Runs made from here on print both `embedder:` and `sim_floor:` in their first log line; the
older artifacts print `embedder:` only, and their floor is whatever `rag/config.py` held at
the time (0.024 / 0.030 for the two named above).
(`bias_results.csv` is an earlier no-LLM prototype run; it predates every fix below and
should not be quoted.)

Reproduce (each LLM run is ~15-25 min of CPU generation on this machine; keep them serial,
they contend for the same 4 threads and latency is part of what is being measured):

```
python evaluate.py --embedder tfidf --out results.csv --dump per_resume.csv
python evaluate.py --embedder dense --out results_dense.csv --dump per_resume_dense.csv
python evaluate.py --fallback --embedder tfidf --variant "Retrieval-only (no LLM)" --out results_fallback_floor024.csv
python sweep_floor.py --embedder tfidf --min-sim 0.08   # -> sweep_tfidf.log, no LLM
python sweep_floor.py --embedder dense --min-sim 0.45   # -> sweep_dense.log, no LLM
python bias_audit.py --limit 6 --embedder tfidf --model "Q4_K_M (4-bit)" --out bias_q4.csv
python bias_audit.py --limit 6 --embedder tfidf --out bias_retrieval_only_control.csv
```

### RAG vs full context, same model

TF-IDF retrieval, floor 0.024 (`results.csv`):

| mode | Spearman | Kendall | MAE | within ±10 | latency | tok/s | prompt words |
|---|---|---|---|---|---|---|---|
| rag | **0.753** | 0.629 | 23.9 | 0.333 | **13.9 s** | 9.6 | **172** |
| full | 0.262 | 0.225 | 30.5 | 0.273 | 21.9 s | 8.6 | 304 |

- **Efficiency claim holds**: the RAG prompt is 43.4% shorter than the full-resume prompt
  and is shorter on 30 of 33 resumes (median 177 vs 297 words). Per-candidate latency
  drops 36.7% and throughput rises 11.6% (9.6 vs 8.6 tok/s), at the same model load time
  (2.03 s) and effectively the same memory (2,074 vs 2,110 MB peak RSS).
- **The efficiency half is embedder-independent.** The dense run reproduces it: 304 -> 178
  words (+41.5%), RAG shorter on the same 30 of 33 resumes, and the same two full-context
  scores for all 33 candidates. What the embedder changes is the *quality* column, not the
  compression (see the TF-IDF-vs-dense section).
- **Where it does not hold**: the 3 resumes where RAG is longer are exactly the 3 under
  90 words (`per_resume.csv`: 39, 56 and 85 words), each by a constant 21 words - that is
  the requirement-to-passage mapping, and when a resume already fits in one chunk there is
  nothing left for retrieval to compress. Below ~100 words the baseline is the cheaper
  prompt, so the honest claim is "RAG compresses any resume with more than one chunk",
  not "RAG always compresses".
- **Quality does not pay for that saving - it comes with it.** RAG separates candidates in
  ways that track the rubric (rho 0.753), while full context barely varies its answer at
  all (see the degeneracy note below - 32 of 33 candidates get the same number).
  Restricting the model to the passages that actually answer each requirement beats letting
  it read everything.
- MAE of ~24 points is still poor in absolute terms, and the +/-10-point hit rate is only
  0.333 (labels span 10-100, mean 58.5): the model orders candidates much better than it
  scores them, so this is a *shortlisting* tool, not a scoring one. Note also that the
  hit rate did **not** improve with the floor (0.364 at floor 0.03 -> 0.333 at 0.024)
  while rho and MAE did, which is what a ranking-shaped fix looks like.

### TF-IDF vs dense retrieval, same model and same budget

`all-MiniLM-L6-v2` at its own swept floor (0.24) against TF-IDF at 0.024, same GGUF, same
33 resumes, same 6 requirements, `temperature=0` (`results.csv` vs `results_dense.csv`):

| embedder | Spearman | Kendall | MAE | within ±10 | signed score offset | mean abs gap to own coverage | latency | prompt words | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| tfidf (floor 0.024) | **0.753** | 0.629 | 23.9 | 0.333 | **-22.0** | 30.8 | **13.9 s** | **172** | **2,074 MB** |
| dense (floor 0.24) | 0.668 | 0.568 | **16.6** | **0.394** | +6.2 | **9.5** | 16.2 s | 178 | 2,463 MB |

- **The ranking difference is not resolvable at n = 33.** 0.753 has a 95% Fisher-z interval
  of [0.553, 0.871] and 0.668 of [0.421, 0.823]; they overlap almost completely, so the
  defensible sentence is "TF-IDF + floor ranked at least as well as dense here", not
  "TF-IDF is better". That is a slightly awkward result for a project whose title says
  *retrieval-augmented*, because the cheap, dependency-free embedder won the headline metric.
- **What is not noise is the scale.** Dense moves the model from systematically harsh
  (mean score 36.5 against labels averaging 58.5) to near-centred (64.7), which is where
  the MAE 23.9 -> 16.6 and hit-rate 0.333 -> 0.394 gains come from - not from a better
  ordering. The two score vectors disagree substantially: Spearman between the TF-IDF and
  dense RAG scores is only **0.462**, 15 of 33 candidates sit 30+ points apart (mean |delta|
  between the two runs is **29.7 points**), and the 23 candidates TF-IDF put below 40 split
  8/15 across the dense bands. For scale: swapping embedders moves a candidate's score about
  twice as much as the entire fairness audit's identity edits did (mean 13.6 points), so the
  choice of embedder is this pipeline's largest measured lever - larger than anything the
  prompt does.
- **Dense retrieval is internally more consistent.** The model's number vs the coverage
  implied by its own `matched` list differs by 30.8 MAE under TF-IDF and 9.5 under dense,
  and only 4 of 33 candidates diverge by more than 30 points (15 under TF-IDF). The dense
  floor is gentler on evidence (21.2% of requirements come back empty vs 51.5% for TF-IDF,
  1.09 vs 1.03 surviving passages per candidate), so the model is handed a set it can
  actually stand behind - at 17% more wall time and 19% more memory.
- Both runs are single-passage retrieval, and both flag degenerate: dense mode puts 19 of 33
  candidates on 80 (modal share 0.576), TF-IDF puts 20 of 33 on 20 (0.606). Same ceiling,
  different ceiling value.
- The three hand-written CVs make the trade concrete (human label / tfidf / dense): weak
  10 / 0 / 0, medium 50 / 20 / 80, strong 90 / 85 / 80. TF-IDF gets the extremes right and
  flattens the middle; dense gets the middle wrong and clusters near the top.


### Both modes score degenerately - read the rho as band agreement

Before comparing the two rho values, look at what the model actually emits (`per_resume.csv`):

| run | distinct scores | distribution | modal share |
|---|---|---|---|
| rag, tfidf | 5 of 33 candidates' worth of resolution | 20 (x20), 80 (x7), 85 (x2), 0 (x3), 75 (x1) | 0.61 |
| rag, dense | 5 | 80 (x19), 85 (x6), 20 (x4), 0 (x3), 25 (x1) | 0.58 |
| full (either embedder) | **2** | 85 (x32), 60 (x1) | 0.97 |

In full-context mode the model gives 32 of 33 candidates exactly 85 - it cannot rank at
all, and its rho of 0.262 is carried by the single candidate it scored differently. So the
defensible sentence is not "RAG beats full context 0.753 vs 0.262"; it is **"RAG separates
candidates into bands that track the rubric, while full context collapses to a constant."**
`evaluate.py` now flags any row whose modal score covers more than half the corpus instead
of letting it through as a correlation.

RAG mode is also coarse: 61% of candidates land on 20, and coverage takes only four values
(0/33/67/83). A Spearman of 0.753 on five levels is agreement about *which tier* a
candidate belongs in, not a fine-grained ordering, and it is why the +/-10 hit rate stays
at 0.333 while the ranking looks good. Any thesis claim here should be phrased as
shortlist tiers, and the tie handling stated: scipy `spearmanr` ranks ties by average, and
the Kendall figures are tau-b, which discounts tied pairs.


`results.csv` reports each variant twice: `spearman` is the LLM's own 0-100 guess,
`cov_spearman` is a deterministic score computed from what retrieval actually found
(weighted count of requirements with surviving evidence, `screen_resume` in
`rag/screen.py`). That second column exists because the two disagree more often than they
agree: in `per_resume.csv`, the model's guess and its own matched list differ by more than
30 points on **15 of 33** resumes (max 63, they coincide exactly on 3) - and every single
one of those 15 has the model scoring *below* what its own citations support. The dense run
diverges on only 4 of 33 (max gap to coverage 9.5 MAE), so the self-contradiction is a
property of what retrieval hands the model, not of the model alone.

| mode | LLM score rho / MAE | coverage rho / MAE |
|---|---|---|
| rag, tfidf | 0.753 / 23.9 | 0.501 / 22.8 |
| rag, dense | 0.668 / 16.6 | 0.566 / 23.1 |
| full (either embedder) | 0.262 / 30.5 | -0.177 / 53.6 |

So the LLM wins on ranking and roughly ties on absolute error - it is a better sorter than
a better scorer. Note that this comparison is **not** an independent validation of
coverage: `score_coverage` counts evidenced requirements, and the synthetic labels were
built from the same rubric, so coverage agreeing with the labels is partly circular. What
coverage usefully exposes is the scale problem, which is a constant offset per mode:

| mode | mean human | mean LLM score | mean signed gap | error quartiles | MAE raw -> after removing the offset |
|---|---|---|---|---|---|
| rag, tfidf | 58.5 | 36.5 | -22.0 | -47 / -14 / -2 | 23.9 -> 19.7 |
| rag, dense | 58.5 | 64.7 | +6.2 | -10 / +3 / +13 | 16.6 -> 16.6 |
| full | 58.5 | 84.2 | +25.8 | | 30.5 -> 25.1 |

With retrieved evidence under TF-IDF the model is systematically harsh (25 of 33 candidates
scored below what their own citations support, 5 above); with the whole resume it is
systematically lenient. The offset is not stable across prompt strategy - it flips sign and
is worth ~26 points of MAE - so neither mode is calibrated, which is exactly why the tool
should report ranks and citations rather than a number to hand to a candidate. The
embedders separate *why* the error is what it is: TF-IDF's error distribution is one-sided
(all four quartiles at or below zero), so most of its 23.9 MAE is a constant harshness that
recalibration would remove, while dense's straddles zero (+6.2 offset, MAE unchanged by
removing it), so its 16.6 is residual disagreement per candidate, not a scale bug. The
other thing coverage exposes is in the first table: in full mode the coverage heuristic
correlates *negatively* with the labels (-0.177), because when every requirement can be
matched against the entire document, "has a passage about X" stops discriminating. Read
together with the floor ablation below: retrieval does most of the ranking work, and the
LLM's judgement is what lifts evidenced-coverage agreement (0.501) to 0.753.


### Run-to-run determinism

Greedy decoding (`temperature=0`) makes the scoring path reproducible, and two things were
checked with the artifacts that exist rather than assumed:

- **Two independent processes at floor 0.03** (`results_q4_tfidf.csv` and
  `results_floor030.csv`) produced byte-identical quality figures: rho 0.560 / 0.262, tau
  0.474 / 0.225, MAE 27.76 / 30.48, within-10 0.364 / 0.273, 172 / 304 prompt words, rag
  peak RSS 2,074 MB. Only the timings moved: 13.72 -> 13.93 s (rag), 22.43 -> 22.83 s (full),
  i.e. ~1.5-3% CPU noise.
- **Full-context mode reproduced across four runs**, including across both embedders:
  `per_resume.csv` (tfidf) and `per_resume_dense.csv` (dense) give the same score for all 33
  candidates in `full` mode (verified by list equality), and rho 0.262 / MAE 30.48 in every
  run made after the parsing fixes - the older `results_tfidf_q4.csv` gives 0.084 there
  because it predates them, which is itself evidence the fix changed a real behaviour.
  That is expected - full mode never touches retrieval - but it is the cleanest
  determinism evidence in the project because it holds across a changed dependency, not just
  a re-run.

**What is NOT reproduced: the headline 0.753.** `results.csv` is a byte-identical copy of
`results_floor024.csv` (same md5), i.e. one process, and no second floor-0.024 LLM run was
made. The 0.753 figure rests on a single run, mitigated but not settled by the two
reproductions above. `python evaluate.py --embedder tfidf --out results_repeat.csv` is the
command that would close it (~20 min of CPU; left unrun deliberately).

Consequence for the fairness section: because the scoring path is deterministic, the
60-point swings there are *input* sensitivity, not sampling noise - which is exactly why the
placebo control was needed to say anything about what the input was.


### The relevance floor is what made RAG work

`index.search` originally returned top-k chunks no matter how weakly they matched, so every
requirement always came back with "evidence" - including for a hospitality CV that has no
data skills at all. Two artifact-backed views of the same pathology:

- **Structurally** (`sweep_tfidf.log`, first row): at floor 0.000 every one of the 33 resumes
  returns evidence for all 6 requirements - weak candidates 6.00, strong candidates 6.00,
  separation exactly 0.00, and 0% of requirements reported as unevidenced. Retrieval simply
  cannot express "this candidate has nothing to say about that".
- **At the model** (`per_resume.csv`, `full` mode - which hands the LLM the entire resume and
  is the same failure in prompt form): the hospitality CV gets **85/100 with 5 of 6
  requirements marked matched**, and is indistinguishable from the strong candidate, who gets
  85 with the same 5 matched.

| candidate (human label) | rag, floor 0.024 | full context |
|---|---|---|
| weak, hospitality CV (10) | **0**, 0/6 matched, 5 missing | 85, 5/6 matched |
| medium (50) | 20, 4/6 matched | 85, 5/6 matched |
| strong (90) | 85, 5/6 matched | 85, 5/6 matched |

Corpus-wide, the floor moves RAG-mode Spearman from **-0.083** (no floor,
`results_tfidf_q4.csv`) to **0.560** (floor 0.03, `results_floor030.csv`) to **0.753** (floor
0.024, `results.csv`). Negative correlation is worse than random, and it is the *efficiency*
path that caused it: a top-k-per-requirement budget guarantees retrieving noise when a
candidate lacks a skill, and the model faithfully reports the noise as proof. The floor costs
some true evidence (the strong candidate keeps 5 of 6 either way, but the corpus loses half
its evidenced requirements) and no input tokens (median 178 -> 177 words), but it does cost
output time: 11.4 -> 13.9 s per candidate, because a candidate with empty evidence sets makes
the model spend tokens listing what is missing. It is still comfortably cheaper than the 21.9
s full-context baseline, so the honest framing is "the floor buys the ranking and pays ~2.5 s
per candidate", not "the floor is free". "No supporting passage" is now a legitimate retrieval
result and is reported as such; it is what turns `missing` from a guess into something the
pipeline observed.

*Provenance note:* an earlier draft of this section tabulated the same three CVs under "no
floor" with LLM scores of 100/80/80. That ablation was run interactively and its output was
never saved, so it is not reproducible from this repository and has been replaced by the two
verifiable views above.

### Where the floor value comes from (and what it costs)

Retrieval-only sweep over all 33 resumes x 6 requirements, no LLM involved, from
`sweep_floor.py` - figures are mean requirements that return at least one surviving passage.
The two grids below are the exact output of `sweep_tfidf.log` and `sweep_dense.log`, so the
chosen value is the row marked `*` in each.

**TF-IDF** (`python sweep_floor.py --embedder tfidf --min-sim 0.08`):

| floor | weak (<40) | medium (40-74) | strong (>=75) | strong-weak | passages/resume | requirements with NO evidence |
|---|---|---|---|---|---|---|
| 0.000 | 6.00 | 6.00 | 6.00 | 0.00 | 1.12 | 0.0% |
| 0.008 | 3.82 | 5.33 | 5.60 | 1.78 | 1.12 | 18.2% |
| 0.016 | 2.00 | 3.67 | 4.50 | 2.50 | 1.06 | 43.9% |
| **0.024** | 1.55 | 3.00 | 4.30 | **2.75** | 1.03 | 51.5% |
| 0.032 | 1.36 | 2.33 | 3.30 | 1.94 | 1.03 | 61.6% |
| 0.040 | 1.27 | 1.75 | 2.30 | 1.03 | 0.94 | 70.7% |
| 0.048 | 1.00 | 1.42 | 1.50 | 0.50 | 0.85 | 78.3% |
| 0.056 | 0.64 | 1.33 | 0.90 | 0.26 | 0.73 | 83.8% |
| 0.064 | 0.45 | 1.25 | 0.80 | 0.35 | 0.67 | 85.9% |
| 0.072 | 0.36 | 1.00 | 0.80 | 0.44 | 0.58 | 87.9% |
| 0.080 | 0.18 | 0.83 | 0.70 | 0.52 | 0.52 | 90.4% |

**Dense, `all-MiniLM-L6-v2`** (`python sweep_floor.py --embedder dense --min-sim 0.45`):

| floor | weak | medium | strong | strong-weak | passages/resume | NO evidence |
|---|---|---|---|---|---|---|
| 0.000 | 5.91 | 6.00 | 6.00 | 0.09 | 1.12 | 0.5% |
| 0.135 | 5.82 | 5.92 | 6.00 | 0.18 | 1.12 | 1.5% |
| 0.180 | 4.64 | 5.92 | 5.90 | 1.26 | 1.09 | 8.6% |
| 0.225 | 3.64 | 5.75 | 5.70 | 2.06 | 1.09 | 16.2% |
| **0.240** | 3.09 | 5.67 | 5.40 | **2.31** | 1.09 | 21.2% |
| 0.270 | 2.36 | 4.33 | 4.60 | 2.24 | 1.09 | 37.4% |
| 0.315 | 1.64 | 3.33 | 3.70 | 2.06 | 1.06 | 52.0% |
| 0.360 | 1.09 | 2.33 | 2.80 | 1.71 | 1.06 | 65.7% |
| 0.405 | 0.73 | 1.50 | 1.40 | 0.67 | 0.88 | 79.8% |
| 0.450 | 0.45 | 0.75 | 0.80 | 0.35 | 0.58 | 88.9% |

Four things these grids settle:

- **At floor 0.000 the tier separation is 0.00 (TF-IDF) / 0.09 (dense)** - top-k never says
  "nothing", so the signal "this candidate lacks the skill" is structurally unavailable to
  the model. This is why the un-floored run correlated *negatively*.
- **The scales are ~10x apart and not interchangeable.** The dense peak sits at 0.240, ten
  times the TF-IDF value; running the TF-IDF floor against dense embeddings would filter
  essentially nothing (gap 0.09), and running 0.24 against TF-IDF would leave almost no
  candidate with evidence at all. `SIM_FLOOR` is a dict for exactly this reason.
- **Past the peak the filter turns on the good candidates.** By 0.048 TF-IDF gives a strong
  candidate 1.50 evidenced requirements against a weak candidate's 1.00, and by 0.405 dense
  gives 1.40 against 0.73 - separation collapses to noise.
- **Dense separates tiers less sharply at its own peak** (2.31 vs 2.75 requirements) while
  leaving far fewer requirements empty (21.2% vs 51.5%). That is the retrieval-side
  explanation for the eval result above: TF-IDF + floor ranks better because it is willing
  to say "no evidence" about half the requirements, dense ranks worse and scores closer
  because it keeps asserting evidence.

The cost is worth stating plainly: dedup plus the floor leaves **about one passage** per
candidate (`per_resume.csv`: mean 1.03, median 1.0, and 2 candidates with none at all;
dense: mean 1.09, 1 candidate with none).
This is single-passage retrieval, not multi-passage aggregation - the 43% prompt
compression is partly a consequence of that narrowness, and a resume that proves one skill
across three sections still shows the model only one of them. The 2 zero-passage candidates
are also the interesting ones: the pipeline returns them with an empty evidence set, the
model gives them 0/100, and that is the behaviour a floor is supposed to produce.


### No-LLM baseline

Retrieval-only scoring (no generation at all) reaches rho **0.306** / tau 0.249 / MAE 54.5 /
within-10 0.061 on the same 33 resumes (`results_fallback_floor024.csv`), so the quantized
LLM contributes real ranking value (0.306 -> 0.753 with TF-IDF, -> 0.668 with dense).

One caveat that changes how to read that gap: **the baseline does not respond to the floor
below its own anchor.** Re-running it at floors 0.000 / 0.024 / 0.030 / 0.050 gives the
identical rho 0.306, because `RetrievalOnlyLLM` rescales similarities through
`SIM_ANCHORS["tfidf"] = (0.05, 0.30)` and clamps everything under 0.05 to zero fit - so any
floor at or below 0.05 is invisible to it. It only starts moving above the anchor
(0.080 -> 0.336, 0.120 -> 0.358). The floor's -0.083 -> 0.753 swing is therefore a property
of what the *LLM* is shown, not something a similarity-threshold scorer would have found on
its own - and the prompt compression and latency rows above belong to the retrieval stage,
so they are not quantization effects. With one quantization level present, this project does
not measure the quantization axis.

### Fairness audit (6 base resumes x 19 variants = 114 screenings, 108 comparisons)

`bias_audit.py --limit 6 --embedder tfidf --model "Q4_K_M (4-bit)"` at the study's floor
(0.024): each base resume is stripped of its candidate line, scored as a baseline, then
scored again with one line added back - a name, a university, a career gap, a
gender/age/marital line, or a **placebo** line that carries no identity at all
("References available on request.", "Document ID: 4471-B (internal tracking).",
"Availability: immediate, open to relocation."). 108 comparisons, signed deltas, `bias_q4.csv`.

| factor | comparisons | mean abs change | median | max | flagged (>10) | >=30 pts | changed at all |
|---|---|---|---|---|---|---|---|
| personal details | 24 | 18.3 | 0 | 60 | 8 | 6 | 8/24 |
| **placebo** | 18 | **14.7** | 0 | 60 | 6 | 5 | 7/18 |
| university | 24 | 13.3 | 0 | 60 | 7 | 6 | 8/24 |
| employment gap | 12 | 13.8 | 0 | 60 | 4 | 3 | 5/12 |
| name | 30 | 9.3 | 0 | 60 | 5 | 5 | 5/30 |

**The placebo is the result, and it demotes the bias claim.** Content-free lines move the
score *more* on average (14.7) than the identity factors do (13.4 pooled), and produce
5 of the 27 swings of 30+ points. The audit was designed to attribute a delta to an identity,
and it cannot: the same edit-anything sensitivity explains the name table. The earlier, more
alarming version of this section (before the placebo existed) reported "all five names lower
the score by ~19 points" - that is now known to be indistinguishable from "a line was added
to the top of this document".

**Where the movement actually comes from: 2 of 6 candidates.**

| base resume | baseline score | comparisons moved | swings >=30 | mean abs change |
|---|---|---|---|---|
| `medium_candidate.txt` | 80 | 16/18 | 14 | 47.5 |
| `r03_medium.txt` | 20 | 13/18 | 13 | 32.2 |
| `r05_strong.txt` | 20 | 3/18 | 0 | 0.8 |
| `r01_weak.txt` | 0 | 1/18 | 0 | 1.1 |
| `r02_strong.txt` | 20 | 0/18 | 0 | 0.0 |
| `r04_weak.txt` | 0 | 0/18 | 0 | 0.0 |

Every 30+ point delta in the whole audit belongs to one of those two resumes. The other four
are effectively invariant - `r02_strong.txt` and `r04_weak.txt` do not move on a single one
of their 18 variants. So what the instrument is measuring is a knife-edge: two candidates sit
on a decision boundary, and any perturbation - identity or placebo - tips them across it, in
steps of 20 to 60 points because the model's output only takes a handful of values (see the
degeneracy section). Four candidates nowhere near a boundary show the correctly behaved
result: identical scores.

**What survives of the name finding.** All five name variants move `medium_candidate.txt`
downward (80 -> 20-25), and no other resume moves for names at all. That is *one* candidate
counted five times, so n_effective = 1, not 30. The differential between the Black-coded name
(Lakisha Washington, -60) and the White-coded ones (-55) is 5 points on that single
candidate, which is inside the placebo's own range (-20 to +40). The honest sentence is
"this audit found no evidence separable from perturbation noise, and cannot: it has n = 1 for
the only identity effect it produced", not "the model is name-blind" and not "the model is
name-biased".

**The retrieval-only control confirms the swing is introduced at generation.** The identical
114 variants scored without an LLM (`bias_retrieval_only_control.csv`, same embedder and
floor) move by at most **2** points across all 108 comparisons - mean 0.20, placebo mean 0.22,
zero flagged. Retrieval is blind to these details; the 60-point steps are the 1.5B model's.

**Design lesson worth stating in the thesis:** a single-factor perturbation audit needs a
placebo factor and needs its deltas reported per base document, because a mean over
comparisons pools one unstable candidate with five stable ones and reports the result as a
group property. To make the identity question answerable at all, each variant needs
replication (several paraphrases or a temperature > 0 sample) so that "the score moved" can
be compared against "the score moves by this much anyway".

## Limitations

- One model, one quantization level (Q4_K_M). Q8_0 and FP16 are absent, so there is no
  quantization-efficiency curve; those variants are skipped rather than faked. The title's
  "quantized LLM" half is therefore a single point on the size/accuracy curve, not a curve.
- Both embedders were measured (TF-IDF and `all-MiniLM-L6-v2`), each at its own swept floor,
  but **each only once**: the dense run is a single process, and the TF-IDF headline (0.753)
  is a single process too (see the determinism section). The 0.753-vs-0.668 ordering between
  embedders is inside the noise at n = 33 and should not be claimed either way.
- Labels are synthetic and rubric-derived, so agreement measures whether the pipeline
  reproduces a stated rubric, not real hiring judgement. n = 33 is small; rho 0.753 has a
  95% Fisher-z interval of [0.553, 0.871] and the dense run's 0.668 of [0.421, 0.823], while
  full context's 0.262 sits below both intervals. The RAG-vs-full gap is the one comparison
  here that survives its own confidence interval; the embedder comparison does not.
- The shipped dataset has a cosmetic text defect that is **not** in the generator any
  more: 10 of the 30 generated resumes carry an experience range that runs backwards
  (`Experience: BI Analyst (fintech, 2023-2022)`), because the graduation-year clamp could
  return a start later than its own end once the role cursor stepped below graduation year.
  `data/generate_dataset.py` now has `role_years()` and
  `tests/test_pipeline.py::test_role_year_ranges_never_run_backwards` pins it, but the
  resumes were **not** regenerated: that would change chunk text for 10 of 33 candidates
  and invalidate every results CSV measured on them, and re-measuring costs ~20 minutes of
  CPU. `python data/check_dataset.py` states this explicitly (12 FAIL / 3 WARN) and proves
  the drift is confined to those 12 `Experience:` lines, so the corpus behind the tables is
  the one labelled "measured on the shipped text" in `ARTIFACTS.md`.
- A second cosmetic finding from the same checker: the two hand-written CVs are 39 and 56
  words, under the 60-word band the evidence budget was sized for. On documents that short
  retrieval has nothing to leave out, so RAG and full context differ only by the prompt
  scaffolding - they are illustrations, not efficient cases.
- The scorer is coarse: 5 distinct values per mode, 61% of candidates on one value under
  TF-IDF. Every correlation reported is a correlation between a 5-level score and a
  0-100 rubric, and `evaluate.py` flags it (`modal_share`).
- The fairness audit found no identity effect separable from its placebo, but that is not
  evidence of fairness: the only identity signal it produced came from one candidate
  (n_effective = 1) and the instrument's noise is 20-60 points. The identity question is
  unresolved, not answered.
- `peak_rss_mb` is process-wide, so RAG-vs-full memory within one run is not separable;
  use `evaluate.py --variant` for a clean per-variant number.
- Latency/tokens/s are from one CPU machine with 4 threads and will not transfer.
  `avg_tok_per_s` here is completion tokens divided by the **whole** `generate()` call
  (prompt prefill included), so it is an end-to-end figure and is lower than the
  decode-only numbers llama.cpp reports; `avg_latency_s` is the number to compare against.
  `avg_tok_per_s` is NaN for the retrieval-only rows by design (a microsecond "generation"
  is not a throughput measurement).

## Fairness audit

`bias_audit.py` holds qualifications fixed and changes one irrelevant detail at a time
(name, university, employment gap, gender/age/marital status) plus a **placebo** line that
carries no identity, then reports the score change against the baseline. It prints both the
magnitude table and a **signed** per-variant table, because a model that docks 5 points for
one name and adds 5 for another looks neutral on absolute change and is not. Run it with
`--embedder` pinned to whatever the study used, and keep `--model "Retrieval-only (no LLM)"`
as the control: the control is what separates "retrieval moved" from "the model moved".
The results section above records what this design did and did not establish.

## Notes

- Scores support, not replace, human decisions.
- The retrieval-only fallback is a ranking baseline, not a calibrated scorer: its MAE
  against 0-100 labels stays large by construction.
- `tests/test_pipeline.py` holds 25 tests and covers the mechanisms the results sections
  argues from - the evidence budget, the relevance floor rejecting spurious passages, a
  rejected requirement never being reported as matched, the loud embedder fallback guard,
  the two floors being on different scales, and the degeneracy metrics that stop a
  near-constant scorer being quoted as a correlation. It runs offline in a couple of
  seconds: TF-IDF only, no GGUF, no GPU, no network.
