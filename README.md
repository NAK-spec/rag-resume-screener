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
budget that scales with the document (`rag/screen.py:73`).

That failure is now measured rather than remembered (`prompt_inflation.py`,
`prompt_inflation.log` - retrieval only, no LLM, seconds to run). Rebuilding the historical
prompt exactly as it stood at commit `9d2e10a` (`TOP_K=4`, no floor, no dedup, no budget) on
the current 33 resumes: the evidence block alone was **1,517 words pasted against a 222-word
document - 6.8x the resume**, and the whole prompt averaged 1,671 words versus the 304-word
full-context baseline it was supposed to compress. The current path carries **69 evidence
words (0.37x the document)** and 172 prompt words total. An earlier draft of this file quoted
"304 -> 416 words" for the same bug; that was measured live during the fix, never written to
an artifact, and re-deriving it lands 4x higher, so it has been replaced by the reproducible
figures above.

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
(`data/check_dataset.log`) is **0 FAIL / 2 WARN** with no generator drift; the 2 WARNs are
the 39- and 56-word hand-written CVs, which are too short for retrieval to have anything to
drop. The 12 reversed experience ranges it used to report are fixed and re-measured - see
Limitations.

## Run

Use the project venv for every command below. On this machine plain `python` resolves to a
system 3.13 whose pandas install is broken (`ModuleNotFoundError:
pandas._libs.pandas_parser`), so the analysis scripts only run under
`.venv\Scripts\python.exe` - and a partially-importing pandas is exactly the kind of failure
that would produce a half-populated results CSV.

```
streamlit run app.py        # the app
python evaluate.py --embedder tfidf --out results_v2.csv --dump per_resume_v2.csv
python sweep_floor.py --embedder tfidf --min-sim 0.08   # what each floor costs (no LLM)
python prompt_inflation.py   # the pre-dedup prompt size, measured (no LLM)
python bias_audit.py --limit 6 --embedder tfidf --model "Q4_K_M (4-bit)" --out bias_q4.csv
python data/check_dataset.py   # dataset integrity + generator drift (no model needed)
python tests/test_pipeline.py    # pytest is not installed; the file is self-running
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
run as four separate LLM processes over the same 33 corrected resumes - twice with
**TF-IDF retrieval** and twice with the dense **`all-MiniLM-L6-v2`** embedder (torch
2.15.0+cpu), each pair the second run being an independent reproduction. Each embedder uses
its own similarity floor,
swept with `sweep_floor.py` (`rag/config.py`: `{"tfidf": 0.024, "dense": 0.24}`), and
the runs are pinned with `--embedder` so a silent fallback cannot pass one run off as the
other. n = 33 resumes, 111 chunks, 6 requirements.

Artifacts, after the 2026-11-03 dataset correction (see Limitations - 12 `Experience:` date
lines were repaired and the study re-measured): the headline is
**`results_v2.csv` + `results_v2_repeat.csv`** and `per_resume_v2*.csv` - two separate
processes whose 33 RAG scores and 33 matched-lists agree exactly, so the 0.753 is now a
reproduced number rather than a single run - with **`results_v2_dense.csv`** as the dense
counterpart on the same corrected corpus (now also paired with
`results_v2_dense_repeat.csv`, see below). The pre-correction equivalents
(`results.csv` / `per_resume.csv`, byte-identical to `results_floor024.csv`, i.e. **one**
process) are kept because they are what produced the earlier draft of these tables and they
let the correction be measured rather than assumed: they differ from `results_v2.csv` by
zero RAG scores and one `matched` count. Also kept: `results_dense.csv` (pre-correction
dense run, superseded by `results_v2_dense.csv`), `results_floor024.csv` +
`results_floor030.csv` (the floor A/B),
`results_tfidf_q4.csv` (pre-floor ablation, kept for that comparison),
`results_fallback*.csv` (no-LLM baseline), `sweep_*.log` (the floor grids, `*_v2` for the
corrected corpus), `bias_q4.csv` / `bias_q4.log`, `bias_retrieval_only_control.csv`.
`ARTIFACTS.md` is the full index - which file is which embedder and floor, which are
duplicates, which are superseded, and which should not be quoted at all.
Runs made from here on print both `embedder:` and `sim_floor:` in their first log line; the
older artifacts print `embedder:` only, and their floor is whatever `rag/config.py` held at
the time (0.024 / 0.030 for the two named above).
(`bias_results.csv` is an earlier no-LLM prototype run; it predates every fix below and
should not be quoted.)

Every headline ranking number now comes from a *pair* of independent processes:
`results_v2.csv` / `results_v2_repeat.csv` (TF-IDF) and `results_v2_dense.csv` /
`results_v2_dense_repeat.csv` (dense). Each pair agrees on all 66 per-resume rows - score,
coverage, matched/missing lists, passage count, prompt and output length - so 0.753 and 0.670
are reproduced numbers rather than single runs. The dense repeat's *timing* columns are
unusable and excluded from every table: this laptop suspended partway through that run, and
one full-mode generation is stamped 421,575 s (4.9 days) because the clock kept running while
the CPU did not. See Determinism and reproducibility.

Reproduce with `.\run_study.ps1` (fast stages only), `.\run_study.ps1 -WithModel -Repeat 2`
for the evaluation pair, `-WithModel -WithDense` for the dense run, `-WithModel -WithBias`
for the audit. `-WithDense`/`-WithBias` without `-WithModel` is an error rather than a silent
no-op. Each LLM run is ~15-25 min of CPU generation on this machine; keep them serial, they
contend for the same 4 threads and latency is part of what is being measured - but note that
latency is only comparable *within* one run, never between two of them. The runner's fast
stages were executed end to end and reproduce the committed artifacts exactly - all three
sweep grids and the fallback per-resume rows match - so `.\run_study.ps1` is a verified path,
not an untested script. The equivalent commands:

```
python evaluate.py --embedder tfidf --out results_v2.csv --dump per_resume_v2.csv
python evaluate.py --embedder tfidf --out results_v2_repeat.csv --dump per_resume_v2_repeat.csv
python evaluate.py --embedder dense --out results_v2_dense.csv --dump per_resume_v2_dense.csv
python evaluate.py --embedder dense --out results_v2_dense_repeat.csv --dump per_resume_v2_dense_repeat.csv
python evaluate.py --fallback --embedder tfidf --variant "Retrieval-only (no LLM)" --out results_v2_fallback.csv
python sweep_floor.py --embedder tfidf --min-sim 0.08   # -> sweep_tfidf_v2.log, no LLM
python sweep_floor.py --embedder dense --min-sim 0.50   # -> sweep_dense_v2.log, no LLM
python sweep_floor.py --embedder dense --min-sim 0.30   # -> sweep_dense_fine_v2.log, the peak grid
python bias_audit.py --limit 6 --embedder tfidf --floor 0.024 --model "Q4_K_M (4-bit)" --out bias_q4.csv
python bias_audit.py --limit 6 --embedder tfidf --out bias_retrieval_only_control.csv
```

### RAG vs full context, same model

TF-IDF retrieval, floor 0.024, on the corrected corpus (`results_v2.csv`, reproduced
reproduced row-for-row on every quality column by `results_v2_repeat.csv`):

| mode | Spearman | Kendall | MAE | within ±10 | latency | tok/s | prompt words |
|---|---|---|---|---|---|---|---|
| rag | **0.753** | 0.629 | 23.9 | 0.333 | **13.3 s** | 10.0 | **172** |
| full | *undefined* | *undefined* | 31.2 | 0.273 | 20.1 s | 9.5 | 304 |

The full-context row has no correlation because it has no rank: all 33 candidates score 85.
`evaluate.py` returns NaN rather than a tie-swollen number, which is the honest reading of
that baseline - not "full context ranks badly" but "full context does not rank". Its MAE is
still quotable (31.2 vs 23.9).

- **Efficiency claim holds**: the RAG prompt is 43.4% shorter than the full-resume prompt
  and is shorter on 30 of 33 resumes (median 177 vs 297 words). Per-candidate latency
  drops 33.6% and throughput rises 5.3% (10.0 vs 9.5 tok/s), at the same model load time
  (1.71 s) and effectively the same memory (2,079 vs 2,119 MB peak RSS).
- **The efficiency half is embedder-independent.** The dense run reproduces it on the same
  corpus: 304 -> 178 words (-41.5%), RAG shorter on the same 30 of 33 resumes, the same two
  prompt medians (177 vs 297), and one full-context score of 85 for all 33 candidates in
  both embedders - as it must be, since `full` mode never reads retrieval. What the embedder
  changes is the *quality* column, not the compression (see the TF-IDF-vs-dense section).
  The same figures come back unchanged in the second dense process, so the -41.5% is
  measured four times over: twice per embedder.
- **Where it does not hold**: the 3 resumes where RAG is longer are exactly the 3 under
  90 words (`per_resume_v2.csv`: 39, 56 and 85 words), each by a constant 21 words - that is
  the requirement-to-passage mapping, and when a resume already fits in one chunk there is
  nothing left for retrieval to compress. Below ~100 words the baseline is the cheaper
  prompt, so the honest claim is "RAG compresses any resume with more than one chunk",
  not "RAG always compresses".
- **Quality does not pay for that saving - it comes with it.** RAG separates candidates in
  ways that track the rubric (rho 0.753), while full context barely varies its answer at
  all (see the degeneracy note below - on the corrected corpus all 33 candidates get the
  same number).
  Restricting the model to the passages that actually answer each requirement beats letting
  it read everything.
- MAE of ~24 points is still poor in absolute terms, and the +/-10-point hit rate is only
  0.333 (labels span 10-100, mean 58.5): the model orders candidates much better than it
  scores them, so this is a *shortlisting* tool, not a scoring one. Note also that the
  hit rate did **not** improve with the floor (0.364 at floor 0.03 -> 0.333 at 0.024)
  while rho and MAE did, which is what a ranking-shaped fix looks like.

### TF-IDF vs dense retrieval, same model and same budget

`all-MiniLM-L6-v2` at its own swept floor (0.24) against TF-IDF at 0.024, same GGUF, same
33 corrected resumes, same 6 requirements, `temperature=0` (`results_v2.csv` vs
`results_v2_dense.csv` - and neither row is a single process any more:
`results_v2_repeat.csv` and `results_v2_dense_repeat.csv` each reproduce their partner on all
66 per-resume rows):

| embedder | Spearman | Kendall | MAE | within ±10 | signed score offset | mean abs gap to own coverage | prompt words | peak RSS |
|---|---|---|---|---|---|---|---|---|
| tfidf (floor 0.024) | **0.753** | 0.629 | 23.9 | 0.333 | **-22.0** | 30.3 | **172** | **2,079 MB** |
| dense (floor 0.24) | 0.670 | 0.576 | **16.6** | **0.394** | +6.1 | **9.9** | 178 | 2,463 MB |

Latency is deliberately **not** a column here. The two runs are separate processes on a
shared CPU and are not comparable: dense is 12.6 s per candidate and tfidf 13.3 s this time,
whereas the pre-correction pair read 16.2 s and 13.9 s - every one of the 33 dense candidates
got 1.1-9.8 s faster on the corrected corpus while its output length barely moved (mean 90.8
-> 88.1 words), so that swing is host state, not content. The comparison that *is* valid is
within a process, and it repeats for both embedders: RAG 13.3 vs full 20.1 s (tfidf), RAG
12.6 vs full 20.0 s (dense) - a 34-37% latency saving either way. Peak RSS is comparable
across runs because it is dominated by the weights, and there the dense embedder does cost
what was claimed: 2,463 vs 2,079 MB (+18.5%).

- **The ranking difference is not resolvable at n = 33.** 0.753 has a 95% Fisher-z interval
  of [0.553, 0.871] and 0.670 of [0.424, 0.824]; they overlap almost completely, so the
  defensible sentence is "TF-IDF + floor ranked at least as well as dense here", not
  "TF-IDF is better". That is a slightly awkward result for a project whose title says
  *retrieval-augmented*, because the cheap, dependency-free embedder won the headline metric.
  Both figures also survived the corpus correction essentially untouched (0.668 -> 0.670,
  16.58 -> 16.55 MAE), so the comparison is not an artifact of the defective text.
- **What is not noise is the scale.** Dense moves the model from systematically harsh
  (mean score 36.5 against labels averaging 58.5) to near-centred (64.5), which is where
  the MAE 23.9 -> 16.6 and hit-rate 0.333 -> 0.394 gains come from - not from a better
  ordering. The two score vectors disagree substantially: Spearman between the TF-IDF and
  dense RAG scores is only **0.462**, 15 of 33 candidates sit 30+ points apart (mean |delta|
  between the two runs is **29.5 points**), and of the 23 candidates TF-IDF puts below 40,
  dense puts 12 on 80 and 3 on 85. For scale: swapping embedders moves a candidate's score
  about twice as much as the entire fairness audit's identity edits did (mean 13.4 points),
  so the choice of embedder is this pipeline's largest measured lever - larger than anything
  the prompt does.
- **Dense retrieval is internally more consistent.** The model's number vs the coverage
  implied by its own `matched` list differs by 30.3 MAE under TF-IDF and 9.9 under dense,
  and only 4 of 33 candidates diverge by more than 30 points (15 under TF-IDF). The dense
  floor is gentler on evidence (20.7% of requirements come back empty vs 51.5% for TF-IDF,
  1.09 vs 1.03 surviving passages per candidate), so the model is handed a set it can
  actually stand behind - and pays for it in memory, not in ranking quality.
- Both runs are single-passage retrieval, and both flag degenerate: dense mode puts 19 of 33
  candidates on 80 (modal share 0.576), TF-IDF puts 20 of 33 on 20 (0.606). Same ceiling,
  different ceiling value.
- The three hand-written CVs make the trade concrete (human label / tfidf / dense): weak
  10 / 0 / 0, medium 50 / 20 / 80, strong 90 / 85 / 80. TF-IDF gets the extremes right and
  flattens the middle; dense gets the middle wrong and clusters near the top.


### Both modes score degenerately - read the rho as band agreement

Before comparing the two rho values, look at what the model actually emits (`per_resume_v2.csv`):

| run | distinct scores | distribution | modal share |
|---|---|---|---|
| rag, tfidf | 5 of 33 candidates' worth of resolution | 20 (x20), 80 (x7), 85 (x2), 0 (x3), 75 (x1) | 0.61 |
| rag, dense | 4 | 80 (x19), 85 (x6), 20 (x5), 0 (x3) | 0.58 |
| full (either embedder) | **1** | 85 (x33) | **1.00** |

In full-context mode the corrected corpus leaves the model with a single answer for all 33
candidates - it cannot rank at all, and `evaluate.py` reports Spearman and Kendall as
`undefined` rather than a number. (Pre-correction it emitted two values and its 0.262 rho was
carried entirely by the one candidate, `r12_weak.txt`, that scored 60; fixing a reversed date
range in that CV removed the last trace of a rank. See "Run-to-run determinism".) So the
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


`results_v2.csv` reports each variant twice: `spearman` is the LLM's own 0-100 guess,
`cov_spearman` is a deterministic score computed from what retrieval actually found
(weighted count of requirements with surviving evidence, `screen_resume` in
`rag/screen.py`). That second column exists because the two disagree more often than they
agree: in `per_resume_v2.csv`, the model's guess and its own matched list differ by more than
30 points on **15 of 33** resumes (max 63, they coincide exactly on 3) - and every single
one of those 15 has the model scoring *below* what its own citations support. The dense run
diverges on only 4 of 33 (mean gap to coverage 9.9 MAE), so the self-contradiction is a
property of what retrieval hands the model, not of the model alone.

| mode | LLM score rho / MAE | coverage rho / MAE |
|---|---|---|
| rag, tfidf | 0.753 / 23.9 | 0.554 / 22.3 |
| rag, dense | 0.670 / 16.6 | 0.535 / 22.6 |
| full (either embedder) | undefined (one score) / 31.2 | -0.267 / 57.3 |

So the LLM wins on ranking and roughly ties on absolute error - it is a better sorter than
a better scorer. Note that this comparison is **not** an independent validation of
coverage: `score_coverage` counts evidenced requirements, and the synthetic labels were
built from the same rubric, so coverage agreeing with the labels is partly circular. What
coverage usefully exposes is the scale problem, which is a constant offset per mode:

| mode | mean human | mean LLM score | mean signed gap | error quartiles | MAE raw -> after removing the offset |
|---|---|---|---|---|---|
| rag, tfidf | 58.5 | 36.5 | -22.0 | -47 / -14 / -2 | 23.9 -> 19.7 |
| rag, dense | 58.5 | 64.5 | +6.1 | -10 / -2 / +13 | 16.6 -> 16.7 |
| full | 58.5 | 85.0 | +26.5 | -4 / +18 / +63 | 31.2 -> 26.0 |

With retrieved evidence under TF-IDF the model is systematically harsh (25 of 33 candidates
scored below what their own citations support, 5 above); with the whole resume it is
systematically lenient. The offset is not stable across prompt strategy - it flips sign and
is worth ~26 points of MAE - so neither mode is calibrated, which is exactly why the tool
should report ranks and citations rather than a number to hand to a candidate. The
embedders separate *why* the error is what it is: TF-IDF's error distribution is one-sided
(all quartiles at or below zero), so most of its 23.9 MAE is a constant harshness that
recalibration would remove (19.7 once the -22.0 offset is centred), while dense's straddles
zero and centring it makes the error very slightly *worse* (16.6 -> 16.7). That is the
sharpest statement of the embedder trade in this study: dense's 16.6 is not a scale bug
waiting for a recalibration, it is irreducible per-candidate disagreement, so its apparent
MAE advantage over TF-IDF cannot be closed by a post-hoc offset the way TF-IDF's can. The
other thing coverage exposes is in the first table: in full mode the coverage heuristic
correlates *negatively* with the labels (-0.267), because when every requirement can be
matched against the entire document, "has a passage about X" stops discriminating. Read
together with the floor ablation below: retrieval does most of the ranking work, and the
LLM's judgement is what lifts evidenced-coverage agreement (0.554) to 0.753.


### Run-to-run determinism

Greedy decoding (`temperature=0`) makes the scoring path reproducible, and the headline is
now backed by the reproduction rather than by an argument:

- **Two independent processes at floor 0.024 on the corrected corpus**
  (`results_v2.csv`, `results_v2_repeat.csv`) agree on all 33 RAG scores, all 33
  full-context scores and every `matched` list, so every quality figure repeats exactly:
  rho 0.753 / undefined, tau 0.629, MAE 23.91 / 31.24, within-10 0.333 / 0.273,
  cov-rho 0.554 / −0.267, 172 / 304 prompt words, rag peak RSS 2,079 MB. Only the timings
  moved: 13.33 -> 12.85 s (rag), 20.06 -> 19.20 s (full), i.e. ~4% CPU noise, with tok/s
  10.0 -> 10.3 and model load 1.71 -> 1.85 s.
- **Earlier, two independent processes at floor 0.03** (`results_q4_tfidf.csv` and
  `results_floor030.csv`) gave the same byte-identical quality figures on the
  pre-correction corpus (rho 0.560 / 0.262, MAE 27.76 / 30.48), so determinism holds across
  both floors, not only at the one being quoted.
- **Full-context mode reproduces across embedders**, as it must: it never touches retrieval.
  On the corrected corpus, `per_resume_v2.csv` and `per_resume_v2_dense.csv` give the same
  score (85) for all 33 candidates in `full` mode, and the two runs' `full` rows agree on
  every quality figure - MAE 31.24, within-10 0.273, cov-rho -0.267, cov-MAE 57.27, 304 prompt
  words. Pre-correction the same held (`per_resume.csv` vs `per_resume_dense.csv`).
- **The dense side is now a pair of its own** (`results_v2_dense.csv`,
  `results_v2_dense_repeat.csv`): all 66 rows agree on score, coverage, matched/missing,
  passage count, prompt words and output words, so 0.670 / 0.576 / 16.55 and the
  one-distinct-value `full` row are each measured twice. It also lands where the
  pre-correction dense run landed (0.668 -> 0.670 rho, 16.58 -> 16.55 MAE) after exactly one
  score moved, so the embedder comparison rests on reproduced numbers on both sides.
- **A repeat run can still be unusable, and this one nearly was.** The dense repeat's
  per-candidate `latency_s` is meaningless: the laptop suspended partway through, and one
  `full` generation is stamped 421,575 s - 4.9 days of wall clock, most of it asleep - which
  drags that mode's mean to 12,813 s and the RAG mean to 21.2 s against its own 13.7 s median
  (first run: 12.6 s mean, 12.5 s median). Greedy decoding kept the quality columns exact
  while the timing columns were destroyed, which is why `ARTIFACTS.md` marks the repeat's
  timing not merely non-comparable but void, and why the quotable dense latencies stay
  12.6 / 20.0 s from the first process. Peak RSS survived untouched (2,462 / 2,479 MB vs
  2,463 / 2,487 MB).

**Latency is only comparable inside one process.** The tfidf pair above moved 13.33 -> 12.85 s
between two runs of the *same* configuration on the *same* corpus, and the dense RAG mean moved
16.19 -> 12.64 s between corpora while all 33 candidates got faster (1.1-9.8 s each) with mean
output length essentially flat (90.8 -> 88.1 words). Nothing about the corrected text makes a
token faster; that is host CPU state. So within-run deltas are quotable (tfidf 13.3 vs 20.1 s,
dense 12.6 vs 20.0 s) and cross-run ones are not (12.6 s dense vs 13.3 s tfidf must not be read
as "dense is quicker"). Peak RSS behaves differently - 2,079 / 2,119 / 2,463 MB is stable enough
across processes to compare, because it is dominated by the loaded weights.

**The corpus correction was measurable, and almost entirely cosmetic.** Regenerating the
dataset changed 12 `Experience:` date strings in 10 files. Re-measuring on it changed **zero of
33 TF-IDF RAG scores**, zero prompt lengths and zero retrieved passages, with one candidate's
`matched` count moving from 5 to 4 (`r13_weak.txt`); under dense the same correction moved
exactly **one of 33** scores, `r13_weak.txt` again, 25 -> 20. The one real effect is the
cautionary tale: in full-context mode the single candidate that had *not* scored 85
pre-correction was `r12_weak.txt` at 60, and `r12_weak.txt` is one of the ten files whose text
changed. After the fix all 33 land on 85 - in both embedders - which is the baseline's only
rank signal, destroyed by editing a two-character date range in one CV. That is the same
knife-edge instability the fairness audit measured, observed from the other direction.

Consequence for the fairness section: because the scoring path is deterministic, the
60-point swings there are *input* sensitivity, not sampling noise - which is exactly why the
placebo control was needed to say anything about what the input was.


### The relevance floor is what made RAG work

`index.search` originally returned top-k chunks no matter how weakly they matched, so every
requirement always came back with "evidence" - including for a hospitality CV that has no
data skills at all. Two artifact-backed views of the same pathology:

- **Structurally** (`sweep_tfidf_v2.log`, first row): at floor 0.000 every one of the 33
  resumes returns evidence for all 6 requirements - weak candidates 6.00, strong candidates
  6.00, separation exactly 0.00, and 0% of requirements reported as unevidenced. Retrieval
  simply cannot express "this candidate has nothing to say about that".
- **At the model** (`per_resume_v2.csv`, `full` mode - which hands the LLM the entire resume
  and is the same failure in prompt form): all three hand-written CVs get **85/100 with the
  same 5 of 6 requirements matched**, so the hospitality CV (label 10) is not merely
  over-scored, it is byte-identical in output to the strong CV (label 90).

| candidate (human label) | rag, floor 0.024 | full context |
|---|---|---|
| weak, hospitality CV (10) | **0**, 0/6 matched, 5 missing | 85, 5/6 matched |
| medium (50) | 20, 4/6 matched | 85, 5/6 matched |
| strong (90) | 85, 5/6 matched | 85, 5/6 matched |

Corpus-wide, the floor moves RAG-mode Spearman from **-0.083** (no floor,
`results_tfidf_q4.csv`) to **0.560** (floor 0.03, `results_floor030.csv`) to **0.753** (floor
0.024, `results_v2.csv`). Negative correlation is worse than random, and it is the *efficiency*
path that caused it: a top-k-per-requirement budget guarantees retrieving noise when a
candidate lacks a skill, and the model faithfully reports the noise as proof. The floor costs
some true evidence (the strong candidate keeps 5 of 6 either way, but the corpus loses half
its evidenced requirements) and no input tokens (median 178 -> 177 words), but it does cost
output time: 11.4 -> 13.3 s per candidate, because a candidate with empty evidence sets makes
the model spend tokens listing what is missing. It is still comfortably cheaper than the 20.1
s full-context baseline, so the honest framing is "the floor buys the ranking and pays ~2 s
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
The two grids below are the exact output of `sweep_tfidf_v2.log` and `sweep_dense_fine_v2.log`
on the corrected corpus, so the chosen value is the row marked `*` in each.

**TF-IDF** (`python sweep_floor.py --embedder tfidf --min-sim 0.08`):

| floor | weak (<40) | medium (40-74) | strong (>=75) | strong-weak | passages/resume | requirements with NO evidence |
|---|---|---|---|---|---|---|
| 0.000 | 6.00 | 6.00 | 6.00 | 0.00 | 1.12 | 0.0% |
| 0.008 | 3.91 | 5.33 | 5.60 | 1.69 | 1.12 | 17.7% |
| 0.016 | 2.00 | 3.67 | 4.50 | 2.50 | 1.06 | 43.9% |
| **0.024** | 1.55 | 3.00 | 4.30 | **2.75** | 1.03 | 51.5% |
| 0.032 | 1.36 | 2.33 | 3.30 | 1.94 | 1.03 | 61.6% |
| 0.040 | 1.27 | 1.75 | 2.30 | 1.03 | 0.94 | 70.7% |
| 0.048 | 1.00 | 1.42 | 1.50 | 0.50 | 0.85 | 78.3% |
| 0.056 | 0.64 | 1.33 | 0.90 | 0.26 | 0.73 | 83.8% |
| 0.064 | 0.45 | 1.25 | 0.80 | 0.35 | 0.67 | 85.9% |
| 0.072 | 0.36 | 1.00 | 0.80 | 0.44 | 0.58 | 87.9% |
| 0.080 | 0.18 | 0.83 | 0.70 | 0.52 | 0.52 | 90.4% |

**Dense, `all-MiniLM-L6-v2`** (`python sweep_floor.py --embedder dense --min-sim 0.30`,
fine grid around the peak; `sweep_dense_v2.log` is the same corpus at 0.050 steps):

| floor | weak | medium | strong | strong-weak | passages/resume | NO evidence |
|---|---|---|---|---|---|---|
| 0.000 | 5.91 | 6.00 | 6.00 | 0.09 | 1.12 | 0.5% |
| 0.030 | 5.91 | 6.00 | 6.00 | 0.09 | 1.12 | 0.5% |
| 0.060 | 5.91 | 6.00 | 6.00 | 0.09 | 1.12 | 0.5% |
| 0.090 | 5.91 | 6.00 | 6.00 | 0.09 | 1.12 | 0.5% |
| 0.120 | 5.82 | 5.92 | 6.00 | 0.18 | 1.12 | 1.5% |
| 0.150 | 5.55 | 5.92 | 6.00 | 0.45 | 1.12 | 3.0% |
| 0.180 | 4.64 | 5.92 | 5.90 | 1.26 | 1.09 | 8.6% |
| 0.210 | 4.00 | 5.75 | 5.70 | 1.70 | 1.09 | 14.1% |
| **0.240** | 3.18 | 5.67 | 5.40 | 2.22 | 1.09 | 20.7% |
| 0.270 | 2.36 | 4.33 | 4.70 | 2.34 | 1.09 | 36.9% |
| 0.300 | 1.91 | 3.67 | 3.90 | 1.99 | 1.06 | 47.5% |

Four things these grids settle:

- **At floor 0.000 the tier separation is 0.00 (TF-IDF) / 0.09 (dense)** - top-k never says
  "nothing", so the signal "this candidate lacks the skill" is structurally unavailable to
  the model. This is why the un-floored run correlated *negatively*.
- **The scales are ~10x apart and not interchangeable.** The dense plateau runs from 0.24 to
  about 0.27, roughly ten times the TF-IDF value; running the TF-IDF floor against dense
  embeddings would filter essentially nothing (gap 0.09), and running 0.24 against TF-IDF
  would leave almost no candidate with evidence at all. `SIM_FLOOR` is a dict for exactly
  this reason.
- **Past the peak the filter turns on the good candidates.** By 0.048 TF-IDF gives a strong
  candidate 1.50 evidenced requirements against a weak candidate's 1.00, and in the 0.050-step
  run (`sweep_dense_v2.log`) dense gives 1.70 at 0.400 and 0.20 at 0.500 against weak's 0.73
  and 0.18 - separation collapses to noise.
- **Dense separates tiers less sharply at its own peak** (2.2-2.3 vs 2.75 requirements) while
  leaving far fewer requirements empty (20.7% vs 51.5%). That is the retrieval-side
  explanation for the eval result above: TF-IDF + floor ranks better because it is willing
  to say "no evidence" about half the requirements, dense ranks worse and scores closer
  because it keeps asserting evidence.

**Why the dense floor stays at 0.240 when 0.250 scores a wider gap.** On the corrected corpus
the coarse grid's argmax is 0.250 (gap 2.40) rather than 0.240 (gap 2.22) - but that 0.18
difference is *two* requirement-match decisions out of the 198 pairs the grid counts (weak
tier: 11 resumes x 3.18 = 35 evidenced matches at 0.240 vs 33 at 0.250; strong tier is
unchanged at 54 either way). Re-tuning to a 2-pair swing would be fitting the threshold to
the noise in one 33-resume sample, and it would invalidate the `SIM_FLOOR` values the
committed `results*.csv` were produced with. 0.240 is kept, and the honest description of the
dense peak is "flat from 0.24 to 0.27", not "at 0.25".

The cost is worth stating plainly: dedup plus the floor leaves **about one passage** per
candidate (`per_resume_v2.csv`: mean 1.03, median 1.0, and 2 candidates with none at all;
dense, `per_resume_v2_dense.csv`: mean 1.09, median 1.0, 1 candidate with none).
This is single-passage retrieval, not multi-passage aggregation - the 43% prompt
compression is partly a consequence of that narrowness, and a resume that proves one skill
across three sections still shows the model only one of them. The 2 zero-passage candidates
are also the interesting ones: the pipeline returns them with an empty evidence set, the
model gives them 0/100, and that is the behaviour a floor is supposed to produce.


### No-LLM baseline

Retrieval-only scoring (no generation at all) reaches rho **0.306** / tau 0.249 / MAE 54.5 /
within-10 0.061 on the same 33 resumes (`results_v2_fallback.csv`, and to the last decimal on
the pre-correction corpus in `results_fallback_floor024.csv`), so the quantized
LLM contributes real ranking value (0.306 -> 0.753 with TF-IDF, -> 0.670 with dense).

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
The audit needed no re-run after the dataset correction, and that is checkable rather than
asserted: `--limit 6` takes the first six resumes in sorted order (`medium_candidate.txt`,
`r01_weak.txt`-`r05_strong.txt`) and none of them is among the ten files whose text changed.

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
  on the corrected corpus, and **each headline number is now two independent processes that
  agree exactly**: `results_v2` / `results_v2_repeat` (TF-IDF) and `results_v2_dense` /
  `results_v2_dense_repeat` (dense), 66 of 66 rows identical on every quality column. What
  that proves is *reproducibility* - no process-level nondeterminism in the retrieval, prompt
  build or greedy decode - not precision: `temperature=0` makes a second run of the same
  input a foregone conclusion, so the repeats add no sampling information and leave every
  Fisher-z interval, and the 0.753-vs-0.670 ordering, exactly as wide as they were with one
  process. The ordering still should not be claimed in either direction.
- Labels are synthetic and rubric-derived, so agreement measures whether the pipeline
  reproduces a stated rubric, not real hiring judgement. n = 33 is small; rho 0.753 has a
  95% Fisher-z interval of [0.553, 0.871] and the dense run's 0.670 of [0.424, 0.824].
  Full context contributes **no interval at all** on the corrected corpus: it scored all 33
  candidates 85, so Spearman is undefined (its pre-correction 0.262 came from a single
  dissenting candidate). The RAG-vs-full comparison therefore rests on the degenerate-versus
  ordered contrast plus MAE (23.9 vs 31.2), not on two overlapping rho intervals; the
  embedder comparison does not survive its own interval either way.
- The dataset's text defect is **fixed and re-measured, and the fix was cosmetic** - which is
  now a measurement rather than an assumption. 10 of the 30 generated resumes had experience
  ranges running backwards (`Experience: BI Analyst (fintech, 2023-2022)`) because the
  graduation-year clamp could return a start later than its own end; `role_years()` in
  `data/generate_dataset.py` clamps to the cursor as well, `tests/test_pipeline.py` pins it,
  and `python data/check_dataset.py` reports 0 FAIL / 2 WARN with no generator drift.
  Re-running the study on the corrected corpus changed 0 of 33 TF-IDF RAG scores, 0 prompt
  lengths and 0 retrieved passages (1 of 33 under dense), and the only effects were one
  candidate's `matched` count (5 -> 4)
  and, in full-context mode, the collapse from 2 distinct scores to 1, because the one CV
  that had not scored 85 was itself one of the ten edited files. Old and new artifacts are
  both committed so the comparison stays checkable (`results.csv` vs `results_v2.csv`).
- A second cosmetic finding from the same checker: the two hand-written CVs are 39 and 56
  words, under the 60-word band the evidence budget was sized for. On documents that short
  retrieval has nothing to leave out, so RAG and full context differ only by the prompt
  scaffolding - they are illustrations, not efficient cases.
- The scorer is coarse: 5 distinct values across 33 candidates in RAG mode (61% on one
  value), and **1** in full-context mode on the corrected corpus. Every correlation reported
  is a correlation between a 5-level score and a 0-100 rubric, and `evaluate.py` flags it
  (`n_distinct_scores`, `modal_share`, and NaN instead of a tie-swollen rho).
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
