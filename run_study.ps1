<#
    run_study.ps1 - reproduce the study in README.md, cheapest checks first.

    .\run_study.ps1 -DryRun          # print the plan, run nothing
    .\run_study.ps1                  # fast stages only: tests, checker, sweeps, baseline
    .\run_study.ps1 -WithModel       # + the Q4_K_M evaluation (~20 min CPU per run)
    .\run_study.ps1 -WithModel -Repeat 2   # ...twice, so the headline is a pair
    .\run_study.ps1 -WithModel -WithDense  # + the dense-embedder run (~20 min CPU)
    .\run_study.ps1 -WithModel -WithDense -Repeat 2  # the committed artifact set: 2 tfidf + 2 dense
    .\run_study.ps1 -WithModel -WithBias   # + the fairness audit (~26 min CPU, 114 screenings)

    CPU cost is the reason for the switches: this machine has no GPU, N_THREADS=4, and each
    model stage is 10-25 minutes of sustained load. The fast stages are what catch a broken
    corpus or a mis-tuned floor, and they cost seconds - run them every time.
    Every stage writes to -OutPrefix so a reproduction never overwrites a committed artifact.
    Tee-Object writes UTF-16LE on Windows PowerShell 5.1, so these logs are not byte-diffable
    against the committed UTF-8 ones: compare with
    `diff <(iconv -f UTF-16LE -t UTF-8 repro_X.log) X.log`, or run under pwsh 7.
    Verified that way: the three sweep grids and the fallback per-resume rows from
    `.\run_study.ps1` reproduce the committed `sweep_*_v2.log` / `results_v2_fallback.csv`
    exactly.
#>
[CmdletBinding()]
param(
    [string]$OutPrefix = "repro",
    [switch]$WithModel,
    [switch]$WithDense,
    [switch]$WithBias,
    [int]$Repeat = 1,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "no .venv at $PSScriptRoot - see README Setup first" }

function Step($label, $block, $cost) {
    Write-Host ("`n=== {0}  ({1})" -f $label, $cost) -ForegroundColor Cyan
    if ($DryRun) { return }
    & $block
    if ($LASTEXITCODE -ne 0) { throw "{0} exited {1}" -f $label, $LASTEXITCODE }
}

Step "tests (25, offline, no model)" { & $py tests/test_pipeline.py } "seconds"

# Labels/files pairing, experience ordering, and whether the generator in this checkout
# still reproduces data/resumes. Exit code 1 means the corpus the CSVs were measured on
# is not the corpus in the repo - stop and decide before spending any CPU.
Step "dataset integrity + drift" { & $py data/check_dataset.py } "seconds"

# The floor is embedder-specific and must be swept on whatever corpus is present now.
Step "floor sweep, tfidf" {
    & $py sweep_floor.py --embedder tfidf --min-sim 0.08 |
        Tee-Object "$OutPrefix`_sweep_tfidf.log"
} "seconds"
Step "floor sweep, dense" {
    & $py sweep_floor.py --embedder dense --min-sim 0.50 |
        Tee-Object "$OutPrefix`_sweep_dense.log"
    # The coarse grid only locates the plateau; the peak decision needs 0.030 resolution,
    # because the coarse argmax sits one step away from the configured floor.
    & $py sweep_floor.py --embedder dense --min-sim 0.30 |
        Tee-Object "$OutPrefix`_sweep_dense_fine.log"
} "seconds (loads torch)"

Step "prompt inflation (pre-dedup vs current, no LLM)" {
    & $py prompt_inflation.py | Tee-Object "$OutPrefix`_prompt_inflation.log"
} "seconds"

# Retrieval-only baseline: the whole pipeline without a GGUF. Its rho is floor-insensitive
# below SIM_ANCHORS["tfidf"][0], so it measures retrieval ordering, not the floor.
Step "no-LLM baseline" {
    & $py evaluate.py --embedder tfidf --fallback --variant "Retrieval-only (no LLM)" `
        --out "$OutPrefix`_fallback.csv" --dump "$OutPrefix`_fallback_per_resume.csv" 2>&1 |
        Tee-Object "$OutPrefix`_fallback.log"
} "~1 min, no model load"

if (-not $WithModel -and ($WithDense -or $WithBias)) {
    # -WithDense / -WithBias without -WithModel would otherwise fall off the `return` below
    # and report success having measured nothing.
    throw "-WithDense and -WithBias need -WithModel (they run the GGUF); passing them alone " +
          "would skip every model stage silently."
}
if (-not $WithModel) {
    Write-Host "`nmodel stages skipped - add -WithModel for the Q4_K_M evaluation." -ForegroundColor Yellow
    return
}

$gguf = "models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
if (-not (Test-Path $gguf)) { throw "$gguf missing - evaluate.py would silently fall back" }
for ($i = 1; $i -le $Repeat; $i++) {
    $tag = if ($Repeat -gt 1) { "$i" } else { "" }
    # --embedder pins the retriever so a missing torch cannot pass a TF-IDF run off as the
    # dense one; --floor is echoed into the log because the logs otherwise record only the
    # embedder. Two runs with the same arguments are the difference between a measurement
    # and a reproduction.
    Step "Q4_K_M evaluation #$i (tfidf, floor 0.024, rag + full over 33 resumes)" {
        & $py evaluate.py --embedder tfidf --floor 0.024 `
            --out "$OutPrefix`_results$tag.csv" --dump "$OutPrefix`_per_resume$tag.csv" 2>&1 |
            Tee-Object "$OutPrefix`_eval$tag.log"
    } "~20 min CPU"
}

if ($WithDense) {
    # Repeatable for the same reason as tfidf: the committed dense row is a pair
    # (results_v2_dense.csv + results_v2_dense_repeat.csv), and a second process is what shows
    # the 0.670 is not one lucky run. It buys reproducibility, not precision - temperature is
    # 0, so the quality columns come back identical and only the timings move.
    for ($i = 1; $i -le $Repeat; $i++) {
        $tag = if ($Repeat -gt 1) { "$i" } else { "" }
        Step "Q4_K_M evaluation #$i (dense, floor 0.24, rag + full over 33 resumes)" {
            & $py evaluate.py --embedder dense --floor 0.24 `
                --out "$OutPrefix`_results_dense$tag.csv" --dump "$OutPrefix`_per_resume_dense$tag.csv" 2>&1 |
                Tee-Object "$OutPrefix`_eval_dense$tag.log"
        } "~20 min CPU"
    }
}

if ($WithBias) {
    Step "fairness audit (6 resumes x 19 variants = 114 screenings)" {
        & $py bias_audit.py --limit 6 --embedder tfidf --floor 0.024 `
            --model "Q4_K_M (4-bit)" --out "$OutPrefix`_bias_q4.csv" 2>&1 |
            Tee-Object "$OutPrefix`_bias_q4.log"
    } "~26 min CPU"
}

Write-Host "`nfinished: artifacts written with prefix '$OutPrefix'." -ForegroundColor Green
