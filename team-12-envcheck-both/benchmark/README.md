# EnvCheck Benchmark

A small benchmark for evaluating EnvPilot on Python environments where a target library has a known breaking change. Each case has a single `bad_env` (the problem environment), a `canonical_solution` that crashes in `bad_env` due to API removal/rename, and a hand-checked `correct_solution` that does the same task using an alternative API and passes the same test in `bad_env`.

EnvPilot's job: given the task description and `bad_env`, generate code that passes `test` in `bad_env`.

## Files

| File | Purpose |
|---|---|
| `build_candidates.py` | Builds `candidates.json` from BigCodeBench v0.1.4 + `manual_cases.py` + `bcb_corrections.py`. Re-runnable. |
| `manual_cases.py` | 8 hand-written cases in BigCodeBench dict format (numpy 2.0 / pandas 2.0 / Pillow 10 / flask 2.3 removals). Each has its own `correct_solution`. |
| `bcb_corrections.py` | `task_id → (old_substr, new_substr)` mapping that derives `correct_solution` from each BCB canonical via a single string replacement. |
| `runner_utils.py` | Shared utilities: build cached `uv` venvs (Python version aware), compose code+test scripts, run with `MPLBACKEND=Agg` + timeout. |
| `verify_ground_truth.py` | Verifies each candidate in its `bad_env`: runs `canonical+test` (expects FAIL with `error_type`), runs `correct+test` (expects PASS). Both must hold for `verified=True`. |
| `candidates.json` | **The benchmark itself.** 19 verified cases with full fields. Generated but committed for convenience. |
| `bigcodebench_pool.json` | Early 50-task random sample from BCB filtered to target libs. Kept for reference; not used by current build pipeline. |
| `pool_stats.json` | Library distribution of the original BCB filtered pool. Kept for reference. |
| `verification_report.json` | (gitignored) Output of last `verify_ground_truth.py` run. |
| `envs/` | (gitignored) Cached venvs keyed by hash of (Python version, pip spec). |

## Usage

```bash
# 1. Re-generate candidates.json from sources (BCB + manual_cases.py + bcb_corrections.py)
uv run --with datasets python benchmark/build_candidates.py

# 2. Verify ground truth: build bad_env per case, run canonical (expect FAIL)
#    and correct_solution (expect PASS) in the same env
python benchmark/verify_ground_truth.py             # all cases
python benchmark/verify_ground_truth.py --case manual_006   # one case
python benchmark/verify_ground_truth.py --first 5           # first 5
python benchmark/verify_ground_truth.py --update            # write `verified` field back into candidates.json

# Force rebuild venvs (otherwise cached under benchmark/envs/<hash>/)
python benchmark/verify_ground_truth.py --rebuild
```

The first `verify_ground_truth.py` run takes 5–10 minutes (creates ~10 unique venvs across cases). Subsequent runs are seconds (envs are cached by `(python_version, sorted(pip_spec))` hash, so cases sharing identical envs share one venv). `runner_utils.py` sets `MPLBACKEND=Agg` for all subprocess runs to avoid Tk-backend imports failing on the python-build-standalone Python 3.8 build.

## Case schema (`candidates.json`)

Each entry is a dict:

```json
{
  "case_id": "bcb_002",
  "task_id": "BigCodeBench/53",
  "libs": ["regex", "pandas", "matplotlib", "seaborn"],
  "library_under_test": "seaborn",
  "bad_version": "0.10.1",
  "good_version": "0.13.2",
  "bad_env_pip": ["matplotlib==3.2.2", "numpy==1.18.5", "pandas==1.0.5", "regex", "seaborn==0.10.1"],
  "bad_python": "3.8",
  "error_type": "AttributeError",
  "kind": "introduction",
  "rule_label": "sns_histplot",
  "reason": "sns.histplot added in seaborn 0.11",
  "evidence_line": "    sns.histplot(data=df, x=\"Age\")",
  "note": "",
  "instruct_prompt": "...",
  "code_prompt": "...",
  "canonical_solution": "...sns.histplot(data=df, x=\"Age\")...",
  "correct_solution":   "...sns.distplot(df[\"Age\"], kde=False)...",
  "test": "...",
  "entry_point": "task_func",
  "verified": true
}
```

- `kind` — `"introduction"` (canonical uses an API that didn't exist in `bad_version`) or `"removal"` (canonical uses an API that was removed in `bad_version`).
- `bad_env_pip` — full pip-installable list. Peer libs are pinned to mutually-compatible snapshot versions to avoid (a) a second confounding break and (b) ABI mismatches (e.g. pandas wheel ↔ numpy version). Transitive ABI deps (numpy ↔ pandas, etc.) are auto-included.
- `bad_python` — venv Python version. Some intro cases (sns.histplot, sns.displot) need Python 3.8 because their old peers (matplotlib 3.2.2, numpy 1.18.5) lack 3.10+ wheels; everything else uses 3.11.
- `canonical_solution` — uses an API that crashes in `bad_env`.
- `correct_solution` — uses an alternative API that yields the same behavior **in the same `bad_env`**, so `correct + test` passes.
- `good_version` — documentation only (no `good_env` is built or run): the lib version where `canonical` works.
- `verified` — `True` only if both: (1) `canonical + test` crashes in `bad_env` with `error_type` in stderr, and (2) `correct + test` passes in `bad_env`. Property (2) is what proves the case is *solvable* — it surfaces unsolvable cases (test depends on `bad_env`-incompatible behavior of an alternative API) automatically.

## Running the eval

`run_eval.py` runs each verified case under two modes and scores the output:

- **`baseline`** — single LLM call (`gemini-2.5-flash`) given the task prompt + `pip freeze` of `bad_env`. No retries, no env probing, no preflight.
- **`envpilot`** — full LangGraph pipeline (analysis → env_probe → kb_query → optional web_search → preflight → generation, with retry on preflight failure up to 3 attempts).

The generated `final_code` from each mode is wrapped with the case's `code_prompt` + `test` and run inside `bad_env`.

```bash
export GOOGLE_API_KEY=...

# Run one case both modes (smoke test, ~30–60s)
uv run python benchmark/run_eval.py --case manual_011

# All 24 cases × both modes × N=1 repeat (~15–30 min, ~300+ Gemini calls)
uv run python benchmark/run_eval.py

# Stability sampling: N=3 repeats per (case, mode) — recommended for the report
uv run python benchmark/run_eval.py --n 3

# Only one mode
uv run python benchmark/run_eval.py --mode envpilot
```

Outputs:
- `benchmark/eval_results.json` — one row per (case, mode, repeat) with `final_code`, `test_passed`, `duration_s`, `total_tokens`, `llm_calls`, `preflight_attempts`, etc.
- `benchmark/eval_summary.json` — aggregate metrics:
  - **Effectiveness**: `first_pass_success_rate` / `final_success_rate` / `crash_rate` per mode
  - **Efficiency**: `mean_duration_s` / `mean_total_tokens` / `mean_llm_calls` / `mean_web_search` / `mean_preflight` / `mean_attempts` per mode
  - **Delta** (envpilot − baseline): `token_overhead`, `tokens_per_extra_pass` (overhead vs. repair savings), `success_rate_lift`, `duration_overhead_s`

Instrumentation: `envcheck/agent/nodes.py` exposes `reset_metrics()` / `get_metrics()` that count LLM calls, tokens (from `usage_metadata`), web_search invocations, preflight runs. `run_eval.py` resets before each run and snapshots after.

The eval flow does **not** modify `candidates.json` and uses the same cached `envs/` as `verify_ground_truth.py`, so re-runs are fast.

## Distribution

24 cases (all `verified=true`):
- **Libraries**: seaborn (10) · numpy (4) · pandas (4) · scikit-learn (2) · Pillow (1) · matplotlib (1) · scipy (1) · flask (1) — 8 libs
- **Direction**: introduction (10) · removal (14)
- **Error types**: AttributeError (17) · TypeError (4) · ImportError (3)
- **Source**: BCB directed regex search (11) · hand-written (13)

## Data sources & attribution

- **BigCodeBench** (`bigcode/bigcodebench`, split `v0.1.4`) — Apache License 2.0 — https://huggingface.co/datasets/bigcode/bigcodebench. The `task_id`, `libs`, `instruct_prompt`, `code_prompt`, `canonical_solution`, `test`, and `entry_point` fields of `bcb_*` cases are derived from this dataset. Selection of cases was via regex search for known breaking-change patterns; the dataset content itself is not modified.
- **Manual cases** — hand-written in BigCodeBench dict format following the `Manual/N` task_id convention. Source-of-truth lives in `manual_cases.py`.
- **Breaking-change version pairs** — cross-referenced from official release notes / migration guides of NumPy, pandas, scikit-learn, scipy, matplotlib, seaborn, Pillow, and Flask.
