# Agent Alignment Testbed

**Team:** Raffi Khondaker (Team-10)  

## Overview
This repo implements an alignment evaluation framework for domain agents, with two core evaluation modes:

- `MARSE` (adaptive red team): multi-turn attacker that adapts attack surfaces over time.
- `ABATE` (static baseline): fixed probe-bank evaluation with reproducible scoring.

Supported target agents:

- `medical`
- `weak_medical` (intentionally weak baseline)
- `financial`
- `customer_service`

## Current Architecture (`src/`)

- `run.py` — unified CLI entrypoint (`experiment`, `cross`, `baseline`)
- `experiments.py` — MARSE + ABATE runners
- `red_team.py` — evaluators (rule / ML / hybrid LLM judge)
- `agents/__init__.py` — target agents + red-team agent
- `backends.py` — backend wiring + agent factory
- `reporting.py` — plot generation
- `app_redteam_combined.py` — Streamlit UI
- `train_marse_ml_judge.py` — train lightweight ML detector
- `ml_violation_detector.py` — ML detector load/train/inference

## Prerequisites

- Python `3.10+` (tested in this project with `3.12`)
- `pip`

## Setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r src/requirements.txt
```

Optional (only needed for ML detector training/inference):

```bash
pip install sentence-transformers scikit-learn numpy joblib
```

If using OpenAI-backed components, set:

```bash
export OPENAI_API_KEY="sk-..."
```

## Important Working Directory Note

Paths in config (like `experiments/`) are **relative to your current working directory**.

- Recommended: run commands from repo root using `python src/run.py ...`
- If you `cd src` first and run `python run.py ...`, outputs will be written under `src/experiments/...`

## How To Run

### 1) Streamlit UI

From repo root:

```bash
streamlit run src/app_redteam_combined.py --server.port 8503
```

### 2) MARSE single target (CLI)

From repo root:

```bash
python src/run.py experiment <target_agent> <n_turns> <stop_on_violation>
```

Examples:

```bash
python src/run.py experiment medical 10 false
python src/run.py experiment weak_medical 10 false
python src/run.py experiment financial 15 true
python src/run.py experiment customer_service 15 false
```

### 3) MARSE cross sweep (attacker × target)

From repo root:

```bash
python src/run.py cross <n_turns> <stop_on_violation> [cross_only=true]
```

Examples:

```bash
# 4x4 sweep: includes same-domain + cross-domain pairs
python src/run.py cross 15 false false

# cross-domain only (default behavior if third arg omitted)
python src/run.py cross 15 false true
```

### 4) ABATE baseline

From repo root:

```bash
python src/run.py baseline
```

Note: baseline currently runs across `medical`, `financial`, and `customer_service`.

### 5) Tests

From repo root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Outputs

Assuming you run from repo root:

- MARSE logs: `experiments/*_campaign_log.json`
- ABATE logs: `experiments/baseline/*_baseline_log.json`
- Plots:
  - experiment mode: `experiments/reports/experiment/<target>/<experiment_id>/`
  - cross mode: `experiments/reports/cross/<attacker>_to_<target>/<experiment_id>/`
  - baseline mode: `experiments/reports/baseline/<baseline_experiment_id>/`

If you run from `src/`, the same structure appears under `src/experiments/...`.

## MARSE Judge Modes

`MARSE_JUDGE_MODE` options:

- `rule` (default)
- `hybrid`
- `llm`

Rule backend options (`MARSE_RULE_BACKEND`):

- `pattern` (default)
- `ml`

Key env vars:

- `MARSE_JUDGE_MODE`
- `MARSE_RULE_BACKEND`
- `MARSE_LLM_JUDGE_BACKEND`
- `MARSE_LLM_JUDGE_MODEL`
- `MARSE_HYBRID_SAMPLE_RATE`
- `MARSE_ML_JUDGE_MODEL_PATH`
- `MARSE_ML_VIOLATION_THRESHOLD`
- `MARSE_ML_UNCERTAINTY_BAND`

## Train Lightweight ML Detector (Optional)

From repo root:

```bash
python src/train_marse_ml_judge.py \
  --log-glob "experiments/**/*_campaign_log.json" \
  --output-artifact experiments/models/marse_ml_detector.joblib
```

Or from labeled JSONL (`attack`, `response`, `label`):

```bash
python src/train_marse_ml_judge.py \
  --input-jsonl src/data/marse_train.jsonl \
  --output-artifact experiments/models/marse_ml_detector.joblib
```

Enable ML rule backend:

```bash
export MARSE_RULE_BACKEND=ml
export MARSE_ML_JUDGE_MODEL_PATH=experiments/models/marse_ml_detector.joblib
export MARSE_ML_VIOLATION_THRESHOLD=0.60
export MARSE_ML_UNCERTAINTY_BAND=0.10
```

With `MARSE_JUDGE_MODE=hybrid`, near-threshold ML outputs can trigger LLM judging (`llm_trigger="ml_uncertain"`).

## Backend Notes

- `TARGET_LLM` and `RED_TEAM_LLM` support: `stub`, `openai`, `vllm`
- `MARSE_LLM_JUDGE_BACKEND` supports: `openai`, `stub`
- `BASELINE_LLM_JUDGE_BACKEND` supports: `openai`, `stub`
- `vllm` mode expects an OpenAI-compatible server at `VLLM_BASE_URL` (default: `http://localhost:8000/v1`)

## Notes

- `weak_medical` is intentionally easier to break and is used as a weak-defense baseline.
- `src/old/` is preserved for history and is not the current maintained execution path.
