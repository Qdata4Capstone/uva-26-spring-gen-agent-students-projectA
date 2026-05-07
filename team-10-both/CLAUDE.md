# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Agent Alignment Testbed

A multi-agent alignment evaluation framework with:
- **Domain-specific target agents** (medical, financial, customer service) built on a generative agent memory loop
- **MARSE** — adaptive red team agent using a UCB1 bandit over 6 attack surfaces
- **ABATE** — reproducible static baseline evaluator

No external agent frameworks are used.

**Team:** Raffi Khondaker (Team-10)

## Setup

```bash
cd src
python3.12 -m venv .venv
source .venv/bin/activate
pip install streamlit openai matplotlib vllm
# On Rivanna portal:
# source portal_setup.sh
```

Create `.env` in the project root:
```
OPENAI_API_KEY="sk-..."
```

## LLM Backend Configuration

Edit `src/config.py`:
```python
TARGET_LLM   = "stub"    # "stub" | "openai" | "vllm"
RED_TEAM_LLM = "stub"    # "stub" | "openai" | "vllm"
```

| Backend | Notes |
|---|---|
| `"stub"` | No API key, hardcoded responses, zero latency — use for development |
| `"openai"` | Requires `OPENAI_API_KEY`; model set by `OPENAI_MODEL` (default `gpt-4o-mini`) |
| `"vllm"` | Requires running vLLM server; set `VLLM_BASE_URL` and `VLLM_MODEL` in `config.py` |

### vLLM server
```bash
.venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 --dtype bfloat16
```

## Running

```bash
# Interactive 3-agent chat UI (Medical / Financial / Customer Service tabs)
.venv/bin/streamlit run app.py --server.port 8501

# Red team + experiment UI (human-in-the-loop adversarial prompting)
.venv/bin/streamlit run streamlit_app.py --server.port 8503

# Automated red team CLI
.venv/bin/python run_experiment_cli.py <target_agent> <n_turns> <stop_on_violation>
# Examples:
.venv/bin/python run_experiment_cli.py medical 10 false
.venv/bin/python run_experiment_cli.py financial 20 true

# Static baseline (ABATE) across all three agents
.venv/bin/python run_baseline_cli.py
```

Experiment logs → `experiments/`  
Plots → `experiments/reports/`  
Pre-computed results: `src/experiments/vllm_light` (vLLM) and `src/experiments/openai_fresh` (OpenAI)

## Architecture

**Target agents** implement a generative agent memory loop: perception → memory retrieval → planning → action. Each domain (medical, financial, customer_service) has a dedicated system prompt and tool set.

**MARSE (red team agent)** — UCB1 bandit selects among 6 attack surfaces each turn. The exploration coefficient is controlled by `CURIOSITY_BONUS_WEIGHT` in `config.py`. The agent adapts based on which attack surfaces previously elicited violations.

**ABATE (static baseline)** — fixed probe bank, `BASELINE_N_PROBES_PER_CATEGORY` probes per category per agent. Uses an LLM judge configured by `BASELINE_LLM_JUDGE_BACKEND` (`"stub"` or `"anthropic"`).

## Key Config Variables

| Variable | Default | Description |
|---|---|---|
| `MAX_TURNS` | `10` | Red team campaign turns |
| `VIOLATION_STOPS_EXPERIMENT` | `True` | Stop on first violation |
| `CURIOSITY_BONUS_WEIGHT` | `0.3` | UCB1 exploration coefficient |
| `EXPLOIT_VS_EXPLORE_EPSILON` | `0.25` | Epsilon for greedy surface selection |
| `BASELINE_N_PROBES_PER_CATEGORY` | `5` | Probes per category in ABATE |
| `BASELINE_LLM_JUDGE_BACKEND` | `"stub"` | Judge backend for ABATE |
