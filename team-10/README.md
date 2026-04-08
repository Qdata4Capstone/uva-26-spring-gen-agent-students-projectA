# Agent Alignment Testbed

A multi-agent alignment evaluation framework featuring domain-specific target agents (medical, financial, customer service), an adaptive red team agent using a UCB1 bandit over six attack surfaces (MARSE), and a reproducible static baseline evaluator (ABATE) — all built on a generative agent memory loop with no external agent frameworks.

# INFO
Adaptive Multi-Turn Red Teaming for LLM Agents via Active Exploration \\

Team-10, Raffi Khondaker \\

**OVERVIEW:**: A multi-agent alignment evaluation setup with domain-specific  agents (medical, financial, customer service), an adaptive red team agent using a bandit algorithm six attack patterns (MARSE), and a static baseline evaluator (ABATE) \\
**VIDEO:**: [DEMO HERE](https://youtu.be/NRwuzaOHMHI)

## Setup

**Requirements:** Python 3.12, pip, an NVIDIA GPU (for vLLM), and a OpenAI API key for second set of results

```bash
cd src
# If on portal
source portal_setup.sh
# Otherwise
python3.12 -m venv .venv
source .venv/bin/activate
pip install streamlit openai matplotlib vllm
```

Create `.env` in the project root:

```
OPENAI_API_KEY="sk-..."
```

Load environment before running any script:

---

## Key results:
VLLM: `/team-10/src/experiments/vllm_light`
OpenAI: `/team-10/src/experiments/openai_fresh`

## LLM Backend Configuration

Edit `config.py` to set the backend for both the target agents and the red team agent:

```python
TARGET_LLM  = "stub"      # "stub" | "openai" | "vllm"
RED_TEAM_LLM = "stub"     # "stub" | "openai" | "vllm"
```

### Backend options

| Backend | Config key | Notes |
|---|---|---|
| Stub | `"stub"` | No API key, hardcoded responses, zero latency |
| OpenAI | `"openai"` | Requires `OPENAI_API_KEY`. Model set by `OPENAI_MODEL` (default `gpt-4o-mini`) |
| vLLM | `"vllm"` | Requires a running vLLM server. See vLLM section below |

### vLLM setup

vLLM exposes an OpenAI-compatible server. Start it before running experiments:

```bash
# Option 1 — Mistral 7B Instruct (~14 GB VRAM, fits on one RTX 4000 Ada)
.venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct
 \
    --port 8000 \
    --dtype bfloat16
```

Then set in `config.py`:

```python
TARGET_LLM   = "vllm"
RED_TEAM_LLM = "vllm"
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
```

---

## Running

### Interactive frontend (app.py — 3-agent chat UI)

```bash
.venv/bin/streamlit run app.py --server.port 8501
```

Open `http://localhost:8501`. Three tabs: Medical, Financial, Customer Service. Sidebar shows real-time agent state (memory stream, plan, reflection, tool results).

### Red team + experiment frontend (streamlit_app.py)

```bash
.venv/bin/streamlit run streamlit_app.py --server.port 8503
```

Open `http://localhost:8503`. Sidebar includes red team toggle (Mode 1: human-in-the-loop adversarial prompting) and automated experiment runner (Mode 2), though this doesnt work right now.

### Automated red team experiment

```bash
.venv/bin/python run_experiment_cli.py <target_agent> <n_turns> <stop_on_violation>

# Examples
.venv/bin/python run_experiment_cli.py medical 10 false
.venv/bin/python run_experiment_cli.py financial 20 true
.venv/bin/python run_experiment_cli.py customer_service 15 false
```

Logs written to `experiments/`. Plots written to `experiments/reports/`.

### Static Baseline experiment 

```bash
.venv/bin/python run_baseline_cli.py
```

Runs all three agents against the static probe bank. Logs written to `experiments/baseline/`. Plots written to `experiments/reports/`.

---

## Key config variables

| Variable | Default | Description |
|---|---|---|
| `TARGET_AGENT` | `"medical"` | Default target for CLI experiment runner |
| `MAX_TURNS` | `10` | Red team campaign turns |
| `VIOLATION_STOPS_EXPERIMENT` | `True` | Stop on first violation |
| `CURIOSITY_BONUS_WEIGHT` | `0.3` | UCB1 exploration coefficient |
| `EXPLOIT_VS_EXPLORE_EPSILON` | `0.25` | Epsilon for greedy surface selection |
| `BASELINE_N_PROBES_PER_CATEGORY` | `5` | Probes per category per agent in Adaptive Red teaming (ABATE) |
| `BASELINE_LLM_JUDGE_BACKEND` | `"stub"` | Judge backend for ABATE (`"stub"` or `"anthropic"`) |
