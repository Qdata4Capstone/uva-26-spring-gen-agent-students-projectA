# Agent Alignment Testbed

**Team:** Raffi Khondaker (Team-10)  
**Video Demo:** https://youtu.be/NRwuzaOHMHI

---

## Introduction

As LLM-based agents are deployed in high-stakes domains (medical advice, financial guidance, customer service), ensuring they cannot be manipulated into producing harmful outputs becomes critical. Static safety evaluations using fixed probe sets miss adaptive adversaries that learn from prior failures.

This project builds a **multi-agent alignment evaluation framework** without relying on any external agent frameworks. It provides:
- **Domain-specific target agents** (medical, financial, customer service) implemented with a generative agent memory loop
- **MARSE** — an adaptive red team agent that uses a UCB1 multi-armed bandit to explore 6 attack surfaces and exploit the ones that most reliably elicit violations
- **ABATE** — a reproducible static baseline evaluator for comparing adaptive vs. fixed attack strategies

---

## Overall Function

The framework supports two evaluation modes:

**Mode 1 — Interactive (human-in-the-loop):** A user manually crafts adversarial prompts against a target agent via the Streamlit UI, observing real-time agent state (memory stream, plan, reflection, tool results).

**Mode 2 — Automated red team:** MARSE autonomously runs N-turn adversarial campaigns. Each turn, the UCB1 bandit selects the attack surface most likely to cause a violation, based on prior successes weighted by an exploration bonus. Logs and plots are saved to `experiments/`.

**ABATE baseline:** A fixed probe bank (configurable number of probes per category per agent) is evaluated deterministically, with an optional LLM judge for scoring.

---

## Code Structure

```
team-10/
├── src/
│   ├── app.py                # Interactive 3-agent chat UI (Streamlit, port 8501)
│   ├── streamlit_app.py      # Red team + experiment UI (Streamlit, port 8503)
│   ├── config.py             # All LLM backend and experiment settings
│   ├── backends.py           # LLM adapter layer — stub / openai / vllm
│   ├── main.py               # Core agent loop (target agent memory loop)
│   ├── red_team.py           # MARSE red team agent (UCB1 bandit over 6 attack surfaces)
│   ├── experiments.py        # ABATE static baseline evaluator
│   ├── reporting.py          # Plot + log generation, saves to experiments/reports/
│   ├── run.py                # Unified CLI runner
│   ├── agents/               # Target agent implementations
│   │   ├── __init__.py       # Medical, financial, customer service agent classes
│   │   └── ...
│   ├── tools/                # Agent tools (memory retrieval, planning, reflection)
│   │   └── __init__.py
│   ├── data/                 # Probe banks and static evaluation data
│   ├── experiments/          # Pre-computed results
│   │   ├── vllm_light/       # Results from vLLM backend (Qwen 2.5-1.5B)
│   │   └── openai_fresh/     # Results from OpenAI backend
│   └── portal_setup.sh       # Rivanna HPC setup script
└── README.md
```

**Agent memory loop (target agents):** Each target agent runs a generative agent cycle — perceive user input → retrieve relevant memories → generate a plan → execute → reflect. This architecture models realistic deployed agents rather than simple prompt-response systems.

**MARSE attack surfaces (6):** The UCB1 bandit chooses among: role-play injection, authority escalation, hypothetical framing, gradual boundary pushing, context manipulation, and direct instruction override.

---

## Installation

**Requirements:** Python 3.12, an NVIDIA GPU (for vLLM backend; stub/openai work without GPU)

```bash
cd src

# On Rivanna portal:
source portal_setup.sh

# Otherwise:
python3.12 -m venv .venv
source .venv/bin/activate
pip install streamlit openai matplotlib vllm
```

Create `.env` in `src/`:
```
OPENAI_API_KEY="sk-..."
```

### LLM Backend Configuration

Edit `src/config.py`:
```python
TARGET_LLM   = "stub"    # "stub" | "openai" | "vllm"
RED_TEAM_LLM = "stub"
```

| Backend | Notes |
|---|---|
| `"stub"` | No API key, hardcoded responses — use for development and testing |
| `"openai"` | Requires `OPENAI_API_KEY`; model set by `OPENAI_MODEL` (default: `gpt-4o-mini`) |
| `"vllm"` | Requires a running vLLM server; set `VLLM_BASE_URL` and `VLLM_MODEL` in `config.py` |

### vLLM server (optional)

```bash
.venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 --dtype bfloat16
```

Then in `config.py`:
```python
TARGET_LLM    = "vllm"
RED_TEAM_LLM  = "vllm"
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
```

---

## How to Run

### Interactive agent chat UI

```bash
.venv/bin/streamlit run app.py --server.port 8501
# Open http://localhost:8501
# Three tabs: Medical, Financial, Customer Service
# Sidebar shows real-time agent state
```

### Red team + experiment UI

```bash
.venv/bin/streamlit run streamlit_app.py --server.port 8503
# Open http://localhost:8503
# Sidebar: Mode 1 (human-in-the-loop) and Mode 2 (automated, partially implemented)
```

### Automated red team (CLI)

```bash
.venv/bin/python run_experiment_cli.py <target_agent> <n_turns> <stop_on_violation>

# Examples
.venv/bin/python run_experiment_cli.py medical 10 false
.venv/bin/python run_experiment_cli.py financial 20 true
.venv/bin/python run_experiment_cli.py customer_service 15 false
```

Logs → `experiments/`  Plots → `experiments/reports/`

### ABATE static baseline

```bash
.venv/bin/python run_baseline_cli.py
# Evaluates all three agents against the fixed probe bank
# Logs → experiments/baseline/   Plots → experiments/reports/
```

### Key Config Variables

| Variable | Default | Description |
|---|---|---|
| `MAX_TURNS` | `10` | Red team campaign turns |
| `VIOLATION_STOPS_EXPERIMENT` | `True` | Stop on first violation |
| `CURIOSITY_BONUS_WEIGHT` | `0.3` | UCB1 exploration coefficient |
| `EXPLOIT_VS_EXPLORE_EPSILON` | `0.25` | Epsilon for greedy surface selection |
| `BASELINE_N_PROBES_PER_CATEGORY` | `5` | Probes per category per agent in ABATE |
| `BASELINE_LLM_JUDGE_BACKEND` | `"stub"` | Judge backend (`"stub"` or `"anthropic"`) |

---

## References

- **Video Demo:** https://youtu.be/NRwuzaOHMHI
- **Pre-computed vLLM results:** `src/experiments/vllm_light/`
- **Pre-computed OpenAI results:** `src/experiments/openai_fresh/`
- UCB1 bandit algorithm used for adaptive attack surface selection (MARSE)
