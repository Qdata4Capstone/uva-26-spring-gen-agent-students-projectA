# Are Medical Agents Reliable Co-workers for Radiologists?

**Team 07:** Mengmeng Ma · Kathleen O'Donovan

A clinical-reasoning evaluation of multi-tool medical agents (built on the
[MedRAX](https://github.com/bowang-lab/MedRAX) framework) over the Eurorad
chest X-ray case bank. We measure not just diagnostic accuracy but several
"reliable co-worker" axes: tool-use recall, communication quality, reasoning
coherence, and uncertainty calibration.

---

## Repository layout

```
team-07/
└── src/
    ├── main.py                       # Launches the Gradio / Chainlit UI
    ├── interface_v2.py               # Gradio interface
    ├── interface_v3.py               # Chainlit interface
    ├── medrax/                       # Agent + tool implementations
    ├── build_clinical_benchmark.py   # Build the Eurorad clinical benchmark
    ├── eval_clinical.py              # Baseline / agent clinical evaluation
    ├── eval_clinical_v2.py           # 3-pass tool-grounded self-reflection
    ├── eval_clinical_v3.py           # Streamlined v3 evaluation pipeline
    ├── eval_unknown.py               # "Unknown" ablation evaluation
    ├── analyze_results.py            # Aggregate evaluation outputs
    └── run_*.bash                    # Convenience runners for each experiment
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU recommended

### Steps
```bash
git clone <repo-url>
cd uva-26-spring-gen-agent-students-projectA/team-07/src
pip install -e .
```

### Configuration
Create a `.env` file in `team-07/src/` with:

```
OPENAI_API_KEY=sk-...
# Optional — point at a local vLLM server instead of OpenAI:
# OPENAI_BASE_URL=http://localhost:8001/v1
```

Optional environment variables (all have sensible defaults):

| Var | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `team-07/src/mydownload` | Where model weights are cached |
| `TEMP_DIR` | `$MODEL_DIR/temp` | Working directory for tool outputs |
| `PROMPT_FILE` | `medrax/docs/system_prompts.txt` | System prompt file |
| `OPENAI_MODEL` | `gpt-4o` | Default OpenAI model |
| `DEVICE` | `cuda` | Torch device |
| `CUDA_VISIBLE_DEVICES` | (unset) | Pin GPU(s) at the shell level |

---

## Running the UI

```bash
# Gradio (default)
python main.py

# Chainlit
python main.py --ui chainlit
```

You can edit `selected_tools` in `main.py` to load only the tools you have
weights for.

---

## Running evaluations

All evaluation scripts read `OPENAI_API_KEY` from `.env`. To use a local
vLLM-served model for inference while still using OpenAI as the judge, set
`OPENAI_BASE_URL` and `JUDGE_OPENAI_API_KEY` separately — see the bash
runners under `run_*.bash`.

```bash
# Single experiment
python eval_clinical.py --model gpt-4o --use-agent --max-cases 32 --xr-only

# All six experiments end-to-end
bash run_all_evals.bash

# Aggregate results
python analyze_results.py
```

---

## License
See [team-07/src/LICENSE](team-07/src/LICENSE).
