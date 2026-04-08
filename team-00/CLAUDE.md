# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Obscura

Legal compliance agent for vision datasets. Given a dataset directory and a natural-language compliance target (e.g., "CVPR 2026 submission"), Obscura audits the dataset against privacy regulations (GDPR, CCPA, conference open-science policies) and applies de-identification automatically.

**Team:** Wentao Zhou, Guangyi Xu, Jinwei Zhou

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
# Download model weights (ego_blur_face_gen2.jit, ego_blur_lp_gen2.jit) — see data/DATA.md
```

## Running

```bash
# Web interface (recommended)
cd src
uvicorn server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000

# CLI demo
cd src
python run_demo.py --input /path/to/dataset --target "CVPR 2026 submission"

# Evaluation (face removal rate, re-id rate, SSIM)
cd src
python evaluate.py --images /path/to/frames --output results.csv
```

## Architecture

The system is an agentic loop controlled by `src/agent/agent.py`:

- **Controller** (`controller.py`) — LLM orchestrator (Claude); decides which tools to invoke and in what order
- **Critic** (`critic.py`) — Adversarial re-identification critic; verifies de-identification quality
- **Tools** (`tools/`):
  - `egoblur_tool.py` — EgoBlur Gen2 face/license-plate blurring (primary)
  - `face_blur.py` — OpenCV fallback blur
  - `pii_redactor.py` — Regex-based PII redaction for text
  - `knowledge.py` — Compliance knowledge base (GDPR, CCPA, conference policies)
  - `file_manager.py` — Output management and report generation
- **Server** (`server.py`) — FastAPI app that streams the agent's reasoning to the browser via WebSocket/SSE
- **Static UI** (`static/`) — Plain HTML/CSS/JS frontend

The controller calls tools iteratively until the compliance target is satisfied, then the critic checks for residual re-identification risk before finalizing.

## Key Dependencies

- Python 3.10+, PyTorch 2.2+
- `anthropic` — Claude API
- `fastapi`, `uvicorn` — web server
- EgoBlur Gen2 `.jit` weights (not bundled; see `data/DATA.md`)
- OpenCV models bundled in `src/agent/tools/`
