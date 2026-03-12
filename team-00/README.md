# Obscura: A Legal Compliance Agent for Vision Dataset

**Team ID:** 00
**Members:** Wentao Zhou, Guangyi Xu, Jinwei Zhou

---

## Overview

Releasing vision datasets (e.g., egocentric video, street-level footage) requires compliance with privacy regulations such as GDPR, CCPA, and conference open-science policies (CVPR, ICCV, NeurIPS). Current practice requires researchers to manually navigate these regulations and apply de-identification tools — a process that is time-consuming, error-prone, and undocumented.

**Obscura** is an agentic system that automates this workflow. Given a dataset directory and a natural-language compliance target (e.g., "I want to submit this to CVPR 2026"), Obscura:

---

## Setup

**Requirements:** Python 3.10+, PyTorch 2.2+

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model weights (see data/DATA.md for full instructions)
#    Place ego_blur_face_gen2.jit and ego_blur_lp_gen2.jit in the project root.
#    The OpenCV models are already bundled in src/agent/tools/.
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Web Interface (recommended)

```bash
cd src
uvicorn server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000 in your browser
```

Upload a dataset directory, describe your compliance target, and Obscura will stream its reasoning and processing steps in real time.

### Command-line Demo

```bash
cd src
python run_demo.py --input /path/to/dataset --target "CVPR 2026 submission"
```

### Evaluate on your own dataset

```bash
cd src
python evaluate.py --images /path/to/frames --output results.csv
```

This runs the full de-identification pipeline and reports face removal rate, re-identification rate, and SSIM for each frame.

---

## Project Structure

```
team-00/
├── src/
│   ├── agent/
│   │   ├── agent.py          # Main agentic loop
│   │   ├── controller.py     # LLM controller (Claude)
│   │   ├── critic.py         # Adversarial Re-ID critic
│   │   └── tools/
│   │       ├── egoblur_tool.py   # EgoBlur Gen2 wrapper
│   │       ├── face_blur.py      # OpenCV fallback blur
│   │       ├── pii_redactor.py   # Regex PII redactor
│   │       ├── knowledge.py      # Compliance knowledge base
│   │       └── file_manager.py   # Output/report management
│   ├── server.py             # FastAPI web server
│   ├── run_demo.py           # CLI demo
│   ├── evaluate.py           # Evaluation script
│   └── static/               # Web UI (HTML/CSS/JS)
├── data/
│   ├── markdown_data/        # 5 sample PII documents for redaction eval
│   └── DATA.md               # Instructions for downloading video data & models
├── requirements.txt
└── README.md
```

---

## Video Demo

> https://youtu.be/CYngcBjpKdM

