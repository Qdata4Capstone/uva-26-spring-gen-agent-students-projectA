# Obscura: Legal Compliance Agent for Vision Datasets

**Team ID:** 00  
**Members:** Wentao Zhou, Guangyi Xu, Jinwei Zhou

---

## Introduction

Releasing vision datasets — egocentric video, street-level footage, medical imagery — requires compliance with privacy regulations such as GDPR, CCPA, and open-science policies from venues like CVPR, ICCV, and NeurIPS. Current practice forces researchers to manually navigate these regulations and apply de-identification tools, a process that is time-consuming, error-prone, and undocumented.

**Obscura** eliminates this manual workflow by providing an end-to-end agentic system that audits a dataset against a natural-language compliance target and automatically applies de-identification.

---

## Overall Function

Given a dataset directory and a compliance goal (e.g., `"I want to submit this to CVPR 2026"`), Obscura:

1. **Audits** the dataset against the applicable privacy regulations using a curated compliance knowledge base (GDPR, CCPA, conference policies)
2. **De-identifies** images using EgoBlur Gen2 (face and license plate blurring) with an OpenCV fallback
3. **Redacts PII** from accompanying text files using regex patterns
4. **Verifies** the result using an adversarial re-identification critic that checks whether faces remain identifiable
5. **Reports** the compliance steps taken in a structured output

The web interface streams the agent's reasoning and processing steps in real time.

---

## Code Structure

```
team-00/
├── src/
│   ├── agent/
│   │   ├── agent.py          # Main agentic loop — iterates until compliance target is met
│   │   ├── controller.py     # LLM controller (Claude); decides which tools to invoke
│   │   ├── critic.py         # Adversarial re-identification critic; verifies de-ID quality
│   │   └── tools/
│   │       ├── egoblur_tool.py   # EgoBlur Gen2 face + license plate blurring (primary)
│   │       ├── face_blur.py      # OpenCV fallback blur
│   │       ├── pii_redactor.py   # Regex-based PII redaction for text
│   │       ├── knowledge.py      # Compliance knowledge base (GDPR, CCPA, conference rules)
│   │       └── file_manager.py   # Output management and report generation
│   ├── server.py             # FastAPI web server with real-time reasoning stream
│   ├── run_demo.py           # CLI demo script
│   ├── evaluate.py           # Evaluation script (face removal rate, re-ID rate, SSIM)
│   └── static/               # Web UI — plain HTML/CSS/JS
├── data/
│   ├── markdown_data/        # 5 sample PII documents for redaction evaluation
│   └── DATA.md               # Instructions for downloading video data and model weights
├── requirements.txt
└── README.md
```

**Agent flow:** `controller.py` receives the compliance target, selects tools from the `tools/` directory, and calls them iteratively. After each de-identification pass, `critic.py` checks for residual re-identification risk. The loop continues until the critic is satisfied or the maximum number of iterations is reached.

---

## Installation

**Requirements:** Python 3.10+, PyTorch 2.2+

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model weights (see data/DATA.md for full instructions)
#    Place ego_blur_face_gen2.jit and ego_blur_lp_gen2.jit in the project root.
#    The OpenCV models are already bundled in src/agent/tools/.

# 4. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## How to Run

### Web Interface (recommended)

```bash
cd src
uvicorn server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000 in your browser
```

Upload a dataset directory, describe your compliance target, and Obscura streams its reasoning and processing steps in real time.

### Command-Line Demo

```bash
cd src
python run_demo.py --input /path/to/dataset --target "CVPR 2026 submission"
```

### Evaluation

Runs the full de-identification pipeline and reports face removal rate, re-identification rate, and SSIM for each frame:

```bash
cd src
python evaluate.py --images /path/to/frames --output results.csv
```

---

## References

- **Video Demo:** https://youtu.be/CYngcBjpKdM
- **EgoBlur Gen2** (Meta): face and license plate blurring model used as the primary de-identification tool
- Privacy regulations covered: GDPR, CCPA, CVPR/ICCV/NeurIPS open-science policies
