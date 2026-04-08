# CardioRAG-CX: Multimodal Cardiac Diagnostic Agent

**Team ID:** Team 6  
**Members:** Chuankai Xu, Xinyue Xu, Youke Zhang

---

## Introduction

Cardiac diagnosis typically requires a physician to synthesize signals from multiple heterogeneous modalities: electrocardiograms (ECG), cardiac MRI or CT scans, and clinical notes. Each modality uses different file formats, domain-specific terminology, and specialist interpretation. **CardioRAG-CX** addresses this challenge by integrating all three modalities into a unified hierarchical agent that produces a structured diagnostic report — using open-source models that can run on-premises.

---

## Overall Function

The system accepts any combination of:
- **ECG files** (WFDB, EDF, CSV, NumPy, GE MUSE XML formats)
- **DICOM/MRI/CT files** (single or multi-file `.dcm` series, pre-converted images)
- **Free-text clinical notes**

A Planner agent orchestrates tool calls in sequence:
1. **ECGFounderTool** classifies the ECG rhythm and generates waveform visualizations (runs on CPU)
2. **LingShuTool** analyzes DICOM/MRI images and extracts radiology findings (runs on GPU)
3. **Qwen3-VL** synthesizes all tool outputs and clinical notes into a final cardiac diagnostic report

The Streamlit UI accepts file uploads and streams the agent's thinking steps and tool outputs in real time. The system degrades gracefully — if the GPU model servers are not running, ECGFounder still operates locally and a plain tool-output report is generated.

---

## Code Structure

```
team-6/
├── src/
│   └── cardioagent_demo/
│       ├── app.py                    # Streamlit GUI — file upload + real-time result display
│       ├── planner.py                # Agent orchestrator — routes inputs to tools, calls Qwen3-VL
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── ecgfounder_tool.py    # ECG analysis: reads .hea/.edf/.csv/.npy/.xml,
│       │   │                         # runs ECGFounder classifier, generates waveform PNG
│       │   └── lingshu_tool.py       # MRI/CT analysis: converts DICOM→PNG, calls LingShu API
│       ├── run.sh                    # All-in-one launcher (starts both model servers + Streamlit)
│       ├── requirements.txt
│       └── README.md
└── data/
    └── README.md                     # Sample data and model download instructions
```

**Key design:** Planner checks which inputs were uploaded, calls the applicable tools, then packages all findings into a single prompt for Qwen3-VL to synthesize. Qwen3-VL runs as a local llama.cpp server with an OpenAI-compatible API, so the Planner interacts with it via standard HTTP calls.

---

## Installation

**Requirements:** Python 3.x, CUDA GPUs (2–4× A100 recommended for full capability)

```bash
cd src/cardioagent_demo
pip install -r requirements.txt
```

No API keys required — all models are self-hosted.

**Model servers** (required for full capability):

| Server | Model | GPU | Port |
|---|---|---|---|
| llama.cpp | Qwen3-VL 32B (GGUF) | GPU 0,1 | 8000 |
| vLLM | LingShu-8B | GPU 2,3 | 8001 |

Download model weights separately:
- `Qwen3VL-32B-Instruct-F16-split-*.gguf` + `mmproj-Qwen3VL-32B-Instruct-F16.gguf`
- LingShu-8B (HuggingFace or local copy)

---

## How to Run

### Full capability (with GPU model servers)

```bash
# Terminal 1 — Qwen3-VL on GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 llama.cpp/llama-server \
  -m models/Qwen3VL-32B-Instruct-F16-split-00001-of-00002.gguf \
  --mmproj models/mmproj-Qwen3VL-32B-Instruct-F16.gguf \
  --n-gpu-layers 99 --port 8000 --host 0.0.0.0 --api-key cardioagent

# Terminal 2 — LingShu-8B on GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 vllm serve lingshu-8b \
  --dtype bfloat16 --max-model-len 4096 --port 8001 \
  --trust-remote-code --api-key lingshu-key

# Terminal 3 — Streamlit UI
cd src/cardioagent_demo
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501`. Upload ECG and/or DICOM files, add clinical notes, click "Run Analysis".

### All-in-one launcher

```bash
cd src/cardioagent_demo
chmod +x run.sh && ./run.sh
```

### Fallback mode (CPU only, no GPU servers needed)

Start only the Streamlit UI — ECGFounder runs locally on CPU. LingShu analysis and Qwen3-VL synthesis are skipped; a plain tool-output report is shown. Useful for UI development and testing.

```bash
cd src/cardioagent_demo
streamlit run app.py --server.port 8501
```

### Deploying on Rivanna (UVA HPC) via SLURM

```bash
#!/bin/bash
#SBATCH --job-name=cardioagent
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00

module load cuda/13.0.2 gcc/11.4.0 anaconda
conda activate cardioagent
cd /path/to/src/cardioagent_demo && ./run.sh
```

---

## Supported Input Formats

| Modality | Formats | Notes |
|---|---|---|
| ECG | `.hea` + `.dat` (WFDB) | PTB-XL, MIMIC-ECG format |
| ECG | `.edf` | European Data Format |
| ECG | `.csv` | Columns = leads; 500 Hz assumed |
| ECG | `.npy` | Shape: `(samples, leads)` or `(leads, samples)` |
| ECG | `.xml` | GE MUSE format |
| MRI/CT | `.dcm` | Single file or multi-file series |
| MRI/CT | `.png` / `.jpg` | Pre-converted images |

---

## References

- **Video Demo:** https://youtu.be/tu9FSAB928M
- **ECGFounder:** Open-source ECG foundation model used for signal classification
- **LingShu-8B:** Open-source radiology vision-language model
- **Qwen3-VL 32B:** Multimodal synthesis backbone (Alibaba DAMO Academy)
