# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: CardioRAG-CX — Multimodal Cardiac Diagnostic Agent

A multi-modal cardiac diagnostic agent using open-source models. Integrates ECG signals, medical imaging (DICOM/MRI), and clinical notes through a hierarchical agent architecture. Designed to run on Rivanna (UVA HPC) with 2–4× A100 GPUs.

**Team:** Chuankai Xu, Xinyue Xu, Youke Zhang (Team 6)

## Setup

```bash
cd src/cardioagent_demo
pip install -r requirements.txt
```

No API keys required — all models are self-hosted.

## Running

### With GPU model servers (full capability)

```bash
# Terminal 1: Qwen3-VL (32B) on GPU 0,1 via llama.cpp server
CUDA_VISIBLE_DEVICES=0,1 llama.cpp/llama-server \
  -m models/Qwen3VL-32B-Instruct-F16-split-00001-of-00002.gguf \
  --mmproj models/mmproj-Qwen3VL-32B-Instruct-F16.gguf \
  --n-gpu-layers 99 --port 8000 --host 0.0.0.0 --api-key cardioagent

# Terminal 2: LingShu-8B (MRI analysis) on GPU 2,3 via vllm
CUDA_VISIBLE_DEVICES=2,3 vllm serve lingshu-8b \
  --dtype bfloat16 --max-model-len 4096 --port 8001 \
  --trust-remote-code --api-key lingshu-key
```

```bash
# Launch Streamlit UI
cd src/cardioagent_demo
streamlit run app.py --server.port 8501
# Or all-in-one:
chmod +x run.sh && ./run.sh
```

Open `http://localhost:8501`, upload ECG + DICOM files, add clinical notes, click "Run Analysis".

### Fallback mode (no GPU servers)

The app works without model servers: ECGFounder runs locally on CPU, LingShu MRI analysis is skipped, Qwen3-VL synthesis is replaced with a plain tool-output report. Useful for UI development.

### SLURM (Rivanna)

```bash
sbatch <<'EOF'
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
EOF
```

## Architecture

```
Streamlit UI (app.py)
      │
Planner (planner.py) — orchestrates tools, tracks thinking steps, calls Qwen3-VL
      ├── ECGFounderTool (tools/ecgfounder_tool.py) — ECG classification + waveform (CPU)
      ├── LingShuTool (tools/lingshu_tool.py)       — DICOM→PNG + MRI analysis (GPU 2,3)
      └── Qwen3-VL (llama.cpp server, GPU 0,1)       — multimodal synthesis → final report
```

The Planner calls each tool in sequence based on what inputs were uploaded, then feeds all tool outputs to Qwen3-VL for final synthesis.

## Supported Input Formats

| Modality | Formats |
|---|---|
| ECG | `.hea`+`.dat` (WFDB/PTB-XL), `.edf`, `.csv` (500 Hz assumed), `.npy`, `.xml` (GE MUSE) |
| MRI/CT | `.dcm` (single or multi-file series), `.png`/`.jpg` |
