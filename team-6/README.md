# CardioRAG-CX: Multimodal Cardiac Agent

**Team ID:** Team 6  

**Members:** Chuankai Xu, Xinyue Xu, Youke Zhang  
## Overview
We develop a **multi-modal cardiac diagnostic agent using open-source models**. The system integrates heterogeneous clinical inputs—including ECG signals, medical imaging, and doctor notes—through a hierarchical agent architecture.

## Architecture

```
┌──────────────────────────────────┐
│      Streamlit GUI (app.py)      │
│  Upload ECG/DICOM → View results │
└───────────────┬──────────────────┘
                │
┌───────────────▼──────────────────┐
│      Planner (planner.py)        │
│  Orchestrates tools, tracks      │
│  thinking steps, calls Qwen3-VL  │
└──┬──────────────┬──────────┬─────┘
   │              │          │
┌──▼─────-─┐ ┌────▼────┐ ┌───▼──────────┐
│ECGFounder│ │LingShu  │ │ Qwen3-VL     │
│Tool      │ │Tool     │ │ (vLLM:8000)  │
│(CPU)     │ │(GPU:2,3)│ │ (GPU:0,1)    │
│          │ │         │ │              │
│WFDB→numpy│ │DICOM→PNG│ │ Synthesize   │
│→classify │ │→analyze │ │ all findings │
│→waveform │ │→findings│ │ into report  │
└──────────┘ └─────────┘ └──────────────┘
```

## File Structure

```
cardioagent_demo/
├── app.py                    # Streamlit GUI
├── planner.py                # Agent orchestrator
├── tools/
│   ├── __init__.py
│   ├── ecgfounder_tool.py    # ECG analysis (reads .hea/.edf/.csv/.npy)
│   └── lingshu_tool.py       # MRI analysis (reads .dcm, calls LingShu API)
├── run.sh                    # One-click launcher
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start model servers (on Rivanna with 2× A100)

```bash
# Terminal 1: Qwen3-VL on GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 llama.cpp/llama-server \
  -m models/Qwen3VL-32B-Instruct-F16-split-00001-of-00002.gguf \
  --mmproj models/mmproj-Qwen3VL-32B-Instruct-F16.gguf \
  --n-gpu-layers 99 \
  --port 8000 \
  --host 0.0.0.0 \
  --api-key cardioagent

# Terminal 2: LingShu-8B on GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 vllm serve lingshu-8b \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --port 8001 \
  --trust-remote-code \
  --api-key lingshu-key
```

### 3. Launch GUI

```bash
streamlit run app.py --server.port 8501
```

### Or use the all-in-one launcher:

```bash
chmod +x run.sh
./run.sh
```

### 4. Open browser

Navigate to `http://localhost:8501`

Upload ECG files (.hea+.dat) and/or DICOM files (.dcm), add clinical notes, click "Run Analysis".

## Works Without GPU Servers

If vLLM servers are not running, the app still works in **fallback mode**:
- ECGFounder tool runs locally on CPU (signal analysis + waveform generation)
- LingShu MRI analysis will be skipped
- Qwen3-VL synthesis will be replaced with a simple tool-output report

This lets you develop and test the GUI without waiting for model servers.

## Supported Input Formats

| Modality | Formats | Notes |
|----------|---------|-------|
| ECG | `.hea`+`.dat` (WFDB) | PTB-XL, MIMIC-ECG format |
| ECG | `.edf` | European Data Format |
| ECG | `.csv` | Columns = leads, 500Hz assumed |
| ECG | `.npy` | Shape: (samples, leads) or (leads, samples) |
| ECG | `.xml` | GE MUSE format |
| MRI/CT | `.dcm` | Single file or multi-file series |
| MRI/CT | `.png`/`.jpg` | Pre-converted images |

## SLURM Job Script (Set-up on Rivanna)

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

cd /your/path/to/the/src
./run.sh
```

## Video demo
https://youtu.be/tu9FSAB928M