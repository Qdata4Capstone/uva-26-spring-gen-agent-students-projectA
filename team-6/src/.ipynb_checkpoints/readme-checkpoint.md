# CardioAgent Demo

Multi-modal cardiac diagnostic agent using open-source models.

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
┌──▼──────┐ ┌────▼────┐ ┌───▼──────────┐
│ECGFounder│ │LingShu  │ │ Qwen3-VL     │
│Tool      │ │Tool     │ │ (vLLM:8000)  │
│(CPU)     │ │(GPU:1)  │ │ (GPU:0)      │
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
# Terminal 1: Qwen3-VL on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dtype bfloat16 --max-model-len 4096 --port 8000 \
    --trust-remote-code --api-key cardioagent

# Terminal 2: LingShu-8B on GPU 1
CUDA_VISIBLE_DEVICES=1 vllm serve lingshu-8b \
    --dtype bfloat16 --max-model-len 4096 --port 8001 \
    --trust-remote-code --api-key lingshu-key
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

## SLURM Job Script (Rivanna)

```bash
#!/bin/bash
#SBATCH --job-name=cardioagent
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00

module load cuda/12.4 gcc/11.4.0 anaconda
conda activate cardioagent

cd /standard/cardioagent/code/cardioagent_demo
./run.sh
```