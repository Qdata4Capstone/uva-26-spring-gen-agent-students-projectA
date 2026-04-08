#!/bin/bash
# ============================================================
# CardioAgent Demo — Launch Script
# UVA Rivanna: 2× A100-40GB
# ============================================================

set -e

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        CardioAgent Demo Launcher         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"

# ── Step 0: Load modules (UVA Rivanna) ──
echo -e "\n${YELLOW}[0/4] Loading modules...${NC}"
if command -v module &> /dev/null; then
    module load cuda/13.0.2 2>/dev/null || true
    module load gcc/11.4.0 2>/dev/null || true
    module load anaconda 2>/dev/null || true
    echo -e "${GREEN}  Modules loaded (Rivanna detected)${NC}"
else
    echo -e "${YELLOW}  Not on Rivanna — skipping module load${NC}"
fi

# ── Step 1: Install dependencies ──
echo -e "\n${YELLOW}[1/4] Checking dependencies...${NC}"
pip install -q streamlit openai wfdb neurokit2 pydicom matplotlib \
    numpy pandas scipy Pillow 2>/dev/null

# Check optional dependencies
python -c "import chromadb" 2>/dev/null && echo -e "${GREEN}  chromadb ✓${NC}" || echo -e "${YELLOW}  chromadb not installed (RAG disabled)${NC}"
python -c "import sentence_transformers" 2>/dev/null && echo -e "${GREEN}  sentence-transformers ✓${NC}" || echo -e "${YELLOW}  sentence-transformers not installed (RAG disabled)${NC}"

echo -e "${GREEN}  Core dependencies ready${NC}"

# ── Step 2: Start Qwen3-VL (GPU 0) ──
echo -e "\n${YELLOW}[2/4] Starting Qwen3-VL Planner on GPU 0...${NC}"

QWEN_MODEL=${QWEN_MODEL:-"../qwen-model/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"}
QWEN_PORT=${QWEN_PORT:-8000}

if curl -s http://localhost:$QWEN_PORT/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}  Qwen3-VL already running on port $QWEN_PORT${NC}"
else
    if command -v vllm &> /dev/null; then
        echo -e "  Starting vLLM server... (this may take 2-5 min to load model)"
        CUDA_VISIBLE_DEVICES=0 vllm serve $QWEN_MODEL \
            --dtype bfloat16 \
            --max-model-len 4096 \
            --port $QWEN_PORT \
            --trust-remote-code \
            --gpu-memory-utilization 0.85 \
            --api-key cardioagent \
            > /tmp/qwen_server.log 2>&1 &
        QWEN_PID=$!
        echo -e "${YELLOW}  Waiting for Qwen3-VL to load (PID: $QWEN_PID)...${NC}"

        for i in {1..120}; do
            if curl -s http://localhost:$QWEN_PORT/v1/models > /dev/null 2>&1; then
                echo -e "${GREEN}  Qwen3-VL ready on port $QWEN_PORT${NC}"
                break
            fi
            sleep 5
            echo -ne "  [$((i*5))s] Still loading...\r"
        done
    else
        echo -e "${YELLOW}  vLLM not installed. Qwen3-VL will be unavailable.${NC}"
        echo -e "${YELLOW}  The app will use fallback mode (tool outputs only).${NC}"
        echo -e "${YELLOW}  Install: pip install vllm${NC}"
    fi
fi

# ── Step 3: Start LingShu-8B (GPU 1) ──
echo -e "\n${YELLOW}[3/4] Starting LingShu-8B on GPU 1...${NC}"

LINGSHU_MODEL=${LINGSHU_MODEL:-"../tools/MRI/Lingshu"}
LINGSHU_PORT=${LINGSHU_PORT:-8001}

if curl -s http://localhost:$LINGSHU_PORT/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}  LingShu-8B already running on port $LINGSHU_PORT${NC}"
else
    if command -v vllm &> /dev/null && [ -d "$LINGSHU_MODEL" ]; then
        CUDA_VISIBLE_DEVICES=1 vllm serve $LINGSHU_MODEL \
            --dtype bfloat16 \
            --max-model-len 4096 \
            --port $LINGSHU_PORT \
            --trust-remote-code \
            --gpu-memory-utilization 0.85 \
            --api-key lingshu-key \
            > /tmp/lingshu_server.log 2>&1 &
        echo -e "${YELLOW}  LingShu loading in background (check /tmp/lingshu_server.log)${NC}"
    else
        echo -e "${YELLOW}  LingShu model not found at $LINGSHU_MODEL${NC}"
        echo -e "${YELLOW}  MRI analysis will be unavailable.${NC}"
        echo -e "${YELLOW}  Set LINGSHU_MODEL=/path/to/model and rerun.${NC}"
    fi
fi

# ── Step 4: Start Streamlit GUI ──
echo -e "\n${YELLOW}[4/4] Starting Streamlit GUI...${NC}"
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  CardioAgent is starting!                ║${NC}"
echo -e "${CYAN}║                                          ║${NC}"
echo -e "${CYAN}║  GUI:       http://localhost:8502         ║${NC}"
echo -e "${CYAN}║  Qwen3-VL:  http://localhost:$QWEN_PORT        ║${NC}"
echo -e "${CYAN}║  LingShu:   http://localhost:$LINGSHU_PORT        ║${NC}"
echo -e "${CYAN}║                                          ║${NC}"
echo -e "${CYAN}║  Upload ECG (.hea) or DICOM (.dcm) to    ║${NC}"
echo -e "${CYAN}║  start analysis.                          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"

cd "$(dirname "$0")"
streamlit run app.py \
    --server.port 8502 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false