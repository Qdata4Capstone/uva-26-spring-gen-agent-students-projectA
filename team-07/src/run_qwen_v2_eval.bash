#!/usr/bin/env bash
# run_qwen_v2_eval.bash
# Runs Qwen3 clinical evaluation using eval_clinical_v2.py (3-pass tool-grounded
# self-reflection) on both Eurorad datasets combined (234 cases).
#
# Pass 1  — vision-only diagnosis (no tools)
# Pass 2  — targeted tool calls to challenge Pass 1 (classifier, GradCAM, report gen)
# Pass 3  — reconciliation: model revises output after addressing tool contradictions
#
# Prerequisites:
#   - Local vLLM server running at http://localhost:8001/v1
#     serving Qwen/Qwen3-VL-8B-Instruct --max-model-len 32768

set -e

cd "$(dirname "$0")"

LOCAL_VLLM_URL="${LOCAL_VLLM_URL:-http://localhost:8001/v1}"
BENCHMARK="${BENCHMARK:-eurorad_clinical_benchmark.jsonl}"

# Judge runs against real OpenAI; model inference goes to local vLLM.
# Export JUDGE_OPENAI_API_KEY (or OPENAI_API_KEY) before running this script.
: "${JUDGE_OPENAI_API_KEY:=${OPENAI_API_KEY}}"
: "${JUDGE_OPENAI_API_KEY:?JUDGE_OPENAI_API_KEY (or OPENAI_API_KEY) must be set for the judge}"

export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="$LOCAL_VLLM_URL"
export JUDGE_OPENAI_API_KEY

# ---------------------------------------------------------------------------
# 1. Qwen3 baseline — Pass 1 only (no agent, no tools)
# ---------------------------------------------------------------------------
echo "=== [1/2] Qwen3 v3 baseline (vision-only, full dataset) ==="

python eval_clinical_v3.py \
    --benchmark "$BENCHMARK" \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --xr-only \
    --max-cases 32 \
    --output clinical_eval_v3_qwen3_baseline.jsonl

echo "Done: clinical_eval_v3_qwen3_baseline.jsonl"

# ---------------------------------------------------------------------------
# 2. Qwen3 agent — all 3 passes (tool-grounded self-reflection)
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/2] Qwen3 v3 agent (self-reflection, full dataset) ==="

python eval_clinical_v3.py \
    --benchmark "$BENCHMARK" \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --xr-only \
    --use-agent \
    --max-cases 32 \
    --output clinical_eval_v3_qwen3_agent.jsonl

echo "Done: clinical_eval_v3_qwen3_agent.jsonl"

echo ""
echo "=== All Qwen3 v3 evaluations complete ==="
