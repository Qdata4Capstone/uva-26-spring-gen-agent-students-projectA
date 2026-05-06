#!/usr/bin/env bash
# run_qwen_full_eval.bash
# Runs Qwen3 clinical evaluation (baseline + agent) on both Eurorad datasets
# combined (eurorad_clinical_benchmark_all.jsonl = dataset1 63 + dataset2 171 = 234 cases).
#
# Prerequisites:
#   - Local vLLM server running at http://localhost:8001/v1
#     serving Qwen/Qwen3-VL-8B-Instruct

set -e

LOCAL_VLLM_URL="${LOCAL_VLLM_URL:-http://localhost:8001/v1}"
BENCHMARK="${BENCHMARK:-eurorad_clinical_benchmark_all.jsonl}"

# Judge runs against real OpenAI; model inference goes to local vLLM.
: "${JUDGE_OPENAI_API_KEY:=${OPENAI_API_KEY}}"
: "${JUDGE_OPENAI_API_KEY:?JUDGE_OPENAI_API_KEY (or OPENAI_API_KEY) must be set for the judge}"

export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="$LOCAL_VLLM_URL"
export JUDGE_OPENAI_API_KEY

# ---------------------------------------------------------------------------
# 1. Qwen3 baseline  (no agent, all 234 cases)
# ---------------------------------------------------------------------------
echo "=== [1/2] Qwen3 baseline (both datasets, 234 cases) ==="

python eval_clinical.py \
    --benchmark "$BENCHMARK" \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --xr-only \
    --output clinical_eval_qwen3_baseline_full.jsonl

echo "Done: clinical_eval_qwen3_baseline_full.jsonl"

# ---------------------------------------------------------------------------
# 2. Qwen3 agent  (all 234 cases)
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/2] Qwen3 agent (both datasets, 234 cases) ==="

python eval_clinical.py \
    --benchmark "$BENCHMARK" \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --xr-only \
    --use-agent \
    --output clinical_eval_qwen3_agent_full.jsonl

echo "Done: clinical_eval_qwen3_agent_full.jsonl"

echo ""
echo "=== All Qwen3 full-dataset evaluations complete ==="
