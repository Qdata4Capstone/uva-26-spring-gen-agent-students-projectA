#!/usr/bin/env bash
# run_all_evals.bash
# Master script for all Project B evaluations.
#
# Runs all 6 experiments:
#   1. GPT-4o  baseline  (eval_clinical.py)
#   2. GPT-4o  agent     (eval_clinical.py --use-agent)
#   3. Qwen3   baseline  (eval_clinical.py)
#   4. Qwen3   agent     (eval_clinical.py --use-agent)
#   5. GPT-4o  unknown   (eval_unknown.py)
#   6. Qwen3   unknown   (eval_unknown.py)

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# OPENAI_API_KEY must be exported by the caller (e.g. via a sourced .env file).
# It is used directly for GPT-4o runs and as the judge key for the Qwen3 runs.
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set — export it or source your .env}"

REAL_OPENAI_KEY="$OPENAI_API_KEY"
LOCAL_VLLM_URL="${LOCAL_VLLM_URL:-http://localhost:8001/v1}"

export JUDGE_OPENAI_API_KEY="$REAL_OPENAI_KEY"

# ---------------------------------------------------------------------------
# 1. GPT-4o baseline  (uses real OpenAI)
# ---------------------------------------------------------------------------
echo "=== [1/6] GPT-4o baseline ==="
export OPENAI_API_KEY="$REAL_OPENAI_KEY"
unset OPENAI_BASE_URL

python eval_clinical.py \
    --model gpt-4o \
    --temperature 0.2 \
    --max-cases 32 \
    --xr-only \
    --output clinical_eval_gpt-4o_baseline_new.jsonl

echo "Done: clinical_eval_gpt-4o_baseline_new.jsonl"

# ---------------------------------------------------------------------------
# 2. GPT-4o agent  (uses real OpenAI)
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/6] GPT-4o agent ==="
export OPENAI_API_KEY="$REAL_OPENAI_KEY"
unset OPENAI_BASE_URL

python eval_clinical.py \
    --model gpt-4o \
    --temperature 0.2 \
    --max-cases 32 \
    --xr-only \
    --use-agent \
    --output clinical_eval_gpt-4o_agent_new.jsonl

echo "Done: clinical_eval_gpt-4o_agent_new.jsonl"

# ---------------------------------------------------------------------------
# 3. Qwen3 baseline  (uses local vllm)
# ---------------------------------------------------------------------------
echo ""
echo "=== [3/6] Qwen3 baseline ==="
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="$LOCAL_VLLM_URL"

python eval_clinical.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --max-cases 32 \
    --xr-only \
    --output clinical_eval_qwen3_baseline_new.jsonl

echo "Done: clinical_eval_qwen3_baseline_new.jsonl"

# ---------------------------------------------------------------------------
# 4. Qwen3 agent  (uses local vllm)
# ---------------------------------------------------------------------------
echo ""
echo "=== [4/6] Qwen3 agent ==="
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="$LOCAL_VLLM_URL"

python eval_clinical.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --max-cases 32 \
    --xr-only \
    --use-agent \
    --output clinical_eval_qwen3_agent_new.jsonl

echo "Done: clinical_eval_qwen3_agent_new.jsonl"

# ---------------------------------------------------------------------------
# 5. GPT-4o unknown eval  (uses real OpenAI)
# ---------------------------------------------------------------------------
echo ""
echo "=== [5/6] GPT-4o unknown eval ==="
export OPENAI_API_KEY="$REAL_OPENAI_KEY"
unset OPENAI_BASE_URL

python eval_unknown.py \
    --model gpt-4o \
    --temperature 0.2 \
    --max-cases 32 \
    --xr-only \
    --variants complete no_image no_history no_both \
    --output unknown_eval_gpt-4o_new.jsonl

echo "Done: unknown_eval_gpt-4o_new.jsonl"

# ---------------------------------------------------------------------------
# 6. Qwen3 unknown eval  (uses local vllm)
# ---------------------------------------------------------------------------
echo ""
echo "=== [6/6] Qwen3 unknown eval ==="
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="$LOCAL_VLLM_URL"

python eval_unknown.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --max-cases 32 \
    --xr-only \
    --variants complete no_image no_history no_both \
    --output unknown_eval_qwen3_new.jsonl

echo "Done: unknown_eval_qwen3_new.jsonl"

# ---------------------------------------------------------------------------
# 7. Analyze all results
# ---------------------------------------------------------------------------
echo ""
echo "=== Generating final comparison tables ==="
python analyze_results.py
