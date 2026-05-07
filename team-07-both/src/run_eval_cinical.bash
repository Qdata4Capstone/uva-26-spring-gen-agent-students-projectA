#!/usr/bin/env bash
# run_eval_cinical.bash
# Runs Qwen3 baseline + agent clinical evaluations against a local vLLM server,
# using OpenAI as the judge. Export JUDGE_OPENAI_API_KEY (or OPENAI_API_KEY)
# before running.

set -e

LOCAL_VLLM_URL="${LOCAL_VLLM_URL:-http://localhost:8001/v1}"

: "${JUDGE_OPENAI_API_KEY:=${OPENAI_API_KEY}}"
: "${JUDGE_OPENAI_API_KEY:?JUDGE_OPENAI_API_KEY (or OPENAI_API_KEY) must be set for the judge}"

export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="$LOCAL_VLLM_URL"
export JUDGE_OPENAI_API_KEY

# Baseline: Qwen3-VL-8B-Instruct alone (no tools, no reflection)
python eval_clinical.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --max-cases 32 \
    --xr-only \
    --output clinical_eval_qwen3_baseline.jsonl

# Agent: Qwen3-VL-8B-Instruct + tools + reflection
python eval_clinical.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --max-cases 32 \
    --xr-only \
    --use-agent \
    --output clinical_eval_qwen3_agent.jsonl
