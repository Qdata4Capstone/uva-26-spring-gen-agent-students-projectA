#!/usr/bin/env bash
# run_eval_unknown.bash
# Runs the "unknown" evaluation against a local Qwen3-VL vLLM server.
# JUDGE_OPENAI_API_KEY (or OPENAI_API_KEY) must be exported by the caller —
# the judge uses real OpenAI even though model inference is local.

set -e

LOCAL_VLLM_URL="${LOCAL_VLLM_URL:-http://localhost:8001/v1}"

: "${JUDGE_OPENAI_API_KEY:=${OPENAI_API_KEY}}"
: "${JUDGE_OPENAI_API_KEY:?JUDGE_OPENAI_API_KEY (or OPENAI_API_KEY) must be set for the judge}"

export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="$LOCAL_VLLM_URL"
export JUDGE_OPENAI_API_KEY

python eval_unknown.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --temperature 0.2 \
    --max-cases 32 \
    --xr-only \
    --variants complete no_image no_history no_both \
    --output unknown_eval_qwen3.jsonl
