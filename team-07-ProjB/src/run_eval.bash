#!/usr/bin/env bash
# run_eval.bash
# Quick smoke-test eval against quickstart.py with the agent enabled.
# Export OPENAI_API_KEY before running.

set -e

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"

python quickstart.py \
    --model gpt-4o \
    --temperature 0.2 \
    --max-cases 10 \
    --log-prefix gpt-4o \
    --use-agent
