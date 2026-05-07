#!/usr/bin/env bash
# analyze_qwen_full.bash
# Evaluates results from run_qwen_full_eval.bash (both eurorad datasets, 234 cases).
# Compares Qwen3 baseline vs agent on the full dataset alongside the 32-case runs.

set -e

cd "$(dirname "$0")"

python analyze_results.py \
    --clinical-files \
        "Qwen3   | baseline (32)"="clinical_eval_qwen3_baseline_new.jsonl" \
        "Qwen3   | agent    (32)"="clinical_eval_qwen3_agent_new.jsonl" \
        "Qwen3   | baseline (all)"="clinical_eval_qwen3_baseline_full.jsonl" \
        "Qwen3   | agent    (all)"="clinical_eval_qwen3_agent_full.jsonl"
