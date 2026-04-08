# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: MedRAX — Medical AI Agent for Radiology

An agentic system designed to assist radiologists with chest X-ray analysis. Also introduces **ChestAgentBench**, a benchmark with 2,500+ complex medical queries across 8 categories.

**Team:** Mengmeng Ma, Kathleen O'Donovan

## Setup

```bash
cd src
pip install -e .
# Set model_dir in main.py to your local model weights directory
# Add OPENAI_API_KEY to .env
```

## Running the Agent

```bash
cd src
python main.py
# If permission issues:
sudo -E env "PATH=$PATH" python main.py
```

Configure which tools to load in `main.py`:

```python
selected_tools = [
    "ImageVisualizerTool",
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    # Add/remove as needed
]
agent, tools_dict = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=selected_tools,
    model_dir="/model-weights"
)
```

## Running Benchmarks

All benchmark scripts are in `src/experiments/`:

```bash
cd src/experiments

# ChestAgentBench
python benchmark_gpt4o.py
python benchmark_llama.py       # Llama 3.2 Vision 90B
python benchmark_chexagent.py

# LLaVA-Med (requires cloning their repo first)
mv benchmark_llavamed.py ~/LLaVA-Med/llava/serve
python -m llava.serve.benchmark_llavamed --model-name llava-med-v1.5-mistral-7b --controller http://localhost:10000

# Inspect logs
python inspect_logs.py [optional: log-file] -n [num-logs]

# Analyze results
python analyze_axes.py results/<logfile>.json ../benchmark/questions/ --model [gpt4|llama|chexagent|llava-med] --max-questions [int]

# CheXbench (requires local dataset files — see experiments/README.md)
# Compare runs
python compare_runs.py results/medmax.json
python compare_runs.py results/medmax.json results/gpt4o.json  # 2-model comparison
python compare_runs.py results/medmax.json results/gpt4o.json results/llama.json  # all models
```

## Architecture

The agent is initialized in `main.py` via `initialize_agent()`, which wires a set of named tools to an LLM backbone (OpenAI). Tools are modular — any tool can be omitted by removing it from `selected_tools`. System prompt lives in `medrax/docs/system_prompts.txt`.

**ChestAgentBench** is a novel benchmark stored under `src/experiments/benchmark/questions/` covering 8 diagnostic categories. Results are logged to `src/experiments/results/` as JSON files.

## Key Dependencies

- Python 3.8+, CUDA/GPU recommended
- OpenAI API key
- Local model weights directory (set `model_dir` in `main.py`)
