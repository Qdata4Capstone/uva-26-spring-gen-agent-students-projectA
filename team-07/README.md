# MedRAX: Medical AI Agent for Radiology

**Team:** Mengmeng Ma · Kathleen O'Donovan

---

## Introduction

Radiology interpretation requires synthesizing imaging findings, clinical context, and domain knowledge — a task that is time-intensive and subject to inter-reader variability. **MedRAX** is a modular medical AI agent that integrates multiple state-of-the-art chest X-ray analysis tools under a single conversational interface backed by an LLM.

Beyond the agent itself, this project introduces **ChestAgentBench**: a novel evaluation benchmark containing over 2,500 complex medical queries across 8 diagnostic categories, designed to rigorously assess the reliability of medical AI agents as radiology co-workers.

---

## Overall Function

MedRAX lets a user (or an automated evaluation pipeline) describe a clinical task in natural language. The agent:

1. Selects the appropriate tool(s) from its toolkit based on the request
2. Calls the selected tools with the supplied chest X-ray image(s) or DICOM file(s)
3. Synthesizes tool outputs into a coherent clinical response

Available tools include image classification, segmentation, visual question answering, report generation, visual grounding, explainability (GradCAM-style), and DICOM handling. Tools are modular — any subset can be enabled at startup.

ChestAgentBench provides a standardized way to evaluate how well any model (GPT-4o, Llama 3.2, CheXagent, LLaVA-Med) handles the benchmark's 8 categories of chest X-ray queries.

---

## Code Structure

```
team-07/
├── src/
│   ├── main.py               # Entry point — configures selected tools and starts the agent
│   ├── interface.py          # CLI/REPL interface for interactive use
│   ├── quickstart.py         # Minimal quickstart script
│   ├── check_api.py          # API connectivity check
│   ├── medrax/               # Core installable package (pip install -e .)
│   │   ├── agent/
│   │   │   └── agent.py      # Agent loop: receives query → selects tools → returns response
│   │   ├── tools/            # Modular diagnostic tools (each maps to one capability)
│   │   │   ├── classification.py      # ChestXRayClassifierTool
│   │   │   ├── segmentation.py        # ChestXRaySegmentationTool
│   │   │   ├── generation.py          # ImageVisualizerTool / generation
│   │   │   ├── report_generation.py   # ReportGenerationTool
│   │   │   ├── xray_vqa.py            # Visual question answering
│   │   │   ├── grounding.py           # Visual grounding / localization
│   │   │   ├── dicom.py               # DICOM file handling
│   │   │   ├── explainability.py      # GradCAM-style explainability
│   │   │   └── llava_med.py           # LLaVA-Med model integration
│   │   ├── llava/            # LLaVA model utilities (conversation, constants)
│   │   ├── utils/            # Shared utilities
│   │   └── docs/
│   │       └── system_prompts.txt     # Agent system prompt
│   ├── benchmark/            # ChestAgentBench framework
│   │   ├── create_benchmark.py        # Benchmark dataset builder
│   │   ├── llm.py                     # LLM call abstraction
│   │   └── utils.py                   # Benchmark utilities
│   ├── experiments/          # Benchmark evaluation scripts
│   │   ├── benchmark_gpt4o.py         # GPT-4o on ChestAgentBench
│   │   ├── benchmark_llama.py         # Llama 3.2 Vision 90B
│   │   ├── benchmark_chexagent.py     # CheXagent
│   │   ├── benchmark_llavamed.py      # LLaVA-Med
│   │   ├── chexbench_gpt4.py          # CheXbench evaluation
│   │   ├── analyze_axes.py            # Per-category accuracy analysis
│   │   ├── compare_runs.py            # Multi-model comparison
│   │   ├── inspect_logs.py            # Log inspector
│   │   └── validate_logs.py           # Log validation
│   └── data/
│       ├── get_cases.py               # Case retrieval utilities
│       └── figures.py                 # Figure generation
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.8+, CUDA/GPU recommended for model inference

```bash
# Clone the repository and navigate to the src directory
cd team-07/src

# Install the medrax package and its dependencies
pip install -e .
```

**Configuration before running:**

1. Set `model_dir` in `main.py` to the path of your local model weights directory
2. Comment out any tools you do not have weights or access for
3. Add your OpenAI API key to a `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   ```

**CheXbench datasets** (required only for CheXbench evaluation):
- [SLAKE](https://www.med-vqa.com/slake/) → save to `data/slake/`
- [Rad-ReStruct](https://github.com/ChantalMP/Rad-ReStruct) → save images to `data/rad-restruct/images/`
- [Open-I](https://openi.nlm.nih.gov/faq) → save images to `data/openi/images/`
- Then fix paths in `chexbench.json` using `data/fix_chexbench.py`

---

## How to Run

### Interactive Agent

```bash
cd src
python main.py

# If you encounter permission issues:
sudo -E env "PATH=$PATH" python main.py
```

Configure which tools to load:

```python
selected_tools = [
    "ImageVisualizerTool",
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    # Add or remove as needed
]
agent, tools_dict = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=selected_tools,
    model_dir="/model-weights"
)
```

### ChestAgentBench Evaluation

```bash
cd src/experiments

# Run a specific model on ChestAgentBench
python benchmark_gpt4o.py
python benchmark_llama.py
python benchmark_chexagent.py

# LLaVA-Med (requires cloning their repo first)
mv benchmark_llavamed.py ~/LLaVA-Med/llava/serve
python -m llava.serve.benchmark_llavamed \
  --model-name llava-med-v1.5-mistral-7b \
  --controller http://localhost:10000

# Inspect logs (defaults to most recent log file)
python inspect_logs.py [log-file] -n [num-logs]

# Analyze results by category
python analyze_axes.py results/<logfile>.json ../benchmark/questions/ \
  --model [gpt4|llama|chexagent|llava-med] --max-questions [int]
```

### Multi-Model Comparison

```bash
cd src/experiments

# Single model
python compare_runs.py results/medmax.json

# Two models side by side
python compare_runs.py results/medmax.json results/gpt4o.json

# All models
python compare_runs.py results/medmax.json results/gpt4o.json results/llama.json results/chexagent.json
```

---

## References

- **ChestAgentBench:** Novel benchmark with 2,500+ queries across 8 chest X-ray diagnostic categories
- **CheXbench:** Prior SoTA benchmark used for baseline comparison
- **LLaVA-Med:** https://github.com/microsoft/LLaVA-Med
- **CheXagent:** Referenced in `benchmark_chexagent.py`
