# IMPROVE.md — MedRAX (team-07-both)

## Project Overview

MedRAX is a medical AI agent for chest X-ray analysis backed by OpenAI GPT-4o and a LangGraph state machine. It integrates 10 specialized tools (classification, segmentation, report generation, grounding, GradCAM, DICOM processing, VQA, image generation) and introduces ChestAgentBench — a 2,500+ question evaluation framework across 8 diagnostic categories. Project B extends Project A with a clinical reasoning evaluation pipeline over the Eurorad case bank, adding two eval scripts (`eval_clinical_v2.py`, `eval_clinical_v3.py`), an "acknowledge unknown" evaluator, and multi-model benchmarking scripts.

---

## What Changed Since Commit 1df9809

Compared to the original `team-07/` at commit `1df9809`, `team-07-both` adds significant new work:

| Area | What Was Added |
|------|---------------|
| Eval pipeline v2 | `src/eval_clinical_v2.py` (851 lines) — 3-pass tool-grounded self-reflection |
| Eval pipeline v3 | `src/eval_clinical_v3.py` (764 lines) — structured JSON output with pre/post reflection phases |
| Unknown eval | `src/eval_unknown.py` — uncertainty calibration / "acknowledge unknown" evaluation |
| Result analysis | `src/analyze_results.py` (309 lines) — aggregates JSONL output into comparison tables |
| Clinical benchmark | `src/build_clinical_benchmark.py` — converts Eurorad → JSONL benchmark |
| Multi-model benchmarks | `src/experiments/benchmark_chexagent.py`, `benchmark_llama.py`, `benchmark_llavamed.py`, `benchmark_gpt4o.py` |
| Multi-model analysis | `src/experiments/analyze_axes.py`, `compare_runs.py`, `inspect_logs.py`, `validate_logs.py` |
| Chainlit UI | `src/interface_v3.py` (300 lines) — async Chainlit chat interface alongside existing Gradio UI |
| Shell scripts | `run_eval.bash`, `run_eval_clinical.bash`, `run_qwen3vl.bash`, etc. — batch experiment runners |
| Tool annotation | `src/annotate_tools.py`, `run_annotate_tool.bash` — retroactive tool call annotation in logs |

---

## Strengths to Preserve

- Modular tool architecture — each tool is independently loadable via `tools_to_use` config
- 3-pass self-reflection evaluation design (`eval_clinical_v3.py`) is novel and rigorous
- Structured JSON output format in v3 evaluation enables automated scoring
- Tool call audit trail (timestamped JSON) enables retrospective analysis
- Comprehensive multi-model benchmarking (GPT-4o, Qwen3-VL, LLaVA-Med, CheXagent)
- Dual UI support (Gradio for deployment, Chainlit for development/debugging)
- Thoughtful system prompt with explicit multi-step clinical workflow examples

---

## P1 — Critical Issues (Will Break at Runtime)

### P1.1 Hardcoded Absolute Path at Module Load Time
**File:** `src/main.py`, line 2  
**Code:**
```python
os.environ["GRADIO_TEMP_DIR"] = "/usa/mengma/myproject/team-07/src/mydownload/temp"
```
This line runs before any argument parsing and silently overwrites `GRADIO_TEMP_DIR` with a path that exists only on one specific machine. Any other user will get errors when Gradio tries to write temporary files.  
**Fix:** Remove the hard-coded line; read from `GRADIO_TEMP_DIR` env var if set, otherwise default to a relative `./temp` directory.

### P1.2 `transformers.__version__` Monkey-Patch
**File:** `src/medrax/tools/xray_vqa.py`, line 69  
**Code:**
```python
transformers.__version__ = "4.40.0"  # Dangerous code, but works for now
```
This mutates a library's internal version string at import time. Any code elsewhere that checks `transformers.__version__` (including other tools in the same process) will see the fake version, potentially bypassing safety checks or triggering incorrect code paths.  
**Fix:** Pin `transformers==4.40.0` in `pyproject.toml` and remove the monkey-patch. If multiple tools need different versions, document that in the README.

### P1.3 Contradictory `gradio` Dependency Constraints
**File:** `src/pyproject.toml`, lines 37 and 51  
**Code:**
```toml
"gradio>=3.0.0",    # line 37
...
"gradio>=5.0.0",    # line 51
```
Two conflicting lower bounds are declared. pip satisfies both (takes the higher bound), but the presence of both is an error-prone maintenance trap — future maintainers may tighten one to `==3.x` and break Gradio 5 features.  
**Fix:** Remove the `>=3.0.0` line; keep only `"gradio>=5.0.0"`.

### P1.4 `GradCAMExplainerTool._arun` Raises `NotImplementedError`
**File:** `src/medrax/tools/explainability.py`, line 194  
**Code:**
```python
async def _arun(self, *args, **kwargs) -> str:
    raise NotImplementedError("GradCAMExplainerTool does not support async")
```
LangGraph's async mode will call `_arun`. Raising here will terminate the agent loop.  
**Fix:** Implement `_arun` by wrapping `_run` in `asyncio.get_event_loop().run_in_executor(None, self._run, ...)`.

### P1.5 Zero Test Coverage
**File:** `tests/__init__.py` (empty)  
No automated tests exist for any of the 10 tools, the agent loop, or the 3 evaluation scripts. The benchmark scripts cannot be verified without model weights, but unit tests with mocked models are fully feasible.  
**Fix:** Add a `pytest` suite targeting at minimum:
- `ChestXRayClassifierTool._run()` with a 224×224 synthetic grayscale PIL image
- `DicomProcessorTool._run()` with one of the demo DCM files in `src/demo/chest/`
- Agent routing logic with a mocked `ChatOpenAI` that returns a tool call then a final message
- System prompt loading from `medrax/docs/system_prompts.txt`
- Score parsing in `eval_clinical.py` and `analyze_results.py`

---

## P2 — Robustness Issues

### P2.1 No Startup Validation for Required Environment
**File:** `src/main.py`  
The agent initializes without checking that `OPENAI_API_KEY` is set, that `MODEL_DIR` exists, or that the model name is valid. Failures manifest at first tool call or first LLM request, far from the source.  
**Fix:** Add a `validate_environment()` function called before `initialize_agent()` that:
- Raises `EnvironmentError` with a clear message if `OPENAI_API_KEY` is missing
- Warns if `MODEL_DIR` does not exist (some users may intentionally run tool-subset mode)
- Validates the model string against a known-good list or a lightweight API call

### P2.2 No Inference Timeouts
**Files:** All tool files (especially `generation.py`, `grounding.py`, `llava_med.py`)  
Diffusion generation, MAIRA-2 grounding, and LLaVA-Med inference can take minutes or hang indefinitely (e.g., on CPU). The agent loop blocks silently.  
**Fix:** Wrap each tool's `_run()` inference block in `concurrent.futures.ThreadPoolExecutor` with a `timeout` (suggest 120s for generation, 60s for others). On timeout, return a structured error JSON so the LLM can report the issue rather than hanging.

### P2.3 No Input Validation for Tool Parameters
**Files:** `src/medrax/tools/explainability.py`, `generation.py`, `grounding.py`  
- `GradCAMExplainerTool` accepts any `target_class` string without checking against the DenseNet's 18-pathology list; an invalid class name silently defaults to the top prediction.
- `ChestXRayGeneratorTool` accepts `num_inference_steps=0` or negative values without error.
- `XRayPhraseGroundingTool` accepts `max_new_tokens` values that can exceed model limits.  
**Fix:** Add Pydantic validators on tool input models; define `VALID_PATHOLOGIES` as a constant in `classification.py` and import it in `explainability.py`.

### P2.4 Evaluation Scripts Fail Silently on Malformed JSON
**Files:** `src/eval_clinical.py`, `src/eval_clinical_v3.py`  
LLM responses are parsed with `json.loads()` inside a bare `try/except Exception` block that logs a warning and returns `None`. Downstream scoring then propagates `None`, producing misleading summary statistics rather than flagging the failure.  
**Fix:** On JSON parse failure, log the raw response and the case ID, then either skip the case (excluded from averages) or assign a score of 0 with a `"parse_error"` flag — never silently propagate `None` into aggregate metrics.

### P2.5 `Gradio share=True` Default
**File:** `src/main.py`, argument parser  
A public Gradio tunnel is created by default. Any user who runs `python main.py` without reading the docs exposes the service to the internet.  
**Fix:** Set `share=False` as the default; make it opt-in via `--share` flag or `GRADIO_SHARE=true` environment variable.

### P2.6 `ImageVisualizerTool` Display Logic Commented Out
**File:** `src/medrax/tools/utils.py`, line ~101  
```python
# self._display_image(image_path, title, description, figsize, cmap)
```
The tool's core function is disabled. When the LLM calls this tool, it returns metadata but no visible output.  
**Fix:** Either uncomment the display call and verify it works in both Gradio and Chainlit contexts, or remove the tool from the available tool list to avoid confusing the LLM.

### P2.7 Inconsistent Tool Return Types
**Files:** All tool files  
Tools return `str` (JSON), `tuple`, or `dict` inconsistently. `interface_v2.py` has ad-hoc formatters for each tool, making the addition of new tools require UI changes.  
**Fix:** Define a `ToolResult(TypedDict)` with fields `status`, `data`, and `display_path`. Update all `_run()` methods and centralize formatting in `interface_v2.py`.

---

## P3 — Code Quality & Maintainability

### P3.1 Evaluation Script Duplication
`eval_clinical.py`, `eval_clinical_v2.py`, and `eval_clinical_v3.py` share substantial logic (case loading, image handling, GPT-4o scoring, JSONL output). The duplication means bug fixes must be applied to three files.  
**Fix:** Extract shared code into `src/eval_utils.py` with functions like `load_benchmark()`, `score_with_llm()`, `write_jsonl_result()`. The three eval scripts then become thin wrappers around the shared utility.

### P3.2 No Structured Logging
All debug output is via `print()`. In multi-model batch evaluations (`run_qwen3vl.bash` runs many cases), it is impossible to filter logs by severity.  
**Fix:** Replace `print()` with `logging.getLogger(__name__)` throughout; add a `--log-level` flag to evaluation scripts.

### P3.3 HuggingFace Model Versions Not Pinned
`transformers` is pinned to a specific commit SHA, but the actual model repository revisions (CheXagent, MAIRA-2, LLaVA-Med) are not. Any of these can change behavior silently.  
**Fix:** Add a `MODEL_REVISIONS` dict in `config.py` mapping model IDs to specific commit SHAs, and pass `revision=MODEL_REVISIONS[model_id]` to every `from_pretrained()` call.

### P3.4 Tool Name Convention Inconsistency
- `explainability.py`: `name = "GradCAMExplainerTool"` (PascalCase)
- `classification.py`: `name = "chest_xray_classifier"` (snake_case)
- `xray_vqa.py`: `name = "chest_xray_expert"`

The LLM sees these names in the tool call spec; inconsistency increases the chance of tool selection errors.  
**Fix:** Standardize all tool names to `snake_case` (e.g., `gradcam_explainer`, `chest_xray_classifier`, `chest_xray_vqa`).

### P3.5 Shell Scripts Have No Error Handling
`run_eval.bash`, `run_qwen3vl.bash`, etc. do not use `set -euo pipefail`. A failed Python script in the middle of a batch run produces no error signal; the script continues as if it succeeded.  
**Fix:** Add `set -euo pipefail` to the top of each bash script.

---

## P4 — Features & Deployment

### P4.1 Add `.env.example`
No template for required environment variables exists. Users must read source code to determine what to set.  
**Fix:** Create `src/.env.example`:
```
OPENAI_API_KEY=sk-...
MODEL_DIR=/path/to/model-weights
TEMP_DIR=./temp
DEVICE=cuda
OPENAI_MODEL=gpt-4o
GRADIO_SERVER_PORT=8585
GRADIO_SHARE=false
```

### P4.2 Implement Lazy Model Loading
All tool weights are loaded at startup regardless of which tools are selected. On a machine without enough VRAM, this will OOM even if only 2 tools are needed.  
**Fix:** Change `all_tools` dict in `main.py` from eagerly-called lambdas to a deferred initialization pattern: store the factory function and only call it on first use inside `Agent.execute_tools()`.

### P4.3 Add LRU Cache for Repeated Image Analysis
The same image can be analyzed multiple times in a conversation (e.g., user asks "classify" then "segment" on the same file). Each call re-runs full inference.  
**Fix:** Add `@functools.lru_cache(maxsize=32)` on a thin wrapper around `_run()` keyed by `(image_path, tool_name, params_hash)`.

### P4.4 Add a Dockerfile
The dependency stack (GDCM, PyDICOM, PyTorch with CUDA, multiple HuggingFace models) is fragile to reproduce.  
**Fix:** Write a `Dockerfile` from `nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04` with all pip installs and a `download_weights.sh` script for model weights. Add a `docker-compose.yml` for one-command startup.

---

## Summary Roadmap

| Priority | Item | File(s) | Effort |
|----------|------|---------|--------|
| P1 | Remove hardcoded `GRADIO_TEMP_DIR` | `main.py:2` | Low |
| P1 | Remove `transformers.__version__` monkey-patch | `xray_vqa.py:69` | Low |
| P1 | Fix duplicate `gradio` constraint | `pyproject.toml:37,51` | Low |
| P1 | Implement `GradCAMExplainerTool._arun` | `explainability.py:194` | Low |
| P1 | Write unit test suite (≥60% coverage) | `tests/` | High |
| P2 | Add startup environment validation | `main.py` | Low |
| P2 | Add inference timeouts (ThreadPoolExecutor) | all tool files | Medium |
| P2 | Add Pydantic input validation for tools | `explainability.py`, `generation.py` | Medium |
| P2 | Fix silent JSON parse failure in eval scripts | `eval_clinical*.py` | Low |
| P2 | Fix `Gradio share=True` default | `main.py` | Low |
| P2 | Fix `ImageVisualizerTool` commented-out logic | `tools/utils.py` | Low |
| P2 | Standardize tool return types to `ToolResult` | all tool files | Medium |
| P3 | Extract shared eval utilities | `eval_clinical*.py` | Medium |
| P3 | Replace `print()` with `logging` | all eval scripts | Low |
| P3 | Pin HuggingFace model revision SHAs | all tool files | Low |
| P3 | Standardize tool name convention | all tool files | Low |
| P3 | Add `set -euo pipefail` to bash scripts | `run_*.bash` | Low |
| P4 | Create `src/.env.example` | — | Low |
| P4 | Implement lazy model loading | `main.py` | Medium |
| P4 | Add LRU cache for repeated image analysis | all tool files | Medium |
| P4 | Dockerfile + docker-compose | — | Medium |
