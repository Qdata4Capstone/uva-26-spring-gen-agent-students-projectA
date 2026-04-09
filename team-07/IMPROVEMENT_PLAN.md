# Improvement Plan — MedRAX (team-07)

## Project Summary
MedRAX is a medical AI agent for radiology that integrates 10 specialized tools (classification, segmentation, report generation, grounding, GradCAM, DICOM processing, and more) under a LangGraph-based conversational interface backed by OpenAI GPT-4o. It also introduces ChestAgentBench, a 2,500-question evaluation framework across 8 diagnostic categories.

---

## Strengths to Preserve
- Modular tool architecture — each tool is independently loadable and testable
- Comprehensive toolset covering detection, localization, generation, and explainability
- Thoughtful system prompt engineering with explicit multi-step workflow examples
- Novel ChestAgentBench evaluation framework
- Tool call logging with timestamped JSON audit files
- Graceful error handling and fallback messaging

---

## Priority 1 — Critical Fixes (Correctness & Portability)

### 1.1 Remove Hardcoded Paths
**Problem:** `main.py` contains absolute paths specific to one machine (`/usa/mengma/myproject/...`) and hardcodes `GRADIO_TEMP_DIR` and CUDA device settings at the top of the file.

**Action:**
- Move all path configuration to environment variables or a `config.yaml` file.
- Provide a `.env.example` with all required fields:
  ```
  MODEL_DIR=/path/to/model/weights
  OPENAI_API_KEY=sk-...
  DEVICE=cuda
  GRADIO_SERVER_PORT=8585
  ```
- Load these at startup using `python-dotenv`; raise `EnvironmentError` with a clear message if any required variable is missing.

### 1.2 Fix Incorrect Model Name
**Problem:** `main.py` sets `model="gpt-5.2"` which does not exist; the agent will fail to initialize with a cryptic API error.

**Action:**
- Change to `"gpt-4o"` (or make it configurable via `OPENAI_MODEL` environment variable).
- Add a startup check that validates the model name against the OpenAI API's model list.

### 1.3 Define Missing RAG Tool Classes
**Problem:** `backends.py` references `MedicalRAGTool()`, `FinancialRAGTool()` (wrong domain), and `CustomerServiceRAGTool()` which do not exist in `tools/__init__.py`, causing `ImportError` at runtime.

**Action:**
- Define a `_MedicalRAGTool` base class (inheriting from `BaseTool`) in `tools/__init__.py` backed by the existing medical knowledge corpus.
- Remove non-medical RAG tool references that are irrelevant to this project.

### 1.4 Standardize Tool Return Types
**Problem:** Tools return inconsistent types — some return JSON strings, some return tuples, some return dicts — making composition and display unreliable.

**Action:**
- Define a shared `ToolResult` dataclass: `{"status": "success"|"error", "data": ..., "display_path": str|None}`.
- Update all `_run()` methods to return `ToolResult` and update `interface.py` to handle the unified format.

---

## Priority 2 — Robustness & Quality

### 2.1 Build a Unit Test Suite
**Problem:** The `tests/` directory contains only `__init__.py`. There are zero tests.

**Action:**
- Write tests for at least:
  - `ChestXRayClassifierTool._run()` with a synthetic grayscale image
  - `DicomProcessorTool._run()` with a sample DICOM file
  - The agentic loop routing (mock LLM responses)
  - System prompt loading from file
- Use `pytest` with `unittest.mock` to mock model weights; aim for ≥60% line coverage.

### 2.2 Implement Async Tool Execution
**Problem:** All `_arun()` methods simply call `_run()` synchronously, defeating the purpose of LangGraph's async support.

**Action:**
- Wrap heavy inference calls in `asyncio.get_event_loop().run_in_executor(None, self._run, ...)` to release the event loop during GPU inference.
- This enables concurrent tool calls when the LLM requests multiple tools in one step.

### 2.3 Add Tool Parameter Validation
**Problem:** `GradCAMExplainerTool` accepts arbitrary class names without checking them against the classifier's 18-pathology list; invalid class names produce silent errors.

**Action:**
- Add a `VALID_CLASSES` constant in `classification.py` listing all 18 pathologies.
- Validate the `target_class` parameter in `GradCAMExplainerTool._run()` against this list; return a clear error message if invalid.

### 2.4 Add Timeout for Tool Inference
**Problem:** Long-running inference calls (e.g., diffusion generation, MAIRA-2 grounding) can hang indefinitely, blocking the agent loop.

**Action:**
- Wrap inference calls in `concurrent.futures.ThreadPoolExecutor` with a `timeout` parameter (e.g., 60 seconds).
- Return a structured error message to the LLM if the timeout is exceeded.

### 2.5 Pin External Model Versions
**Problem:** `pyproject.toml` pins `transformers` to a specific commit but doesn't pin HuggingFace model repository revisions. CheXagent, MAIRA-2, and LLaVA-Med can change silently.

**Action:**
- Record the `revision` SHA for each downloaded HuggingFace model in `config.yaml`.
- Pass `revision=...` to all `from_pretrained()` calls.
- Add a `verify_models.py` script that checks model hashes on startup.

---

## Priority 3 — Features & UX

### 3.1 Add Input Validation for Image Quality
**Problem:** The agent accepts any image path without checking whether it is a frontal chest X-ray or has acceptable quality.

**Action:**
- Before routing to tools, run a lightweight classifier (or heuristic on aspect ratio + pixel intensity distribution) to flag non-X-ray or low-quality inputs.
- Surface a warning in the Gradio UI if the image may not be a frontal CXR.

### 3.2 Standardize Benchmark Output Format
**Problem:** ChestAgentBench questions lack `CERTAINTY`, `ALTERNATIVES`, and `REASONING` fields, making it hard to evaluate whether the agent's logic was sound even if the final answer matched.

**Action:**
- Add these optional fields to the benchmark schema in `create_benchmark.py`.
- Update `analyze_axes.py` to compute partial-credit scoring when reasoning matches even if the final label differs.

### 3.3 Fix Gradio Configuration
**Problem:** `demo.launch(share=True)` in `main.py` creates a public Gradio tunnel, which is a security risk. The port is also hardcoded.

**Action:**
- Set `share=False` by default; make it opt-in via `GRADIO_SHARE=true` environment variable.
- Read the port from `GRADIO_SERVER_PORT` environment variable (default `8585`).

### 3.4 Add Dynamic Model Loading
**Problem:** All model weights are loaded at startup regardless of which tools are enabled, consuming maximum GPU memory.

**Action:**
- Load model weights lazily — only when the tool is first called.
- Provide a `tools_to_use` configuration flag that restricts which tools (and thus which weights) are loaded.

---

## Priority 4 — Testing & Deployment

### 4.1 Add a Dockerfile
**Problem:** The complex dependency stack (GDCM, PyDICOM, PyTorch, multiple HuggingFace models) is hard to reproduce without a container.

**Action:**
- Write a `Dockerfile` with a CUDA base image.
- Document model download steps as `RUN` commands or a separate `download_weights.sh` script.
- Add a `docker-compose.yml` for one-command local startup.

### 4.2 Add a Configuration Template
**Problem:** No `.env.example` or `config.yaml` template exists; new users must read source code to determine what to configure.

**Action:**
- Create `.env.example` and `config.yaml.example` with all required and optional fields, including model paths, API keys, device, and port settings.
- Add a startup configuration validator that prints a friendly checklist of missing values.

### 4.3 Add `/health` Endpoint to Gradio App
**Problem:** There is no way to check if the service is alive and all model weights are loaded.

**Action:**
- Expose a simple HTTP health-check endpoint (using FastAPI mounted alongside Gradio, or a separate lightweight server) that returns model load status and GPU memory usage.

---

## Summary Roadmap

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Remove hardcoded paths → `.env` | Low |
| 1 | Fix model name (`gpt-5.2` → `gpt-4o`) | Low |
| 1 | Define missing RAG tool classes | Medium |
| 1 | Standardize tool return types | Medium |
| 2 | Unit test suite (≥60% coverage) | High |
| 2 | Implement async tool execution | Medium |
| 2 | Validate `GradCAM` class names | Low |
| 2 | Add inference timeout | Low |
| 2 | Pin HuggingFace model revision SHAs | Low |
| 3 | Input image quality validation | Medium |
| 3 | Enrich ChestAgentBench schema | Medium |
| 3 | Fix Gradio `share=True` & port | Low |
| 3 | Lazy model loading | Medium |
| 4 | Dockerfile + docker-compose | Medium |
| 4 | `.env.example` + config validator | Low |
| 4 | Health-check endpoint | Low |
