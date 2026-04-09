# Improvement Plan — CardioRAG-CX (team-6)

## Project Summary
CardioRAG-CX is a multimodal cardiac diagnostic agent that integrates ECG analysis (NeuroKit2 + ECGFounder), MRI/CT analysis (DICOM → LingShu-8B), and cross-modal synthesis (Qwen3-VL 30B) in a Streamlit UI with real-time thinking-step visualization. It runs on HPC (Rivanna) using dual A100 GPUs.

---

## Strengths to Preserve
- Real-time thinking step visualization streamed via callback to Streamlit UI
- Modular tool architecture — each modality is independently runnable
- Graceful degradation when GPU model servers are unavailable
- Five ECG input formats supported (WFDB, EDF, CSV, NPY, XML)
- Multi-step signal processing with NeuroKit2 (R-peak, ST detection, rhythm)
- DICOM → PNG with proper windowing/leveling and montage generation
- OpenAI-compatible API abstraction for both LingShu and Qwen3-VL

---

## Priority 1 — Critical Fixes (Correctness & Security)

### 1.1 Implement ECGFounder Model Loading
**Problem:** `_run_ecgfounder()` in `ecgfounder_tool.py` raises `NotImplementedError`. The core ECGFounder classification (150-class pathology detection) is entirely disabled. Only the rule-based NeuroKit2 analysis runs.

**Action:**
- Implement the method by:
  1. Loading the ECGFounder checkpoint from a configurable path (`ECGFOUNDER_CHECKPOINT` env var).
  2. Running inference on the preprocessed 12-lead signal tensor.
  3. Parsing class probabilities and returning the top-3 diagnoses with confidence scores.
- If the checkpoint is not found, log a warning and fall back to NeuroKit2-only analysis (do not raise).
- Document the checkpoint download source in the README.

### 1.2 Move Hardcoded API Keys to Environment Variables
**Problem:** `planner.py` uses `api_key="cardioagent"` and `lingshu_tool.py` uses `api_key="lingshu-key"` directly in source code. These are visible in git history.

**Action:**
- Replace with `os.environ.get("QWEN_API_KEY", "cardioagent")` and `os.environ.get("LINGSHU_API_KEY", "lingshu-key")`.
- Add both variables to a `.env.example` template.
- Add a startup check that warns if default placeholder values are still in use.

### 1.3 Implement Temp File Cleanup
**Problem:** `app.py` creates per-session temp directories (`/tmp/cardioagent_*`) but never deletes them. The comment at line 479 acknowledges the issue. Long-running servers accumulate gigabytes of image files.

**Action:**
- Register a cleanup function using `atexit.register()` or a Streamlit `on_session_end` callback that deletes the session's temp directory after the results have been displayed.
- Add a configurable `TEMP_RETENTION_MINUTES` variable (default: 30) for deferred cleanup.
- Add a periodic cleanup job (daily cron or startup-time sweep) that removes directories older than the retention period.

### 1.4 Add Input Validation
**Problem:** There are no checks on file sizes, DICOM validity, patient ID format, or frame count. A malformed DICOM causes an unhandled exception that crashes the analysis thread.

**Action:**
- Add file size limits (configurable, default: 500 MB per upload) enforced in `app.py` before saving.
- Validate DICOM files by checking the standard preamble and `SOPClassUID` before passing to `lingshu_tool.py`.
- Validate ECG files by attempting to open them with the appropriate reader before the full pipeline runs.
- Return a clear user-facing error message for each failure type.

---

## Priority 2 — Robustness & Quality

### 2.1 Build a Unit Test Suite
**Problem:** The `tests/` directory has only `__init__.py`. There are no tests, even though `pytest` is installed.

**Action:**
- Write tests for:
  - `ecgfounder_tool.py`: ECG reading (CSV format with synthetic data), preprocessing, lead reordering
  - `lingshu_tool.py`: DICOM → PNG conversion with a synthetic single-frame DICOM (use `pydicom.dataset.Dataset`)
  - `planner.py`: fallback report generation when model servers are unavailable
- Use `unittest.mock` to mock HTTP calls to LingShu and Qwen3-VL servers.
- Aim for ≥ 50% line coverage of the tool files.

### 2.2 Activate Structured Logging
**Problem:** `logger = logging.getLogger(__name__)` is defined in all files but never used. All output goes via `print()` or is embedded in `ThinkingStep` objects.

**Action:**
- Replace all `print()` calls with appropriate `logger.info()` / `logger.warning()` / `logger.error()` calls.
- Configure a `RotatingFileHandler` writing to `logs/cardioagent_{date}.log`.
- Log key events: model server connectivity check, per-step timing, error types.

### 2.3 Fix DICOM Pixel Value Handling (HU Units)
**Problem:** `lingshu_tool.py` normalizes pixel values to [0,1] via min-max scaling, which destroys Hounsfield Unit (HU) values in CT scans. HU values are clinically meaningful (bone = +1000, air = -1000, soft tissue = ~50).

**Action:**
- Detect modality from DICOM `Modality` tag (`"CT"` vs `"MR"`).
- For CT: apply the standard HU formula (`pixel_value × RescaleSlope + RescaleIntercept`) before windowing.
- For MR: use the existing min-max normalization.
- Pass modality and windowing parameters to LingShu as part of the prompt context.

### 2.4 Improve Error Propagation to User
**Problem:** Broad `except Exception as e:` blocks mark a step as "error" but do not surface the error prominently in the UI. Users see a degraded report without understanding what failed.

**Action:**
- Distinguish error categories in the UI: "Model server unavailable", "File format not supported", "Analysis incomplete".
- Show a prominent banner in the Streamlit UI when one or more steps failed, explaining which modality was affected and what the user can do (e.g., "Start the LingShu server to enable MRI analysis").
- Log the full traceback to the log file; show only the user-friendly message in the UI.

### 2.5 Guard DICOM Memory Usage
**Problem:** `pydicom.dcmread()` in a loop over a DICOM series has no frame count limit. A 4D cine DICOM with 500 frames could exhaust GPU memory.

**Action:**
- Add a configurable `MAX_DICOM_FRAMES` limit (default: 64) enforced before the montage generation loop.
- Log a warning when frames are truncated: "Series has 500 frames; using first 64 for analysis."

---

## Priority 3 — Features & Clinical Workflow

### 3.1 Add Confidence Scoring to the Final Report
**Problem:** The Qwen3-VL synthesis returns free text with no quantified uncertainty. Clinical users need to know how confident the system is in each finding.

**Action:**
- Add an explicit confidence elicitation step to the Qwen3-VL prompt: "For each finding, provide a confidence level: HIGH (>85%), MEDIUM (60–85%), or LOW (<60%)."
- Parse these levels from the response and display them as colored badges next to each finding in the Streamlit UI.
- Surface findings from ECGFounder classification (which already returns probabilities) directly in the report with their raw confidence scores.

### 3.2 Add Cross-Modal Consistency Checking
**Problem:** ECG and MRI findings are kept in separate state fields. The Synthesizer does not validate whether they agree (e.g., "ST elevation on ECG but no wall motion abnormality on MRI" is clinically significant disagreement).

**Action:**
- Add a consistency-checking step in `planner.py` between ECG and MRI analysis:
  1. Pass both analyses to the LLM with the prompt: "Do the ECG and imaging findings agree? Identify any discordances."
  2. Append the consistency note to the final report as a dedicated "Cross-Modal Agreement" section.

### 3.3 Add Report Export
**Problem:** The final report is displayed in Streamlit only. Clinicians need archivable, shareable output.

**Action:**
- Add a "Download Report (PDF)" button using a Python PDF library (e.g., `reportlab` or `weasyprint`).
- Include: patient ID, timestamp, ECG waveform image, MRI montage, and the synthesis report in the PDF.
- Add a "Download JSON" button that exports the full `AgentState` as structured data.

### 3.4 Fix Lead Reordering Silent Zero-Fill
**Problem:** When the input ECG has fewer than 12 leads, missing leads are silently zero-filled. This can produce false ST readings on synthetic zero-signal leads.

**Action:**
- After lead reordering, check which leads were zero-filled (those whose source was not found in `name_map`).
- Log a warning and mark those leads as "unavailable" in the ECG metrics output.
- Exclude zero-filled leads from ST detection and the waveform visualization legend.

---

## Priority 4 — Documentation & Deployment

### 4.1 Add a Configuration File Template
**Problem:** Model paths, API URLs, and GPU port numbers are scattered across the codebase with no central `.env` template.

**Action:**
- Create `.env.example` with all configurable values:
  ```
  ECGFOUNDER_CHECKPOINT=/path/to/checkpoint.pth
  QWEN_API_KEY=cardioagent
  QWEN_BASE_URL=http://localhost:8000/v1
  LINGSHU_API_KEY=lingshu-key
  LINGSHU_BASE_URL=http://localhost:8001/v1
  TEMP_RETENTION_MINUTES=30
  MAX_DICOM_FRAMES=64
  ```
- Document all variables and their purpose in the README.

### 4.2 Add Architecture Diagram
**Problem:** The README has a text description but no visual diagram. The data flow through three tools and a synthesis step is non-trivial.

**Action:**
- Add a Mermaid or ASCII diagram to the README showing: file upload → ECG/DICOM parsing → tool agents → Qwen3-VL synthesis → report, with GPU server locations annotated.

### 4.3 Add Troubleshooting Guide
**Problem:** Common failure modes (GPU OOM, LingShu server not starting, ECGFounder checkpoint not found) are not documented.

**Action:**
- Add a "Troubleshooting" section to the README covering the top 5 failure scenarios with diagnostic steps and solutions.

---

## Summary Roadmap

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Implement ECGFounder model loading | High |
| 1 | Move API keys to environment variables | Low |
| 1 | Implement temp file cleanup with TTL | Low |
| 1 | Add input validation (file size, DICOM, ECG) | Medium |
| 2 | Build unit test suite (≥50% coverage) | High |
| 2 | Activate structured logging | Low |
| 2 | Fix DICOM HU pixel value handling | Medium |
| 2 | Improve error propagation to UI | Medium |
| 2 | Guard DICOM frame count memory usage | Low |
| 3 | Add confidence scoring to final report | Medium |
| 3 | Add cross-modal consistency checking | Medium |
| 3 | Add report export (PDF + JSON) | Medium |
| 3 | Fix lead reordering silent zero-fill | Low |
| 4 | Create `.env.example` template | Low |
| 4 | Add architecture diagram to README | Low |
| 4 | Add troubleshooting guide | Low |
