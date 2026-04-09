# Improvement Plan — Obscura (team-00)

## Project Summary
Obscura is a legal compliance agent that de-identifies vision datasets for GDPR/CCPA and conference (CVPR/ICCV/NeurIPS) submission. It runs a 5-step agentic pipeline: intent parsing → clip classification → de-identification → EXIF stripping → re-ID critic validation.

---

## Strengths to Preserve
- Clean 5-step pipeline with clear separation of concerns (controller, tools, critic)
- Dual de-identification tools (EgoBlur Gen2 + OpenCV fallback)
- Adversarial re-ID critic with self-correction loop (up to 3 retries)
- Comprehensive compliance knowledge base (7 entities, strictest-rules-win merging)
- Real-time SSE streaming from FastAPI backend
- Concurrent clip processing via `ThreadPoolExecutor`

---

## Priority 1 — Critical Fixes (Correctness & Portability)

### 1.1 Remove Hardcoded Paths
**Problem:** Absolute paths to model weights and dataset directories are scattered across `egoblur_tool.py`, `server.py`, and `run_demo.py`, making the code non-portable.

**Action:**
- Replace all hardcoded paths with environment variables loaded from a `.env` file.
- Add a `.env.example` template documenting all required variables:
  ```
  EGOBLUR_ENV=/path/to/egoblur
  EGOBLUR_FACE_MODEL=/path/to/face_gen2.jit
  EGOBLUR_PLATE_MODEL=/path/to/plate_gen2.jit
  DATASET_DIR=/path/to/dataset
  OUTPUT_DIR=/path/to/output
  ```
- Validate all paths exist on startup; raise a clear error if missing.

### 1.2 Fix Unused `resolve_conflicts()` Integration
**Problem:** `controller.resolve_conflicts()` is defined but never called in `agent.py`. Conflicts between user preferences and regulations are silently ignored.

**Action:**
- Invoke `resolve_conflicts()` at the end of `_step_license_check()`.
- Log the resolution reasoning to the SSE stream so users understand why their preference was overridden.

### 1.3 Fix Face Alignment in Critic
**Problem:** `critic.py` computes face embeddings without calling `rec.alignCrop()`, producing less reliable re-ID comparisons.

**Action:**
- Add `aligned_face = self._recognizer.alignCrop(image, face_bbox)` before calling `self._recognizer.feature(aligned_face)` in the critic's similarity calculation.

### 1.4 Add Configuration Schema Validation
**Problem:** `regulations.json` is loaded without validation; a JSON typo causes a silent runtime crash.

**Action:**
- Define a Pydantic model for the knowledge base schema.
- Validate on startup and surface a clear error with the offending field name.

---

## Priority 2 — Robustness & Quality

### 2.1 Add File Logging & Audit Trail
**Problem:** All logs go to stdout/stderr only; SSE logs are transient. No permanent audit record exists.

**Action:**
- Add a `RotatingFileHandler` to the agent logger writing to `logs/obscura_{date}.log`.
- Include a reference to the log file path in the compliance report JSON.

### 2.2 Add Video Input Support
**Problem:** The pipeline only processes pre-extracted frames. EgoBlur's `process_video()` method exists but is never called.

**Action:**
- Detect video file extensions (`.mp4`, `.avi`, `.mov`) in `agent.py`'s input scanner.
- Call `egoblur_tool.process_video()` on video inputs; fall back to frame-by-frame extraction using OpenCV if EgoBlur is unavailable.

### 2.3 Improve Re-ID Fallback Validator
**Problem:** When DNN models are unavailable, the critic falls back to a crude 32×32 perceptual hash with no face detection.

**Action:**
- Even in fallback mode, run OpenCV's built-in `YuNet` detector (CPU-only).
- Add SSIM comparison (already implemented in `evaluate.py`) as the fallback metric instead of the perceptual hash.

### 2.4 Integrate `evaluate.py` into Critic
**Problem:** `evaluate.py` is a standalone script; SSIM metrics are never included in compliance reports.

**Action:**
- Call `evaluate_image_pair()` within `critic.py`'s validation loop.
- Add `ssim_score` and `face_count_before/after` fields to the compliance report JSON.

### 2.5 Replace Magic Numbers with Named Constants
**Problem:** Retry scale factors (`1.15 + retry*0.1`), blur strengths (`51 + retry*20`), and SSIM thresholds (`0.363`) are unexplained.

**Action:**
- Define all tunable values in a `constants.py` or `config.py` with inline documentation explaining the rationale.

---

## Priority 3 — Features & UX

### 3.1 Add Numeric Progress to SSE Events
**Problem:** Users cannot estimate time remaining on large datasets because SSE only sends text logs.

**Action:**
- Add a structured progress event type:
  ```json
  {"type": "progress", "current": 5, "total": 100, "clip": "clip_003"}
  ```
- Update the frontend to render a progress bar driven by these events.

### 3.2 Add BEFORE/AFTER Sample Output to Report
**Problem:** The compliance report contains no visual evidence of anonymization quality.

**Action:**
- Sample 1 representative frame per clip; save the original and blurred pair side-by-side.
- Embed the file paths (or base64 thumbnails) in the compliance report JSON.
- Generate a minimal HTML summary report alongside the JSON.

### 3.3 Support Web UI File Upload
**Problem:** `server.py` assumes datasets are pre-staged at a fixed disk path; there is no browser upload mechanism.

**Action:**
- Add a `POST /upload` endpoint accepting `multipart/form-data` (zip archive or individual files).
- Store uploads in a per-session temp directory; pass that path to the agent.

### 3.4 Update Knowledge Base with Version & Date Fields
**Problem:** `regulations.json` has no version or `last_updated` field; compliance data can silently become stale.

**Action:**
- Add `"version"` and `"last_updated"` fields to each regulation entry.
- Log a warning at startup if any entry is older than 180 days.

---

## Priority 4 — Testing & Deployment

### 4.1 Add Integration Tests with a Minimal Test Dataset
**Problem:** `test_agent.py` mocks all dependencies; no test exercises the real pipeline.

**Action:**
- Create a `tests/fixtures/` directory with 5 synthetic PNG frames (generate with OpenCV at test time).
- Write a pytest integration test that runs the full pipeline on these frames and asserts that the compliance report is generated and no faces remain detectable.

### 4.2 Add a Dockerfile
**Problem:** There is no container recipe; reproducing the environment requires manual steps referencing specific machine paths.

**Action:**
- Write a two-stage `Dockerfile`: base image installs Python + OpenCV + PyTorch; second stage copies application code.
- Document model weight download steps in the `Dockerfile` with `RUN` commands or a `download_models.sh` script.

### 4.3 Add `/health` Endpoint
**Problem:** The FastAPI server has no health-check route, making it unmonitorable.

**Action:**
- Add `GET /health` returning model availability status, disk space, and uptime.
- Return HTTP 200 if all critical models are loaded, HTTP 503 otherwise.

---

## Summary Roadmap

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Remove hardcoded paths → `.env` | Low |
| 1 | Invoke `resolve_conflicts()` in pipeline | Low |
| 1 | Fix face alignment in critic | Low |
| 1 | Add Pydantic validation for knowledge base | Low |
| 2 | File logging & audit trail | Low |
| 2 | Video input support | Medium |
| 2 | Improve re-ID fallback with SSIM | Medium |
| 2 | Integrate `evaluate.py` into critic | Medium |
| 2 | Replace magic numbers with constants | Low |
| 3 | Numeric SSE progress events | Low |
| 3 | BEFORE/AFTER samples in report | Medium |
| 3 | Web UI file upload | Medium |
| 3 | Knowledge base versioning | Low |
| 4 | Integration tests with fixtures | Medium |
| 4 | Dockerfile | Medium |
| 4 | `/health` endpoint | Low |
