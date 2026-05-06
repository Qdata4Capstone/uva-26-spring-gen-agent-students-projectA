# IMPROVE.md — team-6-ProjB (CardioAgent)

## P1 — Critical / Security

- **5+ hardcoded Rivanna paths and localhost URLs** (`app_dis.py:78-84`, `eval_v5.py:75-81`): `/scratch/rzv4ve/cardioagent/`, `http://localhost:8000/v1`, `http://localhost:8001/v1` are baked into source. Move to environment variables (`QWEN_URL`, `LINGSHU_URL`, `VECTOR_DB_PATH`) loaded via `.env`.
- **No startup config validation**: App starts even if model servers are unreachable or data directories don't exist. Add a preflight check that probes `QWEN_URL` and `LINGSHU_URL` health endpoints before accepting requests.
- **Missing data existence checks** (`eval_v5.py:186-189`): `.npy` files are loaded with `mmap_mode` but no existence check. A missing file raises a cryptic `FileNotFoundError` deep in eval; check all 4 files at startup and fail with a clear message.

## P2 — Reliability

- **Silent embedder init failure** (`planner_v4.py:107-115`): `vector_memory = None` is set silently when any embedder fails to load. Downstream RAG queries then silently fall back to keyword search with no log or user warning. Log which embedder failed and why.
- **Silent ECG parse failure** (`planner_v4.py:268-276`): Nested try/except swallows `wfdb` parsing errors; the ECG signal is set to `None` with no indication in the output report. Surface a warning in the agent trace.
- **Corrupted vector DB not detected**: If the vector DB files exist but are corrupt (partial write, disk error), loading fails deep in FAISS with an opaque error. Add a checksum or a small query probe on startup.
- **No retry on model server calls**: If Qwen3-VL or LingShu vLLM server has a transient blip, the planner fails the entire case. Add 1–2 retries with exponential backoff on HTTP 5xx responses.

## P3 — Quality

- **Default planner parameters hardcoded in signature** (`planner_v4.py:68-70`): `base_url` defaults embedded in the constructor make the class hard to configure without subclassing. Accept a config dict or read from env.
- **RRF fusion underperforms D1 but is kept as default** (`doc/architecture.md:81-84`): Results show RRF trailing single-pathway retrieval, yet RRF is still the default. Either fix the fusion weights or flip the default and document why.
- **Index build commands assume data exists** (`doc/architecture.md:136-148`): `build-train`, `build-cxr`, `build-symile` have no documented failure modes if source files are missing. Add guards or a `--dry-run` flag.
- **No automated tests for eval pipeline**: `eval_v5.py` is a single script with no unit tests. A broken regex or schema change would only be caught at eval time. Add at minimum a smoke test with a mock patient record.

## P4 — Enhancements

- **Unified config module missing**: Five files each parse their own hardcoded constants. A single `config.py` loading from `.env` / `config.yaml` would eliminate drift.
- **Eval script not runnable standalone**: `eval_v5.py` assumes it is run from a specific working directory. Accept `--data-dir` and `--output-dir` CLI arguments.
- **No confidence calibration validation**: Architecture doc promises "calibrated confidence per hypothesis" but there is no calibration curve or reliability diagram in the eval output. Add ECE (Expected Calibration Error) to `eval_v5.py`.
- **Phase timing not logged**: The 6-phase planner has no per-phase latency instrumentation. Add timestamps at each phase boundary to identify bottlenecks.
