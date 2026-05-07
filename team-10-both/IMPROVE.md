# IMPROVE.md — team-10 (Agent Alignment Testbed — Project B Extensions)

This file covers improvements specific to the Project B additions: the ML violation detector,
unified `run.py` CLI, combined Streamlit UI (`app_redteam_combined.py`), and `weak_medical`
target agent. For Project A improvements see `IMPROVEMENT_PLAN.md`.

## P1 — Critical Fixes

- **ML detector not integrated into live evaluation loop**: `ml_violation_detector.py` and `train_marse_ml_judge.py` exist as standalone modules but are not wired into `experiments.py` or `red_team.py`. The ML judge is trained and evaluated in isolation but never replaces or augments the keyword/LLM judge during a live experiment. Add a config flag (`VIOLATION_DETECTOR = "rule" | "ml" | "llm"`) and integrate the ML detector as a selectable backend.
- **ML model not persisted across runs**: `train_marse_ml_judge.py` trains the sentence-transformer + scikit-learn classifier but the output `.joblib` file path is not documented and not checked at startup. If the model file is missing, the ML detector silently falls back (or crashes). Validate the model file exists before starting any experiment that uses `ml` detection; log a clear error if absent.
- **`weak_medical` agent has no documented threat model**: The intentionally weak baseline agent is present but its weaknesses are not enumerated. Without knowing what it is weak against, it cannot be used to calibrate the red team's sensitivity. Add a `docs/weak_medical_design.md` explaining which perceive-layer rules are deliberately disabled.

## P2 — Reliability

- **Combined Streamlit UI (`app_redteam_combined.py`) shares state unsafely**: If two browser tabs run experiments simultaneously, both write to the same `experiments/` directory with timestamp-based filenames. A race between two concurrent runs can interleave log entries. Use session-scoped subdirectories or a lock file.
- **`run.py cross` sweep not resumable**: A 4×4 cross-domain sweep (16 pairs × N turns) can take tens of minutes. If interrupted, there is no checkpoint; the sweep restarts from scratch. Add a `--resume` flag that skips pairs whose log file already exists.
- **`run.py baseline` silently skips `weak_medical`**: ABATE baseline runs across `medical`, `financial`, and `customer_service` but does not include `weak_medical`, making it impossible to measure the detector's false-negative rate on the deliberately weak agent. Add `weak_medical` as an optional baseline target.
- **No timeout on individual experiment turns**: A single LLM call that hangs (network issue, rate limit) blocks the entire experiment indefinitely. Add a per-turn timeout (e.g., 30 s) with a graceful skip-and-log behaviour.

## P3 — Quality

- **ML training data provenance not recorded**: `train_marse_ml_judge.py` uses experiment logs as training data but does not record which log files were used or their timestamps in the saved model artifact. A model trained on stale or biased logs will silently underperform. Save a `training_manifest.json` alongside the `.joblib` file.
- **No evaluation split for ML detector**: Training and evaluation of the ML detector appear to use the same experiment logs with no held-out test set. Add a proper train/validation/test split (e.g., 60/20/20 by experiment timestamp) and report precision, recall, and F1 on the test set.
- **`app_redteam_combined.py` duplicates logic from `app.py` and `streamlit_app.py`**: All three apps implement session management, agent initialization, and result rendering independently. Extract shared components into a `ui_helpers.py` module.
- **Config not tagged in ML model artifact**: The scikit-learn model does not store the `sentence-transformers` model name or hyperparameters used during training. If `config.py` changes (e.g., different embedding model), the saved classifier becomes incompatible with no warning. Embed config metadata in the `.joblib` artifact.

## P4 — Enhancements

- **ML detector not used for real-time UCB1 feedback**: MARSE currently updates bandit scores using the rule/LLM judge verdict. The ML detector, being faster, could provide a real-time signal after each turn to adjust UCB1 scores mid-experiment. Feed ML detector output into the bandit update loop as a low-latency signal.
- **`run.py` has no `--dry-run` mode**: Adding `--dry-run` would print the experiment plan (agents, turn counts, attack surfaces, output paths) without executing it — useful for validating config before a long sweep.
- **No HTML/PDF report export from combined UI**: `reporting.py` generates matplotlib plots but the combined Streamlit UI has no export button. Add a "Download Report" button that bundles the experiment JSON + plots into a zip file.
- **`weak_medical` not included in cross-domain sweep**: The cross-domain `run.py cross` mode pairs all agents against all attackers but excludes `weak_medical`. Including it as an attacker target would measure whether cross-domain attacks are more effective against a weak agent than within-domain attacks.
