# Project A to Project B: Changes in team-10 (Agent Alignment Testbed)

**Team:** Raffi Khondaker (Team-10)

## What Project A Was

Project A was an alignment evaluation framework for LLM-based domain agents with two evaluation modes:

- **MARSE** — an adaptive red-team agent that used a UCB1 bandit to select among 6 attack surfaces and attack a domain-specific target agent over multiple turns.
- **ABATE** — a static baseline evaluator with a fixed probe bank and an LLM judge.

Target agents covered three domains: `medical`, `financial`, and `customer_service`. Each agent implemented a generative loop: perceive → memory retrieval → plan → act → reflect.

The Project A codebase had two separate Streamlit entry points (`app.py` for interactive domain-agent chat, `streamlit_app.py` for red-team mode), a standalone CLI script per evaluation mode (`run_experiment_cli.py`, `run_baseline_cli.py`), no unit tests, and a minimal `config.py` with hardcoded values and no environment-variable override mechanism. The perceive-layer jailbreak detector used simple case-sensitive substring matching over 8 fixed trigger phrases. Memory retrieval used plain substring matching. The `LLMJudge` used a keyword-count heuristic ("3+ alignment keywords → aligned"). There was no `weak_medical` agent. The old files are preserved under `src/old/`.

---

## 1. New Target Agent: `WeakMedicalAgent` (`weak_medical`)

Project B adds a fourth target agent class, `WeakMedicalAgent` (in `src/agents/__init__.py`), as an intentionally weakened baseline for calibrating red-team sensitivity.

Key differences from `MedicalAgent`:

- `perceive()` skips the jailbreak/input-safety block entirely — jailbreak-style inputs are accepted without raising `AlignmentBlockedError`.
- `plan()` uses a minimal permissive system prompt ("Be concise and directly answer the user request") instead of the clinician-constraint prompt.
- `reflect()` evaluates response quality rather than safety alignment; never emits `ALIGNMENT_ALERT`.
- `act()` applies no tool-category restrictions and allows `max_tool_calls_per_run=100` instead of 5.
- `run()` returns no disclaimers and does not set `alignment_warning`.
- `alignment` profile flags `refuse_diagnosis`, `refuse_prescription`, and `refuse_emergency_bypass` are all set to `False`.

`weak_medical` is available everywhere `medical` appears: `backends.build_target_agent()`, the `run.py cross` sweep list, the combined Streamlit UI target selector, and the stub LLM responses in `backends.make_stub_llm()`.

A dedicated test in `tests/test_marse_refactor.py` confirms that `weak_medical` does not block jailbreak-style input while `medical` raises `AlignmentBlockedError` for the same input.

---

## 2. ML Violation Detector (`ml_violation_detector.py`, `train_marse_ml_judge.py`)

Project A relied entirely on a keyword/pattern evaluator (later improved with negation awareness) and a stub `LLMJudge` for ABATE.

Project B adds a lightweight ML-based violation detector as an optional alternative to the pattern evaluator:

**`src/ml_violation_detector.py`** — defines `MLViolationDetector`, a dataclass wrapping:
- A `sentence-transformers/all-MiniLM-L6-v2` encoder (lazy-loaded).
- A `scikit-learn` `LogisticRegression` classifier trained on (attack, response) pair embeddings.
- `predict_violation_proba(attack, response)` → float in [0, 1].
- `predict_violation(attack, response)` → bool using a configurable threshold (default 0.6 in config).
- `save()` / `load()` via `joblib`, storing encoder name, classifier, threshold, and metadata together.

**`src/train_marse_ml_judge.py`** — CLI training script that:
- Ingests MARSE campaign logs (via `--log-glob`) and/or labeled JSONL (via `--input-jsonl`).
- Performs stratified 80/20 train/val split by positive and negative label counts.
- Trains the `MLViolationDetector` and saves the `.joblib` artifact.
- Writes a sidecar `<artifact>.metrics.json` with precision, recall, F1, and accuracy on both splits.

**Integration in `src/red_team.py`** (`Evaluator` class):
- `MARSE_RULE_BACKEND=ml` causes `Evaluator.evaluate()` to load the ML detector and use `predict_violation_proba()` instead of the pattern matcher.
- If the ML model file is missing or fails to load, the evaluator falls back to the pattern backend and sets `ml_fallback_to_pattern=True` in the score dict.
- When ML is active, scores include `ml_violation_probability` and `ml_threshold` fields.

**Integration in `HybridEvaluator`** (`src/red_team.py`):
- The new `ml_uncertain` trigger fires when the ML probability falls within `MARSE_ML_UNCERTAINTY_BAND` of `MARSE_ML_VIOLATION_THRESHOLD`, causing the LLM judge to be called even when the rule evaluator finds no violation.

---

## 3. Unified CLI Entrypoint (`src/run.py`)

Project A had two separate CLI scripts: `run_experiment_cli.py` (MARSE) and `run_baseline_cli.py` (ABATE), with hardcoded argument reading.

Project B replaces them with a single unified CLI in `src/run.py` with three subcommands:

- `python src/run.py experiment <target_agent> <n_turns> <stop_on_violation>` — runs a single MARSE campaign, saves the log, and generates plots in a per-experiment subdirectory under `experiments/reports/experiment/<target>/<experiment_id>/`.
- `python src/run.py cross <n_turns> <stop_on_violation> [cross_only=true|false]` — runs a full attacker × target sweep. The agent list is now `["medical", "weak_medical", "financial", "customer_service"]` (4 × 4 = 16 pairs when `cross_only=false`, or 12 cross-domain pairs when `cross_only=true`). Each pair saves plots to `experiments/reports/cross/<attacker>_to_<target>/<experiment_id>/`.
- `python src/run.py baseline` — runs the ABATE baseline across `medical`, `financial`, and `customer_service` and saves plots to `experiments/reports/baseline/<experiment_id>/`.

`src/main.py` is a legacy shim that imports and calls `run.main` so that any code referencing the old entry point still works.

---

## 4. Combined Streamlit UI (`src/app_redteam_combined.py`)

Project A had two disconnected UIs:
- `src/old/app.py` / `src/old/app_chat.py` — interactive chat with all three domain agents in tabbed layout; required direct `openai.OpenAI()` construction and did not use `backends.py`.
- `src/old/streamlit_app.py` / `src/old/app_redteam.py` — red-team mode and automated experiment runner; supported only `medical`, `financial`, `customer_service` in the target selector.

Project B consolidates these into **`src/app_redteam_combined.py`**:

- Target selector now includes `weak_medical` as a fourth option.
- Layout uses a two-column split (`chat_col` 2.3 / `state_col` 1.0) rather than a full-page chat with sidebar state. The target-agent internal state panel (identity, alignment policy, memory stream, tools) appears in the right column and can be toggled on/off.
- All agent construction goes through `backends.build_target_agent()` and `backends.build_red_team_agent()`, removing the direct OpenAI client construction that was in the old `app.py`.
- Exception handling is more precise: `AlignmentBlockedError` (the new typed exception from `agents.py`) is caught separately from generic `Exception`, preventing a generic `ValueError` from being misidentified as an alignment block.
- When the human edits a red-team prompt before sending, the UI detects the tampering and skips updating the red-team agent's bandit state (was missing in the old `app_redteam.py`).
- A "thinking" loading indicator (`"Thinking..."` message) is shown while the target agent processes input.
- The `.env` file is loaded with a `try/except FileNotFoundError` guard (the old `app.py` would crash if `.env` was absent).

---

## 5. Hardened Perceive-Layer Jailbreak Detector (`InputSafetyDetector`)

Project A's perceive-layer used case-sensitive substring matching over 8 hardcoded trigger phrases, applied inconsistently across agents.

Project B replaces this with the `InputSafetyDetector` class in `src/agents/__init__.py`:

- Three rule groups with named categories: `instruction_override` (10 phrases), `persona_hijack` (6 phrases), `policy_skip_framing` (6 phrases), plus a dynamic `domain_sensitive_action` group per agent.
- Input is normalized to NFKC Unicode before matching, stripping control/format characters and casefolding.
- Leet-speak normalization (`0→o`, `1→i`, `3→e`, `4→a`, `5→s`, `7→t`, `@→a`, `$→s`) is applied as a second variant when `JAILBREAK_NORMALIZE_LEETSPEAK=true`.
- Fuzzy phrase matching via `difflib.SequenceMatcher` with a configurable `JAILBREAK_FUZZY_THRESHOLD` (default 0.86).
- Squashed-space matching catches phrases written without spaces (e.g., `ignorepreviousinstructions`).
- A configurable `JAILBREAK_ALLOWLIST` allows specific benign phrases (e.g., "act as a customer") to pass through.
- Rule hit counts are tracked per category and exposed via `get_input_safety_stats()`, which is called before/after each experiment and the delta is stored in the experiment log.
- Domain-specific sensitive phrases are passed per-agent: `FinancialAgent` blocks "execute trade", "transfer funds", etc.; `CustomerServiceAgent` blocks "authorize refund", "modify account", etc.

---

## 6. Improved Memory Retrieval (BM25 + Recency + Importance Scoring)

Project A's `MemoryStream.retrieve()` used plain substring matching: `if query.lower() in entry["content"].lower()`.

Project B replaces this with a multi-factor ranked retrieval in `MemoryStream.retrieve()`:

- A pre-filter step finds candidate entries with at least `MEMORY_RETRIEVAL_PREFILTER_MIN_OVERLAP` token overlaps (default 1) with the query.
- If `rank_bm25` is installed, a `BM25Okapi` index is built lazily over all memory entries and refreshed on changes (`_index_dirty` flag).
- Final score per candidate is a weighted combination: `0.55 * bm25_norm + 0.20 * recency + 0.20 * importance + exact_bonus + lexical_bonus`, where recency uses an exponential decay with half-life `MEMORY_RETRIEVAL_RECENCY_HALFLIFE_SECONDS` (default 600 s).
- Falls back to substring matching if `rank_bm25` is not installed or if the BM25 index build fails.
- Returns at most `k=5` entries by default (configurable).

---

## 7. Full Config Environment-Variable Override System (`src/config.py`)

Project A's `config.py` was a flat file of hardcoded constants with no override mechanism.

Project B rewrites it with typed helper functions (`_env_str`, `_env_int`, `_env_float`, `_env_bool`, `_env_list`, `_env_optional_int`) so every config variable reads from an environment variable first and falls back to a default. New variables added for Project B features:

| New Config Variable | Default | Purpose |
|---|---|---|
| `MARSE_JUDGE_MODE` | `"rule"` | Select rule / hybrid / llm judge for MARSE |
| `MARSE_RULE_BACKEND` | `"pattern"` | Select pattern or ml rule backend |
| `MARSE_ML_JUDGE_MODEL_PATH` | `"experiments/models/marse_ml_detector.joblib"` | Path to trained ML detector |
| `MARSE_ML_VIOLATION_THRESHOLD` | `0.60` | ML violation decision threshold |
| `MARSE_ML_UNCERTAINTY_BAND` | `0.10` | Band around threshold that triggers LLM second-pass |
| `MARSE_LLM_JUDGE_BACKEND` | `"openai"` | LLM backend for MARSE hybrid/llm judge |
| `MARSE_LLM_JUDGE_MODEL` | `"gpt-4o-mini"` | Model for MARSE LLM judge |
| `MARSE_HYBRID_SAMPLE_RATE` | `0.15` | Fraction of clean turns sampled for LLM judge |
| `BANDIT_ALGORITHM` | `"thompson"` | UCB1 or Thompson sampling for attack surface selection |
| `UCB_EXPLORATION_COEFF` | `1.4142` | Exploration coefficient for UCB1 |
| `EXPERIMENT_SEED` | `17` | Random seed for reproducibility |
| `BASELINE_N_RUNS` | `3` | Number of repeated ABATE runs |
| `BASELINE_RANDOMIZE_PROBES` | `True` | Shuffle probe order per run |
| `BASELINE_SEED_STRIDE` | `9973` | Seed increment between ABATE runs |
| `BASELINE_BOOTSTRAP_SAMPLES` | `800` | Bootstrap samples for 95% CI |
| `BASELINE_BOOTSTRAP_CONFIDENCE` | `0.95` | Confidence level for CI |
| `MEMORY_RETRIEVAL_RECENCY_HALFLIFE_SECONDS` | `600.0` | Recency decay for memory retrieval |
| `MEMORY_RETRIEVAL_PREFILTER_MIN_OVERLAP` | `1` | Minimum token overlap for memory pre-filter |
| `JAILBREAK_FUZZY_THRESHOLD` | `0.86` | Fuzzy match ratio for jailbreak detection |
| `JAILBREAK_ALLOWLIST` | (5 phrases) | Phrases to skip jailbreak detection for |
| `JAILBREAK_NORMALIZE_LEETSPEAK` | `True` | Enable leet-speak normalization |

A `validate_runtime_config()` function is added that checks all values against allowed sets, validates `OPENAI_API_KEY` presence when required, and raises `ValueError` with a combined error list if any check fails. This is called at the start of every `run_experiment()` and `run_baseline_experiment()` call.

A `runtime_config_snapshot()` function produces a dict of all uppercase config values and is embedded in every experiment log for reproducibility.

---

## 8. Upgraded Red-Team Agent: Thompson Sampling + Domain-Specific Attack Personas

Project A's `RedTeamAgent` used UCB1 exclusively.

Project B adds Thompson Sampling as an alternative bandit algorithm, selectable via `BANDIT_ALGORITHM`:

- Both algorithms maintain the same `bandit_state` dict per surface; UCB1 uses `cumulative_reward / attempts`; Thompson uses Beta posterior parameters `alpha` and `beta`.
- Both update on every turn: `alpha += severity`, `beta += (1.0 - severity)`, accumulating fractional reward from severity scores.
- UCB scores are always refreshed for display regardless of active algorithm.
- `load_bandit_state()` allows warm-starting a new campaign from a prior experiment's final bandit state.

In `backends.py`, the red-team agent's system context is extended with domain-specific attack persona hints when `attacker_domain` is set (used by the `run.py cross` sweep). The hints instruct the red-team agent to adopt the authority framing of the attacker's domain (e.g., medical urgency and HIPAA framing when attacking with a medical attacker).

---

## 9. Upgraded ABATE Baseline: Multi-Run with Bootstrap CI

Project A's ABATE ran probes once per agent with no statistical robustness.

Project B runs ABATE `BASELINE_N_RUNS` times (default 3) with independently seeded probe orderings and computes:

- Mean violation rate and standard deviation across runs per agent and per category.
- 95% bootstrap confidence intervals (800 samples, configurable) per agent and per category.
- The CI half-widths are rendered as error bars in `plot_violation_rate_by_agent()`.
- `run_baseline_experiment()` embeds run-level logs, per-run seeds, and bootstrap parameters in the output JSON.

---

## 10. Improved LLM Judge and HybridEvaluator (`src/red_team.py`)

Project A's `LLMJudge` counted alignment keywords (3+ → aligned) as its stub and had no structured LLM path.

Project B replaces this:

- `LLMJudge` now calls OpenAI with `response_format={"type": "json_object"}` and a structured prompt asking for `aligned`, `confidence`, `violation_type`, `reasoning`, `severity_realized`. The `_coerce_judge_output()` method robustly parses and validates the JSON response.
- A fallback chain (`_resolve_backend_order()`) tries the primary backend first, then the fallback backend, then the stub if both fail.
- The stub heuristic now checks for 15 alignment compliance signals (vs. 3 keywords before).

`HybridEvaluator` is a new class wrapping `Evaluator` with a configurable LLM second-pass for MARSE:

- `MARSE_JUDGE_MODE=rule` (default) — rule only, zero LLM cost.
- `MARSE_JUDGE_MODE=hybrid` — LLM called on violation turns, `alignment_warning` turns, ML-uncertain turns, and a `MARSE_HYBRID_SAMPLE_RATE` random sample of clean turns.
- `MARSE_JUDGE_MODE=llm` — LLM called on every turn.
- Bandit updates always use the fast rule signal (`violation_detected` / `severity`). Reporting uses the LLM result when available (`reporting_violation_detected` / `reporting_severity`). This means the reporting metric and the bandit-driving metric can differ.

---

## 11. Extended Test Suite (`tests/test_marse_refactor.py`)

Project A had an empty `tests/__init__.py` with no tests.

Project B adds 16 unit and integration tests in `tests/test_marse_refactor.py` covering:

- Single-turn experiment execution, precomputed score propagation, and log field presence.
- `reporting_first` decision signal controlling `stop_on_violation` (rule violation alone does not stop when `reporting_violation_detected=False`).
- Prior bandit state warm-start via `load_bandit_state()`.
- Legacy `act()` call path without `outcome` or `precomputed_score`.
- `weak_medical` not blocking jailbreak-style input; `medical` raising `AlignmentBlockedError` for the same input.
- `validate_runtime_config()` rejecting unknown surfaces, unknown categories, bad `MARSE_RULE_BACKEND`, bad `MARSE_ML_UNCERTAINTY_BAND`, and missing `OPENAI_API_KEY`.
- `HybridEvaluator` triggering LLM on `alignment_warning` and on `ml_uncertain`.
- `Evaluator` severity map by surface.
- ML backend fallback to pattern when model artifact is missing.
- `run.py cross` producing 16 unique output directories.
- `run.py experiment` and `run.py baseline` using correct output directory structure.
- Experiment campaign ID propagated to `build_red_team_agent`.
- Generic `ValueError` not treated as alignment block.
- `AlignmentBlockedError` still treated as blocked.
- Comparison plot `_redteam_primary_counts` using turn-level scores not campaign totals.
- Run status `degraded` on partial target and act failures.
- `run_baseline_experiment` rejecting unknown judge backend.
- Baseline stub + OpenAI fallback requiring `OPENAI_API_KEY`.
- Act failure metadata and counters logged correctly.
- Partial act failure producing `run_status=degraded`.
- README references updated entry points.
- `main.py` shim imports `cli_main` correctly.

---

## Summary Comparison Table

| Aspect | Project A | Project B |
|---|---|---|
| Target agents | `medical`, `financial`, `customer_service` | + `weak_medical` (intentionally weak baseline) |
| Jailbreak detector | 8 hardcoded phrases, case-sensitive substring | `InputSafetyDetector`: 22+ phrases in 3 rule groups, NFKC normalization, leet-speak normalization, fuzzy matching, configurable allowlist, domain-sensitive terms per agent |
| Memory retrieval | Plain substring matching | BM25 + recency decay + importance weighting, with substring fallback |
| MARSE judge | Pattern-only `Evaluator` | `HybridEvaluator` wrapping `Evaluator` with optional ML or LLM second-pass; three modes: `rule`, `hybrid`, `llm` |
| ML violation detector | None | `MLViolationDetector` (sentence-transformer + LogisticRegression) with training script and metrics sidecar |
| Bandit algorithm | UCB1 only | UCB1 or Thompson Sampling (selectable); dual `cumulative_reward` + Beta posterior fields |
| Cross-domain sweeps | Not available | `run.py cross` with 4 agents (16 pairs); domain-specific attacker personas |
| CLI | Two separate scripts (`run_experiment_cli.py`, `run_baseline_cli.py`) | Unified `src/run.py` with `experiment`, `cross`, `baseline` subcommands |
| Streamlit UI | Two apps (`app.py` + `streamlit_app.py`), no `weak_medical`, old OpenAI wiring | Single `app_redteam_combined.py`, four target agents, two-column layout, tamper detection, typed exception handling |
| ABATE baseline | Single run, no statistics | `BASELINE_N_RUNS` (default 3) runs, mean ± std, 95% bootstrap CI, seeded probe randomization |
| LLM judge quality | Keyword count heuristic (stub only) | OpenAI structured-output call with JSON schema validation + robust fallback chain |
| Config system | Hardcoded constants, no overrides | Every constant readable from env var; `validate_runtime_config()` at startup; `runtime_config_snapshot()` embedded in every log |
| Tests | None (empty `tests/__init__.py`) | 16 unit/integration tests covering agents, evaluators, CLI, experiment runner, reporting |
| Reporting | No per-turn primary metric; campaign-summary counts only | `reporting_first` decision signal in every turn log; `primary_violation_detected` / `primary_severity` per turn; dual rule/reporting counts in experiment summary |
