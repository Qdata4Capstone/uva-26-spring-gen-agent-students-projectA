# MedRAX — Project A to Project B Change Summary

**Team 07:** Mengmeng Ma · Kathleen O'Donovan
**Research Question (Project B):** *Are Medical Agents Reliable Co-workers for Radiologists?*

---

## What Was Project A (Original MedRAX)

Project A delivered the core MedRAX framework: a conversational medical AI agent for chest
X-ray analysis built on top of OpenAI GPT-4o and a LangGraph state machine. Its main
contributions were:

- A modular **10-tool architecture** covering classification, segmentation, report
  generation, phrase grounding, GradCAM explainability, DICOM processing, image generation
  (Roentgen diffusion), LLaVA-Med VQA, CheXagent VQA, and image visualization.
- A LangGraph **Agent** class (`medrax/agent/agent.py`) with a
  `process → execute_tools → process` loop, timestamped JSON tool-call logging, and
  LangChain `MemorySaver` for conversation state.
- A **Gradio web UI** (`interface_v2.py`) with custom CSS, image upload, streaming chat,
  and per-tool result rendering.
- **ChestAgentBench** — a 2,500+ question multiple-choice benchmark across 8 diagnostic
  categories, located in `src/experiments/benchmark/`.
- Benchmark runner scripts in `src/experiments/` for GPT-4o (`benchmark_gpt4o.py`),
  Llama 3.2 Vision 90B (`benchmark_llama.py`), CheXagent (`benchmark_chexagent.py`), and
  LLaVA-Med (`benchmark_llavamed.py`), each comparing accuracy on ChestAgentBench.
- Result analysis utilities: `analyze_axes.py`, `compare_runs.py`, `inspect_logs.py`,
  `validate_logs.py`.
- A system prompt defining a multi-step clinical workflow
  (`medrax/docs/system_prompts.txt`).

---

## Changes Introduced in Project B

### 1. New Research Focus and Evaluation Benchmark (Eurorad Clinical Cases)

Project B shifts from multiple-choice accuracy on ChestAgentBench to open-ended
**clinical reasoning quality** on real radiology case reports from the
[Eurorad](https://www.eurorad.org) database.

**New file: `src/build_clinical_benchmark.py`**

Converts `eurorad_dataset/extracted_preview.json` (scraped via `src/data/get_cases.py`)
into a structured JSONL benchmark file (`eurorad_clinical_benchmark.jsonl`). Each record
contains: `case_id`, `title`, `patient_age`, `patient_gender`, `clinical_history`,
`images` (local file paths), `image_descriptions`, `gt_imaging_findings`,
`gt_differentials`, and `gt_final_diagnosis`. A `--min-images` flag filters cases lacking
local images.

This is a new data asset that did not exist in Project A. The benchmark uses genuine
radiology case records rather than the synthetic multiple-choice format of
ChestAgentBench.

---

### 2. New Evaluation Pipeline v1 — Four-Step Clinical Reasoning (`eval_clinical.py`)

**New file: `src/eval_clinical.py`** (633 lines)

A two-mode pipeline (baseline and agent) that forces the model through a **four-step
structured clinical workflow** per case:

1. History intake — extract key clinical facts
2. Imaging analysis — findings from image plus optional tool calls
3. Differential diagnosis — ranked list of up to 5 candidates
4. Final impression — single primary diagnosis with confidence level

**Agent mode** adds a two-phase approach: Phase 1 invokes the LangGraph agent freely to
call imaging tools; Phase 2 runs GPT to synthesize a structured JSON response enriched
with the tool outputs.

**New scoring dimensions not present in Project A:**

| Metric | What it measures |
|--------|-----------------|
| `imaging_findings` | Semantic match of agent findings to Eurorad ground truth |
| `diagnosis_match` | Semantic equivalence of final diagnosis to ground truth |
| `consistency` | Whether stated findings logically support the stated diagnosis |
| `diff_top1 / diff_top3` | Whether ground-truth diagnosis appears in differentials |
| `tool_recall` | Fraction of expected imaging tools actually called |
| `communication_quality` | Radiological terminology, structure, specificity |
| `reasoning_coherence` | Integration of history, findings, and differentials |
| `uncertainty_appropriate` | Calibration of expressed confidence vs. actual correctness |

All scores are computed by a GPT-4o judge. A **Reliable Co-worker (RCW) score** aggregates
communication quality, reasoning coherence, and uncertainty appropriateness. Results are
written to JSONL log files.

---

### 3. New Evaluation Pipeline v2 — Tool-Grounded Self-Reflection (`eval_clinical_v2.py`)

**New file: `src/eval_clinical_v2.py`** (851 lines)

The key conceptual novelty of Project B. Tools are called **after** an initial diagnosis
(to falsify it), not before (as data sources). Three structured self-reflection patterns
are applied:

- **Pattern A — Targeted Hypothesis Testing:** the classifier is called after the
  preliminary diagnosis to challenge it; the agent checks whether the top classifier score
  supports or contradicts the diagnosis.
- **Pattern B — Spatial Grounding via GradCAM:** the heatmap is used to verify whether
  the model's stated finding locations match the image regions the classifier attended to.
- **Pattern C — Cross-Tool Consistency Check:** the report generator provides an
  independent narrative; mismatches between its findings and the model's findings trigger
  explicit reconciliation.

**Three-pass pipeline per case:**

1. Pass 1 — Vision-only synthesis (GPT from images plus history, no tools).
2. Pass 2 — Targeted tool calls (LangChain agent calls all three tools to challenge the
   Pass 1 diagnosis).
3. Pass 3 — Reconciliation (GPT receives the initial diagnosis plus per-tool evidence and
   must address each contradiction, revising any field).

This is a fundamentally different evaluation philosophy from both ChestAgentBench and
eval_clinical.py v1.

---

### 4. New Evaluation Pipeline v3 — Pre/Post Reflection (`eval_clinical_v3.py`)

**New file: `src/eval_clinical_v3.py`** (764 lines)

A streamlined three-phase pipeline combining v1 tool gathering with a final
self-verification pass:

- Phase 1 (agent only) — Tool gathering, identical to eval_clinical.py.
- Phase 2 (both modes) — Synthesis; initial clinical JSON recorded as `pre_reflection`.
- Phase 3 (agent only) — Self-verification: model receives its Phase 2 JSON plus tool
  outputs (no images) and checks for contradictions; revised output recorded as
  `post_reflection`.

Non-agent mode runs Phase 2 only, serving as the pure baseline. The dual
`pre_reflection` / `post_reflection` output format enables direct measurement of how much
the self-verification pass changes the agent's conclusions.

---

### 5. New Uncertainty Calibration Evaluation (`eval_unknown.py`)

**New file: `src/eval_unknown.py`**

Tests whether the agent correctly expresses uncertainty when critical clinical information
is deliberately withheld. Each Eurorad case is evaluated under four variants:

| Variant | What is withheld |
|---------|-----------------|
| `complete` | Nothing (original case) |
| `no_image` | Image removed; history only |
| `no_history` | Clinical history removed; image only |
| `no_both` | Both image and history removed |

New metrics introduced:

- `overconfidence_rate` — fraction of cases rated "high" confidence (dangerous when
  information is missing)
- `uncertainty_rate` — fraction rated "low" or "medium" confidence (appropriate)
- `hedging_rate` — fraction with explicit uncertainty language in the output
- `confidence_drop` — fraction where confidence fell relative to the complete variant

---

### 6. New Result Aggregation and Comparison Tool (`analyze_results.py`)

**New file: `src/analyze_results.py`** (309 lines)

Reads all clinical eval and unknown eval JSONL output files and prints two final
comparison tables suitable for the Project B report:

- **Table 1 — Communication Quality:** rows are model × mode (baseline/agent); columns
  are Findings, Dx@1, Dx@3, Dx Score, Consistency, RCW.
- **Table 2 — Acknowledge Unknown:** rows are model × variant; columns are Dx Score,
  High%, Low/Med%, Hedged%, Conf Drop%.

Handles both old and new JSONL formats via normalization. This replaces the per-benchmark
`analyze_axes.py` approach from Project A with a unified cross-experiment table.

---

### 7. New Chainlit UI (`interface_v3.py`)

**New file: `src/interface_v3.py`** (300 lines)

A fully async Chainlit-based chat interface running alongside the existing Gradio UI.
Key differences from `interface_v2.py`:

- Built on `chainlit` instead of `gradio`; uses `@cl.on_app_startup`, `@cl.on_message`,
  and `@cl.step` decorators.
- Displays each tool call as a labeled step with emoji icons per tool name (e.g.,
  "📊 Chest X-ray Classifier").
- Async message streaming; Chainlit manages its own agent initialization lifecycle via
  environment variables passed from `main.py`.
- Launched via `python main.py --ui chainlit`, which calls `subprocess.run` with
  `chainlit run interface_v3.py`.

The original Gradio UI remains the default (`python main.py`).

---

### 8. Updated Entry Point (`main.py`) — Dual UI Support

`main.py` was extended to accept a `--ui {gradio,chainlit}` argument. In Chainlit mode,
it passes all configuration through environment variables (`MODEL_DIR`, `TEMP_DIR`,
`PROMPT_FILE`, `OPENAI_MODEL`, `DEVICE`) and launches Chainlit in a subprocess. All
hardcoded absolute paths from the original version were replaced with environment-variable
lookups with sensible defaults relative to the script directory.

---

### 9. Multi-Model Support for Clinical Evaluations

The eval scripts and batch runners add support for **Qwen3-VL** via a local vLLM server
in addition to GPT-4o. The `OPENAI_BASE_URL` environment variable redirects the OpenAI
client to the local vLLM endpoint while a separate `JUDGE_OPENAI_API_KEY` always routes
the GPT-4o judge to real OpenAI. This enables cost-effective evaluation of open-weight
models without changing the evaluation script code.

**New benchmark scripts added to `src/experiments/` that also support the agent mode:**
- `benchmark_gpt4o.py` — extended to accept `--use-agent` flag for ChestAgentBench
- `benchmark_chexagent.py`, `benchmark_llama.py`, `benchmark_llavamed.py` — model-specific
  runners, present in Project A but extended in Project B with tool-process scoring
  (tool recall, tool precision, process score = recall × precision)

---

### 10. Tool Annotation Utility (`annotate_tools.py`)

**New file: `src/annotate_tools.py`**

Adds `expected_tools` and `tool_required` fields to ChestAgentBench JSONL via:
1. A fast rule-based pass using the question's `categories` field.
2. A GPT-4o verification pass for ambiguous cases (skippable with `--rules-only`).

This retroactively augments the Project A benchmark to support tool-process scoring, which
requires knowing which tools should ideally be called for each question.

---

### 11. Batch Shell Scripts

**New files: `run_eval.bash`, `run_eval_cinical.bash`, `run_qwen3vl.bash`,
`run_qwen_full_eval.bash`, `run_qwen_v2_eval.bash`, `run_bench_build.bash`,
`run_annotate_tool.bash`, `run_all_evals.bash`, `run_eval_unknown.bash`**

`run_all_evals.bash` is the primary Project B runner. It executes all six experiments in
sequence: GPT-4o baseline, GPT-4o agent, Qwen3 baseline, Qwen3 agent, GPT-4o unknown
eval, Qwen3 unknown eval — then runs `analyze_results.py` to produce the final comparison
tables. The script uses `set -e` for error propagation and reads `OPENAI_API_KEY` from
the environment.

Project A had no equivalent top-level experiment orchestration script.

---

## Summary Comparison Table

| Dimension | Project A | Project B |
|-----------|-----------|-----------|
| **Primary benchmark** | ChestAgentBench (2,500+ multiple-choice Qs, 8 categories) | Eurorad clinical case bank (open-ended, real radiology reports) |
| **Evaluation style** | Multiple-choice accuracy (letter match) | Structured clinical reasoning scored by GPT-4o judge across 8 dimensions |
| **Agent mode** | Direct GPT call or MedRAX agent for ChestAgentBench | Three eval variants: baseline, v1 (tool-gather-then-synthesize), v2 (3-pass self-reflection), v3 (pre/post reflection) |
| **Self-reflection** | None | v2 (3-pass: vision-only → targeted tool falsification → reconciliation) and v3 (pre/post reflection) |
| **Uncertainty evaluation** | None | `eval_unknown.py` with 4 input-ablation variants; measures overconfidence, hedging, confidence drop |
| **UI** | Gradio only (`interface_v2.py`) | Gradio (default) + Chainlit (`interface_v3.py`) via `--ui chainlit` flag |
| **Multi-model support** | GPT-4o, Llama 3.2 Vision 90B, CheXagent, LLaVA-Med | Adds Qwen3-VL via local vLLM; separate judge vs. inference API keys |
| **Result aggregation** | Per-model `analyze_axes.py` + `compare_runs.py` | Unified `analyze_results.py` producing cross-model, cross-mode comparison tables |
| **Tool-process scoring** | None | Tool recall, tool precision, process score (R×P) on ChestAgentBench; tool recall on Eurorad |
| **Benchmark augmentation** | Raw ChestAgentBench JSONL | `annotate_tools.py` adds `expected_tools` and `tool_required` per question |
| **Configuration** | Hardcoded paths in `main.py` | All paths and keys via environment variables with sensible defaults |
| **Experiment orchestration** | Manual per-script runs | `run_all_evals.bash` runs all 6 experiments + analysis end-to-end |
| **Tests** | Empty `tests/__init__.py` | Still no tests (identified as P1 gap in `IMPROVE.md`) |
