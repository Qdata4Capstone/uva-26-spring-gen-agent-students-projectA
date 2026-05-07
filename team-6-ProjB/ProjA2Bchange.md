# Project A to Project B: Changes in CardioRAG-CX / CardioAgent (Team 6)

## What Project A Was

**CardioRAG-CX** (Project A) is a multimodal cardiac diagnostic agent that accepts ECG files, DICOM/MRI images, and free-text clinical notes and produces a structured diagnostic report. It runs entirely on self-hosted models (no external API keys). The architecture in Project A is a three-step linear pipeline driven by a single orchestrator class (`CardioAgentPlanner` in `src/planner.py`):

1. **ECGFounderTool** — reads raw ECG files in five formats (WFDB, EDF, CSV, NumPy, GE MUSE XML), runs NeuroKit2 signal analysis, generates a 12-lead waveform image, and optionally runs the ECGFounder 150-class classifier (disabled via `NotImplementedError` in the submitted code).
2. **LingShuTool** — converts DICOM/MRI files to PNG and queries the LingShu-8B medical vision-language model on a local vLLM server (port 8001).
3. **Qwen3-VL 32B** — multimodal synthesis backbone served by llama.cpp (port 8000) that ingests all tool outputs and images to produce the final diagnosis.

The Streamlit UI (`src/app.py`) streams each `ThinkingStep` in real time. When the GPU model servers are unavailable the system degrades to a plain text fallback report. The `tests/` directory contains only `__init__.py`; there are no automated tests.

---

## 1. Agent Pipeline Architecture: From 3 Steps to 6 Phases (DeepRare-Inspired)

**Project A** uses a flat 3-step pipeline: Tool → Tool → Synthesize.

**Project B** restructures this into a DeepRare-style 6-phase agentic planner (`src/planner_v4.py`, class `CardioAgentPlannerV4`):

| Phase | Name | What It Does |
|------:|------|--------------|
| 1 | Tool execution | Detect available modalities; run `_run_ecg`, `_run_image`, `_run_labs`, `_run_notes` tools in parallel |
| 2 | Phenotype extraction | `PhenotypeExtractor.extract(state)` surfaces structured clinical terms (e.g., "tachycardia", "ST elevation") from all tool outputs |
| 3 | Hypothesis generation | `HypothesisGenerator.generate(...)` produces a ranked differential diagnosis list with confidence scores and evidence links |
| 4 | RAG retrieval | Vector-based retrieval (new in v4) with keyword fallback |
| 5 | Reflection loop | `ReflectionEngine.run(...)` re-scores hypotheses against retrieved cases, checks cross-modal consistency, tracks agreement/contradiction counts |
| 6 | Final synthesis | `_run_synthesis(...)` assembles a final structured report with a fenced JSON output block |

`CardioAgentPlannerV4` inherits from `CardioAgentPlannerV3` (a DeepRare-style base) and overrides only Phase 4. Phases 1–3 and 5–6 are inherited unchanged, making v4 strictly additive.

The `AgentState` data model also evolved: Project A's `AgentState` tracked only `ecg_result`, `mri_result`, `planner_output`, `final_report`, and `images`. Project B's `AgentStateV3` additionally carries `modalities_available`, `phenotype_terms`, `hypothesis_ranking`, `rag_cases`, `cross_modal_consistency`, `reflection_rounds`, `reflection_log`, and `lab_result`.

---

## 2. New Retrieval Pathway: Embedding-Based Vector RAG with Multi-Modal Fusion

This is the largest single addition in Project B.

**Project A** has no retrieval system at all — the synthesizer works entirely from the current patient's data.

**Project B** introduces a full vector retrieval subsystem (`src/vec_memory.py`, class `VectorCaseMemory`) backed by ChromaDB with three separate persistent collections:

| Collection | Encoder | Embedding Dim | Source Data |
|------------|---------|--------------|-------------|
| `ecg_embeddings` | ECG-FM (fairseq_signals) | 768 | MIMIC-IV-ECG 12-lead signals |
| `cxr_embeddings` | BioViL-T (hi-ml-multimodal) | 512 | MIMIC-CXR-JPG frontal images |
| `text_embeddings` | BiomedBERT (microsoft/BiomedNLP) | 768 | Clinical notes, lab text, radiology reports |

**Online retrieval flow:** for each modality present in the query, the system embeds the input using the corresponding encoder, queries the ChromaDB collection for `per_modality_k=10` candidates, then fuses results across modalities using one of two strategies:

- **Reciprocal Rank Fusion (RRF)** (default): `RRF(d) = Σ 1/(60 + rank_m(d))` — parameter-free, robust to heterogeneous score scales.
- **Weighted score averaging**: weights `ecg=0.4, cxr=0.4, text=0.2`.

Each retrieved case is returned as a `RetrievedCase` dataclass carrying `case_id`, `source`, `diagnosis`, `report`, `score`, `modality_scores` (per-modality similarity), and `modalities_matched`.

**Offline index building** is supported via a CLI with three subcommands: `build-train` (unified ECG + CXR + text from MIMIC-IV), `build-cxr` (CXR-only from MIMIC-CXR-JPG + CheXpert labels), and `build-symile` (Symile-MIMIC train split). The `build-train` command implements ECG-to-CXR temporal matching: for each ECG record it selects the patient's CXR closest in acquisition time, preferring frontal views (PA > AP > lateral).

---

## 3. New Embedding Tools: Three Domain-Specific Encoders

**Project A** has no embedding models. Signal analysis uses NeuroKit2; image analysis delegates entirely to the LingShu API.

**Project B** adds `src/tools/embedding_tool.py` with three dedicated embedder classes:

- **`ECGFMEmbedder`**: loads an ECG-FM checkpoint via `fairseq_signals`. Includes robust loading logic that handles OmegaConf MISSING values, patches the `dataclasses.field` default factory, clears stale `sys.modules` entries, and falls back from `fairseq.checkpoint_utils` to direct model building. Preprocesses ECG to `(12, 5000)` at 500 Hz and mean-pools the transformer output to a 768-d vector.
- **`BioViLTEmbedder`**: wraps `health_multimodal.image.utils.get_image_inference()` from the `hi-ml-multimodal` package. Converts any image input (path, PIL Image, or NumPy array) to grayscale and extracts the `projected_global_embedding`.
- **`BiomedBERTEmbedder`**: loads `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` via HuggingFace Transformers and uses the CLS token embedding.

All three embedders expose `.embed(input)` → `(dim,)` and `.embed_batch(inputs)` → `(N, dim)`, both L2-normalized.

---

## 4. New Reranking Stage: MedCPT Cross-Encoder

**Project A** has no reranking.

**Project B** adds `src/tools/reranker.py` with two classes:

- **`MedCPTReranker`**: a two-stage retrieval pipeline. After RRF fusion returns `4 × top_k` candidates, the cross-encoder (default: `ncbi/MedCPT-Cross-Encoder`) jointly scores each `(query_text, candidate_report)` pair. The query is built from the planner's phenotype terms and top hypothesis rather than raw inputs, so the reranker sees the agent's evolving clinical intent. Results are trimmed to `rerank_top_n`. This is the D5 ablation setting.
- **`LLMRelevanceGate`** (optional): an additional filtering layer that uses a Qwen3-VL call to score each candidate's clinical relevance, keeping only those above a configurable threshold.

---

## 5. ECG Tool Redesigned: ECGTool v2 Replaces ECGFounderTool

**Project A** uses `ECGFounderTool` (`src/ecgfounder_tool.py`) which relies on NeuroKit2 for signal analysis and has a `_run_ecgfounder()` method that raises `NotImplementedError`.

**Project B** replaces this with `ECGTool` (`src/tools/ecg_tool_v2.py`) with these concrete changes:

- **Dependency**: drops NeuroKit2; uses only `scipy.signal` (`find_peaks`, `butter`, `filtfilt`) for R-peak detection and ST analysis. This eliminates the division-by-zero crashes documented in the IMPROVEMENT_PLAN.
- **Path fixing**: adds `_fix_path()` which correctly handles `.dat` files (strips extension to WFDB base), directory inputs (scans for first `.hea`), and ambiguous extensions.
- **NaN handling**: explicitly calls `np.nan_to_num(signal, nan=0.0)` after WFDB reads, addressing the MIMIC-IV data quality issue.
- **NumPy shape handling**: `_read_npy()` now handles `(1, 5000, 12)` Symile-MIMIC batch-dimension arrays and ambiguous `(samples, leads)` vs `(leads, samples)` shapes.
- **R-peak detection**: uses a threshold cascade (`[0.5, 0.3, 0.2, 0.1]`) with physiological validity filtering (200 ms < RR < 3000 ms) instead of NeuroKit2's `ecg_process`.
- **ST analysis**: uses `np.median` over beats instead of `np.mean` for robustness.
- **Findings field**: each finding now carries a `"modality": "ecg"` tag for downstream fusion.
- **Fallback metric**: adds `_basic_metrics()` which returns a safe stub when R-peak detection fails entirely, instead of raising.

The `ECGFMEmbedder` in `embedding_tool.py` is the separate retrieval-pathway encoder for the same ECG signal; these two tools serve distinct roles (diagnostic tool vs. vector index query).

---

## 6. Expanded Input Modalities: Labs and Clinical Notes Added

**Project A** accepts ECG files, DICOM/MRI images, and free-text clinical notes as a single string passed through to Qwen3-VL.

**Project B** separates labs from clinical notes as a distinct modality. `PatientData` (the v3/v4 input dataclass) includes:
- `ecg_path`: `.npy` or WFDB record prefix
- `image_path` + `image_type`: `"cxr"` (chest X-ray) or `"echo"`
- `lab_results`: structured dict `{itemid: {value, ref_range, ...}}`
- `lab_text`: free-text lab summary
- `clinical_notes`: free-text notes

Phase 1 routes each modality to its tool: `_run_labs()` and `_run_notes()` are new dedicated handlers that did not exist in Project A. Lab text is also used as a modality for text-embedding-based retrieval in Phase 4.

---

## 7. Formal Evaluation Suite with Ablation Framework

**Project A** has no evaluation scripts. The `tests/` folder is empty.

**Project B** adds a structured evaluation and ablation system:

- **`src/eval_v5.py`**: evaluates three pipeline ablation settings (A, B, C) against two benchmark datasets. Metrics computed per label and overall: AUROC, AUPRC, F1, Precision, Recall, Specificity, Macro/Micro AUROC, Hamming distance, Exact Match. Multi-stage prediction extraction: fenced JSON → regex pattern → keyword fallback, with method tracking.

- **`src/ablation_modality.py`**: evaluates seven vector-RAG ablation settings (D1–D5, D-rand, D-shuf):
  - D1: ECG-FM retrieval only
  - D2: BioViL-T (CXR) only
  - D3: BiomedBERT (text) only
  - D4: All modalities + RRF fusion
  - D5: All modalities + RRF + MedCPT cross-encoder reranker
  - D-rand / D-shuf: counterfactual controls (random cases / shuffled embeddings)

**Key empirical results** (from `doc/results.md`):
- DeepRare planner (B) lifts Macro F1 by +17.1 points over Qwen3-VL zero-shot (A).
- Keyword RAG (C) adds +0.77 F1 points over B.
- Single-modality ECG-FM retrieval (D1, Macro AUROC 0.5891) beats full RRF fusion (D4, 0.5829) — the headline finding motivating the MedCPT reranker (D5).

---

## 8. Streamlit UI: Enhanced Diagnostic Transparency

**Project A** (`src/app.py`) streams `ThinkingStep` objects in real time and displays ECG waveform and MRI montage images.

**Project B** (`src/app_dis.py`) adds:

- **Live phase visualization**: each of the six planner phases streams individually with status indicators.
- **RAG transparency panel**: shows retrieved cases with per-modality similarity scores and reranker scores side-by-side.
- **Diagnostic reasoning panel**: displays phenotype terms, ranked hypotheses with confidence, and reflection round details.
- **CXR anatomical annotation**: `CXR_REGIONS` dict maps LingShu textual location references (e.g., "left lower lobe", "cardiac silhouette") to bounding-box coordinates for overlay rendering.
- **Follow-up chat**: a conversational refinement interface allowing the physician to query Qwen3-VL about the specific case after the report is generated.

---

## 9. Documentation Expansion

**Project A** has a single `README.md` plus the externally authored `IMPROVEMENT_PLAN.md`.

**Project B** adds a `doc/` directory with three files:
- `doc/architecture.md`: full Mermaid flowchart, component table, data-flow phase-by-phase, design decisions (why per-modality embedders, why RRF, why two LLM servers), and an extension guide.
- `doc/results.md`: empirical results tables for both evaluation blocks with analysis and limitations.
- `doc/usage.md`: step-by-step operational guide.
- `IMPROVE.md`: Project-B-specific improvement backlog with prioritized items.

---

## Summary Comparison Table

| Dimension | Project A | Project B |
|-----------|-----------|-----------|
| **Pipeline structure** | 3-step linear (ECG tool → MRI tool → Qwen3-VL synthesize) | 6-phase DeepRare-style (Tools → Phenotypes → Hypotheses → Vector RAG → Reflection → Synthesis) |
| **Orchestrator class** | `CardioAgentPlanner` (planner.py) | `CardioAgentPlannerV4` (planner_v4.py), inherits from v3 |
| **ECG analysis tool** | `ECGFounderTool` — NeuroKit2 + ECGFounder (disabled, raises NotImplementedError) | `ECGTool` (ecg_tool_v2.py) — scipy-only, NaN-safe, robust path fixing, no unimplemented methods |
| **MRI/CXR analysis** | `LingShuTool` — DICOM→PNG→LingShu-8B (port 8001) | Same `LingShuTool` (unchanged interface) |
| **Lab results** | Passed as part of clinical notes string | Separate `lab_results` dict + `lab_text` field; dedicated `_run_labs()` phase |
| **RAG retrieval** | None | Vector RAG via ChromaDB: ECG-FM + BioViL-T + BiomedBERT with RRF/score fusion |
| **RAG knowledge base** | None | MIMIC-IV-ECG + MIMIC-CXR-JPG + Symile-MIMIC, offline index built via CLI |
| **Reranking** | None | Optional MedCPT cross-encoder (D5 ablation) + optional LLM relevance gate |
| **Phenotype extraction** | None — Qwen3-VL sees raw tool outputs directly | `PhenotypeExtractor` produces structured term list (term, source, evidence) |
| **Hypothesis generation** | None — single synthesis call | `HypothesisGenerator` produces ranked differential list with confidence + evidence links |
| **Reflection loop** | None | `ReflectionEngine` re-scores hypotheses against RAG cases, tracks cross-modal consistency |
| **Embedding models** | None | ECGFMEmbedder (768-d), BioViLTEmbedder (512-d), BiomedBERTEmbedder (768-d) |
| **Evaluation** | None (empty tests/) | `eval_v5.py` (A/B/C ablation) + `ablation_modality.py` (D1–D5, D-rand, D-shuf) |
| **Metrics** | None | AUROC, AUPRC, F1, Precision, Recall, Specificity, Hamming, Exact Match per label and macro/micro |
| **UI features** | Real-time ThinkingStep stream, ECG waveform, MRI montage | All of A, plus RAG scores panel, phenotype/hypothesis display, CXR region overlay, follow-up chat |
| **Documentation** | README.md + IMPROVEMENT_PLAN.md | doc/architecture.md, doc/results.md, doc/usage.md, IMPROVE.md |
| **Modality input** | ECG files, DICOM/MRI, clinical notes string | ECG, CXR/DICOM, structured lab results, lab text, clinical notes (five distinct modalities) |
| **State data model** | `AgentState` (7 fields) | `AgentStateV3` (14+ fields including phenotype_terms, hypothesis_ranking, rag_cases, reflection_log) |
| **Synthesis LLM** | Qwen3-VL 32B via llama.cpp (port 8000) | Same (port 8000); Lingshu also used for Phase 1 notes/lab interpretation and Phase 5 reflection |
