# Step-by-Step Demo Run

End-to-end walkthrough for reproducing CardioAgent. The flow is:

1. Set up the environment.
2. Preprocess MIMIC-IV-ECG into the JSONL index.
3. Build the multi-modal vector index in ChromaDB.
4. Start the model servers (Qwen3-VL + Lingshu).
5. Run **Block 1** ablations (A / B / C — pipeline study, 200 samples).
6. Run **Block 2** ablations (D1–D5 — modality study, 2,000 samples).
7. Print the comparison table.
8. (Optional) Launch the Streamlit demo for live, transparent inference.

All commands assume `$PWD = team-x-ProjB/` and:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Concrete data paths used in our runs match the constants at the top of
`src/eval/eval_v5.py` and `src/ablations/ablation_modality.py`. Replace
`/scratch/rzv4ve/cardioagent/...` with your own root if running elsewhere.

---

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # core: torch, transformers, chromadb,
                                       # streamlit, pandas, scikit-learn,
                                       # vllm, openai, wfdb
```

The ECG encoder requires **fairseq-signals** (not vendored — install separately):

```bash
git clone https://github.com/Jwoo5/fairseq-signals.git
cd fairseq-signals && pip install -e . && cd ..
```

Required model weights / checkpoints:

| Model        | Path used in code                                           |
| ------------ | ----------------------------------------------------------- |
| Qwen3-VL     | served via vLLM (see §4)                                    |
| Lingshu      | served via vLLM (see §4)                                    |
| ECG-FM       | `tools/ECG/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt`     |
| BioViL-T     | downloaded from HF on first use                             |
| BiomedBERT   | `tools/Biomedbert/blobs/` (HF snapshot)                     |
| MedCPT       | `ncbi/MedCPT-Cross-Encoder` (HF, used in D5)                |

---

## 1. Preprocess MIMIC-IV-ECG

Build the JSONL index that downstream tools read:

```bash
python src/preprocess/mimic_ecg_preprocess.py \
    --data-dir /path/to/MIMIC-IV-ECG-1 \
    --out-dir  /path/to/results_db
```

**Output:** `results_db/mimic_iv_ecg_index.jsonl` — one JSON record per ECG with
`patient_id`, `subject_id`, `study_id`, `ecg_path`, `ecg_time`, `report`,
`split`.

---

## 2. Build the vector index (ChromaDB)

`src/memory/vec_memory_new.py` exposes three subcommands. The unified MIMIC-IV
index is the main one; `build-cxr` is for the CXR-only collection used by D2.

### 2a. Unified ECG + CXR + text index from MIMIC-IV (primary index)

```bash
python src/memory/vec_memory_new.py build-train \
    --ecg-index    /path/to/results_db/mimic_iv_ecg_index.jsonl \
    --ecg-data-dir /path/to/MIMIC-IV-ECG-1 \
    --cxr-data-dir /path/to/mimic-cxr-jpg-2.0.0.physionet.org \
    --cxr-metadata /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
    --db-path      /path/to/vector_db_multi \
    --ecg-model    /path/to/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt \
    --cxr-model    biovilt \
    --text-model   /path/to/Biomedbert/blobs \
    --batch-size   256 \
    --device       cuda
```

This populates three ChromaDB collections inside `vector_db_multi/`:

- `ecg_embeddings`   — ECG-FM vectors
- `cxr_embeddings`   — BioViL-T vectors
- `text_embeddings`  — BiomedBERT vectors over reports / labs

Records are joined across modalities by `subject_id`. ECG ↔ CXR are
temporally matched: for each ECG study we pick the patient's CXR closest in
time, preferring frontal views (PA > AP > lateral).

### 2b. (Optional) CXR-only collection from MIMIC-CXR-JPG + CheXpert

Use this if you want a denser CXR index than the ECG-aligned subset above:

```bash
python src/memory/vec_memory_new.py build-cxr \
    --cxr-data-dir    /path/to/mimic-cxr-jpg-2.0.0.physionet.org \
    --cxr-metadata    /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
    --chexpert-labels /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
    --cxr-split       /path/to/mimic-cxr-2.0.0-split.csv.gz \
    --db-path         /path/to/vector_db_multi \
    --frontal-only \
    --batch-size      64 \
    --device          cuda
```

### 2c. (Optional) Symile train split as an extra RAG source

```bash
python src/memory/vec_memory_new.py build-symile \
    --data-dir /path/to/symile-mimic \
    --db-path  /path/to/vector_db_multi \
    --ecg-model /path/to/ecg-fm/...pt \
    --cxr-model biovilt \
    --text-model /path/to/Biomedbert/blobs
```

A wrapper shell script that calls `build-train` with our exact flags lives at
`src/scripts/vec_memory.sh`.

---

## 3. Start the model servers

CardioAgent talks to two OpenAI-compatible endpoints:

| Server         | Default port | Source file                      |
| -------------- | -----------: | -------------------------------- |
| Qwen3-VL       |       `8000` | served via `vllm serve`          |
| Lingshu (med)  |       `8001` | `src/apps/lingshu_server.py`     |

Open two terminals.

**Terminal 1 — Qwen3-VL (vision-language):**

```bash
vllm serve Qwen/Qwen3-VL-Chat \
    --port 8000 --tensor-parallel-size 1 --dtype bfloat16
```

**Terminal 2 — Lingshu medical LLM:**

```bash
python src/apps/lingshu_server.py --port 8001
```

Wait until both report `ready`. Sanity check:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
```

---

## 4. Block 1 — Pipeline ablation (A / B / C)

Driven by `src/eval/eval_v5.py`. Each setting writes
`<name>_predictions.json` and `<name>_metrics.json` into `--output-dir`.

```bash
python src/eval/eval_v5.py \
    --ablation A B C \
    --data-source symile \
    --max-samples 200 \
    --data-dir-symile /path/to/symile-mimic \
    --mimic-cxr-dir   /path/to/mimic-cxr-jpg \
    --db-dir          /path/to/results_db \
    --output-dir      ablation_results_v5 \
    --qwen-url    http://localhost:8000/v1 \
    --lingshu-url http://localhost:8001/v1
```

Settings:

| Flag      | Description                                                 |
| :-------- | ----------------------------------------------------------- |
| **`A`**   | Qwen3-VL zero-shot — CXR + labs only, no tools, no RAG       |
| **`B`**   | DeepRare planner + ECG/CXR tools (phenotype + reflection)   |
| **`C`**   | DeepRare + naïve keyword RAG over MIMIC-IV-ECG (367 K)      |

Optionally evaluate on MIMIC-IV-ECG diagnoses too: `--data-source both`.

---

## 5. Block 2 — Vector-RAG modality ablation (D1–D5)

Driven by `src/ablations/ablation_modality.py`. Run all five settings
sequentially:

```bash
python src/ablations/ablation_modality.py \
    --retrieval-mode all \
    --data-source symile \
    --max-samples 2000 \
    --vector-db-path /path/to/vector_db_multi \
    --data-dir-symile /path/to/symile-mimic \
    --output-dir ablation_results_v5_cxr_rag \
    --qwen-url    http://localhost:8000/v1 \
    --lingshu-url http://localhost:8002/v1 \
    --device cuda
```

Modes:

| Mode         | Retrieval                                                        |
| :----------- | ---------------------------------------------------------------- |
| **`D1`**     | ECG-FM only                                                      |
| **`D2`**     | BioViL-T (CXR) only                                              |
| **`D3`**     | BiomedBERT (text) only                                           |
| **`D4`**     | All three modalities + RRF fusion                                |
| **`D5`**     | All three modalities + RRF + **MedCPT cross-encoder reranker**   |
| **`D-rand`** | Counterfactual: random cases injected as RAG context             |
| **`D-shuf`** | Counterfactual: shuffled embeddings                              |
| **`all`**    | D1 → D5 sequentially                                             |
| **`full`**   | D1 → D5 + counterfactuals                                        |

Per-modality shell wrappers also exist:

```bash
bash src/scripts/ablation_ecg_rag.sh        # D1
bash src/scripts/ablation_cxr_rag.sh        # D2
bash src/scripts/ablation_text_rag.sh       # D3
bash src/scripts/ablation_full_fusion.sh    # D4 (and/or D5)
```

---

## 6. Compare results

Reads every `*_metrics.json` in `--output-dir` and prints the comparison table:

```bash
python src/eval/eval_v5.py --mode compare \
    --output-dir ablation_results_v5
```

You should see Block 1 numbers matching the table in
[`results.md`](results.md). Run the same command pointed at
`ablation_results_v5_cxr_rag` for Block 2.

---

## 7. (Optional) Streamlit demo — `app_dis.py`

The interactive UI streams every phase of the agent (planning → tool execution
→ phenotype → hypothesis → vector retrieval → reflection → synthesis) and
exposes the retrieved cases with their per-modality scores.

```bash
streamlit run src/apps/app_dis.py --server.port 8501
```

Open `http://localhost:8501`. Upload:

- An **ECG** file (`.hea`/`.npy`/WFDB triple),
- A **CXR** image (`.jpg`/`.png`), and / or
- **Lab values** (JSON or CSV).

Click **Run**. You'll see each step appear with status, duration, and any
intermediate output. The "Retrieved cases" panel shows the top-k cases from
the vector index with per-modality similarity scores.

The simpler single-page variant is `src/apps/app.py`.

---

## End-to-end smoke test (≤ 5 min)

If you only want to verify the whole stack works:

```bash
# Servers up (§3), index built (§2a, even on --max-records 1000)
python src/eval/eval_v5.py --ablation B \
    --data-source symile --max-samples 5 \
    --output-dir /tmp/smoke
cat /tmp/smoke/B_*_metrics.json
```

If `B_..._metrics.json` exists with non-trivial `macro_auroc`, the pipeline is
wired up correctly.
