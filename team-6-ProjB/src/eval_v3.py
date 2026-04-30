"""
CardioAgent Ablation v4 — Expanded MIMIC-IV-ECG (330K samples)
================================================================
Extends eval_v3 to support the full MIMIC-IV-ECG dataset alongside
Symile-MIMIC, using planner_v3 with DeepRare-inspired features.

Four ablation settings:

  A: Qwen3-VL only         — CXR image zero-shot (baseline)
  B: + ECG + LingShu        — v2 pipeline (no RAG, no reflection)
  C: + RAG                  — v2 + RAG retrieval from unified index
  D: + Reflection + Pheno   — v3 full pipeline (DeepRare-inspired)

Fixes over previous version:
  - [FIX #1] MIMIC4ECGTestData.get_ground_truth() added
  - [FIX #2] evaluate_predictions() now computes AUROC, AUPRC, specificity
  - [FIX #3] evaluate_predictions() is actually called after inference
  - [FIX #4] Symile test set path added for ablation D
  - [FIX #5] print_metrics() and _metrics.json output added

Usage:
    python eval_v4.py --ablation D --data-source symile --max-samples 100
    python eval_v4.py --ablation D --data-source mimic4ecg --max-samples 500
    python eval_v4.py --ablation D --data-source both --max-samples 200
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ──

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Edema", "Lung Opacity",
    "No Finding", "Pleural Effusion",
]

ECG_DIAGNOSES = [
    "Sinus Rhythm", "Sinus Tachycardia", "Sinus Bradycardia",
    "Atrial Fibrillation", "Atrial Flutter",
    "Left Bundle Branch Block", "Right Bundle Branch Block",
    "Left Ventricular Hypertrophy", "Right Ventricular Hypertrophy",
    "ST Elevation", "ST Depression", "T Wave Inversion",
    "First Degree AV Block", "Prolonged QT",
    "Left Axis Deviation", "Right Axis Deviation",
    "Left Atrial Abnormality", "Right Atrial Abnormality",
    "Normal ECG", "Abnormal ECG",
]

DATA_DIR_SYMILE = "/scratch/rzv4ve/cardioagent/data/symile-mimic"
MIMIC_CXR_JPG_DIR = "/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg"
DATA_DIR_MIMIC4 = "/scratch/rzv4ve/cardioagent/data/mimic-iv-ecg"
DB_DIR = "/scratch/rzv4ve/cardioagent/results_db"
RESULTS_DIR = "/scratch/rzv4ve/cardioagent/ablation_results_v4"
QWEN_URL = "http://localhost:8000/v1"
LINGSHU_URL = "http://localhost:8001/v1"

PATHOLOGY_KEYWORDS = {
    "Atelectasis": ["atelectasis", "lung collapse", "volume loss"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement",
                     "enlarged cardiac silhouette"],
    "Edema": ["edema", "pulmonary edema", "pulmonary congestion", "fluid overload"],
    "Lung Opacity": ["lung opacity", "opacity", "consolidation", "infiltrate",
                     "opacification", "ground glass"],
    "No Finding": ["no finding", "no abnormality", "normal", "unremarkable",
                   "no acute", "clear lungs", "within normal"],
    "Pleural Effusion": ["pleural effusion", "effusion", "pleural fluid",
                         "costophrenic blunting"],
}

JSON_OUTPUT_INSTRUCTION = (
    "IMPORTANT: End your response with a JSON block like: "
    '{"predictions": {"Atelectasis": 0 or 1, "Cardiomegaly": 0 or 1, '
    '"Edema": 0 or 1, "Lung Opacity": 0 or 1, "No Finding": 0 or 1, '
    '"Pleural Effusion": 0 or 1}}'
)


# ═══════════════════════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════════════════════

class SymileTestData:
    """[FIX #4] Symile-MIMIC test data — was entirely missing from v4."""

    def __init__(self, data_dir: str = DATA_DIR_SYMILE, split: str = "test"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.npy_dir = self.data_dir / "data_npy" / split

        logger.info(f"Loading Symile-MIMIC {split} data...")
        self.ecg = np.load(self.npy_dir / f"ecg_{split}.npy", mmap_mode="r")
        self.cxr = np.load(self.npy_dir / f"cxr_{split}.npy", mmap_mode="r")
        self.hadm_ids = np.load(self.npy_dir / f"hadm_id_{split}.npy")
        self.labs_pct = np.load(self.npy_dir / f"labs_percentiles_{split}.npy", mmap_mode="r")
        self.labs_miss = np.load(self.npy_dir / f"labs_missingness_{split}.npy", mmap_mode="r")
        self.csv = pd.read_csv(self.data_dir / f"{split}.csv")
        self.n = self.ecg.shape[0]
        logger.info(f"Loaded {self.n} samples")

        try:
            from symile_adapter import MIMIC_LAB_NAMES
            self._lab_names = MIMIC_LAB_NAMES
        except ImportError:
            self._lab_names = {}

    def get_ground_truth(self, idx: int) -> dict:
        row = self.csv.iloc[idx]
        return {p: row.get(p, np.nan) for p in PATHOLOGIES}

    def get_ecg(self, idx: int) -> np.ndarray:
        return self.ecg[idx].squeeze(0).T.astype(np.float32)

    def get_cxr_png_path(self, idx: int, cache_dir: str) -> str:
        path = os.path.join(cache_dir, f"cxr_eval_{idx}.png")
        if os.path.exists(path):
            return path
        from PIL import Image
        cxr = self.cxr[idx]
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        cxr = np.clip((cxr * std + mean) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(cxr).save(path)
        return path

    def get_cxr_best_path(self, idx: int, cache_dir: str) -> tuple:
        row = self.csv.iloc[idx]
        cxr_rel = row.get("cxr_path", "")
        if pd.notna(cxr_rel) and cxr_rel:
            orig = os.path.join(MIMIC_CXR_JPG_DIR, str(cxr_rel))
            if os.path.exists(orig):
                return orig, "original"
        return self.get_cxr_png_path(idx, cache_dir), "npy_recovered"

    def get_lab_dict(self, idx: int) -> Optional[dict]:
        miss = self.labs_miss[idx]
        row = self.csv.iloc[idx]
        values = {}
        lab_cols = [c for c in self.csv.columns if c.startswith("5")]
        for j, col in enumerate(lab_cols):
            if j < len(miss) and miss[j] == 1:
                val = row.get(col, np.nan)
                if pd.notna(val):
                    values[self._lab_names.get(col, col)] = float(val)
        return values if values else None

    def get_lab_text(self, idx: int) -> Optional[str]:
        d = self.get_lab_dict(idx)
        return "\n".join(f"{k}: {v}" for k, v in d.items()) if d else None


class MIMIC4ECGTestData:
    """[FIX #1] Added get_ground_truth() for ECG-specific evaluation."""

    def __init__(self, data_dir: str, index_path: str, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.records = []
        with open(index_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("split") == "test":
                        self.records.append(r)
                        if max_samples and len(self.records) >= max_samples:
                            break
        self.n = len(self.records)
        logger.info(f"MIMIC-IV-ECG test set: {self.n} samples")

    def __len__(self):
        return self.n

    def get_record(self, idx: int) -> dict:
        return self.records[idx]

    def get_ecg_path(self, idx: int) -> Optional[str]:
        r = self.records[idx]
        wf_path = r.get("waveform_path", "")
        if wf_path:
            full_path = self.data_dir / wf_path
            if full_path.with_suffix(".hea").exists():
                return str(full_path)
        return None

    def get_ground_truth(self, idx: int) -> dict:
        """
        [FIX #1] Extract ground truth ECG diagnoses from indexed machine reports.
        Returns: {diagnosis: 1.0 or 0.0} for each ECG_DIAGNOSES label.
        """
        record = self.records[idx]
        findings = record.get("ecg_findings", [])
        machine_report = record.get("machine_report", "").lower()

        gt = {}
        finding_names = {f.get("finding", "").lower() for f in findings}

        for dx in ECG_DIAGNOSES:
            dx_lower = dx.lower()
            matched = any(dx_lower in fn for fn in finding_names)
            if not matched and machine_report:
                matched = dx_lower in machine_report
            gt[dx] = 1.0 if matched else 0.0

        return gt


# ═══════════════════════════════════════════════════════════════════════
# Prediction Extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_predictions(text: str, label_set: list = None) -> dict:
    if label_set is None:
        label_set = PATHOLOGIES

    preds = {p: 0 for p in label_set}

    # Try JSON extraction
    try:
        import re
        json_match = re.search(r'\{[^{}]*"predictions"[^{}]*\{([^{}]*)\}[^{}]*\}', text)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if "predictions" in parsed:
                for p in label_set:
                    if p in parsed["predictions"]:
                        preds[p] = int(parsed["predictions"][p])
                return preds
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: keyword matching
    text_lower = text.lower()
    if label_set is PATHOLOGIES or label_set == PATHOLOGIES:
        for pathology, keywords in PATHOLOGY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    preds[pathology] = 1
                    break
        if any(preds[p] == 1 for p in PATHOLOGIES if p != "No Finding"):
            preds["No Finding"] = 0
    else:
        for dx in label_set:
            if dx.lower() in text_lower:
                preds[dx] = 1

    return preds


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Metrics  [FIX #2]
# ═══════════════════════════════════════════════════════════════════════

def evaluate_predictions(
    ground_truth: List[dict],
    predictions: List[dict],
    label_set: List[str] = None,
) -> dict:
    """
    [FIX #2] Full metrics: AUROC, AUPRC, F1, Precision, Recall, Specificity.
    Previous version imported roc_auc_score/average_precision_score but never used them.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score,
    )

    if label_set is None:
        label_set = PATHOLOGIES

    results = {"per_label": {}, "overall": {}}
    all_y_true = []
    all_y_pred = []

    for label in label_set:
        y_true = []
        y_pred = []

        for gt, pred in zip(ground_truth, predictions):
            gt_val = gt.get(label, np.nan)
            if pd.isna(gt_val):
                continue
            y_true.append(1 if gt_val == 1.0 else 0)
            y_pred.append(pred.get(label, 0))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_samples = len(y_true)
        n_pos = int(y_true.sum())
        n_neg = n_samples - n_pos

        lr = {
            "n_evaluated": n_samples,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "prevalence": round(n_pos / max(n_samples, 1), 4),
        }

        if n_samples < 5 or n_pos == 0 or n_neg == 0:
            logger.warning(f"  {label}: skip (n={n_samples}, pos={n_pos}, neg={n_neg})")
            lr.update({"auroc": None, "auprc": None, "f1": None,
                       "precision": None, "recall": None, "specificity": None})
        else:
            try:
                lr["auroc"] = round(roc_auc_score(y_true, y_pred), 4)
            except ValueError:
                lr["auroc"] = None
            try:
                lr["auprc"] = round(average_precision_score(y_true, y_pred), 4)
            except ValueError:
                lr["auprc"] = None

            lr["f1"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
            lr["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
            lr["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 4)

            tn = int(((1 - y_true) * (1 - y_pred)).sum())
            fp = int(((1 - y_true) * y_pred).sum())
            lr["specificity"] = round(tn / max(tn + fp, 1), 4)

        results["per_label"][label] = lr
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Overall
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    aurocs = [r["auroc"] for r in results["per_label"].values() if r.get("auroc") is not None]
    auprcs = [r["auprc"] for r in results["per_label"].values() if r.get("auprc") is not None]
    f1s = [r["f1"] for r in results["per_label"].values() if r.get("f1") is not None]
    recs = [r["recall"] for r in results["per_label"].values() if r.get("recall") is not None]
    precs = [r["precision"] for r in results["per_label"].values() if r.get("precision") is not None]

    results["overall"]["macro_auroc"] = round(np.mean(aurocs), 4) if aurocs else None
    results["overall"]["macro_auprc"] = round(np.mean(auprcs), 4) if auprcs else None
    results["overall"]["macro_f1"] = round(np.mean(f1s), 4) if f1s else None
    results["overall"]["macro_precision"] = round(np.mean(precs), 4) if precs else None
    results["overall"]["macro_recall"] = round(np.mean(recs), 4) if recs else None

    if len(all_y_true) > 0 and len(np.unique(all_y_true)) > 1:
        try:
            results["overall"]["micro_auroc"] = round(roc_auc_score(all_y_true, all_y_pred), 4)
        except ValueError:
            results["overall"]["micro_auroc"] = None
        results["overall"]["micro_f1"] = round(
            f1_score(all_y_true, all_y_pred, average="micro", zero_division=0), 4)

    # Hamming loss
    errs = []
    for gt, pred in zip(ground_truth, predictions):
        e, c = 0, 0
        for lbl in label_set:
            gv = gt.get(lbl, np.nan)
            if pd.isna(gv):
                continue
            c += 1
            if (1 if gv == 1.0 else 0) != pred.get(lbl, 0):
                e += 1
        if c > 0:
            errs.append(e / c)
    results["overall"]["hamming_loss"] = round(np.mean(errs), 4) if errs else None

    # Exact match
    ex, tot = 0, 0
    for gt, pred in zip(ground_truth, predictions):
        ok, has = True, False
        for lbl in label_set:
            gv = gt.get(lbl, np.nan)
            if pd.isna(gv):
                continue
            has = True
            if (1 if gv == 1.0 else 0) != pred.get(lbl, 0):
                ok = False
                break
        if has:
            tot += 1
            if ok:
                ex += 1
    results["overall"]["exact_match"] = round(ex / max(tot, 1), 4)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Results Printing  [FIX #5]
# ═══════════════════════════════════════════════════════════════════════

def print_metrics(name: str, metrics: dict, label_set: List[str] = None):
    if label_set is None:
        label_set = PATHOLOGIES

    print(f"\n{'='*80}")
    print(f"  ABLATION {name}")
    print(f"{'='*80}")

    meta = metrics.get("meta", {})
    if meta:
        print(f"  Samples: {meta.get('n_succeeded', '?')}/{meta.get('n_total', '?')} "
              f"({meta.get('n_failed', 0)} failed), {meta.get('total_time_s', '?')}s")

    header = (f"{'Label':<30} {'N':>5} {'Prev':>6} {'AUROC':>7} {'AUPRC':>7} "
              f"{'F1':>7} {'Prec':>7} {'Rec':>7} {'Spec':>7}")
    print(f"\n{header}")
    print("-" * len(header))

    for label in label_set:
        r = metrics["per_label"].get(label, {})
        n = r.get("n_evaluated", 0)
        prev = r.get("prevalence", 0)
        fmt = lambda v: f"{v:.4f}" if v is not None else "   N/A"
        print(f"{label:<30} {n:>5} {prev:>6.3f} {fmt(r.get('auroc')):>7} "
              f"{fmt(r.get('auprc')):>7} {fmt(r.get('f1')):>7} "
              f"{fmt(r.get('precision')):>7} {fmt(r.get('recall')):>7} "
              f"{fmt(r.get('specificity')):>7}")

    o = metrics.get("overall", {})
    print(f"\n  Macro AUROC:     {o.get('macro_auroc', 'N/A')}")
    print(f"  Macro AUPRC:     {o.get('macro_auprc', 'N/A')}")
    print(f"  Macro F1:        {o.get('macro_f1', 'N/A')}")
    print(f"  Micro AUROC:     {o.get('micro_auroc', 'N/A')}")
    print(f"  Micro F1:        {o.get('micro_f1', 'N/A')}")
    print(f"  Hamming Loss:    {o.get('hamming_loss', 'N/A')}")
    print(f"  Exact Match:     {o.get('exact_match', 'N/A')}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Ablation D Runner
# ═══════════════════════════════════════════════════════════════════════

class AblationD:
    name = "D_deeprare_inspired"

    def __init__(self, qwen_url=QWEN_URL, lingshu_url=LINGSHU_URL, db_dir=DB_DIR):
        from planner import CardioAgentPlannerV3, PatientData as PD

        rag_paths = [os.path.join(db_dir, f) for f in [
            "unified_index.jsonl", "mimic_iv_ecg_index.jsonl",
            "symile_index.jsonl", "index.jsonl",
        ]]
        rag_paths = [p for p in rag_paths if os.path.exists(p)]

        self.planner = CardioAgentPlannerV3(
            qwen_api_url=qwen_url, lingshu_api_url=lingshu_url,
            rag_index_paths=rag_paths, enable_reflection=True,
            max_reflection_rounds=2,
        )
        self.PatientData = PD

    def predict_symile(self, data, idx, cache_dir):
        ecg = data.get_ecg(idx)
        ecg_path = os.path.join(cache_dir, f"ecg_eval_{idx}.npy")
        np.save(ecg_path, ecg)
        cxr_path, cxr_source = data.get_cxr_best_path(idx, cache_dir)
        lab_dict = data.get_lab_dict(idx)
        lab_text = data.get_lab_text(idx)

        state = self.planner.run(self.PatientData(
            patient_id=f"eval_{idx}", ecg_path=ecg_path,
            image_path=cxr_path, image_type="cxr",
            lab_results=lab_dict, lab_text=lab_text if not lab_dict else None,
            clinical_notes=JSON_OUTPUT_INSTRUCTION,
        ))
        preds = extract_predictions(state.final_report, PATHOLOGIES)
        return {
            "predictions": preds, "raw_text": state.final_report,
            "cxr_source": cxr_source, "elapsed_s": state.elapsed_s,
            "hypothesis_ranking": [{"rank": h.rank, "dx": h.diagnosis, "conf": h.confidence}
                                   for h in state.hypothesis_ranking[:5]],
            "phenotypes": [p["term"] for p in state.phenotype_terms],
            "reflection_rounds": state.reflection_rounds,
            "n_rag_cases": len(state.rag_cases),
        }

    def predict_mimic4(self, test_data, idx, cache_dir):
        ecg_path = test_data.get_ecg_path(idx)
        record = test_data.get_record(idx)

        state = self.planner.run(self.PatientData(
            patient_id=record.get("patient_id", f"mimic4_{idx}"),
            ecg_path=ecg_path,
            clinical_notes=f"Machine ECG report: {record.get('machine_report', '')}\n{JSON_OUTPUT_INSTRUCTION}",
        ))
        cxr_preds = extract_predictions(state.final_report, PATHOLOGIES)
        ecg_preds = extract_predictions(state.final_report, ECG_DIAGNOSES)
        return {
            "predictions": cxr_preds, "ecg_predictions": ecg_preds,
            "raw_text": state.final_report, "elapsed_s": state.elapsed_s,
            "hypothesis_ranking": [{"rank": h.rank, "dx": h.diagnosis, "conf": h.confidence}
                                   for h in state.hypothesis_ranking[:5]],
            "phenotypes": [p["term"] for p in state.phenotype_terms],
            "reflection_rounds": state.reflection_rounds,
            "n_rag_cases": len(state.rag_cases),
        }


# ═══════════════════════════════════════════════════════════════════════
# Run Ablation  [FIX #3]
# ═══════════════════════════════════════════════════════════════════════

def run_ablation(runner, data, max_samples, results_dir, ablation_name,
                 data_type="symile"):
    """
    [FIX #3] Unified runner that collects predictions AND evaluates.
    Previous version saved predictions but never called evaluate_predictions().
    """
    cache_dir = os.path.join(results_dir, f"cache_{ablation_name}")
    os.makedirs(cache_dir, exist_ok=True)

    is_symile = (data_type == "symile")
    label_set = PATHOLOGIES if is_symile else ECG_DIAGNOSES
    n = min(data.n, max_samples) if max_samples else data.n

    logger.info(f"\n{'='*60}")
    logger.info(f"Ablation {ablation_name}: {n} samples ({data_type})")
    logger.info(f"Labels: {len(label_set)} ({data_type})")
    logger.info(f"{'='*60}")

    ground_truth = []
    predictions = []
    raw_outputs = []
    t_start = time.time()
    succeeded, failed = 0, 0

    for idx in range(n):
        try:
            gt = data.get_ground_truth(idx)
            ground_truth.append(gt)

            t0 = time.time()
            if is_symile:
                result = runner.predict_symile(data, idx, cache_dir)
                preds = result["predictions"]
            else:
                result = runner.predict_mimic4(data, idx, cache_dir)
                preds = result.get("ecg_predictions", {})
            elapsed = time.time() - t0

            predictions.append(preds)
            raw_outputs.append({
                "index": idx,
                "ground_truth": {k: v for k, v in gt.items() if pd.notna(v)},
                "predictions": preds,
                "hypothesis_ranking": result.get("hypothesis_ranking", []),
                "phenotypes": result.get("phenotypes", []),
                "reflection_rounds": result.get("reflection_rounds", 0),
                "n_rag_cases": result.get("n_rag_cases", 0),
                "raw_text": result.get("raw_text", "")[:500],
                "inference_time_s": round(elapsed, 2),
            })

            succeeded += 1
            if succeeded % 10 == 0:
                te = time.time() - t_start
                rate = succeeded / te
                logger.info(f"  [{ablation_name}] {succeeded}/{n} "
                            f"({rate:.2f}/s, ETA {(n-succeeded)/rate:.0f}s, {failed} fail)")

        except Exception as e:
            failed += 1
            ground_truth.append(data.get_ground_truth(idx))
            predictions.append({lbl: 0 for lbl in label_set})
            if failed <= 5:
                logger.warning(f"  Sample {idx} failed: {e}")

    total_time = time.time() - t_start

    # ── [FIX #3] Actually call evaluate_predictions ──
    metrics = evaluate_predictions(ground_truth, predictions, label_set)
    metrics["meta"] = {
        "ablation": ablation_name, "data_source": data_type,
        "label_type": "cxr_pathologies" if is_symile else "ecg_diagnoses",
        "n_total": n, "n_succeeded": succeeded, "n_failed": failed,
        "total_time_s": round(total_time, 1),
        "avg_time_per_sample_s": round(total_time / max(n, 1), 2),
    }

    # Save predictions
    preds_path = os.path.join(results_dir, f"{ablation_name}_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(raw_outputs, f, indent=2, default=str)

    # Save metrics  [FIX #5]
    metrics_path = os.path.join(results_dir, f"{ablation_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print results  [FIX #5]
    print_metrics(ablation_name, metrics, label_set)
    logger.info(f"Saved: {preds_path}")
    logger.info(f"Saved: {metrics_path}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# CLI  [FIX #3 + #4 + #5]
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CardioAgent Ablation v4")
    parser.add_argument("--ablation", nargs="+", default=["D"],
                        choices=["A", "B", "C", "D", "all"])
    parser.add_argument("--data-source", default="symile",
                        choices=["symile", "mimic4ecg", "both"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--data-dir-symile", default=DATA_DIR_SYMILE)
    parser.add_argument("--data-dir-mimic4", default=DATA_DIR_MIMIC4)
    parser.add_argument("--db-dir", default=DB_DIR)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--qwen-url", default=QWEN_URL)
    parser.add_argument("--lingshu-url", default=LINGSHU_URL)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ablations = ["A", "B", "C", "D"] if "all" in args.ablation else args.ablation
    all_metrics = {}

    for abl in ablations:
        if abl in ["A", "B", "C"]:
            logger.info(f"Ablation {abl}: use eval_v3.py")
            continue

        runner = AblationD(args.qwen_url, args.lingshu_url, args.db_dir)

        # [FIX #4] Symile path
        if args.data_source in ["symile", "both"]:
            try:
                sdata = SymileTestData(args.data_dir_symile)
                m = run_ablation(runner, sdata, args.max_samples,
                                 args.output_dir, "D_symile", "symile")
                all_metrics["D_symile"] = m
            except Exception as e:
                logger.error(f"Symile failed: {e}", exc_info=True)

        if args.data_source in ["mimic4ecg", "both"]:
            idx_path = os.path.join(args.db_dir, "mimic_iv_ecg_index.jsonl")
            if os.path.exists(idx_path):
                try:
                    mdata = MIMIC4ECGTestData(args.data_dir_mimic4, idx_path,
                                              args.max_samples)
                    m = run_ablation(runner, mdata, args.max_samples,
                                     args.output_dir, "D_mimic4ecg", "mimic4ecg")
                    all_metrics["D_mimic4ecg"] = m
                except Exception as e:
                    logger.error(f"MIMIC-IV-ECG failed: {e}", exc_info=True)
            else:
                logger.error(f"Index not found: {idx_path}")
                logger.error("Run mimic_iv_ecg_preprocess.py --mode index-meta first")

    # Summary
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*70}")
        h = f"{'Run':<20} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'EM':>8}"
        print(h)
        print("-" * len(h))
        for name, m in all_metrics.items():
            o = m.get("overall", {})
            f = lambda v: f"{v:.4f}" if v is not None else "   N/A"
            print(f"{name:<20} {f(o.get('macro_auroc')):>8} "
                  f"{f(o.get('macro_auprc')):>8} {f(o.get('macro_f1')):>8} "
                  f"{f(o.get('exact_match')):>8}")

    summary_path = os.path.join(args.output_dir, "all_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()