"""
CardioAgent Ablation v5 — DeepRare-Inspired Evaluation
========================================================
Unified evaluation for the planner_v3 architecture across three
ablation settings and two data sources.

Ablation settings:
  A: Qwen3-VL zero-shot — CXR image + labs, no tools, no DeepRare
  B: DeepRare pipeline   — tools + phenotypes + reflection, NO RAG
  C: DeepRare + RAG      — full pipeline with CaseMemory retrieval

Data sources:
  symile     — Symile-MIMIC test set (CXR pathology labels)
  mimic4ecg  — MIMIC-IV-ECG test set (ECG diagnosis labels)
  both       — Run on both datasets

Usage:
    # Single ablation on Symile
    python eval_v5.py --ablation A --data-source symile --max-samples 200

    # Full comparison (A vs B vs C) on Symile
    python eval_v5.py --ablation all --data-source symile --max-samples 100

    # DeepRare ± RAG on both datasets
    python eval_v5.py --ablation B C --data-source both --max-samples 100

    # Compare saved results
    python eval_v5.py --mode compare
"""

import os
import sys
import json
import time
import re
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

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

# ── Default paths (Rivanna) ──
DATA_DIR_SYMILE = "/scratch/rzv4ve/cardioagent/data/symile-mimic"
MIMIC_CXR_JPG_DIR = "/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg"
DATA_DIR_MIMIC4 = "/scratch/rzv4ve/cardioagent/data/mimic-iv-ecg"
DB_DIR = "/scratch/rzv4ve/cardioagent/results_db"
RESULTS_DIR = "/scratch/rzv4ve/cardioagent/ablation_results_v5"
QWEN_URL = "http://localhost:8000/v1"
LINGSHU_URL = "http://localhost:8001/v1"

# ── MIMIC-IV Lab Item ID → Human-Readable Name ──
MIMIC_LAB_NAMES = {
    "51221": "Hematocrit", "51265": "Platelet Count", "50912": "Creatinine",
    "50971": "Potassium", "51222": "Hemoglobin", "51301": "WBC",
    "51249": "MCHC", "51279": "RBC", "51250": "MCV", "51248": "MCH",
    "51277": "RDW", "51006": "BUN", "50983": "Sodium", "50902": "Chloride",
    "50882": "Bicarbonate", "50868": "Anion Gap", "50931": "Glucose",
    "50960": "Magnesium", "50893": "Calcium Total", "50970": "Phosphate",
    "51237": "INR", "51274": "PT", "51275": "PTT", "51146": "Basophils",
    "51256": "Neutrophils", "51254": "Monocytes", "51200": "Eosinophils",
    "51244": "Lymphocytes", "52172": "Specific Gravity (Urine)",
    "50934": "Lipase", "51678": "Troponin T", "50947": "Total Protein",
    "50861": "ALT", "50878": "AST", "50813": "Lactate",
    "50863": "Alkaline Phosphatase", "50885": "Bilirubin Total",
    "50820": "pH", "50862": "Albumin", "50802": "Base Excess",
    "50821": "pO2", "50804": "Calculated Total CO2", "50818": "pCO2",
    "52075": "Protein (Urine)", "52073": "Nitrite (Urine)",
    "52074": "pH (Urine)", "52069": "Ketone (Urine)",
    "51133": "Absolute Basophil Count", "50910": "CK-MB",
    "52135": "Glucose (Urine)",
}

# ── Reference Ranges for Abnormal Flagging ──
LAB_REFERENCE_RANGES = {
    "51221": (36, 50, "%"), "51222": (12.0, 17.5, "g/dL"),
    "50912": (0.6, 1.2, "mg/dL"), "50971": (3.5, 5.0, "mEq/L"),
    "50983": (136, 145, "mEq/L"), "51301": (4.5, 11.0, "K/uL"),
    "51265": (150, 400, "K/uL"), "51006": (7, 20, "mg/dL"),
    "50931": (70, 100, "mg/dL"), "51237": (0.8, 1.2, "ratio"),
    "51678": (0, 0.01, "ng/mL"), "50813": (0.5, 2.2, "mmol/L"),
    "50862": (3.5, 5.0, "g/dL"), "50820": (7.35, 7.45, ""),
    "50882": (22, 29, "mEq/L"), "50910": (0, 5, "ng/mL"),
    "50861": (7, 56, "U/L"), "50878": (10, 40, "U/L"),
    "50885": (0.1, 1.2, "mg/dL"), "50902": (98, 106, "mEq/L"),
    "50868": (8, 12, "mEq/L"), "50960": (1.7, 2.2, "mg/dL"),
    "50893": (8.5, 10.5, "mg/dL"), "50970": (2.5, 4.5, "mg/dL"),
    "50821": (80, 100, "mmHg"), "50818": (35, 45, "mmHg"),
    "50947": (6.0, 8.3, "g/dL"), "50863": (44, 147, "U/L"),
    "51279": (4.0, 5.5, "M/uL"), "51250": (80, 100, "fL"),
    "51248": (27, 33, "pg"), "51249": (32, 36, "g/dL"),
    "51277": (11.5, 14.5, "%"), "51274": (11, 13.5, "sec"),
    "51275": (25, 35, "sec"),
}

# ── Pathology keyword dictionaries for extraction fallback ──
PATHOLOGY_KEYWORDS = {
    "Atelectasis": [
        "atelectasis", "lung collapse", "basilar collapse", "volume loss",
        "subsegmental", "plate-like", "discoid opacity", "passive collapse",
    ],
    "Cardiomegaly": [
        "cardiomegaly", "enlarged heart", "cardiac enlargement",
        "enlarged cardiac", "cardiothoracic ratio",
        "biventricular enlargement", "ventricular dilation",
        "dilated heart", "cardiac dilation", "enlarged silhouette",
    ],
    "Edema": [
        "edema", "pulmonary edema", "congestion", "pulmonary congestion",
        "vascular congestion", "fluid overload", "volume overload",
        "interstitial thickening", "kerley", "perihilar haz",
        "cephalization", "alveolar edema",
    ],
    "Lung Opacity": [
        "lung opacity", "opacity", "consolidation", "infiltrate",
        "opacification", "airspace disease", "ground glass",
        "hazy opacity", "haziness", "parenchymal", "patchy opacity",
    ],
    "No Finding": [
        "no finding", "no abnormality", "normal", "unremarkable",
        "no acute", "clear lungs", "no significant", "within normal",
        "no evidence of", "negative study",
    ],
    "Pleural Effusion": [
        "pleural effusion", "effusion", "pleural fluid",
        "costophrenic blunting", "meniscus sign", "fluid collection",
        "layering fluid", "hydrothorax", "blunting",
    ],
}

JSON_OUTPUT_INSTRUCTION = (
    "\n\nIMPORTANT: End your response with EXACTLY this JSON block.\n"
    "Use 1 for POSITIVE and 0 for NEGATIVE:\n"
    '```json\n'
    '{"Atelectasis": 0, "Cardiomegaly": 0, "Edema": 0, '
    '"Lung Opacity": 0, "No Finding": 0, "Pleural Effusion": 0}\n'
    '```\n'
    "Replace 0 with 1 if that condition IS present."
)


# ═══════════════════════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════════════════════

class SymileTestData:
    """Symile-MIMIC test data loader for CXR pathology evaluation."""

    def __init__(self, data_dir: str = DATA_DIR_SYMILE, split: str = "test"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.npy_dir = self.data_dir / "data_npy" / split

        logger.info(f"Loading Symile-MIMIC {split} data...")
        self.ecg = np.load(self.npy_dir / f"ecg_{split}.npy", mmap_mode="r")
        self.cxr = np.load(self.npy_dir / f"cxr_{split}.npy", mmap_mode="r")
        self.hadm_ids = np.load(self.npy_dir / f"hadm_id_{split}.npy")
        self.labs_pct = np.load(
            self.npy_dir / f"labs_percentiles_{split}.npy", mmap_mode="r",
        )
        self.labs_miss = np.load(
            self.npy_dir / f"labs_missingness_{split}.npy", mmap_mode="r",
        )
        self.csv = pd.read_csv(self.data_dir / f"{split}.csv")
        self.n = self.ecg.shape[0]
        logger.info(f"  Loaded {self.n} samples")

    def get_ground_truth(self, idx: int) -> dict:
        row = self.csv.iloc[idx]
        return {p: row.get(p, np.nan) for p in PATHOLOGIES}

    def get_ecg(self, idx: int) -> np.ndarray:
        """(1, 5000, 12) → (12, 5000) float32"""
        return self.ecg[idx].squeeze(0).T.astype(np.float32)

    def get_cxr_png_path(self, idx: int, cache_dir: str) -> str:
        path = os.path.join(cache_dir, f"cxr_eval_{idx}.png")
        if os.path.exists(path):
            return path
        from PIL import Image
        cxr = self._denormalize_cxr(idx)
        Image.fromarray(cxr).save(path)
        return path

    def get_cxr_original_path(self, idx: int) -> Optional[str]:
        row = self.csv.iloc[idx]
        cxr_rel = row.get("cxr_path", "")
        if pd.isna(cxr_rel) or not cxr_rel:
            return None
        full = os.path.join(MIMIC_CXR_JPG_DIR, str(cxr_rel))
        return full if os.path.exists(full) else None

    def get_cxr_best_path(self, idx: int, cache_dir: str) -> tuple:
        orig = self.get_cxr_original_path(idx)
        if orig:
            return orig, "original"
        return self.get_cxr_png_path(idx, cache_dir), "npy_recovered"

    def get_cxr_best_base64(self, idx: int, cache_dir: str = None) -> tuple:
        import base64 as b64mod
        orig = self.get_cxr_original_path(idx)
        if orig:
            with open(orig, "rb") as f:
                return b64mod.b64encode(f.read()).decode(), "original"
        return self._get_cxr_base64(idx), "npy_recovered"

    def _get_cxr_base64(self, idx: int) -> str:
        if not hasattr(self, "_b64_cache"):
            self._b64_cache = {}
        if idx in self._b64_cache:
            return self._b64_cache[idx]
        import base64
        from io import BytesIO
        from PIL import Image

        cxr = self._denormalize_cxr(idx)
        buf = BytesIO()
        Image.fromarray(cxr).save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        self._b64_cache[idx] = b64
        return b64

    def _denormalize_cxr(self, idx: int) -> np.ndarray:
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        cxr = self.cxr[idx].copy()
        for c in range(3):
            cxr[c] = cxr[c] * STD[c] + MEAN[c]
        cxr = np.transpose(cxr, (1, 2, 0))
        return np.clip(cxr * 255, 0, 255).astype(np.uint8)

    def prebatch_cxr(self, indices: list, cache_dir: str = None):
        """Pre-convert CXR batch for faster inference."""
        import base64
        from io import BytesIO
        from PIL import Image

        if not hasattr(self, "_b64_cache"):
            self._b64_cache = {}

        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        logger.info(f"  Pre-converting {len(indices)} CXR images...")
        t0 = time.time()
        batch = self.cxr[indices].copy()
        batch = batch * STD[np.newaxis] + MEAN[np.newaxis]
        batch = np.clip(batch * 255, 0, 255).astype(np.uint8)
        batch = np.transpose(batch, (0, 2, 3, 1))

        for i, idx in enumerate(indices):
            buf = BytesIO()
            Image.fromarray(batch[i]).save(buf, format="JPEG", quality=85)
            self._b64_cache[idx] = base64.b64encode(buf.getvalue()).decode()
            if cache_dir:
                path = os.path.join(cache_dir, f"cxr_eval_{idx}.png")
                if not os.path.exists(path):
                    Image.fromarray(batch[i]).save(path)

        logger.info(f"  Pre-converted in {time.time()-t0:.1f}s")

    def get_lab_text(self, idx: int) -> str:
        row = self.csv.iloc[idx]
        parts = []
        abnormals = []
        for itemid, name in MIMIC_LAB_NAMES.items():
            val = row.get(itemid, np.nan)
            if pd.notna(val):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue
                if itemid in LAB_REFERENCE_RANGES:
                    low, high, unit = LAB_REFERENCE_RANGES[itemid]
                    if val < low:
                        parts.append(f"{name}: {val} {unit} (LOW, ref {low}-{high})")
                        abnormals.append(f"{name} low ({val})")
                    elif val > high:
                        parts.append(f"{name}: {val} {unit} (HIGH, ref {low}-{high})")
                        abnormals.append(f"{name} high ({val})")
                    else:
                        parts.append(f"{name}: {val} {unit}")
                else:
                    parts.append(f"{name}: {val}")
        result = ""
        if parts:
            result = "Lab Results: " + ", ".join(parts)
        if abnormals:
            result += f"\nABNORMAL LABS ({len(abnormals)}): " + "; ".join(abnormals)
        return result

    def get_lab_dict(self, idx: int) -> dict:
        row = self.csv.iloc[idx]
        labs = {}
        for itemid, name in MIMIC_LAB_NAMES.items():
            val = row.get(itemid, np.nan)
            if pd.notna(val):
                try:
                    labs[name] = float(val)
                except (ValueError, TypeError):
                    pass
        return labs


class MIMIC4ECGTestData:
    """MIMIC-IV-ECG test data loader for ECG diagnosis evaluation."""

    def __init__(
        self,
        data_dir: str,
        index_path: str,
        max_samples: Optional[int] = None,
    ):
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
        logger.info(f"  MIMIC-IV-ECG test set: {self.n} samples")

    def __len__(self):
        return self.n

    def get_record(self, idx: int) -> dict:
        return self.records[idx]

    def get_ecg_path(self, idx: int) -> Optional[str]:
        r = self.records[idx]
        wf = r.get("waveform_path", "")
        if wf:
            full = self.data_dir / wf
            if full.with_suffix(".hea").exists():
                return str(full)
        return None

    def get_ground_truth(self, idx: int) -> dict:
        record = self.records[idx]
        findings = record.get("ecg_findings", [])
        machine_report = record.get("machine_report", "").lower()
        finding_names = {f.get("finding", "").lower() for f in findings}
        gt = {}
        for dx in ECG_DIAGNOSES:
            dx_lower = dx.lower()
            matched = any(dx_lower in fn for fn in finding_names)
            if not matched and machine_report:
                matched = dx_lower in machine_report
            gt[dx] = 1.0 if matched else 0.0
        return gt


# ═══════════════════════════════════════════════════════════════════════
# Prediction Extraction (3-strategy: JSON → POSITIVE/NEGATIVE → keyword)
# ═══════════════════════════════════════════════════════════════════════

def extract_predictions(text: str, label_set: list = None) -> dict:
    """
    Extract predictions from free-text agent output.
    Priority: JSON block → POSITIVE/NEGATIVE patterns → keyword matching.
    """
    if label_set is None:
        label_set = PATHOLOGIES

    # Strategy 1: JSON block
    preds = _try_parse_json(text, label_set)
    if preds:
        preds["_extraction_method"] = "json"
        return preds

    # Strategy 2: POSITIVE/NEGATIVE patterns (CXR pathologies only)
    if label_set is PATHOLOGIES or label_set == PATHOLOGIES:
        preds = _try_parse_pos_neg(text, label_set)
        if preds:
            preds["_extraction_method"] = "pos_neg"
            return preds

    # Strategy 3: Keyword fallback
    preds = _keyword_fallback(text, label_set)
    preds["_extraction_method"] = "keyword"
    return preds


def _try_parse_json(text: str, label_set: list) -> Optional[dict]:
    """Try to extract JSON dict from model output."""
    # For CXR pathologies, look for known key
    first_label = label_set[0]
    patterns = [
        r'```json\s*(\{[^}]+\})\s*```',
        r'```\s*(\{[^}]+\})\s*```',
        rf'(\{{["\']?{re.escape(first_label)}["\']?[^}}]+\}})',
    ]
    # Also try the nested {"predictions": {...}} format from eval_v4
    nested = re.search(
        r'\{[^{}]*"predictions"[^{}]*\{([^{}]*)\}[^{}]*\}', text,
    )
    if nested:
        try:
            parsed = json.loads(nested.group(0))
            if "predictions" in parsed:
                result = {}
                for p in label_set:
                    if p in parsed["predictions"]:
                        result[p] = int(parsed["predictions"][p])
                    else:
                        result[p] = 0
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                raw = match.group(1).strip()
                parsed = json.loads(raw)
                matched = sum(1 for p in label_set if p in parsed)
                if matched >= min(3, len(label_set)):
                    result = {}
                    for p in label_set:
                        val = parsed.get(p, 0)
                        if isinstance(val, (int, float)):
                            result[p] = 1 if val >= 0.5 else 0
                        elif isinstance(val, str):
                            result[p] = 1 if val.lower() in (
                                "1", "positive", "yes", "true", "present",
                            ) else 0
                        elif isinstance(val, bool):
                            result[p] = 1 if val else 0
                        else:
                            result[p] = 0
                    # Consistency: if any pathology positive, No Finding = 0
                    if "No Finding" in result:
                        if any(result.get(p, 0) == 1 for p in label_set
                               if p != "No Finding"):
                            result["No Finding"] = 0
                    return result
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def _try_parse_pos_neg(text: str, label_set: list) -> Optional[dict]:
    """Parse 'Atelectasis: POSITIVE' style output."""
    result = {}
    found = 0
    for p in label_set:
        pattern = (
            rf'{re.escape(p)}\s*[:\-–]\s*'
            r'(POSITIVE|NEGATIVE|Present|Absent|Yes|No)'
        )
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).lower()
            result[p] = 1 if val in ("positive", "present", "yes") else 0
            found += 1
    if found >= min(3, len(label_set)):
        if "No Finding" in result:
            if any(result.get(p, 0) == 1 for p in label_set
                   if p != "No Finding"):
                result["No Finding"] = 0
        for p in label_set:
            if p not in result:
                result[p] = 0
        return result
    return None


def _keyword_fallback(text: str, label_set: list) -> dict:
    """Keyword matching fallback."""
    text_lower = text.lower()
    preds = {}

    if label_set is PATHOLOGIES or label_set == PATHOLOGIES:
        for pathology, keywords in PATHOLOGY_KEYWORDS.items():
            preds[pathology] = 1 if any(kw in text_lower for kw in keywords) else 0
        if any(preds[p] == 1 for p in PATHOLOGIES if p != "No Finding"):
            preds["No Finding"] = 0
    else:
        for dx in label_set:
            preds[dx] = 1 if dx.lower() in text_lower else 0

    return preds


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════

def evaluate_predictions(
    ground_truth: List[dict],
    predictions: List[dict],
    label_set: List[str] = None,
) -> dict:
    """
    Full multi-label metrics: AUROC, AUPRC, F1, Precision, Recall,
    Specificity, Hamming loss, Exact match.
    Per-label exclusion for uncertain (NaN) labels.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score,
    )

    if label_set is None:
        label_set = PATHOLOGIES

    results = {"per_label": {}, "overall": {}}
    all_y_true, all_y_pred = [], []

    for label in label_set:
        y_true, y_pred = [], []
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
            lr.update({
                "auroc": None, "auprc": None, "f1": None,
                "precision": None, "recall": None, "specificity": None,
            })
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
            lr["precision"] = round(
                precision_score(y_true, y_pred, zero_division=0), 4)
            lr["recall"] = round(
                recall_score(y_true, y_pred, zero_division=0), 4)
            tn = int(((1 - y_true) * (1 - y_pred)).sum())
            fp = int(((1 - y_true) * y_pred).sum())
            lr["specificity"] = round(tn / max(tn + fp, 1), 4)

        results["per_label"][label] = lr
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())

    # ── Overall metrics ──
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    def _safe_mean(vals):
        return round(np.mean(vals), 4) if vals else None

    aurocs = [r["auroc"] for r in results["per_label"].values()
              if r.get("auroc") is not None]
    auprcs = [r["auprc"] for r in results["per_label"].values()
              if r.get("auprc") is not None]
    f1s = [r["f1"] for r in results["per_label"].values()
           if r.get("f1") is not None]
    precs = [r["precision"] for r in results["per_label"].values()
             if r.get("precision") is not None]
    recs = [r["recall"] for r in results["per_label"].values()
            if r.get("recall") is not None]

    results["overall"]["macro_auroc"] = _safe_mean(aurocs)
    results["overall"]["macro_auprc"] = _safe_mean(auprcs)
    results["overall"]["macro_f1"] = _safe_mean(f1s)
    results["overall"]["macro_precision"] = _safe_mean(precs)
    results["overall"]["macro_recall"] = _safe_mean(recs)

    if len(all_y_true) > 0 and len(np.unique(all_y_true)) > 1:
        try:
            results["overall"]["micro_auroc"] = round(
                roc_auc_score(all_y_true, all_y_pred), 4)
        except ValueError:
            results["overall"]["micro_auroc"] = None
        results["overall"]["micro_f1"] = round(
            f1_score(all_y_true, all_y_pred, average="micro", zero_division=0),
            4,
        )

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
    results["overall"]["hamming_loss"] = _safe_mean(errs)

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
# Results Printing
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
        if meta.get("extraction_methods"):
            print(f"  Extraction methods: {meta['extraction_methods']}")
        if meta.get("cxr_sources"):
            print(f"  CXR sources: {meta['cxr_sources']}")

    label_col = "Pathology" if label_set == PATHOLOGIES else "Label"
    header = (
        f"{label_col:<30} {'N':>5} {'Prev':>6} {'AUROC':>7} {'AUPRC':>7} "
        f"{'F1':>7} {'Prec':>7} {'Rec':>7} {'Spec':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    fmt = lambda v: f"{v:.4f}" if v is not None else "   N/A"
    for label in label_set:
        r = metrics["per_label"].get(label, {})
        print(
            f"{label:<30} {r.get('n_evaluated',0):>5} "
            f"{r.get('prevalence',0):>6.3f} "
            f"{fmt(r.get('auroc')):>7} {fmt(r.get('auprc')):>7} "
            f"{fmt(r.get('f1')):>7} {fmt(r.get('precision')):>7} "
            f"{fmt(r.get('recall')):>7} {fmt(r.get('specificity')):>7}"
        )

    o = metrics.get("overall", {})
    print(f"\n  Macro AUROC:     {fmt(o.get('macro_auroc'))}")
    print(f"  Macro AUPRC:     {fmt(o.get('macro_auprc'))}")
    print(f"  Macro F1:        {fmt(o.get('macro_f1'))}")
    print(f"  Micro AUROC:     {fmt(o.get('micro_auroc'))}")
    print(f"  Micro F1:        {fmt(o.get('micro_f1'))}")
    print(f"  Hamming Loss:    {fmt(o.get('hamming_loss'))}")
    print(f"  Exact Match:     {fmt(o.get('exact_match'))}")
    print()


def print_comparison(all_metrics: dict):
    """Print side-by-side comparison table."""
    print(f"\n{'='*80}")
    print(f"  ABLATION COMPARISON")
    print(f"{'='*80}")

    fmt = lambda v: f"{v:.4f}" if v is not None else "   N/A"

    # Overall comparison
    header = f"{'Run':<30} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'EM':>8} {'Time':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, m in all_metrics.items():
        o = m.get("overall", {})
        meta = m.get("meta", {})
        ts = meta.get("total_time_s", "?")
        print(
            f"{name:<30} {fmt(o.get('macro_auroc')):>8} "
            f"{fmt(o.get('macro_auprc')):>8} "
            f"{fmt(o.get('macro_f1')):>8} "
            f"{fmt(o.get('exact_match')):>8} "
            f"{str(ts)+' s':>8}"
        )

    # Per-label F1 comparison (only if same label set)
    first_key = next(iter(all_metrics))
    label_set = list(all_metrics[first_key]["per_label"].keys())
    all_same = all(
        set(m["per_label"].keys()) == set(label_set) for m in all_metrics.values()
    )

    if all_same:
        print(f"\n{'Per-Label F1':}")
        header = f"{'Label':<30}"
        for name in all_metrics:
            short = name[:15]
            header += f" {short:>15}"
        print(header)
        print("-" * len(header))

        for label in label_set:
            row = f"{label:<30}"
            for name, m in all_metrics.items():
                f1 = m["per_label"].get(label, {}).get("f1")
                row += f" {fmt(f1):>15}"
            print(row)
    print()


# ═══════════════════════════════════════════════════════════════════════
# Ablation Runners
# ═══════════════════════════════════════════════════════════════════════

class AblationA:
    """
    Qwen3-VL zero-shot: directly read CXR image + raw lab text.
    No ECG tool, no LingShu, no DeepRare, no RAG.
    """
    name = "A_qwen_zeroshot"
    description = "Qwen3-VL zero-shot (CXR + labs)"

    def __init__(self, qwen_url: str = QWEN_URL):
        from openai import OpenAI
        self.client = OpenAI(base_url=qwen_url, api_key="cardioagent")

    def predict_symile(self, data: SymileTestData, idx: int, cache_dir: str) -> dict:
        b64, cxr_source = data.get_cxr_best_base64(idx, cache_dir)
        lab_text = data.get_lab_text(idx)

        prompt = (
            "You are a clinical physician. You are given a chest X-ray "
            "and the patient's blood test results. "
            "Determine which of these conditions are present:\n"
            "1. Atelectasis\n2. Cardiomegaly\n3. Edema\n"
            "4. Lung Opacity\n5. No Finding (normal)\n6. Pleural Effusion\n\n"
        )
        if lab_text:
            prompt += f"Patient Lab Results:\n{lab_text}\n\n"
        prompt += (
            "Consider both imaging and lab values. "
            "Provide your assessment for each condition."
            + JSON_OUTPUT_INSTRUCTION
        )

        response = self.client.chat.completions.create(
            model="qwen3-vl",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=1500,
            temperature=0.1,
        )
        text = response.choices[0].message.content
        preds = extract_predictions(text, PATHOLOGIES)
        return {
            "predictions": preds, "raw_text": text, "cxr_source": cxr_source,
        }

    def predict_mimic4(self, test_data: MIMIC4ECGTestData, idx: int,
                       cache_dir: str) -> dict:
        """ECG-only zero-shot: give Qwen the machine report as text."""
        record = test_data.get_record(idx)
        prompt = (
            "You are a clinical cardiologist. Based on the following ECG "
            "machine report, determine which conditions are present.\n\n"
            f"Machine ECG Report: {record.get('machine_report', 'N/A')}\n\n"
            "For each condition, output 1 if present, 0 if absent."
            + JSON_OUTPUT_INSTRUCTION
        )

        response = self.client.chat.completions.create(
            model="qwen3-vl",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1,
        )
        text = response.choices[0].message.content
        ecg_preds = extract_predictions(text, ECG_DIAGNOSES)
        return {
            "predictions": extract_predictions(text, PATHOLOGIES),
            "ecg_predictions": ecg_preds,
            "raw_text": text,
        }


class AblationB:
    """
    DeepRare pipeline: tools + phenotypes + reflection, NO RAG.
    Uses planner_v3 with CaseMemory disabled.
    """
    name = "B_deeprare_norag"
    description = "DeepRare pipeline (no RAG)"

    def __init__(self, qwen_url: str = QWEN_URL, lingshu_url: str = LINGSHU_URL):
        from planner_v2 import CardioAgentPlannerV3, PatientData
        self.planner = CardioAgentPlannerV3(
            qwen_api_url=qwen_url,
            lingshu_api_url=lingshu_url,
            rag_index_paths=None,          # ← No RAG
            enable_reflection=True,
            max_reflection_rounds=2,
        )
        self.PatientData = PatientData

    def _extract_v3_metadata(self, state) -> dict:
        """Extract DeepRare-specific metadata from AgentStateV3."""
        return {
            "hypothesis_ranking": [
                {"rank": h.rank, "dx": h.diagnosis,
                 "conf": round(h.confidence, 4)}
                for h in state.hypothesis_ranking[:5]
            ],
            "phenotypes": [p["term"] for p in state.phenotype_terms],
            "reflection_rounds": state.reflection_rounds,
            "n_rag_cases": len(state.rag_cases),
            "cross_modal_agreements": state.cross_modal_consistency.get(
                "n_agreements", 0),
            "cross_modal_contradictions": state.cross_modal_consistency.get(
                "n_contradictions", 0),
        }

    def predict_symile(self, data: SymileTestData, idx: int, cache_dir: str) -> dict:
        ecg = data.get_ecg(idx)
        ecg_path = os.path.join(cache_dir, f"ecg_eval_{idx}.npy")
        np.save(ecg_path, ecg)
        cxr_path, cxr_source = data.get_cxr_best_path(idx, cache_dir)
        lab_dict = data.get_lab_dict(idx)
        lab_text = data.get_lab_text(idx)

        state = self.planner.run(self.PatientData(
            patient_id=f"eval_{idx}",
            ecg_path=ecg_path,
            image_path=cxr_path,
            image_type="cxr",
            lab_results=lab_dict if lab_dict else None,
            lab_text=lab_text if not lab_dict else None,
            clinical_notes=JSON_OUTPUT_INSTRUCTION,
        ))
        preds = extract_predictions(state.final_report, PATHOLOGIES)
        result = {"predictions": preds, "raw_text": state.final_report,
                  "cxr_source": cxr_source, "elapsed_s": state.elapsed_s}
        result.update(self._extract_v3_metadata(state))
        return result

    def predict_mimic4(self, test_data: MIMIC4ECGTestData, idx: int,
                       cache_dir: str) -> dict:
        ecg_path = test_data.get_ecg_path(idx)
        record = test_data.get_record(idx)

        state = self.planner.run(self.PatientData(
            patient_id=record.get("patient_id", f"mimic4_{idx}"),
            ecg_path=ecg_path,
            clinical_notes=(
                f"Machine ECG report: {record.get('machine_report', '')}\n"
                + JSON_OUTPUT_INSTRUCTION
            ),
        ))
        result = {
            "predictions": extract_predictions(state.final_report, PATHOLOGIES),
            "ecg_predictions": extract_predictions(state.final_report, ECG_DIAGNOSES),
            "raw_text": state.final_report, "elapsed_s": state.elapsed_s,
        }
        result.update(self._extract_v3_metadata(state))
        return result


class AblationC:
    """
    DeepRare pipeline + CaseMemory RAG.
    Same as B but with RAG retrieval enabled.
    """
    name = "C_deeprare_rag"
    description = "DeepRare pipeline + RAG"

    def __init__(self, qwen_url: str = QWEN_URL, lingshu_url: str = LINGSHU_URL,
                 db_dir: str = DB_DIR):
        from planner_v2 import CardioAgentPlannerV3, PatientData

        # Gather all available index files
        rag_paths = [os.path.join(db_dir, f) for f in [
            "unified_index.jsonl", "mimic_iv_ecg_index.jsonl",
            "symile_index.jsonl", "index.jsonl",
        ]]
        rag_paths = [p for p in rag_paths if os.path.exists(p)]
        logger.info(f"  AblationC RAG indices: {[Path(p).name for p in rag_paths]}")

        self.planner = CardioAgentPlannerV3(
            qwen_api_url=qwen_url,
            lingshu_api_url=lingshu_url,
            rag_index_paths=rag_paths,     # ← RAG enabled
            enable_reflection=True,
            max_reflection_rounds=2,
            rag_top_k=5,
        )
        self.PatientData = PatientData

    def _extract_v3_metadata(self, state) -> dict:
        return {
            "hypothesis_ranking": [
                {"rank": h.rank, "dx": h.diagnosis,
                 "conf": round(h.confidence, 4)}
                for h in state.hypothesis_ranking[:5]
            ],
            "phenotypes": [p["term"] for p in state.phenotype_terms],
            "reflection_rounds": state.reflection_rounds,
            "n_rag_cases": len(state.rag_cases),
            "rag_case_ids": [
                {"id": c.get("patient_id", "?"),
                 "dx": c.get("diagnosis", "")[:60]}
                for c in state.rag_cases[:5]
            ],
            "cross_modal_agreements": state.cross_modal_consistency.get(
                "n_agreements", 0),
            "cross_modal_contradictions": state.cross_modal_consistency.get(
                "n_contradictions", 0),
        }

    def predict_symile(self, data: SymileTestData, idx: int, cache_dir: str) -> dict:
        ecg = data.get_ecg(idx)
        ecg_path = os.path.join(cache_dir, f"ecg_eval_{idx}.npy")
        np.save(ecg_path, ecg)
        cxr_path, cxr_source = data.get_cxr_best_path(idx, cache_dir)
        lab_dict = data.get_lab_dict(idx)
        lab_text = data.get_lab_text(idx)

        state = self.planner.run(self.PatientData(
            patient_id=f"eval_{idx}",
            ecg_path=ecg_path,
            image_path=cxr_path,
            image_type="cxr",
            lab_results=lab_dict if lab_dict else None,
            lab_text=lab_text if not lab_dict else None,
            clinical_notes=JSON_OUTPUT_INSTRUCTION,
        ))
        preds = extract_predictions(state.final_report, PATHOLOGIES)
        result = {"predictions": preds, "raw_text": state.final_report,
                  "cxr_source": cxr_source, "elapsed_s": state.elapsed_s}
        result.update(self._extract_v3_metadata(state))
        return result

    def predict_mimic4(self, test_data: MIMIC4ECGTestData, idx: int,
                       cache_dir: str) -> dict:
        ecg_path = test_data.get_ecg_path(idx)
        record = test_data.get_record(idx)

        state = self.planner.run(self.PatientData(
            patient_id=record.get("patient_id", f"mimic4_{idx}"),
            ecg_path=ecg_path,
            clinical_notes=(
                f"Machine ECG report: {record.get('machine_report', '')}\n"
                + JSON_OUTPUT_INSTRUCTION
            ),
        ))
        result = {
            "predictions": extract_predictions(state.final_report, PATHOLOGIES),
            "ecg_predictions": extract_predictions(state.final_report, ECG_DIAGNOSES),
            "raw_text": state.final_report, "elapsed_s": state.elapsed_s,
        }
        result.update(self._extract_v3_metadata(state))
        return result


# ═══════════════════════════════════════════════════════════════════════
# Unified Runner
# ═══════════════════════════════════════════════════════════════════════

def run_ablation(
    runner,
    data,
    max_samples: int,
    results_dir: str,
    ablation_name: str,
    data_type: str = "symile",
) -> dict:
    """
    Unified runner: inference → extraction → evaluation → save.
    Works for both Symile (CXR pathologies) and MIMIC4ECG (ECG diagnoses).
    """
    cache_dir = os.path.join(results_dir, f"cache_{ablation_name}")
    os.makedirs(cache_dir, exist_ok=True)

    is_symile = (data_type == "symile")
    label_set = PATHOLOGIES if is_symile else ECG_DIAGNOSES
    n = min(data.n, max_samples) if max_samples else data.n

    logger.info(f"\n{'='*60}")
    logger.info(f"Ablation {ablation_name}: {n} samples ({data_type})")
    logger.info(f"Labels: {len(label_set)} ({label_set[0]}...{label_set[-1]})")
    logger.info(f"Runner: {runner.description}")
    logger.info(f"{'='*60}")

    # Pre-batch CXR if Symile + data has prebatch method
    if is_symile and hasattr(data, "prebatch_cxr"):
        data.prebatch_cxr(list(range(n)), cache_dir=cache_dir)

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
                preds = result.get("ecg_predictions", result["predictions"])
            elapsed = time.time() - t0

            # Strip extraction method marker before storing
            clean_preds = {k: v for k, v in preds.items()
                          if k != "_extraction_method"}
            predictions.append(clean_preds)

            raw_outputs.append({
                "index": idx,
                "ground_truth": {
                    k: v for k, v in gt.items() if pd.notna(v)
                },
                "predictions": clean_preds,
                "extraction_method": preds.get("_extraction_method", "unknown"),
                "cxr_source": result.get("cxr_source", "N/A"),
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
                logger.info(
                    f"  [{ablation_name}] {succeeded}/{n} "
                    f"({rate:.2f}/s, ETA {(n-succeeded)/rate:.0f}s, "
                    f"{failed} failed)"
                )

        except Exception as e:
            failed += 1
            ground_truth.append(data.get_ground_truth(idx))
            predictions.append({lbl: 0 for lbl in label_set})
            if failed <= 5:
                logger.warning(f"  Sample {idx} failed: {e}")
            elif failed == 6:
                logger.warning(f"  Suppressing further failure messages...")

    total_time = time.time() - t_start
    logger.info(
        f"  [{ablation_name}] Done: {succeeded} succeeded, {failed} failed, "
        f"{total_time:.1f}s total"
    )

    # ── Evaluate ──
    metrics = evaluate_predictions(ground_truth, predictions, label_set)

    # Extraction method counts
    method_counts = {}
    for out in raw_outputs:
        m = out.get("extraction_method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1

    # CXR source counts (Symile only)
    cxr_source_counts = {}
    for out in raw_outputs:
        s = out.get("cxr_source", "N/A")
        if s != "N/A":
            cxr_source_counts[s] = cxr_source_counts.get(s, 0) + 1

    # DeepRare-specific aggregates (B and C)
    avg_phenotypes = 0
    avg_hypotheses = 0
    avg_rag_cases = 0
    avg_reflection = 0
    if succeeded > 0:
        avg_phenotypes = np.mean([
            len(o.get("phenotypes", [])) for o in raw_outputs
        ])
        avg_hypotheses = np.mean([
            len(o.get("hypothesis_ranking", [])) for o in raw_outputs
        ])
        avg_rag_cases = np.mean([
            o.get("n_rag_cases", 0) for o in raw_outputs
        ])
        avg_reflection = np.mean([
            o.get("reflection_rounds", 0) for o in raw_outputs
        ])

    metrics["meta"] = {
        "ablation": ablation_name,
        "runner": runner.description,
        "data_source": data_type,
        "label_type": "cxr_pathologies" if is_symile else "ecg_diagnoses",
        "n_total": n,
        "n_succeeded": succeeded,
        "n_failed": failed,
        "total_time_s": round(total_time, 1),
        "avg_time_per_sample_s": round(total_time / max(n, 1), 2),
        "extraction_methods": method_counts,
        "cxr_sources": cxr_source_counts,
        "avg_phenotypes": round(avg_phenotypes, 1),
        "avg_hypotheses": round(avg_hypotheses, 1),
        "avg_rag_cases": round(avg_rag_cases, 1),
        "avg_reflection_rounds": round(avg_reflection, 1),
    }

    # ── Save ──
    preds_path = os.path.join(results_dir, f"{ablation_name}_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(raw_outputs, f, indent=2, default=str)

    metrics_path = os.path.join(results_dir, f"{ablation_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print_metrics(ablation_name, metrics, label_set)
    logger.info(f"  Saved: {preds_path}")
    logger.info(f"  Saved: {metrics_path}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    global MIMIC_CXR_JPG_DIR

    parser = argparse.ArgumentParser(
        description="CardioAgent Ablation v5 (DeepRare-Inspired)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_v5.py --ablation A --data-source symile --max-samples 200
  python eval_v5.py --ablation B C --data-source symile --max-samples 100
  python eval_v5.py --ablation all --data-source both --max-samples 50
  python eval_v5.py --mode compare
        """,
    )
    parser.add_argument(
        "--mode", choices=["run", "compare"], default="run",
        help="'run' to execute ablations, 'compare' to print saved results",
    )
    parser.add_argument(
        "--ablation", nargs="+", default=["all"],
        choices=["A", "B", "C", "all"],
        help="Which ablation(s) to run",
    )
    parser.add_argument(
        "--data-source", default="symile",
        choices=["symile", "mimic4ecg", "both"],
        help="Which dataset(s) to evaluate on",
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--data-dir-symile", default=DATA_DIR_SYMILE)
    parser.add_argument("--data-dir-mimic4", default=DATA_DIR_MIMIC4)
    parser.add_argument("--mimic-cxr-dir", default=MIMIC_CXR_JPG_DIR)
    parser.add_argument("--db-dir", default=DB_DIR)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--qwen-url", default=QWEN_URL)
    parser.add_argument("--lingshu-url", default=LINGSHU_URL)
    args = parser.parse_args()

    MIMIC_CXR_JPG_DIR = args.mimic_cxr_dir

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Compare mode ──
    if args.mode == "compare":
        all_metrics = {}
        for f in sorted(Path(args.output_dir).glob("*_metrics.json")):
            name = f.stem.replace("_metrics", "")
            with open(f) as fh:
                all_metrics[name] = json.load(fh)
        if all_metrics:
            print_comparison(all_metrics)
        else:
            print(f"No results found in {args.output_dir}")
        return

    # ── Run mode ──
    ablations = ["A", "B", "C"] if "all" in args.ablation else args.ablation
    all_metrics = {}

    for abl in ablations:
        # Create runner
        if abl == "A":
            runner = AblationA(qwen_url=args.qwen_url)
        elif abl == "B":
            runner = AblationB(
                qwen_url=args.qwen_url, lingshu_url=args.lingshu_url,
            )
        elif abl == "C":
            runner = AblationC(
                qwen_url=args.qwen_url, lingshu_url=args.lingshu_url,
                db_dir=args.db_dir,
            )

        # Run on Symile
        if args.data_source in ["symile", "both"]:
            try:
                sdata = SymileTestData(args.data_dir_symile)
                tag = f"{runner.name}_symile"
                m = run_ablation(
                    runner, sdata, args.max_samples,
                    args.output_dir, tag, "symile",
                )
                all_metrics[tag] = m
            except Exception as e:
                logger.error(f"Symile failed for {abl}: {e}", exc_info=True)

        # Run on MIMIC-IV-ECG
        if args.data_source in ["mimic4ecg", "both"]:
            idx_path = os.path.join(args.db_dir, "mimic_iv_ecg_index.jsonl")
            if os.path.exists(idx_path):
                try:
                    mdata = MIMIC4ECGTestData(
                        args.data_dir_mimic4, idx_path, args.max_samples,
                    )
                    tag = f"{runner.name}_mimic4ecg"
                    m = run_ablation(
                        runner, mdata, args.max_samples,
                        args.output_dir, tag, "mimic4ecg",
                    )
                    all_metrics[tag] = m
                except Exception as e:
                    logger.error(f"MIMIC4ECG failed for {abl}: {e}",
                                 exc_info=True)
            else:
                logger.error(f"MIMIC-IV-ECG index not found: {idx_path}")
                logger.error("Run mimic_iv_ecg_preprocess.py first")

    # ── Summary ──
    if len(all_metrics) > 1:
        print_comparison(all_metrics)

    summary_path = os.path.join(args.output_dir, "all_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()