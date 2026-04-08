#!/usr/bin/env python3
"""
Evaluation script for the Legal Compliance Agent.
Computes Re-Identification Rate (via ArcFace/SFace) and SSIM
across the full dataset (original vs de-identified).
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_DIR = Path("/bigtemp2/tsx4zn/legal/images")
OUTPUT_DIR  = Path("/bigtemp2/tsx4zn/legal/output/deidentified")

# ── Face detection + recognition models ──────────────────────
_MODEL_DIR = Path(__file__).parent / "agent" / "models"
_DETECTOR_PATH   = _MODEL_DIR / "face_detection_yunet_2023mar.onnx"
_RECOGNIZER_PATH = _MODEL_DIR / "face_recognition_sface_2021dec.onnx"


def build_detector_recognizer():
    """Create per-thread detector + recognizer instances."""
    det = cv2.FaceDetectorYN.create(str(_DETECTOR_PATH), "", (320, 320))
    rec = cv2.FaceRecognizerSF.create(str(_RECOGNIZER_PATH), "")
    return det, rec


def get_face_embeddings(img, det, rec):
    """Detect faces and return list of (bbox, embedding)."""
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None:
        return []
    results = []
    for face in faces:
        aligned = rec.alignCrop(img, face)
        feat = rec.feature(aligned)
        results.append((face, feat))
    return results


def evaluate_image_pair(orig_path, proc_path, det, rec):
    """
    Evaluate a single image pair.
    Returns dict with: re_id metrics, ssim, face counts.
    """
    orig = cv2.imread(str(orig_path))
    proc = cv2.imread(str(proc_path))
    if orig is None or proc is None:
        return None

    # ── SSIM ──
    # Convert to grayscale for SSIM
    g_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    g_proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    ssim_val = ssim(g_orig, g_proc)

    # ── Re-ID ──
    orig_faces = get_face_embeddings(orig, det, rec)
    proc_faces = get_face_embeddings(proc, det, rec)

    n_orig_faces = len(orig_faces)
    n_proc_faces = len(proc_faces)

    max_cosine = 0.0
    re_identified = False

    if n_orig_faces > 0 and n_proc_faces > 0:
        for _, of in orig_faces:
            for _, pf in proc_faces:
                cos = float(rec.match(of, pf, cv2.FaceRecognizerSF_FR_COSINE))
                max_cosine = max(max_cosine, cos)
        # Threshold for "re-identified": cosine > 0.363 (OpenCV SFace recommendation)
        re_identified = max_cosine > 0.363

    return {
        "orig": str(orig_path.name),
        "ssim": round(ssim_val, 4),
        "n_orig_faces": n_orig_faces,
        "n_proc_faces": n_proc_faces,
        "max_cosine": round(max_cosine, 4),
        "re_identified": re_identified,
    }


def evaluate_clip(clip_name, det, rec):
    """Evaluate all image pairs in a clip."""
    orig_dir = DATASET_DIR / clip_name
    proc_dir = OUTPUT_DIR / clip_name

    if not proc_dir.exists():
        return clip_name, []

    orig_imgs = sorted(
        f for f in orig_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    results = []
    for orig_img in orig_imgs:
        proc_img = proc_dir / orig_img.name
        if not proc_img.exists():
            continue
        r = evaluate_image_pair(orig_img, proc_img, det, rec)
        if r:
            r["clip"] = clip_name
            results.append(r)
    return clip_name, results


def main():
    t0 = time.time()

    # Check models exist
    if not _DETECTOR_PATH.exists() or not _RECOGNIZER_PATH.exists():
        logger.error("Face detection/recognition models not found. Run the agent first to download them.")
        sys.exit(1)

    clips = sorted(d.name for d in DATASET_DIR.iterdir() if d.is_dir())
    logger.info(f"Evaluating {len(clips)} clips...")

    all_results = []
    det, rec = build_detector_recognizer()

    for i, clip in enumerate(clips, 1):
        clip_name, results = evaluate_clip(clip, det, rec)
        all_results.extend(results)
        if i % 10 == 0 or i == len(clips):
            logger.info(f"  Progress: {i}/{len(clips)} clips ({len(all_results)} images)")

    elapsed = time.time() - t0
    logger.info(f"Evaluation complete in {elapsed:.1f}s")

    # ── Aggregate statistics ──
    total = len(all_results)
    if total == 0:
        logger.error("No image pairs found!")
        return

    ssim_vals = [r["ssim"] for r in all_results]
    cosine_vals = [r["max_cosine"] for r in all_results]

    # Images where original had faces
    with_faces = [r for r in all_results if r["n_orig_faces"] > 0]
    # Images where faces survived de-identification
    faces_survived = [r for r in with_faces if r["n_proc_faces"] > 0]
    # Images where re-identification succeeded (cosine > threshold)
    re_id_success = [r for r in all_results if r["re_identified"]]

    # Face detection rate after processing (lower = better)
    face_removal_rate = 1.0 - (len(faces_survived) / len(with_faces)) if with_faces else 1.0
    # Re-identification rate (lower = better)
    re_id_rate = len(re_id_success) / len(with_faces) if with_faces else 0.0

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n  Dataset: {len(clips)} clips, {total} image pairs")
    print(f"  Evaluation time: {elapsed:.1f}s")

    print(f"\n  ── SSIM (Structural Similarity) ──")
    print(f"  Mean SSIM:    {np.mean(ssim_vals):.4f}")
    print(f"  Median SSIM:  {np.median(ssim_vals):.4f}")
    print(f"  Std SSIM:     {np.std(ssim_vals):.4f}")
    print(f"  Min SSIM:     {np.min(ssim_vals):.4f}")
    print(f"  Max SSIM:     {np.max(ssim_vals):.4f}")

    print(f"\n  ── Re-Identification Analysis ──")
    print(f"  Images with faces (original):  {len(with_faces)}/{total} ({100*len(with_faces)/total:.1f}%)")
    print(f"  Faces survived de-id:          {len(faces_survived)}/{len(with_faces)} ({100*(1-face_removal_rate):.1f}%)" if with_faces else "  Faces survived de-id:          N/A")
    print(f"  Face removal rate:             {100*face_removal_rate:.1f}%")
    print(f"  Re-identification successes:   {len(re_id_success)}/{len(with_faces)} ({100*re_id_rate:.1f}%)" if with_faces else "  Re-identification successes:   N/A")
    print(f"  Re-identification rate:        {100*re_id_rate:.2f}%")
    print(f"  Mean cosine similarity:        {np.mean(cosine_vals):.4f}")
    if with_faces:
        face_cosines = [r["max_cosine"] for r in with_faces]
        print(f"  Mean cosine (faces only):      {np.mean(face_cosines):.4f}")

    print("\n" + "=" * 70)

    # ── Per-clip summary ──
    print("\n  Per-clip breakdown:")
    print(f"  {'Clip':<40s} {'Images':>6s} {'SSIM':>8s} {'Faces':>6s} {'ReID':>6s} {'MaxCos':>8s}")
    print("  " + "-" * 74)
    clip_stats = {}
    for r in all_results:
        c = r["clip"]
        if c not in clip_stats:
            clip_stats[c] = {"ssims": [], "faces": 0, "reid": 0, "cosines": [], "count": 0}
        clip_stats[c]["ssims"].append(r["ssim"])
        clip_stats[c]["cosines"].append(r["max_cosine"])
        clip_stats[c]["count"] += 1
        if r["n_orig_faces"] > 0:
            clip_stats[c]["faces"] += 1
        if r["re_identified"]:
            clip_stats[c]["reid"] += 1

    for clip in sorted(clip_stats.keys()):
        s = clip_stats[clip]
        print(f"  {clip:<40s} {s['count']:>6d} {np.mean(s['ssims']):>8.4f} {s['faces']:>6d} {s['reid']:>6d} {np.max(s['cosines']):>8.4f}")

    # ── Save results ──
    output_path = Path("/bigtemp2/tsx4zn/legal/output/evaluation_results.json")
    summary = {
        "dataset": {
            "clips": len(clips),
            "total_images": total,
        },
        "ssim": {
            "mean": round(float(np.mean(ssim_vals)), 4),
            "median": round(float(np.median(ssim_vals)), 4),
            "std": round(float(np.std(ssim_vals)), 4),
            "min": round(float(np.min(ssim_vals)), 4),
            "max": round(float(np.max(ssim_vals)), 4),
        },
        "re_identification": {
            "images_with_faces": len(with_faces),
            "faces_survived": len(faces_survived),
            "face_removal_rate": round(face_removal_rate, 4),
            "re_id_successes": len(re_id_success),
            "re_id_rate": round(re_id_rate, 4),
            "mean_cosine": round(float(np.mean(cosine_vals)), 4),
            "threshold": 0.363,
        },
        "per_image": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full results saved to: {output_path}")

    # Also save CSV for easy analysis
    csv_path = Path("/bigtemp2/tsx4zn/legal/output/evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "orig", "ssim", "n_orig_faces", "n_proc_faces", "max_cosine", "re_identified"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"  CSV results saved to:  {csv_path}")
    print()


if __name__ == "__main__":
    main()
