"""
Critic / Adversarial Validator
Re-ID test after de-identification: checks if faces remain identifiable.
Uses OpenCV DNN face embeddings as a lightweight ArcFace proxy.
If re-ID confidence exceeds threshold, signals the agent to re-process.
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# OpenCV FaceRecognizerSF (uses a lightweight ArcFace-compatible model)
# Available in OpenCV >= 4.5.4 via cv2.FaceRecognizerSF
_FACE_RECOGNIZER_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
_FACE_DETECT_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

_MODEL_DIR = Path(__file__).parent / "models"
_RECOGNIZER_PATH = _MODEL_DIR / "face_recognition_sface_2021dec.onnx"
_DETECTOR_PATH   = _MODEL_DIR / "face_detection_yunet_2023mar.onnx"


def _download_if_missing(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        logger.info(f"Downloading {dest.name}...")
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        logger.warning(f"Could not download {dest.name}: {e}")
        return False


class ReIDCritic:
    """
    Adversarial Re-ID validator.
    Detects faces in processed images and measures similarity to reference faces.
    Signals failure if similarity (cosine) exceeds re_id_threshold.

    Falls back to perceptual hash comparison if face recognition model unavailable.
    """

    def __init__(self, re_id_threshold: float = 0.1):
        self.re_id_threshold = re_id_threshold
        self._detector  = None
        self._recognizer = None
        self._initialized = False
        self._use_dnn = False

    def _init(self):
        if self._initialized:
            return

        det_ok = _download_if_missing(_FACE_DETECT_MODEL_URL, _DETECTOR_PATH)
        rec_ok = _download_if_missing(_FACE_RECOGNIZER_MODEL_URL, _RECOGNIZER_PATH)

        if det_ok and rec_ok and _DETECTOR_PATH.exists() and _RECOGNIZER_PATH.exists():
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(_DETECTOR_PATH), "", (320, 320)
                )
                self._recognizer = cv2.FaceRecognizerSF.create(
                    str(_RECOGNIZER_PATH), ""
                )
                self._use_dnn = True
                logger.info("Re-ID critic: using OpenCV YuNet + SFace (ArcFace-compatible)")
            except Exception as e:
                logger.warning(f"Could not init DNN re-ID: {e}. Using perceptual hash fallback.")
        else:
            logger.info("Re-ID critic: using perceptual hash fallback (no DNN model)")

        self._initialized = True

    def _get_face_embeddings(self, image: np.ndarray) -> list[np.ndarray]:
        """Extract face embeddings from an image."""
        if not self._use_dnn:
            return []

        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)
        if faces is None:
            return []

        embeddings = []
        for face in faces:
            aligned = self._recognizer.alignCrop(image, face)
            feat = self._recognizer.feature(aligned)
            embeddings.append(feat)
        return embeddings

    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Cosine similarity between two face feature vectors."""
        score = self._recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        # score is in [0,1] where 1 = identical
        return float(score)

    def _perceptual_hash_similarity(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> float:
        """
        Fallback: compare images using normalized pixel correlation.
        Lower = more different (better anonymization).
        """
        g1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (32, 32)).flatten().astype(float)
        g2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (32, 32)).flatten().astype(float)
        g1 -= g1.mean(); g2 -= g2.mean()
        norm1 = np.linalg.norm(g1); norm2 = np.linalg.norm(g2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(g1, g2) / (norm1 * norm2))

    def check_image(
        self,
        original_path: str,
        processed_path: str,
    ) -> dict:
        """
        Compare original vs processed image for re-identifiability.
        Returns:
          passed: bool  (True = anonymization OK)
          max_similarity: float  (lower is better)
          threshold: float
        """
        self._init()

        orig = cv2.imread(original_path)
        proc = cv2.imread(processed_path)

        if orig is None or proc is None:
            return {
                "passed": False,
                "max_similarity": 1.0,
                "threshold": self.re_id_threshold,
                "reason": "could not read image",
            }

        if self._use_dnn:
            orig_feats = self._get_face_embeddings(orig)
            proc_feats = self._get_face_embeddings(proc)

            if not orig_feats:
                # No faces in original — trivially passes
                return {
                    "passed": True, "max_similarity": 0.0,
                    "threshold": self.re_id_threshold,
                    "reason": "no_faces_in_original",
                    "original": original_path, "processed": processed_path,
                }

            if not proc_feats:
                # No faces detected in processed — good anonymization
                return {
                    "passed": True, "max_similarity": 0.0,
                    "threshold": self.re_id_threshold,
                    "reason": "no_faces_detected_after_processing",
                    "original": original_path, "processed": processed_path,
                }

            # Check every pair
            max_sim = 0.0
            for of in orig_feats:
                for pf in proc_feats:
                    sim = self._cosine_similarity(of, pf)
                    max_sim = max(max_sim, sim)

            passed = max_sim < self.re_id_threshold
            return {
                "passed": passed,
                "max_similarity": round(max_sim, 4),
                "threshold": self.re_id_threshold,
                "method": "arcface_sface",
                "original": original_path,
                "processed": processed_path,
            }

        else:
            # Perceptual hash fallback
            sim = self._perceptual_hash_similarity(orig, proc)
            # If images are very similar (high correlation), anonymization failed
            passed = sim < 0.85  # heuristic threshold for fallback
            return {
                "passed": passed,
                "max_similarity": round(sim, 4),
                "threshold": 0.85,
                "method": "perceptual_hash_fallback",
                "original": original_path,
                "processed": processed_path,
                "note": "DNN re-ID model unavailable; using perceptual hash",
            }

    def check_directory(
        self,
        original_dir: str,
        processed_dir: str,
        sample_rate: float = 1.0,
    ) -> dict:
        """
        Validate a processed clip directory against its originals.
        sample_rate: fraction of images to check (1.0 = all).
        """
        orig_dir = Path(original_dir)
        proc_dir = Path(processed_dir)

        orig_files = sorted(
            f for f in orig_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

        if not orig_files:
            return {"passed": True, "checked": 0, "failed": 0, "results": []}

        # Sample
        if sample_rate < 1.0:
            import random
            n = max(1, int(len(orig_files) * sample_rate))
            orig_files = random.sample(orig_files, n)

        results = []
        for orig_f in orig_files:
            proc_f = proc_dir / orig_f.name
            if not proc_f.exists():
                results.append({
                    "passed": False,
                    "reason": "processed file missing",
                    "original": str(orig_f),
                    "processed": str(proc_f),
                })
                continue
            r = self.check_image(str(orig_f), str(proc_f))
            results.append(r)

        passed_all = all(r.get("passed") for r in results)
        return {
            "passed": passed_all,
            "checked": len(results),
            "failed": sum(1 for r in results if not r.get("passed")),
            "max_similarity_seen": max((r.get("max_similarity", 0) for r in results), default=0),
            "results": results,
        }
