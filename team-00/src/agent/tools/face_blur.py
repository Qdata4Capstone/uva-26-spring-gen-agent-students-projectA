"""
Face Blur Tool (MTCNN-style)
Uses OpenCV's DNN face detector (ResNet-SSD) for robust face detection
and Gaussian/pixelation blur for anonymization.
Falls back to Haar cascades if DNN model unavailable.
"""

import cv2
import numpy as np
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# Global lock: prevents concurrent OpenCV DNN init / model download across threads
_INIT_LOCK = threading.Lock()

# OpenCV DNN face detector model URLs (ResNet-SSD)
_MODEL_DIR = Path(__file__).parent.parent / "models"
_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
_PROTOTXT_PATH = _MODEL_DIR / "deploy.prototxt"
_CAFFEMODEL_PATH = _MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"


def _ensure_dnn_model() -> bool:
    """Download DNN face model if not present."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not _PROTOTXT_PATH.exists():
            logger.info("Downloading face detector prototxt...")
            urllib.request.urlretrieve(_PROTOTXT_URL, str(_PROTOTXT_PATH))
        if not _CAFFEMODEL_PATH.exists():
            logger.info("Downloading face detector caffemodel...")
            urllib.request.urlretrieve(_CAFFEMODEL_URL, str(_CAFFEMODEL_PATH))
        return True
    except Exception as e:
        logger.warning(f"Could not download DNN model: {e}. Will use Haar cascade.")
        return False


class FaceBlurTool:
    """
    MTCNN-style face detection and blur using OpenCV DNN.
    Supports Gaussian blur, pixelation, and black-box modes.
    """

    name = "face_blur"
    description = (
        "Detect and blur faces in images using OpenCV DNN (ResNet-SSD). "
        "Supports standard camera footage. Fallback to Haar cascades."
    )

    def __init__(
        self,
        blur_method: Literal["gaussian", "pixelate", "blackbox"] = "gaussian",
        confidence_threshold: float = 0.5,
        blur_strength: int = 51,
        scale_factor: float = 1.3,
    ):
        self.blur_method = blur_method
        self.confidence_threshold = confidence_threshold
        self.blur_strength = blur_strength  # kernel size for gaussian, block size for pixelate
        self.scale_factor = scale_factor
        self._net = None
        self._haar = None
        self._use_dnn = False
        self._initialized = False

    def _init_detector(self):
        if self._initialized:
            return
        with _INIT_LOCK:
            # Double-check inside the lock
            if self._initialized:
                return
            self._init_detector_locked()

    def _init_detector_locked(self):
        dnn_available = _ensure_dnn_model()
        if dnn_available and _PROTOTXT_PATH.exists() and _CAFFEMODEL_PATH.exists():
            try:
                self._net = cv2.dnn.readNetFromCaffe(
                    str(_PROTOTXT_PATH), str(_CAFFEMODEL_PATH)
                )
                self._use_dnn = True
                logger.info("Using DNN face detector (ResNet-SSD)")
            except Exception as e:
                logger.warning(f"DNN load failed: {e}, falling back to Haar")

        if not self._use_dnn:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            logger.info("Using Haar cascade face detector")

        self._initialized = True

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Returns list of (x, y, w, h) face bounding boxes.
        """
        self._init_detector()
        h, w = image.shape[:2]
        boxes = []

        if self._use_dnn:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    # Apply scale factor
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    bw = int((x2 - x1) * self.scale_factor)
                    bh = int((y2 - y1) * self.scale_factor)
                    x1 = max(0, cx - bw // 2)
                    y1 = max(0, cy - bh // 2)
                    x2 = min(w, cx + bw // 2)
                    y2 = min(h, cy + bh // 2)
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, fw, fh) in faces:
                # Apply scale factor
                cx, cy = x + fw // 2, y + fh // 2
                fw2 = int(fw * self.scale_factor)
                fh2 = int(fh * self.scale_factor)
                x = max(0, cx - fw2 // 2)
                y = max(0, cy - fh2 // 2)
                fw2 = min(fw2, w - x)
                fh2 = min(fh2, h - y)
                boxes.append((x, y, fw2, fh2))

        return boxes

    def blur_region(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply blur to a specific region of the image."""
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return image

        if self.blur_method == "gaussian":
            k = self.blur_strength | 1  # ensure odd
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
        elif self.blur_method == "pixelate":
            block = max(1, self.blur_strength // 5)
            small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                               interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:  # blackbox
            blurred = np.zeros_like(roi)

        image[y:y+h, x:x+w] = blurred
        return image

    def process_image(self, input_path: str, output_path: str) -> dict:
        """
        Detect and blur faces in a single image.
        Returns dict with status and face count.
        """
        img = cv2.imread(input_path)
        if img is None:
            return {"status": "error", "reason": f"Cannot read {input_path}"}

        faces = self.detect_faces(img)
        for (x, y, w, h) in faces:
            img = self.blur_region(img, x, y, w, h)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, img)
        return {
            "status": "success",
            "output": output_path,
            "faces_detected": len(faces),
            "method": "dnn" if self._use_dnn else "haar",
        }

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: tuple = (".png", ".jpg", ".jpeg"),
    ) -> dict:
        """Process all images in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "processed": 0, "failed": 0, "total_faces": 0, "outputs": []
        }

        image_files = [f for f in sorted(input_dir.iterdir())
                       if f.suffix.lower() in extensions]

        for img_path in image_files:
            out_path = output_dir / img_path.name
            result = self.process_image(str(img_path), str(out_path))
            if result["status"] == "success":
                results["processed"] += 1
                results["total_faces"] += result.get("faces_detected", 0)
                results["outputs"].append(str(out_path))
            else:
                results["failed"] += 1

        return results

    def get_model_info(self) -> dict:
        self._init_detector()
        return {
            "tool": "FaceBlur (OpenCV DNN ResNet-SSD)",
            "detector": "dnn" if self._use_dnn else "haar_cascade",
            "blur_method": self.blur_method,
            "blur_strength": self.blur_strength,
            "best_for": "standard / fixed-camera footage",
        }
