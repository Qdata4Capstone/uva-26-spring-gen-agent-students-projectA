"""
EgoBlur Tool Wrapper
Wraps Meta's EgoBlur Gen2 CLI for first-person / ego-centric de-identification.
Directly processes images or videos; output is the blurred result.

Usage of underlying CLI:
  egoblur-gen2 \\
    --camera_name camera-rgb \\
    --face_model_path /bigtemp2/tsx4zn/legal/ego_blur_face_gen2.jit \\
    --lp_model_path /bigtemp2/tsx4zn/legal/ego_blur_lp_gen2.jit \\
    --input_image_path <in.png> \\
    --output_image_path <out.png>
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EGOBLUR_ENV    = "/bigtemp/tsx4zn/egoblur"
EGOBLUR_BIN    = f"{EGOBLUR_ENV}/bin/egoblur-gen2"
FACE_MODEL     = "/bigtemp2/tsx4zn/legal/ego_blur_face_gen2.jit"
LP_MODEL       = "/bigtemp2/tsx4zn/legal/ego_blur_lp_gen2.jit"

# camera_name selects camera-specific default thresholds
DEFAULT_CAMERA = "camera-rgb"


class EgoBlurTool:
    """
    Wraps egoblur-gen2 for ego-centric image/video anonymization.
    Takes image or video as input; writes blurred result to output path.
    """

    name = "egoblur"
    description = (
        "Blur faces and license plates in ego-centric (first-person) images or videos "
        "using Meta EgoBlur Gen2. Input: raw image/video path. Output: blurred file."
    )

    def __init__(
        self,
        face_model_path: str = FACE_MODEL,
        lp_model_path: str = LP_MODEL,
        camera_name: str = DEFAULT_CAMERA,
        face_threshold: Optional[float] = None,   # None = use camera default
        lp_threshold: Optional[float] = None,
        scale_factor: float = 1.15,
        blur_faces: bool = True,
        blur_lp: bool = True,
    ):
        self.face_model_path = face_model_path
        self.lp_model_path = lp_model_path
        self.camera_name = camera_name
        self.face_threshold = face_threshold
        self.lp_threshold = lp_threshold
        self.scale_factor = scale_factor
        self.blur_faces = blur_faces
        self.blur_lp = blur_lp

    @property
    def is_available(self) -> bool:
        return (
            os.path.isfile(EGOBLUR_BIN)
            and (os.path.isfile(self.face_model_path) or os.path.isfile(self.lp_model_path))
        )

    def _build_cmd(
        self,
        input_image: Optional[str] = None,
        output_image: Optional[str] = None,
        input_video: Optional[str] = None,
        output_video: Optional[str] = None,
    ) -> list[str]:
        cmd = [EGOBLUR_BIN, "--camera_name", self.camera_name]

        if self.blur_faces and os.path.isfile(self.face_model_path):
            cmd += ["--face_model_path", self.face_model_path]
            if self.face_threshold is not None:
                cmd += ["--face_model_score_threshold", str(self.face_threshold)]

        if self.blur_lp and os.path.isfile(self.lp_model_path):
            cmd += ["--lp_model_path", self.lp_model_path]
            if self.lp_threshold is not None:
                cmd += ["--lp_model_score_threshold", str(self.lp_threshold)]

        cmd += ["--scale_factor_detections", str(self.scale_factor)]

        if input_image and output_image:
            cmd += ["--input_image_path", input_image, "--output_image_path", output_image]
        elif input_video and output_video:
            cmd += ["--input_video_path", input_video, "--output_video_path", output_video]

        return cmd

    def process_image(self, input_path: str, output_path: str) -> dict:
        """Blur a single image. Returns status dict."""
        if not self.is_available:
            return {"status": "error", "reason": "egoblur or model files not found"}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cmd = self._build_cmd(input_image=input_path, output_image=output_path)
        env = {**os.environ, "PATH": f"{EGOBLUR_ENV}/bin:" + os.environ.get("PATH", "")}

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, env=env
            )
            if result.returncode == 0:
                return {"status": "success", "output": output_path}
            else:
                logger.error(f"egoblur-gen2 error: {result.stderr}")
                return {"status": "error", "reason": result.stderr.strip()}
        except subprocess.TimeoutExpired:
            return {"status": "error", "reason": "timeout"}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def process_video(self, input_path: str, output_path: str) -> dict:
        """Blur a video file. Returns status dict."""
        if not self.is_available:
            return {"status": "error", "reason": "egoblur or model files not found"}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cmd = self._build_cmd(input_video=input_path, output_video=output_path)
        env = {**os.environ, "PATH": f"{EGOBLUR_ENV}/bin:" + os.environ.get("PATH", "")}

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env
            )
            if result.returncode == 0:
                return {"status": "success", "output": output_path}
            else:
                logger.error(f"egoblur-gen2 error: {result.stderr}")
                return {"status": "error", "reason": result.stderr.strip()}
        except subprocess.TimeoutExpired:
            return {"status": "error", "reason": "timeout"}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def process_clip_dir(
        self,
        input_dir: str,
        output_dir: str,
        extensions: tuple = (".png", ".jpg", ".jpeg"),
    ) -> dict:
        """
        Process all images in a clip directory (e.g. one clip folder from the dataset).
        Returns summary dict.
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"processed": 0, "failed": 0, "outputs": []}
        image_files = sorted(
            f for f in input_dir.iterdir() if f.suffix.lower() in extensions
        )

        for img_path in image_files:
            out_path = output_dir / img_path.name
            r = self.process_image(str(img_path), str(out_path))
            if r["status"] == "success":
                results["processed"] += 1
                results["outputs"].append(str(out_path))
            else:
                results["failed"] += 1
                logger.warning(f"Failed {img_path.name}: {r.get('reason')}")

        return results

    def get_model_info(self) -> dict:
        return {
            "tool": "EgoBlur Gen2 (Meta)",
            "face_model": self.face_model_path,
            "lp_model": self.lp_model_path,
            "face_model_exists": os.path.isfile(self.face_model_path),
            "lp_model_exists": os.path.isfile(self.lp_model_path),
            "camera_name": self.camera_name,
            "available": self.is_available,
            "best_for": "ego-centric / first-person video",
        }
