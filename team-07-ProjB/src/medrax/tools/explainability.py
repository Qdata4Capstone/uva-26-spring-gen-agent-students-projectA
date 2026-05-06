# medrax/tools/explainability.py

import os
import json
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun


# ── Input schema ──────────────────────────────────────────────────────────────

class GradCAMInput(BaseModel):
    image_path: str = Field(description="Path to the chest X-ray image (PNG/JPG/DICOM).")
    target_class: Optional[str] = Field(
        default=None,
        description=(
            "Name of the pathology class to explain (e.g., 'Pneumonia', 'Cardiomegaly'). "
            "If None, the highest-scoring class is used."
        ),
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Optional path to save the GradCAM overlay image. If None, a temp path is used."
    )


# ── Tool class ────────────────────────────────────────────────────────────────

class GradCAMExplainerTool(BaseTool):
    """
    Wraps a GradCAM explainer on TorchXRayVision's DenseNet-121.
    Produces a heatmap overlay highlighting which image regions most
    influenced the model's prediction for a given pathology class.
    """

    name: str = "GradCAMExplainerTool"
    description: str = (
        "Generates a GradCAM saliency heatmap explaining WHY the classifier predicted "
        "a given pathology class for a chest X-ray. Use this when the user asks why "
        "the model made a prediction, or requests a heatmap/saliency map/visual explanation. "
        "REQUIRED arguments: image_path (str) — the original X-ray file path. "
        "OPTIONAL: target_class (str) — single class name like 'Lung Opacity' or 'Effusion'. "
        "After this tool returns, ALWAYS call ImageVisualizerTool with the HEATMAP_PATH value."
    )
    args_schema: Type[BaseModel] = GradCAMInput

    # Tool-level state (set at init, not by the LLM)
    model_dir: str = "/model-weights"
    temp_dir: str = "temp"
    device: str = "cuda"

    # Internal — populated in __init__, excluded from LangChain schema
    _model: object = None
    _pathologies: list = []

    def __init__(self, model_dir: str = "/model-weights", temp_dir: str = "temp", device: str = "cuda"):
        super().__init__()
        self.model_dir = model_dir
        self.temp_dir = temp_dir
        self.device = device
        os.makedirs(temp_dir, exist_ok=True)
        self._load_model()

    def _load_model(self):
        """Load TorchXRayVision DenseNet-121 (same backbone as ChestXRayClassifierTool)."""
        self._model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self._model = self._model.to(self.device)
        self._model.eval()
        self._pathologies = self._model.pathologies

    def _preprocess(self, image_path: str):
        """Load and preprocess image to the format expected by TorchXRayVision."""
        img = Image.open(image_path).convert("L")  # grayscale
        img_np = np.array(img, dtype=np.float32)
        img_np = xrv.datasets.normalize(img_np, 255)
        img_np = img_np[None, ...]  # add channel dim → (1, H, W)
        # Resize to 224x224
        transform = xrv.datasets.XRayCenterCrop()
        img_np = transform(img_np)
        tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)  # (1, 1, 224, 224)
        return tensor

    def _run(
    self,
    image_path: str,
    target_class: Optional[str] = None,
    output_path: Optional[str] = None,
    run_manager=None,
) -> str:
        import traceback

        # Validate and clean image_path
        image_path = str(image_path).strip().strip("'\"")
        if image_path.startswith("{") or image_path.startswith("["):
            return (
                "Error: image_path must be a file path string like '/path/to/xray.png', "
                "not a dictionary or JSON object. Please pass only the file path."
            )
        if not os.path.exists(image_path):
            return f"Error: File not found at '{image_path}'. Please provide a valid image file path."

        # Sanitize target_class
        if target_class:
            target_class = str(target_class).strip().strip("'\"")
            if "{" in target_class:
                try:
                    import json as _json
                    scores = _json.loads(target_class.replace("'", '"'))
                    target_class = max(scores, key=lambda k: scores[k])
                except Exception:
                    target_class = target_class.split(":")[0].strip("{").strip()

        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            tensor = self._preprocess(image_path)

            # Resolve target class index
            if target_class:
                matches = [
                    i for i, p in enumerate(self._pathologies)
                    if p and target_class.lower() in p.lower()
                ]
                if not matches:
                    available = [p for p in self._pathologies if p]
                    return (
                        f"Class '{target_class}' not found. "
                        f"Available classes: {', '.join(available)}"
                    )
                class_idx = matches[0]
            else:
                with torch.no_grad():
                    preds = self._model(tensor).cpu().numpy()[0]
                class_idx = int(np.nanargmax(preds))

            class_name = self._pathologies[class_idx]

            # Run GradCAM
            target_layer = self._model.features[-1]
            cam = GradCAM(model=self._model, target_layers=[target_layer])
            grayscale_cam = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(class_idx)]
            )[0]

            # Build overlay
            img_rgb = Image.open(image_path).convert("RGB")

            # grayscale_cam shape is (H, W) — resize image to match it exactly
            cam_h, cam_w = grayscale_cam.shape
            img_rgb = img_rgb.resize((cam_w, cam_h))
            img_array = np.array(img_rgb, dtype=np.float32) / 255.0

            # Normalize to [0, 1] if needed (show_cam_on_image requires this)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0

            visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

            # Save
            if output_path is None:
                base = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(self.temp_dir, f"{base}_gradcam_{class_name}.png")
            Image.fromarray(visualization).save(output_path)

            # Score
            with torch.no_grad():
                score = float(torch.sigmoid(
                    self._model(tensor)[0, class_idx]
                ).cpu())

            # ── Return with explicit next-step instruction ──────────────────
            return (
                f"GradCAM heatmap generated successfully.\n"
                f"HEATMAP_PATH: {output_path}\n"
                f"Next step: call ImageVisualizerTool with image_path='{output_path}' to display the heatmap to the user.\n"
                f"Explained class: {class_name} (score: {score:.2%})\n"
                f"Interpretation: Brighter/redder regions in the heatmap had the strongest "
                f"influence on the '{class_name}' prediction."
            )

        except Exception as e:
            return f"GradCAM failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"


    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError("GradCAMExplainerTool does not support async")