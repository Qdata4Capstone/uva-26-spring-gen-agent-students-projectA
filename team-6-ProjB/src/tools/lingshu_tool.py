"""
LingShu MCP Tool
=================
Wraps LingShu-8B (running as vLLM server) as an MCP-compatible tool.
Handles: raw DICOM (.dcm) → preprocess to PNG → call LingShu API → structured output

LingShu-8B runs separately as: vllm serve lingshu-8b --port 8001
This tool handles DICOM→PNG conversion and API calls.
"""

import json
import os
import base64
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LingShuTool:
    """
    Cardiac MRI/CT analysis via LingShu-8B.
    Pipeline: DICOM → PNG + metadata → LingShu vLLM API → findings
    """

    def __init__(
        self,
        lingshu_api_url: str = "http://localhost:8001/v1",
        lingshu_model_name: str = "lingshu-8b",
        output_dir: str = "/tmp/cardioagent",
    ):
        self.api_url = lingshu_api_url
        self.model_name = lingshu_model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ──

    def analyze(self, dicom_input: str, patient_id: str = "unknown", question: str = "") -> dict:
        """
        Analyze cardiac DICOM data.
        dicom_input: path to .dcm file OR directory of .dcm files
        Returns structured analysis with step tracking.
        """
        result = {"patient_id": patient_id, "input_path": dicom_input, "steps": []}

        # Step 1: Determine input type
        if os.path.isdir(dicom_input):
            result["steps"].append({"step": "detect_input", "status": "success", "detail": "DICOM directory detected"})
            prep = self._process_dicom_series(dicom_input, patient_id)
        elif dicom_input.lower().endswith(".dcm"):
            result["steps"].append({"step": "detect_input", "status": "success", "detail": "Single DICOM file detected"})
            prep = self._process_single_dicom(dicom_input, patient_id)
        elif dicom_input.lower().endswith((".png", ".jpg", ".jpeg")):
            result["steps"].append({"step": "detect_input", "status": "success", "detail": "Image file — no DICOM conversion needed"})
            prep = {"images": [dicom_input], "metadata_text": "Pre-processed image.", "montage_path": dicom_input}
        else:
            result["error"] = f"Unsupported file type: {dicom_input}"
            return result

        if "error" in prep:
            result["steps"].append({"step": "preprocess", "status": "error", "detail": prep["error"]})
            result["error"] = prep["error"]
            return result

        result["steps"].append({
            "step": "preprocess",
            "status": "success",
            "detail": f"Generated {len(prep.get('images', []))} images. {prep.get('metadata_text', '')[:100]}"
        })
        result["images"] = prep.get("images", [])
        result["montage_path"] = prep.get("montage_path")
        result["metadata"] = prep.get("metadata_text", "")

        # Step 2: Call LingShu
        try:
            analysis = self._call_lingshu(
                image_path=prep.get("montage_path", prep["images"][0]),
                metadata=prep.get("metadata_text", ""),
                question=question,
            )
            result["lingshu_analysis"] = analysis
            result["steps"].append({
                "step": "lingshu_inference",
                "status": "success",
                "detail": f"LingShu returned {len(analysis)} chars of analysis"
            })
        except Exception as e:
            result["steps"].append({"step": "lingshu_inference", "status": "error", "detail": str(e)})
            result["lingshu_analysis"] = f"LingShu call failed: {e}"

        # Step 3: Compile findings
        result["findings"] = self._parse_findings(result.get("lingshu_analysis", ""))
        result["steps"].append({"step": "compile_findings", "status": "success", "detail": f"{len(result['findings'])} findings extracted"})

        return result

    # ── DICOM Processing ──

    def _process_single_dicom(self, dcm_path: str, patient_id: str) -> dict:
        """Convert single DICOM to PNG + metadata"""
        import pydicom
        from PIL import Image

        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(float)

        # Apply window/level if available
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = float(ds.WindowCenter[0]) if hasattr(ds.WindowCenter, '__iter__') else float(ds.WindowCenter)
            ww = float(ds.WindowWidth[0]) if hasattr(ds.WindowWidth, '__iter__') else float(ds.WindowWidth)
            arr = np.clip((arr - (wc - ww / 2)) / ww * 255, 0, 255)
        elif arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255

        img = Image.fromarray(arr.astype(np.uint8))
        if img.mode != "RGB":
            img = img.convert("RGB")

        png_path = os.path.join(self.output_dir, f"mri_{patient_id}_single.png")
        img.save(png_path)

        metadata = self._extract_metadata(ds)
        return {"images": [png_path], "montage_path": png_path, "metadata_text": metadata}

    def _process_dicom_series(self, dcm_dir: str, patient_id: str) -> dict:
        """Convert DICOM series to montage + individual slice PNGs"""
        import pydicom
        from PIL import Image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dcm_files = sorted(Path(dcm_dir).rglob("*.dcm"))
        if not dcm_files:
            return {"error": f"No .dcm files found in {dcm_dir}"}

        # Group by slice location
        slices = {}
        sample_ds = None
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f))
                if not hasattr(ds, "pixel_array"):
                    continue
                if sample_ds is None:
                    sample_ds = ds
                loc = round(float(getattr(ds, "SliceLocation", 0)), 1)
                slices.setdefault(loc, []).append(ds)
            except Exception:
                continue

        if not slices:
            return {"error": "No valid DICOM files with pixel data"}

        # Take first frame from each slice location
        locations = sorted(slices.keys())
        n = min(len(locations), 16)
        cols = min(4, n)
        rows = max(1, (n + cols - 1) // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), facecolor="#0d1117")
        if n == 1:
            axes = np.array([[axes]])
        axes = np.array(axes).reshape(-1)

        images = []
        for i in range(n):
            loc = locations[i]
            arr = slices[loc][0].pixel_array.astype(float)
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min())

            axes[i].imshow(arr, cmap="gray")
            axes[i].set_title(f"Slice {loc:.0f}mm", fontsize=7, color="#8b949e")
            axes[i].axis("off")

            # Save individual slice
            slice_path = os.path.join(self.output_dir, f"mri_{patient_id}_slice_{i}.png")
            pil_img = Image.fromarray((arr * 255).astype(np.uint8))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.save(slice_path)
            images.append(slice_path)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        montage_path = os.path.join(self.output_dir, f"mri_{patient_id}_montage.png")
        fig.savefig(montage_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)

        metadata = self._extract_metadata(sample_ds)
        metadata += f"\nTotal slices: {len(locations)}, Total frames: {sum(len(v) for v in slices.values())}"

        return {"images": images, "montage_path": montage_path, "metadata_text": metadata}

    def _extract_metadata(self, ds) -> str:
        """Extract DICOM metadata as text for LingShu context"""
        parts = []
        fields = [
            ("Modality", "Modality"), ("Series", "SeriesDescription"),
            ("Slice Thickness", "SliceThickness"), ("Pixel Spacing", "PixelSpacing"),
            ("Rows×Cols", None), ("Field Strength", "MagneticFieldStrength"),
            ("Sequence", "SequenceName"), ("Patient Position", "PatientPosition"),
        ]
        for label, attr in fields:
            if attr is None:
                val = f"{getattr(ds, 'Rows', '?')}×{getattr(ds, 'Columns', '?')}"
            else:
                val = getattr(ds, attr, "N/A")
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    val = list(val)
            parts.append(f"{label}: {val}")

        return " | ".join(parts)

    # ── LingShu API Call ──

    def _call_lingshu(self, image_path: str, metadata: str = "", question: str = "") -> str:
        """Call LingShu-8B via OpenAI-compatible vLLM API"""
        from openai import OpenAI

        client = OpenAI(base_url=self.api_url, api_key="lingshu-key")

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        prompt = f"Medical image metadata: {metadata}\n\n"
        prompt += question or (
            "Analyze this cardiac MRI image. Provide:\n"
            "1. Chamber assessment (dimensions, wall thickness)\n"
            "2. Wall motion analysis (normal/hypokinetic/akinetic, specify regions)\n"
            "3. Myocardial tissue characteristics\n"
            "4. Any structural abnormalities\n"
            "5. Overall impression and severity"
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=2000,
            temperature=0.1,
        )
        return response.choices[0].message.content

    # ── Parse Findings ──

    def _parse_findings(self, analysis_text: str) -> list:
        """Extract structured findings from LingShu's free-text response"""
        findings = []
        keywords = {
            "akinesis": "severe", "akinetic": "severe", "dyskinesis": "severe",
            "hypokinesis": "moderate", "hypokinetic": "moderate",
            "wall thinning": "moderate", "dilation": "moderate", "dilated": "moderate",
            "thickening": "mild", "hypertrophy": "moderate",
            "effusion": "moderate", "pericardial": "mild",
            "normal": "normal", "no abnormality": "normal",
            "scar": "severe", "fibrosis": "moderate", "edema": "moderate",
        }

        text_lower = analysis_text.lower()
        for keyword, severity in keywords.items():
            if keyword in text_lower:
                # Extract sentence containing the keyword
                for sentence in analysis_text.split("."):
                    if keyword.lower() in sentence.lower():
                        findings.append({
                            "finding": sentence.strip()[:200],
                            "severity": severity,
                            "keyword": keyword,
                            "modality": "cmr",
                        })
                        break

        if not findings:
            findings.append({
                "finding": analysis_text[:300],
                "severity": "unknown",
                "modality": "cmr",
            })

        return findings


# ── MCP Server ──

def create_mcp_server():
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server("lingshu-cardiac")
    tool = LingShuTool()

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="analyze_cardiac_mri",
                description=(
                    "Analyze cardiac MRI/CT. Accepts raw DICOM files/directories or PNG images. "
                    "Converts DICOM to images, calls LingShu-8B for analysis. "
                    "Returns: wall motion, chamber assessment, tissue characterization."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dicom_input": {"type": "string", "description": "Path to .dcm file, DICOM directory, or image file"},
                        "patient_id": {"type": "string", "default": "unknown"},
                        "question": {"type": "string", "default": ""},
                    },
                    "required": ["dicom_input"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "analyze_cardiac_mri":
            result = tool.analyze(
                arguments["dicom_input"],
                arguments.get("patient_id", "unknown"),
                arguments.get("question", ""),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server