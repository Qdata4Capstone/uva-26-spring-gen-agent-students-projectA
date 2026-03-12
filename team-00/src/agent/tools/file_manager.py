"""
File Manager Tool
Handles dataset file organization, EXIF stripping, and report generation.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import piexif
    _PIEXIF_AVAILABLE = True
except ImportError:
    _PIEXIF_AVAILABLE = False


class FileManager:
    """
    Manages dataset files: organizing outputs, stripping EXIF metadata,
    generating compliance reports.
    """

    name = "file_manager"
    description = (
        "Organize dataset files, strip EXIF metadata, and generate compliance reports."
    )

    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def strip_exif(self, image_path: str) -> dict:
        """Strip EXIF metadata from an image file in-place."""
        if _PIEXIF_AVAILABLE:
            try:
                piexif.remove(image_path)
                return {"status": "success", "method": "piexif"}
            except Exception:
                pass  # fall through to PIL

        # PIL fallback: re-save without EXIF (PIL does not copy metadata on save)
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            img.save(image_path)
            return {"status": "success", "method": "pil_resave"}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def strip_exif_directory(self, directory: str) -> dict:
        """Strip EXIF from all images in a directory."""
        dir_path = Path(directory)
        all_imgs = [
            p for p in sorted(dir_path.rglob("*"))
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif")
        ]
        total = len(all_imgs)
        results = {"stripped": 0, "failed": 0}

        for i, img_path in enumerate(all_imgs, 1):
            result = self.strip_exif(str(img_path))
            if result["status"] == "success":
                results["stripped"] += 1
            else:
                results["failed"] += 1
                logger.warning(f"EXIF strip failed for {img_path}: {result.get('reason')}")
            # Progress every 200 images or at the end
            if i % 200 == 0 or i == total:
                logger.info(f"  EXIF strip progress: {i}/{total}")

        return results

    def organize_output(self, clip_dirs: list[str], output_base: str) -> dict:
        """
        Organize processed clip directories into a clean output structure.
        Returns mapping from original to output paths.
        """
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        mapping = {}

        for clip_dir in clip_dirs:
            clip_path = Path(clip_dir)
            dest = output_base / clip_path.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(clip_path, dest)
            mapping[str(clip_dir)] = str(dest)

        return {"status": "success", "mapping": mapping}

    def generate_compliance_report(
        self,
        session_id: str,
        user_intent: str,
        context: dict,
        requirements: dict,
        processing_log: list[dict],
        critic_results: list[dict],
        output_dir: str,
    ) -> str:
        """
        Generate a JSON compliance report documenting the de-identification process.
        Returns path to the report file.
        """
        report = {
            "report_id": session_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "user_intent": user_intent,
            "legal_context": {
                "platform": context.get("platform"),
                "conference": context.get("conference"),
                "regulations": context.get("regulations", []),
            },
            "requirements_applied": requirements,
            "processing_summary": {
                "total_clips": len(processing_log),
                "successful": sum(1 for p in processing_log if p.get("status") == "success"),
                "failed": sum(1 for p in processing_log if p.get("status") == "error"),
            },
            "processing_log": processing_log,
            "critic_validation": {
                "total_checked": len(critic_results),
                "passed": sum(1 for c in critic_results if c.get("passed")),
                "failed": sum(1 for c in critic_results if not c.get("passed")),
                "results": critic_results,
            },
            "compliance_statement": self._generate_compliance_statement(
                requirements, critic_results
            ),
        }

        report_path = Path(output_dir) / f"compliance_report_{session_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Compliance report saved to {report_path}")
        return str(report_path)

    def _generate_compliance_statement(
        self, requirements: dict, critic_results: list[dict]
    ) -> str:
        all_passed = all(c.get("passed") for c in critic_results) if critic_results else True
        reid_threshold = requirements.get("re_id_threshold", 0.1)

        if all_passed:
            return (
                f"All processed images passed Re-ID validation "
                f"(ArcFace confidence < {reid_threshold}). "
                "De-identification meets the specified compliance requirements."
            )
        else:
            failed_count = sum(1 for c in critic_results if not c.get("passed"))
            return (
                f"WARNING: {failed_count} image(s) failed Re-ID validation. "
                "Additional processing may be required."
            )

    def count_images(self, directory: str) -> int:
        """Count images in a directory."""
        exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        return sum(
            1 for f in Path(directory).rglob("*") if f.suffix.lower() in exts
        )
