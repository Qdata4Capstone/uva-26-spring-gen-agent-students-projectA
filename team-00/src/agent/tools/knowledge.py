"""
Knowledge Base Tool
Retrieves legal regulations, platform TOS, and conference policies.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = Path(__file__).parent.parent / "knowledge" / "regulations.json"


class KnowledgeBase:
    """
    Retrieves platform TOS, conference open-science policies, and
    regulatory requirements (GDPR, CCPA, etc.).
    """

    name = "knowledge_retrieval"
    description = (
        "Retrieve privacy/legal requirements for platforms (YouTube, TikTok), "
        "academic conferences (CVPR, ICCV, NeurIPS), and regulations (GDPR, CCPA)."
    )

    def __init__(self, knowledge_path: Optional[str] = None):
        path = Path(knowledge_path) if knowledge_path else _KNOWLEDGE_PATH
        with open(path, "r", encoding="utf-8") as f:
            self._db = json.load(f)

    def _normalize(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def query(self, query: str) -> dict:
        """
        Query the knowledge base with a natural language string.
        Returns matching requirements from platforms, conferences, and regulations.
        """
        q = self._normalize(query)
        results = {}

        # Match platforms
        for key, val in self._db.get("platforms", {}).items():
            if self._normalize(key) in q or self._normalize(val.get("name", "")) in q:
                results[f"platform:{key}"] = val

        # Match conferences
        for key, val in self._db.get("conferences", {}).items():
            if self._normalize(key) in q or self._normalize(val.get("name", "")) in q:
                results[f"conference:{key}"] = val

        # Match regulations
        for key, val in self._db.get("regulations", {}).items():
            if self._normalize(key) in q or self._normalize(val.get("name", "")) in q:
                results[f"regulation:{key}"] = val

        return results

    def get_requirements(self, context: dict) -> dict:
        """
        Given a context dict with keys like 'platform', 'conference', 'regulation',
        return merged requirements.
        """
        requirements = {
            "face_blur_required": False,
            "license_plate_blur_required": False,
            "metadata_strip_required": False,
            "de_identification_required": False,
            "compliance_report_required": False,
            "irreversible_anonymization_required": False,
            "re_id_threshold": 0.1,
            "ego_centric_detection": False,
            "sources": [],
            "notes": [],
        }

        sources_to_check = []
        if platform := context.get("platform"):
            key = self._normalize(platform)
            for pkey, pval in self._db.get("platforms", {}).items():
                if self._normalize(pkey) == key or self._normalize(pval.get("name", "")) == key:
                    sources_to_check.append(pval)
                    requirements["sources"].append(f"platform:{pkey}")

        if conference := context.get("conference"):
            key = self._normalize(conference)
            for ckey, cval in self._db.get("conferences", {}).items():
                if self._normalize(ckey) == key or self._normalize(cval.get("name", "")) == key:
                    sources_to_check.append(cval)
                    requirements["sources"].append(f"conference:{ckey}")

        for reg in context.get("regulations", []):
            key = self._normalize(reg)
            for rkey, rval in self._db.get("regulations", {}).items():
                if self._normalize(rkey) == key or self._normalize(rval.get("name", "")) == key:
                    sources_to_check.append(rval)
                    requirements["sources"].append(f"regulation:{rkey}")

        # Merge with strictest rules (OR logic for booleans, MIN for thresholds)
        for source in sources_to_check:
            for bool_key in [
                "face_blur_required", "license_plate_blur_required",
                "metadata_strip_required", "de_identification_required",
                "compliance_report_required", "irreversible_anonymization_required",
                "ego_centric_detection",
            ]:
                if source.get(bool_key):
                    requirements[bool_key] = True
            # Use strictest (lowest) re-ID threshold
            if "re_id_threshold" in source:
                requirements["re_id_threshold"] = min(
                    requirements["re_id_threshold"], source["re_id_threshold"]
                )
            if note := source.get("notes"):
                requirements["notes"].append(note)

        return requirements

    def list_conferences(self) -> list[str]:
        return [v.get("name", k) for k, v in self._db.get("conferences", {}).items()]

    def list_platforms(self) -> list[str]:
        return [v.get("name", k) for k, v in self._db.get("platforms", {}).items()]

    def list_regulations(self) -> list[str]:
        return [v.get("name", k) for k, v in self._db.get("regulations", {}).items()]

    def get_conflict_resolution_info(self) -> dict:
        return self._db.get("conflict_resolution", {})
