"""
LLM Controller — The Brain
Uses Claude (claude-haiku-4-5 for speed) to:
  1. Parse user intent (conference / platform / regulation)
  2. Detect ego-centric video proportion
  3. Reason about tool selection and conflict resolution
  4. Decompose the task into [license-check → de-id → validate] steps
"""

import json
import logging
import os
import re
from typing import Any, Optional

import anthropic

logger = logging.getLogger(__name__)

# Use haiku for fast reasoning steps; upgrade to sonnet for complex conflict resolution
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_STRONG_MODEL  = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are a legal compliance assistant for vision datasets.
Your job is to analyze user intent and plan de-identification of image/video datasets
to meet privacy laws, platform TOS, and conference policies.

You have access to these tools:
- egoblur: Best for ego-centric/first-person video. Blurs faces + license plates directly.
- face_blur: OpenCV DNN face blurring. Best for fixed-camera/standard footage.
- knowledge_retrieval: Fetch platform TOS, conference policies, GDPR/CCPA rules.
- file_manager: Strip EXIF metadata, generate compliance reports.
- re_id_critic: Validate de-identification quality (ArcFace re-ID test).

When the user describes their situation, output a JSON plan with these fields:
{
  "intent_summary": "<1-sentence summary of user goal>",
  "platform": "<youtube|tiktok|null>",
  "conference": "<cvpr2026|cvpr2025|iccv2025|neurips2025|null>",
  "regulations": ["gdpr", "ccpa"],
  "ego_centric_ratio_estimate": 0.0,
  "selected_tool": "<egoblur|face_blur|both>",
  "task_sequence": ["license_check", "de_identification", "exif_strip", "validation", "report"],
  "conflict_notes": "<any conflicts between user preference and compliance, or empty string>",
  "compliance_priority": "<note if compliance overrides user preference>"
}

Rules:
- If input mentions "first person", "ego", "Aria", "GoPro", "dashcam" → set ego_centric_ratio_estimate > 0.5
- If conference is CVPR → ego_centric_detection is mandatory
- GDPR requires irreversible anonymization → prefer egoblur over face_blur
- Always include "validation" and "report" in task_sequence
- If user wants realism but compliance requires full anonymization, note the conflict and choose compliance
"""


class LLMController:
    """
    Parses user instructions, reasons about legal requirements,
    and produces a structured task plan for the agent.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def parse_intent(self, user_message: str, knowledge_context: Optional[str] = None) -> dict:
        """
        Parse user message into a structured task plan.
        Returns the JSON plan dict.
        """
        messages = [{"role": "user", "content": user_message}]
        if knowledge_context:
            messages[0]["content"] += f"\n\n[Relevant regulations]\n{knowledge_context}"

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=messages,
            )
            raw = resp.content[0].text.strip()
            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                raise ValueError(f"No JSON found in response: {raw}")
            return plan
        except Exception as e:
            logger.error(f"LLM parse_intent failed: {e}")
            # Keyword-based fallback — extract intent without LLM
            return self._keyword_fallback(user_message)

    def _keyword_fallback(self, text: str) -> dict:
        """
        Regex-based intent extraction when LLM is unavailable.
        Covers the most common patterns from specs.md use cases.
        """
        t = text.lower()

        # Platform detection
        platform = None
        if "youtube" in t:
            platform = "youtube"
        elif "tiktok" in t:
            platform = "tiktok"

        # Conference detection
        conference = None
        for conf in ["cvpr2026", "cvpr 2026", "cvpr2025", "cvpr 2025",
                     "iccv2025", "iccv 2025", "neurips2025", "neurips 2025"]:
            if conf in t:
                conference = conf.replace(" ", "")
                break
        # Also catch "cvpr" alone → default to latest
        if conference is None and "cvpr" in t:
            conference = "cvpr2026"
        if conference is None and "iccv" in t:
            conference = "iccv2025"
        if conference is None and "neurips" in t:
            conference = "neurips2025"

        # Regulation detection
        regulations = []
        if "gdpr" in t:
            regulations.append("gdpr")
        if "ccpa" in t:
            regulations.append("ccpa")

        # Ego-centric detection
        ego_keywords = ["ego", "first person", "first-person",
                        "gopro", "aria", "helmet", "wearable", "fpv"]
        ego_ratio = 0.8 if any(k in t for k in ego_keywords) else 0.0

        # Always use egoblur — it handles detection + blurring in one step,
        # no need for a separate face detector.
        return {
            "intent_summary": text[:100],
            "platform": platform,
            "conference": conference,
            "regulations": regulations,
            "ego_centric_ratio_estimate": ego_ratio,
            "selected_tool": "egoblur",
            "task_sequence": [
                "license_check", "de_identification", "exif_strip", "validation", "report"
            ],
            "conflict_notes": (
                "GDPR requires irreversible anonymization — using egoblur." if "gdpr" in t else ""
            ),
            "compliance_priority": "compliance" if regulations or conference else "",
            "_fallback": True,
        }

    def resolve_conflicts(
        self, user_preferences: dict, requirements: dict
    ) -> tuple[dict, str]:
        """
        Use LLM to reason about conflicts between user preferences and requirements.
        Returns (resolved_requirements, explanation).
        """
        prompt = (
            f"User preferences: {json.dumps(user_preferences, indent=2)}\n"
            f"Compliance requirements: {json.dumps(requirements, indent=2)}\n\n"
            "Identify conflicts and decide the final processing strategy. "
            "Always prioritize compliance. Output JSON:\n"
            '{"resolved": {...merged requirements...}, "explanation": "..."}'
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("resolved", requirements), result.get("explanation", "")
        except Exception as e:
            logger.warning(f"Conflict resolution LLM call failed: {e}")
        return requirements, "Defaulting to strictest requirements."

    def classify_clip(self, clip_name: str, sample_image_path: Optional[str] = None) -> str:
        """
        Classify a clip as ego-centric or standard based on its name/path.
        Returns "egocentric" or "standard".
        """
        # Heuristic: known ego-centric dataset patterns
        ego_patterns = [
            r"ego", r"aria", r"gopro", r"helmet", r"head", r"pov",
            r"first.?person", r"fpv", r"wearable",
        ]
        name_lower = clip_name.lower()
        for pat in ego_patterns:
            if re.search(pat, name_lower):
                return "egocentric"

        # If we have a sample image, ask LLM with vision
        if sample_image_path and os.path.isfile(sample_image_path):
            try:
                import base64
                with open(sample_image_path, "rb") as f:
                    img_b64 = base64.standard_b64encode(f.read()).decode()
                resp = self.client.messages.create(
                    model=_STRONG_MODEL,
                    max_tokens=64,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Is this image from an ego-centric (first-person / "
                                    "wearable camera) perspective? Reply with only "
                                    "'egocentric' or 'standard'."
                                ),
                            },
                        ],
                    }],
                )
                answer = resp.content[0].text.strip().lower()
                return "egocentric" if "ego" in answer else "standard"
            except Exception as e:
                logger.warning(f"Vision classification failed: {e}")

        return "standard"

    def summarize_run(self, processing_log: list[dict], critic_results: list[dict]) -> str:
        """Generate a human-readable summary of the agent run."""
        total = len(processing_log)
        ok = sum(1 for p in processing_log if p.get("status") == "success")
        failed_reid = sum(1 for c in critic_results if not c.get("passed"))

        prompt = (
            f"De-identification complete. Processed {total} clips ({ok} succeeded). "
            f"Re-ID validation: {failed_reid} clip(s) failed threshold. "
            "Write a 2-sentence compliance summary for the researcher."
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception:
            return f"Processed {total} clips, {ok} succeeded. {failed_reid} re-ID failures."
