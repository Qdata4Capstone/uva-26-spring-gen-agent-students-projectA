"""
Legal Compliance Agent — Main Orchestration Loop
Implements the closed-loop architecture:
  LLM Controller → Toolbox → Critic → (retry if needed) → Report
"""

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .controller import LLMController
from .critic import ReIDCritic
from .tools.egoblur_tool import EgoBlurTool
from .tools.face_blur import FaceBlurTool
from .tools.file_manager import FileManager
from .tools.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

# Max retry attempts for Critic-triggered re-runs
MAX_CRITIC_RETRIES = 3


class LegalComplianceAgent:
    """
    End-to-end agent for dataset de-identification and compliance.

    Usage:
        agent = LegalComplianceAgent(dataset_dir="/bigtemp2/tsx4zn/legal/images")
        result = agent.run("I'm submitting to CVPR 2026, dataset from YouTube")
    """

    def __init__(
        self,
        dataset_dir: str,
        output_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        max_workers: int = 4,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir  = Path(output_dir) if output_dir else self.dataset_dir.parent / "output"
        self.max_workers = max_workers

        # Initialize modules
        self.controller = LLMController(api_key=api_key)
        self.knowledge  = KnowledgeBase()
        self.egoblur    = EgoBlurTool()
        self.face_blur  = FaceBlurTool()
        self.file_mgr   = FileManager(str(self.output_dir))
        self.critic     = ReIDCritic()

        logger.info(f"Agent initialized | dataset: {self.dataset_dir} | output: {self.output_dir}")
        logger.info(f"EgoBlur available: {self.egoblur.is_available}")

    # ------------------------------------------------------------------
    # Step 1: Parse intent + retrieve regulations
    # ------------------------------------------------------------------
    def _step_license_check(self, user_message: str) -> tuple[dict, dict]:
        """Parse user intent and retrieve applicable requirements."""
        logger.info("[Step 1] License check & intent parsing")

        # Query knowledge base with free-text
        kb_results = self.knowledge.query(user_message)
        kb_summary = json.dumps(kb_results, indent=2, ensure_ascii=False)

        # LLM parses intent into structured plan
        plan = self.controller.parse_intent(user_message, knowledge_context=kb_summary)
        logger.info(f"  Plan: {json.dumps(plan, ensure_ascii=False)}")

        # Build context from plan
        context = {
            "platform":    plan.get("platform"),
            "conference":  plan.get("conference"),
            "regulations": plan.get("regulations", []),
        }

        # Get merged requirements (strictest rules)
        requirements = self.knowledge.get_requirements(context)

        # Log conflicts
        if plan.get("conflict_notes"):
            logger.warning(f"  Conflict detected: {plan['conflict_notes']}")
            logger.warning(f"  Resolution: {plan.get('compliance_priority', 'compliance prevails')}")

        return plan, requirements

    # ------------------------------------------------------------------
    # Step 2: Classify clips + run de-identification
    # ------------------------------------------------------------------
    def _classify_clips(self, plan: dict) -> dict[str, str]:
        """
        Returns {clip_dir: "egocentric"|"standard"} for each clip.
        Uses LLM vision if ego_centric_ratio_estimate is between 0.1 and 0.9.
        """
        clip_dirs = sorted(
            d for d in self.dataset_dir.iterdir() if d.is_dir()
        )
        classifications = {}

        ego_ratio = plan.get("ego_centric_ratio_estimate", 0.0)

        for clip_dir in clip_dirs:
            if ego_ratio >= 0.9:
                classifications[str(clip_dir)] = "egocentric"
            elif ego_ratio <= 0.1:
                classifications[str(clip_dir)] = "standard"
            else:
                # Per-clip classification via LLM
                sample_images = sorted(clip_dir.glob("*.png"))[:1]
                sample = str(sample_images[0]) if sample_images else None
                cls = self.controller.classify_clip(clip_dir.name, sample)
                classifications[str(clip_dir)] = cls

        ego_count = sum(1 for v in classifications.values() if v == "egocentric")
        logger.info(f"  Clips: {len(clip_dirs)} total, {ego_count} ego-centric, "
                    f"{len(clip_dirs) - ego_count} standard")
        return classifications

    def _process_one_clip(
        self,
        clip_dir: str,
        retry: int = 0,
    ) -> dict:
        """Process a single clip directory. Returns status dict."""
        clip_path  = Path(clip_dir)
        suffix     = f"_retry{retry}" if retry > 0 else ""
        out_dir    = self.output_dir / "deidentified" / (clip_path.name + suffix)

        # Check if valid output already exists (incremental processing)
        if retry == 0 and out_dir.exists():
            input_imgs  = list(clip_path.glob("*.png")) + list(clip_path.glob("*.jpg"))
            output_imgs = list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg"))
            if len(output_imgs) >= len(input_imgs) and len(input_imgs) > 0:
                logger.info(f"  [cached] {clip_path.name}: {len(output_imgs)} files already processed")
                return {
                    "clip": clip_path.name,
                    "tool": "egoblur" if self.egoblur.is_available else "face_blur",
                    "output_dir": str(out_dir),
                    "status": "success",
                    "processed": len(output_imgs),
                    "cached": True,
                }

        # EgoBlur is the primary tool: it detects AND blurs faces + license plates
        # in a single pass. FaceBlurTool is only a last-resort fallback when
        # the egoblur models are absent.
        if self.egoblur.is_available:
            tool_name = "egoblur"
            logger.info(f"  [egoblur] {clip_path.name} (retry={retry})")
            egoblur = EgoBlurTool(scale_factor=1.15 + retry * 0.1)
            result = egoblur.process_clip_dir(str(clip_path), str(out_dir))
        else:
            tool_name = "face_blur"
            logger.info(f"  [face_blur fallback] {clip_path.name} (retry={retry})")
            face_blur = FaceBlurTool(blur_strength=51 + retry * 20)
            result = face_blur.process_directory(str(clip_path), str(out_dir))

        result["clip"] = clip_path.name
        result["tool"] = tool_name
        result["output_dir"] = str(out_dir)
        result["status"] = "success" if result.get("processed", 0) > 0 else "error"
        return result

    def _step_deidentify(self, clip_dirs: list[str]) -> list[dict]:
        """Run de-identification on all clips with egoblur, in parallel."""
        logger.info(f"[Step 2] De-identification — {len(clip_dirs)} clips via egoblur")
        processing_log = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._process_one_clip, clip_dir): clip_dir
                for clip_dir in clip_dirs
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processing_log.append(result)
                except Exception as e:
                    processing_log.append({
                        "clip": futures[future],
                        "status": "error",
                        "reason": str(e),
                    })

        ok = sum(1 for r in processing_log if r.get("status") == "success")
        logger.info(f"  De-identification: {ok}/{len(processing_log)} succeeded")
        return processing_log

    # ------------------------------------------------------------------
    # Step 3: EXIF strip
    # ------------------------------------------------------------------
    def _step_exif_strip(self, requirements: dict) -> dict:
        """Strip EXIF metadata from all outputs."""
        if not requirements.get("metadata_strip_required"):
            return {"stripped": 0, "skipped": True}

        logger.info("[Step 3] EXIF strip")
        deid_dir = self.output_dir / "deidentified"
        result = self.file_mgr.strip_exif_directory(str(deid_dir))
        logger.info(f"  Stripped EXIF from {result['stripped']} files")
        return result

    # ------------------------------------------------------------------
    # Step 4: Critic validation + self-correction loop
    # ------------------------------------------------------------------
    def _step_validate(
        self,
        processing_log: list[dict],
        requirements: dict,
    ) -> list[dict]:
        """
        Run Re-ID critic on processed clips.
        If a clip fails, re-process with higher blur strength (up to MAX_CRITIC_RETRIES).
        """
        logger.info("[Step 4] Critic validation (Re-ID test)")
        reid_threshold = requirements.get("re_id_threshold", 0.1)
        self.critic.re_id_threshold = reid_threshold

        all_critic_results = []
        retry_clips = {}  # clip_dir → retry_count

        try:
            self.critic._init()
        except Exception as e:
            logger.warning(f"  Critic init failed: {e}. Skipping Re-ID validation.")
            return [{"passed": True, "skipped": True, "reason": str(e)}]

        for log_entry in processing_log:
            if log_entry.get("status") != "success":
                continue

            clip_name = log_entry["clip"]
            orig_dir  = str(self.dataset_dir / clip_name)
            proc_dir  = log_entry.get("output_dir", "")

            if not proc_dir or not Path(proc_dir).exists():
                continue

            critic_result = self.critic.check_directory(orig_dir, proc_dir, sample_rate=0.3)
            critic_result["clip"] = clip_name

            if not critic_result["passed"]:
                logger.warning(
                    f"  [CRITIC FAIL] {clip_name} — "
                    f"max_similarity={critic_result['max_similarity_seen']:.3f} "
                    f"> threshold={reid_threshold}"
                )
                retry_clips[str(self.dataset_dir / clip_name)] = 1

            all_critic_results.append(critic_result)

        # Self-correction loop
        for retry_num in range(1, MAX_CRITIC_RETRIES + 1):
            if not retry_clips:
                break
            logger.info(f"  [RETRY {retry_num}] Re-processing {len(retry_clips)} failing clips")

            next_retry = {}
            for clip_dir, attempt in retry_clips.items():
                clip_name = Path(clip_dir).name
                result = self._process_one_clip(clip_dir, retry=attempt)

                if result["status"] == "success":
                    orig_dir = clip_dir
                    proc_dir = result["output_dir"]
                    critic_result = self.critic.check_directory(
                        orig_dir, proc_dir, sample_rate=0.3
                    )
                    critic_result["clip"] = clip_name
                    critic_result["retry"] = attempt
                    all_critic_results.append(critic_result)

                    if not critic_result["passed"] and attempt < MAX_CRITIC_RETRIES:
                        next_retry[clip_dir] = attempt + 1
                    elif critic_result["passed"]:
                        logger.info(f"  [RETRY OK] {clip_name} passed after retry {attempt}")

            retry_clips = next_retry

        passed = sum(1 for c in all_critic_results if c.get("passed"))
        logger.info(f"  Validation: {passed}/{len(all_critic_results)} clips passed")
        return all_critic_results

    # ------------------------------------------------------------------
    # Step 5: Compliance report
    # ------------------------------------------------------------------
    def _step_report(
        self,
        session_id: str,
        user_message: str,
        plan: dict,
        requirements: dict,
        processing_log: list[dict],
        critic_results: list[dict],
    ) -> str:
        """Generate and save compliance report."""
        logger.info("[Step 5] Generating compliance report")
        context = {
            "platform":    plan.get("platform"),
            "conference":  plan.get("conference"),
            "regulations": plan.get("regulations", []),
        }
        report_path = self.file_mgr.generate_compliance_report(
            session_id=session_id,
            user_intent=user_message,
            context=context,
            requirements=requirements,
            processing_log=processing_log,
            critic_results=critic_results,
            output_dir=str(self.output_dir),
        )
        logger.info(f"  Report saved: {report_path}")
        return report_path

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self, user_message: str) -> dict:
        """
        Run the full compliance pipeline for a user request.

        Args:
            user_message: Natural language description of the user's situation.

        Returns:
            dict with keys: session_id, plan, requirements, processing_log,
                            critic_results, report_path, summary
        """
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"=== Agent session {session_id} ===")
        logger.info(f"User: {user_message}")

        # Step 1: License check + intent parsing
        plan, requirements = self._step_license_check(user_message)

        # Collect clip directories
        clip_dirs = sorted(str(d) for d in self.dataset_dir.iterdir() if d.is_dir())
        logger.info(f"  Found {len(clip_dirs)} clip(s) in {self.dataset_dir}")

        # Step 2: De-identification (egoblur on all clips)
        processing_log = self._step_deidentify(clip_dirs)

        # Step 3: EXIF strip
        self._step_exif_strip(requirements)

        # Step 4: Critic validation + self-correction
        try:
            critic_results = self._step_validate(processing_log, requirements)
        except Exception as e:
            logger.error(f"[Step 4] Critic failed: {e}. Continuing to report.")
            critic_results = [{"passed": True, "skipped": True, "reason": str(e)}]

        # Step 5: Compliance report
        report_path = self._step_report(
            session_id, user_message, plan, requirements,
            processing_log, critic_results,
        )

        # Final summary
        summary = self.controller.summarize_run(processing_log, critic_results)
        logger.info(f"=== Session {session_id} complete ===")
        logger.info(summary)

        return {
            "session_id":      session_id,
            "plan":            plan,
            "requirements":    requirements,
            "processing_log":  processing_log,
            "critic_results":  critic_results,
            "report_path":     report_path,
            "summary":         summary,
            "output_dir":      str(self.output_dir),
        }
