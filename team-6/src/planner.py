"""
CardioAgent Planner
====================
Orchestrates Qwen3-VL (main reasoning) + ECGFounder + LingShu tools.
Manages the complete analysis workflow with step-by-step thinking.
"""

import json
import base64
import time
import os
from typing import Optional, Callable
from dataclasses import dataclass, field


@dataclass
class ThinkingStep:
    """One step in the agent's thinking process"""
    step_name: str
    status: str           # "running", "success", "error", "skipped"
    detail: str
    duration_s: float = 0
    substeps: list = field(default_factory=list)
    output_data: dict = field(default_factory=dict)


@dataclass
class AgentState:
    """Complete state of a single analysis run"""
    patient_id: str
    thinking_steps: list = field(default_factory=list)
    ecg_result: dict = field(default_factory=dict)
    mri_result: dict = field(default_factory=dict)
    planner_output: str = ""
    final_report: str = ""
    images: dict = field(default_factory=dict)
    started_at: float = 0
    finished_at: float = 0

    def add_step(self, step: ThinkingStep):
        self.thinking_steps.append(step)

    @property
    def elapsed_s(self):
        end = self.finished_at if self.finished_at else time.time()
        return round(end - self.started_at, 1) if self.started_at else 0


class CardioAgentPlanner:
    """
    Main orchestrator.

    Workflow:
    1. Receive raw data (ECG file, DICOM dir/file)
    2. Dispatch to ECGFounder tool → get ECG findings
    3. Dispatch to LingShu tool → get MRI findings
    4. Compile all findings + images
    5. Send everything to Qwen3-VL for integrated diagnosis
    6. Return structured result with full thinking trace
    """

    def __init__(
        self,
        qwen_api_url: str = "http://localhost:8000/v1",
        qwen_model_name: str = "gpt-3.5-turbo",
        lingshu_api_url: str = "http://localhost:8001/v1",
        ecg_checkpoint: Optional[str] = None,
        on_step_update: Optional[Callable] = None,
    ):
        self.qwen_url = qwen_api_url
        self.qwen_model = qwen_model_name
        self.on_step = on_step_update  # callback for GUI updates

        # Initialize tools
        from ecgfounder_tool import ECGFounderTool
        from lingshu_tool import LingShuTool

        self.ecg_tool = ECGFounderTool(
            checkpoint_path=ecg_checkpoint,
            output_dir="/tmp/cardioagent",
        )
        self.lingshu_tool = LingShuTool(
            lingshu_api_url=lingshu_api_url,
            output_dir="/tmp/cardioagent",
        )

    def _notify(self, step: ThinkingStep, state: AgentState):
        """Notify GUI of step update"""
        state.add_step(step)
        if self.on_step:
            self.on_step(step, state)

    def run(
        self,
        patient_id: str = "unknown",
        ecg_path: Optional[str] = None,
        dicom_input: Optional[str] = None,
        clinical_notes: str = "",
    ) -> AgentState:
        """
        Execute full CardioAgent pipeline.
        Returns AgentState with all thinking steps and results.
        """
        state = AgentState(patient_id=patient_id, started_at=time.time())

        # ── Step 0: Planning ──
        step = ThinkingStep(
            step_name="planning",
            status="success",
            detail=self._build_plan(ecg_path, dicom_input, clinical_notes),
        )
        self._notify(step, state)

        # ── Step 1: ECG Analysis ──
        if ecg_path and os.path.exists(ecg_path):
            step = ThinkingStep(step_name="ecg_analysis", status="running", detail="Starting ECG analysis...")
            self._notify(step, state)

            t0 = time.time()
            try:
                ecg_result = self.ecg_tool.analyze(ecg_path, patient_id)
                state.ecg_result = ecg_result

                if ecg_result.get("waveform_image"):
                    state.images["ecg_waveform"] = ecg_result["waveform_image"]

                step = ThinkingStep(
                    step_name="ecg_analysis",
                    status="success",
                    detail=self._summarize_ecg(ecg_result),
                    duration_s=round(time.time() - t0, 1),
                    substeps=ecg_result.get("steps", []),
                    output_data={"findings": ecg_result.get("findings", []), "metrics": ecg_result.get("metrics", {})},
                )
            except Exception as e:
                step = ThinkingStep(
                    step_name="ecg_analysis", status="error",
                    detail=f"ECG analysis failed: {e}", duration_s=round(time.time() - t0, 1),
                )
            self._notify(step, state)
        else:
            step = ThinkingStep(step_name="ecg_analysis", status="skipped", detail="No ECG data provided")
            self._notify(step, state)

        # ── Step 2: MRI Analysis ──
        if dicom_input and os.path.exists(dicom_input):
            step = ThinkingStep(step_name="mri_analysis", status="running", detail="Starting MRI analysis...")
            self._notify(step, state)

            t0 = time.time()
            try:
                mri_result = self.lingshu_tool.analyze(dicom_input, patient_id)
                state.mri_result = mri_result

                if mri_result.get("montage_path"):
                    state.images["mri_montage"] = mri_result["montage_path"]

                step = ThinkingStep(
                    step_name="mri_analysis",
                    status="success",
                    detail=self._summarize_mri(mri_result),
                    duration_s=round(time.time() - t0, 1),
                    substeps=mri_result.get("steps", []),
                    output_data={"findings": mri_result.get("findings", [])},
                )
            except Exception as e:
                step = ThinkingStep(
                    step_name="mri_analysis", status="error",
                    detail=f"MRI analysis failed: {e}", duration_s=round(time.time() - t0, 1),
                )
            self._notify(step, state)
        else:
            step = ThinkingStep(step_name="mri_analysis", status="skipped", detail="No MRI data provided")
            self._notify(step, state)

        # ── Step 3: Planner Synthesis (Qwen3-VL) ──
        step = ThinkingStep(step_name="planner_synthesis", status="running", detail="Qwen3-VL synthesizing...")
        self._notify(step, state)

        t0 = time.time()
        try:
            planner_output = self._call_qwen_planner(state, clinical_notes)
            state.planner_output = planner_output
            state.final_report = planner_output

            step = ThinkingStep(
                step_name="planner_synthesis",
                status="success",
                detail=f"Generated {len(planner_output)} chars of integrated analysis",
                duration_s=round(time.time() - t0, 1),
            )
        except Exception as e:
            # Fallback: generate report from tool findings alone
            state.final_report = self._fallback_report(state, clinical_notes)
            step = ThinkingStep(
                step_name="planner_synthesis",
                status="error",
                detail=f"Qwen3-VL call failed: {e}. Using fallback report.",
                duration_s=round(time.time() - t0, 1),
            )
        self._notify(step, state)

        state.finished_at = time.time()
        return state

    # ── Helper Methods ──

    def _build_plan(self, ecg_path, dicom_input, clinical_notes) -> str:
        """Describe what the agent will do"""
        modalities = []
        if ecg_path:
            modalities.append(f"ECG ({Path(ecg_path).suffix})")
        if dicom_input:
            modalities.append(f"MRI/CT ({'directory' if os.path.isdir(str(dicom_input)) else 'file'})")

        plan = f"Analyzing {len(modalities)} modalities: {', '.join(modalities)}.\n"
        plan += "Execution plan:\n"
        if ecg_path:
            plan += "  1. ECGFounder: read → preprocess → classify → waveform plot\n"
        if dicom_input:
            plan += "  2. LingShu-8B: DICOM→PNG → medical image analysis\n"
        plan += "  3. Qwen3-VL: synthesize all findings into integrated diagnosis"
        if clinical_notes:
            plan += f"\nClinical context: {clinical_notes[:100]}..."
        return plan

    def _summarize_ecg(self, result: dict) -> str:
        metrics = result.get("metrics", {})
        findings = result.get("findings", [])
        hr = metrics.get("heart_rate_bpm", "?")
        rhythm = metrics.get("rhythm_detail", "?")
        n_st = len(metrics.get("st_findings", []))
        return f"HR: {hr} bpm | {rhythm} | {n_st} ST abnormalities | {len(findings)} total findings"

    def _summarize_mri(self, result: dict) -> str:
        findings = result.get("findings", [])
        analysis = result.get("lingshu_analysis", "")
        return f"{len(findings)} findings | LingShu response: {len(analysis)} chars"

    def _call_qwen_planner(self, state: AgentState, clinical_notes: str) -> str:
        """Send all collected data to Qwen3-VL for final synthesis"""
        from openai import OpenAI

        client = OpenAI(base_url=self.qwen_url, api_key="cardioagent")
        content = []

        # Add images
        for key, img_path in state.images.items():
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

        # Build comprehensive text prompt
        prompt = "You are CardioAgent, an expert multi-modal cardiac diagnostic system.\n\n"

        if clinical_notes:
            prompt += f"## Clinical Context\n{clinical_notes}\n\n"

        # ECG findings
        if state.ecg_result:
            prompt += "## ECG Analysis (ECGFounder)\n"
            metrics = state.ecg_result.get("metrics", {})
            prompt += f"Heart Rate: {metrics.get('heart_rate_bpm', 'N/A')} bpm\n"
            prompt += f"Rhythm: {metrics.get('rhythm_detail', 'N/A')}\n"
            for st in metrics.get("st_findings", []):
                prompt += f"ST {st['type']} in {st['lead']}: {st['st_deviation_mv']:.2f} mV\n"
            for f in state.ecg_result.get("findings", []):
                prompt += f"Finding: {f['finding']} (severity: {f['severity']})\n"
            prompt += "\n"

        # MRI findings
        if state.mri_result:
            prompt += "## Cardiac MRI Analysis (LingShu)\n"
            prompt += state.mri_result.get("lingshu_analysis", "No analysis available")
            prompt += "\n\n"

        prompt += """## Your Task
Based on ALL available data (images above + tool findings), provide:

1. **Integrated Diagnosis**: What is the most likely diagnosis?
2. **Evidence Summary**: Key findings from each modality
3. **Cross-Modal Correlation**: Do ECG and MRI findings agree? Which coronary territory?
4. **Risk Assessment**: Low / Intermediate / High risk, with justification
5. **Confidence Level**: How confident are you? What additional data would help?
6. **Recommended Next Steps**: Further testing, treatment considerations

Be specific and reference the actual findings from the tools above."""

        content.append({"type": "text", "text": prompt})

        response = client.chat.completions.create(
            model=self.qwen_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=3000,
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _fallback_report(self, state: AgentState, clinical_notes: str) -> str:
        """Generate report without Qwen3-VL (when server is unavailable)"""
        parts = ["# CardioAgent Analysis Report (Offline Mode)\n"]

        if clinical_notes:
            parts.append(f"**Clinical Context:** {clinical_notes}\n")

        if state.ecg_result:
            parts.append("## ECG Findings")
            metrics = state.ecg_result.get("metrics", {})
            parts.append(f"- Heart Rate: {metrics.get('heart_rate_bpm', 'N/A')} bpm")
            parts.append(f"- Rhythm: {metrics.get('rhythm_detail', 'N/A')}")
            for f in state.ecg_result.get("findings", []):
                parts.append(f"- {f['finding']} [{f['severity']}]")
            parts.append("")

        if state.mri_result:
            parts.append("## MRI Findings")
            for f in state.mri_result.get("findings", []):
                parts.append(f"- {f.get('finding', 'N/A')} [{f.get('severity', 'N/A')}]")
            parts.append("")

        parts.append("*Note: Full Qwen3-VL synthesis unavailable. Showing raw tool outputs.*")
        return "\n".join(parts)


# Allow import of Path
from pathlib import Path