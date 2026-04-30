"""
interface_v3.py — Chainlit-based MedRAX interface

Standalone:
    chainlit run team-07/src/interface_v3.py

Programmatic (inject a pre-built agent):
    from interface_v3 import set_agent
    set_agent(agent, tools_dict)
    # then start chainlit externally
"""

import os
import base64
import json
import re
import shutil
import time
from pathlib import Path

import chainlit as cl

# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

TOOL_ICONS = {
    "chest_xray_classifier":       "📊",
    "chest_xray_report_generator": "📋",
    "grad_cam_explainer":          "🔥",
    "image_visualizer":            "🖼️",
    "dicom_processor":             "📁",
    "chest_xray_segmentation":     "🫁",
    "xray_vqa":                    "💬",
    "xray_phrase_grounding":       "📍",
    "llava_med":                   "🔬",
}

TOOL_LABELS = {
    "chest_xray_classifier":       "Chest X-ray Classifier",
    "chest_xray_report_generator": "Report Generator",
    "grad_cam_explainer":          "GradCAM Explainer",
    "image_visualizer":            "Image Visualizer",
    "dicom_processor":             "DICOM Processor",
    "chest_xray_segmentation":     "Segmentation",
    "xray_vqa":                    "X-ray VQA",
    "xray_phrase_grounding":       "Phrase Grounding",
    "llava_med":                   "LLaVA-Med",
}

# ---------------------------------------------------------------------------
# Tool output formatters
# ---------------------------------------------------------------------------

def _extract_score(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    if isinstance(value, dict):
        for k in ("score", "probability", "confidence", "value", "prob"):
            if k in value:
                return _extract_score(value[k])
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
    return 0.0


def _fmt_classifier(raw: str) -> str:
    try:
        data  = json.loads(raw)
        preds = data.get("predictions") or data.get("results") or data.get("scores") or data
        if isinstance(preds, dict):
            top   = sorted(preds.items(), key=lambda x: _extract_score(x[1]), reverse=True)[:8]
            lines = ["**Top predictions**\n"]
            for label, score in top:
                pct = _extract_score(score) * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                lines.append(f"`{label:<38}` {bar} **{pct:.1f}%**")
            return "\n".join(lines)
    except Exception:
        pass
    return raw


def _fmt_report(raw: str) -> str:
    try:
        data   = json.loads(raw)
        report = data.get("report") or data.get("text") or data.get("output") or raw
        return str(report).strip()
    except Exception:
        return raw.strip()


def _fmt_gradcam(raw: str) -> str:
    try:
        data = json.loads(raw)
        desc = (data.get("description") or data.get("interpretation")
                or data.get("text") or data.get("output") or raw)
        return str(desc).strip()
    except Exception:
        lines = [ln for ln in raw.splitlines() if not ln.startswith("HEATMAP_PATH:")]
        return "\n".join(lines).strip()


def _fmt_generic(raw: str) -> str:
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            data = data[0]
        return f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
    except Exception:
        return raw.strip()


TOOL_FORMATTERS = {
    "chest_xray_classifier":       _fmt_classifier,
    "chest_xray_report_generator": _fmt_report,
    "grad_cam_explainer":          _fmt_gradcam,
}


_ABS_PATH_RE = re.compile(r"(?<![\w./])/(?:[\w.\-]+/)+[\w.\-]+")


def _redact_paths(text: str) -> str:
    """Replace absolute filesystem paths with `[basename]` so internal
    directory structure is not exposed when the UI is shared publicly."""
    if not text:
        return text
    def _sub(m):
        base = m.group(0).rsplit("/", 1)[-1]
        return f"[{base}]" if base else "[file]"
    return _ABS_PATH_RE.sub(_sub, text)


def _format_tool_output(tool_name: str, raw_content: str) -> str:
    key       = tool_name.lower().replace("tool", "").replace("-", "_").strip("_")
    formatter = TOOL_FORMATTERS.get(key, _fmt_generic)
    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, list):
            raw_content = json.dumps(parsed[0]) if parsed else raw_content
    except Exception:
        pass
    return _redact_paths(formatter(raw_content))


# ---------------------------------------------------------------------------
# Clinical assessment card (Project B: structured radiologist output)
# ---------------------------------------------------------------------------

_CLINICAL_KEYS = {
    "history_summary", "imaging_findings", "differentials",
    "final_diagnosis", "confidence", "reasoning",
}


def _parse_clinical_json(text: str) -> dict | None:
    """
    Extract a clinical-assessment JSON object from the agent's response.
    Looks for ```json ... ``` fences first, then falls back to any {...} block.
    Returns the parsed dict only if at least 3 expected keys are present.
    """
    if not text:
        return None

    candidates: list[str] = []

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates.extend(fenced)

    # Fallback: greedy outermost braces
    if not candidates:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            candidates.append(m.group(0))

    for raw in candidates:
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if isinstance(data, dict) and len(set(data.keys()) & _CLINICAL_KEYS) >= 3:
            return data
    return None


def _render_clinical_card(data: dict) -> str:
    """Render a clinical-assessment dict as a Markdown card."""
    history = data.get("history_summary") or []
    if isinstance(history, str):
        history = [history]

    findings      = (data.get("imaging_findings") or "").strip()
    differentials = data.get("differentials") or []
    if isinstance(differentials, str):
        differentials = [differentials]

    diagnosis  = (data.get("final_diagnosis") or "").strip()
    confidence = (data.get("confidence") or "").strip().lower()
    reasoning  = (data.get("reasoning") or "").strip()

    badges = {
        "high":   "🟢 **HIGH**",
        "medium": "🟡 **MEDIUM**",
        "low":    "🔴 **LOW**",
    }
    badge = badges.get(confidence, f"⚪ **{(confidence or 'N/A').upper()}**")

    out: list[str] = ["### 🩻 Clinical Assessment", ""]

    if history:
        out.append("**📋 History Summary**")
        for fact in history:
            out.append(f"- {fact}")
        out.append("")

    if findings:
        out.append("**🔍 Imaging Findings**")
        out.append(findings)
        out.append("")

    if differentials:
        out.append("**🩺 Differential Diagnoses**")
        for i, dx in enumerate(differentials[:5], 1):
            out.append(f"{i}. {dx}")
        out.append("")

    if diagnosis:
        out.append("**🎯 Final Diagnosis**")
        out.append(f"{diagnosis} · {badge} confidence")
        out.append("")

    if reasoning:
        out.append("**💭 Reasoning**")
        out.append(reasoning)

    return "\n".join(out).rstrip()


# ---------------------------------------------------------------------------
# Module-level agent (initialized once, shared across all sessions)
# ---------------------------------------------------------------------------

_agent:      object | None = None
_tools_dict: dict  | None  = None
_upload_dir = Path(__file__).parent / "temp"


def set_agent(agent, tools_dict: dict) -> None:
    """Inject an already-initialized agent (called from main.py or tests)."""
    global _agent, _tools_dict
    _agent      = agent
    _tools_dict = tools_dict


# ---------------------------------------------------------------------------
# App startup — initialize once when `chainlit run` starts the server
# ---------------------------------------------------------------------------

@cl.on_app_startup
async def on_app_startup():
    """
    Load the MedRAX agent once when the Chainlit server starts.
    Skipped if set_agent() was already called (programmatic injection).
    """
    global _agent, _tools_dict

    if _agent is not None:
        return

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[interface_v3] OPENAI_API_KEY not set — agent will not be available.")
        return

    try:
        import sys
        _src = Path(__file__).parent
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))

        from main import initialize_agent

        model_dir = os.getenv("MODEL_DIR", str(_src / "mydownload"))
        temp_dir  = os.getenv("TEMP_DIR", str(_src / "mydownload" / "temp"))
        prompt_file = os.getenv(
            "PROMPT_FILE",
            str(_src / "medrax" / "docs" / "system_prompts.txt"),
        )
        model  = os.getenv("OPENAI_MODEL", "gpt-4o")
        device = os.getenv("DEVICE", "cuda")

        openai_kwargs: dict = {"api_key": api_key}
        if base_url := os.getenv("OPENAI_BASE_URL"):
            openai_kwargs["base_url"] = base_url

        selected_tools = [
            "ImageVisualizerTool",
            "DicomProcessorTool",
            "ChestXRayClassifierTool",
            "ChestXRaySegmentationTool",
            "ChestXRayReportGeneratorTool",
            "XRayVQATool",
            "LlavaMedTool",
            "XRayPhraseGroundingTool",
            "GradCAMExplainerTool",
        ]

        _agent, _tools_dict = initialize_agent(
            prompt_file   = prompt_file,
            tools_to_use  = selected_tools,
            model_dir     = model_dir,
            temp_dir      = temp_dir,
            device        = device,
            model         = model,
            temperature   = 0.7,
            top_p         = 0.95,
            openai_kwargs = openai_kwargs,
        )
        print("[interface_v3] Agent initialized.")

    except Exception as exc:
        print(f"[interface_v3] Agent init failed: {exc}")


# ---------------------------------------------------------------------------
# Starter prompts shown in a fresh chat
# ---------------------------------------------------------------------------

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Structured clinical assessment",
            message=(
                "Provide a structured clinical assessment of this case. "
                "Include history summary, imaging findings, top-3 differentials, "
                "final diagnosis, and confidence level."
            ),
            icon="/public/learn.svg",
        ),
        cl.Starter(
            label="Classify pathologies",
            message="Please classify any abnormalities visible in this X-ray.",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Explain with GradCAM",
            message="Show me a GradCAM heatmap highlighting the most abnormal regions.",
            icon="/public/terminal.svg",
        ),
        cl.Starter(
            label="Full analysis",
            message=(
                "Perform a comprehensive analysis: classify pathologies, "
                "generate a report, and highlight areas of concern."
            ),
            icon="/public/write.svg",
        ),
    ]


# ---------------------------------------------------------------------------
# Chat session start
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    _upload_dir.mkdir(exist_ok=True)
    cl.user_session.set("thread_id",          str(time.time()))
    cl.user_session.set("original_file_path", None)
    cl.user_session.set("display_file_path",  None)

    status = "ready" if _agent is not None else "⚠️ agent not initialized"
    await cl.Message(
        content=(
            f"## 🫁 Medical Reasoning Agent for Chest X-ray\n\n"
            "Upload a chest X-ray (JPEG / PNG / DICOM) and ask me anything about it.\n\n"
            "_Tip: drag a file into the chat box or click the attachment icon._"
        ),
    ).send()


# ---------------------------------------------------------------------------
# Main message handler
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    if _agent is None:
        await cl.Message(
            content="⚠️ Agent is not initialized. Check your API key and model config."
        ).send()
        return

    thread_id          = cl.user_session.get("thread_id")
    original_file_path = cl.user_session.get("original_file_path")
    display_file_path  = cl.user_session.get("display_file_path")

    # ── Handle uploaded files ─────────────────────────────────────────────
    # Chainlit delivers user uploads as cl.File OR cl.Image (sibling subclasses
    # of Element), depending on detected MIME type. Accept either by inspecting
    # the `path` attribute directly.
    _upload_dir.mkdir(parents=True, exist_ok=True)
    for element in message.elements or []:
        src_path = getattr(element, "path", None)
        if not src_path:
            print(f"[interface_v3] skipping element with no path: {type(element).__name__}")
            continue
        if not Path(src_path).exists():
            print(f"[interface_v3] skipping element — path does not exist: {src_path}")
            continue

        source = Path(src_path)
        suffix = source.suffix.lower() or ".jpg"
        saved  = _upload_dir / f"upload_{int(time.time() * 1000)}{suffix}"
        shutil.copy2(str(source), str(saved))
        original_file_path = str(saved)

        if suffix == ".dcm" and _tools_dict and "DicomProcessorTool" in _tools_dict:
            output, _ = _tools_dict["DicomProcessorTool"]._run(str(saved))
            display_file_path = output["image_path"]
        else:
            display_file_path = str(saved)

        cl.user_session.set("original_file_path", original_file_path)
        cl.user_session.set("display_file_path",  display_file_path)
        print(f"[interface_v3] saved upload → {original_file_path}  (display: {display_file_path})")

        if display_file_path and Path(display_file_path).exists():
            await cl.Message(
                content="",
                elements=[cl.Image(
                    path    = display_file_path,
                    name    = "uploaded-xray",
                    display = "inline",
                )],
            ).send()

    # ── Build agent input messages ────────────────────────────────────────
    # Always send the display (PNG/JPEG) path to the multimodal model — DICOM
    # bytes can't be inlined as image/jpeg.
    image_path_for_llm = display_file_path or original_file_path
    image_path_for_tool = original_file_path or display_file_path
    agent_messages = []

    if image_path_for_llm and Path(image_path_for_llm).exists():
        agent_messages.append({"role": "user",
                               "content": f"image_path: {image_path_for_tool}"})
        suffix = Path(image_path_for_llm).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        with open(image_path_for_llm, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode("utf-8")
        agent_messages.append({
            "role":    "user",
            "content": [{"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{img_b64}"}}],
        })
        print(f"[interface_v3] attached image to LLM payload: {image_path_for_llm} ({mime})")
    else:
        print(f"[interface_v3] no image attached — original={original_file_path} "
              f"display={display_file_path}")

    if message.content and message.content.strip():
        agent_messages.append({
            "role":    "user",
            "content": [{"type": "text", "text": message.content}],
        })

    if not agent_messages:
        await cl.Message(content="Please provide a question or upload an image.").send()
        return

    # Did the user actually ask for segmentation this turn? If not, the seg
    # tool may still run (the agent decides), but we won't surface its image.
    user_text_lc      = (message.content or "").lower()
    wants_segmentation = any(
        kw in user_text_lc
        for kw in ("segment", "outline", "boundary", "boundaries", "mask")
    )

    # ── Stream through the LangGraph agent ───────────────────────────────
    answer_msg   = cl.Message(content="")
    answer_sent  = False
    seg_images: list[str] = []   # collected, attached to final answer if wanted

    try:
        for event in _agent.workflow.stream(
            {"messages": agent_messages},
            {"configurable": {"thread_id": thread_id}},
        ):
            if not isinstance(event, dict):
                continue

            # ── Tool execution ─────────────────────────────────────────
            if "execute" in event:
                for tool_msg in event["execute"]["messages"]:
                    tool_name   = tool_msg.name
                    raw_content = tool_msg.content
                    if not raw_content:
                        continue

                    key       = tool_name.lower().replace("tool", "").strip("_")
                    icon      = TOOL_ICONS.get(key, "🔧")
                    label     = TOOL_LABELS.get(key, tool_name)
                    formatted = _format_tool_output(tool_name, str(raw_content))

                    async with cl.Step(
                        name         = f"{icon}  {label}",
                        type         = "tool",
                        default_open = False,
                    ) as step:
                        step.output = formatted

                    # Parse tool result for image side-effects
                    tool_result = None
                    try:
                        parsed = json.loads(raw_content)
                        tool_result = parsed[0] if isinstance(parsed, list) else parsed
                    except Exception:
                        tool_result = raw_content

                    img_path = _extract_image_path(key, tool_result, raw_content)
                    if img_path and Path(img_path).exists():
                        # Segmentation: defer to the final answer (and only
                        # if the user explicitly asked for it).
                        if "segmentation" in key:
                            if wants_segmentation:
                                seg_images.append(img_path)
                                cl.user_session.set("display_file_path", img_path)
                            # else: silently drop — tool step still shows it ran
                        else:
                            # gradcam / image_visualizer keep current behaviour
                            display_file_path = img_path
                            cl.user_session.set("display_file_path", display_file_path)
                            await cl.Message(
                                content  = "",
                                elements = [cl.Image(
                                    path    = img_path,
                                    name    = key,
                                    display = "inline",
                                )],
                            ).send()

            # ── Agent synthesis / final answer ─────────────────────────
            elif "process" in event:
                content = event["process"]["messages"][-1].content
                if not content:
                    continue
                content = _redact_paths(content).strip()
                if not content:
                    continue

                # If the agent returned a structured clinical JSON, render it
                # as a radiology-report card instead of raw text.
                clinical = _parse_clinical_json(content)
                rendered = _render_clinical_card(clinical) if clinical else content

                answer_msg.content = rendered
                if not answer_sent:
                    await answer_msg.send()
                    answer_sent = True
                else:
                    await answer_msg.update()

    except Exception as exc:
        await cl.Message(content=f"⚠️ Error: {exc}").send()
        return

    # Attach segmentation outputs to the model's reply (only if requested)
    if seg_images and wants_segmentation:
        seg_elements = [
            cl.Image(path=p, name="segmentation", display="inline")
            for p in seg_images
        ]
        if answer_sent:
            answer_msg.elements = seg_elements
            await answer_msg.update()
        else:
            await cl.Message(content="", elements=seg_elements).send()
            answer_sent = True

    if not answer_sent:
        await cl.Message(content="_(No response from agent.)_").send()


# ---------------------------------------------------------------------------
# Helper: extract image paths produced by tool side-effects
# ---------------------------------------------------------------------------

def _extract_image_path(key: str, tool_result, raw_content: str) -> str | None:
    if "image_visualizer" in key:
        if isinstance(tool_result, dict):
            return tool_result.get("image_path")

    if "gradcam" in key or "grad_cam" in key:
        if isinstance(tool_result, dict):
            path = tool_result.get("heatmap_saved_to")
            if path:
                return path
        if isinstance(raw_content, str):
            for line in raw_content.splitlines():
                if line.startswith("HEATMAP_PATH:"):
                    return line.split("HEATMAP_PATH:", 1)[-1].strip()

    if "segmentation" in key:
        if isinstance(tool_result, dict):
            return tool_result.get("image_path") or tool_result.get("output_path")

    return None
