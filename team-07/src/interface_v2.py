import re
import json
import base64
import shutil
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import gradio as gr
from gradio import ChatMessage


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CSS = """
/* ── Base ── */
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif !important;
    background: #f5f4f0 !important;
    color: #1a1a1a !important;
}

/* Force all text dark */
*, p, span, div, label, textarea, input, button {
    color: #1a1a1a;
}

/* Markdown text */
.prose, .prose p, .prose li, .md p, .md li {
    color: #1a1a1a !important;
}

/* ── Header ── */
#header {
    padding: 18px 28px 12px;
    background: #ffffff;
    border-bottom: 1px solid #e5e5e5;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ── Main panels ── */
#chat-panel {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e5e5;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
#image-panel {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e5e5;
    overflow: hidden;
    padding: 16px;
}

/* ── Chatbot messages ── */
#chatbot {
    background: transparent !important;
    border: none !important;
    padding: 8px 4px !important;
}
#chatbot .message-wrap {
    padding: 6px 20px !important;
}

/* User bubble */
#chatbot .user .message {
    background: #ede9fe !important;
    color: #1a1a1a !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 10px 16px !important;
    max-width: 80% !important;
    font-size: 0.9rem !important;
    box-shadow: none !important;
    border: none !important;
}

/* Assistant bubble */
#chatbot .bot .message {
    background: #f9f9f7 !important;
    color: #1a1a1a !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 10px 16px !important;
    max-width: 88% !important;
    font-size: 0.9rem !important;
    border: 1px solid #ebebeb !important;
    box-shadow: none !important;
}

/* ── Grouped analysis block — styled entirely via inline HTML, no overrides here ── */
#chatbot details summary::-webkit-details-marker,
.chatbot details summary::-webkit-details-marker { display: none; }

/* ── Final answer bubble — slightly larger, no border ── */
#chatbot .bot:last-of-type .message {
    background: #ffffff !important;
    border: none !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
}

/* ── Input row ── */
#input-row {
    padding: 12px 16px;
    border-top: 1px solid #ebebeb;
    background: #ffffff;
    display: flex;
    gap: 8px;
    align-items: flex-end;
}
#msg-box textarea {
    border-radius: 22px !important;
    border: 1.5px solid #d8d8d8 !important;
    padding: 10px 18px !important;
    font-size: 0.9rem !important;
    resize: none !important;
    background: #fafafa !important;
    color: #1a1a1a !important;
    transition: border-color 0.15s;
}
#msg-box textarea:focus {
    border-color: #7c3aed !important;
    outline: none !important;
    background: #ffffff !important;
    color: #1a1a1a !important;
}
#send-btn {
    border-radius: 50% !important;
    min-width: 40px !important;
    max-width: 40px !important;
    height: 40px !important;
    padding: 0 !important;
    background: #7c3aed !important;
    color: white !important;
    border: none !important;
    font-size: 1.1rem !important;
    cursor: pointer !important;
    transition: background 0.15s;
}
#send-btn:hover { background: #6d28d9 !important; }

/* ── Image panel ── */
#image-display {
    border-radius: 10px !important;
    border: 1px solid #ebebeb !important;
    overflow: hidden !important;
}
#image-display img { object-fit: contain !important; }

/* ── Upload / action buttons ── */
.upload-btn, .action-btn {
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 7px 14px !important;
    border: 1px solid #d8d8d8 !important;
    background: #fafafa !important;
    color: #444 !important;
    cursor: pointer !important;
    transition: background 0.12s, border-color 0.12s;
}
.upload-btn:hover, .action-btn:hover {
    background: #f0eeff !important;
    border-color: #7c3aed !important;
    color: #7c3aed !important;
}
#clear-btn {
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    background: transparent !important;
    color: #888 !important;
    border: 1px solid #e0e0e0 !important;
}
#clear-btn:hover {
    background: #fff0f0 !important;
    color: #dc2626 !important;
    border-color: #dc2626 !important;
}

/* ── Status pill ── */
#status-bar {
    font-size: 0.75rem;
    color: #888;
    padding: 4px 20px 6px;
    min-height: 22px;
}
"""


# ---------------------------------------------------------------------------
# Tool output formatters
# ---------------------------------------------------------------------------

def _extract_score(value) -> float:
    """Safely extract a numeric score from a value that may be a float, str, or dict."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    if isinstance(value, dict):
        # Try common keys
        for key in ("score", "probability", "confidence", "value", "prob"):
            if key in value:
                return _extract_score(value[key])
        # Fall back to first numeric value found
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
    return 0.0


def _fmt_classifier(raw: str) -> str:
    try:
        data  = json.loads(raw)
        preds = data.get("predictions") or data.get("results") or data.get("scores") or data
        if isinstance(preds, dict):
            sorted_preds = sorted(preds.items(), key=lambda x: _extract_score(x[1]), reverse=True)
            lines = ["Top predictions:\n"]
            for label, score in sorted_preds[:8]:
                pct = _extract_score(score) * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                lines.append(f"  {label:<40} {bar} {pct:5.1f}%")
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
        lines = [l for l in raw.splitlines() if not l.startswith("HEATMAP_PATH:")]
        return "\n".join(lines).strip()


def _fmt_generic(raw: str) -> str:
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            data = data[0]
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return raw.strip()


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

TOOL_FORMATTERS = {
    "chest_xray_classifier":       _fmt_classifier,
    "chest_xray_report_generator": _fmt_report,
    "grad_cam_explainer":          _fmt_gradcam,
}


def _format_tool_output(tool_name: str, raw_content: str) -> str:
    key = tool_name.lower().replace("tool", "").replace("-", "_").strip("_")
    formatter = TOOL_FORMATTERS.get(key, _fmt_generic)
    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, list):
            raw_content = json.dumps(parsed[0]) if parsed else raw_content
    except Exception:
        pass
    return formatter(raw_content)


# ---------------------------------------------------------------------------
# One-line summary per tool  (used in grouped analysis block)
# ---------------------------------------------------------------------------

def _one_line_summary(tool_name: str, raw_content: str) -> str:
    """Extract a single informative line from a tool result."""
    key = tool_name.lower().replace("tool", "").replace("-", "_").strip("_")
    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
    except Exception:
        parsed = raw_content

    # Classifier → top 2 predictions
    if "classifier" in key:
        preds = {}
        if isinstance(parsed, dict):
            preds = parsed.get("predictions") or parsed.get("scores") or parsed
        if isinstance(preds, dict) and preds:
            top = sorted(preds.items(), key=lambda x: _extract_score(x[1]), reverse=True)[:2]
            return "  ·  ".join(f"{l} {_extract_score(s)*100:.0f}%" for l, s in top)

    # Report generator → first sentence of impression/findings
    if "report" in key:
        text = ""
        if isinstance(parsed, dict):
            text = parsed.get("report") or parsed.get("impression") or parsed.get("text") or ""
        text = str(text or raw_content).strip()
        # Take first sentence (up to 120 chars)
        first = re.split(r'(?<=[.!?])\s', text)[0]
        return first[:120] + ("…" if len(first) > 120 else "")

    # GradCAM → region mentioned
    if "gradcam" in key or "grad_cam" in key:
        text = ""
        if isinstance(parsed, dict):
            text = parsed.get("description") or parsed.get("interpretation") or parsed.get("text") or ""
        text = str(text or raw_content).strip()
        lines = [l for l in text.splitlines() if l and not l.startswith("HEATMAP_PATH:")]
        first = lines[0] if lines else text
        return first[:120] + ("…" if len(first) > 120 else "")

    # Generic → first non-empty line
    text = str(parsed if isinstance(parsed, str) else json.dumps(parsed))
    first = next((l.strip() for l in text.splitlines() if l.strip()), text)
    return first[:120] + ("…" if len(first) > 120 else "")


def _build_analysis_block(tool_buffer: list, running: bool = False) -> ChatMessage:
    """
    Build a single grouped ChatMessage rendered as a collapsed <details> block.
    Uses raw HTML with inline styles so colours are theme-independent.
    """
    rows = []
    for name, summary in tool_buffer:
        key   = name.lower().replace("tool", "").strip("_")
        icon  = TOOL_ICONS.get(key, "🔧")
        label = TOOL_LABELS.get(key, name)
        rows.append(
            f'<div style="padding:2px 0;color:#3b1f8c;">'
            f'<span style="color:#7c3aed;margin-right:6px;">✓</span>'
            f'<span style="margin-right:6px;">{icon}</span>'
            f'<span style="display:inline-block;min-width:200px;color:#3b1f8c;">{label}</span>'
            f'<span style="color:#555;font-size:0.85em;">{summary}</span>'
            f'</div>'
        )

    if running:
        rows.append(
            '<div style="color:#7c3aed;padding:2px 0;">⟳ &nbsp;running…</div>'
        )

    n        = len(tool_buffer)
    title    = f"🔍 Analysis &nbsp;·&nbsp; {n} tool{'s' if n != 1 else ''} used"
    rows_html = "\n".join(rows)

    html = (
        f'<details style="background:#f4f3ff;border:1px solid #c4b8ff;'
        f'border-radius:10px;margin:4px 0;overflow:hidden;">'
        f'<summary style="padding:7px 14px;font-size:0.78rem;font-weight:600;'
        f'color:#6d28d9;background:#ede9fe;cursor:pointer;list-style:none;'
        f'user-select:none;">{title}</summary>'
        f'<div style="padding:8px 14px;font-family:monospace;font-size:0.78rem;'
        f'background:#f4f3ff;line-height:1.8;">'
        f'{rows_html}'
        f'</div>'
        f'</details>'
    )
    return ChatMessage(role="assistant", content=html)


# ---------------------------------------------------------------------------
# ChatInterface
# ---------------------------------------------------------------------------

class ChatInterface:
    def __init__(self, agent, tools_dict):
        self.agent        = agent
        self.tools_dict   = tools_dict
        self.upload_dir   = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id  = None
        self.original_file_path = None
        self.display_file_path  = None

    def handle_upload(self, file_path: str) -> str:
        if not file_path:
            return None
        source    = Path(file_path)
        timestamp = int(time.time())
        suffix    = source.suffix.lower()
        saved     = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved)
        self.original_file_path = str(saved)
        if suffix == ".dcm":
            output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved))
            self.display_file_path = output["image_path"]
        else:
            self.display_file_path = str(saved)
        return self.display_file_path

    def add_message(
        self, message: str, display_image: str, history: List[dict]
    ) -> Tuple[List[dict], gr.Textbox]:
        image_path = self.original_file_path or display_image
        if image_path is not None:
            history.append({"role": "user", "content": {"path": image_path}})
        if message:
            history.append({"role": "user", "content": message})
        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(
        self, message: str, display_image: Optional[str], chat_history: List[ChatMessage]
    ) -> AsyncGenerator:
        chat_history = chat_history or []

        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages   = []
        image_path = self.original_file_path or display_image

        if image_path is not None:
            messages.append({"role": "user", "content": f"image_path: {image_path}"})
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            messages.append({
                "role": "user",
                "content": [{"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}],
            })

        if message:
            messages.append({"role": "user",
                             "content": [{"type": "text", "text": message}]})

        # ── Per-turn state ────────────────────────────────────────────────
        tool_buffer   = []   # (tool_name, one_line_summary) accumulated this turn
        analysis_idx  = None # index in chat_history of the grouped analysis block
        pending_imgs  = []   # image paths produced by tools, shown after final answer

        try:
            for event in self.agent.workflow.stream(
                {"messages": messages},
                {"configurable": {"thread_id": self.current_thread_id}},
            ):
                if not isinstance(event, dict):
                    continue

                # ── Tool execution — buffer and show grouped block ─────────
                if "execute" in event:
                    for msg in event["execute"]["messages"]:
                        tool_name   = msg.name
                        raw_content = msg.content
                        if not raw_content:
                            continue

                        # 1-line summary for the grouped block
                        summary = _one_line_summary(tool_name, str(raw_content))
                        tool_buffer.append((tool_name, summary))

                        # Insert or update the single analysis block
                        block = _build_analysis_block(tool_buffer, running=True)
                        if analysis_idx is None:
                            chat_history.append(block)
                            analysis_idx = len(chat_history) - 1
                        else:
                            chat_history[analysis_idx] = block

                        # Collect image side-effects (show after final answer)
                        tool_result = None
                        try:
                            parsed      = json.loads(raw_content)
                            tool_result = parsed[0] if isinstance(parsed, list) else parsed
                        except Exception:
                            tool_result = raw_content

                        norm = tool_name.lower()

                        if "image_visualizer" in norm:
                            if isinstance(tool_result, dict):
                                p = tool_result.get("image_path")
                                if p:
                                    self.display_file_path = p
                                    pending_imgs.append(p)

                        elif "gradcam" in norm or "grad_cam" in norm:
                            heatmap = None
                            if isinstance(tool_result, dict):
                                heatmap = tool_result.get("heatmap_saved_to")
                            if not heatmap and isinstance(raw_content, str):
                                for line in raw_content.splitlines():
                                    if line.startswith("HEATMAP_PATH:"):
                                        heatmap = line.split("HEATMAP_PATH:", 1)[-1].strip()
                                        break
                            if heatmap and Path(heatmap).exists():
                                self.display_file_path = heatmap
                                pending_imgs.append(heatmap)

                        elif "segmentation" in norm:
                            seg = None
                            if isinstance(tool_result, dict):
                                seg = (tool_result.get("image_path")
                                       or tool_result.get("output_path"))
                            if seg and Path(seg).exists():
                                self.display_file_path = seg
                                pending_imgs.append(seg)

                        yield chat_history, self.display_file_path, ""

                # ── Agent reasoning / final answer ────────────────────────
                elif "process" in event:
                    content = event["process"]["messages"][-1].content
                    if not content:
                        continue
                    content = re.sub(r"temp/[^\s]*", "", content).strip()

                    # Finalise the analysis block (remove the ⟳ spinner)
                    if analysis_idx is not None and tool_buffer:
                        chat_history[analysis_idx] = _build_analysis_block(
                            tool_buffer, running=False
                        )
                        # Reset so the next tool group gets a fresh block
                        tool_buffer  = []
                        analysis_idx = None

                    # Show any tool-generated images before the final answer
                    for img_path in pending_imgs:
                        if Path(img_path).exists():
                            chat_history.append(
                                ChatMessage(role="assistant",
                                            content={"path": img_path}))
                    pending_imgs = []

                    # Final answer — wrapped in a div for distinct styling
                    chat_history.append(ChatMessage(
                        role="assistant",
                        content=content,
                    ))
                    yield chat_history, self.display_file_path, ""

        except Exception as e:
            chat_history.append(ChatMessage(
                role="assistant",
                content=f"Something went wrong: {e}",
                metadata={"title": "⚠️ Error"},
            ))
            yield chat_history, self.display_file_path, ""


# ---------------------------------------------------------------------------
# Demo builder
# ---------------------------------------------------------------------------

def create_demo(agent, tools_dict):
    interface = ChatInterface(agent, tools_dict)

    with gr.Blocks(css=CSS, title="MedRAX", theme=gr.themes.Soft()) as demo:

        # ── Header ───────────────────────────────────────────────────────
        with gr.Row(elem_id="header"):
            gr.HTML("""
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:36px;height:36px;background:#7c3aed;border-radius:8px;
                                display:flex;align-items:center;justify-content:center;
                                font-size:1.3rem;">🫁</div>
                    <div>
                        <div style="font-size:1.05rem;font-weight:700;color:#1a1a1a;">MedRAX</div>
                        <div style="font-size:0.75rem;color:#888;">
                            Medical Reasoning Agent for Chest X-ray
                        </div>
                    </div>
                </div>
            """)

        # ── Main layout ───────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            # Left: chat ─────────────────────────────────────────────────
            with gr.Column(scale=3, elem_id="chat-panel"):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=680,
                    show_label=False,
                    container=False,
                    avatar_images=(None, "assets/medrax_logo.jpg"),
                    render_markdown=True,
                    sanitize_html=False,
                )

                status = gr.Markdown("", elem_id="status-bar")

                with gr.Row(elem_id="input-row"):
                    txt = gr.Textbox(
                        elem_id="msg-box",
                        placeholder="Ask about the X-ray…",
                        show_label=False,
                        container=False,
                        scale=10,
                        lines=1,
                        max_lines=5,
                    )
                    send_btn = gr.Button("↑", elem_id="send-btn", scale=1, min_width=42)

            # Right: image + controls ─────────────────────────────────────
            with gr.Column(scale=2, elem_id="image-panel"):
                image_display = gr.Image(
                    elem_id="image-display",
                    label="",
                    type="filepath",
                    height=480,
                    show_label=False,
                    container=False,
                )

                gr.Markdown(
                    "<div style='font-size:0.75rem;color:#aaa;margin:8px 0 4px;'>"
                    "Upload image</div>"
                )
                with gr.Row():
                    upload_btn = gr.UploadButton(
                        "📎  X-Ray", file_types=["image"],
                        elem_classes="upload-btn",
                    )
                    dicom_btn = gr.UploadButton(
                        "📄  DICOM", file_types=["file"],
                        elem_classes="upload-btn",
                    )

                gr.Markdown(
                    "<div style='font-size:0.75rem;color:#aaa;margin:12px 0 4px;'>"
                    "Session</div>"
                )
                with gr.Row():
                    clear_btn      = gr.Button("🗑  Clear chat",
                                               elem_id="clear-btn",
                                               elem_classes="action-btn")
                    new_thread_btn = gr.Button("✦  New thread",
                                               elem_classes="action-btn")

        # ── Event wiring ──────────────────────────────────────────────────

        def clear_chat():
            interface.original_file_path = None
            interface.display_file_path  = None
            return [], None, ""

        def new_thread():
            interface.current_thread_id = str(time.time())
            return [], interface.display_file_path, ""

        def handle_upload(file):
            return interface.handle_upload(file.name)

        def start_processing():
            return gr.update(value="_Thinking…_")

        def stop_processing():
            return gr.update(value="")

        for trigger in (txt.submit, send_btn.click):
            chat_msg = trigger(
                interface.add_message,
                inputs=[txt, image_display, chatbot],
                outputs=[chatbot, txt],
            )
            bot_msg = (
                chat_msg
                .then(start_processing, outputs=[status])
                .then(
                    interface.process_message,
                    inputs=[txt, image_display, chatbot],
                    outputs=[chatbot, image_display, txt],
                )
            )
            bot_msg.then(stop_processing, outputs=[status])
            bot_msg.then(lambda: gr.Textbox(interactive=True), outputs=[txt])

        upload_btn.upload(handle_upload, inputs=upload_btn, outputs=image_display)
        dicom_btn.upload(handle_upload,  inputs=dicom_btn,  outputs=image_display)
        clear_btn.click(clear_chat,      outputs=[chatbot, image_display, status])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display, status])

    return demo
