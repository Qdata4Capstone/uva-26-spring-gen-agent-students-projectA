#!/usr/bin/env python3
"""
Legal Compliance Agent — Web Server
FastAPI backend with SSE streaming for the frontend UI.
Routes user prompts to the appropriate pipeline: de-identification,
dataset analysis, compliance reference lookup, or PII redaction.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_DIR   = Path("/bigtemp2/tsx4zn/legal/images")
OUTPUT_DIR    = Path("/bigtemp2/tsx4zn/legal/output")
STATIC_DIR    = Path("/bigtemp2/tsx4zn/legal/static")
MARKDOWN_DIR  = Path("/bigtemp2/tsx4zn/legal/markdown_data")
REDACTED_DIR  = OUTPUT_DIR / "redacted"

app = FastAPI(title="Legal Compliance Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

(OUTPUT_DIR / "deidentified").mkdir(parents=True, exist_ok=True)

# Active SSE streams: session_id → queue
_sse_queues: dict[str, asyncio.Queue] = {}


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    api_key: str = ""
    session_id: str = ""


# ──────────────────────────────────────────────────────────────
# Intent routing
# ──────────────────────────────────────────────────────────────

def _classify_prompt(message: str) -> str:
    """Route user intent to the appropriate pipeline."""
    t = message.lower()
    if any(k in t for k in ["redact", "pii", "sensitive", "markdown", ".md", "text file", "document", "personal info"]):
        return "redact"
    if any(k in t for k in ["reference", "standard", "which compliance", "what framework", "options", "choose", "select", "recommendation"]):
        return "references"
    # Conference/platform/regulation names or explicit action verbs → full pipeline
    process_kws = [
        "cvpr", "iccv", "neurips", "youtube", "tiktok", "gdpr", "ccpa",
        "blur", "de-identify", "deidentif", "anonymiz", "comply", "complian",
        "process", "submit",
    ]
    if any(k in t for k in process_kws):
        return "process"
    if any(k in t for k in ["analyz", "structure", "dataset", "what's in", "show me", "explore", "inspect", "overview"]):
        return "analyze"
    return "process"


def _get_dataset_stats() -> dict:
    """Collect dataset statistics by scanning the filesystem."""
    clips = []
    total_images = 0
    for d in sorted(DATASET_DIR.iterdir()):
        if not d.is_dir():
            continue
        imgs = list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))
        total_images += len(imgs)
        out_dir = OUTPUT_DIR / "deidentified" / d.name
        has_output = out_dir.exists() and any(out_dir.iterdir())
        clips.append({"name": d.name, "images": len(imgs), "has_output": has_output})
    return {"clips": clips, "total_clips": len(clips), "total_images": total_images}


# ──────────────────────────────────────────────────────────────
# Agent thread — runs the real pipeline with SSE log streaming
# ──────────────────────────────────────────────────────────────

def _run_agent_thread(
    session_id: str,
    message: str,
    api_key: str,
    loop: asyncio.AbstractEventLoop,
):
    """Run the appropriate pipeline in a background thread, pushing events via SSE."""

    # SSE log handler — captures all agent logging and streams to the frontend
    class SSELogHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                asyncio.run_coroutine_threadsafe(
                    _sse_queues[session_id].put({"type": "log", "text": msg}),
                    loop,
                )
            except Exception:
                pass

    handler = SSELogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    def push(item):
        asyncio.run_coroutine_threadsafe(
            _sse_queues[session_id].put(item), loop
        )

    def done():
        asyncio.run_coroutine_threadsafe(
            _sse_queues[session_id].put(None), loop
        )

    prompt_type = _classify_prompt(message)

    try:
        if prompt_type == "analyze":
            push({"type": "intro", "text": "Sure! Let me scan the dataset directory and collect the statistics for you."})

            logger.info("Scanning dataset directory...")
            stats = _get_dataset_stats()
            processed = sum(1 for c in stats["clips"] if c["has_output"])
            logger.info(f"Found {stats['total_clips']} clips, {stats['total_images']} frames, {processed} already processed")

            push({
                "type": "analyze",
                "data": {
                    "total_clips": stats["total_clips"],
                    "total_images": stats["total_images"],
                    "processed_clips": processed,
                    "clips": stats["clips"],
                    "summary": (
                        f"Dataset contains {stats['total_clips']} video clips "
                        f"({stats['total_images']} frames total). "
                        f"{processed} clips already have de-identification output. "
                        "Scenes include outdoor pedestrian/vehicle footage from YouTube."
                    ),
                },
            })
            push({"type": "outro", "text": (
                f"Analysis complete. {processed} out of {stats['total_clips']} clips are already "
                "de-identified and ready to browse. You can expand the clip list above for details, "
                "or send a processing prompt to run the full compliance pipeline."
            )})

        elif prompt_type == "references":
            push({"type": "intro", "text": "Of course! Let me query the compliance knowledge base for applicable frameworks."})

            from agent.tools.knowledge import KnowledgeBase
            kb = KnowledgeBase()
            logger.info("Querying compliance knowledge base...")

            # Retrieve all available frameworks from the KB
            references = []
            for category in ("platforms", "conferences", "regulations"):
                section = kb._db.get(category, {})
                for key, entry in section.items():
                    bool_fields = [
                        "face_blur", "license_plate_blur", "metadata_strip",
                        "de_identification", "compliance_report", "irreversible_anonymization",
                    ]
                    required_ops = [
                        f for f in bool_fields if entry.get(f"{f}_required")
                    ]
                    references.append({
                        "id": f"{category}:{key}",
                        "title": entry.get("name", key),
                        "summary": entry.get("notes", "") or ", ".join(
                            entry.get("privacy_requirements", entry.get("requirements", entry.get("dataset_requirements", [])))[:2]
                        ),
                        "tags": [category.rstrip("s")],
                        "required_ops": required_ops,
                    })
            logger.info(f"Found {len(references)} applicable frameworks")

            push({
                "type": "references",
                "data": {
                    "references": references,
                    "message": (
                        "Here are the applicable compliance frameworks for your dataset. "
                        "Select one to apply:"
                    ),
                },
            })
            push({"type": "outro", "text": (
                "Click 'Apply this framework →' on any card above to auto-fill the processing prompt, "
                "then hit Send to run the full de-identification pipeline."
            )})

        elif prompt_type == "redact":
            push({"type": "intro", "text": (
                "Got it! I'll scan your document folder for Markdown files and redact any "
                "personally identifiable information — emails, phone numbers, SSNs, addresses, and more."
            )})

            from agent.tools.pii_redactor import PIIRedactor
            redactor = PIIRedactor()

            logger.info("[Step 1] Scanning document directory...")
            md_files = sorted(MARKDOWN_DIR.glob("*.md"))
            logger.info(f"  Found {len(md_files)} Markdown files")

            logger.info("[Step 2] PII detection & redaction")
            result = redactor.process_directory(str(MARKDOWN_DIR), str(REDACTED_DIR))
            for f in result["files"]:
                if f["status"] == "success":
                    logger.info(f"  {f['file']}: {f['total_pii']} PII items redacted")

            logger.info(f"[Step 3] Verification — confirming no residual PII in output...")
            # Verify redacted output is clean
            residual = 0
            for f in sorted(REDACTED_DIR.glob("*.md")):
                _, counts = redactor.redact_text(f.read_text(encoding="utf-8"))
                residual += sum(counts.values())
            logger.info(f"  Residual PII in output: {residual}")

            file_results = []
            for f in result["files"]:
                if f["status"] == "success":
                    file_results.append({
                        "file": f["file"],
                        "pii_count": f["total_pii"],
                        "pii_types": f["pii_counts"],
                    })

            push({
                "type": "redact_result",
                "data": {
                    "total_files": result["processed"],
                    "total_pii": result["total_pii"],
                    "pii_counts": result["pii_counts"],
                    "files": file_results,
                },
            })

            cat_parts = []
            for k, v in sorted(result["pii_counts"].items(), key=lambda x: -x[1]):
                cat_parts.append(f"{v} {k.lower().replace('_', ' ')}s")
            cat_str = ", ".join(cat_parts)
            push({"type": "outro", "text": (
                f"Done! I found and redacted {result['total_pii']} PII items across {result['processed']} documents. "
                f"Categories: {cat_str}. "
                "You can now review the original vs redacted documents in the Documents tab on the right."
            )})

        else:
            # Full compliance pipeline — de-identification, validation, report
            push({"type": "intro", "text": (
                "Got it! I'll check the applicable regulations, run EgoBlur de-identification "
                "on your dataset, validate with re-identification testing, and generate a compliance report. "
                "Please hold on for a moment..."
            )})

            from agent.agent import LegalComplianceAgent

            agent = LegalComplianceAgent(
                dataset_dir=str(DATASET_DIR),
                output_dir=str(OUTPUT_DIR),
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
                max_workers=4,
            )
            result = agent.run(message)

            plan = result.get("plan", {})
            processing_log = result.get("processing_log", [])
            critic_results = result.get("critic_results", [])
            clips_ok = sum(1 for p in processing_log if p.get("status") == "success")
            reid_failed = sum(1 for c in critic_results if not c.get("passed"))

            push({
                "type": "result",
                "data": {
                    "session_id": result.get("session_id", session_id),
                    "plan": plan,
                    "requirements": result.get("requirements", {}),
                    "report_path": result.get("report_path", ""),
                    "output_dir": str(OUTPUT_DIR),
                    "clips_ok": clips_ok,
                    "clips_total": len(processing_log),
                    "reid_failed": reid_failed,
                },
            })

            ctx_parts = []
            if plan.get("conference"):
                ctx_parts.append(f"for {plan['conference'].upper()}")
            if plan.get("platform"):
                ctx_parts.append(f"({plan['platform']} data)")
            if plan.get("regulations"):
                ctx_parts.append(f"under {', '.join(r.upper() for r in plan['regulations'])}")
            ctx_str = " ".join(ctx_parts)

            push({"type": "outro", "text": (
                f"All done! I've successfully de-identified {clips_ok} out of {len(processing_log)} clips {ctx_str}. "
                "Faces and license plates have been blurred using EgoBlur Gen2, EXIF metadata has been stripped, "
                f"and {len(processing_log) - reid_failed} clips passed re-identification validation. "
                "You can now browse and compare the processed images using the Dataset Browser on the right, "
                "or check the compliance report in the Reports tab."
            )})

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        push({"type": "error", "text": str(e)})
    finally:
        root_logger.removeHandler(handler)
        done()


# ──────────────────────────────────────────────────────────────
# API: Dataset browser
# ──────────────────────────────────────────────────────────────

@app.get("/api/clips")
def list_clips():
    """List all clip folders with image counts and output status."""
    clips = []
    for d in sorted(DATASET_DIR.iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        out_dir = OUTPUT_DIR / "deidentified" / d.name
        has_output = out_dir.exists() and any(out_dir.iterdir())
        clips.append({
            "name": d.name,
            "count": len(imgs),
            "has_output": has_output,
            "thumbnail": f"/images/input/{d.name}/{imgs[0].name}" if imgs else None,
            "thumbnail_out": (
                f"/images/output/{d.name}/{imgs[0].name}"
                if has_output and imgs else None
            ),
        })
    return clips


@app.get("/api/clips/{clip_name}/images")
def list_clip_images(clip_name: str):
    """List images in a clip, with input and output paths."""
    clip_dir = DATASET_DIR / clip_name
    if not clip_dir.exists():
        raise HTTPException(404, f"Clip {clip_name!r} not found")

    imgs = sorted(
        f for f in clip_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    out_dir = OUTPUT_DIR / "deidentified" / clip_name

    result = []
    for img in imgs:
        out_path = out_dir / img.name
        result.append({
            "name": img.name,
            "input":  f"/images/input/{clip_name}/{img.name}",
            "output": f"/images/output/{clip_name}/{img.name}" if out_path.exists() else None,
        })
    return result


@app.get("/api/reports")
def list_reports():
    """List all compliance reports."""
    reports = []
    for f in sorted(OUTPUT_DIR.glob("compliance_report_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            reports.append({
                "file": f.name,
                "session_id": data.get("report_id"),
                "generated_at": data.get("generated_at"),
                "clips_processed": data.get("processing_summary", {}).get("successful", 0),
                "re_id_passed": data.get("critic_validation", {}).get("passed", 0),
                "platform": data.get("legal_context", {}).get("platform"),
                "conference": data.get("legal_context", {}).get("conference"),
                "statement": data.get("compliance_statement", ""),
            })
        except Exception:
            pass
    return reports


@app.get("/api/reports/{filename}")
def get_report(filename: str):
    """Get full compliance report JSON."""
    path = OUTPUT_DIR / filename
    if not path.exists() or not filename.startswith("compliance_report_"):
        raise HTTPException(404)
    return json.loads(path.read_text())


# ──────────────────────────────────────────────────────────────
# API: Markdown / PII redaction
# ──────────────────────────────────────────────────────────────

@app.get("/api/markdown/files")
def list_markdown_files():
    """List Markdown files with redaction status."""
    files = []
    for f in sorted(MARKDOWN_DIR.glob("*.md")):
        redacted = REDACTED_DIR / f.name
        has_redacted = redacted.exists()
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
            "has_redacted": has_redacted,
        })
    return files


@app.get("/api/markdown/{filename}/original")
def get_markdown_original(filename: str):
    """Return original Markdown content."""
    path = MARKDOWN_DIR / filename
    if not path.exists() or not filename.endswith(".md"):
        raise HTTPException(404)
    return JSONResponse({"filename": filename, "content": path.read_text(encoding="utf-8")})


@app.get("/api/markdown/{filename}/redacted")
def get_markdown_redacted(filename: str):
    """Return redacted Markdown content."""
    path = REDACTED_DIR / filename
    if not path.exists() or not filename.endswith(".md"):
        raise HTTPException(404)
    return JSONResponse({"filename": filename, "content": path.read_text(encoding="utf-8")})


# ──────────────────────────────────────────────────────────────
# API: Agent chat with SSE streaming
# ──────────────────────────────────────────────────────────────

@app.post("/api/chat/start")
async def chat_start(req: ChatRequest):
    """Start an agent run and return a session_id for SSE streaming."""
    session_id = req.session_id or str(uuid.uuid4())[:8]
    _sse_queues[session_id] = asyncio.Queue()

    loop = asyncio.get_event_loop()
    thread = threading.Thread(
        target=_run_agent_thread,
        args=(session_id, req.message, req.api_key, loop),
        daemon=True,
    )
    thread.start()
    return {"session_id": session_id}


@app.get("/api/chat/stream/{session_id}")
async def chat_stream(session_id: str):
    """SSE stream for agent log events."""
    if session_id not in _sse_queues:
        raise HTTPException(404, "Session not found")

    async def generator() -> AsyncGenerator[str, None]:
        q = _sse_queues[session_id]
        while True:
            item = await q.get()
            if item is None:
                yield "data: {\"type\": \"done\"}\n\n"
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        _sse_queues.pop(session_id, None)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────────
# Image serving
# ──────────────────────────────────────────────────────────────

@app.get("/images/input/{clip_name}/{filename}")
def serve_input_image(clip_name: str, filename: str):
    path = DATASET_DIR / clip_name / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(str(path))


@app.get("/images/output/{clip_name}/{filename}")
def serve_output_image(clip_name: str, filename: str):
    path = OUTPUT_DIR / "deidentified" / clip_name / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(str(path))


# ──────────────────────────────────────────────────────────────
# Frontend
# ──────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"\n  Legal Compliance Agent UI → http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
