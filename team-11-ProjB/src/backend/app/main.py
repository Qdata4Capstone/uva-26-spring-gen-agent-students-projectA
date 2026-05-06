"""
FinSynth – FastAPI backend.

Endpoints:
    POST /api/analyze          Stream an SSE analysis for a given ticker.
    GET  /api/health           Health check.
"""

from __future__ import annotations

import json
import logging
import logging.config
import time

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .config import get_settings
from .graph.workflow import run_analysis
from .schemas import AnalyzeRequest, ChatRequest

# ── Logging configuration ─────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
    # Quiet noisy third-party libraries
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "langchain": {"level": "WARNING"},
        "langgraph": {"level": "WARNING"},
    },
})

logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinSynth API",
    description="Financial Synthesis AI Agent – multi-agent investment analysis",
    version="0.1.0",
)

# ── CORS ──────────────────────────────────────────────────────────────
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(
    "FinSynth API started | model=%s | cors_origins=%s",
    settings.ollama_model,
    settings.cors_origins,
)


# ── Health ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    logger.debug("Health check requested")
    return {"status": "ok", "service": "finsynth"}


# ── Analysis (SSE stream) ────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Kick off the multi-agent analysis and stream progress + report via SSE.

    Events emitted:
        thinking  – agent progress messages
        report    – final Markdown report
        error     – an error occurred
        done      – stream is finished
    """
    ticker = request.ticker.upper()
    logger.info("POST /api/analyze | ticker=%s | mode=%s", ticker, request.workflow_mode)
    t_start = time.monotonic()
    event_count = 0

    async def event_generator():
        nonlocal event_count
        async for event in run_analysis(request.ticker, request.workflow_mode):
            event_type = event["event"]
            event_count += 1
            if event_type == "error":
                logger.error(
                    "SSE error event | ticker=%s | message=%s",
                    ticker,
                    event["data"].get("message", ""),
                )
            elif event_type == "done":
                elapsed = time.monotonic() - t_start
                logger.info(
                    "Analysis stream complete | ticker=%s | events_emitted=%d | elapsed=%.2fs",
                    ticker,
                    event_count,
                    elapsed,
                )
            else:
                logger.debug("SSE event | ticker=%s | event=%s", ticker, event_type)
            yield {
                "event": event_type,
                "data": json.dumps(event["data"]),
            }

    return EventSourceResponse(event_generator())


# ── Chat (SSE stream) ─────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Stream a chat response grounded in the provided report via SSE.

    Events emitted:
        token  – incremental assistant text chunk
        error  – an error occurred
        done   – stream is finished
    """
    logger.info("POST /api/chat | messages=%d", len(request.messages))

    system_prompt = (
        "You are a financial analyst assistant. "
        "Answer questions about the investment report below. "
        "Be concise and cite specific figures from the report when relevant.\n\n"
        f"=== INVESTMENT REPORT ===\n{request.report}\n=== END REPORT ==="
    )

    ollama_messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        ollama_messages.append({"role": msg.role, "content": msg.content})

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=settings.ollama_timeout_sec) as client:
                async with client.stream(
                    "POST",
                    f"{settings.ollama_base_url}/api/chat",
                    json={"model": settings.ollama_model, "messages": ollama_messages, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        yield {"event": "error", "data": json.dumps({"message": f"Ollama error: {response.status_code}"})}
                        yield {"event": "done", "data": json.dumps({})}
                        return
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield {"event": "token", "data": json.dumps({"content": content})}
                        if chunk.get("done"):
                            break
        except Exception as exc:
            logger.error("Chat stream error: %s", exc)
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}
        yield {"event": "done", "data": json.dumps({})}

    return EventSourceResponse(event_generator())

