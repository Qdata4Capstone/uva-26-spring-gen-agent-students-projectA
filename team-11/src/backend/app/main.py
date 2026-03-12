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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .config import get_settings
from .graph.workflow import run_analysis
from .schemas import AnalyzeRequest

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
        "google": {"level": "WARNING"},
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
    settings.gemini_model,
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
    logger.info("POST /api/analyze | ticker=%s", ticker)
    t_start = time.monotonic()
    event_count = 0

    async def event_generator():
        nonlocal event_count
        async for event in run_analysis(request.ticker):
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

