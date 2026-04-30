"""
Chat Router
===========
Endpoints:
  POST /api/chat         — standard chat (tool-use loop resolved, full response)
  POST /api/chat/stream  — streaming chat (tool-use → stream final response)
  GET  /api/tools        — list available MCP tools
  GET  /api/mcp/status   — MCP server connection status

Request flow:
  1. Safety check (crisis keywords → immediate crisis response, bypass LLM)
  2. Claude tool-use loop via MCP (Claude decides which tools to call)
  3. Return response + any PubMed articles Claude retrieved
"""

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    PubMedArticle,
)
from app.services import anthropic_service, tool_router
from app.services.mcp_client import mcp_client
from app.services.safety_service import check_safety
from app.services.user_service import get_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Standard chat ──────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Standard (non-streaming) chat endpoint.

    Claude autonomously decides whether to call MCP tools (pubmed_search,
    crisis_resources) based on the user's message. All tool calls are
    resolved before the response is returned.
    """
    session_id = req.session_id or str(uuid.uuid4())
    safety = check_safety(req.message, user_profile=req.user_profile)

    if safety.is_crisis:
        return ChatResponse(reply=safety.crisis_response, session_id=session_id, is_crisis=True)

    routed = await tool_router.route_user_message(req.message, user_profile=req.user_profile)
    if safety.is_crisis:
        return ChatResponse(
            reply=routed.crisis_response,
            session_id=session_id,
            is_crisis=True,
        )

    try:
        reply, articles = await anthropic_service.get_chat_response(
            message=req.message,
            history=req.history,
            distress_tier=safety.tier,
            user_profile=req.user,
            prefetched_pubmed=routed.pubmed_articles,
            prefetched_trials=routed.clinical_trials,
        )
        return ChatResponse(
            reply=reply,
            session_id=session_id,
            is_crisis=False,
            pubmed_articles=articles,
            clinical_trials=routed.clinical_trials,
            pubmed_query_used=routed.pubmed_query_used,
        )
    except Exception as exc:
        logger.exception("Error in /api/chat")
        raise HTTPException(status_code=500, detail="An internal error occurred.") from exc


# ── Streaming chat ─────────────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat via Server-Sent Events.

    Event sequence:
      data: {"type":"meta", "session_id":"...", "pubmed_articles":[...], "is_crisis":false}
      data: {"type":"delta", "delta":"Hello "}
      data: {"type":"delta", "delta":"there..."}
      data: [DONE]
    """
    session_id = req.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncIterator[str]:
        # Immediate SSE comment so the client knows the stream opened (before slow LLM work).
        yield ": stream-open\n\n"

        safety = check_safety(req.message, user_profile=req.user_profile)

        if safety.is_crisis:
            yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id, 'pubmed_articles': [], 'is_crisis': True})}\n\n"
            yield f"data: {json.dumps({'type': 'delta', 'delta': safety.crisis_response})}\n\n"
            yield "data: [DONE]\n\n"
            return

        routed = await tool_router.route_user_message(req.message, user_profile=req.user_profile)
        if safety.is_crisis:
            meta = json.dumps({
                "type": "meta",
                "session_id": session_id,
                "pubmed_articles": [],
                "clinical_trials": [],
                "pubmed_query_used": None,
                "is_crisis": True,
            })
            yield f"data: {meta}\n\n"
            crisis_text = routed.crisis_response
            delta = json.dumps({"type": "delta", "delta": crisis_text})
            yield f"data: {delta}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Signal prefetch finished so the client knows work is proceeding (long gap before first meta otherwise).
        yield f"data: {json.dumps({'type': 'ack', 'stage': 'prefetch'})}\n\n"

        # 2. Stream through the two-phase generator
        articles_sent = False

        try:
            async for chunk in anthropic_service.stream_chat_response(
                message=req.message,
                history=req.history,
                distress_tier=safety.tier,
                user_profile=req.user_profile,
                prefetched_pubmed=routed.pubmed_articles,
                prefetched_trials=routed.clinical_trials,
            ):
                # First chunk is always the articles sentinel from the service
                if not articles_sent:
                    articles: list[PubMedArticle] = []
                    try:
                        sentinel = json.loads(chunk)
                        articles_data = sentinel.get("__articles__", [])
                        if isinstance(articles_data, list):
                            for a in articles_data:
                                if not isinstance(a, dict):
                                    continue
                                try:
                                    articles.append(PubMedArticle(**a))
                                except Exception:
                                    logger.warning("Skipping malformed PubMed article in stream meta")
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        logger.warning("Bad articles sentinel: %s", exc)

                    meta = json.dumps({
                        "type":               "meta",
                        "session_id":         session_id,
                        "pubmed_articles":    [a.model_dump() for a in articles],
                        "clinical_trials":    [t.model_dump() for t in routed.clinical_trials],
                        "pubmed_query_used":  routed.pubmed_query_used,
                        "is_crisis":          False,
                    })
                    yield f"data: {meta}\n\n"
                    articles_sent = True
                    continue

                # Subsequent chunks are text deltas
                payload = json.dumps({"type": "delta", "delta": chunk})
                yield f"data: {payload}\n\n"

        except Exception as exc:
            logger.exception("chat/stream generator failed: %s", exc)
            err_meta = json.dumps({
                "type":               "meta",
                "session_id":         session_id,
                "pubmed_articles":    [],
                "clinical_trials":    [t.model_dump() for t in routed.clinical_trials],
                "pubmed_query_used":  routed.pubmed_query_used,
                "is_crisis":          False,
            })
            yield f"data: {err_meta}\n\n"
            err_delta = json.dumps({
                "type":  "delta",
                "delta": "Something went wrong on the server. Please try again.",
            })
            yield f"data: {err_delta}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── MCP introspection ──────────────────────────────────────────────────────

@router.get("/tools")
async def list_tools():
    """List all MCP tools currently available to Claude."""
    tools = await mcp_client.list_anthropic_tools()
    return {"tools": tools, "count": len(tools)}


@router.get("/mcp/status")
async def mcp_status():
    """Return the MCP server connection status."""
    connected = mcp_client.is_connected
    tools = await mcp_client.list_anthropic_tools() if connected else []
    return {
        "connected": connected,
        "tool_count": len(tools),
        "tools": [t["name"] for t in tools],
    }
