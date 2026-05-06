"""
Anthropic Service
=================
Manages all interactions with Claude using the Anthropic Messages API.

This module implements the real MCP tool-use loop:

  1. Send user message + conversation history to Claude, with MCP tools attached.
  2. Claude decides autonomously whether and which tools to call.
  3. If Claude returns stop_reason="tool_use":
       a. Extract tool_use blocks from the response.
       b. Execute each tool via the MCP client (which calls the MCP server subprocess).
       c. Append tool_result blocks back to the conversation.
       d. Call Claude again with the updated messages.
       e. Repeat until Claude returns stop_reason="end_turn".
  4. Return the final text response + any articles retrieved during tool calls.

Streaming
---------
Tool use is resolved non-streaming (Claude must complete its reasoning before
we know which tools to call). The final response after tool resolution is
streamed via SSE so the user sees text appear progressively.
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import anthropic

from app.config import settings
from app.models.schemas import ClinicalTrialSummary, ConversationMessage, PubMedArticle
from app.services.mcp_client import mcp_client

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Mental_Health_Bot, a compassionate, evidence-informed mental health support assistant.

HARD RULES — never break these:
1. You are NOT a licensed therapist, psychologist, psychiatrist, or doctor.
2. You NEVER diagnose any medical or mental health condition.
3. You NEVER prescribe or recommend specific medications, dosages, or medical procedures.
4. You NEVER claim certainty on medical matters.
5. If a user expresses thoughts of self-harm, suicide, or immediate danger you must
   STOP normal conversation and use the crisis_resources tool to get helplines, then
   present them clearly and compassionately.
6. Always encourage users to consult a qualified mental health professional for
   personalised guidance.
7. Respond with warmth, empathy, and respect. Never shame or frighten the user.
8. Keep language clear and accessible — avoid clinical jargon where possible.
9. When you summarise research findings from pubmed_search, explain them in plain
   English and clearly state this is general information, not personal medical advice.
10. Only cite articles that were actually returned by the pubmed_search tool or the
    pre-retrieved PubMed JSON in your context. Never fabricate citations.
11. When pre-retrieved clinical trial JSON is present, describe trials accurately
    (status, title) and link users to the trial URL — do not invent trials.

Your role:
- Offer emotional support and a non-judgemental listening space.
- Use pubmed_search when the user asks about therapies, treatments, medications,
  or the evidence base for a condition.
- Use crisis_resources when the user seems to be in significant distress.
- Gently encourage professional help, therapy, and evidence-based self-care.
- Remind users at the end of complex answers that speaking to a professional is worthwhile.
"""

DISTRESS_ADDON = """
ADDITIONAL INSTRUCTION FOR THIS MESSAGE:
The user's message shows signs of significant emotional distress.
Acknowledge their feelings warmly before providing any information.
Remind them early in your response that professional support is available.
"""

# ── Helpers ────────────────────────────────────────────────────────────────

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        key = (settings.anthropic_api_key or "").strip()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in backend/.env")
        _client = anthropic.AsyncAnthropic(api_key=key)
    return _client


def _build_system(distress_tier: int, user_profile: Any = None) -> str:
    user_context = _build_user_context(user_profile)
    base = SYSTEM_PROMPT + (DISTRESS_ADDON if distress_tier == 2 else "")

    if user_context:
        base += "\n\nUSER CONTEXT:\n" + user_context + "\n\n"
        base += "Use this context to personalize responses, but do NOT make assumptions beyond it."

    return base


def _format_history(history: list[ConversationMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in history]


def _extract_text_blocks(content: list) -> str:
    return "".join(
        block.text for block in content
        if hasattr(block, "type") and block.type == "text"
    )


def _extract_tool_uses(content: list) -> list:
    return [b for b in content if hasattr(b, "type") and b.type == "tool_use"]


def _dedupe_cap_pubmed(articles: list[PubMedArticle]) -> list[PubMedArticle]:
    """Keep PubMed order, drop duplicate PMIDs, cap at configured limit for the UI."""
    lim = settings.max_research_sources_per_turn
    seen: set[str] = set()
    out: list[PubMedArticle] = []
    for art in articles:
        pmid = (art.pmid or "").strip()
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        out.append(art)
        if len(out) >= lim:
            break
    return out


def _prefetch_context_block(
    articles: list[PubMedArticle] | None,
    trials: list[ClinicalTrialSummary] | None,
) -> str:
    """Extra system instructions listing JSON the model should ground on."""
    lim = settings.max_research_sources_per_turn
    arts = list(articles or [])[:lim]
    tr = list(trials or [])[:lim]
    chunks: list[str] = []
    if arts:
        chunks.append(
            "Pre-retrieved PubMed articles (JSON). Use only these for literature claims "
            "unless you run pubmed_search again for a tighter query:\n"
            + json.dumps([a.model_dump() for a in arts], ensure_ascii=False),
        )
    if tr:
        chunks.append(
            "Pre-retrieved clinical trials (JSON). Summarise accurately when relevant:\n"
            + json.dumps([t.model_dump() for t in tr], ensure_ascii=False),
        )
    if not chunks:
        return ""
    return (
        "\n## Pre-retrieved evidence\n"
        + "\n\n".join(chunks)
        + f"\n\nThe user interface lists at most {lim} PubMed sources; avoid extra pubmed_search "
        "calls unless the user clearly needs different evidence."
    )


def _coerce_user_profile(user_profile: Any) -> dict | None:
    if user_profile is None:
        return None
    if isinstance(user_profile, dict):
        return user_profile
    if hasattr(user_profile, "model_dump"):
        return user_profile.model_dump()
    return None


def _build_user_context(user_profile: dict | None) -> str:
    """
    Build a structured, instruction-oriented user context block.

    Goals:
    - Personalize tone, style, and explanations
    - Respect user preferences without overfitting
    - Avoid unsafe assumptions (especially clinical)
    """

    if not user_profile:
        return ""

    lines: list[str] = []
    instructions: list[str] = []

    # Identity (light personalization only)
    if name := user_profile.get("preferred_name"):
        lines.append(f"- Preferred name: {name}")
        instructions.append("Address the user by their preferred name occasionally, but not excessively.")

    if age := user_profile.get("age"):
        lines.append(f"- Age: {age}")
        if age < 18:
            instructions.append("Use simpler language and be extra supportive and careful.")

    if gender := user_profile.get("gender"):
        lines.append(f"- Gender: {gender}")

    # UVA context (affects resources)
    if user_profile.get("is_uva_student"):
        lines.append("- Affiliation: University of Virginia")
        instructions.append("When suggesting resources, prioritize UVA-specific services when relevant.")

    # Communication preferences
    literacy = user_profile.get("literacy_level")
    if literacy:
        lines.append(f"- Preferred explanation level: {literacy}")
        if literacy == "Child":
            instructions.append("Explain concepts very simply using short sentences and minimal jargon.")
        elif literacy == "General Adult":
            instructions.append("Use clear, everyday language with minimal jargon.")
        elif literacy == "Medical/Advanced":
            instructions.append("You may include more detailed and technical explanations when appropriate.")

    tone = user_profile.get("preferred_tone")
    if tone:
        lines.append(f"- Preferred tone: {tone}")
        if tone == "Empathetic":
            instructions.append("Lead with empathy and emotional validation.")
        elif tone == "Soft":
            instructions.append("Use gentle, reassuring language.")
        elif tone == "Direct":
            instructions.append("Be concise and clear, while remaining respectful.")
        elif tone == "Clinical":
            instructions.append("Use a more neutral, structured, and clinical tone while staying supportive.")

    modality = user_profile.get("therapeutic_modality")
    if modality:
        lines.append(f"- Preferred therapeutic style: {modality}")
        if modality == "CBT":
            instructions.append("Incorporate cognitive-behavioral techniques such as reframing thoughts.")
        elif modality == "Mindfulness":
            instructions.append("Incorporate mindfulness-based suggestions such as grounding or breathing exercises.")
        elif modality == "DBT":
            instructions.append("Incorporate DBT-style strategies like distress tolerance and emotional regulation.")

    # Mental health context (handle carefully)
    if diagnoses := user_profile.get("formal_diagnoses"):
        if diagnoses:
            lines.append(f"- Reported diagnoses: {', '.join(diagnoses)}")
            instructions.append(
                "Be mindful of these conditions, but do NOT assume symptoms or make diagnostic claims."
            )

    if meds := user_profile.get("active_medications"):
        if meds:
            lines.append(f"- Active medications: {', '.join(meds)}")
            instructions.append(
                "Do NOT give medication advice. Be cautious when discussing treatment."
            )

    if triggers := user_profile.get("known_triggers"):
        if triggers:
            lines.append(f"- Known triggers: {', '.join(triggers)}")
            instructions.append(
                "Avoid unnecessarily emphasizing or describing known triggers."
            )

    if mood := user_profile.get("mood_baseline"):
        lines.append(f"- Baseline mood: {mood}")

    # Safety preferences
    if not user_profile.get("emergency_opt_in", True):
        instructions.append(
            "Do not aggressively push crisis resources unless clearly necessary."
        )

    if not user_profile.get("conversation_history_enabled", True):
        instructions.append(
            "Do not reference prior messages or assume continuity beyond the current message."
        )

    retention = user_profile.get("data_retention_period_days")
    if retention:
        lines.append(f"- Data retention preference: {retention} days")

    # Final assembly
    context_block = "USER PROFILE:\n" + "\n".join(lines) if lines else ""
    instruction_block = (
        "\n\nPERSONALIZATION INSTRUCTIONS:\n" + "\n".join(f"- {i}" for i in instructions)
        if instructions
        else ""
    )

    return context_block + instruction_block


# ── Tool-use loop ──────────────────────────────────────────────────────────

# Claude + network can stall; fail fast instead of hanging the UI indefinitely.
_ANTHROPIC_MESSAGE_TIMEOUT_SEC = 120.0
_MCP_TOOL_TIMEOUT_SEC = 90.0
_MCP_LIST_TOOLS_TIMEOUT_SEC = 45.0


async def _list_anthropic_tools_safe() -> list[dict]:
    try:
        return await asyncio.wait_for(
            mcp_client.list_anthropic_tools(),
            timeout=_MCP_LIST_TOOLS_TIMEOUT_SEC,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP list_anthropic_tools failed or timed out: %s", exc)
        return []


async def _call_mcp_tool_safe(name: str, arguments: dict[str, Any]):
    return await asyncio.wait_for(
        mcp_client.call_tool(name, arguments),
        timeout=_MCP_TOOL_TIMEOUT_SEC,
    )


async def _execute_tool_use_loop(
    messages: list[dict],
    tools: list[dict],
    distress_tier: int,
    user_profile: Any = None,
    system_suffix: str = "",
    seed_pubmed_articles: list[PubMedArticle] | None = None,
) -> tuple[str, list[PubMedArticle]]:
    """
    Run the Claude ↔ MCP tool-use loop until Claude returns end_turn.

    ``seed_pubmed_articles`` are merged with any articles from pubmed_search in-loop.
    ``system_suffix`` is appended to the system prompt (e.g. pre-fetched JSON context).
    """
    client = _get_client()
    system = _build_system(distress_tier, user_profile) + (system_suffix or "")
    retrieved_articles: list[PubMedArticle] = list(seed_pubmed_articles or [])
    max_rounds = 5  # safety cap to prevent infinite loops

    for round_num in range(max_rounds):
        logger.info("Tool-use loop round %d/%d", round_num + 1, max_rounds)

        try:
            response = await client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.max_response_tokens,
                system=system,
                tools=tools or [],
                messages=messages,
                timeout=_ANTHROPIC_MESSAGE_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Claude request failed or timed out: %s", exc)
            return (
                "The AI service timed out or was unavailable. Please try again in a moment.",
                retrieved_articles,
            )

        logger.info("Claude stop_reason=%s", response.stop_reason)

        # ── End turn: Claude is done ───────────────────────────────────
        if response.stop_reason == "end_turn":
            return _extract_text_blocks(response.content), retrieved_articles

        # ── Tool use: Claude wants to call one or more tools ───────────
        if response.stop_reason == "tool_use":
            tool_uses = _extract_tool_uses(response.content)

            # Append assistant message (with tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call via the MCP client
            tool_results = []
            for tool_use in tool_uses:
                logger.info("Claude calling tool: %s(%s)", tool_use.name, tool_use.input)
                try:
                    mcp_result = await _call_mcp_tool_safe(tool_use.name, tool_use.input)
                    result_text = mcp_client.extract_text(mcp_result)

                    # If this was a pubmed_search, parse and store articles for the UI
                    if tool_use.name == "pubmed_search":
                        try:
                            raw = json.loads(result_text)
                            for art in raw:
                                retrieved_articles.append(PubMedArticle(**art))
                        except Exception:
                            pass

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tool_use.id,
                        "content":     result_text,
                    })

                except Exception as exc:  # noqa: BLE001
                    logger.error("Tool %s failed: %s", tool_use.name, exc)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tool_use.id,
                        "content":     f"Tool error: {exc}",
                        "is_error":    True,
                    })

            # Append tool results as the next user turn
            messages.append({"role": "user", "content": tool_results})
            continue  # Go back to Claude with tool results

        # Unexpected stop reason — just extract any text and stop
        logger.warning("Unexpected stop_reason: %s", response.stop_reason)
        return _extract_text_blocks(response.content), retrieved_articles

    logger.warning("Tool-use loop hit max rounds (%d)", max_rounds)
    return "I'm sorry, something went wrong processing your request. Please try again.", retrieved_articles


# ── Public API ─────────────────────────────────────────────────────────────

async def get_chat_response(
    message: str,
    history: list[ConversationMessage],
    distress_tier: int = 0,
    user_profile: Any = None,
    prefetched_pubmed: list[PubMedArticle] | None = None,
    prefetched_trials: list[ClinicalTrialSummary] | None = None,
) -> tuple[str, list[PubMedArticle]]:
    """
    Send a message to Claude with MCP tools available.

    Pre-fetched PubMed / trials from the tool router are injected into the system prompt
    and seed the returned article list so the UI stays consistent even if Claude skips tools.
    """
    if not (settings.anthropic_api_key or "").strip():
        arts = _dedupe_cap_pubmed(list(prefetched_pubmed or []))
        return (
            "Configure ANTHROPIC_API_KEY in backend/.env to enable AI replies. "
            "Any pre-fetched sources still appear below.",
            arts,
        )

    tools = await _list_anthropic_tools_safe()
    messages = _format_history(history[-20:])
    messages.append({"role": "user", "content": message})

    suffix = _prefetch_context_block(prefetched_pubmed, prefetched_trials)
    reply, articles = await _execute_tool_use_loop(
        messages,
        tools,
        distress_tier,
        user_profile,
        system_suffix=suffix,
        seed_pubmed_articles=list(prefetched_pubmed or []),
    )
    return reply, _dedupe_cap_pubmed(articles)


async def stream_chat_response(
    message: str,
    history: list[ConversationMessage],
    distress_tier: int = 0,
    user_profile: Any = None,
    prefetched_pubmed: list[PubMedArticle] | None = None,
    prefetched_trials: list[ClinicalTrialSummary] | None = None,
) -> AsyncIterator[str]:
    """
    Two-phase streaming response:

    Phase 1 (non-streaming): Run the tool-use loop to resolve all tool calls.
    Phase 2 (streaming):     Stream the final Claude response as text deltas.

    First yielded value is a JSON sentinel ``{"__articles__": [...]}`` (deduped/capped).
    """
    if not (settings.anthropic_api_key or "").strip():
        deduped = _dedupe_cap_pubmed(list(prefetched_pubmed or []))
        yield json.dumps({"__articles__": [a.model_dump() for a in deduped]})
        yield (
            "Configure ANTHROPIC_API_KEY in backend/.env to enable AI replies. "
            "Any pre-fetched sources still appear above."
        )
        return

    tools = await _list_anthropic_tools_safe()
    messages = _format_history(history[-20:])
    messages.append({"role": "user", "content": message})

    client = _get_client()
    suffix = _prefetch_context_block(prefetched_pubmed, prefetched_trials)
    system = _build_system(distress_tier, user_profile) + suffix
    retrieved_articles: list[PubMedArticle] = list(prefetched_pubmed or [])

    # Run tool use loop non-streaming until end_turn
    max_rounds = 5
    for _ in range(max_rounds):
        try:
            response = await client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.max_response_tokens,
                system=system,
                tools=tools or [],
                messages=messages,
                timeout=_ANTHROPIC_MESSAGE_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Claude streaming-phase request failed or timed out: %s", exc)
            yield json.dumps({
                "__articles__": [a.model_dump() for a in _dedupe_cap_pubmed(retrieved_articles)],
            })
            yield "The AI service timed out or was unavailable. Please try again in a moment."
            return

        if response.stop_reason == "end_turn":
            # Phase 2: stream the final text
            final_text = _extract_text_blocks(response.content)

            capped = _dedupe_cap_pubmed(retrieved_articles)
            yield json.dumps({"__articles__": [a.model_dump() for a in capped]})

            # Stream the text in chunks to simulate streaming
            chunk_size = 4
            words = final_text.split(" ")
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield chunk
            return

        if response.stop_reason == "tool_use":
            tool_uses = _extract_tool_uses(response.content)
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                try:
                    mcp_result = await _call_mcp_tool_safe(tu.name, tu.input)
                    result_text = mcp_client.extract_text(mcp_result)

                    if tu.name == "pubmed_search":
                        try:
                            for art in json.loads(result_text):
                                retrieved_articles.append(PubMedArticle(**art))
                        except Exception:
                            pass

                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tu.id, "content": result_text
                    })
                except Exception as exc:  # noqa: BLE001
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tu.id,
                        "content": f"Tool error: {exc}", "is_error": True
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Fallback
        break

    capped = _dedupe_cap_pubmed(retrieved_articles)
    yield json.dumps({"__articles__": [a.model_dump() for a in capped]})
    yield "I'm sorry, I couldn't complete that request. Please try again."
