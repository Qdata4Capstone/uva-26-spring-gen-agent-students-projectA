"""
LangGraph agent — Claude <-> MCP tool-use loop.

  START -> call_claude -> (end_turn) -> END
                       -> (tool_use) -> execute_tools -> call_claude -> ...
"""

import json
import logging
from typing import Literal, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.models.schemas import PubMedArticle
from app.services.mcp_client import mcp_client

logger = logging.getLogger(__name__)

MAX_ROUNDS = 5

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


class AgentState(TypedDict):
    messages: list[dict]
    tools: list[dict]
    system: str
    retrieved_articles: list[dict]
    final_text: str
    rounds: int
    stop_reason: str


async def call_claude_node(state: AgentState) -> dict:
    response = await _get_client().messages.create(
        model=settings.claude_model,
        max_tokens=settings.max_response_tokens,
        system=state["system"],
        tools=state["tools"] or [],
        messages=state["messages"],
    )

    new_rounds = state["rounds"] + 1
    stop_reason = response.stop_reason
    logger.info("Claude stop_reason=%s (round %d/%d)", stop_reason, new_rounds, MAX_ROUNDS)

    if stop_reason == "end_turn":
        final_text = "".join(
            block.text for block in response.content
            if hasattr(block, "type") and block.type == "text"
        )
        return {"final_text": final_text, "rounds": new_rounds, "stop_reason": stop_reason}

    if stop_reason == "tool_use":
        # SDK content block objects must be serialized to dicts for state storage
        content_dicts = []
        for block in response.content:
            if block.type == "text":
                content_dicts.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content_dicts.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        new_messages = state["messages"] + [{"role": "assistant", "content": content_dicts}]
        return {"messages": new_messages, "rounds": new_rounds, "stop_reason": stop_reason}

    logger.warning("Unexpected stop_reason: %s — halting loop", stop_reason)
    final_text = "".join(
        block.text for block in response.content
        if hasattr(block, "type") and block.type == "text"
    )
    return {
        "final_text": final_text or "I'm sorry, something went wrong. Please try again.",
        "rounds": new_rounds,
        "stop_reason": "end_turn",
    }


async def execute_tools_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    tool_uses = [
        b for b in last_msg.get("content", [])
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]

    tool_results = []
    new_articles = list(state["retrieved_articles"])

    for tu in tool_uses:
        logger.info("Executing MCP tool: %s(%s)", tu["name"], tu["input"])
        try:
            mcp_result = await mcp_client.call_tool(tu["name"], tu["input"])
            result_text = mcp_client.extract_text(mcp_result)

            if tu["name"] == "pubmed_search":
                try:
                    for art in json.loads(result_text):
                        new_articles.append(art)
                except Exception:
                    pass

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_text,
            })
        except Exception as exc:
            logger.error("Tool %s failed: %s", tu["name"], exc)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": f"Tool error: {exc}",
                "is_error": True,
            })

    new_messages = state["messages"] + [{"role": "user", "content": tool_results}]
    return {"messages": new_messages, "retrieved_articles": new_articles}


def route_after_claude(state: AgentState) -> Literal["execute_tools", "__end__"]:
    if state["stop_reason"] == "tool_use" and state["rounds"] < MAX_ROUNDS:
        return "execute_tools"
    return END


def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("call_claude", call_claude_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_edge(START, "call_claude")
    builder.add_conditional_edges(
        "call_claude",
        route_after_claude,
        {"execute_tools": "execute_tools", END: END},
    )
    builder.add_edge("execute_tools", "call_claude")
    return builder.compile()


_graph = _build_graph()


async def run_agent(
    messages: list[dict],
    tools: list[dict],
    system: str,
) -> tuple[str, list[PubMedArticle]]:
    result = await _graph.ainvoke({
        "messages": messages,
        "tools": tools,
        "system": system,
        "retrieved_articles": [],
        "final_text": "",
        "rounds": 0,
        "stop_reason": "",
    })

    articles: list[PubMedArticle] = []
    for art_dict in result.get("retrieved_articles", []):
        try:
            articles.append(PubMedArticle(**art_dict))
        except Exception:
            pass

    final_text = (
        result.get("final_text")
        or "I'm sorry, I couldn't process your request. Please try again."
    )
    return final_text, articles
