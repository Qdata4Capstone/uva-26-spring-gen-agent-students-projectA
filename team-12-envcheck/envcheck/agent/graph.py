"""
EnvPilot Agent Graph — LangGraph state graph with 5-phase workflow.

State graph topology:

    [START] -> analysis -> env_probe -> kb_query -+-> web_search -> kb_update -> preflight
                                                  |                                  |
                                                  +-> preflight -----+               |
                                                                     |               |
                                                  +------------------+               |
                                                  |                                  |
                                                  +-> generation -> [END]            |
                                                  |                                  |
                                                  +<---- (retry on failure) <--------+

Usage:
    from envcheck.agent.graph import build_graph

    app = build_graph()
    result = app.invoke({
        "task_description": "Create a numpy script...",
        "env_path": "./environments/case_numpy_2x",
    })
    print(result["final_code"])
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from envcheck.agent.nodes import (
    analysis_node,
    env_probe_node,
    generation_node,
    kb_query_node,
    kb_update_node,
    plan_node,
    preflight_node,
    route_after_kb_query,
    route_after_preflight,
    web_search_node,
)
from envcheck.agent.state import EnvPilotState


def build_graph() -> StateGraph:
    """Build and compile the EnvPilot state graph.

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    graph = StateGraph(EnvPilotState)

    # Add all nodes
    graph.add_node("analysis", analysis_node)
    graph.add_node("env_probe", env_probe_node)
    graph.add_node("kb_query", kb_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("kb_update", kb_update_node)
    graph.add_node("plan", plan_node)
    graph.add_node("preflight", preflight_node)
    graph.add_node("generation", generation_node)

    # Linear edges
    graph.add_edge(START, "analysis")
    graph.add_edge("analysis", "env_probe")
    graph.add_edge("env_probe", "kb_query")

    # Conditional: after KB query, either web_search (when info gaps) or
    # straight to plan (when KB has enough).
    graph.add_conditional_edges(
        "kb_query",
        route_after_kb_query,
        {"web_search": "web_search", "plan": "plan"},
    )

    # Web search -> KB update -> plan
    graph.add_edge("web_search", "kb_update")
    graph.add_edge("kb_update", "plan")

    # Plan synthesizes proposed_apis, preflight verifies them deterministically
    graph.add_edge("plan", "preflight")

    # Conditional: preflight passed -> generation; failed -> back to plan
    # (with failure feedback). Capped at MAX_PLAN_ATTEMPTS in route function.
    graph.add_conditional_edges(
        "preflight",
        route_after_preflight,
        {"plan": "plan", "generation": "generation"},
    )

    # Generation -> END
    graph.add_edge("generation", END)

    return graph.compile()


def get_default_initial_state(
    task_description: str,
    env_path: str = "",
) -> dict:
    """Create a default initial state for the graph.

    Args:
        task_description: The coding task to accomplish.
        env_path: Optional path to a virtual environment.

    Returns:
        Initial state dict ready for graph.invoke().
    """
    return {
        "task_description": task_description,
        "messages": [],
        "identified_packages": [],
        "critical_apis": [],
        "uncertainty_score": 50,
        "env_path": env_path,
        "env_info": {},
        "kb_results": [],
        "kb_has_gaps": False,
        "web_results": [],
        "kb_updates": [],
        "proposed_apis": [],
        "plan_attempts": 0,
        "preflight_code": "",
        "preflight_result": {},
        "preflight_attempts": 0,
        "final_code": "",
        "phase": "init",
        "error": "",
    }
