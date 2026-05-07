"""
EnvPilot MCP Server — Exposes 4 tools via the Model Context Protocol.

Tools:
  1. envcheck       — Probe the local Python environment for installed packages
  2. knowledge_base — Query/search/upsert the breaking changes knowledge base
  3. web_search     — Search the web for API docs and breaking changes
  4. preflight_test — Run a smoke test in an isolated virtual environment

Usage:
    # Start via stdio (default, for Cursor / Claude Desktop)
    python -m envcheck.mcp_server

    # Or import and run programmatically
    from envcheck.mcp_server import mcp
    mcp.run()
"""

import platform
import sys
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

from envcheck.knowledge_base import BreakingChangeRule, PatternType, Severity
from envcheck.knowledge_base_store import KnowledgeBaseStore
from envcheck.preflight_runner import run_preflight, to_dict as preflight_to_dict
from envcheck.version_detector import get_installed_packages, get_package_version
from envcheck.web_searcher import WebSearcher

mcp = FastMCP("envpilot")

_kb_store: Optional[KnowledgeBaseStore] = None
_web_searcher: Optional[WebSearcher] = None


def _get_kb_store() -> KnowledgeBaseStore:
    global _kb_store
    if _kb_store is None:
        _kb_store = KnowledgeBaseStore()
    return _kb_store


def _get_web_searcher() -> WebSearcher:
    global _web_searcher
    if _web_searcher is None:
        _web_searcher = WebSearcher()
    return _web_searcher


@mcp.tool()
def envcheck(packages: list[str], env_path: str = "") -> dict:
    """Probe the local Python environment for installed package versions.

    Args:
        packages: List of package names to check (e.g., ["numpy", "pandas"]).
        env_path: Optional path to a virtual environment. If empty, uses the
                  current Python interpreter's environment.

    Returns:
        Dictionary with python version, OS info, and installed package versions.
    """
    result: dict = {
        "python": platform.python_version(),
        "os": platform.system().lower(),
        "os_version": platform.release(),
        "packages": {},
    }

    if env_path and Path(env_path).exists():
        installed = get_installed_packages(env_path)
    else:
        # Use current environment: find site-packages from sys.path
        env_path_resolved = Path(sys.prefix)
        installed = get_installed_packages(env_path_resolved)

    for pkg_name in packages:
        normalized = pkg_name.lower()
        if normalized in installed:
            pkg = installed[normalized]
            result["packages"][pkg_name] = {
                "installed": True,
                "version": pkg.version,
            }
        else:
            result["packages"][pkg_name] = {
                "installed": False,
                "version": None,
            }

    return result


@mcp.tool()
def knowledge_base(
    action: str,
    library: str = "",
    symbol: str = "",
    query_text: str = "",
    rule: Optional[dict] = None,
) -> dict:
    """Query, search, or update the breaking changes knowledge base.

    Args:
        action: One of "query", "search", or "upsert".
            - "query": Exact match lookup by library and/or symbol.
            - "search": Full-text search across all rule fields.
            - "upsert": Insert or update a rule (requires the 'rule' parameter).
        library: Filter by library name (for "query" action).
        symbol: Filter by symbol/function name (for "query" action).
        query_text: Search text (for "search" action).
        rule: Rule dict to upsert (for "upsert" action). Must include:
              rule_id, library, removed_in, pattern_type, module_path, symbol,
              old_api, new_api, error_type, description.

    Returns:
        Dictionary with results or confirmation.
    """
    store = _get_kb_store()

    if action == "query":
        rules = store.query(
            library=library or None,
            symbol=symbol or None,
        )
        return {
            "action": "query",
            "count": len(rules),
            "results": store.to_dict_list(rules),
        }

    elif action == "search":
        if not query_text:
            return {"action": "search", "error": "query_text is required for search"}
        rules = store.search_fts(query_text)
        return {
            "action": "search",
            "query": query_text,
            "count": len(rules),
            "results": store.to_dict_list(rules),
        }

    elif action == "upsert":
        if not rule:
            return {"action": "upsert", "error": "rule dict is required for upsert"}
        try:
            br = BreakingChangeRule(
                rule_id=rule["rule_id"],
                library=rule["library"],
                removed_in=rule["removed_in"],
                pattern_type=PatternType(rule["pattern_type"]),
                module_path=rule["module_path"],
                symbol=rule["symbol"],
                old_api=rule["old_api"],
                new_api=rule["new_api"],
                error_type=rule["error_type"],
                description=rule["description"],
                severity=Severity(rule.get("severity", "error")),
                method_kwargs=rule.get("method_kwargs"),
                base_type_hint=rule.get("base_type_hint"),
            )
            store.upsert(br, source=rule.get("source", "web_search"))
            return {"action": "upsert", "status": "ok", "rule_id": br.rule_id}
        except (KeyError, ValueError) as e:
            return {"action": "upsert", "error": f"Invalid rule: {e}"}

    else:
        return {"error": f"Unknown action '{action}'. Use 'query', 'search', or 'upsert'."}


@mcp.tool()
def web_search(library: str, query: str, max_results: int = 5) -> dict:
    """Search the web for API documentation, breaking changes, and migration guides.

    Uses DuckDuckGo to find relevant information about library APIs,
    especially useful when the local knowledge base lacks information.

    Args:
        library: The library name (e.g., "numpy", "pandas").
        query: Search terms (e.g., "trapz removed replacement").
        max_results: Maximum number of results to return.

    Returns:
        Dictionary with search results containing title, url, and snippet.
    """
    searcher = _get_web_searcher()
    results = searcher.search_api_docs(library, query, max_results=max_results)
    return {
        "library": library,
        "query": query,
        "count": len(results),
        "results": searcher.to_dict_list(results),
    }


@mcp.tool()
def preflight_test(code: str, env_path: str, timeout: int = 15) -> dict:
    """Run a smoke test script in an isolated virtual environment.

    Executes a minimal Python script to verify that a proposed code plan
    works in the target environment before generating the full project code.

    Args:
        code: Python source code to execute as a smoke test.
        env_path: Path to the virtual environment to run the test in.
        timeout: Maximum execution time in seconds.

    Returns:
        Dictionary with success status, stdout, stderr, exit_code, and duration_ms.
    """
    result = run_preflight(code, env_path, timeout=timeout)
    return preflight_to_dict(result)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
