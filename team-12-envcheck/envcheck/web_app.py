"""
EnvPilot Web App — FastAPI backend with SSE streaming for the 5-phase workflow.

Provides:
  - GET  /                  → Serve the single-page frontend
  - POST /api/run           → Start a workflow run (returns run_id)
  - GET  /api/run/{id}/stream → SSE stream of phase progress
  - POST /api/scan          → Quick scan (no LLM, just envcheck scanner)
  - GET  /api/kb/stats      → Knowledge base stats
  - GET  /api/kb/search     → Search the knowledge base
  - GET  /api/health        → Health check

Usage:
    python -m envcheck.web_app
    # or: uvicorn envcheck.web_app:app --reload --port 8000
"""

import asyncio
import json
import logging
import platform
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("envpilot.web")

app = FastAPI(
    title="EnvPilot",
    description="Proactive code environment diagnostic system",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for active runs
_runs: dict[str, dict[str, Any]] = {}

STATIC_DIR = Path(__file__).parent / "static"


# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>EnvPilot</h1><p>Static files not found.</p>")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.2.0",
        "python": platform.python_version(),
        "os": platform.system(),
    }


@app.get("/api/kb/stats")
async def kb_stats():
    """Return knowledge base statistics."""
    from envcheck.knowledge_base_store import KnowledgeBaseStore
    store = KnowledgeBaseStore()
    libraries = store.get_all_libraries()
    count = store.count()
    store.close()
    return {
        "total_rules": count,
        "libraries": sorted(libraries),
        "library_count": len(libraries),
    }


@app.get("/api/kb/search")
async def kb_search(q: str = Query(..., min_length=1)):
    """Search the knowledge base."""
    from envcheck.knowledge_base_store import KnowledgeBaseStore
    store = KnowledgeBaseStore()
    rules = store.search_fts(q)
    results = store.to_dict_list(rules)
    store.close()
    return {"query": q, "count": len(results), "results": results}


@app.get("/api/kb/rules")
async def kb_rules(library: str = ""):
    """List all rules, optionally filtered by library."""
    from envcheck.knowledge_base_store import KnowledgeBaseStore
    store = KnowledgeBaseStore()
    if library:
        rules = store.query(library=library)
    else:
        rules = store.get_all_for_scanner()
    results = store.to_dict_list(rules)
    store.close()
    return {"count": len(results), "results": results}


@app.post("/api/scan")
async def quick_scan(body: dict):
    """Quick static scan without LLM — just run the envcheck scanner."""
    from envcheck.scanner import scan_source

    code = body.get("code", "")
    env_path = body.get("env_path", "")

    if not code:
        return {"error": "code is required"}

    if not env_path:
        env_path = sys.prefix

    report = scan_source(code, env_path=env_path)
    findings = []
    for f in report.findings:
        findings.append({
            "lineno": f.lineno,
            "matched_code": f.matched_code,
            "rule_id": f.rule.rule_id,
            "description": f.rule.description,
            "old_api": f.rule.old_api,
            "new_api": f.rule.new_api,
            "severity": f.severity.value,
            "installed_version": f.installed_version,
            "library": f.rule.library,
        })

    return {
        "files_scanned": report.files_scanned,
        "total_findings": report.total_findings,
        "scan_time_ms": round(report.scan_time_ms, 1),
        "findings": findings,
        "errors": report.errors,
    }


@app.post("/api/run")
async def start_run(body: dict):
    """Start a new EnvPilot workflow run. Returns a run_id for SSE streaming."""
    task = body.get("task", "")
    env_path = body.get("env_path", "")

    if not task:
        return {"error": "task is required"}

    import os
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        return {"error": "GEMINI_API_KEY environment variable not set. Export it before starting the server."}

    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {
        "task": task,
        "env_path": env_path,
        "status": "pending",
        "events": [],
        "result": None,
        "done": asyncio.Event(),
    }

    asyncio.create_task(_execute_run(run_id))

    return {"run_id": run_id, "status": "started"}


@app.get("/api/run/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE stream for a running workflow. Streams phase events in real-time."""
    if run_id not in _runs:
        return {"error": "run not found"}

    return StreamingResponse(
        _sse_generator(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# SSE streaming logic
# ============================================================================

async def _sse_generator(run_id: str):
    """Generate SSE events for a workflow run."""
    run = _runs[run_id]
    sent = 0

    while True:
        # Send any new events
        while sent < len(run["events"]):
            event = run["events"][sent]
            data = json.dumps(event, ensure_ascii=False)
            yield f"data: {data}\n\n"
            sent += 1

        if run["done"].is_set():
            # Send final result
            if run["result"]:
                data = json.dumps(run["result"], ensure_ascii=False)
                yield f"event: complete\ndata: {data}\n\n"
            break

        await asyncio.sleep(0.3)


async def _execute_run(run_id: str):
    """Execute the EnvPilot workflow and push events to the run store."""
    run = _runs[run_id]
    run["status"] = "running"

    task = run["task"]
    env_path = run["env_path"]

    def push_event(phase: str, status: str, detail: Any = None):
        run["events"].append({
            "phase": phase,
            "status": status,
            "detail": detail,
        })

    def llm_call(prompt: str) -> dict:
        from envcheck.agent.nodes import _llm_call
        return _llm_call(prompt)

    try:
        push_event("init", "started", {"task": task, "env_path": env_path})

        # Phase 1: Analysis
        push_event("analysis", "running")
        from envcheck.agent.prompts import ANALYSIS_PROMPT
        prompt = ANALYSIS_PROMPT.format(task_description=task)
        analysis = await asyncio.to_thread(llm_call, prompt)
        packages = analysis.get("identified_packages", [])
        uncertainty = analysis.get("uncertainty_score", 50)
        push_event("analysis", "complete", {
            "packages": packages,
            "uncertainty_score": uncertainty,
            "reasoning": analysis.get("reasoning", ""),
        })

        # Phase 2: Environment Probe
        push_event("env_probe", "running")
        env_info = await asyncio.to_thread(_probe_env, packages, env_path)
        push_event("env_probe", "complete", env_info)

        # Phase 3: Knowledge Base Query
        push_event("kb_query", "running")
        kb_data = await asyncio.to_thread(_query_kb, packages)
        push_event("kb_query", "complete", {
            "rules_found": len(kb_data["results"]),
            "libraries_covered": kb_data["libraries"],
        })

        # Phase 3b: Web Search (conditional)
        web_results = []
        if uncertainty > 20 or kb_data["has_gaps"]:
            push_event("web_search", "running")
            web_results = await asyncio.to_thread(_web_search, packages, env_info)
            push_event("web_search", "complete", {
                "results_found": len(web_results),
            })

            # Phase 3c: KB Update (after web search)
            push_event("kb_update", "running")
            try:
                from envcheck.knowledge_base_store import KnowledgeBaseStore
                store = KnowledgeBaseStore()
                updated_count = 0
                for wr in web_results:
                    if isinstance(wr, dict) and wr.get("rule_id"):
                        from envcheck.knowledge_base import BreakingChangeRule
                        try:
                            br = BreakingChangeRule(**{k: wr[k] for k in BreakingChangeRule.__dataclass_fields__ if k in wr})
                            store.upsert(br, source="web_search")
                            updated_count += 1
                        except Exception:
                            pass
                push_event("kb_update", "complete", {"rules_upserted": updated_count})
            except Exception as e:
                push_event("kb_update", "complete", {"rules_upserted": 0, "note": str(e)})
        else:
            push_event("web_search", "skipped", {"reason": "confidence sufficient"})
            push_event("kb_update", "skipped", {"reason": "no web search needed"})

        # Phase 4: Preflight (if env_path provided)
        preflight_result = {"success": True, "note": "skipped"}
        if env_path:
            push_event("preflight", "running")
            from envcheck.agent.prompts import PREFLIGHT_PROMPT
            pf_prompt = PREFLIGHT_PROMPT.format(
                env_info=json.dumps(env_info, indent=2),
                kb_results=json.dumps(kb_data["results"], indent=2),
                web_results=json.dumps(web_results, indent=2),
                task_description=task,
            )
            pf_plan = await asyncio.to_thread(llm_call, pf_prompt)
            pf_code = pf_plan.get("preflight_code", "")
            if pf_code and env_path:
                from envcheck.preflight_runner import run_preflight, to_dict
                pf_res = await asyncio.to_thread(run_preflight, pf_code, env_path)
                preflight_result = to_dict(pf_res)
            push_event("preflight", "complete", preflight_result)
        else:
            push_event("preflight", "skipped", {"reason": "no env_path"})

        # Phase 5: Generation
        push_event("generation", "running")
        from envcheck.agent.prompts import GENERATION_PROMPT
        gen_prompt = GENERATION_PROMPT.format(
            env_info=json.dumps(env_info, indent=2),
            kb_results=json.dumps(kb_data["results"], indent=2),
            preflight_result=json.dumps(preflight_result, indent=2),
            task_description=task,
        )
        gen_result = await asyncio.to_thread(llm_call, gen_prompt)
        final_code = gen_result.get("final_code", "")
        notes = gen_result.get("notes", "")
        push_event("generation", "complete", {
            "code_length": len(final_code),
            "notes": notes,
        })

        run["result"] = {
            "status": "success",
            "final_code": final_code,
            "notes": notes,
            "packages": packages,
            "uncertainty_score": uncertainty,
            "findings_count": len(kb_data["results"]),
        }

    except Exception as e:
        logger.exception(f"Run {run_id} failed")
        push_event("error", "failed", {"error": str(e)})
        run["result"] = {"status": "error", "error": str(e)}

    finally:
        run["status"] = "done"
        run["done"].set()


# ============================================================================
# Helper functions (run in thread pool)
# ============================================================================

def _probe_env(packages: list[str], env_path: str) -> dict:
    from envcheck.version_detector import get_installed_packages

    info: dict[str, Any] = {"packages": {}, "python": platform.python_version()}

    if env_path:
        installed = get_installed_packages(env_path)
    else:
        installed = get_installed_packages(Path(sys.prefix))

    for pkg in packages:
        normalized = pkg.lower()
        if normalized in installed:
            info["packages"][pkg] = {
                "installed": True,
                "version": installed[normalized].version,
            }
        else:
            info["packages"][pkg] = {"installed": False, "version": None}

    return info


def _query_kb(packages: list[str]) -> dict:
    from envcheck.knowledge_base_store import KnowledgeBaseStore
    store = KnowledgeBaseStore()
    all_results = []
    libraries_found = []
    for pkg in packages:
        rules = store.query(library=pkg)
        if rules:
            libraries_found.append(pkg)
        all_results.extend(store.to_dict_list(rules))
    has_gaps = len(all_results) == 0 or len(libraries_found) < len(packages)
    store.close()
    return {
        "results": all_results,
        "libraries": libraries_found,
        "has_gaps": has_gaps,
    }


def _web_search(packages: list[str], env_info: dict) -> list[dict]:
    from envcheck.web_searcher import WebSearcher
    searcher = WebSearcher()
    results = []
    for pkg in packages:
        pkg_info = env_info.get("packages", {}).get(pkg, {})
        version = pkg_info.get("version", "")
        query = f"breaking changes migration guide"
        if version:
            query = f"version {version} {query}"
        res = searcher.search_api_docs(pkg, query, max_results=3)
        results.extend(searcher.to_dict_list(res))
    return results


def main():
    """Run the web app with uvicorn."""
    import uvicorn
    uvicorn.run(
        "envcheck.web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
