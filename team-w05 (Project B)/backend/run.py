"""
Windows-compatible dev server launcher.

uvicorn --reload on Windows forces WindowsSelectorEventLoopPolicy, which
breaks asyncio subprocess creation (used by MCP's stdio transport). This
script patches that out so the MCP server subprocess can be spawned correctly.
"""

import sys

if sys.platform == "win32":
    import asyncio
    # Keep the default ProactorEventLoop on Windows so that
    # asyncio.create_subprocess_exec works (needed by MCP stdio_client).
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Prevent uvicorn's reload mode from switching back to SelectorEventLoop.
    import uvicorn.loops.asyncio as _uvla
    _uvla.asyncio_setup = lambda use_subprocess=False: None

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        reload=True,
        host="127.0.0.1",
        port=8000,
    )
