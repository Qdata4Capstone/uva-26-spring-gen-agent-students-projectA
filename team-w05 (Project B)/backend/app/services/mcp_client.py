import asyncio
import json
import logging
import os
import sys
import threading
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from app.config import settings

logger = logging.getLogger(__name__)


def _new_event_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()
    return asyncio.new_event_loop()


class MCPClient:
    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools_cache: list[dict] | None = None
        self._bg_loop: asyncio.AbstractEventLoop = _new_event_loop()
        self._thread = threading.Thread(
            target=self._bg_loop.run_forever,
            name="mcp-event-loop",
            daemon=True,
        )
        self._thread.start()

    def _run(self, coro) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return asyncio.wrap_future(future)

    async def connect(self) -> None:
        return await self._run(self._connect())

    async def _connect(self) -> None:
        if self._session is not None:
            return

        logger.info("Spawning MCP server: %s %s", settings.mcp_python, settings.mcp_server_path)

        server_params = StdioServerParameters(
            command=settings.mcp_python,
            args=[settings.mcp_server_path],
            env={**os.environ, "NCBI_API_KEY": settings.ncbi_api_key} if settings.ncbi_api_key else None,
        )

        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        session: ClientSession = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        self._session = session
        logger.info("MCP server connected: %s", settings.mcp_server_path)

    async def disconnect(self) -> None:
        return await self._run(self._disconnect())

    async def _disconnect(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            self._tools_cache = None
            logger.info("MCP server disconnected.")

    async def list_anthropic_tools(self) -> list[dict]:
        return await self._run(self._list_anthropic_tools())

    async def _list_anthropic_tools(self) -> list[dict]:
        if self._tools_cache is not None:
            return self._tools_cache
        if self._session is None:
            logger.warning("MCP session unavailable — returning empty tool list.")
            return []

        result = await self._session.list_tools()
        self._tools_cache = [
            {"name": tool.name, "description": tool.description or "", "input_schema": tool.inputSchema}
            for tool in result.tools
        ]
        logger.info("Discovered %d MCP tools: %s", len(self._tools_cache), [t["name"] for t in self._tools_cache])
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        return await self._run(self._call_tool(name, arguments))

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        if self._session is None:
            raise RuntimeError("MCP session is not connected.")
        logger.info("Calling MCP tool %r with args %s", name, arguments)
        result = await self._session.call_tool(name, arguments)
        if result.isError:
            logger.warning("MCP tool %r returned an error.", name)
        return result

    def extract_text(self, result: CallToolResult) -> str:
        return "\n".join(item.text for item in result.content if hasattr(item, "text"))

    def extract_json(self, result: CallToolResult) -> Any:
        text = self.extract_text(result)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    @property
    def is_connected(self) -> bool:
        return self._session is not None


mcp_client = MCPClient()
