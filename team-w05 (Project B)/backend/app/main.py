import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import chat, health, user
from app.services.mcp_client import mcp_client
from app.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting — connecting to MCP server...")
    try:
        await mcp_client.connect()
        tools = await mcp_client.list_anthropic_tools()
        logger.info("MCP ready. Tools: %s", [t["name"] for t in tools])
    except Exception as exc:
        logger.error("MCP server failed to start: %s. Continuing without tools.", exc, exc_info=True)

    yield

    logger.info("Shutting down — disconnecting MCP server...")
    await mcp_client.disconnect()


app = FastAPI(
    title="Mental Health Bot API",
    description="Supportive AI chatbot backed by PubMed research via MCP. NOT a replacement for professional care.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(user.router, tags=["User"])
