"""
Configuration — all settings loaded from environment variables via Pydantic.
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Absolute path to the repo root (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_MCP_SERVER = str(_REPO_ROOT / "mcp_server" / "server.py")


class Settings(BaseSettings):
    # ── Anthropic ────────────────────────────────────────────
    # Empty allows the API process to boot (e.g. /health, local router checks).
    # Chat uses a short placeholder reply until a real key is set in .env.
    anthropic_api_key: str = ""

    # ── App ──────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Claude ───────────────────────────────────────────────
    claude_model: str = "claude-sonnet-4-20250514"
    max_response_tokens: int = 1024
    session_memory_turns: int = 10

    # ── MCP Server ───────────────────────────────────────────
    # Path to the MCP server entry-point script.
    # Override via MCP_SERVER_PATH env var if your layout differs.
    mcp_server_path: str = _DEFAULT_MCP_SERVER
    # Python executable used to spawn the MCP server subprocess.
    # Defaults to the same interpreter running this process so the subprocess
    # inherits the same virtualenv / site-packages (including mcp).
    mcp_python: str = sys.executable

    # ── NCBI / PubMed ────────────────────────────────────────
    ncbi_api_key: str = ""

    # PubMed + clinical trials per turn (prefetch + final merged list sent to the client).
    max_research_sources_per_turn: int = Field(default=3, ge=1, le=10)

    @field_validator("mcp_server_path", mode="before")
    @classmethod
    def mcp_server_path_non_empty(cls, v: object) -> str:
        """`.env` often has `MCP_SERVER_PATH=` which would override the default with ''."""
        if v is None:
            return _DEFAULT_MCP_SERVER
        if isinstance(v, str) and not v.strip():
            return _DEFAULT_MCP_SERVER
        return str(v)

    @field_validator("mcp_python", mode="before")
    @classmethod
    def mcp_python_non_empty(cls, v: object) -> str:
        if v is None:
            return sys.executable
        if isinstance(v, str) and not v.strip():
            return sys.executable
        return str(v)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
