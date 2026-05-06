"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings backed by .env file."""

    financial_api_key: str = ""
    brave_search_api_key: str = ""

    # LLM config
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4"
    # Seconds for each Ollama HTTP request (large reports can run long on local GPUs).
    ollama_timeout_sec: float = 600.0

    # Server config
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
