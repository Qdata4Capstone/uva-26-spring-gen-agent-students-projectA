"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings backed by .env file."""

    gemini_api_key: str = ""
    financial_api_key: str = ""
    brave_search_api_key: str = ""

    # LLM config
    gemini_model: str = "gemini-2.5-flash"

    # Server config
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
