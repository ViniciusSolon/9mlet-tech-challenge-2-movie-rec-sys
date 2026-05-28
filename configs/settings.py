"""Application settings loaded from environment."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.paths import get_data_dir, get_raw_dir


class Settings(BaseSettings):
    """Environment-backed configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tmdb_api_key: str = ""
    data_dir: Path = Field(default_factory=get_data_dir)
    raw_data_path: Path = Field(default_factory=get_raw_dir)
    tmdb_language: str = "en-US"
    tmdb_min_interval_sec: float = 0.26
    tmdb_max_retries: int = 4

    def ensure_tmdb_key(self) -> str:
        """Return API key or raise if missing."""
        key = self.tmdb_api_key.strip()
        if not key:
            msg = "TMDB_API_KEY is empty; set it in .env"
            raise ValueError(msg)
        return key


def load_settings() -> Settings:
    """Load settings from ``.env`` in the current working directory."""
    return Settings()
