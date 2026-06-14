"""Application settings loaded from environment.

All fields map 1-to-1 with variables in `.env` / `.env.example`.
Add new variables there before adding them here.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.paths import get_data_dir, get_raw_dir


class Settings(BaseSettings):
    """Environment-backed configuration for the full pipeline.

    Sections
    --------
    TMDB scraping  : tmdb_*
    MLflow         : mlflow_*
    DVC            : dvc_*
    Reproducibility: *_seed / pythonhashseed
    Data paths     : data_dir, raw_data_path
    Model / eval   : top_k, implicit_rating_threshold
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # TMDB / OMDb  (Etapa de scraping)
    # ------------------------------------------------------------------
    tmdb_api_key: str = ""
    omdb_api_key: str = ""
    tmdb_language: str = "en-US"
    tmdb_min_interval_sec: float = 0.26
    tmdb_max_retries: int = 4

    # ------------------------------------------------------------------
    # MLflow  (Etapa 3)
    # ------------------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "movielens-recommender"

    # ------------------------------------------------------------------
    # DVC  (Etapa 3)
    # ------------------------------------------------------------------
    dvc_remote_url: str = "./dvc-storage"

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    data_dir: Path = Field(default_factory=get_data_dir)
    raw_data_path: Path = Field(default_factory=get_raw_dir)

    # ------------------------------------------------------------------
    # Reproducibility seeds  (Etapa 4 / Bloco 2.6)
    # ------------------------------------------------------------------
    pythonhashseed: int = 42
    torch_seed: int = 42
    numpy_seed: int = 42

    # ------------------------------------------------------------------
    # Model / evaluation  (Etapa 4)
    # ------------------------------------------------------------------
    top_k: int = 10
    implicit_rating_threshold: float = 4.0

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        """Ensure top_k is a positive integer."""
        if v <= 0:
            msg = "top_k must be a positive integer"
            raise ValueError(msg)
        return v

    def ensure_tmdb_key(self) -> str:
        """Return TMDB API key or raise if missing."""
        key = self.tmdb_api_key.strip()
        if not key:
            msg = "TMDB_API_KEY is empty; set it in .env"
            raise ValueError(msg)
        return key


def load_settings() -> Settings:
    """Load settings from ``.env`` in the current working directory."""
    return Settings()
