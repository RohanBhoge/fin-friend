from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List

# Calculate project root (assumes this file is in backend/app/core/config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class Settings(BaseSettings):
    """Central configuration for the FinFriend backend."""

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str

    # ── JWT / Auth ────────────────────────────────────────────────────────
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    RESET_TOKEN_EXPIRE_MINUTES: int = 15

    # ── Google Gemini ─────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""

    # ── SMTP / Email ─────────────────────────────────────────────────────
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "noreply@finfriend.ai"

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:5173"

    # ── Environment ───────────────────────────────────────────────────────
    ENVIRONMENT: str = "development"

    # ── RAG / FAISS ───────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = str(PROJECT_ROOT / "faiss_index")
    DOCUMENTS_PATH: str = str(PROJECT_ROOT / "documents")

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton instance used across the application
settings = Settings()
