"""
AI Knowledge Distillation Platform - Application Configuration.

Loads settings from environment variables / .env file using Pydantic Settings.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Project ---
    PROJECT_NAME: str = "AI Knowledge Distillation Platform"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://kdp_user:kdp_password@localhost:5432/knowledge_distillation"

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Qdrant ---
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "knowledge_base"

    # --- Google Gemini API ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # --- Embedding Model ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # --- File Storage ---
    UPLOAD_DIR: str = "./storage/uploads"
    MODELS_DIR: str = "./storage/models"
    DATASETS_DIR: str = "./storage/datasets"

    # --- Document Processing ---
    MAX_UPLOAD_SIZE_MB: int = 100
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # --- Student Training ---
    BASE_STUDENT_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    LORA_RANK: int = 16
    LORA_ALPHA: int = 32
    TRAINING_EPOCHS: int = 10
    TRAINING_BATCH_SIZE: int = 4
    LEARNING_RATE: float = 2e-4

    # --- Teacher Fallback ---
    CONFIDENCE_THRESHOLD: float = 0.7

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        for dir_path in [self.UPLOAD_DIR, self.MODELS_DIR, self.DATASETS_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
