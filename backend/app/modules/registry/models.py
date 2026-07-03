"""
Registry Module - SQLAlchemy Models.

Database models for the model version registry.
"""

import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    String,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, TimestampMixin, UUIDMixin


class ModelVersion(Base, UUIDMixin, TimestampMixin):
    """Represents a versioned, deployable student model."""

    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    training_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("training_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    base_model: Mapped[str] = mapped_column(String(256), nullable=False)
    model_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    quantization_format: Mapped[str | None] = mapped_column(String(50), nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    training_run = relationship("TrainingRun", back_populates="model_version")

    def __repr__(self) -> str:
        return f"<ModelVersion(id={self.id}, version={self.version}, active={self.is_active})>"
