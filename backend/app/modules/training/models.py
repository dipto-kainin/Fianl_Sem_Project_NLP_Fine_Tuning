"""
Training Module - SQLAlchemy Models.

Database models for training runs.
"""

import enum
import uuid

from sqlalchemy import (
    Enum,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime

from app.database import Base, TimestampMixin, UUIDMixin


class TrainingStatus(str, enum.Enum):
    """Training run status."""
    QUEUED = "queued"
    TRAINING = "training"
    MERGING = "merging"
    QUANTIZING = "quantizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingRun(Base, UUIDMixin, TimestampMixin):
    """Represents a model training run."""

    __tablename__ = "training_runs"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    base_model: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[TrainingStatus] = mapped_column(
        Enum(TrainingStatus), default=TrainingStatus.QUEUED
    )
    lora_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="training_runs")
    model_version = relationship(
        "ModelVersion", back_populates="training_run", uselist=False
    )

    def __repr__(self) -> str:
        return f"<TrainingRun(id={self.id}, status={self.status})>"
