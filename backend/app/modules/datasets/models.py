"""
Datasets Module - SQLAlchemy Models.

Database models for training datasets and samples.
"""

import enum
import uuid

from sqlalchemy import (
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, TimestampMixin, UUIDMixin


class DatasetStatus(str, enum.Enum):
    """Dataset generation status."""
    GENERATING = "generating"
    READY = "ready"
    ARCHIVED = "archived"


class DifficultyLevel(str, enum.Enum):
    """Difficulty level of a training sample."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Dataset(Base, UUIDMixin, TimestampMixin):
    """Represents a versioned training dataset."""

    __tablename__ = "datasets"

    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    total_samples: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[DatasetStatus] = mapped_column(
        Enum(DatasetStatus), default=DatasetStatus.GENERATING
    )
    categories: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    samples = relationship(
        "DatasetSample", back_populates="dataset", cascade="all, delete-orphan"
    )
    training_runs = relationship("TrainingRun", back_populates="dataset")

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, version={self.version}, samples={self.total_samples})>"


class DatasetSample(Base, UUIDMixin, TimestampMixin):
    """A single training sample in a dataset."""

    __tablename__ = "dataset_samples"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[DifficultyLevel] = mapped_column(
        Enum(DifficultyLevel), default=DifficultyLevel.MEDIUM
    )
    priority: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    dataset = relationship("Dataset", back_populates="samples")

    def __repr__(self) -> str:
        return f"<DatasetSample(id={self.id}, dataset={self.dataset_id})>"
