"""
Teacher Module - SQLAlchemy Models.

Database models for Teacher LLM outputs.
"""

import uuid

from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, TimestampMixin, UUIDMixin


class TeacherOutput(Base, UUIDMixin, TimestampMixin):
    """Stores structured output from the Teacher LLM for a chunk."""

    __tablename__ = "teacher_outputs"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Structured outputs
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    entities: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    relationships: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    qa_pairs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    explanations: Mapped[str | None] = mapped_column(Text, nullable=True)
    faqs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)

    # Token usage tracking
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    chunk = relationship("Chunk", back_populates="teacher_output")

    def __repr__(self) -> str:
        return f"<TeacherOutput(id={self.id}, chunk_id={self.chunk_id})>"
