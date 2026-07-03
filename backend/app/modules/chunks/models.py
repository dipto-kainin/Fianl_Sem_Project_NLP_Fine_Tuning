"""
Chunks Module - SQLAlchemy Models.

Database models for document chunks.
"""

import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base, TimestampMixin, UUIDMixin


class Chunk(Base, UUIDMixin, TimestampMixin):
    """Represents a semantic chunk of a document."""

    __tablename__ = "chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    section_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_numbers: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    embedding_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, doc="Qdrant point ID"
    )

    # Relationships
    document = relationship("Document", back_populates="chunks")
    teacher_output = relationship(
        "TeacherOutput", back_populates="chunk", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, doc={self.document_id}, index={self.chunk_index})>"
