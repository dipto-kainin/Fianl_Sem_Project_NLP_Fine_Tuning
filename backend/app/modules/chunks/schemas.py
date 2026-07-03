"""
Chunks Module - Pydantic Schemas.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel


class ChunkResponse(BaseModel):
    """Response schema for a single chunk."""

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    text: str
    token_count: int
    section_title: str | None = None
    page_numbers: list[int] | None = None
    metadata_: dict | None = None
    embedding_id: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChunkListResponse(BaseModel):
    """Response schema for listing chunks."""

    chunks: list[ChunkResponse]
    total: int
