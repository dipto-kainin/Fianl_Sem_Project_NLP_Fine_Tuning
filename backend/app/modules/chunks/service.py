"""
Chunks Module - Business Logic Service.
"""

import logging
import uuid

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chunks.models import Chunk
from app.utils.exceptions import ChunkNotFoundError

logger = logging.getLogger(__name__)


async def create_chunk(db: AsyncSession, **kwargs) -> Chunk:
    """Create a chunk record in the database."""
    chunk = Chunk(**kwargs)
    db.add(chunk)
    await db.flush()
    await db.refresh(chunk)
    return chunk


async def create_chunks_batch(
    db: AsyncSession,
    chunks_data: list[dict],
) -> list[Chunk]:
    """Create multiple chunk records in batch."""
    chunks = [Chunk(**data) for data in chunks_data]
    db.add_all(chunks)
    await db.flush()
    for chunk in chunks:
        await db.refresh(chunk)
    return chunks


async def get_chunk(db: AsyncSession, chunk_id: uuid.UUID) -> Chunk:
    """Get a chunk by ID."""
    result = await db.execute(select(Chunk).where(Chunk.id == chunk_id))
    chunk = result.scalar_one_or_none()
    if not chunk:
        raise ChunkNotFoundError(str(chunk_id))
    return chunk


async def get_chunks_by_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[Chunk], int]:
    """Get all chunks for a document with pagination."""
    query = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .offset(skip)
        .limit(limit)
    )
    count_query = (
        select(func.count(Chunk.id))
        .where(Chunk.document_id == document_id)
    )

    result = await db.execute(query)
    chunks = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return chunks, total


async def update_chunk_embedding_id(
    db: AsyncSession,
    chunk_id: uuid.UUID,
    embedding_id: str,
) -> Chunk:
    """Update a chunk's embedding_id (Qdrant point ID)."""
    chunk = await get_chunk(db, chunk_id)
    chunk.embedding_id = embedding_id
    await db.flush()
    await db.refresh(chunk)
    return chunk


async def delete_chunks_by_document(
    db: AsyncSession,
    document_id: uuid.UUID,
) -> int:
    """Delete all chunks for a document. Returns count deleted."""
    result = await db.execute(
        select(Chunk).where(Chunk.document_id == document_id)
    )
    chunks = result.scalars().all()
    count = 0
    for chunk in chunks:
        await db.delete(chunk)
        count += 1
    await db.flush()
    return count
