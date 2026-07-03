"""
Teacher Module - Business Logic Service.
"""

import logging
import uuid

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.teacher.models import TeacherOutput
from app.modules.teacher.schemas import TeacherOutputResponse

logger = logging.getLogger(__name__)


async def store_teacher_output(
    db: AsyncSession,
    chunk_id: uuid.UUID,
    summary: str | None = None,
    entities: dict | None = None,
    relationships: dict | None = None,
    qa_pairs: list | None = None,
    explanations: str | None = None,
    faqs: list | None = None,
    tags: list[str] | None = None,
    tokens_used: int = 0,
) -> TeacherOutput:
    """Store teacher LLM output for a chunk."""
    output = TeacherOutput(
        chunk_id=chunk_id,
        summary=summary,
        entities=entities,
        relationships=relationships,
        qa_pairs=qa_pairs,
        explanations=explanations,
        faqs=faqs,
        tags=tags,
        tokens_used=tokens_used,
    )
    db.add(output)
    await db.flush()
    await db.refresh(output)
    logger.info(f"Stored teacher output for chunk {chunk_id}")
    return output


async def get_teacher_output_by_chunk(
    db: AsyncSession,
    chunk_id: uuid.UUID,
) -> TeacherOutput | None:
    """Get teacher output for a specific chunk."""
    result = await db.execute(
        select(TeacherOutput).where(TeacherOutput.chunk_id == chunk_id)
    )
    return result.scalar_one_or_none()


async def get_teacher_outputs_by_document(
    db: AsyncSession,
    document_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
) -> tuple[list[TeacherOutput], int]:
    """Get all teacher outputs for chunks belonging to a document."""
    from app.modules.chunks.models import Chunk

    query = (
        select(TeacherOutput)
        .join(Chunk, TeacherOutput.chunk_id == Chunk.id)
        .where(Chunk.document_id == document_id)
        .offset(skip)
        .limit(limit)
    )
    count_query = (
        select(func.count(TeacherOutput.id))
        .join(Chunk, TeacherOutput.chunk_id == Chunk.id)
        .where(Chunk.document_id == document_id)
    )

    result = await db.execute(query)
    outputs = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return outputs, total


async def get_teacher_stats(db: AsyncSession) -> dict:
    """Get aggregated Teacher LLM usage statistics."""
    result = await db.execute(
        select(
            func.count(TeacherOutput.id).label("total"),
            func.sum(TeacherOutput.tokens_used).label("total_tokens"),
            func.avg(TeacherOutput.tokens_used).label("avg_tokens"),
        )
    )
    row = result.one()
    return {
        "total_chunks_processed": row.total or 0,
        "total_tokens_used": row.total_tokens or 0,
        "average_tokens_per_chunk": float(row.avg_tokens or 0),
    }
