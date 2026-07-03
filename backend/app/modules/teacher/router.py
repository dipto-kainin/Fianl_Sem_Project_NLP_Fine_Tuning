"""
Teacher Module - API Router.

Endpoints for Teacher LLM operations and statistics.
"""

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.modules.teacher import service
from app.modules.teacher.schemas import (
    TeacherOutputListResponse,
    TeacherOutputResponse,
    TeacherProcessRequest,
    TeacherStatsResponse,
)

router = APIRouter(prefix="/teacher", tags=["Teacher LLM"])


@router.get("/outputs/document/{document_id}", response_model=TeacherOutputListResponse)
async def list_teacher_outputs(
    document_id: uuid.UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """List all Teacher LLM outputs for a document's chunks."""
    outputs, total = await service.get_teacher_outputs_by_document(
        db, document_id, skip=skip, limit=limit
    )
    return TeacherOutputListResponse(
        outputs=[TeacherOutputResponse.model_validate(o) for o in outputs],
        total=total,
    )


@router.get("/outputs/chunk/{chunk_id}", response_model=TeacherOutputResponse | None)
async def get_teacher_output(
    chunk_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get Teacher LLM output for a specific chunk."""
    output = await service.get_teacher_output_by_chunk(db, chunk_id)
    if output:
        return TeacherOutputResponse.model_validate(output)
    return None


@router.post("/process")
async def trigger_teacher_processing(
    request: TeacherProcessRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger Teacher LLM processing for specified chunks.
    Processing runs asynchronously via a background task.
    """
    from app.workers.teacher_tasks import process_chunks_with_teacher

    task = process_chunks_with_teacher.delay(
        chunk_ids=[str(cid) for cid in request.chunk_ids] if request.chunk_ids else None,
        document_id=str(request.document_id) if request.document_id else None,
    )
    return {"task_id": task.id, "message": "Teacher processing started."}


@router.get("/stats", response_model=TeacherStatsResponse)
async def get_teacher_stats(db: AsyncSession = Depends(get_db)):
    """Get aggregated Teacher LLM usage statistics."""
    stats = await service.get_teacher_stats(db)
    return TeacherStatsResponse(**stats)
