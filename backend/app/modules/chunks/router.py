"""
Chunks Module - API Router.

Endpoints for listing and retrieving document chunks.
"""

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.modules.chunks import service
from app.modules.chunks.schemas import ChunkListResponse, ChunkResponse

router = APIRouter(prefix="/chunks", tags=["Chunks"])


@router.get("/document/{document_id}", response_model=ChunkListResponse)
async def list_chunks_by_document(
    document_id: uuid.UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """List all chunks for a specific document, ordered by chunk index."""
    chunks, total = await service.get_chunks_by_document(
        db, document_id, skip=skip, limit=limit
    )
    return ChunkListResponse(
        chunks=[ChunkResponse.model_validate(c) for c in chunks],
        total=total,
    )


@router.get("/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    chunk_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific chunk by ID."""
    chunk = await service.get_chunk(db, chunk_id)
    return ChunkResponse.model_validate(chunk)
