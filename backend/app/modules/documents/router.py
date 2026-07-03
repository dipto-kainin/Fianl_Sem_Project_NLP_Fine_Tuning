"""
Documents Module - API Router.

Endpoints for document upload, listing, retrieval, processing, and deletion.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.modules.documents import service
from app.modules.documents.models import DocumentStatus
from app.modules.documents.schemas import (
    DocumentListResponse,
    DocumentProcessRequest,
    DocumentProcessResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.utils.file_storage import save_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document for processing.

    Supported formats: PDF, DOCX, TXT, Markdown, HTML, EPUB.
    The document will be stored and can then be processed via the /process endpoint.
    """
    # Save file to storage
    file_path, file_size = await save_upload(file, sub_dir="documents")

    # Create document record
    document = await service.upload_document(
        db=db,
        filename=file.filename or "unknown",
        file_path=file_path,
        file_size=file_size,
    )

    return DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status,
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=200, description="Maximum records to return"),
    status: DocumentStatus | None = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
):
    """List all uploaded documents with pagination and optional status filtering."""
    documents, total = await service.list_documents(db, skip=skip, limit=limit, status=status)
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=total,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a document by ID with its current processing status."""
    document = await service.get_document(db, document_id)
    return DocumentResponse.model_validate(document)


@router.post("/{document_id}/process", response_model=DocumentProcessResponse)
async def process_document(
    document_id: uuid.UUID,
    config: DocumentProcessRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger the full processing pipeline for a document.

    Pipeline: Parse → Chunk → Embed → Teacher Label → Store in Vector DB.
    Processing runs asynchronously via a background task.
    """
    from app.workers.document_tasks import process_document_pipeline

    document = await service.get_document(db, document_id)

    # Update status to processing
    await service.update_document_status(db, document_id, DocumentStatus.PROCESSING)

    # Dispatch Celery task
    config = config or DocumentProcessRequest()
    task = process_document_pipeline.delay(
        str(document_id),
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        run_teacher=config.run_teacher,
    )

    return DocumentProcessResponse(
        id=document.id,
        status=DocumentStatus.PROCESSING,
        task_id=task.id,
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and all associated data (chunks, embeddings, etc.)."""
    await service.delete_document(db, document_id)
    return None
