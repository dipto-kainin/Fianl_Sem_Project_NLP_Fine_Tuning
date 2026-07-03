"""
Documents Module - Business Logic Service.

Handles document CRUD operations and processing orchestration.
"""

import logging
import uuid
from pathlib import Path

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.modules.documents.models import Document, DocumentStatus, FileType
from app.utils.exceptions import (
    DocumentNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.utils.file_storage import save_upload, delete_file, get_file_extension

logger = logging.getLogger(__name__)
settings = get_settings()

# Map file extensions to FileType enum
EXTENSION_MAP = {
    "pdf": FileType.PDF,
    "docx": FileType.DOCX,
    "txt": FileType.TXT,
    "md": FileType.MD,
    "markdown": FileType.MD,
    "html": FileType.HTML,
    "htm": FileType.HTML,
    "epub": FileType.EPUB,
}


async def upload_document(
    db: AsyncSession,
    filename: str,
    file_path: str,
    file_size: int,
) -> Document:
    """
    Create a document record after file upload.

    Args:
        db: Database session.
        filename: Original filename.
        file_path: Storage path of the uploaded file.
        file_size: File size in bytes.

    Returns:
        Created Document instance.
    """
    # Validate file type
    ext = get_file_extension(filename)
    file_type = EXTENSION_MAP.get(ext)
    if not file_type:
        raise UnsupportedFileTypeError(ext)

    # Validate file size
    size_mb = file_size / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise FileTooLargeError(size_mb, settings.MAX_UPLOAD_SIZE_MB)

    document = Document(
        filename=filename,
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    await db.flush()
    await db.refresh(document)

    logger.info(f"Document uploaded: {document.id} ({filename}, {size_mb:.1f} MB)")
    return document


async def get_document(db: AsyncSession, document_id: uuid.UUID) -> Document:
    """
    Get a document by ID.

    Args:
        db: Database session.
        document_id: Document UUID.

    Returns:
        Document instance.

    Raises:
        DocumentNotFoundError: If document doesn't exist.
    """
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise DocumentNotFoundError(str(document_id))
    return document


async def list_documents(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    status: DocumentStatus | None = None,
) -> tuple[list[Document], int]:
    """
    List documents with pagination and optional filtering.

    Args:
        db: Database session.
        skip: Number of records to skip.
        limit: Maximum records to return.
        status: Optional status filter.

    Returns:
        Tuple of (documents list, total count).
    """
    query = select(Document)
    count_query = select(func.count(Document.id))

    if status:
        query = query.where(Document.status == status)
        count_query = count_query.where(Document.status == status)

    query = query.order_by(Document.created_at.desc()).offset(skip).limit(limit)

    result = await db.execute(query)
    documents = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return documents, total


async def update_document_status(
    db: AsyncSession,
    document_id: uuid.UUID,
    status: DocumentStatus,
    error_message: str | None = None,
    **kwargs,
) -> Document:
    """
    Update a document's status and optional fields.

    Args:
        db: Database session.
        document_id: Document UUID.
        status: New status.
        error_message: Optional error message for failed status.
        **kwargs: Additional fields to update (raw_text, page_count, etc.).

    Returns:
        Updated Document instance.
    """
    document = await get_document(db, document_id)
    document.status = status
    if error_message:
        document.error_message = error_message
    for key, value in kwargs.items():
        if hasattr(document, key):
            setattr(document, key, value)
    await db.flush()
    await db.refresh(document)
    return document


async def delete_document(db: AsyncSession, document_id: uuid.UUID) -> bool:
    """
    Delete a document and its file from storage.

    Args:
        db: Database session.
        document_id: Document UUID.

    Returns:
        True if deleted successfully.
    """
    document = await get_document(db, document_id)

    # Delete the file from storage
    await delete_file(document.file_path)

    # Delete from database (cascades to chunks)
    await db.delete(document)
    await db.flush()

    logger.info(f"Document deleted: {document_id}")
    return True
