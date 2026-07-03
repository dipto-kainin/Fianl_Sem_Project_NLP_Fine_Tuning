"""
Documents Module - Pydantic Schemas.

Request/response schemas for the documents API.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.modules.documents.models import DocumentStatus, FileType


# --- Response Schemas ---

class DocumentResponse(BaseModel):
    """Response schema for a single document."""

    id: uuid.UUID
    filename: str
    file_type: FileType
    file_size: int
    status: DocumentStatus
    language: str | None = None
    page_count: int | None = None
    metadata: dict | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def _coerce_metadata(cls, data):
        """
        When built from an ORM object, `data.metadata` resolves to SQLAlchemy's
        MetaData() instance (a name clash). Read `metadata_` (the real column
        attribute) and map it to our `metadata` field instead.
        """
        if hasattr(data, "metadata_"):
            # ORM object path — copy as plain dict
            raw = data.metadata_
            if hasattr(raw, "__dict__") and not isinstance(raw, dict):
                raw = None
            # Return a plain dict representation for Pydantic
            return {
                "id": data.id,
                "filename": data.filename,
                "file_type": data.file_type,
                "file_size": data.file_size,
                "status": data.status,
                "language": data.language,
                "page_count": data.page_count,
                "metadata": raw,
                "error_message": data.error_message,
                "created_at": data.created_at,
                "updated_at": data.updated_at,
            }
        return data


class DocumentListResponse(BaseModel):
    """Response schema for listing documents."""

    documents: list[DocumentResponse]
    total: int


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""

    id: uuid.UUID
    filename: str
    file_type: FileType
    file_size: int
    status: DocumentStatus
    message: str = "Document uploaded successfully. Processing will begin shortly."

    model_config = {"from_attributes": True}


class DocumentProcessResponse(BaseModel):
    """Response after triggering document processing."""

    id: uuid.UUID
    status: DocumentStatus
    task_id: str
    message: str = "Document processing started."


# --- Request Schemas ---

class DocumentProcessRequest(BaseModel):
    """Request to configure document processing."""

    chunk_size: int = Field(default=512, ge=100, le=4096, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=64, ge=0, le=512, description="Overlap between chunks in tokens")
    run_teacher: bool = Field(default=True, description="Whether to run Teacher LLM pipeline")
