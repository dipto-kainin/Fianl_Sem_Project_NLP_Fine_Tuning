"""
Datasets Module - Pydantic Schemas.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.modules.datasets.models import DatasetStatus, DifficultyLevel


class DatasetSampleResponse(BaseModel):
    """Response schema for a single dataset sample."""
    id: uuid.UUID
    instruction: str
    context: str | None = None
    response: str
    difficulty: DifficultyLevel
    category: str | None = None
    source_chunk_id: uuid.UUID | None = None
    priority: int = 1

    model_config = {"from_attributes": True}


class DatasetSampleUpdateRequest(BaseModel):
    """Request to update a dataset sample."""
    instruction: str | None = Field(None, description="Updated instruction text")
    response: str | None = Field(None, description="Updated response text")
    context: str | None = Field(None, description="Updated context text")
    priority: int | None = Field(None, ge=1, le=3, description="Priority 1–3")


class DatasetResponse(BaseModel):
    """Response schema for a dataset."""
    id: uuid.UUID
    version: str
    total_samples: int
    status: DatasetStatus
    categories: dict | None = None
    description: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetListResponse(BaseModel):
    """Response for listing datasets."""
    datasets: list[DatasetResponse]
    total: int


class DatasetSampleListResponse(BaseModel):
    """Response for listing dataset samples."""
    samples: list[DatasetSampleResponse]
    total: int


class DatasetGenerateRequest(BaseModel):
    """Request to generate a new training dataset."""
    version: str | None = Field(
        None,
        description="Dataset version string. Auto-generated if not provided."
    )
    document_ids: list[uuid.UUID] | None = Field(
        None,
        description="Specific documents to include. If None, uses all processed documents."
    )
    description: str | None = Field(None, description="Description of this dataset version")
