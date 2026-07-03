"""
Registry Module - Pydantic Schemas.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel


class ModelVersionResponse(BaseModel):
    """Response schema for a model version."""
    id: uuid.UUID
    version: str
    training_run_id: uuid.UUID | None = None
    base_model: str
    model_path: str
    quantization_format: str | None = None
    file_size: int | None = None
    metrics: dict | None = None
    dataset_version: str | None = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelVersionListResponse(BaseModel):
    """Response for listing model versions."""
    models: list[ModelVersionResponse]
    total: int
