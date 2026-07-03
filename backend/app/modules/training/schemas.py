"""
Training Module - Pydantic Schemas.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.modules.training.models import TrainingStatus


class TrainingStartRequest(BaseModel):
    """Request to start a training run."""
    dataset_id: uuid.UUID = Field(..., description="Dataset to train on")
    base_model: str | None = Field(None, description="Base model ID (uses default if not provided)")
    lora_rank: int = Field(default=16, ge=4, le=128, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=8, le=256, description="LoRA alpha")
    epochs: int = Field(default=10, ge=1, le=50, description="Training epochs")
    batch_size: int = Field(default=4, ge=1, le=64, description="Training batch size")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    quantize: bool = Field(default=True, description="Quantize model after training")
    quantization_format: str = Field(default="q4_k_m", description="Quantization format")
    use_context: bool = Field(default=True, description="Whether to include context in the training samples (disable for closed-book QA)")
    user_prompt: str | None = Field(default=None, description="Optional user persona/goal prompt (e.g. 'you are an HR reading a resume'). Will be amplified by Gemini and used during document Q&A generation.")
    model_name: str | None = Field(default=None, description="Optional custom name for the trained model version")


class TrainingRunResponse(BaseModel):
    """Response schema for a training run."""
    id: uuid.UUID
    dataset_id: uuid.UUID | None
    base_model: str
    status: TrainingStatus
    lora_config: dict | None = None
    training_config: dict | None = None
    metrics: dict | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    created_at: datetime
    model_version: str | None = None

    @field_validator("model_version", mode="before")
    @classmethod
    def get_version_string(cls, v):
        if v is not None and hasattr(v, "version"):
            return v.version
        if isinstance(v, str):
            return v
        return None

    model_config = {"from_attributes": True}


class TrainingRunListResponse(BaseModel):
    """Response for listing training runs."""
    runs: list[TrainingRunResponse]
    total: int


class TrainingStartResponse(BaseModel):
    """Response after starting a training run."""
    id: uuid.UUID
    status: TrainingStatus
    task_id: str
    message: str = "Training job queued."
