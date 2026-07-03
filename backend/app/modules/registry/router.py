"""
Registry Module - API Router.

Endpoints for managing model versions.
"""

import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.modules.registry import service
from app.modules.registry.schemas import (
    ModelVersionListResponse,
    ModelVersionResponse,
)

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("/", response_model=ModelVersionListResponse)
async def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List all registered model versions."""
    models, total = await service.list_model_versions(db, skip=skip, limit=limit)
    return ModelVersionListResponse(
        models=[ModelVersionResponse.model_validate(m) for m in models],
        total=total,
    )


@router.get("/active", response_model=ModelVersionResponse | None)
async def get_active_model(db: AsyncSession = Depends(get_db)):
    """Get the currently active (deployed) model version."""
    model = await service.get_active_model(db)
    if model:
        return ModelVersionResponse.model_validate(model)
    return None


@router.get("/base-model")
async def get_base_model():
    """Get the default base student model name."""
    from app.config import get_settings
    return {"base_model": get_settings().BASE_STUDENT_MODEL}


@router.get("/{model_id}", response_model=ModelVersionResponse)
async def get_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a model version by ID with its details and metrics."""
    model = await service.get_model_version(db, model_id)
    return ModelVersionResponse.model_validate(model)


@router.post("/{model_id}/activate", response_model=ModelVersionResponse)
async def activate_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Set a model version as the active deployment version."""
    model = await service.activate_model(db, model_id)
    return ModelVersionResponse.model_validate(model)


@router.get("/{model_id}/download")
async def download_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Download the quantized model file for on-device deployment."""
    model = await service.get_model_version(db, model_id)
    return FileResponse(
        path=model.model_path,
        filename=f"student-model-{model.version}.gguf",
        media_type="application/octet-stream",
    )


@router.get("/{model_id}/metrics")
async def get_model_metrics(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed training and evaluation metrics for a model version."""
    model = await service.get_model_version(db, model_id)
    return {
        "model_id": str(model.id),
        "version": model.version,
        "metrics": model.metrics or {},
        "base_model": model.base_model,
        "dataset_version": model.dataset_version,
        "quantization_format": model.quantization_format,
    }
