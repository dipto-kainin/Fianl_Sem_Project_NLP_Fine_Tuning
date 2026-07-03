"""
Datasets Module - API Router.

Endpoints for generating, listing, and downloading training datasets.
"""

import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.modules.datasets import service
from app.modules.datasets.schemas import (
    DatasetGenerateRequest,
    DatasetListResponse,
    DatasetResponse,
    DatasetSampleListResponse,
    DatasetSampleResponse,
    DatasetSampleUpdateRequest,
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/generate", response_model=DatasetResponse, status_code=201)
async def generate_dataset(
    request: DatasetGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a new training dataset from Teacher LLM outputs.

    Converts all QA pairs, FAQs, and explanations into supervised
    instruction-context-response training samples. Exports as JSONL.
    """
    dataset = await service.generate_dataset(
        db=db,
        version=request.version,
        document_ids=request.document_ids,
        description=request.description,
    )
    return DatasetResponse.model_validate(dataset)


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List all training datasets with pagination."""
    datasets, total = await service.list_datasets(db, skip=skip, limit=limit)
    return DatasetListResponse(
        datasets=[DatasetResponse.model_validate(d) for d in datasets],
        total=total,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a dataset by ID."""
    dataset = await service.get_dataset(db, dataset_id)
    return DatasetResponse.model_validate(dataset)


@router.get("/{dataset_id}/samples", response_model=DatasetSampleListResponse)
async def list_dataset_samples(
    dataset_id: uuid.UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Preview samples from a dataset with pagination."""
    samples, total = await service.get_dataset_samples(
        db, dataset_id, skip=skip, limit=limit
    )
    return DatasetSampleListResponse(
        samples=[DatasetSampleResponse.model_validate(s) for s in samples],
        total=total,
    )


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Download the dataset as a JSONL file."""
    dataset = await service.get_dataset(db, dataset_id)
    if not dataset.file_path:
        return {"error": "Dataset file not available."}
    return FileResponse(
        path=dataset.file_path,
        filename=f"dataset_{dataset.version}.jsonl",
        media_type="application/jsonl",
    )


@router.patch("/{dataset_id}/samples/{sample_id}", response_model=DatasetSampleResponse)
async def update_sample(
    dataset_id: uuid.UUID,
    sample_id: uuid.UUID,
    request: DatasetSampleUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Edit a single dataset sample (instruction, response, context, priority)."""
    sample = await service.update_sample(
        db=db,
        sample_id=sample_id,
        instruction=request.instruction,
        response=request.response,
        context=request.context,
        priority=request.priority,
    )
    return DatasetSampleResponse.model_validate(sample)


@router.delete("/{dataset_id}/samples/{sample_id}", status_code=204)
async def delete_sample(
    dataset_id: uuid.UUID,
    sample_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a single dataset sample."""
    await service.delete_sample(db=db, sample_id=sample_id)
    return None
