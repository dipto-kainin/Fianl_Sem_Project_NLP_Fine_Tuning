"""
Training Module - API Router.

Endpoints for managing student model training runs.
"""

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.dependencies import get_db
from app.modules.training import service
from app.modules.training.models import TrainingStatus
from app.modules.training.schemas import (
    TrainingRunListResponse,
    TrainingRunResponse,
    TrainingStartRequest,
    TrainingStartResponse,
)

settings = get_settings()
router = APIRouter(prefix="/training", tags=["Training"])


@router.post("/start", response_model=TrainingStartResponse, status_code=201)
async def start_training(
    request: TrainingStartRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Start a new LoRA fine-tuning training run.

    Training runs asynchronously on a background worker. Use the returned
    run ID to check status and metrics.
    """
    from app.workers.training_tasks import run_training_pipeline

    lora_config = {
        "r": request.lora_rank,
        "alpha": request.lora_alpha,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "dropout": 0.05,
    }
    training_config = {
        "epochs": request.epochs,
        "batch_size": request.batch_size,
        "learning_rate": request.learning_rate,
        "quantize": request.quantize,
        "quantization_format": request.quantization_format,
        "use_context": request.use_context,
        "model_name": request.model_name.strip() if request.model_name else None,
    }

    # If the user supplied a persona/goal prompt, amplify it with Gemini
    # and store the result so the Celery worker can use it during SFT formatting
    if request.user_prompt and request.user_prompt.strip():
        from app.modules.teacher.gemini_client import amplify_user_prompt
        amplified = amplify_user_prompt(request.user_prompt.strip())
        training_config["system_prompt"] = amplified
    else:
        training_config["system_prompt"] = None

    # Create training run record
    run = await service.create_training_run(
        db=db,
        dataset_id=request.dataset_id,
        base_model=request.base_model or settings.BASE_STUDENT_MODEL,
        lora_config=lora_config,
        training_config=training_config,
    )

    # Dispatch Celery task
    task = run_training_pipeline.delay(str(run.id))

    return TrainingStartResponse(
        id=run.id,
        status=run.status,
        task_id=task.id,
    )


@router.get("/", response_model=TrainingRunListResponse)
async def list_training_runs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List all training runs with pagination."""
    runs, total = await service.list_training_runs(db, skip=skip, limit=limit)
    return TrainingRunListResponse(
        runs=[TrainingRunResponse.model_validate(r) for r in runs],
        total=total,
    )


@router.get("/{run_id}", response_model=TrainingRunResponse)
async def get_training_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a training run by ID with current status and metrics."""
    run = await service.get_training_run(db, run_id)
    return TrainingRunResponse.model_validate(run)


@router.post("/{run_id}/cancel")
async def cancel_training_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Cancel a queued or running training job."""
    run = await service.get_training_run(db, run_id)
    if run.status in (TrainingStatus.QUEUED, TrainingStatus.TRAINING):
        await service.update_training_status(
            db, run_id, TrainingStatus.CANCELLED
        )
        return {"message": "Training run cancelled."}
    return {"message": f"Cannot cancel run with status: {run.status}"}


@router.post("/{run_id}/retry", response_model=TrainingStartResponse)
async def retry_training_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retry a failed or cancelled training run."""
    from app.workers.training_tasks import run_training_pipeline

    run = await service.get_training_run(db, run_id)
    
    # Update status to QUEUED and reset properties
    run.status = TrainingStatus.QUEUED
    run.started_at = None
    run.completed_at = None
    run.error_message = None
    run.metrics = {}
    
    await db.commit()
    await db.refresh(run)

    # Dispatch Celery task
    task = run_training_pipeline.delay(str(run.id))

    return TrainingStartResponse(
        id=run.id,
        status=run.status,
        task_id=task.id,
    )


@router.delete("/{run_id}", status_code=204)
async def delete_training_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a training run record and its files from disk."""
    import shutil
    from pathlib import Path
    from sqlalchemy import select
    from app.modules.registry.models import ModelVersion
    from app.modules.training.models import TrainingRun
    from app.utils.student_inference import clear_student_model_cache

    # 1. Fetch training run
    run = await service.get_training_run(db, run_id)

    # 2. Clear student model cache if this model was loaded
    clear_student_model_cache()

    # 3. Delete from database (cascade handles related objects, but delete ModelVersion explicitly to be safe)
    version_result = await db.execute(
        select(ModelVersion).where(ModelVersion.training_run_id == run_id)
    )
    versions = version_result.scalars().all()
    for version in versions:
        await db.delete(version)

    await db.delete(run)
    await db.commit()

    # 4. Remove files from disk
    if run.output_dir:
        output_path = Path(run.output_dir)
        if output_path.exists() and output_path.is_dir():
            try:
                shutil.rmtree(output_path)
            except Exception as e:
                pass

    return None
