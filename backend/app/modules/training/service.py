"""
Training Module - Business Logic Service.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.modules.training.models import TrainingRun, TrainingStatus
from app.utils.exceptions import TrainingRunNotFoundError

logger = logging.getLogger(__name__)
settings = get_settings()


async def create_training_run(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    base_model: str | None = None,
    lora_config: dict | None = None,
    training_config: dict | None = None,
) -> TrainingRun:
    """Create a new training run record."""
    run = TrainingRun(
        dataset_id=dataset_id,
        base_model=base_model or settings.BASE_STUDENT_MODEL,
        status=TrainingStatus.QUEUED,
        lora_config=lora_config,
        training_config=training_config,
    )
    db.add(run)
    await db.flush()
    await db.refresh(run)
    logger.info(f"Created training run: {run.id}")
    return run


async def get_training_run(db: AsyncSession, run_id: uuid.UUID) -> TrainingRun:
    """Get a training run by ID."""
    result = await db.execute(
        select(TrainingRun)
        .options(selectinload(TrainingRun.model_version))
        .where(TrainingRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise TrainingRunNotFoundError(str(run_id))
    return run


async def list_training_runs(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[TrainingRun], int]:
    """List all training runs with pagination."""
    query = (
        select(TrainingRun)
        .options(selectinload(TrainingRun.model_version))
        .order_by(TrainingRun.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    count_query = select(func.count(TrainingRun.id))

    result = await db.execute(query)
    runs = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return runs, total


async def update_training_status(
    db: AsyncSession,
    run_id: uuid.UUID,
    status: TrainingStatus,
    error_message: str | None = None,
    metrics: dict | None = None,
    output_dir: str | None = None,
) -> TrainingRun:
    """Update a training run's status and optional fields."""
    run = await get_training_run(db, run_id)
    run.status = status

    if status == TrainingStatus.TRAINING and not run.started_at:
        run.started_at = datetime.now(timezone.utc)
    if status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
        run.completed_at = datetime.now(timezone.utc)
    if error_message:
        run.error_message = error_message
    if metrics:
        run.metrics = metrics
    if output_dir:
        run.output_dir = output_dir

    await db.flush()
    await db.refresh(run)
    return run
