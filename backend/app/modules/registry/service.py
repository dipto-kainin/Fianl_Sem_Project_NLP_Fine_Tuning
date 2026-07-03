"""
Registry Module - Business Logic Service.
"""

import logging
import uuid

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.registry.models import ModelVersion
from app.utils.exceptions import ModelVersionNotFoundError

logger = logging.getLogger(__name__)


async def register_model(
    db: AsyncSession,
    version: str,
    base_model: str,
    model_path: str,
    training_run_id: uuid.UUID | None = None,
    quantization_format: str | None = None,
    file_size: int | None = None,
    metrics: dict | None = None,
    dataset_version: str | None = None,
) -> ModelVersion:
    """Register a new model version in the registry."""
    model = ModelVersion(
        version=version,
        training_run_id=training_run_id,
        base_model=base_model,
        model_path=model_path,
        quantization_format=quantization_format,
        file_size=file_size,
        metrics=metrics,
        dataset_version=dataset_version,
        is_active=False,
    )
    db.add(model)
    await db.flush()
    await db.refresh(model)
    logger.info(f"Registered model version: {version}")
    return model


async def get_model_version(db: AsyncSession, model_id: uuid.UUID) -> ModelVersion:
    """Get a model version by ID."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise ModelVersionNotFoundError(str(model_id))
    return model


async def get_active_model(db: AsyncSession) -> ModelVersion | None:
    """Get the currently active model version."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)
    )
    return result.scalar_one_or_none()


async def list_model_versions(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[ModelVersion], int]:
    """List all model versions with pagination."""
    query = (
        select(ModelVersion)
        .order_by(ModelVersion.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    count_query = select(func.count(ModelVersion.id))

    result = await db.execute(query)
    models = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return models, total


async def activate_model(db: AsyncSession, model_id: uuid.UUID) -> ModelVersion:
    """Set a model version as the active deployment version."""
    from app.utils.student_inference import clear_student_model_cache

    # Deactivate all current models
    await db.execute(
        update(ModelVersion).values(is_active=False)
    )

    # Activate the specified model
    model = await get_model_version(db, model_id)
    model.is_active = True
    await db.flush()
    await db.refresh(model)

    # Clear cached model to load the new one
    clear_student_model_cache()

    logger.info(f"Activated model version: {model.version}")
    return model
