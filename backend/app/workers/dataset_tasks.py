"""
Dataset Generation Tasks.

Celery tasks for generating training datasets from Teacher outputs.
"""

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.dataset_tasks.generate_dataset_task")
def generate_dataset_task(
    version: str | None = None,
    document_ids: list[str] | None = None,
    description: str | None = None,
):
    """Generate a training dataset asynchronously."""
    import asyncio
    import uuid
    from app.database import AsyncSessionLocal
    from app.modules.datasets.service import generate_dataset

    async def _run():
        async with AsyncSessionLocal() as session:
            doc_uuids = [uuid.UUID(did) for did in document_ids] if document_ids else None
            dataset = await generate_dataset(
                db=session,
                version=version,
                document_ids=doc_uuids,
                description=description,
            )
            await session.commit()
            return {
                "dataset_id": str(dataset.id),
                "version": dataset.version,
                "total_samples": dataset.total_samples,
            }

    return asyncio.run(_run())
