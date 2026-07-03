"""
Datasets Module - Business Logic Service.

Converts Teacher LLM outputs into supervised training datasets.
"""

import json
import logging
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.modules.chunks.models import Chunk
from app.modules.datasets.models import Dataset, DatasetSample, DatasetStatus, DifficultyLevel
from app.modules.teacher.models import TeacherOutput
from app.utils.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)
settings = get_settings()


async def generate_dataset(
    db: AsyncSession,
    version: str | None = None,
    document_ids: list[uuid.UUID] | None = None,
    description: str | None = None,
) -> Dataset:
    """
    Generate a supervised training dataset from Teacher LLM outputs.

    Converts QA pairs, FAQs, and explanations into instruction-context-response
    training samples.

    Args:
        db: Database session.
        version: Dataset version string (auto-generated if None).
        document_ids: Optional filter to specific documents.
        description: Description for this dataset version.

    Returns:
        Created Dataset with samples.
    """
    # Auto-generate version
    if not version:
        count_result = await db.execute(select(func.count(Dataset.id)))
        count = count_result.scalar_one()
        version = f"v{count + 1}"

    # Create dataset record
    dataset = Dataset(
        version=version,
        status=DatasetStatus.GENERATING,
        description=description,
    )
    db.add(dataset)
    await db.flush()

    # Query teacher outputs - only those NOT already used in a previous dataset
    # This ensures each new version only contains fresh/unseen data
    already_used_subquery = (
        select(DatasetSample.source_chunk_id)
        .where(DatasetSample.source_chunk_id.isnot(None))
        .scalar_subquery()
    )
    query = (
        select(TeacherOutput)
        .join(Chunk, TeacherOutput.chunk_id == Chunk.id)
        .where(TeacherOutput.chunk_id.not_in(already_used_subquery))
    )
    if document_ids:
        query = query.where(Chunk.document_id.in_(document_ids))

    result = await db.execute(query)
    teacher_outputs = list(result.scalars().all())

    if not teacher_outputs:
        dataset.status = DatasetStatus.READY
        dataset.total_samples = 0
        await db.flush()
        return dataset

    # Convert teacher outputs to training samples
    samples: list[DatasetSample] = []
    category_counter: Counter = Counter()

    for output in teacher_outputs:
        chunk_id = output.chunk_id

        # Get the chunk text for context
        chunk_result = await db.execute(select(Chunk).where(Chunk.id == chunk_id))
        chunk = chunk_result.scalar_one_or_none()
        context_text = chunk.text if chunk else ""

        # 1. Convert QA pairs
        if output.qa_pairs:
            for qa in output.qa_pairs:
                if isinstance(qa, dict) and "instruction" in qa and "answer" in qa:
                    difficulty = _estimate_difficulty(qa["answer"])
                    category = (output.tags[0] if output.tags else "general")
                    category_counter[category] += 1

                    samples.append(DatasetSample(
                        dataset_id=dataset.id,
                        instruction=qa["instruction"],
                        context=context_text[:2000],  # Truncate context
                        response=qa["answer"],
                        difficulty=difficulty,
                        category=category,
                        source_chunk_id=chunk_id,
                        priority=max(1, min(3, int(qa.get("priority", 1)))),
                    ))

    # Batch insert samples
    db.add_all(samples)

    # Update dataset metadata
    dataset.total_samples = len(samples)
    dataset.status = DatasetStatus.READY
    dataset.categories = dict(category_counter)

    # Export to JSONL file
    file_path = await _export_to_jsonl(dataset, samples)
    dataset.file_path = file_path

    await db.flush()
    await db.refresh(dataset)

    logger.info(f"Generated dataset {version}: {len(samples)} samples")
    return dataset


async def _export_to_jsonl(dataset: Dataset, samples: list[DatasetSample]) -> str:
    """Export dataset samples to a JSONL file."""
    output_dir = Path(settings.DATASETS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"dataset_{dataset.version}.jsonl"

    with open(file_path, "w", encoding="utf-8") as f:
        for sample in samples:
            entry = {
                "instruction": sample.instruction,
                "context": sample.context or "",
                "response": sample.response,
                "difficulty": sample.difficulty.value,
                "category": sample.category or "general",
                "priority": getattr(sample, "priority", 1),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(samples)} samples to {file_path}")
    return str(file_path)


def _estimate_difficulty(answer: str) -> DifficultyLevel:
    """Estimate difficulty based on answer length/complexity."""
    word_count = len(answer.split())
    if word_count < 20:
        return DifficultyLevel.EASY
    elif word_count < 80:
        return DifficultyLevel.MEDIUM
    else:
        return DifficultyLevel.HARD


async def update_sample(
    db: AsyncSession,
    sample_id: uuid.UUID,
    instruction: str | None,
    response: str | None,
    context: str | None,
    priority: int | None,
) -> DatasetSample:
    """Update a dataset sample and re-export the parent dataset's JSONL."""
    result = await db.execute(select(DatasetSample).where(DatasetSample.id == sample_id))
    sample = result.scalar_one_or_none()
    if not sample:
        raise ValueError(f"Sample not found: {sample_id}")

    if instruction is not None:
        sample.instruction = instruction
    if response is not None:
        sample.response = response
    if context is not None:
        sample.context = context
    if priority is not None:
        sample.priority = max(1, min(3, int(priority)))

    await db.commit()
    await db.refresh(sample)

    # Re-export the JSONL so training picks up the edit
    await _reexport_dataset_jsonl(db, sample.dataset_id)
    return sample


async def delete_sample(
    db: AsyncSession,
    sample_id: uuid.UUID,
) -> uuid.UUID:
    """Delete a dataset sample and re-export the parent dataset's JSONL."""
    result = await db.execute(select(DatasetSample).where(DatasetSample.id == sample_id))
    sample = result.scalar_one_or_none()
    if not sample:
        raise ValueError(f"Sample not found: {sample_id}")

    dataset_id = sample.dataset_id
    await db.delete(sample)

    # Decrement total_samples
    ds_result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = ds_result.scalar_one_or_none()
    if dataset and dataset.total_samples > 0:
        dataset.total_samples -= 1

    await db.commit()

    # Re-export the JSONL
    await _reexport_dataset_jsonl(db, dataset_id)
    return dataset_id


async def _reexport_dataset_jsonl(db: AsyncSession, dataset_id: uuid.UUID) -> None:
    """Re-export all samples for a dataset to its JSONL file."""
    ds_result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = ds_result.scalar_one_or_none()
    if not dataset:
        return

    samples_result = await db.execute(
        select(DatasetSample).where(DatasetSample.dataset_id == dataset_id)
    )
    all_samples = list(samples_result.scalars().all())
    await _export_to_jsonl(dataset, all_samples)
    logger.info(f"Re-exported JSONL for dataset {dataset_id} ({len(all_samples)} samples)")


async def get_dataset(db: AsyncSession, dataset_id: uuid.UUID) -> Dataset:
    """Get a dataset by ID."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise DatasetNotFoundError(str(dataset_id))
    return dataset


async def list_datasets(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[Dataset], int]:
    """List all datasets with pagination."""
    query = select(Dataset).order_by(Dataset.created_at.desc()).offset(skip).limit(limit)
    count_query = select(func.count(Dataset.id))

    result = await db.execute(query)
    datasets = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return datasets, total


async def get_dataset_samples(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[DatasetSample], int]:
    """Get paginated samples for a dataset."""
    query = (
        select(DatasetSample)
        .where(DatasetSample.dataset_id == dataset_id)
        .offset(skip)
        .limit(limit)
    )
    count_query = (
        select(func.count(DatasetSample.id))
        .where(DatasetSample.dataset_id == dataset_id)
    )

    result = await db.execute(query)
    samples = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return samples, total
