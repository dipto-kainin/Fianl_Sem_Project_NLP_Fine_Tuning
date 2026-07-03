"""
Training Pipeline Tasks.

Celery tasks for LoRA fine-tuning, adapter merging, quantization,
and model registration.
"""

import logging
import uuid
from pathlib import Path

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _get_sync_session():
    """Create a synchronous database session for Celery workers."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.config import get_settings

    settings = get_settings()
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2").replace("postgresql+psycopg2", "postgresql")
    engine = create_engine(sync_url)
    Session = sessionmaker(bind=engine)
    return Session()


@celery_app.task(name="app.workers.training_tasks.run_training_pipeline", bind=True)
def run_training_pipeline(self, run_id: str):
    """
    Full training pipeline:
    1. Load dataset
    2. LoRA fine-tuning
    3. Merge adapters
    4. Quantize model
    5. Register in model registry
    """
    from app.modules.training.models import TrainingRun, TrainingStatus
    from app.modules.training.trainer import run_lora_training, merge_lora_adapters
    from app.modules.training.quantizer import quantize_to_gguf, get_model_size
    from app.modules.datasets.models import Dataset
    from app.modules.registry.models import ModelVersion
    from app.config import get_settings
    from datetime import datetime, timezone

    settings = get_settings()
    session = _get_sync_session()

    try:
        # Load training run
        run = session.query(TrainingRun).filter(
            TrainingRun.id == uuid.UUID(run_id)
        ).first()
        if not run:
            raise ValueError(f"Training run not found: {run_id}")

        # Update status
        run.status = TrainingStatus.TRAINING
        run.started_at = datetime.now(timezone.utc)
        session.commit()

        # Load dataset
        dataset = session.query(Dataset).filter(
            Dataset.id == run.dataset_id
        ).first()
        if not dataset or not dataset.file_path:
            raise ValueError("Dataset not found or has no file")

        # Configure output directory
        output_dir = str(Path(settings.MODELS_DIR) / f"run_{run_id}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Extract training config
        tc = run.training_config or {}
        lc = run.lora_config or {}

        # Step 1: LoRA Training
        self.update_state(state="TRAINING", meta={"step": "lora_training"})
        logger.info(f"Step 1: Starting LoRA training for run {run_id}")

        metrics = run_lora_training(
            dataset_path=dataset.file_path,
            output_dir=output_dir,
            base_model=run.base_model,
            lora_rank=lc.get("r", settings.LORA_RANK),
            lora_alpha=lc.get("alpha", settings.LORA_ALPHA),
            epochs=tc.get("epochs", settings.TRAINING_EPOCHS),
            batch_size=tc.get("batch_size", settings.TRAINING_BATCH_SIZE),
            learning_rate=tc.get("learning_rate", settings.LEARNING_RATE),
            use_context=tc.get("use_context", True),
            system_prompt=tc.get("system_prompt", None),
        )

        run.metrics = metrics
        run.output_dir = output_dir
        session.commit()

        # Step 2: Merge adapters
        self.update_state(state="MERGING", meta={"step": "merging_adapters"})
        run.status = TrainingStatus.MERGING
        session.commit()

        logger.info("Step 2: Merging LoRA adapters")
        merged_dir = merge_lora_adapters(
            base_model=run.base_model,
            adapter_dir=output_dir,
            output_dir=output_dir,
        )

        # Step 3: Quantize (if enabled)
        model_path = merged_dir
        quantization_format = tc.get("quantization_format", "q4_k_m")

        if tc.get("quantize", True):
            self.update_state(state="QUANTIZING", meta={"step": "quantizing"})
            run.status = TrainingStatus.QUANTIZING
            session.commit()

            logger.info(f"Step 3: Quantizing to {quantization_format}")
            model_path = quantize_to_gguf(
                model_dir=merged_dir,
                output_dir=output_dir,
                quantization_type=quantization_format,
            )

        # Step 4: Register model version
        self.update_state(state="REGISTERING", meta={"step": "registering"})
        logger.info("Step 4: Registering model version")

        # Auto-generate version if not specified
        existing_count = session.query(ModelVersion).count()
        custom_name = tc.get("model_name")
        if custom_name:
            import re
            custom_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', custom_name)
            base_custom_name = custom_name
            suffix = 1
            while session.query(ModelVersion).filter(ModelVersion.version == custom_name).first() is not None:
                custom_name = f"{base_custom_name}_{suffix}"
                suffix += 1
            version_name = custom_name
        else:
            version_name = f"v{existing_count + 1}"
        
        # Deactivate all existing versions first so the new one becomes the sole active version
        session.query(ModelVersion).update({ModelVersion.is_active: False})
        
        model_version = ModelVersion(
            version=version_name,
            training_run_id=uuid.UUID(run_id),
            base_model=run.base_model,
            model_path=model_path,
            quantization_format=quantization_format if tc.get("quantize", True) else None,
            file_size=get_model_size(model_path),
            metrics=metrics,
            dataset_version=dataset.version,
            is_active=True,
        )
        session.add(model_version)

        # Mark training run as completed
        run.status = TrainingStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        session.commit()

        logger.info(f"Training pipeline complete for run {run_id}")
        return {
            "run_id": run_id,
            "model_version": model_version.version,
            "model_path": model_path,
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"Training pipeline failed for run {run_id}: {e}", exc_info=True)
        try:
            run = session.query(TrainingRun).filter(
                TrainingRun.id == uuid.UUID(run_id)
            ).first()
            if run:
                run.status = TrainingStatus.FAILED
                run.error_message = str(e)[:2000]
                run.completed_at = datetime.now(timezone.utc)
                session.commit()
        except Exception:
            session.rollback()
        raise

    finally:
        session.close()
