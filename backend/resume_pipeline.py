import sys
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.workers.training_tasks import _get_sync_session
from app.modules.training.models import TrainingRun, TrainingStatus
from app.modules.training.trainer import merge_lora_adapters
from app.modules.training.quantizer import quantize_to_gguf, get_model_size
from app.modules.registry.models import ModelVersion
from app.modules.datasets.models import Dataset
from app.config import get_settings

def resume_pipeline(run_id_str: str):
    run_id = uuid.UUID(run_id_str)
    settings = get_settings()
    session = _get_sync_session()

    try:
        # Load training run
        run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if not run:
            print(f"Error: Training run not found: {run_id}")
            return

        print(f"Resuming pipeline for run: {run.id}")
        print(f"Base model: {run.base_model}")
        print(f"Status: {run.status}")

        dataset = session.query(Dataset).filter(Dataset.id == run.dataset_id).first()
        if not dataset:
            print("Error: Dataset not found")
            return

        output_dir = str(Path(settings.MODELS_DIR) / f"run_{run_id}")
        print(f"Output directory: {output_dir}")

        tc = run.training_config or {}
        
        # Step 2: Merge adapters
        run.status = TrainingStatus.MERGING
        session.commit()
        print("\n=== Step 2: Merging LoRA adapters on GPU/CPU ===")
        merged_dir = merge_lora_adapters(
            base_model=run.base_model,
            adapter_dir=output_dir,
            output_dir=output_dir,
        )
        print(f"Merge complete. Merged model directory: {merged_dir}")

        # Step 3: Quantize (if enabled)
        model_path = merged_dir
        quantization_format = tc.get("quantization_format", "q4_k_m")

        if tc.get("quantize", True):
            run.status = TrainingStatus.QUANTIZING
            session.commit()
            print(f"\n=== Step 3: Quantizing to {quantization_format} ===")
            model_path = quantize_to_gguf(
                model_dir=merged_dir,
                output_dir=output_dir,
                quantization_type=quantization_format,
            )
            print(f"Quantization complete. Quantized path: {model_path}")

        # Step 4: Register model version
        print("\n=== Step 4: Registering model version ===")
        
        # Deactivate all existing versions
        session.query(ModelVersion).update({ModelVersion.is_active: False})

        existing_count = session.query(ModelVersion).count()
        metrics = run.metrics or {}
        
        model_version = ModelVersion(
            version=f"v{existing_count + 1}",
            training_run_id=run_id,
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

        print(f"\nSuccess! Model registered as version {model_version.version} and set as active.")

    except Exception as e:
        session.rollback()
        print(f"\nError during merge/quantization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resume_pipeline.py <run_id>")
        sys.exit(1)
    resume_pipeline(sys.argv[1])
