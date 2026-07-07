"""
AI Knowledge Distillation Platform - Student Model Inference.

Loads the active Student LLM from the model registry and generates answers
using retrieved RAG context. Caches the model in memory.
"""

import logging
from pathlib import Path
from functools import lru_cache

import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.modules.registry.models import ModelVersion
from app.modules.training.models import TrainingRun

logger = logging.getLogger(__name__)
settings = get_settings()

# Global cache for loaded models and tokenizers: {model_id: (model, tokenizer, version_label, system_prompt)}
_model_cache = {}


def clear_student_model_cache():
    """Clear all cached student models to force reload and reclaim VRAM."""
    global _model_cache
    _model_cache.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleared Student model cache and reclaimed VRAM.")


async def get_student_model_and_tokenizer(db: AsyncSession, force_base: bool = False):
    """
    Load and cache student model and tokenizer.
    Loads from the merged HuggingFace model directory, or raw base model if force_base is True.
    """
    global _model_cache

    active_version = None
    system_prompt = None
    if not force_base:
        # Find the active model in the registry, fall back to the latest registered model
        result = await db.execute(
            select(ModelVersion).where(ModelVersion.is_active == True)
        )
        active_version = result.scalar_one_or_none()

        if not active_version:
            # Fallback: Get the latest trained model version
            fallback_result = await db.execute(
                select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(1)
            )
            active_version = fallback_result.scalar_one_or_none()

        if active_version and active_version.training_run_id:
            # Fetch custom system prompt used during training
            run_result = await db.execute(
                select(TrainingRun).where(TrainingRun.id == active_version.training_run_id)
            )
            run = run_result.scalar_one_or_none()
            if run and run.training_config:
                system_prompt = run.training_config.get("system_prompt")

    # Determine load details
    if active_version and not force_base:
        model_id_to_load = str(active_version.id)
        version_label = active_version.version
    else:
        model_id_to_load = "default_base_model"
        version_label = "Base Model (Default)"

    # Check cache
    if model_id_to_load in _model_cache:
        model, tokenizer, label, cached_prompt = _model_cache[model_id_to_load]
        return model, tokenizer, label, cached_prompt

    # Clear memory cache if another model is loaded to avoid GPU OOM (keep at most 1)
    if len(_model_cache) >= 1:
        clear_student_model_cache()

    if active_version and not force_base:
        # Determine paths
        model_dir = Path(active_version.model_path)
        hf_dir = None

        if model_dir.is_dir() and (model_dir / "config.json").exists():
            hf_dir = model_dir
        else:
            parent_dir = model_dir.parent
            if (parent_dir / "merged" / "config.json").exists():
                hf_dir = parent_dir / "merged"
            elif active_version.training_run_id:
                # Look up training run output_dir
                run_result = await db.execute(
                    select(TrainingRun).where(TrainingRun.id == active_version.training_run_id)
                )
                run = run_result.scalar_one_or_none()
                if run and run.output_dir:
                    candidate = Path(run.output_dir) / "merged"
                    if (candidate / "config.json").exists():
                        hf_dir = candidate

        if not hf_dir:
            logger.warning(
                f"Could not find HuggingFace model directory for version {active_version.version}. "
                f"Path checked: {active_version.model_path}"
            )
            return None, None, None, None
        path_to_load = str(hf_dir)
    else:
        path_to_load = settings.BASE_STUDENT_MODEL

    logger.info(f"Loading Student model '{version_label}' from {path_to_load}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(path_to_load, trust_remote_code=True, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(path_to_load, trust_remote_code=True)
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model in 8-bit quantization to save RAM if cuda is available, else cpu
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        use_8bit = torch.cuda.is_available()
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path_to_load,
                device_map=device_map,
                load_in_8bit=use_8bit,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload if GPU OOM/dispatch warning
                local_files_only=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                path_to_load,
                device_map=device_map,
                load_in_8bit=use_8bit,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload if GPU OOM/dispatch warning
            )

        _model_cache[model_id_to_load] = (model, tokenizer, version_label, system_prompt)
        logger.info(f"Student model version '{version_label}' successfully loaded.")

        return model, tokenizer, version_label, system_prompt

    except Exception as e:
        logger.error(f"Failed to load Student model: {e}", exc_info=True)
        return None, None, None, None


async def generate_student_answer(
    db: AsyncSession,
    query: str,
    context_chunks: list[str],
    force_base: bool = False,
) -> tuple[str | None, str | None]:
    """
    Generate an answer using the active Student model.

    Args:
        db: Database session.
        query: User query.
        context_chunks: Retrieved context text.
        force_base: If True, load the default model instead of active fine-tuned model.

    Returns:
        Tuple of (answer_text, version_string) or (None, None) if loading fails.
    """
    model, tokenizer, version, system_prompt = await get_student_model_and_tokenizer(db, force_base=force_base)
    if not model or not tokenizer:
        return None, None

    # Format prompt using the tokenizer's chat template
    context_str = "\n".join(context_chunks)
    if system_prompt:
        system_content = system_prompt
    elif context_str:
        system_content = "You are a helpful QA assistant. Provide a direct, concise answer based on the context in at most 3 lines. Do not write letters, emails, sign-offs, or signature templates."
    else:
        system_content = "You are a helpful assistant. Provide a direct, concise answer in at most 3 lines. Do not write letters, emails, sign-offs, or signature templates."

    if context_str:
        user_content = f"{query}\nContext: {context_str}"
    else:
        user_content = query

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    logger.info(f"Generating student answer using model version: {version}")

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        # Decode only the generated response
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return answer, version

    except Exception as e:
        logger.error(f"Error during student inference: {e}", exc_info=True)
        return f"Error during student model generation: {str(e)}", version
