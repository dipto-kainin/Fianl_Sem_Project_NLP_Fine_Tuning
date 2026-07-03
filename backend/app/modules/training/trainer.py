"""
Training Module - LoRA Fine-Tuning Pipeline.

Implements the student model training using PEFT LoRA adapters with
QLoRA (4-bit quantization) for memory efficiency.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def run_lora_training(
    dataset_path: str,
    output_dir: str,
    base_model: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    use_context: bool = True,
    system_prompt: str | None = None,
) -> dict:
    """
    Run LoRA fine-tuning on the base model using the provided dataset.

    Steps:
    1. Load base model with 4-bit quantization (QLoRA)
    2. Configure LoRA adapters
    3. Load and format the training dataset
    4. Train with SFTTrainer from TRL
    5. Save LoRA adapters

    Args:
        dataset_path: Path to the JSONL training dataset.
        output_dir: Directory to save trained adapters.
        base_model: HuggingFace model ID.
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA alpha scaling.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        learning_rate: Learning rate.

    Returns:
        Dict with training metrics.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer, SFTConfig

    # Set environment variables for memory efficiency
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    logger.info(f"Starting LoRA training: model={base_model}, dataset={dataset_path}")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    # Enable gradient checkpointing for lower memory footprint
    model.gradient_checkpointing_enable()

    # Step 3: Configure LoRA
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Step 4: Load dataset
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Replicate samples by priority weight so high-priority facts get more gradient updates
    # priority 3 = 3 copies, priority 2 = 2 copies, priority 1 = 1 copy
    import random as _random
    weighted_rows = []
    for sample in raw_dataset:
        weight = max(1, min(3, int(sample.get("priority", 1))))
        weighted_rows.extend([sample] * weight)
    _random.shuffle(weighted_rows)

    from datasets import Dataset as HFDataset
    dataset = HFDataset.from_list(weighted_rows)
    logger.info(
        f"Weighted dataset: {len(raw_dataset)} raw samples → {len(dataset)} weighted "
        f"(priority 1×/2×/3× replication applied)"
    )

    def format_instruction(sample):
        """Format sample into instruction-following template."""
        if system_prompt:
            # Use the user-supplied (amplified) HR/persona system prompt verbatim
            sp = system_prompt
        elif use_context and sample.get("context"):
            sp = "You are a helpful QA assistant. Provide a direct, concise answer based on the context in at most 3 lines. Do not write letters, emails, sign-offs, or signature templates."
        else:
            sp = "You are a helpful assistant. Provide a direct, concise answer in at most 3 lines. Do not write letters, emails, sign-offs, or signature templates."

        if use_context and sample.get("context"):
            user_prompt = f"{sample['instruction']}\nContext: {sample['context']}"
        else:
            user_prompt = sample['instruction']

        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": sample['response']}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Step 5: Train with SFTTrainer
    is_memorization = not use_context
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1 if is_memorization else (4 if batch_size == 1 else 2 if batch_size == 2 else 1),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=learning_rate,
        weight_decay=0.0 if is_memorization else 0.01,
        warmup_ratio=0.0 if is_memorization else 0.03,
        lr_scheduler_type="constant" if is_memorization else "cosine",
        logging_steps=5 if is_memorization else 10,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_length=1024,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_instruction,
        processing_class=tokenizer,
        args=sft_config,
    )

    # Log trainable parameters info
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Run training
    train_result = trainer.train()

    # Save the LoRA adapters
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Collect metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "total_steps": train_result.global_step,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percent": round(100 * trainable_params / total_params, 2),
    }

    # Save metrics
    with open(Path(output_dir) / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training complete. Loss: {metrics['train_loss']:.4f}")
    return metrics


def merge_lora_adapters(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
) -> str:
    """
    Merge LoRA adapters into the base model.

    Args:
        base_model: HuggingFace model ID.
        adapter_dir: Directory containing LoRA adapters.
        output_dir: Directory to save merged model.

    Returns:
        Path to the merged model directory.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Merging LoRA adapters from {adapter_dir}")

    # Load base model (float16 on CPU to avoid offloading/meta device corruption during merge)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load and merge adapters
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_dir = Path(output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    logger.info(f"Merged model saved to {merged_dir}")
    return str(merged_dir)
