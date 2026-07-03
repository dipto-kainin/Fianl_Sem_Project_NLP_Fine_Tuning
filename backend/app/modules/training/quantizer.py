"""
Training Module - Model Quantization.

Converts trained models to quantized formats (GGUF) for on-device deployment.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def quantize_to_gguf(
    model_dir: str,
    output_dir: str,
    quantization_type: str = "q4_k_m",
) -> str:
    """
    Quantize a model to GGUF format for on-device inference.

    This uses llama.cpp's convert and quantize utilities.
    Falls back to a Python-based approach if llama.cpp is not available.

    Args:
        model_dir: Path to the merged HuggingFace model.
        output_dir: Directory to save the quantized model.
        quantization_type: Quantization type (q4_k_m, q8_0, etc.).

    Returns:
        Path to the quantized GGUF file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gguf_filename = f"model-{quantization_type}.gguf"
    gguf_path = output_path / gguf_filename

    try:
        # Try using llama.cpp's conversion script
        # Step 1: Convert HF model to GGUF f16
        f16_path = output_path / "model-f16.gguf"

        convert_cmd = [
            "python", "-m", "llama_cpp.convert",
            "--outfile", str(f16_path),
            "--outtype", "f16",
            str(model_dir),
        ]
        logger.info(f"Converting to GGUF: {' '.join(convert_cmd)}")

        result = subprocess.run(
            convert_cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed: {result.stderr}")

        # Step 2: Quantize
        quantize_cmd = [
            "llama-quantize",
            str(f16_path),
            str(gguf_path),
            quantization_type.upper(),
        ]
        logger.info(f"Quantizing: {' '.join(quantize_cmd)}")

        result = subprocess.run(
            quantize_cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Quantization failed: {result.stderr}")

        # Clean up f16 intermediate file
        if f16_path.exists():
            f16_path.unlink()

    except (FileNotFoundError, RuntimeError) as e:
        logger.warning(f"llama.cpp quantization failed ({e}), trying alternative approach...")
        gguf_path = _quantize_with_transformers(model_dir, output_path, quantization_type)

    file_size = gguf_path.stat().st_size if gguf_path.exists() else 0
    logger.info(f"Quantized model saved: {gguf_path} ({file_size / 1024 / 1024:.1f} MB)")

    return str(gguf_path)


def _quantize_with_transformers(
    model_dir: str,
    output_path: Path,
    quantization_type: str,
) -> Path:
    """
    Fallback quantization using transformers + bitsandbytes.

    Saves the model with GPTQ or bitsandbytes quantization.
    This produces a HuggingFace-compatible quantized model rather than GGUF.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Fallback quantization using bitsandbytes ({quantization_type})")

    # Determine quantization level
    if "4" in quantization_type:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Save quantized model
    quant_dir = output_path / f"quantized-{quantization_type}"
    quant_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(quant_dir))
    tokenizer.save_pretrained(str(quant_dir))

    logger.info(f"Fallback quantized model saved to {quant_dir}")
    return quant_dir


def get_model_size(path: str) -> int:
    """Get total size of model files in bytes."""
    model_path = Path(path)
    if model_path.is_file():
        return model_path.stat().st_size
    elif model_path.is_dir():
        return sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    return 0
