"""
AI Knowledge Distillation Platform - File Storage Utilities.

Handles file upload, storage, and retrieval on local filesystem.
"""

import os
import uuid
import aiofiles
from pathlib import Path
from datetime import datetime

from fastapi import UploadFile

from app.config import get_settings

settings = get_settings()


async def save_upload(file: UploadFile, sub_dir: str = "") -> tuple[str, int]:
    """
    Save an uploaded file to the storage directory.

    Args:
        file: The uploaded file from FastAPI.
        sub_dir: Optional subdirectory within UPLOAD_DIR.

    Returns:
        Tuple of (file_path, file_size_bytes).
    """
    # Generate a unique filename to prevent collisions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    original_ext = Path(file.filename).suffix.lower() if file.filename else ""
    safe_name = f"{timestamp}_{unique_id}{original_ext}"

    # Build target directory
    target_dir = Path(settings.UPLOAD_DIR)
    if sub_dir:
        target_dir = target_dir / sub_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / safe_name

    # Stream write to disk
    file_size = 0
    async with aiofiles.open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            await f.write(chunk)
            file_size += len(chunk)

    return str(file_path), file_size


async def delete_file(file_path: str) -> bool:
    """
    Delete a file from storage.

    Args:
        file_path: Path to the file.

    Returns:
        True if deleted, False if file didn't exist.
    """
    path = Path(file_path)
    if path.exists():
        path.unlink()
        return True
    return False


async def read_file_bytes(file_path: str) -> bytes:
    """
    Read file contents as bytes.

    Args:
        file_path: Path to the file.

    Returns:
        File contents as bytes.
    """
    async with aiofiles.open(file_path, "rb") as f:
        return await f.read()


def get_file_extension(filename: str) -> str:
    """Get the lowercase file extension without the dot."""
    return Path(filename).suffix.lower().lstrip(".")


def ensure_directory(dir_path: str) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path
