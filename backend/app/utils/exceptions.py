"""
AI Knowledge Distillation Platform - Custom Exceptions.
"""

from fastapi import HTTPException, status


class DocumentNotFoundError(HTTPException):
    """Raised when a document is not found."""

    def __init__(self, document_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )


class ChunkNotFoundError(HTTPException):
    """Raised when a chunk is not found."""

    def __init__(self, chunk_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk '{chunk_id}' not found.",
        )


class DatasetNotFoundError(HTTPException):
    """Raised when a dataset is not found."""

    def __init__(self, dataset_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )


class TrainingRunNotFoundError(HTTPException):
    """Raised when a training run is not found."""

    def __init__(self, run_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run '{run_id}' not found.",
        )


class ModelVersionNotFoundError(HTTPException):
    """Raised when a model version is not found."""

    def __init__(self, model_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{model_id}' not found.",
        )


class UnsupportedFileTypeError(HTTPException):
    """Raised when an unsupported file type is uploaded."""

    def __init__(self, file_type: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: '{file_type}'. Supported: PDF, DOCX, TXT, MD, HTML, EPUB.",
        )


class FileTooLargeError(HTTPException):
    """Raised when uploaded file exceeds the maximum size."""

    def __init__(self, size_mb: float, max_mb: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb:.1f} MB) exceeds maximum ({max_mb} MB).",
        )


class DocumentProcessingError(HTTPException):
    """Raised when document processing fails."""

    def __init__(self, document_id: str, reason: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document '{document_id}': {reason}",
        )


class TeacherAPIError(HTTPException):
    """Raised when the Teacher LLM API call fails."""

    def __init__(self, reason: str):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Teacher LLM API error: {reason}",
        )


class TrainingError(HTTPException):
    """Raised when model training fails."""

    def __init__(self, run_id: str, reason: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training run '{run_id}' failed: {reason}",
        )


class QdrantConnectionError(HTTPException):
    """Raised when Qdrant is unreachable."""

    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database (Qdrant) is unavailable.",
        )
