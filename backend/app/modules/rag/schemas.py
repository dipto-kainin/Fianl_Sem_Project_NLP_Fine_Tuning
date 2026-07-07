"""
RAG Module - Pydantic Schemas.
"""

import uuid
from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Request to ask a question via RAG."""
    query: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    document_id: uuid.UUID | None = Field(
        None, description="Optional: restrict search to a specific document"
    )
    tags_filter: list[str] | None = Field(
        None, description="Optional: filter by topic tags"
    )
    use_rag: bool = Field(
        default=True, description="Whether to use RAG retrieval. If False, the query goes directly to the model without context."
    )
    model: str = Field(
        default="teacher", description="Model selection: 'teacher', 'default', 'student', or 'compare'"
    )


class RAGSourceChunk(BaseModel):
    """A source chunk used in the RAG answer."""
    chunk_id: str
    document_id: str
    text: str
    summary: str = ""
    section_title: str = ""
    similarity_score: float


class RAGQueryResponse(BaseModel):
    """Response from the RAG query endpoint."""
    answer: str | None = None
    default_answer: str | None = None
    student_answer: str | None = None
    student_version: str | None = None
    sources: list[RAGSourceChunk]
    tokens_used: int = 0
    query: str
    use_rag: bool = True
    used_rag: bool = True


class RAGSearchRequest(BaseModel):
    """Request for semantic search without answer generation."""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    document_id: uuid.UUID | None = None
    tags_filter: list[str] | None = None


class RAGSearchResponse(BaseModel):
    """Response from semantic search."""
    results: list[RAGSourceChunk]
    query: str
