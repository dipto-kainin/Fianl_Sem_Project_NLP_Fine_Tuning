"""
Knowledge Module - Pydantic Schemas.
"""

from pydantic import BaseModel


class KnowledgeSearchResult(BaseModel):
    """A single search result from the knowledge base."""
    id: str
    score: float
    chunk_id: str
    document_id: str
    text: str
    summary: str = ""
    section_title: str = ""
    tags: list[str] = []


class KnowledgeBaseInfo(BaseModel):
    """Information about the knowledge base collection."""
    collection_name: str
    vectors_count: int | None = None
    points_count: int | None = None
    status: str
    vector_size: int | None = None
    error: str | None = None
