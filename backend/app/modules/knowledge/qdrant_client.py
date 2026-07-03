"""
Knowledge Module - Qdrant Vector Database Client.

Manages the vector database for storing and searching document knowledge.
"""

import logging
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Get cached Qdrant client instance."""
    client = QdrantClient(url=settings.QDRANT_URL)
    logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")
    return client


def ensure_collection() -> None:
    """Create the knowledge base collection if it doesn't exist."""
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        # Create payload indices for fast filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="tags",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Created Qdrant collection: {collection_name}")
    else:
        logger.info(f"Qdrant collection already exists: {collection_name}")


def upsert_knowledge(
    point_id: str,
    vector: list[float],
    chunk_id: str,
    document_id: str,
    text: str,
    summary: str = "",
    tags: list[str] | None = None,
    entities: dict | None = None,
    section_title: str = "",
    metadata: dict | None = None,
) -> None:
    """
    Upsert a knowledge point into Qdrant.

    Args:
        point_id: Unique point identifier.
        vector: Embedding vector.
        chunk_id: Source chunk UUID (as string).
        document_id: Source document UUID (as string).
        text: Chunk text content.
        summary: Teacher-generated summary.
        tags: Topic tags.
        entities: Extracted entities.
        section_title: Section heading.
        metadata: Additional metadata.
    """
    client = get_qdrant_client()

    payload = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "summary": summary,
        "section_title": section_title,
        "tags": tags or [],
        "entities": entities or {},
        "metadata": metadata or {},
    }

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )


def upsert_knowledge_batch(
    points: list[dict],
) -> None:
    """
    Batch upsert knowledge points into Qdrant.

    Args:
        points: List of dicts with keys: point_id, vector, and payload fields.
    """
    client = get_qdrant_client()

    point_structs = [
        PointStruct(
            id=p["point_id"],
            vector=p["vector"],
            payload={
                "chunk_id": p.get("chunk_id", ""),
                "document_id": p.get("document_id", ""),
                "text": p.get("text", ""),
                "summary": p.get("summary", ""),
                "section_title": p.get("section_title", ""),
                "tags": p.get("tags", []),
                "entities": p.get("entities", {}),
                "metadata": p.get("metadata", {}),
            },
        )
        for p in points
    ]

    # Qdrant supports batch upsert up to ~1000 points
    batch_size = 100
    for i in range(0, len(point_structs), batch_size):
        batch = point_structs[i : i + batch_size]
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=batch,
        )
    logger.info(f"Upserted {len(point_structs)} points to Qdrant")


def search_similar(
    query_vector: list[float],
    top_k: int = 5,
    document_id: str | None = None,
    tags_filter: list[str] | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Search for similar vectors in the knowledge base.

    Args:
        query_vector: Query embedding vector.
        top_k: Number of results to return.
        document_id: Optional filter by document.
        tags_filter: Optional filter by tags.
        score_threshold: Optional minimum similarity score.

    Returns:
        List of search results with payload and score.
    """
    client = get_qdrant_client()

    # Build filters
    conditions = []
    if document_id:
        conditions.append(
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id),
            )
        )
    if tags_filter:
        for tag in tags_filter:
            conditions.append(
                FieldCondition(
                    key="tags",
                    match=MatchValue(value=tag),
                )
            )

    query_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    return [
        {
            "id": str(point.id),
            "score": point.score,
            "chunk_id": point.payload.get("chunk_id", ""),
            "document_id": point.payload.get("document_id", ""),
            "text": point.payload.get("text", ""),
            "summary": point.payload.get("summary", ""),
            "section_title": point.payload.get("section_title", ""),
            "tags": point.payload.get("tags", []),
        }
        for point in results.points
    ]


def delete_by_document(document_id: str) -> None:
    """Delete all vectors belonging to a document."""
    client = get_qdrant_client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )
    logger.info(f"Deleted Qdrant points for document: {document_id}")


def get_collection_info() -> dict:
    """Get information about the knowledge base collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(settings.QDRANT_COLLECTION)
        return {
            "collection_name": settings.QDRANT_COLLECTION,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value if info.status else "unknown",
            "vector_size": settings.EMBEDDING_DIMENSION,
        }
    except Exception as e:
        return {
            "collection_name": settings.QDRANT_COLLECTION,
            "status": "unavailable",
            "error": str(e),
        }
