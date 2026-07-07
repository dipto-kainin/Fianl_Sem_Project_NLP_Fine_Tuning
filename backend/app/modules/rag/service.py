"""
RAG Module - Retrieval-Augmented Generation Service.

Implements the RAG pipeline: embed query → vector search → generate answer.
"""

import logging

from app.modules.knowledge.qdrant_client import search_similar
from app.modules.teacher.gemini_client import generate_rag_answer
from app.utils.embeddings import generate_embedding

logger = logging.getLogger(__name__)


from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.student_inference import generate_student_answer

async def rag_query(
    db: AsyncSession,
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    tags_filter: list[str] | None = None,
    use_rag: bool = True,
    model: str = "teacher",
) -> dict:
    """
    Execute a full RAG pipeline.

    1. Embed the query using sentence-transformers (if RAG enabled)
    2. Search Qdrant for similar chunks (if RAG enabled)
    3. Pass retrieved context to Gemini (Teacher) or Student (Default/Fine-tuned) models for answer generation
    4. Return answers with source citations

    Args:
        db: Database session.
        query: User's question.
        top_k: Number of chunks to retrieve.
        document_id: Optional document filter.
        tags_filter: Optional tag filters.
        use_rag: Whether to run semantic retrieval.
        model: Model selection: 'teacher', 'default', 'student', or 'compare'.

    Returns:
        Dict with answers, sources, and metadata.
    """
    logger.info(f"RAG query: '{query[:100]}...' (model={model}, use_rag={use_rag}, top_k={top_k})")

    context_chunks = []
    chunk_summaries = []
    sources = []

    if use_rag:
        # Step 1: Embed the query
        query_vector = generate_embedding(query)

        # Step 2: Search for similar chunks
        search_results = search_similar(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
            tags_filter=tags_filter,
        )

        if search_results:
            # Step 3: Prepare context for answer generation
            context_chunks = [r["text"] for r in search_results]
            chunk_summaries = [r.get("summary", "") for r in search_results]

            # Build source citations
            sources = [
                {
                    "chunk_id": r["chunk_id"],
                    "document_id": r["document_id"],
                    "text": r["text"][:500],  # Truncate for response
                    "summary": r.get("summary", ""),
                    "section_title": r.get("section_title", ""),
                    "similarity_score": r["score"],
                }
                for r in search_results
            ]

    # Step 4: Generate answers based on model selection
    answer = None
    default_answer = None
    student_answer = None
    student_version = None
    tokens_used = 0

    # A. Gemini (Teacher)
    if model in ("teacher", "compare"):
        try:
            answer, tokens_used = generate_rag_answer(
                query=query,
                context_chunks=context_chunks,
                chunk_summaries=chunk_summaries,
            )
        except Exception as e:
            logger.error(f"Teacher (Gemini) model generation failed: {e}")
            answer = f"Teacher (Gemini) generation failed (offline or network error): {str(e)}"
            tokens_used = 0

    # B. Default Student Model (TinyLlama Base)
    if model in ("default", "compare"):
        try:
            default_answer, _ = await generate_student_answer(
                db, query, context_chunks, force_base=True
            )
        except Exception as e:
            logger.error(f"Default model generation failed: {e}")
            default_answer = f"Default model generation failed: {str(e)}"

    # C. Fine-tuned Student Model
    if model in ("student", "compare"):
        try:
            student_answer, student_version = await generate_student_answer(
                db, query, context_chunks, force_base=False
            )
            # If no fine-tuned model exists, indicate it
            if student_version == "Base Model (Default)":
                student_version = "No fine-tuned model active yet (using default base)"
        except Exception as e:
            logger.error(f"Fine-tuned student model generation failed: {e}")
            student_answer = f"Fine-tuned student generation failed: {str(e)}"

    logger.info(f"RAG answer generated. Sources retrieved: {len(sources)}")

    return {
        "answer": answer,
        "default_answer": default_answer,
        "student_answer": student_answer,
        "student_version": student_version,
        "sources": sources,
        "tokens_used": tokens_used,
        "query": query,
        "use_rag": use_rag,
    }


def semantic_search(
    query: str,
    top_k: int = 10,
    document_id: str | None = None,
    tags_filter: list[str] | None = None,
) -> dict:
    """
    Semantic search without answer generation.

    Args:
        query: Search query.
        top_k: Number of results.
        document_id: Optional document filter.
        tags_filter: Optional tag filters.

    Returns:
        Dict with search results.
    """
    query_vector = generate_embedding(query)

    search_results = search_similar(
        query_vector=query_vector,
        top_k=top_k,
        document_id=document_id,
        tags_filter=tags_filter,
    )

    results = [
        {
            "chunk_id": r["chunk_id"],
            "document_id": r["document_id"],
            "text": r["text"],
            "summary": r.get("summary", ""),
            "section_title": r.get("section_title", ""),
            "similarity_score": r["score"],
        }
        for r in search_results
    ]

    return {"results": results, "query": query}
