"""
RAG Module - API Router.

Endpoints for Retrieval-Augmented Generation and semantic search.
"""

from fastapi import APIRouter

from app.modules.rag import service
from app.modules.rag.schemas import (
    RAGQueryRequest,
    RAGQueryResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGSourceChunk,
)

router = APIRouter(prefix="/rag", tags=["RAG"])


from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies import get_db

@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question and get an AI-generated answer based on your documents.

    Pipeline: Embed query → Vector search → Context assembly → Gemini answer generation.
    Can also query the active fine-tuned Student model side-by-side if available.
    Returns both the Teacher and Student answers along with source citations.
    """
    result = await service.rag_query(
        db=db,
        query=request.query,
        top_k=request.top_k,
        document_id=str(request.document_id) if request.document_id else None,
        tags_filter=request.tags_filter,
        use_rag=request.use_rag,
        model=request.model,
    )
    return RAGQueryResponse(
        answer=result["answer"],
        default_answer=result.get("default_answer"),
        student_answer=result["student_answer"],
        student_version=result["student_version"],
        sources=[RAGSourceChunk(**s) for s in result["sources"]],
        tokens_used=result["tokens_used"],
        query=result["query"],
        use_rag=result["use_rag"],
    )


@router.post("/reload-model")
async def reload_student_model():
    """
    Clear the cached student model so the next query loads
    the latest trained weights fresh from disk.
    """
    from app.utils.student_inference import clear_student_model_cache
    clear_student_model_cache()
    return {"message": "Student model cache cleared. Next query will load fresh weights."}


@router.post("/search", response_model=RAGSearchResponse)
async def semantic_search(request: RAGSearchRequest):
    """
    Semantic search across the knowledge base.

    Returns similar chunks ranked by relevance without generating an AI answer.
    Useful for browsing and exploring document content.
    """
    result = service.semantic_search(
        query=request.query,
        top_k=request.top_k,
        document_id=str(request.document_id) if request.document_id else None,
        tags_filter=request.tags_filter,
    )
    return RAGSearchResponse(
        results=[RAGSourceChunk(**r) for r in result["results"]],
        query=result["query"],
    )
