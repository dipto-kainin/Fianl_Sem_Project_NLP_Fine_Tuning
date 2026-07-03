"""
Knowledge Module - API Router.

Endpoints for querying the knowledge base and viewing stats.
"""

from fastapi import APIRouter

from app.modules.knowledge import qdrant_client
from app.modules.knowledge.schemas import KnowledgeBaseInfo

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


@router.get("/info", response_model=KnowledgeBaseInfo)
async def get_knowledge_base_info():
    """Get information about the knowledge base (vector count, status, etc.)."""
    info = qdrant_client.get_collection_info()
    return KnowledgeBaseInfo(**info)


@router.post("/initialize")
async def initialize_knowledge_base():
    """Initialize the knowledge base collection in Qdrant."""
    qdrant_client.ensure_collection()
    return {"message": "Knowledge base collection initialized."}
