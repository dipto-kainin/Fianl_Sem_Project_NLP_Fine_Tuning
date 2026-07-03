"""
AI Knowledge Distillation Platform - FastAPI Application Entry Point.

Sets up the FastAPI application, includes all routers, configures CORS,
and manages the application lifecycle.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db, close_db

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("🚀 Starting AI Knowledge Distillation Platform")

    # Create storage directories
    settings.ensure_directories()

    # Initialize database tables (development only)
    if settings.DEBUG:
        await init_db()
        logger.info("📦 Database tables created")

    # Initialize Qdrant collection
    try:
        from app.modules.knowledge.qdrant_client import ensure_collection
        ensure_collection()
        logger.info("🔍 Qdrant collection initialized")
    except Exception as e:
        logger.warning(f"⚠️  Qdrant not available: {e}")

    yield

    # Shutdown
    await close_db()
    logger.info("👋 Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Backend API for the AI Knowledge Distillation Platform. "
        "Upload documents, extract structured knowledge using a Teacher LLM (Gemini), "
        "power RAG-based Q&A, generate training datasets, and fine-tune "
        "lightweight Student models for on-device inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
from app.modules.documents.router import router as documents_router
from app.modules.chunks.router import router as chunks_router
from app.modules.teacher.router import router as teacher_router
from app.modules.knowledge.router import router as knowledge_router
from app.modules.rag.router import router as rag_router
from app.modules.datasets.router import router as datasets_router
from app.modules.training.router import router as training_router
from app.modules.registry.router import router as registry_router

app.include_router(documents_router, prefix=settings.API_V1_PREFIX)
app.include_router(chunks_router, prefix=settings.API_V1_PREFIX)
app.include_router(teacher_router, prefix=settings.API_V1_PREFIX)
app.include_router(knowledge_router, prefix=settings.API_V1_PREFIX)
app.include_router(rag_router, prefix=settings.API_V1_PREFIX)
app.include_router(datasets_router, prefix=settings.API_V1_PREFIX)
app.include_router(training_router, prefix=settings.API_V1_PREFIX)
app.include_router(registry_router, prefix=settings.API_V1_PREFIX)


# --- Health Check ---
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API overview."""
    return {
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "documents": f"{settings.API_V1_PREFIX}/documents",
            "chunks": f"{settings.API_V1_PREFIX}/chunks",
            "teacher": f"{settings.API_V1_PREFIX}/teacher",
            "knowledge": f"{settings.API_V1_PREFIX}/knowledge",
            "rag": f"{settings.API_V1_PREFIX}/rag",
            "datasets": f"{settings.API_V1_PREFIX}/datasets",
            "training": f"{settings.API_V1_PREFIX}/training",
            "models": f"{settings.API_V1_PREFIX}/models",
        },
    }
