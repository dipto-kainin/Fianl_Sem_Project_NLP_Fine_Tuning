"""
Teacher LLM Processing Tasks.

Celery tasks for processing chunks with the Teacher LLM independently
of the document processing pipeline.
"""

import logging
import uuid

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _get_sync_session():
    """Create a synchronous database session for Celery workers."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.config import get_settings

    settings = get_settings()
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2").replace("postgresql+psycopg2", "postgresql")
    engine = create_engine(sync_url)
    Session = sessionmaker(bind=engine)
    return Session()


@celery_app.task(name="app.workers.teacher_tasks.process_chunks_with_teacher", bind=True)
def process_chunks_with_teacher(
    self,
    chunk_ids: list[str] | None = None,
    document_id: str | None = None,
):
    """
    Process specified chunks with the Teacher LLM.

    Can be triggered for:
    - Specific chunk IDs
    - All chunks of a document
    - All unprocessed chunks (if both are None)
    """
    from sqlalchemy import select
    from app.modules.chunks.models import Chunk
    from app.modules.teacher.models import TeacherOutput
    from app.modules.teacher.gemini_client import process_chunk_with_teacher
    from app.modules.knowledge.qdrant_client import get_qdrant_client
    from app.config import get_settings

    settings = get_settings()
    session = _get_sync_session()

    try:
        # Determine which chunks to process
        if chunk_ids:
            chunks = session.query(Chunk).filter(
                Chunk.id.in_([uuid.UUID(cid) for cid in chunk_ids])
            ).all()
        elif document_id:
            chunks = session.query(Chunk).filter(
                Chunk.document_id == uuid.UUID(document_id)
            ).all()
        else:
            # Get all chunks without teacher output
            processed_ids = session.query(TeacherOutput.chunk_id).subquery()
            chunks = session.query(Chunk).filter(
                ~Chunk.id.in_(select(processed_ids))
            ).all()

        total = len(chunks)
        logger.info(f"Processing {total} chunks with Teacher LLM")
        processed = 0
        failed = 0

        for i, chunk in enumerate(chunks):
            try:
                # Skip if already processed
                existing = session.query(TeacherOutput).filter(
                    TeacherOutput.chunk_id == chunk.id
                ).first()
                if existing:
                    logger.info(f"Chunk {chunk.id} already processed, skipping")
                    continue

                self.update_state(
                    state="PROCESSING",
                    meta={"current": i + 1, "total": total}
                )

                teacher_output, tokens_used = process_chunk_with_teacher(
                    chunk_text=chunk.text,
                    section_title=chunk.section_title or "",
                )

                db_output = TeacherOutput(
                    chunk_id=chunk.id,
                    summary=teacher_output.summary,
                    entities={"entities": [e.model_dump() for e in teacher_output.entities]},
                    relationships={"relationships": [r.model_dump() for r in teacher_output.relationships]},
                    qa_pairs=[qa.model_dump() for qa in teacher_output.qa_pairs],
                    explanations=teacher_output.explanation,
                    faqs=teacher_output.faqs,
                    tags=teacher_output.tags,
                    tokens_used=tokens_used,
                )
                session.add(db_output)

                # Update Qdrant payload
                if chunk.embedding_id:
                    client = get_qdrant_client()
                    client.set_payload(
                        collection_name=settings.QDRANT_COLLECTION,
                        payload={
                            "summary": teacher_output.summary,
                            "tags": teacher_output.tags,
                        },
                        points=[chunk.embedding_id],
                    )

                session.commit()
                processed += 1

            except Exception as e:
                logger.error(f"Teacher failed for chunk {chunk.id}: {e}")
                session.rollback()
                failed += 1
                continue

        logger.info(f"Teacher processing complete: {processed} processed, {failed} failed")
        return {
            "total": total,
            "processed": processed,
            "failed": failed,
        }

    finally:
        session.close()
