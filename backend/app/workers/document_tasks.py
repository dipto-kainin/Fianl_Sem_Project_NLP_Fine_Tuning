"""
Document Processing Tasks.

Celery tasks for the document processing pipeline:
Parse → Chunk → Embed → Store in Qdrant.
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
    # Convert async URL to sync URL
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2").replace("postgresql+psycopg2", "postgresql")
    engine = create_engine(sync_url)
    Session = sessionmaker(bind=engine)
    return Session()


@celery_app.task(name="app.workers.document_tasks.process_document_pipeline", bind=True)
def process_document_pipeline(
    self,
    document_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    run_teacher: bool = True,
):
    """
    Full document processing pipeline.

    Steps:
    1. Parse the document
    2. Chunk the parsed text
    3. Generate embeddings for each chunk
    4. Store chunks + embeddings in Qdrant
    5. (Optional) Run Teacher LLM on each chunk
    """
    from app.modules.documents.parser import parse_document
    from app.modules.chunks.chunker import chunk_document
    from app.utils.embeddings import generate_embeddings_batch
    from app.modules.knowledge.qdrant_client import upsert_knowledge_batch, ensure_collection

    logger.info(f"Starting pipeline for document: {document_id}")

    session = _get_sync_session()

    try:
        # Fetch document
        from app.modules.documents.models import Document, DocumentStatus
        from app.modules.chunks.models import Chunk

        doc = session.query(Document).filter(Document.id == uuid.UUID(document_id)).first()
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        doc.status = DocumentStatus.PROCESSING
        session.commit()

        # Step 1: Parse
        self.update_state(state="PARSING", meta={"step": "parsing"})
        logger.info(f"Step 1: Parsing {doc.file_type.value} document")
        parsed = parse_document(doc.file_path, doc.file_type.value)

        doc.raw_text = parsed.raw_text[:50000]  # Store truncated raw text
        doc.page_count = parsed.page_count
        doc.language = parsed.language
        doc.metadata_ = parsed.metadata
        session.commit()

        # Step 2: Chunk
        self.update_state(state="CHUNKING", meta={"step": "chunking"})
        logger.info(f"Step 2: Chunking document ({len(parsed.raw_text)} chars)")
        chunk_results = chunk_document(parsed, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Save chunks to database
        db_chunks = []
        for cr in chunk_results:
            chunk = Chunk(
                document_id=uuid.UUID(document_id),
                chunk_index=cr.chunk_index,
                text=cr.text,
                token_count=cr.token_count,
                section_title=cr.section_title,
                page_numbers=cr.page_numbers if cr.page_numbers else None,
                metadata_=cr.metadata,
            )
            session.add(chunk)
            db_chunks.append(chunk)
        session.flush()

        logger.info(f"Created {len(db_chunks)} chunks")

        # Step 3: Generate embeddings
        self.update_state(state="EMBEDDING", meta={"step": "embedding", "chunks": len(db_chunks)})
        logger.info(f"Step 3: Generating embeddings for {len(db_chunks)} chunks")
        texts = [c.text for c in db_chunks]
        embeddings = generate_embeddings_batch(texts)

        # Step 4: Store in Qdrant
        self.update_state(state="STORING", meta={"step": "storing"})
        logger.info("Step 4: Storing in Qdrant")
        ensure_collection()

        points = []
        for i, (chunk, embedding) in enumerate(zip(db_chunks, embeddings)):
            point_id = str(uuid.uuid4())
            chunk.embedding_id = point_id
            points.append({
                "point_id": point_id,
                "vector": embedding,
                "chunk_id": str(chunk.id),
                "document_id": document_id,
                "text": chunk.text,
                "section_title": chunk.section_title or "",
                "tags": [],
                "metadata": chunk.metadata_ or {},
            })

        upsert_knowledge_batch(points)
        session.commit()

        # Step 5: Run Teacher LLM (optional)
        if run_teacher:
            self.update_state(state="TEACHER", meta={"step": "teacher", "chunks": len(db_chunks)})
            logger.info("Step 5: Running Teacher LLM pipeline")
            _run_teacher_on_chunks(session, db_chunks, document_id)

            # Step 6: Auto-generate dataset for this document
            self.update_state(state="DATASET", meta={"step": "dataset_generation"})
            logger.info("Step 6: Auto-generating dataset for this document")
            try:
                _auto_generate_document_dataset(document_id, doc.filename)
            except Exception as ds_err:
                logger.error(f"Auto dataset generation failed for {document_id}: {ds_err}", exc_info=True)

        # Mark as processed
        doc.status = DocumentStatus.PROCESSED
        session.commit()

        # Check if we should trigger auto-learning (10 processed documents threshold)
        try:
            processed_count = session.query(Document).filter(Document.status == DocumentStatus.PROCESSED).count()
            logger.info(f"Processed documents count: {processed_count}/10 for auto-learning")
            if processed_count >= 10:
                logger.info("Auto-learning threshold (10 documents) reached! Triggering training.")
                _trigger_auto_train_sequence()
        except Exception as auto_err:
            logger.error(f"Failed to check/trigger auto-learning: {auto_err}", exc_info=True)

        logger.info(f"Pipeline complete for document: {document_id}")
        return {
            "document_id": document_id,
            "chunks_created": len(db_chunks),
            "status": "processed",
        }

    except Exception as e:
        logger.error(f"Pipeline failed for document {document_id}: {e}", exc_info=True)
        try:
            doc = session.query(Document).filter(Document.id == uuid.UUID(document_id)).first()
            if doc:
                doc.status = DocumentStatus.FAILED
                doc.error_message = str(e)[:2000]
                session.commit()
        except Exception:
            session.rollback()
        raise

    finally:
        session.close()


def _run_teacher_on_chunks(session, chunks, document_id: str):
    """Process all chunks with the Teacher LLM."""
    from app.modules.teacher.gemini_client import process_chunk_with_teacher
    from app.modules.teacher.models import TeacherOutput
    from app.modules.knowledge.qdrant_client import get_qdrant_client
    from app.config import get_settings
    from qdrant_client.models import PointStruct

    settings = get_settings()

    for i, chunk in enumerate(chunks):
        try:
            logger.info(f"Teacher processing chunk {i+1}/{len(chunks)}")
            teacher_output, tokens_used = process_chunk_with_teacher(
                chunk_text=chunk.text,
                section_title=chunk.section_title or "",
            )

            # Store teacher output in database
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

            # Update Qdrant payload with teacher labels
            if chunk.embedding_id:
                client = get_qdrant_client()
                client.set_payload(
                    collection_name=settings.QDRANT_COLLECTION,
                    payload={
                        "summary": teacher_output.summary,
                        "tags": teacher_output.tags,
                        "entities": {"entities": [e.model_dump() for e in teacher_output.entities]},
                    },
                    points=[chunk.embedding_id],
                )

            session.commit()

        except Exception as e:
            logger.error(f"Teacher failed for chunk {chunk.id}: {e}")
            session.rollback()
            continue  # Continue with remaining chunks


def _auto_generate_document_dataset(document_id: str, filename: str):
    """
    Auto-generate a dataset scoped to a single document after it's processed (Synchronous).
    """
    import re
    import json
    import uuid as uuid_mod
    from pathlib import Path
    from collections import Counter
    from app.config import get_settings
    from app.modules.chunks.models import Chunk
    from app.modules.datasets.models import Dataset, DatasetSample, DatasetStatus, DifficultyLevel
    from app.modules.teacher.models import TeacherOutput

    settings = get_settings()

    # Build a safe version string from the filename (strip extension, sanitize)
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    safe_version = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)[:60]  # max 60 chars
    short_id = document_id[:8]
    version = f"{safe_version}_{short_id}"

    session = _get_sync_session()
    try:
        # 1. Create dataset record
        dataset = Dataset(
            version=version,
            status=DatasetStatus.GENERATING,
            description=f"Auto-generated from document: {filename}",
        )
        session.add(dataset)
        session.flush()

        # 2. Get already used chunk IDs
        already_used_ids = [
            r[0] for r in session.query(DatasetSample.source_chunk_id)
            .filter(DatasetSample.source_chunk_id.isnot(None))
            .all()
        ]

        # 3. Query teacher outputs for this document
        doc_uuid = uuid_mod.UUID(document_id)
        query = session.query(TeacherOutput).join(Chunk, TeacherOutput.chunk_id == Chunk.id).filter(
            Chunk.document_id == doc_uuid
        )
        if already_used_ids:
            query = query.filter(TeacherOutput.chunk_id.not_in(already_used_ids))

        teacher_outputs = query.all()

        if not teacher_outputs:
            dataset.status = DatasetStatus.READY
            dataset.total_samples = 0
            session.commit()
            logger.info(f"Auto-dataset created: version='{version}', samples=0, doc={filename}")
            return

        # 4. Generate samples
        samples = []
        category_counter = Counter()

        for output in teacher_outputs:
            chunk = session.query(Chunk).filter(Chunk.id == output.chunk_id).first()
            context_text = chunk.text if chunk else ""

            if output.qa_pairs:
                for qa in output.qa_pairs:
                    if isinstance(qa, dict) and "instruction" in qa and "answer" in qa:
                        # Estimate difficulty
                        word_count = len(qa["answer"].split())
                        difficulty = DifficultyLevel.EASY if word_count < 20 else DifficultyLevel.MEDIUM if word_count < 80 else DifficultyLevel.HARD
                        category = (output.tags[0] if output.tags else "general")
                        category_counter[category] += 1

                        samples.append(DatasetSample(
                            dataset_id=dataset.id,
                            instruction=qa["instruction"],
                            context=context_text[:2000],
                            response=qa["answer"],
                            difficulty=difficulty,
                            category=category,
                            source_chunk_id=output.chunk_id,
                        ))

        session.add_all(samples)
        dataset.total_samples = len(samples)
        dataset.status = DatasetStatus.READY
        dataset.categories = dict(category_counter)

        # 5. Export to JSONL file
        output_dir = Path(settings.DATASETS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"dataset_{version}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                entry = {
                    "instruction": sample.instruction,
                    "context": sample.context or "",
                    "response": sample.response,
                    "difficulty": sample.difficulty.value,
                    "category": sample.category or "general",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        dataset.file_path = str(file_path)
        session.commit()
        logger.info(f"Auto-dataset created: version='{version}', samples={len(samples)}, doc={filename}")

    except Exception as e:
        session.rollback()
        logger.error(f"Error during auto-generating dataset: {e}", exc_info=True)
        raise
    finally:
        session.close()


def _trigger_auto_train_sequence():
    """Generates dataset and starts training automatically when 10 sources are accumulated."""
    import asyncio
    from app.database import AsyncSessionLocal
    from app.modules.datasets.service import generate_dataset
    from app.modules.training.service import create_training_run
    from app.workers.training_tasks import run_training_pipeline

    async def _async_trigger():
        async with AsyncSessionLocal() as session:
            # 1. Create dataset from all processed documents
            logger.info("Auto-learning: Generating training dataset...")
            dataset = await generate_dataset(
                db=session,
                version=None,
                description="Auto-generated dataset from 10+ documents"
            )
            await session.commit()
            logger.info(f"Auto-learning: Created dataset {dataset.version} ({dataset.total_samples} samples)")

            # 2. Create training run config
            logger.info("Auto-learning: Creating training run...")
            run = await create_training_run(
                db=session,
                dataset_id=dataset.id,
            )
            await session.commit()
            logger.info(f"Auto-learning: Created training run {run.id}")
            return run.id

    try:
        run_id = asyncio.run(_async_trigger())
        # 3. Dispatch LoRA training task in background
        run_training_pipeline.delay(str(run_id))
        logger.info("Auto-learning: Fine-tuning worker successfully queued.")
    except Exception as e:
        logger.error(f"Auto-learning sequence failed: {e}", exc_info=True)
