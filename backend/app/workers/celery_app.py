"""
Celery Application Configuration.

Defines the Celery app with Redis broker, task queues, and routing.
"""

from celery import Celery

from app.config import get_settings
# Import all database models to ensure SQLAlchemy mapper configuration
# resolves relationship references correctly on Celery worker startup.
from app.modules.documents.models import Document
from app.modules.chunks.models import Chunk
from app.modules.teacher.models import TeacherOutput
from app.modules.datasets.models import Dataset, DatasetSample
from app.modules.training.models import TrainingRun
from app.modules.registry.models import ModelVersion

settings = get_settings()

celery_app = Celery(
    "kdp_workers",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.workers.document_tasks",
        "app.workers.teacher_tasks",
        "app.workers.dataset_tasks",
        "app.workers.training_tasks",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task settings
    task_track_started=True,
    task_time_limit=7200,       # 2 hours max per task
    task_soft_time_limit=6600,  # Soft limit at 1h50m
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1,  # Restart workers after 1 task to free GPU memory and prevent leaks

    # Result settings
    result_expires=86400,  # Results expire after 24 hours

    # Task routing
    task_routes={
        "app.workers.document_tasks.*": {"queue": "documents"},
        "app.workers.teacher_tasks.*": {"queue": "teacher"},
        "app.workers.dataset_tasks.*": {"queue": "default"},
        "app.workers.training_tasks.*": {"queue": "training"},
    },

    # Default queue
    task_default_queue="default",
)

# Auto-discover tasks in worker modules
celery_app.autodiscover_tasks([
    "app.workers",
])
