"""Celery application instance and configuration."""

from celery import Celery

from src.config import get_settings
from src.celery_app.config import CELERY_BEAT_SCHEDULE

settings = get_settings()

# Create Celery app instance
celery_app = Celery(
    "mobibox",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker settings
    worker_prefetch_multiplier=1,  # Only fetch one task at a time per worker
    worker_concurrency=4,  # Number of concurrent worker processes
    # Task result settings
    result_expires=3600,  # Results expire after 1 hour
    # Beat schedule
    beat_schedule=CELERY_BEAT_SCHEDULE,
    # Task routing (optional, for future scaling)
    task_routes={
        "src.celery_app.tasks.har_tasks.*": {"queue": "har"},
        "src.celery_app.tasks.atomic_tasks.*": {"queue": "atomic"},
        "src.celery_app.tasks.summary_tasks.*": {"queue": "summary"},
    },
    # Default queue
    task_default_queue="default",
)

# Autodiscover tasks from all task modules
celery_app.autodiscover_tasks([
    "src.celery_app.tasks.har_tasks",
    "src.celery_app.tasks.atomic_tasks",
    "src.celery_app.tasks.summary_tasks",
])