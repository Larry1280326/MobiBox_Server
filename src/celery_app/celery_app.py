"""Celery application instance and configuration."""

import logging

from celery import Celery
from celery.signals import setup_logging

from src.config import get_settings
from src.celery_app.config import CELERY_BEAT_SCHEDULE

# Enable debug logging for HAR service to see IMU query details
logging.getLogger("src.celery_app.services.har_service").setLevel(logging.DEBUG)

settings = get_settings()

# Create Celery app instance
celery_app = Celery(
    "mobibox",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)


@setup_logging.connect
def configure_celery_logging(**kwargs):
    """Configure Celery logging with rotational file handlers."""
    from src.logging_config import get_file_handler, DEFAULT_LOGS_DIR

    # Disable Celery's default logging configuration
    # We'll set up our own handlers

    # Get the log file based on the current process
    # Celery worker uses 'celery_worker.log', beat uses 'celery_beat.log'
    import sys

    if "beat" in " ".join(sys.argv):
        log_file = "celery_beat.log"
    else:
        log_file = "celery_worker.log"

    # Create rotating file handler
    file_handler = get_file_handler(log_file)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Add file handler to root logger
    root_logger.addHandler(file_handler)

    # Configure celery logger
    celery_logger = logging.getLogger("celery")
    celery_logger.setLevel(logging.INFO)
    celery_logger.addHandler(file_handler)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=False,
    # Worker settings
    worker_prefetch_multiplier=1,  # Only fetch one task at a time per worker
    worker_concurrency=1,  # Number of concurrent worker processes (reduced for ML model memory)
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