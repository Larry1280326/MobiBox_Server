"""Celery schemas for MobiBox."""

from src.celery_app.schemas.har_schemas import HARLabel
from src.celery_app.schemas.atomic_schemas import AtomicActivity

__all__ = ["HARLabel", "AtomicActivity"]