"""Celery application for MobiBox activity recognition and intervention system."""

from src.celery_app.celery_app import celery_app

__all__ = ["celery_app"]