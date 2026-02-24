"""Celery tasks for Atomic Activities generation."""

import asyncio
import logging
from typing import List

from celery import shared_task

from src.celery_app.celery_app import celery_app
from src.celery_app.config import ATOMIC_TASK_RATE_LIMIT, ATOMIC_DEBOUNCE_SECONDS
from src.celery_app.services.atomic_service import (
    generate_all_atomic_labels,
    insert_atomic_activity,
)
from src.celery_app.schemas.atomic_schemas import AtomicActivityResult
from src.database import get_supabase_client

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run async coroutine in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# In-memory debounce tracking
_last_atomic_process_time: dict[str, float] = {}


def _should_process_user(user: str) -> bool:
    """Check if enough time has passed since last atomic processing for this user."""
    import time
    last_time = _last_atomic_process_time.get(user, 0)
    current_time = time.time()
    return current_time - last_time >= ATOMIC_DEBOUNCE_SECONDS


def _mark_user_processed(user: str) -> None:
    """Mark user as processed at current time."""
    import time
    _last_atomic_process_time[user] = time.time()


@celery_app.task(
    bind=True,
    rate_limit=ATOMIC_TASK_RATE_LIMIT,
    name="process_atomic_activities_batch"
)
def process_atomic_activities_batch(self, user_list: List[str]) -> dict:
    """
    Generate atomic activities for a batch of users.

    This task is triggered when new document data is uploaded.
    It generates 7 dimensions of atomic activity labels.

    Args:
        user_list: List of user identifiers to process

    Returns:
        Summary of processing results
    """
    logger.info(f"Processing atomic activities batch for {len(user_list)} users")

    client = get_supabase_client()
    results = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "activities": [],
    }

    async def process_all():
        for user in user_list:
            # Check debounce
            if not _should_process_user(user):
                logger.debug(f"Skipping user {user} due to debounce")
                results["skipped"] += 1
                continue

            try:
                # Generate all atomic labels
                activity = await generate_all_atomic_labels(user, client)

                # Insert to database
                await insert_atomic_activity(activity, client)

                results["processed"] += 1
                results["activities"].append({
                    "user": user,
                    "har_label": activity.har_label,
                    "app_category": activity.app_category,
                })
                _mark_user_processed(user)

            except Exception as e:
                logger.error(f"Error processing atomic activities for user {user}: {e}")
                results["errors"] += 1

    _run_async(process_all())

    logger.info(f"Atomic activities batch complete: {results}")
    return results


@shared_task(name="process_atomic_single")
def process_atomic_single(user: str) -> dict:
    """
    Process atomic activities for a single user.

    Convenience task for triggering atomic processing for one user.

    Args:
        user: User identifier

    Returns:
        Processing result
    """
    return process_atomic_activities_batch.delay([user])