"""Celery tasks for HAR (Human Activity Recognition) processing."""

import asyncio
import logging
from typing import List

from celery import shared_task

from src.celery_app.celery_app import celery_app
from src.celery_app.config import (
    HAR_TASK_RATE_LIMIT,
    HAR_DEBOUNCE_SECONDS,
    HAR_IMU_WINDOW_SECONDS,
    HAR_DATA_DELAY_SECONDS,
)
from src.celery_app.services.har_service import process_har_for_user
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


# In-memory debounce tracking (for single worker; for multi-worker use Redis)
_last_har_process_time: dict[str, float] = {}


def _should_process_user(user: str) -> bool:
    """Check if enough time has passed since last HAR processing for this user."""
    import time
    last_time = _last_har_process_time.get(user, 0)
    current_time = time.time()
    return current_time - last_time >= HAR_DEBOUNCE_SECONDS


def _mark_user_processed(user: str) -> None:
    """Mark user as processed at current time."""
    import time
    _last_har_process_time[user] = time.time()


@celery_app.task(bind=True, rate_limit=HAR_TASK_RATE_LIMIT, name="process_har_batch")
def process_har_batch(self, user_list: List[str]) -> dict:
    """
    Process HAR for a batch of users.

    This task is triggered when new IMU data is uploaded.
    It processes users that haven't been processed recently (debounced).

    Args:
        user_list: List of user identifiers to process

    Returns:
        Summary of processing results
    """
    logger.info(f"Processing HAR batch for {len(user_list)} users")

    client = get_supabase_client()
    results = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "labels": [],
    }

    async def process_all():
        for user in user_list:
            # Check debounce
            if not _should_process_user(user):
                logger.debug(f"Skipping user {user} due to debounce")
                results["skipped"] += 1
                continue

            try:
                result = await process_har_for_user(user, client)
                if result:
                    results["processed"] += 1
                    results["labels"].append({
                        "user": user,
                        "label": result.label,
                        "confidence": result.confidence,
                    })
                    _mark_user_processed(user)
                else:
                    results["skipped"] += 1
                    logger.debug(f"No IMU data for user {user}")
            except Exception as e:
                logger.error(f"Error processing HAR for user {user}: {e}")
                results["errors"] += 1

    _run_async(process_all())

    logger.info(f"HAR batch complete: {results}")
    return results


@shared_task(name="process_har_single")
def process_har_single(user: str) -> dict:
    """
    Process HAR for a single user.

    Convenience task for triggering HAR processing for one user.

    Args:
        user: User identifier

    Returns:
        Processing result
    """
    return process_har_batch.delay([user])


@celery_app.task(
    bind=True,
    rate_limit=HAR_TASK_RATE_LIMIT,
    name="src.celery_app.tasks.har_tasks.process_har_periodic",
)
def process_har_periodic(self) -> dict:
    """
    Periodic task to process HAR for all active users.

    This task runs every 2 seconds (configured in beat schedule) and processes
    HAR for users that have recent IMU data and haven't been processed recently.

    Returns:
        Summary of processing results
    """
    logger.info("Running periodic HAR processing")

    client = get_supabase_client()
    results = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "labels": [],
    }

    async def process_active_users():
        import time
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        china_tz = ZoneInfo("Asia/Shanghai")

        # Compute delayed window to match batch upload behavior:
        # we look at [now - delay - window, now - delay]
        delayed_end = datetime.now(china_tz) - timedelta(seconds=HAR_DATA_DELAY_SECONDS)
        delayed_start = delayed_end - timedelta(seconds=HAR_IMU_WINDOW_SECONDS)

        logger.debug(
            "Periodic HAR active-user window: %s to %s (delay=%ss, window=%ss)",
            delayed_start.isoformat(),
            delayed_end.isoformat(),
            HAR_DATA_DELAY_SECONDS,
            HAR_IMU_WINDOW_SECONDS,
        )

        response = await asyncio.to_thread(
            lambda: client.table("imu")
            .select("user")
            .gte("timestamp", delayed_start.isoformat())
            .lte("timestamp", delayed_end.isoformat())
            .execute()
        )

        if not response.data:
            logger.debug("No users with IMU data in delayed window")
            return

        # Get unique users
        users = list(set(item["user"] for item in response.data))
        logger.info(f"Found {len(users)} users with IMU data in delayed window")

        for user in users:
            # Check debounce
            if not _should_process_user(user):
                logger.debug(f"Skipping user {user} due to debounce")
                results["skipped"] += 1
                continue

            try:
                result = await process_har_for_user(user, client)
                if result:
                    results["processed"] += 1
                    results["labels"].append({
                        "user": user,
                        "label": result.label,
                        "confidence": result.confidence,
                    })
                    _mark_user_processed(user)
                else:
                    results["skipped"] += 1
                    logger.debug(f"No IMU data for user {user}")
            except Exception as e:
                logger.error(f"Error processing HAR for user {user}: {e}")
                results["errors"] += 1

    _run_async(process_active_users())

    logger.info(f"Periodic HAR complete: {results}")
    return results