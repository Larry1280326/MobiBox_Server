"""Celery tasks for generating interventions and summary logs.

Scheduled tasks:
- generate_hourly_interventions: Runs every hour
- generate_hourly_summary: Runs every hour
- generate_daily_summary: Runs once daily at midnight
"""

import asyncio
import logging
from typing import List

from celery import shared_task

from src.celery_app.celery_app import celery_app
from src.celery_app.services.summary_service import (
    get_all_users_with_activities,
    compress_atomic_activities,
    generate_intervention,
    generate_summary,
    insert_intervention,
    insert_summary_log,
)
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


@celery_app.task(name="generate_hourly_interventions")
def generate_hourly_interventions() -> dict:
    """
    Generate interventions for all users with recent activity.

    Runs every hour via Celery Beat.
    Compresses the last hour of atomic activities and generates
    personalized health interventions.
    """
    logger.info("Starting hourly intervention generation")

    async def process():
        client = get_supabase_client()

        # Get users with activity in the last hour
        users = await get_all_users_with_activities(hours=1, client=client)
        logger.info(f"Found {len(users)} users with recent activity")

        results = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "interventions": [],
        }

        for user in users:
            try:
                # Compress activities
                compressed = await compress_atomic_activities(user, hours=1, client=client)

                if not compressed.get("total_records"):
                    results["skipped"] += 1
                    continue

                # Generate intervention
                intervention = await generate_intervention(user, compressed)

                if intervention:
                    # Insert to database
                    await insert_intervention(intervention, client)
                    results["processed"] += 1
                    results["interventions"].append({
                        "user": user,
                        "type": intervention.get("intervention_type"),
                        "priority": intervention.get("priority"),
                    })
                else:
                    results["skipped"] += 1

            except Exception as e:
                logger.error(f"Error generating intervention for user {user}: {e}")
                results["errors"] += 1

        return results

    results = _run_async(process())
    logger.info(f"Hourly intervention generation complete: {results}")
    return results


@celery_app.task(name="generate_hourly_summary")
def generate_hourly_summary() -> dict:
    """
    Generate hourly summary logs for all users with recent activity.

    Runs every hour via Celery Beat.
    Creates a narrative summary of the user's activities.
    """
    logger.info("Starting hourly summary generation")

    async def process():
        client = get_supabase_client()

        # Get users with activity in the last hour
        users = await get_all_users_with_activities(hours=1, client=client)
        logger.info(f"Found {len(users)} users with recent activity")

        results = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "summaries": [],
        }

        for user in users:
            try:
                # Compress activities
                compressed = await compress_atomic_activities(user, hours=1, client=client)

                if not compressed.get("total_records"):
                    results["skipped"] += 1
                    continue

                # Generate summary
                summary_log = await generate_summary(user, compressed, log_type="hourly")

                if summary_log:
                    # Insert to database
                    await insert_summary_log(summary_log, client)
                    results["processed"] += 1
                    results["summaries"].append({
                        "user": user,
                        "title": summary_log.get("title"),
                    })
                else:
                    results["skipped"] += 1

            except Exception as e:
                logger.error(f"Error generating summary for user {user}: {e}")
                results["errors"] += 1

        return results

    results = _run_async(process())
    logger.info(f"Hourly summary generation complete: {results}")
    return results


@celery_app.task(name="generate_daily_summary")
def generate_daily_summary() -> dict:
    """
    Generate daily summary logs for all users with activity.

    Runs once daily at midnight via Celery Beat.
    Creates a comprehensive summary of the user's day.
    """
    logger.info("Starting daily summary generation")

    async def process():
        client = get_supabase_client()

        # Get users with activity in the last 24 hours
        users = await get_all_users_with_activities(hours=24, client=client)
        logger.info(f"Found {len(users)} users with activity in the last 24 hours")

        results = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "summaries": [],
        }

        for user in users:
            try:
                # Compress activities for the full day
                compressed = await compress_atomic_activities(user, hours=24, client=client)

                if not compressed.get("total_records"):
                    results["skipped"] += 1
                    continue

                # Generate daily summary
                summary_log = await generate_summary(user, compressed, log_type="daily")

                if summary_log:
                    # Insert to database
                    await insert_summary_log(summary_log, client)
                    results["processed"] += 1
                    results["summaries"].append({
                        "user": user,
                        "title": summary_log.get("title"),
                    })
                else:
                    results["skipped"] += 1

            except Exception as e:
                logger.error(f"Error generating daily summary for user {user}: {e}")
                results["errors"] += 1

        return results

    results = _run_async(process())
    logger.info(f"Daily summary generation complete: {results}")
    return results


# Manual trigger tasks (for testing/admin use)

@shared_task(name="trigger_intervention_for_user")
def trigger_intervention_for_user(user: str) -> dict:
    """
    Manually trigger intervention generation for a specific user.

    Args:
        user: User identifier

    Returns:
        Generated intervention data
    """
    async def process():
        client = get_supabase_client()

        compressed = await compress_atomic_activities(user, hours=1, client=client)
        if not compressed.get("total_records"):
            return {"error": "No recent activity data for user"}

        intervention = await generate_intervention(user, compressed)
        if intervention:
            await insert_intervention(intervention, client)

        return intervention or {}

    return _run_async(process())


@shared_task(name="trigger_summary_for_user")
def trigger_summary_for_user(user: str, hours: int = 1) -> dict:
    """
    Manually trigger summary generation for a specific user.

    Args:
        user: User identifier
        hours: Number of hours to summarize

    Returns:
        Generated summary data
    """
    async def process():
        client = get_supabase_client()

        compressed = await compress_atomic_activities(user, hours=hours, client=client)
        if not compressed.get("total_records"):
            return {"error": "No activity data for user in specified period"}

        log_type = "daily" if hours >= 24 else "hourly"
        summary_log = await generate_summary(user, compressed, log_type=log_type)

        if summary_log:
            await insert_summary_log(summary_log, client)

        return summary_log or {}

    return _run_async(process())