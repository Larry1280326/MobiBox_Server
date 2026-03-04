"""Celery tasks for generating interventions and summary logs.

Scheduled tasks:
- generate_hourly_summary: Runs every hour at minute 0
- generate_hourly_interventions: Runs every hour at minute 5 (after summaries)
- generate_daily_summary: Runs once daily at midnight

Data flow:
  atomic_activities -> summary_logs -> interventions
"""

import asyncio
import logging

from celery import shared_task

from src.celery_app.celery_app import celery_app
from src.celery_app.services.summary_service import (
    get_all_users_with_activities,
    compress_atomic_activities,
    generate_summary,
    insert_summary_log,
)
from src.celery_app.services.intervention_service import (
    generate_intervention_from_summary,
    get_recent_summaries,
    insert_intervention,
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
    Generate interventions based on recent summaries.

    Runs every hour via Celery Beat (5 minutes after summary generation).
    Reads from summary_logs table instead of atomic_activities.
    """
    logger.info("Starting hourly intervention generation from summaries")

    async def process():
        client = get_supabase_client()

        # Get recent summaries from the last hour
        summaries = await get_recent_summaries(hours=1, client=client)
        logger.info(f"Found {len(summaries)} recent summaries")

        results = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "interventions": [],
        }

        for summary in summaries:
            user = summary.get("user")
            if not user:
                results["skipped"] += 1
                continue

            try:
                # Generate intervention from summary
                intervention = await generate_intervention_from_summary(user, summary)

                if intervention:
                    # Link intervention to source summary
                    intervention["summary_id"] = summary.get("id")

                    # Insert to database
                    await insert_intervention(intervention, client)
                    results["processed"] += 1
                    results["interventions"].append({
                        "user": user,
                        "type": intervention.get("intervention_type"),
                        "priority": intervention.get("priority"),
                        "summary_id": summary.get("id"),
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
                        "log_type": summary_log.get("log_type"),
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
                        "log_type": summary_log.get("log_type"),
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
def trigger_intervention_for_user(user: str, hours: int = 1) -> dict:
    """
    Manually trigger intervention generation for a specific user.

    Generates intervention based on the most recent summary for the user.

    Args:
        user: User identifier
        hours: Number of hours to look back for summaries (default: 1)

    Returns:
        Generated intervention data
    """
    async def process():
        client = get_supabase_client()

        # Get recent summaries for this user
        summaries = await get_recent_summaries(hours=hours, client=client)
        user_summaries = [s for s in summaries if s.get("user") == user]

        if not user_summaries:
            return {"error": "No recent summaries found for user"}

        # Use the most recent summary
        summary = user_summaries[-1] if user_summaries else None

        intervention = await generate_intervention_from_summary(user, summary)
        if intervention:
            intervention["summary_id"] = summary.get("id")
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