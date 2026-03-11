"""Celery tasks for data archival to Supabase Storage."""

import asyncio
import logging

from celery import shared_task

from src.celery_app.celery_app import celery_app
from src.celery_app.services.archive_service import run_archival

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run async coroutine in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    name="archive_data_periodic",
)
def archive_data_periodic(self) -> dict:
    """Periodic task to archive old data to Supabase Storage.

    This task runs daily (configured in beat schedule) and archives
    records older than their retention period to CSV files in storage,
    then deletes them from the database.

    Returns:
        Dictionary with archival statistics for all tables
    """
    logger.info("Starting periodic data archival")

    try:
        results = _run_async(run_archival())

        # Log summary
        total_archived = sum(
            r.get("archived", 0) for r in results.values() if isinstance(r, dict)
        )
        total_deleted = sum(
            r.get("deleted", 0) for r in results.values() if isinstance(r, dict)
        )

        logger.info(
            f"Archival complete: {total_archived} records archived, "
            f"{total_deleted} records deleted"
        )

        return results

    except Exception as e:
        logger.error(f"Archival task failed: {e}")
        return {"error": str(e)}


@shared_task(name="archive_table_manual")
def archive_table_manual(table_name: str, retention_days: int) -> dict:
    """Manually trigger archival for a specific table.

    Args:
        table_name: Name of the table to archive
        retention_days: Number of days to keep data

    Returns:
        Dictionary with archival statistics
    """
    from src.celery_app.services.archive_service import ArchiveService

    logger.info(f"Manual archival requested for {table_name} (retention: {retention_days} days)")

    async def run_manual_archival():
        service = ArchiveService()
        return await service.archive_table(
            table_name=table_name,
            retention_days=retention_days,
            batch_size=service.settings.archive_batch_size,
        )

    return _run_async(run_manual_archival())


@shared_task(name="get_archive_stats")
def get_archive_stats() -> dict:
    """Get statistics about archiveable data.

    Returns:
        Dictionary with counts of records older than retention periods
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from src.database import get_supabase_client
    from src.config import get_settings

    settings = get_settings()
    client = get_supabase_client()
    china_tz = ZoneInfo("Asia/Shanghai")

    stats = {}

    tables = [
        ("imu", settings.retention_imu_days),
        ("har", settings.retention_har_days),
        ("atomic_activities", settings.retention_atomic_days),
        ("uploads", settings.retention_uploads_days),
        ("summary_logs", settings.retention_summary_logs_days),
        ("interventions", settings.retention_interventions_days),
    ]

    async def count_records():
        for table_name, retention_days in tables:
            cutoff_date = datetime.now(china_tz) - timedelta(days=retention_days)
            cutoff_iso = cutoff_date.isoformat()

            try:
                # Use count with head=True for efficiency
                import asyncio
                response = await asyncio.to_thread(
                    lambda: client.table(table_name)
                    .select("id", count="exact")
                    .lt("timestamp", cutoff_iso)
                    .execute()
                )
                stats[table_name] = {
                    "archiveable_count": response.count if hasattr(response, 'count') else len(response.data),
                    "retention_days": retention_days,
                }
            except Exception as e:
                logger.error(f"Failed to get stats for {table_name}: {e}")
                stats[table_name] = {"error": str(e)}

        return stats

    return _run_async(count_records())