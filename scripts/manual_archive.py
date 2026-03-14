#!/usr/bin/env python3
"""Manual archival script for all tables.

Usage:
    cd /path/to/MobiBox_Server
    python -m scripts.manual_archive

    # Or run directly:
    python scripts/manual_archive.py

    # Archive specific table:
    python -m scripts.manual_archive --table imu

    # Archive before specific date:
    python -m scripts.manual_archive --before 2026-03-01
"""

import argparse
import asyncio
import io
import logging
import random
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import get_settings
from src.database import get_supabase_admin_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")

# Retry configuration (matching archive_service.py)
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 60.0  # seconds
BATCH_DELAY = 1.0  # delay between successful batches

# Table names
TABLES = {
    "imu": "IMU data",
    "uploads": "Upload metadata",
    "har": "HAR labels",
    "atomic_activities": "Atomic activities",
    "summary_logs": "Summary logs",
    "interventions": "Interventions",
}


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (transient)."""
    error_str = str(error).lower()
    # 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout
    # Connection errors, rate limiting
    retryable_codes = ['502', '503', '504', '429', 'timeout', 'connection', 'rate']
    return any(code in error_str for code in retryable_codes)


async def retry_with_backoff(func, *args, **kwargs):
    """Execute a function with exponential backoff retry logic.

    Retries indefinitely for retryable errors.
    Non-retryable errors are raised immediately.
    """
    attempt = 0

    while True:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if not is_retryable_error(e):
                raise  # Non-retryable error, fail immediately

            attempt += 1
            # Calculate delay with exponential backoff and jitter, capped at MAX_DELAY
            delay = min(BASE_DELAY * (2 ** min(attempt, 6)) + random.uniform(0, 1), MAX_DELAY)
            logger.warning(f"Retryable error on attempt {attempt}: {e}")
            logger.info(f"Waiting {delay:.1f} seconds before retry...")
            await asyncio.sleep(delay)


async def count_records(client, table_name: str, cutoff_iso: str) -> int:
    """Count records older than cutoff."""
    async def _count():
        return await asyncio.to_thread(
            lambda: client.table(table_name)
            .select("id", count="exact")
            .lt("timestamp", cutoff_iso)
            .execute()
        )
    response = await retry_with_backoff(_count)
    return getattr(response, 'count', 0) if response else 0


async def fetch_batch(client, table_name: str, cutoff_iso: str, batch_size: int = 1000) -> list:
    """Fetch a batch of records older than cutoff."""
    async def _fetch():
        return await asyncio.to_thread(
            lambda: client.table(table_name)
            .select("*")
            .lt("timestamp", cutoff_iso)
            .order("timestamp", desc=False)
            .limit(batch_size)
            .execute()
        )
    response = await retry_with_backoff(_fetch)
    return response.data if response.data else []


async def upload_to_storage(client, bucket: str, path: str, content: bytes, content_type: str) -> bool:
    """Upload content to Supabase Storage."""
    try:
        async def _upload():
            return await asyncio.to_thread(
                lambda: client.storage.from_(bucket).upload(
                    path=path,
                    file=content,
                    file_options={"content-type": content_type},
                )
            )
        await retry_with_backoff(_upload)
        logger.info(f"Uploaded archive to {bucket}/{path}")
        return True

    except Exception as e:
        error_str = str(e).lower()
        if "already exists" in error_str or "duplicate" in error_str:
            try:
                async def _update():
                    return await asyncio.to_thread(
                        lambda: client.storage.from_(bucket).update(
                            path=path,
                            file=content,
                            file_options={"content-type": content_type},
                        )
                    )
                await retry_with_backoff(_update)
                logger.info(f"Updated archive at {bucket}/{path}")
                return True
            except Exception as update_error:
                logger.error(f"Failed to update archive {path}: {update_error}")
                return False
        else:
            logger.error(f"Failed to upload archive {path}: {e}")
            return False


async def delete_records(client, table_name: str, record_ids: list) -> int:
    """Delete archived records from database."""
    if not record_ids:
        return 0

    async def _delete():
        return await asyncio.to_thread(
            lambda: client.table(table_name)
            .delete()
            .in_("id", record_ids)
            .execute()
        )
    await retry_with_backoff(_delete)
    logger.info(f"Deleted {len(record_ids)} records from {table_name}")
    return len(record_ids)


async def log_archival(client, table_name: str, archived: int, deleted: int,
                       storage_paths: list, total_size: int, status: str, error: str = None):
    """Log archival operation to database."""
    async def _log():
        return await asyncio.to_thread(
            lambda: client.table("archival_logs").insert({
                "table_name": table_name,
                "records_archived": archived,
                "records_deleted": deleted,
                "storage_path": storage_paths[0] if storage_paths else None,
                "file_size_bytes": total_size,
                "status": status,
                "error_message": error,
            }).execute()
        )
    await retry_with_backoff(_log)


def get_storage_path(table_name: str, record_date: datetime, batch_num: int = 0) -> str:
    """Generate storage path for archived data."""
    year = record_date.strftime("%Y")
    month = record_date.strftime("%m")
    day = record_date.strftime("%Y-%m-%d")

    if batch_num > 0:
        filename = f"{day}_batch{batch_num:04d}.parquet"
    else:
        filename = f"{day}.parquet"

    return f"archives/{table_name}/{year}/{month}/{filename}"


def records_to_parquet(records: list) -> bytes:
    """Convert records to compressed Parquet format."""
    if not records:
        return b""

    table = pa.Table.from_pylist(records)
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
    )
    return buffer.getvalue()


async def archive_table(client, bucket: str, table_name: str, retention_days: int,
                        before_date: datetime = None) -> dict:
    """Archive old records from a table to storage and delete them.

    Args:
        client: Supabase admin client
        bucket: Storage bucket name
        table_name: Name of the table
        retention_days: Number of days to keep data
        before_date: Archive records before this date (default: calculated from retention)

    Returns:
        Dictionary with archival statistics
    """
    # Calculate cutoff date
    if before_date is None:
        cutoff_date = datetime.now(CHINA_TZ) - timedelta(days=retention_days)
    else:
        cutoff_date = before_date

    cutoff_iso = cutoff_date.isoformat()

    logger.info(f"{'='*60}")
    logger.info(f"Archiving table: {table_name} ({TABLES.get(table_name, 'Unknown')})")
    logger.info(f"Retention: {retention_days} days")
    logger.info(f"Cutoff timestamp: {cutoff_iso}")
    logger.info(f"{'='*60}")

    # Count total records to archive
    total_count = await count_records(client, table_name, cutoff_iso)
    logger.info(f"Records to archive: {total_count}")

    if total_count == 0:
        logger.info(f"No records to archive for {table_name}")
        await log_archival(client, table_name, 0, 0, [], 0, "completed")
        return {"table": table_name, "archived": 0, "deleted": 0}

    # Track totals
    total_archived = 0
    total_deleted = 0
    total_size = 0
    storage_paths = []
    batch_num = 0

    # Process in batches
    while True:
        batch_num += 1

        # Fetch batch
        records = await fetch_batch(client, table_name, cutoff_iso)
        if not records:
            logger.info(f"No more records to process for {table_name}")
            break

        logger.info(f"Processing batch {batch_num}: {len(records)} records")

        # Convert to Parquet
        try:
            parquet_bytes = records_to_parquet(records)
            file_size = len(parquet_bytes)
        except Exception as e:
            logger.error(f"Failed to convert batch {batch_num} to Parquet: {e}")
            await log_archival(client, table_name, total_archived, total_deleted,
                             storage_paths, total_size, "partial" if total_archived > 0 else "failed",
                             f"Parquet conversion failed: {e}")
            return {"table": table_name, "archived": total_archived, "deleted": total_deleted, "error": str(e)}

        # Generate storage path
        oldest_record = records[0]
        if "timestamp" in oldest_record and oldest_record["timestamp"]:
            ts_str = oldest_record["timestamp"]
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            record_date = datetime.fromisoformat(ts_str)
        else:
            record_date = datetime.now(CHINA_TZ)

        storage_path = get_storage_path(table_name, record_date, batch_num)

        # Upload to storage
        upload_success = await upload_to_storage(
            client, bucket, storage_path, parquet_bytes,
            "application/vnd.apache.parquet"
        )

        if not upload_success:
            logger.error(f"Failed to upload batch {batch_num} for {table_name}")
            await log_archival(client, table_name, total_archived, total_deleted,
                             storage_paths, total_size, "partial" if total_archived > 0 else "failed",
                             f"Upload failed at batch {batch_num}")
            return {"table": table_name, "archived": total_archived, "deleted": total_deleted,
                    "storage_paths": storage_paths, "error": "upload_failed"}

        storage_paths.append(storage_path)
        total_size += file_size

        # Delete archived records
        record_ids = [r["id"] for r in records if "id" in r]
        if record_ids:
            deleted = await delete_records(client, table_name, record_ids)
            total_deleted += deleted

        total_archived += len(records)
        logger.info(f"Progress: {total_archived}/{total_count} records archived")

        # Check if last batch
        if len(records) < 1000:
            logger.info(f"Last batch processed for {table_name}")
            break

        # Delay between batches
        await asyncio.sleep(BATCH_DELAY)

    # Log final operation
    await log_archival(client, table_name, total_archived, total_deleted,
                      storage_paths, total_size, "completed")

    logger.info(f"Archival complete for {table_name}: {total_archived} archived, {total_deleted} deleted")

    return {
        "table": table_name,
        "archived": total_archived,
        "deleted": total_deleted,
        "storage_paths": storage_paths,
        "total_size_bytes": total_size,
    }


async def main(tables: list = None, before_date: datetime = None):
    """Run archival for specified tables (or all tables).

    Args:
        tables: List of table names to archive (default: all)
        before_date: Archive records before this date (default: use retention settings)
    """
    settings = get_settings()
    client = get_supabase_admin_client()

    # Check if archival is enabled
    if not settings.archive_enabled:
        logger.warning("Archival is disabled in settings (archive_enabled=False)")
        logger.warning("Set archive_enabled=True in .env to enable archival")
        return

    print(f"\n{'='*60}")
    print("MANUAL DATA ARCHIVAL")
    print(f"{'='*60}")
    print(f"Started at: {datetime.now(CHINA_TZ).isoformat()}")
    print(f"Archive enabled: {settings.archive_enabled}")
    print(f"{'='*60}\n")

    # Define tables with their retention settings
    all_tables = [
        ("imu", settings.retention_imu_days, TABLES["imu"]),
        ("uploads", settings.retention_uploads_days, TABLES["uploads"]),
        ("har", settings.retention_har_days, TABLES["har"]),
        ("atomic_activities", settings.retention_atomic_days, TABLES["atomic_activities"]),
        ("summary_logs", settings.retention_summary_logs_days, TABLES["summary_logs"]),
        ("interventions", settings.retention_interventions_days, TABLES["interventions"]),
    ]

    # Filter to specified tables if provided
    if tables:
        all_tables = [(t, r, n) for t, r, n in all_tables if t in tables]
        if not all_tables:
            print(f"Error: No valid tables found. Valid tables: {list(TABLES.keys())}")
            return

    results = {}

    for table_name, retention_days, _ in all_tables:
        try:
            results[table_name] = await archive_table(
                client=client,
                bucket=settings.storage_bucket,
                table_name=table_name,
                retention_days=retention_days,
                before_date=before_date,
            )
        except Exception as e:
            logger.error(f"Failed to archive {table_name}: {e}")
            results[table_name] = {"error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("ARCHIVAL SUMMARY")
    print(f"{'='*60}")

    for table_name, result in results.items():
        if "error" in result:
            print(f"{table_name}: ❌ Error - {result['error']}")
        else:
            print(f"{table_name}: ✅ {result.get('archived', 0)} records archived, "
                  f"{result.get('deleted', 0)} deleted")
            if result.get("storage_paths"):
                for path in result["storage_paths"][:3]:
                    print(f"         Storage: {path}")
                if len(result["storage_paths"]) > 3:
                    print(f"         ... and {len(result['storage_paths']) - 3} more files")

    print(f"\nCompleted at: {datetime.now(CHINA_TZ).isoformat()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manually archive old data to Supabase Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Archive all tables using retention settings
    python -m scripts.manual_archive

    # Archive specific table
    python -m scripts.manual_archive --table imu

    # Archive before specific date
    python -m scripts.manual_archive --before 2026-03-01

    # Archive specific table before date
    python -m scripts.manual_archive --table har --before 2026-03-01
        """
    )
    parser.add_argument(
        "--table", "-t",
        help="Specific table to archive (default: all tables)",
        choices=list(TABLES.keys()),
    )
    parser.add_argument(
        "--before", "-b",
        help="Archive records before this date (YYYY-MM-DD)",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=CHINA_TZ),
    )
    parser.add_argument(
        "--list-tables", "-l",
        action="store_true",
        help="List available tables and exit",
    )

    args = parser.parse_args()

    if args.list_tables:
        print("\nAvailable tables:")
        for name, desc in TABLES.items():
            print(f"  {name}: {desc}")
        sys.exit(0)

    tables = [args.table] if args.table else None

    asyncio.run(main(tables=tables, before_date=args.before))