"""Manual archival script for imu and uploads tables.

Usage:
    cd /Users/larry/Desktop/MobiBox/MobiBox_Server
    python -m scripts.manual_archive

Or run directly:
    python scripts/manual_archive.py
"""

import asyncio
import io
import logging
import random
from datetime import datetime
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

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 60.0  # seconds
BATCH_DELAY = 1.0  # delay between successful batches


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (transient)."""
    error_str = str(error).lower()
    # 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout
    # Connection errors, rate limiting
    retryable_codes = ['502', '503', '504', '429', 'timeout', 'connection', 'rate']
    return any(code in error_str for code in retryable_codes)


async def retry_with_backoff(func, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """Execute a function with exponential backoff retry logic."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if not is_retryable_error(e):
                raise  # Non-retryable error, fail immediately

            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff and jitter
                delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                logger.warning(f"Retryable error on attempt {attempt + 1}/{max_retries}: {e}")
                logger.info(f"Waiting {delay:.1f} seconds before retry...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Max retries ({max_retries}) exceeded")
                raise

    raise last_error


async def fetch_batch(client, table_name: str, cutoff_iso: str, batch_size: int):
    """Fetch a batch of records with retry logic."""
    async def _fetch():
        return await asyncio.to_thread(
            lambda: client.table(table_name)
            .select("*")
            .lt("timestamp", cutoff_iso)
            .order("timestamp", desc=False)
            .limit(batch_size)
            .execute()
        )
    return await retry_with_backoff(_fetch)


async def count_records(client, table_name: str, cutoff_iso: str):
    """Count records with retry logic."""
    async def _count():
        return await asyncio.to_thread(
            lambda: client.table(table_name)
            .select("id", count="exact")
            .lt("timestamp", cutoff_iso)
            .execute()
        )
    return await retry_with_backoff(_count)


async def delete_records(client, table_name: str, record_ids: list):
    """Delete records with retry logic."""
    async def _delete():
        return await asyncio.to_thread(
            lambda: client.table(table_name)
            .delete()
            .in_("id", record_ids)
            .execute()
        )
    return await retry_with_backoff(_delete)


async def upload_to_storage(client, bucket: str, path: str, content: bytes, content_type: str):
    """Upload to storage with retry logic."""
    async def _upload():
        return await asyncio.to_thread(
            lambda: client.storage.from_(bucket).upload(
                path=path,
                file=content,
                file_options={"content-type": content_type},
            )
        )
    return await retry_with_backoff(_upload)


async def update_storage(client, bucket: str, path: str, content: bytes, content_type: str):
    """Update storage file with retry logic."""
    async def _update():
        return await asyncio.to_thread(
            lambda: client.storage.from_(bucket).update(
                path=path,
                file=content,
                file_options={"content-type": content_type},
            )
        )
    return await retry_with_backoff(_update)


async def archive_table(table_name: str, before_date: datetime | None = None):
    """Archive records before specified date (default: today).

    This function processes records in batches, handling Supabase's 1000 record limit
    per request. It archives all records before the cutoff date in a loop.

    Args:
        table_name: Name of the table ('imu' or 'uploads')
        before_date: Archive records before this date (default: start of today)

    Returns:
        Dict with archival statistics
    """
    settings = get_settings()
    client = get_supabase_admin_client()

    # Default: archive everything before today
    if before_date is None:
        before_date = datetime.now(CHINA_TZ).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    cutoff_iso = before_date.isoformat()

    logger.info(f"{'='*60}")
    logger.info(f"Archiving table: {table_name}")
    logger.info(f"Cutoff timestamp: {cutoff_iso}")
    logger.info(f"{'='*60}")

    # Step 1: Count records with retry
    count_response = await count_records(client, table_name, cutoff_iso)

    total_count = getattr(count_response, 'count', 0)
    logger.info(f"Records to archive: {total_count}")

    if total_count == 0:
        logger.info("No records to archive.")
        return {"table": table_name, "archived": 0, "deleted": 0}

    # Track totals across all batches
    total_archived = 0
    total_deleted = 0
    total_size = 0
    storage_paths = []

    # Step 2: Process in batches (Supabase max limit is 1000)
    batch_size = 1000
    batch_num = 0

    while True:
        batch_num += 1
        logger.info(f"\n--- Processing batch {batch_num} ---")

        # Fetch one batch with retry logic
        response = await fetch_batch(client, table_name, cutoff_iso, batch_size)

        if not response.data:
            logger.info("No more records to process.")
            break

        batch_records = response.data
        logger.info(f"Fetched {len(batch_records)} records in batch {batch_num}")

        # Step 3: Convert to Parquet
        table = pa.Table.from_pylist(batch_records)
        buffer = io.BytesIO()
        pq.write_table(
            table,
            buffer,
            compression="snappy",
            use_dictionary=True,
            write_statistics=True,
        )
        parquet_bytes = buffer.getvalue()
        file_size = len(parquet_bytes)

        # Step 4: Generate storage path with batch number
        oldest_record = batch_records[0]
        if "timestamp" in oldest_record and oldest_record["timestamp"]:
            ts_str = oldest_record["timestamp"]
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            record_date = datetime.fromisoformat(ts_str)
        else:
            record_date = datetime.now(CHINA_TZ)

        year = record_date.strftime("%Y")
        month = record_date.strftime("%m")
        day = record_date.strftime("%Y-%m-%d")
        # Add batch number to filename to avoid overwriting
        storage_path = f"archives/{table_name}/{year}/{month}/{day}_batch{batch_num:04d}.parquet"

        logger.info(f"Storage path: {storage_path}")
        logger.info(f"Parquet size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

        # Step 5: Upload to Supabase Storage with retry
        bucket = settings.storage_bucket

        try:
            await upload_to_storage(
                client, bucket, storage_path, parquet_bytes,
                "application/vnd.apache.parquet"
            )
            logger.info(f"✅ Uploaded to {bucket}/{storage_path}")
        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str or "duplicate" in error_str:
                await update_storage(
                    client, bucket, storage_path, parquet_bytes,
                    "application/vnd.apache.parquet"
                )
                logger.info(f"✅ Updated {bucket}/{storage_path}")
            else:
                raise

        storage_paths.append(storage_path)
        total_size += file_size

        # Step 6: Delete archived records with retry
        record_ids = [r["id"] for r in batch_records if "id" in r]

        if record_ids:
            await delete_records(client, table_name, record_ids)
            total_deleted += len(record_ids)
            logger.info(f"✅ Deleted {len(record_ids)} records from {table_name}")

        total_archived += len(batch_records)
        logger.info(f"Progress: {total_archived}/{total_count} records archived")

        # Check if we got fewer records than requested (last batch)
        if len(batch_records) < batch_size:
            logger.info("Last batch processed.")
            break

        # Delay between batches to avoid rate limiting
        await asyncio.sleep(BATCH_DELAY)

    # Step 7: Log to archival_logs table with retry
    async def _log():
        return await asyncio.to_thread(
            lambda: client.table("archival_logs").insert({
                "table_name": table_name,
                "records_archived": total_archived,
                "records_deleted": total_deleted,
                "storage_path": storage_paths[0] if storage_paths else None,
                "file_size_bytes": total_size,
                "status": "completed",
            }).execute()
        )
    await retry_with_backoff(_log)
    logger.info(f"✅ Logged archival operation")

    return {
        "table": table_name,
        "archived": total_archived,
        "deleted": total_deleted,
        "storage_paths": storage_paths,
        "total_size_bytes": total_size,
    }


async def main():
    """Run archival for imu and uploads tables."""
    print(f"\n{'='*60}")
    print("MANUAL DATA ARCHIVAL")
    print(f"{'='*60}")
    print(f"Started at: {datetime.now(CHINA_TZ).isoformat()}\n")

    results = {}

    # Archive IMU table
    try:
        results["imu"] = await archive_table("imu")
    except Exception as e:
        logger.error(f"Failed to archive imu: {e}")
        results["imu"] = {"error": str(e)}

    # Archive uploads table
    try:
        results["uploads"] = await archive_table("uploads")
    except Exception as e:
        logger.error(f"Failed to archive uploads: {e}")
        results["uploads"] = {"error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("ARCHIVAL SUMMARY")
    print(f"{'='*60}")

    for table, result in results.items():
        if "error" in result:
            print(f"{table}: ❌ Error - {result['error']}")
        else:
            print(f"{table}: ✅ {result.get('archived', 0)} records archived, "
                  f"{result.get('deleted', 0)} deleted")
            if result.get("storage_path"):
                print(f"         Storage: {result['storage_path']}")

    print(f"\nCompleted at: {datetime.now(CHINA_TZ).isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())