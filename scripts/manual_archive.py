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


async def archive_table(table_name: str, before_date: datetime | None = None):
    """Archive records before specified date (default: today).

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

    # Step 1: Count records
    count_response = await asyncio.to_thread(
        lambda: client.table(table_name)
        .select("id", count="exact")
        .lt("timestamp", cutoff_iso)
        .execute()
    )

    total_count = getattr(count_response, 'count', 0)
    logger.info(f"Records to archive: {total_count}")

    if total_count == 0:
        logger.info("No records to archive.")
        return {"table": table_name, "archived": 0, "deleted": 0}

    # Step 2: Fetch records in batches
    batch_size = 10000
    all_records = []
    offset = 0

    while len(all_records) < total_count:
        response = await asyncio.to_thread(
            lambda o=offset: client.table(table_name)
            .select("*")
            .lt("timestamp", cutoff_iso)
            .order("timestamp", desc=False)
            .range(o, o + batch_size - 1)
            .execute()
        )

        if not response.data:
            break

        all_records.extend(response.data)
        offset += batch_size
        logger.info(f"Fetched {len(all_records)}/{total_count} records...")

        if len(response.data) < batch_size:
            break

    logger.info(f"Total records fetched: {len(all_records)}")

    if not all_records:
        return {"table": table_name, "archived": 0, "deleted": 0}

    # Step 3: Convert to Parquet
    table = pa.Table.from_pylist(all_records)
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

    logger.info(f"Parquet size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

    # Step 4: Generate storage path
    oldest_record = all_records[0]
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
    storage_path = f"archives/{table_name}/{year}/{month}/{day}.parquet"

    logger.info(f"Storage path: {storage_path}")

    # Step 5: Upload to Supabase Storage
    bucket = settings.storage_bucket

    try:
        await asyncio.to_thread(
            lambda: client.storage.from_(bucket).upload(
                path=storage_path,
                file=parquet_bytes,
                file_options={"content-type": "application/vnd.apache.parquet"},
            )
        )
        logger.info(f"✅ Uploaded to {bucket}/{storage_path}")
    except Exception as e:
        error_str = str(e).lower()
        if "already exists" in error_str or "duplicate" in error_str:
            # Update existing file
            await asyncio.to_thread(
                lambda: client.storage.from_(bucket).update(
                    path=storage_path,
                    file=parquet_bytes,
                    file_options={"content-type": "application/vnd.apache.parquet"},
                )
            )
            logger.info(f"✅ Updated {bucket}/{storage_path}")
        else:
            raise

    # Step 6: Delete archived records
    record_ids = [r["id"] for r in all_records if "id" in r]

    if record_ids:
        # Delete in batches to avoid query size limits
        delete_batch_size = 5000
        total_deleted = 0

        for i in range(0, len(record_ids), delete_batch_size):
            batch_ids = record_ids[i:i + delete_batch_size]
            await asyncio.to_thread(
                lambda ids=batch_ids: client.table(table_name)
                .delete()
                .in_("id", ids)
                .execute()
            )
            total_deleted += len(batch_ids)
            logger.info(f"Deleted {total_deleted}/{len(record_ids)} records...")

        logger.info(f"✅ Deleted {total_deleted} records from {table_name}")

    # Step 7: Log to archival_logs table
    await asyncio.to_thread(
        lambda: client.table("archival_logs").insert({
            "table_name": table_name,
            "records_archived": len(all_records),
            "records_deleted": len(record_ids),
            "storage_path": storage_path,
            "file_size_bytes": file_size,
            "status": "completed",
        }).execute()
    )
    logger.info(f"✅ Logged archival operation")

    return {
        "table": table_name,
        "archived": len(all_records),
        "deleted": len(record_ids),
        "storage_path": storage_path,
        "file_size_bytes": file_size,
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