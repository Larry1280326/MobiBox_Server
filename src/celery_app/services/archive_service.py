"""Service for archiving old data to Supabase Storage using Parquet format."""

import asyncio
import io
import logging
import random
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from supabase import Client

from src.config import get_settings
from src.database import get_supabase_admin_client

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")

# Table names
IMU_TABLE = "imu"
HAR_TABLE = "har"
ATOMIC_ACTIVITIES_TABLE = "atomic_activities"
UPLOADS_TABLE = "uploads"
SUMMARY_LOGS_TABLE = "summary_logs"
INTERVENTIONS_TABLE = "interventions"
INTERVENTION_FEEDBACKS_TABLE = "intervention_feedbacks"
SUMMARY_LOG_FEEDBACKS_TABLE = "summary_log_feedbacks"
ARCHIVAL_LOGS_TABLE = "archival_logs"

# Supabase REST API has a maximum of 1000 records per request
SUPABASE_MAX_LIMIT = 1000

# Retry configuration
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


async def retry_with_backoff(func, *args, **kwargs):
    """Execute a function with exponential backoff retry logic.

    Retries indefinitely until success for retryable errors.
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


def _records_to_parquet(records: list[dict]) -> bytes:
    """Convert records to compressed Parquet format.

    Uses PyArrow with Snappy compression for optimal storage efficiency.
    Parquet provides ~10-100x compression over CSV while preserving types.

    Args:
        records: List of record dictionaries

    Returns:
        Parquet file as bytes
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        return b""

    # Infer schema from data
    # Group all records to build schema
    table = pa.Table.from_pylist(records)

    # Write to buffer with Snappy compression
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        compression="snappy",  # Fast compression with good ratio
        use_dictionary=True,  # Dictionary encoding for strings
        write_statistics=True,  # Enable statistics for query optimization
    )

    return buffer.getvalue()


class ArchiveService:
    """Service for archiving database records to Supabase Storage."""

    def __init__(self, client: Optional[Client] = None):
        """Initialize archive service.

        Args:
            client: Optional Supabase admin client (uses service role for storage access)
        """
        self.client = client or get_supabase_admin_client()
        self.settings = get_settings()

    def _get_storage_path(self, table_name: str, date: datetime, batch_num: int = 0) -> str:
        """Generate storage path for archived data.

        Args:
            table_name: Name of the table being archived
            date: Date for the archive file
            batch_num: Batch number (0 for single file, >0 for batched files)

        Returns:
            Storage path in format: archives/{table_name}/{year}/{month}/{filename}.parquet
        """
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%Y-%m-%d")

        if batch_num > 0:
            filename = f"{day}_batch{batch_num:04d}.parquet"
        else:
            filename = f"{day}.parquet"

        return f"archives/{table_name}/{year}/{month}/{filename}"

    async def count_records_for_archival(
        self,
        table_name: str,
        retention_days: int,
    ) -> int:
        """Count records older than retention period.

        Args:
            table_name: Name of the table
            retention_days: Number of days to keep data

        Returns:
            Number of records to archive
        """
        cutoff_date = datetime.now(CHINA_TZ) - timedelta(days=retention_days)
        cutoff_iso = cutoff_date.isoformat()

        async def _count():
            return await asyncio.to_thread(
                lambda: self.client.table(table_name)
                .select("id", count="exact")
                .lt("timestamp", cutoff_iso)
                .execute()
            )

        response = await retry_with_backoff(_count)
        return getattr(response, 'count', 0) if response else 0

    async def fetch_records_batch(
        self,
        table_name: str,
        retention_days: int,
        columns: Optional[list[str]] = None,
    ) -> list[dict]:
        """Fetch a single batch of records older than retention period.

        Note: Supabase REST API has a maximum limit of 1000 records per request.

        Args:
            table_name: Name of the table
            retention_days: Number of days to keep data
            columns: Optional list of columns to select (default: all)

        Returns:
            List of records to archive (max 1000)
        """
        cutoff_date = datetime.now(CHINA_TZ) - timedelta(days=retention_days)
        cutoff_iso = cutoff_date.isoformat()

        select_columns = ", ".join(columns) if columns else "*"

        async def _fetch():
            return await asyncio.to_thread(
                lambda: self.client.table(table_name)
                .select(select_columns)
                .lt("timestamp", cutoff_iso)
                .order("timestamp", desc=False)
                .limit(SUPABASE_MAX_LIMIT)
                .execute()
            )

        response = await retry_with_backoff(_fetch)
        return response.data if response.data else []

    async def upload_to_storage(
        self,
        bucket: str,
        path: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Upload content to Supabase Storage.

        Args:
            bucket: Storage bucket name
            path: File path in storage
            content: File content as bytes
            content_type: MIME type

        Returns:
            True if upload successful
        """
        try:
            async def _upload():
                return await asyncio.to_thread(
                    lambda: self.client.storage.from_(bucket).upload(
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
            # Handle case where file already exists (update instead)
            if "already exists" in error_str or "duplicate" in error_str:
                try:
                    async def _update():
                        return await asyncio.to_thread(
                            lambda: self.client.storage.from_(bucket).update(
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

    async def delete_archived_records(
        self,
        table_name: str,
        record_ids: list[int],
        id_column: str = "id",
    ) -> int:
        """Delete archived records from database.

        Args:
            table_name: Name of the table
            record_ids: List of record IDs to delete
            id_column: Name of the ID column

        Returns:
            Number of records deleted
        """
        if not record_ids:
            return 0

        try:
            async def _delete():
                return await asyncio.to_thread(
                    lambda: self.client.table(table_name)
                    .delete()
                    .in_(id_column, record_ids)
                    .execute()
                )
            await retry_with_backoff(_delete)
            logger.info(f"Deleted {len(record_ids)} records from {table_name}")
            return len(record_ids)

        except Exception as e:
            logger.error(f"Failed to delete records from {table_name}: {e}")
            return 0

    async def log_archival_operation(
        self,
        table_name: str,
        records_archived: int,
        records_deleted: int,
        storage_paths: list[str],
        total_size_bytes: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Log archival operation to database for audit trail.

        Args:
            table_name: Name of the table archived
            records_archived: Number of records archived
            records_deleted: Number of records deleted
            storage_paths: List of storage paths (one per batch)
            total_size_bytes: Total size of all archive files in bytes
            status: Status of archival (completed, failed, partial)
            error_message: Error message if failed
        """
        try:
            async def _log():
                return await asyncio.to_thread(
                    lambda: self.client.table(ARCHIVAL_LOGS_TABLE)
                    .insert({
                        "table_name": table_name,
                        "records_archived": records_archived,
                        "records_deleted": records_deleted,
                        "storage_path": storage_paths[0] if storage_paths else None,
                        "file_size_bytes": total_size_bytes,
                        "status": status,
                        "error_message": error_message,
                    })
                    .execute()
                )
            await retry_with_backoff(_log)
            logger.debug(f"Logged archival operation for {table_name}")
        except Exception as e:
            logger.error(f"Failed to log archival operation: {e}")

    async def archive_table(
        self,
        table_name: str,
        retention_days: int,
        batch_size: int = 10000,  # Unused, kept for API compatibility
        columns: Optional[list[str]] = None,
        id_column: str = "id",
    ) -> dict:
        """Archive old records from a table to storage and delete them.

        This method processes records in batches of 1000 (Supabase's max limit)
        and archives all records older than the retention period.

        Args:
            table_name: Name of the table
            retention_days: Number of days to keep data
            batch_size: Unused (kept for API compatibility)
            columns: Columns to archive (default: all)
            id_column: Name of the ID column

        Returns:
            Dictionary with archival statistics
        """
        if not self.settings.archive_enabled:
            logger.info(f"Archival disabled, skipping {table_name}")
            return {"table": table_name, "archived": 0, "deleted": 0, "skipped": True}

        logger.info(f"Starting archival for {table_name} (retention: {retention_days} days)")

        # Count total records to archive
        total_count = await self.count_records_for_archival(
            table_name=table_name,
            retention_days=retention_days,
        )

        logger.info(f"Records to archive for {table_name}: {total_count}")

        if total_count == 0:
            logger.info(f"No records to archive for {table_name}")
            await self.log_archival_operation(
                table_name=table_name,
                records_archived=0,
                records_deleted=0,
                storage_paths=[],
                total_size_bytes=0,
                status="completed",
            )
            return {"table": table_name, "archived": 0, "deleted": 0}

        # Track totals across all batches
        total_archived = 0
        total_deleted = 0
        total_size = 0
        storage_paths = []
        batch_num = 0

        # Process in batches until all records are archived
        while True:
            batch_num += 1

            # Fetch one batch
            records = await self.fetch_records_batch(
                table_name=table_name,
                retention_days=retention_days,
                columns=columns,
            )

            if not records:
                logger.info(f"No more records to process for {table_name}")
                break

            logger.info(f"Processing batch {batch_num} for {table_name}: {len(records)} records")

            # Convert to Parquet
            try:
                parquet_bytes = _records_to_parquet(records)
                file_size = len(parquet_bytes)
            except Exception as e:
                logger.error(f"Failed to convert batch {batch_num} to Parquet: {e}")
                await self.log_archival_operation(
                    table_name=table_name,
                    records_archived=total_archived,
                    records_deleted=total_deleted,
                    storage_paths=storage_paths,
                    total_size_bytes=total_size,
                    status="partial" if total_archived > 0 else "failed",
                    error_message=f"Parquet conversion failed at batch {batch_num}: {str(e)}",
                )
                return {
                    "table": table_name,
                    "archived": total_archived,
                    "deleted": total_deleted,
                    "storage_paths": storage_paths,
                    "error": "parquet_conversion_failed",
                }

            # Generate storage path with batch number
            oldest_record = records[0]
            if "timestamp" in oldest_record and oldest_record["timestamp"]:
                ts_str = oldest_record["timestamp"]
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                record_date = datetime.fromisoformat(ts_str)
            else:
                record_date = datetime.now(CHINA_TZ)

            storage_path = self._get_storage_path(table_name, record_date, batch_num)

            # Upload to storage
            upload_success = await self.upload_to_storage(
                bucket=self.settings.storage_bucket,
                path=storage_path,
                content=parquet_bytes,
                content_type="application/vnd.apache.parquet",
            )

            if not upload_success:
                logger.error(f"Failed to upload batch {batch_num} for {table_name}")
                await self.log_archival_operation(
                    table_name=table_name,
                    records_archived=total_archived,
                    records_deleted=total_deleted,
                    storage_paths=storage_paths,
                    total_size_bytes=total_size,
                    status="partial" if total_archived > 0 else "failed",
                    error_message=f"Upload failed at batch {batch_num}",
                )
                return {
                    "table": table_name,
                    "archived": total_archived,
                    "deleted": total_deleted,
                    "storage_paths": storage_paths,
                    "error": "upload_failed",
                }

            storage_paths.append(storage_path)
            total_size += file_size

            # Delete archived records
            record_ids = [r[id_column] for r in records if id_column in r]
            if record_ids:
                deleted = await self.delete_archived_records(
                    table_name=table_name,
                    record_ids=record_ids,
                    id_column=id_column,
                )
                total_deleted += deleted

            total_archived += len(records)
            logger.info(f"Progress for {table_name}: {total_archived}/{total_count} records archived")

            # Check if we got fewer records than max (last batch)
            if len(records) < SUPABASE_MAX_LIMIT:
                logger.info(f"Last batch processed for {table_name}")
                break

            # Delay between batches to avoid rate limiting
            await asyncio.sleep(BATCH_DELAY)

        # Log final archival operation
        await self.log_archival_operation(
            table_name=table_name,
            records_archived=total_archived,
            records_deleted=total_deleted,
            storage_paths=storage_paths,
            total_size_bytes=total_size,
            status="completed",
        )

        logger.info(f"Archival complete for {table_name}: {total_archived} records archived, {total_deleted} deleted")

        return {
            "table": table_name,
            "archived": total_archived,
            "deleted": total_deleted,
            "storage_paths": storage_paths,
            "total_size_bytes": total_size,
        }

    async def archive_all_tables(self) -> dict:
        """Archive all configured tables.

        Returns:
            Dictionary with archival statistics for all tables
        """
        results = {}

        # Archive IMU data (highest volume, shortest retention)
        results["imu"] = await self.archive_table(
            table_name=IMU_TABLE,
            retention_days=self.settings.retention_imu_days,
        )

        # Archive HAR labels
        results["har"] = await self.archive_table(
            table_name=HAR_TABLE,
            retention_days=self.settings.retention_har_days,
        )

        # Archive atomic activities
        results["atomic_activities"] = await self.archive_table(
            table_name=ATOMIC_ACTIVITIES_TABLE,
            retention_days=self.settings.retention_atomic_days,
        )

        # Archive uploads
        results["uploads"] = await self.archive_table(
            table_name=UPLOADS_TABLE,
            retention_days=self.settings.retention_uploads_days,
        )

        # Archive summary logs (longer retention)
        results["summary_logs"] = await self.archive_table(
            table_name=SUMMARY_LOGS_TABLE,
            retention_days=self.settings.retention_summary_logs_days,
        )

        # Archive interventions
        results["interventions"] = await self.archive_table(
            table_name=INTERVENTIONS_TABLE,
            retention_days=self.settings.retention_interventions_days,
        )

        return results


async def run_archival() -> dict:
    """Run archival process for all tables.

    Returns:
        Dictionary with archival statistics
    """
    service = ArchiveService()
    return await service.archive_all_tables()