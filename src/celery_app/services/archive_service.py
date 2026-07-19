"""Service for archiving old data to local filesystem using Parquet format."""

import io
import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from bson import ObjectId

from src.config import get_settings
from src.database import get_database

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")

# Collection names
IMU_COLLECTION = "imu"
HAR_COLLECTION = "har"
ATOMIC_ACTIVITIES_COLLECTION = "atomic_activities"
UPLOADS_COLLECTION = "uploads"
SUMMARY_LOGS_COLLECTION = "summary_logs"
INTERVENTIONS_COLLECTION = "interventions"
ARCHIVAL_LOGS_COLLECTION = "archival_logs"


def _records_to_parquet(records: list[dict]) -> bytes:
    """Convert records to compressed Parquet format."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        return b""

    table = pa.Table.from_pylist(records)

    buffer = io.BytesIO()
    pq.write_table(
        table, buffer,
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
    )

    return buffer.getvalue()


class ArchiveService:
    """Service for archiving database records to local filesystem."""

    def __init__(self):
        """Initialize archive service."""
        self.settings = get_settings()

    def _get_storage_path(self, collection_name: str, date: datetime) -> str:
        """Generate storage path for archived data.

        Returns:
            Storage path relative to archive_dir:
            {archive_dir}/{collection_name}/{year}/{month}/{filename}.parquet
        """
        year = date.strftime("%Y")
        month = date.strftime("%m")
        filename = f"{date.strftime('%Y-%m-%d')}.parquet"
        return f"{collection_name}/{year}/{month}/{filename}"

    async def fetch_records_for_archival(
        self,
        collection_name: str,
        retention_days: int,
        batch_size: int,
    ) -> list[dict]:
        """Fetch records older than retention period."""
        db = await get_database()
        cutoff_date = datetime.now(CHINA_TZ) - timedelta(days=retention_days)

        cursor = db[collection_name].find({
            "timestamp": {"$lt": cutoff_date},
        }).sort("timestamp", 1).limit(batch_size)

        return await cursor.to_list(None)

    def _ensure_archive_dir(self, path: str):
        """Ensure the archive directory exists."""
        full_path = os.path.join(self.settings.archive_dir, os.path.dirname(path))
        os.makedirs(full_path, exist_ok=True)

    async def upload_to_storage(
        self,
        path: str,
        content: bytes,
    ) -> bool:
        """Save content to local filesystem.

        Args:
            path: Relative file path
            content: File content as bytes

        Returns:
            True if successful
        """
        try:
            self._ensure_archive_dir(path)
            full_path = os.path.join(self.settings.archive_dir, path)
            with open(full_path, "wb") as f:
                f.write(content)
            logger.info(f"Saved archive to {full_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save archive {path}: {e}")
            return False

    async def delete_archived_records(
        self,
        collection_name: str,
        record_ids: list[ObjectId],
    ) -> int:
        """Delete archived records from database.

        Args:
            collection_name: Name of the collection
            record_ids: List of MongoDB ObjectIds to delete

        Returns:
            Number of records deleted
        """
        if not record_ids:
            return 0

        db = await get_database()
        try:
            result = await db[collection_name].delete_many(
                {"_id": {"$in": record_ids}}
            )
            logger.info(f"Deleted {result.deleted_count} records from {collection_name}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to delete records from {collection_name}: {e}")
            return 0

    async def log_archival_operation(
        self,
        table_name: str,
        records_archived: int,
        records_deleted: int,
        storage_path: Optional[str],
        file_size_bytes: Optional[int],
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Log archival operation to database for audit trail."""
        db = await get_database()
        try:
            await db[ARCHIVAL_LOGS_COLLECTION].insert_one({
                "table_name": table_name,
                "records_archived": records_archived,
                "records_deleted": records_deleted,
                "storage_path": storage_path,
                "file_size_bytes": file_size_bytes,
                "archival_timestamp": datetime.now(CHINA_TZ),
                "status": status,
                "error_message": error_message,
                "created_at": datetime.now(CHINA_TZ),
            })
            logger.debug(f"Logged archival operation for {table_name}")
        except Exception as e:
            logger.error(f"Failed to log archival operation: {e}")

    async def archive_collection(
        self,
        collection_name: str,
        retention_days: int,
        batch_size: int,
    ) -> dict:
        """Archive old records from a collection to local filesystem and delete them.

        Returns:
            Dictionary with archival statistics
        """
        if not self.settings.archive_enabled:
            logger.info(f"Archival disabled, skipping {collection_name}")
            return {"table": collection_name, "archived": 0, "deleted": 0, "skipped": True}

        logger.info(f"Starting archival for {collection_name} (retention: {retention_days} days)")

        records = await self.fetch_records_for_archival(
            collection_name=collection_name,
            retention_days=retention_days,
            batch_size=batch_size,
        )

        if not records:
            logger.info(f"No records to archive for {collection_name}")
            await self.log_archival_operation(
                table_name=collection_name,
                records_archived=0,
                records_deleted=0,
                storage_path=None,
                file_size_bytes=None,
                status="completed",
            )
            return {"table": collection_name, "archived": 0, "deleted": 0}

        # Convert to Parquet format
        try:
            parquet_bytes = _records_to_parquet(records)
            file_size = len(parquet_bytes)
            logger.info(
                f"Converted {len(records)} records to Parquet ({file_size} bytes) for {collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to convert records to Parquet: {e}")
            await self.log_archival_operation(
                table_name=collection_name,
                records_archived=len(records),
                records_deleted=0,
                storage_path=None,
                file_size_bytes=None,
                status="failed",
                error_message=f"Parquet conversion failed: {str(e)}",
            )
            return {"table": collection_name, "archived": 0, "deleted": 0, "error": "parquet_conversion_failed"}

        # Generate storage path
        if records and "timestamp" in records[0]:
            ts = records[0]["timestamp"]
            record_date = ts if isinstance(ts, datetime) else datetime.now(CHINA_TZ)
        else:
            record_date = datetime.now(CHINA_TZ)

        storage_path = self._get_storage_path(collection_name, record_date)

        # Save to local filesystem
        success = await self.upload_to_storage(path=storage_path, content=parquet_bytes)

        if not success:
            await self.log_archival_operation(
                table_name=collection_name,
                records_archived=len(records),
                records_deleted=0,
                storage_path=storage_path,
                file_size_bytes=file_size,
                status="failed",
                error_message="Failed to save archive file",
            )
            return {"table": collection_name, "archived": 0, "deleted": 0, "error": "upload_failed"}

        # Delete archived records
        record_ids = [r["_id"] for r in records if "_id" in r]
        deleted_count = await self.delete_archived_records(
            collection_name=collection_name,
            record_ids=record_ids,
        )

        await self.log_archival_operation(
            table_name=collection_name,
            records_archived=len(records),
            records_deleted=deleted_count,
            storage_path=storage_path,
            file_size_bytes=file_size,
            status="completed",
        )

        return {
            "table": collection_name,
            "archived": len(records),
            "deleted": deleted_count,
            "storage_path": storage_path,
            "file_size_bytes": file_size,
        }

    async def archive_all_tables(self) -> dict:
        """Archive all configured collections."""
        results = {}
        batch_size = self.settings.archive_batch_size

        results["imu"] = await self.archive_collection(
            IMU_COLLECTION, self.settings.retention_imu_days, batch_size,
        )
        results["har"] = await self.archive_collection(
            HAR_COLLECTION, self.settings.retention_har_days, batch_size,
        )
        results["atomic_activities"] = await self.archive_collection(
            ATOMIC_ACTIVITIES_COLLECTION, self.settings.retention_atomic_days, batch_size,
        )
        results["uploads"] = await self.archive_collection(
            UPLOADS_COLLECTION, self.settings.retention_uploads_days, batch_size,
        )
        results["summary_logs"] = await self.archive_collection(
            SUMMARY_LOGS_COLLECTION, self.settings.retention_summary_logs_days, batch_size,
        )
        results["interventions"] = await self.archive_collection(
            INTERVENTIONS_COLLECTION, self.settings.retention_interventions_days, batch_size,
        )

        return results


async def run_archival() -> dict:
    """Run archival process for all tables."""
    service = ArchiveService()
    return await service.archive_all_tables()
