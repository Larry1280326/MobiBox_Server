"""Service for archiving old data to Supabase Storage."""

import csv
import io
import json
import logging
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


class ArchiveService:
    """Service for archiving database records to Supabase Storage."""

    def __init__(self, client: Optional[Client] = None):
        """Initialize archive service.

        Args:
            client: Optional Supabase admin client (uses service role for storage access)
        """
        self.client = client or get_supabase_admin_client()
        self.settings = get_settings()

    def _get_storage_path(self, table_name: str, date: datetime) -> str:
        """Generate storage path for archived data.

        Args:
            table_name: Name of the table being archived
            date: Date for the archive file

        Returns:
            Storage path in format: archives/{table_name}/{year}/{month}/{filename}
        """
        year = date.strftime("%Y")
        month = date.strftime("%m")
        filename = f"{date.strftime('%Y-%m-%d')}.csv"
        return f"archives/{table_name}/{year}/{month}/{filename}"

    def _records_to_csv(self, records: list[dict], columns: list[str]) -> str:
        """Convert records to CSV string.

        Args:
            records: List of record dictionaries
            columns: Column names to include

        Returns:
            CSV string
        """
        if not records:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for record in records:
            # Convert datetime objects to ISO format
            row = {}
            for col in columns:
                value = record.get(col)
                if isinstance(value, datetime):
                    row[col] = value.isoformat()
                else:
                    row[col] = value
            writer.writerow(row)

        return output.getvalue()

    async def fetch_records_for_archival(
        self,
        table_name: str,
        retention_days: int,
        batch_size: int,
        columns: Optional[list[str]] = None,
    ) -> list[dict]:
        """Fetch records older than retention period.

        Args:
            table_name: Name of the table
            retention_days: Number of days to keep data
            batch_size: Maximum number of records to fetch
            columns: Optional list of columns to select (default: all)

        Returns:
            List of records to archive
        """
        cutoff_date = datetime.now(CHINA_TZ) - timedelta(days=retention_days)
        cutoff_iso = cutoff_date.isoformat()

        select_columns = ", ".join(columns) if columns else "*"

        import asyncio

        response = await asyncio.to_thread(
            lambda: self.client.table(table_name)
            .select(select_columns)
            .lt("timestamp", cutoff_iso)
            .order("timestamp", desc=False)
            .limit(batch_size)
            .execute()
        )

        return response.data if response.data else []

    async def upload_to_storage(
        self,
        bucket: str,
        path: str,
        content: str,
        content_type: str = "text/csv",
    ) -> bool:
        """Upload content to Supabase Storage.

        Args:
            bucket: Storage bucket name
            path: File path in storage
            content: File content as string
            content_type: MIME type

        Returns:
            True if upload successful
        """
        import asyncio

        try:
            # Convert string to bytes
            file_bytes = content.encode("utf-8")

            response = await asyncio.to_thread(
                lambda: self.client.storage.from_(bucket).upload(
                    path=path,
                    file=file_bytes,
                    file_options={"content-type": content_type},
                )
            )

            logger.info(f"Uploaded archive to {bucket}/{path}")
            return True

        except Exception as e:
            # Handle case where file already exists (update instead)
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                try:
                    await asyncio.to_thread(
                        lambda: self.client.storage.from_(bucket).update(
                            path=path,
                            file=file_bytes,
                            file_options={"content-type": content_type},
                        )
                    )
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

        import asyncio

        try:
            await asyncio.to_thread(
                lambda: self.client.table(table_name)
                .delete()
                .in_(id_column, record_ids)
                .execute()
            )

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
        storage_path: Optional[str],
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Log archival operation to database for audit trail.

        Args:
            table_name: Name of the table archived
            records_archived: Number of records archived
            records_deleted: Number of records deleted
            storage_path: Path to the archive file
            status: Status of archival (completed, failed, partial)
            error_message: Error message if failed
        """
        import asyncio

        try:
            await asyncio.to_thread(
                lambda: self.client.table(ARCHIVAL_LOGS_TABLE)
                .insert({
                    "table_name": table_name,
                    "records_archived": records_archived,
                    "records_deleted": records_deleted,
                    "storage_path": storage_path,
                    "status": status,
                    "error_message": error_message,
                })
                .execute()
            )
            logger.debug(f"Logged archival operation for {table_name}")
        except Exception as e:
            logger.error(f"Failed to log archival operation: {e}")

    async def archive_table(
        self,
        table_name: str,
        retention_days: int,
        batch_size: int,
        columns: Optional[list[str]] = None,
        id_column: str = "id",
    ) -> dict:
        """Archive old records from a table to storage and delete them.

        Args:
            table_name: Name of the table
            retention_days: Number of days to keep data
            batch_size: Maximum records per batch
            columns: Columns to archive (default: all)
            id_column: Name of the ID column

        Returns:
            Dictionary with archival statistics
        """
        if not self.settings.archive_enabled:
            logger.info(f"Archival disabled, skipping {table_name}")
            return {"table": table_name, "archived": 0, "deleted": 0, "skipped": True}

        logger.info(f"Starting archival for {table_name} (retention: {retention_days} days)")

        # Fetch records to archive
        records = await self.fetch_records_for_archival(
            table_name=table_name,
            retention_days=retention_days,
            batch_size=batch_size,
            columns=columns,
        )

        if not records:
            logger.info(f"No records to archive for {table_name}")
            # Log that no records needed archiving
            await self.log_archival_operation(
                table_name=table_name,
                records_archived=0,
                records_deleted=0,
                storage_path=None,
                status="completed",
            )
            return {"table": table_name, "archived": 0, "deleted": 0}

        # Get column names from first record if not specified
        archive_columns = columns or list(records[0].keys())

        # Convert to CSV
        csv_content = self._records_to_csv(records, archive_columns)

        # Generate storage path
        oldest_record = records[0]
        if "timestamp" in oldest_record:
            record_date = datetime.fromisoformat(oldest_record["timestamp"].replace("Z", "+00:00"))
        else:
            record_date = datetime.now(CHINA_TZ)

        storage_path = self._get_storage_path(table_name, record_date)

        # Upload to storage
        upload_success = await self.upload_to_storage(
            bucket=self.settings.storage_bucket,
            path=storage_path,
            content=csv_content,
        )

        if not upload_success:
            error_msg = "Failed to upload archive file"
            logger.error(f"Failed to upload archive for {table_name}, not deleting records")
            # Log failure
            await self.log_archival_operation(
                table_name=table_name,
                records_archived=len(records),
                records_deleted=0,
                storage_path=storage_path,
                status="failed",
                error_message=error_msg,
            )
            return {
                "table": table_name,
                "archived": 0,
                "deleted": 0,
                "error": "upload_failed",
            }

        # Delete archived records
        record_ids = [r[id_column] for r in records if id_column in r]
        deleted_count = await self.delete_archived_records(
            table_name=table_name,
            record_ids=record_ids,
            id_column=id_column,
        )

        # Log successful archival
        await self.log_archival_operation(
            table_name=table_name,
            records_archived=len(records),
            records_deleted=deleted_count,
            storage_path=storage_path,
            status="completed",
        )

        return {
            "table": table_name,
            "archived": len(records),
            "deleted": deleted_count,
            "storage_path": storage_path,
        }

    async def archive_all_tables(self) -> dict:
        """Archive all configured tables.

        Returns:
            Dictionary with archival statistics for all tables
        """
        results = {}
        batch_size = self.settings.archive_batch_size

        # Archive IMU data (highest volume, shortest retention)
        results["imu"] = await self.archive_table(
            table_name=IMU_TABLE,
            retention_days=self.settings.retention_imu_days,
            batch_size=batch_size,
        )

        # Archive HAR labels
        results["har"] = await self.archive_table(
            table_name=HAR_TABLE,
            retention_days=self.settings.retention_har_days,
            batch_size=batch_size,
        )

        # Archive atomic activities
        results["atomic_activities"] = await self.archive_table(
            table_name=ATOMIC_ACTIVITIES_TABLE,
            retention_days=self.settings.retention_atomic_days,
            batch_size=batch_size,
        )

        # Archive uploads
        results["uploads"] = await self.archive_table(
            table_name=UPLOADS_TABLE,
            retention_days=self.settings.retention_uploads_days,
            batch_size=batch_size,
        )

        # Archive summary logs (longer retention)
        results["summary_logs"] = await self.archive_table(
            table_name=SUMMARY_LOGS_TABLE,
            retention_days=self.settings.retention_summary_logs_days,
            batch_size=batch_size,
        )

        # Archive interventions
        results["interventions"] = await self.archive_table(
            table_name=INTERVENTIONS_TABLE,
            retention_days=self.settings.retention_interventions_days,
            batch_size=batch_size,
        )

        return results


async def run_archival() -> dict:
    """Run archival process for all tables.

    Returns:
        Dictionary with archival statistics
    """
    service = ArchiveService()
    return await service.archive_all_tables()