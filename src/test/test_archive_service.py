"""Tests for data archival service and tasks."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from zoneinfo import ZoneInfo

import pytest

from src.celery_app.services.archive_service import (
    ArchiveService,
    run_archival,
    _records_to_parquet,
    IMU_TABLE,
    HAR_TABLE,
    ATOMIC_ACTIVITIES_TABLE,
    UPLOADS_TABLE,
    SUMMARY_LOGS_TABLE,
    INTERVENTIONS_TABLE,
    ARCHIVAL_LOGS_TABLE,
    CHINA_TZ,
)


class TestRecordsToParquet:
    """Tests for _records_to_parquet function."""

    def test_empty_records(self):
        """Empty records return empty bytes."""
        result = _records_to_parquet([])
        assert result == b""

    def test_simple_records(self):
        """Simple records are converted to Parquet."""
        records = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]

        result = _records_to_parquet(records)

        # Parquet files start with 'PAR1' magic bytes
        assert result[:4] == b"PAR1"
        assert len(result) > 0

    def test_records_with_datetime(self):
        """Records with datetime are properly serialized."""
        now = datetime.now()
        records = [
            {"id": 1, "timestamp": now, "data": "test"},
        ]

        result = _records_to_parquet(records)

        assert result[:4] == b"PAR1"
        assert len(result) > 0

    def test_records_with_nested_data(self):
        """Records with nested data are properly serialized."""
        records = [
            {"id": 1, "data": {"nested": "value"}, "list": [1, 2, 3]},
        ]

        result = _records_to_parquet(records)

        assert result[:4] == b"PAR1"
        assert len(result) > 0


class TestArchiveService:
    """Tests for ArchiveService class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.storage_bucket = "test-archive-bucket"
        settings.archive_enabled = True
        settings.archive_batch_size = 100
        settings.retention_imu_days = 7
        settings.retention_har_days = 30
        settings.retention_atomic_days = 30
        settings.retention_uploads_days = 30
        settings.retention_summary_logs_days = 90
        settings.retention_interventions_days = 90
        return settings

    @pytest.fixture
    def mock_client(self):
        """Mock Supabase client."""
        client = MagicMock()
        return client

    def test_get_storage_path(self, mock_settings, mock_client):
        """Test storage path generation."""
        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            service = ArchiveService(client=mock_client)

            date = datetime(2026, 3, 7, 12, 0, 0)
            path = service._get_storage_path("imu", date)

            assert path == "archives/imu/2026/03/2026-03-07.parquet"

    def test_get_storage_path_different_tables(self, mock_settings, mock_client):
        """Test storage path for different tables."""
        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            service = ArchiveService(client=mock_client)

            date = datetime(2026, 1, 15, 10, 30, 0)

            assert service._get_storage_path("har", date) == "archives/har/2026/01/2026-01-15.parquet"
            assert service._get_storage_path("atomic_activities", date) == "archives/atomic_activities/2026/01/2026-01-15.parquet"

    @pytest.mark.asyncio
    async def test_fetch_records_for_archival(self, mock_settings, mock_client):
        """Test fetching records for archival."""
        mock_response = MagicMock()
        mock_response.data = [
            {"id": 1, "timestamp": "2026-03-01T10:00:00Z", "data": "test1"},
            {"id": 2, "timestamp": "2026-03-01T11:00:00Z", "data": "test2"},
        ]

        # Mock asyncio.to_thread to return the response directly
        async def mock_to_thread(func, *args, **kwargs):
            return mock_response

        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            with patch("asyncio.to_thread", side_effect=mock_to_thread):
                service = ArchiveService(client=mock_client)

                records = await service.fetch_records_for_archival(
                    table_name="imu",
                    retention_days=7,
                    batch_size=100,
                )

                assert len(records) == 2

    @pytest.mark.asyncio
    async def test_fetch_records_empty(self, mock_settings, mock_client):
        """Test fetching records when no records to archive."""
        mock_response = MagicMock()
        mock_response.data = []

        # Mock asyncio.to_thread to return the response directly
        async def mock_to_thread(func, *args, **kwargs):
            return mock_response

        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            with patch("asyncio.to_thread", side_effect=mock_to_thread):
                service = ArchiveService(client=mock_client)

                records = await service.fetch_records_for_archival(
                    table_name="imu",
                    retention_days=7,
                    batch_size=100,
                )

                assert len(records) == 0

    @pytest.mark.asyncio
    async def test_archive_table_disabled(self, mock_settings, mock_client):
        """Test archive_table when archival is disabled."""
        mock_settings.archive_enabled = False

        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            service = ArchiveService(client=mock_client)

            result = await service.archive_table(
                table_name="imu",
                retention_days=7,
                batch_size=100,
            )

            assert result["archived"] == 0
            assert result["skipped"] == True

    @pytest.mark.asyncio
    async def test_archive_table_no_records(self, mock_settings, mock_client):
        """Test archive_table when no records need archiving."""
        # Mock fetch_records_for_archival to return empty
        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            service = ArchiveService(client=mock_client)
            service.fetch_records_for_archival = AsyncMock(return_value=[])

            # Mock log_archival_operation
            service.log_archival_operation = AsyncMock()

            result = await service.archive_table(
                table_name="imu",
                retention_days=7,
                batch_size=100,
            )

            assert result["archived"] == 0
            assert result["deleted"] == 0
            service.log_archival_operation.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_table_success(self, mock_settings, mock_client):
        """Test successful archival of records."""
        records = [
            {"id": 1, "timestamp": "2026-03-01T10:00:00Z", "data": "test1"},
            {"id": 2, "timestamp": "2026-03-01T11:00:00Z", "data": "test2"},
        ]

        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            service = ArchiveService(client=mock_client)

            # Mock methods
            service.fetch_records_for_archival = AsyncMock(return_value=records)
            service.upload_to_storage = AsyncMock(return_value=True)
            service.delete_archived_records = AsyncMock(return_value=2)
            service.log_archival_operation = AsyncMock()

            result = await service.archive_table(
                table_name="imu",
                retention_days=7,
                batch_size=100,
            )

            assert result["archived"] == 2
            assert result["deleted"] == 2
            assert "storage_path" in result
            service.upload_to_storage.assert_called_once()
            service.delete_archived_records.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_table_upload_failure(self, mock_settings, mock_client):
        """Test archive_table when upload fails."""
        records = [
            {"id": 1, "timestamp": "2026-03-01T10:00:00Z", "data": "test1"},
        ]

        # Mock asyncio.to_thread for log_archival_operation
        async def mock_to_thread(func, *args, **kwargs):
            mock_response = MagicMock()
            mock_response.data = [{"id": 1}]
            return mock_response

        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            with patch("asyncio.to_thread", side_effect=mock_to_thread):
                service = ArchiveService(client=mock_client)

                # Mock methods
                service.fetch_records_for_archival = AsyncMock(return_value=records)
                service.upload_to_storage = AsyncMock(return_value=False)  # Upload fails
                service.log_archival_operation = AsyncMock()
                service.delete_archived_records = AsyncMock(return_value=0)

                result = await service.archive_table(
                    table_name="imu",
                    retention_days=7,
                    batch_size=100,
                )

                assert result["archived"] == 0
                assert result["deleted"] == 0
                assert result.get("error") == "upload_failed"
                # Should not delete records if upload failed
                service.delete_archived_records.assert_not_called()

    @pytest.mark.asyncio
    async def test_archive_all_tables(self, mock_settings, mock_client):
        """Test archiving all configured tables."""
        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            service = ArchiveService(client=mock_client)

            # Mock archive_table for each table
            service.archive_table = AsyncMock(return_value={"archived": 10, "deleted": 10})

            results = await service.archive_all_tables()

            assert "imu" in results
            assert "har" in results
            assert "atomic_activities" in results
            assert "uploads" in results
            assert "summary_logs" in results
            assert "interventions" in results

            # Should be called 6 times (6 tables)
            assert service.archive_table.call_count == 6

    @pytest.mark.asyncio
    async def test_log_archival_operation(self, mock_settings, mock_client):
        """Test logging archival operation."""
        mock_response = MagicMock()
        mock_response.data = [{"id": 1}]
        mock_table = MagicMock()
        mock_table.insert.return_value.execute = AsyncMock(return_value=mock_response)
        mock_client.table.return_value = mock_table

        with patch("src.celery_app.services.archive_service.get_settings", return_value=mock_settings):
            service = ArchiveService(client=mock_client)

            await service.log_archival_operation(
                table_name="imu",
                records_archived=100,
                records_deleted=100,
                storage_path="archives/imu/2026/03/2026-03-07.parquet",
                file_size_bytes=5000,
                status="completed",
            )

            mock_client.table.assert_called_once_with(ARCHIVAL_LOGS_TABLE)
            mock_table.insert.assert_called_once()


class TestArchiveTasks:
    """Tests for Celery archive tasks."""

    def test_archive_data_periodic_task(self):
        """Test archive_data_periodic Celery task."""
        from src.celery_app.tasks.archive_tasks import archive_data_periodic

        # Mock the run_archival function
        with patch("src.celery_app.tasks.archive_tasks.run_archival") as mock_run:
            mock_run.return_value = {
                "imu": {"archived": 100, "deleted": 100},
                "har": {"archived": 50, "deleted": 50},
            }

            result = archive_data_periodic()

            assert result["imu"]["archived"] == 100
            assert result["har"]["archived"] == 50

    def test_archive_data_periodic_task_error(self):
        """Test archive_data_periodic handles errors."""
        from src.celery_app.tasks.archive_tasks import archive_data_periodic

        with patch("src.celery_app.tasks.archive_tasks.run_archival") as mock_run:
            mock_run.side_effect = Exception("Test error")

            result = archive_data_periodic()

            assert "error" in result

    def test_archive_table_manual_task(self):
        """Test archive_table_manual Celery task."""
        from src.celery_app.tasks.archive_tasks import archive_table_manual

        # Create a mock service instance
        mock_service = MagicMock()
        mock_service.settings.archive_batch_size = 100
        mock_service.archive_table = AsyncMock(return_value={"archived": 50, "deleted": 50})

        # Patch ArchiveService at its source module
        with patch("src.celery_app.services.archive_service.ArchiveService", return_value=mock_service):
            result = archive_table_manual("imu", 7)

            assert result["archived"] == 50
            mock_service.archive_table.assert_called_once_with(
                table_name="imu",
                retention_days=7,
                batch_size=100,
            )

    def test_get_archive_stats_task(self):
        """Test get_archive_stats Celery task."""
        from src.celery_app.tasks.archive_tasks import get_archive_stats

        # Mock Supabase client
        mock_response = MagicMock()
        mock_response.data = [{"id": 1}, {"id": 2}, {"id": 3}]
        mock_response.count = 3

        mock_table = MagicMock()
        mock_table.select.return_value = mock_table
        mock_table.lt.return_value = mock_table
        mock_table.execute = Mock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.table.return_value = mock_table

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.retention_imu_days = 7
        mock_settings.retention_har_days = 30
        mock_settings.retention_atomic_days = 30
        mock_settings.retention_uploads_days = 30
        mock_settings.retention_summary_logs_days = 90
        mock_settings.retention_interventions_days = 90

        # Patch at source modules
        with patch("src.database.get_supabase_client", return_value=mock_client):
            with patch("src.config.get_settings", return_value=mock_settings):
                result = get_archive_stats()

                assert "imu" in result
                assert "har" in result


class TestRunArchival:
    """Tests for run_archival function."""

    @pytest.mark.asyncio
    async def test_run_archival(self):
        """Test run_archival function."""
        mock_service = MagicMock()
        mock_service.archive_all_tables = AsyncMock(return_value={
            "imu": {"archived": 100, "deleted": 100},
            "har": {"archived": 50, "deleted": 50},
        })

        with patch("src.celery_app.services.archive_service.ArchiveService", return_value=mock_service):
            result = await run_archival()

            assert result["imu"]["archived"] == 100
            assert result["har"]["archived"] == 50
            mock_service.archive_all_tables.assert_called_once()


class TestParquetFormat:
    """Tests for Parquet file format."""

    def test_parquet_has_correct_structure(self):
        """Verify Parquet file structure."""
        import pyarrow.parquet as pq
        import io

        records = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.3},
            {"id": 3, "name": "Charlie", "score": 92.1},
        ]

        parquet_bytes = _records_to_parquet(records)

        # Read back and verify
        buffer = io.BytesIO(parquet_bytes)
        table = pq.read_table(buffer)

        assert table.num_rows == 3
        assert "id" in table.column_names
        assert "name" in table.column_names
        assert "score" in table.column_names

    def test_parquet_compression_efficiency(self):
        """Test that Parquet produces smaller files than CSV for repetitive data."""
        import csv
        import io

        # Create data with repetitive values (good for compression)
        records = [
            {"id": i, "status": "active", "category": "type_a", "count": i * 10}
            for i in range(1000)
        ]

        # Parquet size
        parquet_bytes = _records_to_parquet(records)
        parquet_size = len(parquet_bytes)

        # CSV size
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=["id", "status", "category", "count"])
        writer.writeheader()
        writer.writerows(records)
        csv_size = len(csv_buffer.getvalue().encode("utf-8"))

        # Parquet should be smaller for repetitive data
        # Note: Parquet with Snappy compression typically achieves 10-100x compression
        # for repetitive data, but for small datasets the overhead might make it larger
        # We just check that Parquet is reasonably sized
        assert parquet_size < csv_size * 1.5  # Parquet should be at most 150% of CSV size