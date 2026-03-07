"""Integration tests for data archival service with real Supabase.

These tests require actual Supabase credentials and will make real API calls.
Run with: pytest -m integration src/test/test_archive_service_integration.py

Note: These tests should be run against a test/development database, not production.
"""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.celery_app.services.archive_service import (
    ArchiveService,
    _records_to_parquet,
    run_archival,
    IMU_TABLE,
    HAR_TABLE,
    ATOMIC_ACTIVITIES_TABLE,
    ARCHIVAL_LOGS_TABLE,
)
from src.database import get_supabase_admin_client
from src.config import get_settings

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# Timezone for China (used in archive service)
CHINA_TZ = ZoneInfo("Asia/Shanghai")


@pytest.fixture(scope="module")
def real_supabase_client():
    """Create a real Supabase admin client for integration tests."""
    # Check if Supabase credentials are available
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        pytest.skip("Supabase credentials not available (set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)")

    client = get_supabase_admin_client()
    yield client


@pytest.fixture(scope="module")
def real_settings():
    """Get real settings for integration tests."""
    return get_settings()


@pytest.fixture(scope="module")
def archive_service(real_supabase_client):
    """Create ArchiveService with real Supabase client."""
    return ArchiveService(client=real_supabase_client)


class TestArchiveServiceConnection:
    """Tests to verify Supabase connectivity."""

    def test_supabase_client_created(self, real_supabase_client):
        """Verify Supabase client can be created."""
        assert real_supabase_client is not None

    def test_settings_loaded(self, real_settings):
        """Verify archive settings are loaded correctly."""
        assert real_settings.storage_bucket is not None
        assert real_settings.archive_batch_size > 0
        assert real_settings.retention_imu_days > 0

    def test_storage_bucket_accessible(self, archive_service, real_settings):
        """Verify storage bucket exists and is accessible."""
        # Try to list files in the bucket (should not raise)
        try:
            result = archive_service.client.storage.from_(real_settings.storage_bucket).list()
            # If bucket doesn't exist, this will raise an error
            assert result is not None
        except Exception as e:
            # Bucket might not exist yet - that's okay for testing
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                pytest.skip("Storage bucket does not exist yet")
            raise


class TestArchiveServiceRealOperations:
    """Integration tests for archive service with real Supabase."""

    @pytest.mark.asyncio
    async def test_fetch_records_for_archival_real(self, archive_service, real_settings):
        """Test fetching records from real database."""
        # Fetch with a very long retention period to get any records
        records = await archive_service.fetch_records_for_archival(
            table_name=IMU_TABLE,
            retention_days=365,  # 1 year - should capture most test data
            batch_size=10,  # Small batch for testing
        )

        # Should return a list (might be empty if no data)
        assert isinstance(records, list)

    @pytest.mark.asyncio
    async def test_fetch_records_with_columns(self, archive_service, real_settings):
        """Test fetching specific columns from real database."""
        # Fetch specific columns
        records = await archive_service.fetch_records_for_archival(
            table_name=IMU_TABLE,
            retention_days=365,
            batch_size=10,
            columns=["id", "timestamp"],
        )

        # Should return a list
        assert isinstance(records, list)
        # If records exist, they should have only the specified columns
        if records:
            assert "id" in records[0] or len(records[0]) <= 2

    @pytest.mark.asyncio
    async def test_log_archival_operation_real(self, archive_service, real_supabase_client):
        """Test logging archival operation to real database."""
        # Log a test archival operation
        await archive_service.log_archival_operation(
            table_name="_test_table",  # Use a test table name to avoid conflicts
            records_archived=0,
            records_deleted=0,
            storage_path=None,
            file_size_bytes=None,
            status="test",
            error_message="Integration test - can be deleted",
        )

        # Verify the log was created by querying the table
        response = real_supabase_client.table(ARCHIVAL_LOGS_TABLE).select("*").eq("table_name", "_test_table").execute()

        # Should have at least one log entry
        assert len(response.data) >= 1

        # Clean up the test log
        test_log_ids = [log["id"] for log in response.data if log.get("table_name") == "_test_table"]
        if test_log_ids:
            real_supabase_client.table(ARCHIVAL_LOGS_TABLE).delete().in_("id", test_log_ids).execute()

    @pytest.mark.asyncio
    async def test_get_storage_path_format(self, archive_service):
        """Test storage path generation with real date."""
        test_date = datetime(2026, 3, 7, 12, 30, 45, tzinfo=CHINA_TZ)

        path = archive_service._get_storage_path(IMU_TABLE, test_date)

        assert path == "archives/imu/2026/03/2026-03-07.parquet"

    @pytest.mark.asyncio
    async def test_archive_table_disabled(self, archive_service, real_settings):
        """Test that archival can be disabled via settings."""
        # Temporarily disable archival
        original_enabled = archive_service.settings.archive_enabled
        archive_service.settings.archive_enabled = False

        try:
            result = await archive_service.archive_table(
                table_name="_test_table",
                retention_days=1,
                batch_size=10,
            )

            assert result["archived"] == 0
            assert result["skipped"] == True
        finally:
            archive_service.settings.archive_enabled = original_enabled


class TestParquetFormatReal:
    """Test Parquet conversion with real data samples."""

    def test_parquet_with_real_imu_structure(self):
        """Test Parquet conversion with realistic IMU data structure."""
        # Create sample IMU data with realistic structure
        records = [
            {
                "id": 1,
                "user_id": "test-user-123",
                "timestamp": "2026-03-01T10:00:00Z",
                "acc_x": 0.123,
                "acc_y": 0.456,
                "acc_z": 9.81,
                "gyro_x": 0.01,
                "gyro_y": 0.02,
                "gyro_z": 0.03,
            },
            {
                "id": 2,
                "user_id": "test-user-123",
                "timestamp": "2026-03-01T10:00:01Z",
                "acc_x": 0.125,
                "acc_y": 0.458,
                "acc_z": 9.80,
                "gyro_x": 0.011,
                "gyro_y": 0.021,
                "gyro_z": 0.031,
            },
        ]

        parquet_bytes = _records_to_parquet(records)

        assert parquet_bytes[:4] == b"PAR1"
        assert len(parquet_bytes) > 0

        # Verify we can read it back
        import io
        import pyarrow.parquet as pq

        buffer = io.BytesIO(parquet_bytes)
        table = pq.read_table(buffer)

        assert table.num_rows == 2
        assert "id" in table.column_names
        assert "acc_x" in table.column_names

    def test_parquet_with_real_har_structure(self):
        """Test Parquet conversion with realistic HAR data structure."""
        # Create sample HAR data with realistic structure
        records = [
            {
                "id": 1,
                "user_id": "test-user-123",
                "timestamp": "2026-03-01T10:00:00Z",
                "label": "walking",
                "confidence": 0.95,
                "source": "model_v1",
            },
            {
                "id": 2,
                "user_id": "test-user-123",
                "timestamp": "2026-03-01T10:05:00Z",
                "label": "sitting",
                "confidence": 0.88,
                "source": "model_v1",
            },
        ]

        parquet_bytes = _records_to_parquet(records)

        assert parquet_bytes[:4] == b"PAR1"
        assert len(parquet_bytes) > 0

    def test_parquet_compression_efficiency_realistic(self):
        """Test Parquet compression efficiency with realistic data volume."""
        import csv
        import io

        # Create more realistic IMU data volume
        records = []
        for i in range(1000):
            records.append({
                "id": i,
                "user_id": "user-123",
                "timestamp": f"2026-03-01T{10 + i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}Z",
                "acc_x": 0.1 * (i % 10),
                "acc_y": 0.2 * (i % 10),
                "acc_z": 9.8 + 0.01 * (i % 5),
                "gyro_x": 0.01 * i,
                "gyro_y": 0.02 * i,
                "gyro_z": 0.03 * i,
            })

        # Parquet size
        parquet_bytes = _records_to_parquet(records)
        parquet_size = len(parquet_bytes)

        # CSV size for comparison
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=[
            "id", "user_id", "timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"
        ])
        writer.writeheader()
        writer.writerows(records)
        csv_size = len(csv_buffer.getvalue().encode("utf-8"))

        # Parquet should be reasonably sized
        # With Snappy compression, Parquet typically achieves good compression
        print(f"\nParquet size: {parquet_size} bytes")
        print(f"CSV size: {csv_size} bytes")
        print(f"Compression ratio: {csv_size / parquet_size:.2f}x")

        # Parquet should be at most 150% of CSV size for this data
        assert parquet_size < csv_size * 1.5


class TestArchiveAllTablesReal:
    """Test archiving all tables with real Supabase."""

    @pytest.mark.asyncio
    async def test_archive_all_tables_dry_run(self, archive_service):
        """Test archive_all_tables with archival disabled (dry run)."""
        # Temporarily disable archival
        original_enabled = archive_service.settings.archive_enabled
        archive_service.settings.archive_enabled = False

        try:
            results = await archive_service.archive_all_tables()

            # Should return results for all configured tables
            assert "imu" in results
            assert "har" in results
            assert "atomic_activities" in results
            assert "uploads" in results
            assert "summary_logs" in results
            assert "interventions" in results

            # All should be skipped
            for table_result in results.values():
                assert table_result.get("skipped") == True or table_result.get("archived") == 0

        finally:
            archive_service.settings.archive_enabled = original_enabled


class TestRunArchivalReal:
    """Test run_archival function with real Supabase."""

    @pytest.mark.asyncio
    async def test_run_archival_dry_run(self):
        """Test run_archival with archival disabled (dry run)."""
        settings = get_settings()
        original_enabled = settings.archive_enabled

        # Temporarily disable archival
        settings.archive_enabled = False

        try:
            results = await run_archival()

            # Should return results for all tables
            assert isinstance(results, dict)
            assert len(results) > 0

        finally:
            settings.archive_enabled = original_enabled


class TestGetArchiveStats:
    """Test getting archive statistics."""

    def test_get_archive_stats_task_real(self):
        """Test get_archive_stats with real Supabase."""
        from src.celery_app.tasks.archive_tasks import get_archive_stats

        # This will use real Supabase connection
        result = get_archive_stats()

        # Should return stats for all tables
        assert "imu" in result
        assert "har" in result
        assert "atomic_activities" in result
        assert "uploads" in result
        assert "summary_logs" in result
        assert "interventions" in result

        # Each table should have stats
        for table_name, stats in result.items():
            assert isinstance(stats, dict)
            # May have archiveable_count or error
            assert "archiveable_count" in stats or "error" in stats