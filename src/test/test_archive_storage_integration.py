"""Integration tests for archive service storage operations.

These tests upload actual files to Supabase Storage for verification.
Files are stored in the /tests folder to avoid conflicts with production data.
"""

import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.celery_app.services.archive_service import (
    ArchiveService,
    _records_to_parquet,
)
from src.config import get_settings

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

CHINA_TZ = ZoneInfo("Asia/Shanghai")


@pytest.fixture(scope="module")
def archive_service():
    """Create ArchiveService with real Supabase client."""
    return ArchiveService()


@pytest.fixture(scope="module")
def real_settings():
    """Get real settings."""
    return get_settings()


class TestStorageUpload:
    """Tests for uploading files to Supabase Storage."""

    @pytest.mark.asyncio
    async def test_upload_small_parquet_to_tests_folder(self, archive_service, real_settings):
        """Upload a small Parquet file to /tests folder in storage bucket."""
        # Create dummy IMU data
        test_records = [
            {
                "id": i,
                "user_id": "test-user-001",
                "timestamp": (datetime.now(CHINA_TZ) - timedelta(days=i)).isoformat(),
                "acc_x": 0.1 * i,
                "acc_y": 0.2 * i,
                "acc_z": 9.8,
                "gyro_x": 0.01,
                "gyro_y": 0.02,
                "gyro_z": 0.03,
                "label": "walking" if i % 2 == 0 else "sitting",
            }
            for i in range(10)
        ]

        # Convert to Parquet
        parquet_bytes = _records_to_parquet(test_records)

        print(f"\nParquet file size: {len(parquet_bytes)} bytes")

        # Generate test path
        test_date = datetime.now(CHINA_TZ)
        test_path = f"tests/imu/{test_date.strftime('%Y/%m')}/test-{test_date.strftime('%Y-%m-%d-%H%M%S')}.parquet"

        # Upload to storage
        success = await archive_service.upload_to_storage(
            bucket=real_settings.storage_bucket,
            path=test_path,
            content=parquet_bytes,
            content_type="application/vnd.apache.parquet",
        )

        assert success, f"Failed to upload to {test_path}"
        print(f"Successfully uploaded to: {test_path}")

        # Verify the file exists by listing
        bucket = archive_service.client.storage.from_(real_settings.storage_bucket)
        files = bucket.list(f"tests/imu/{test_date.strftime('%Y/%m')}")

        # Find our uploaded file
        uploaded_file = next((f for f in files if f["name"].startswith("test-")), None)
        assert uploaded_file is not None, "Uploaded file not found in bucket listing"
        print(f"Verified file exists: {uploaded_file['name']}")

    @pytest.mark.asyncio
    async def test_upload_har_data_to_tests_folder(self, archive_service, real_settings):
        """Upload HAR data to /tests folder."""
        # Create dummy HAR data
        test_records = [
            {
                "id": i,
                "user_id": "test-user-001",
                "timestamp": (datetime.now(CHINA_TZ) - timedelta(hours=i)).isoformat(),
                "label": ["walking", "sitting", "standing", "running", "lying_down"][i % 5],
                "confidence": 0.85 + (0.1 * (i % 3)),
                "source": "test_model_v1",
            }
            for i in range(20)
        ]

        parquet_bytes = _records_to_parquet(test_records)

        print(f"\nHAR Parquet file size: {len(parquet_bytes)} bytes")

        test_date = datetime.now(CHINA_TZ)
        test_path = f"tests/har/{test_date.strftime('%Y/%m')}/test-{test_date.strftime('%Y-%m-%d-%H%M%S')}.parquet"

        success = await archive_service.upload_to_storage(
            bucket=real_settings.storage_bucket,
            path=test_path,
            content=parquet_bytes,
            content_type="application/vnd.apache.parquet",
        )

        assert success, f"Failed to upload to {test_path}"
        print(f"Successfully uploaded HAR data to: {test_path}")

    @pytest.mark.asyncio
    async def test_upload_atomic_activities_to_tests_folder(self, archive_service, real_settings):
        """Upload atomic activities data to /tests folder."""
        # Create dummy atomic activities data
        activities = ["walking", "sitting", "standing", "running", "lying_down", "going_upstairs", "going_downstairs"]

        test_records = [
            {
                "id": i,
                "user_id": "test-user-001",
                "timestamp": (datetime.now(CHINA_TZ) - timedelta(minutes=i * 5)).isoformat(),
                "activity": activities[i % len(activities)],
                "duration_seconds": 30 + (i * 10) % 60,
                "confidence": 0.9 - (0.05 * (i % 3)),
            }
            for i in range(15)
        ]

        parquet_bytes = _records_to_parquet(test_records)

        print(f"\nAtomic activities Parquet file size: {len(parquet_bytes)} bytes")

        test_date = datetime.now(CHINA_TZ)
        test_path = f"tests/atomic_activities/{test_date.strftime('%Y/%m')}/test-{test_date.strftime('%Y-%m-%d-%H%M%S')}.parquet"

        success = await archive_service.upload_to_storage(
            bucket=real_settings.storage_bucket,
            path=test_path,
            content=parquet_bytes,
            content_type="application/vnd.apache.parquet",
        )

        assert success, f"Failed to upload to {test_path}"
        print(f"Successfully uploaded atomic activities to: {test_path}")

    @pytest.mark.asyncio
    async def test_list_test_files_in_bucket(self, archive_service, real_settings):
        """List all test files in the storage bucket."""
        bucket = archive_service.client.storage.from_(real_settings.storage_bucket)

        # List files in tests folder
        try:
            test_folders = bucket.list("tests")
            print(f"\nFolders in /tests:")
            for folder in test_folders:
                print(f"  - {folder['name']}")

            # List files in each subfolder
            for folder in test_folders:
                subfolder_path = f"tests/{folder['name']}"
                try:
                    files = bucket.list(subfolder_path)
                    if files:
                        print(f"\nFiles in {subfolder_path}:")
                        for f in files:
                            print(f"  - {f['name']} ({f.get('metadata', {}).get('size', 'unknown size')})")
                except Exception as e:
                    print(f"  Could not list {subfolder_path}: {e}")

        except Exception as e:
            print(f"Could not list tests folder: {e}")
            pytest.skip("Tests folder does not exist yet")

    @pytest.mark.asyncio
    async def test_download_and_verify_parquet(self, archive_service, real_settings):
        """Download an uploaded Parquet file and verify its contents."""
        import pyarrow.parquet as pq

        bucket = archive_service.client.storage.from_(real_settings.storage_bucket)

        # First, upload a test file
        test_records = [
            {"id": i, "value": i * 10, "label": f"test_{i}"} for i in range(5)
        ]
        parquet_bytes = _records_to_parquet(test_records)

        test_path = f"tests/verify/test-{datetime.now(CHINA_TZ).strftime('%Y%m%d%H%M%S')}.parquet"

        success = await archive_service.upload_to_storage(
            bucket=real_settings.storage_bucket,
            path=test_path,
            content=parquet_bytes,
            content_type="application/vnd.apache.parquet",
        )

        assert success, "Upload failed"

        # Download the file
        try:
            response = bucket.download(test_path)

            # Verify the content
            buffer = io.BytesIO(response)
            table = pq.read_table(buffer)

            assert table.num_rows == 5, f"Expected 5 rows, got {table.num_rows}"
            assert "id" in table.column_names
            assert "value" in table.column_names
            assert "label" in table.column_names

            print(f"\nVerified Parquet file: {test_path}")
            print(f"  Rows: {table.num_rows}")
            print(f"  Columns: {table.column_names}")

        except Exception as e:
            pytest.fail(f"Failed to download and verify file: {e}")


class TestFullArchivalFlow:
    """Test the complete archival flow with test data."""

    @pytest.mark.asyncio
    async def test_archive_test_table_dry_run(self, archive_service):
        """Test archiving a test table (with archival disabled for safety)."""
        # Disable archival to ensure we don't actually delete anything
        original_enabled = archive_service.settings.archive_enabled
        archive_service.settings.archive_enabled = False

        try:
            result = await archive_service.archive_table(
                table_name="_test_table_for_archival",
                retention_days=30,
                batch_size=100,
            )

            assert result["archived"] == 0
            assert result.get("skipped") == True or result.get("error") is not None
            print(f"\nDry run result: {result}")

        finally:
            archive_service.settings.archive_enabled = original_enabled

    @pytest.mark.asyncio
    async def test_archive_flow_with_mock_data(self, archive_service, real_settings):
        """Test the complete archival flow with mock data (upload only, no delete)."""
        # Create mock records
        mock_records = [
            {
                "id": i,
                "user_id": "test-user-001",
                "timestamp": (datetime.now(CHINA_TZ) - timedelta(days=i + 10)).isoformat(),
                "acc_x": 0.1 * i,
                "acc_y": 0.2 * i,
                "acc_z": 9.8,
                "gyro_x": 0.01,
                "gyro_y": 0.02,
                "gyro_z": 0.03,
            }
            for i in range(50)
        ]

        # Convert to Parquet
        parquet_bytes = _records_to_parquet(mock_records)
        file_size = len(parquet_bytes)

        print(f"\nMock data Parquet size: {file_size} bytes for {len(mock_records)} records")

        # Generate storage path
        test_date = datetime.now(CHINA_TZ) - timedelta(days=10)
        storage_path = f"tests/archive_flow/{test_date.strftime('%Y/%m')}/test-{test_date.strftime('%Y-%m-%d')}.parquet"

        # Upload
        success = await archive_service.upload_to_storage(
            bucket=real_settings.storage_bucket,
            path=storage_path,
            content=parquet_bytes,
            content_type="application/vnd.apache.parquet",
        )

        assert success, "Upload failed"
        print(f"Uploaded to: {storage_path}")

        # Log the operation
        await archive_service.log_archival_operation(
            table_name="_test_imu",
            records_archived=len(mock_records),
            records_deleted=0,  # We're not deleting in this test
            storage_path=storage_path,
            file_size_bytes=file_size,
            status="test",
        )

        print(f"Logged archival operation for {len(mock_records)} records")