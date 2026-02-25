"""Integration tests for the full intervention generation pipeline with real Supabase.

This test module covers the complete data flow with actual database operations:
1. IMU Data -> HAR Labels (har_service)
2. HAR Labels + Upload Data -> Atomic Activities (atomic_service)
3. Atomic Activities -> Summary Logs (summary_service)
4. Summary Logs -> Interventions (intervention_service)

IMPORTANT: These tests require a valid Supabase connection and will write/read real data.
All test data is cleaned up after tests complete.
"""

import asyncio
import os
import pytest
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch, AsyncMock
from dotenv import load_dotenv

load_dotenv()

# Skip all tests if running in CI without Supabase credentials
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_ANON_KEY"),
    reason="Supabase credentials not available",
)


def has_service_role_key() -> bool:
    """Check if service role key is available for admin operations."""
    return bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

from supabase import Client

from src.database import get_supabase_client, get_supabase_admin_client
from src.celery_app.services.har_service import (
    get_imu_window,
    run_mock_har_model,
    insert_har_label,
    process_har_for_user,
)
from src.celery_app.services.atomic_service import (
    get_document_window,
    get_har_window,
    generate_all_atomic_labels,
    insert_atomic_activity,
    generate_step_label,
    generate_phone_usage_label,
    generate_social_label,
    generate_movement_label,
)
from src.celery_app.services.summary_service import (
    compress_atomic_activities,
    get_all_users_with_activities,
    generate_summary,
    insert_summary_log,
    SummaryOutput,
)
from src.celery_app.services.intervention_service import (
    get_recent_summaries,
    generate_intervention_from_summary,
    insert_intervention,
    InterventionOutput,
)
from src.celery_app.schemas.har_schemas import HARLabel
from src.celery_app.schemas.atomic_schemas import AtomicActivity


# ============================================================================
# Test Configuration
# ============================================================================

TEST_USER_PREFIX = "test_pipeline_user_"
CLEANUP_TABLES = ["interventions", "summary_logs", "atomic_activities", "har", "imu", "uploads", "user"]


def generate_test_user_id() -> str:
    """Generate unique test user ID with timestamp."""
    return f"{TEST_USER_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def supabase_client() -> Client:
    """Create real Supabase client for integration tests."""
    return get_supabase_client()


@pytest.fixture(scope="module")
def supabase_admin_client() -> Optional[Client]:
    """Create admin Supabase client for cleanup operations."""
    if has_service_role_key():
        return get_supabase_admin_client()
    return None


@pytest.fixture
async def test_user(supabase_client: Client, supabase_admin_client: Optional[Client]):
    """Create a test user and yield the user ID, then cleanup."""
    user_id = generate_test_user_id()

    # Create test user
    user_data = {"name": user_id}
    await asyncio.to_thread(
        lambda: supabase_client.table("user").insert(user_data).execute()
    )

    yield user_id

    # Cleanup: Delete all test data
    # Use admin client if available, otherwise use regular client
    cleanup_client = supabase_admin_client if supabase_admin_client else supabase_client

    for table in CLEANUP_TABLES:
        try:
            tbl = table  # Capture for closure
            pattern = f"{TEST_USER_PREFIX}%"
            if table == "user":
                await asyncio.to_thread(
                    lambda t=tbl, c=cleanup_client, p=pattern: c.table(t)
                    .delete()
                    .filter("name", "ilike", p)
                    .execute()
                )
            else:
                await asyncio.to_thread(
                    lambda t=tbl, c=cleanup_client, p=pattern: c.table(t)
                    .delete()
                    .filter("user", "ilike", p)
                    .execute()
                )
        except Exception as e:
            print(f"Cleanup warning for {table}: {e}")


@pytest.fixture
def sample_imu_data_factory():
    """Factory for creating sample IMU data."""
    def _create(user: str, count: int = 5, base_time: Optional[datetime] = None):
        if base_time is None:
            base_time = datetime.now(timezone.utc)

        data = []
        for i in range(count):
            data.append({
                "user": user,
                "acc_X": 0.1 + i * 0.05,
                "acc_Y": 0.2 + i * 0.03,
                "acc_Z": 9.8 + i * 0.1,
                "gyro_X": 0.01,
                "gyro_Y": 0.02,
                "gyro_Z": 0.01,
                "mag_X": 0.1,
                "mag_Y": 0.1,
                "mag_Z": 0.1,
                "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
            })
        return data
    return _create


@pytest.fixture
def sample_upload_data_factory():
    """Factory for creating sample upload data."""
    def _create(user: str, count: int = 5, base_time: Optional[datetime] = None):
        if base_time is None:
            base_time = datetime.now(timezone.utc)

        data = []
        for i in range(count):
            data.append({
                "user": user,
                "current_app": "Instagram" if i % 2 == 0 else "WhatsApp",
                "stepcount_sensor": 1000 + i * 5,
                "screen_on_ratio": 0.5 + i * 0.1,
                "network_traffic": 10240.0 + i * 1000,
                "nearbyBluetoothCount": i % 5,
                "gpsLat": 37.7749 + i * 0.0001,
                "gpsLon": -122.4194 + i * 0.0001,
                "address": "123 Main St",
                "volume": 50,
                "battery": 80,
                "wifi_connected": True,
                "wifi_ssid": "TestWiFi",
                "timestamp": (base_time + timedelta(seconds=i * 2)).isoformat(),
            })
        return data
    return _create


# ============================================================================
# HAR Service Integration Tests
# ============================================================================

class TestHarServiceIntegration:
    """Integration tests for HAR service with real database."""

    @pytest.mark.asyncio
    async def test_insert_and_retrieve_imu_data(
        self,
        supabase_client: Client,
        test_user: str,
        sample_imu_data_factory,
    ):
        """Test inserting and retrieving IMU data."""
        # Insert IMU data
        imu_data = sample_imu_data_factory(test_user, count=3)

        for data in imu_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("imu").insert(d).execute()
            )

        # Wait a moment for data to be available
        await asyncio.sleep(0.5)

        # Retrieve IMU data
        result = await get_imu_window(test_user, seconds=120, client=supabase_client)

        assert len(result) >= 3
        assert all(r["user"] == test_user for r in result)
        assert "acc_X" in result[0]
        assert "acc_Y" in result[0]
        assert "acc_Z" in result[0]

    @pytest.mark.asyncio
    async def test_insert_har_label_to_database(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test inserting HAR label to database."""
        result = await insert_har_label(
            user=test_user,
            label="walking",
            confidence=0.85,
            source="test_integration",
            client=supabase_client,
        )

        assert result is not None
        assert result.get("user") == test_user
        assert result.get("har_label") == "walking"

    @pytest.mark.asyncio
    async def test_full_har_processing_pipeline(
        self,
        supabase_client: Client,
        test_user: str,
        sample_imu_data_factory,
    ):
        """Test complete HAR processing: insert IMU -> process -> verify HAR record."""
        # Insert IMU data
        imu_data = sample_imu_data_factory(test_user, count=5)
        for data in imu_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("imu").insert(d).execute()
            )

        await asyncio.sleep(0.5)

        # Process HAR for user
        result = await process_har_for_user(test_user, client=supabase_client)

        assert result is not None
        assert result.user == test_user
        assert result.label is not None
        assert result.confidence is not None
        assert result.source == "mock_har"

        # Verify HAR record in database
        har_records = await get_har_window(test_user, seconds=60, client=supabase_client)
        assert len(har_records) >= 1
        assert har_records[0]["user"] == test_user


# ============================================================================
# Atomic Activity Service Integration Tests
# ============================================================================

class TestAtomicServiceIntegration:
    """Integration tests for Atomic Activity service with real database."""

    @pytest.mark.asyncio
    async def test_insert_and_retrieve_upload_data(
        self,
        supabase_client: Client,
        test_user: str,
        sample_upload_data_factory,
    ):
        """Test inserting and retrieving upload/document data."""
        upload_data = sample_upload_data_factory(test_user, count=3)

        for data in upload_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("uploads").insert(d).execute()
            )

        await asyncio.sleep(0.5)

        # Retrieve upload data
        result = await get_document_window(test_user, seconds=60, client=supabase_client)

        assert len(result) >= 3
        assert all(r["user"] == test_user for r in result)

    @pytest.mark.asyncio
    async def test_generate_step_label_real_data(
        self,
        supabase_client: Client,
        test_user: str,
        sample_upload_data_factory,
    ):
        """Test step label generation with real database data."""
        upload_data = sample_upload_data_factory(test_user, count=5)
        for data in upload_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("uploads").insert(d).execute()
            )

        await asyncio.sleep(0.5)

        result = await generate_step_label(test_user, window_seconds=60, client=supabase_client)

        # Should return a valid step label
        assert result is not None
        assert result in ["almost stationary", "low", "medium", "high", "very high"]

    @pytest.mark.asyncio
    async def test_generate_phone_usage_label_real_data(
        self,
        supabase_client: Client,
        test_user: str,
        sample_upload_data_factory,
    ):
        """Test phone usage label generation with real database data."""
        upload_data = sample_upload_data_factory(test_user, count=5)
        for data in upload_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("uploads").insert(d).execute()
            )

        await asyncio.sleep(0.5)

        result = await generate_phone_usage_label(test_user, window_seconds=60, client=supabase_client)

        assert result is not None
        assert result in ["idle", "low", "medium", "high", "very high"]

    @pytest.mark.asyncio
    async def test_generate_social_label_real_data(
        self,
        supabase_client: Client,
        test_user: str,
        sample_upload_data_factory,
    ):
        """Test social label generation with real database data."""
        upload_data = sample_upload_data_factory(test_user, count=5)
        for data in upload_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("uploads").insert(d).execute()
            )

        await asyncio.sleep(0.5)

        result = await generate_social_label(test_user, window_seconds=60, client=supabase_client)

        assert result is not None
        assert result in [
            "alone", "alone or with someone", "with someone", "in group/public space"
        ]

    @pytest.mark.asyncio
    async def test_generate_movement_label_real_data(
        self,
        supabase_client: Client,
        test_user: str,
        sample_upload_data_factory,
    ):
        """Test movement label generation with real database data."""
        upload_data = sample_upload_data_factory(test_user, count=5)
        for data in upload_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("uploads").insert(d).execute()
            )

        await asyncio.sleep(0.5)

        result = await generate_movement_label(test_user, window_seconds=120, client=supabase_client)

        assert result is not None
        assert result in ["stationary", "slow", "medium", "fast"]

    @pytest.mark.asyncio
    async def test_insert_atomic_activity_real(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test inserting atomic activity to real database."""
        activity = AtomicActivity(
            user=test_user,
            timestamp=datetime.now(timezone.utc),
            har_label="walking",
            app_category="social communication app",
            step_label="medium",
            phone_usage="high",
            social_label="alone",
            movement_label="medium",
            location_label="work",
        )

        result = await insert_atomic_activity(activity, client=supabase_client)

        assert result is not None
        assert result.get("user") == test_user
        assert result.get("har_label") == "walking"
        assert result.get("app_category") == "social communication app"

    @pytest.mark.asyncio
    async def test_generate_all_atomic_labels_with_mocked_llm(
        self,
        supabase_client: Client,
        test_user: str,
        sample_imu_data_factory,
        sample_upload_data_factory,
    ):
        """Test generating all atomic labels with real data and mocked LLM."""
        # Insert IMU data
        imu_data = sample_imu_data_factory(test_user, count=3)
        for data in imu_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("imu").insert(d).execute()
            )

        # Insert upload data
        upload_data = sample_upload_data_factory(test_user, count=5)
        for data in upload_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("uploads").insert(d).execute()
            )

        # Insert HAR label
        await insert_har_label(test_user, "walking", 0.85, client=supabase_client)

        await asyncio.sleep(0.5)

        # Mock LLM calls for HAR, app, and location labels
        with patch("src.celery_app.services.atomic_service.query_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "walking"  # For HAR and app category

            activity = await generate_all_atomic_labels(test_user, client=supabase_client)

        assert activity is not None
        assert activity.user == test_user
        assert activity.step_label is not None
        assert activity.phone_usage is not None
        assert activity.social_label is not None
        assert activity.movement_label is not None


# ============================================================================
# Summary Service Integration Tests
# ============================================================================

class TestSummaryServiceIntegration:
    """Integration tests for Summary service with real database."""

    @pytest.mark.asyncio
    async def test_compress_atomic_activities_real_data(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test compressing atomic activities from real database."""
        # Insert multiple atomic activities
        for i in range(5):
            activity = AtomicActivity(
                user=test_user,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i * 10),
                har_label="walking" if i < 3 else "sitting",
                app_category="social communication app" if i < 2 else "office/working app",
                step_label="medium",
                phone_usage="high",
                social_label="alone",
                movement_label="medium",
                location_label="work",
            )
            await insert_atomic_activity(activity, client=supabase_client)

        await asyncio.sleep(0.5)

        result = await compress_atomic_activities(test_user, hours=1, client=supabase_client)

        assert result["total_records"] >= 5
        assert result["user"] == test_user
        assert "har" in result["summary"]
        assert "app_usage" in result["summary"]
        assert result["dominant"]["activity"] is not None

    @pytest.mark.asyncio
    async def test_get_all_users_with_activities_real(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test getting users with activities from real database."""
        # Insert atomic activity
        activity = AtomicActivity(
            user=test_user,
            timestamp=datetime.now(timezone.utc),
            har_label="walking",
        )
        await insert_atomic_activity(activity, client=supabase_client)

        await asyncio.sleep(0.5)

        users = await get_all_users_with_activities(hours=1, client=supabase_client)

        assert test_user in users

    @pytest.mark.asyncio
    async def test_insert_summary_log_real(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test inserting summary log to real database."""
        summary_log = {
            "user": test_user,
            "log_type": "hourly",
            "summary": "Test summary content",
            "start_timestamp": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = await insert_summary_log(summary_log, client=supabase_client)

        assert result is not None
        assert result.get("user") == test_user
        assert result.get("log_type") == "hourly"

    @pytest.mark.asyncio
    async def test_generate_summary_with_mocked_llm(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test generating summary with mocked LLM."""
        # Insert atomic activities
        for i in range(3):
            activity = AtomicActivity(
                user=test_user,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i * 10),
                har_label="sitting",
                app_category="office/working app",
            )
            await insert_atomic_activity(activity, client=supabase_client)

        await asyncio.sleep(0.5)

        # Compress activities
        compressed = await compress_atomic_activities(test_user, hours=1, client=supabase_client)

        # Mock LLM for summary generation
        mock_summary = SummaryOutput(
            title="Test Summary",
            summary="This is a test summary.",
            highlights=["Highlight 1", "Highlight 2"],
            recommendations=["Recommendation 1"],
        )

        with patch("src.celery_app.services.summary_service.generate_structured_output", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_summary

            result = await generate_summary(test_user, compressed, log_type="hourly")

        assert result is not None
        assert result["user"] == test_user
        assert result["title"] == "Test Summary"
        assert result["log_type"] == "hourly"


# ============================================================================
# Intervention Service Integration Tests
# ============================================================================

class TestInterventionServiceIntegration:
    """Integration tests for Intervention service with real database."""

    @pytest.mark.asyncio
    async def test_get_recent_summaries_real(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test getting recent summaries from real database."""
        # Insert a summary log
        summary_log = {
            "user": test_user,
            "log_type": "hourly",
            "summary": "Test summary for intervention",
            "start_timestamp": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await insert_summary_log(summary_log, client=supabase_client)

        await asyncio.sleep(0.5)

        summaries = await get_recent_summaries(hours=2, client=supabase_client)

        # Filter for our test user
        user_summaries = [s for s in summaries if s.get("user") == test_user]
        assert len(user_summaries) >= 1

    @pytest.mark.asyncio
    async def test_insert_intervention_real(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test inserting intervention to real database."""
        intervention = {
            "user": test_user,
            "intervention_content": "Take a break and stretch!",
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
            "end_timestamp": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        }

        result = await insert_intervention(intervention, client=supabase_client)

        assert result is not None
        assert result.get("user") == test_user
        assert result.get("intervention_content") == "Take a break and stretch!"

    @pytest.mark.asyncio
    async def test_generate_intervention_from_summary_mocked_llm(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test generating intervention from summary with mocked LLM."""
        summary_log = {
            "user": test_user,
            "title": "Sedentary Hour",
            "summary": "You have been sitting for most of the hour.",
            "highlights": ["60 minutes of sitting", "Low step count"],
            "recommendations": ["Consider standing up"],
            "dominant_activities": {"activity": "sitting", "location": "work"},
            "activity_counts": {"har": {"sitting": 60}},
            "period_hours": 1,
        }

        mock_intervention = InterventionOutput(
            intervention_type="movement_reminder",
            message="Time to stand up and move!",
            priority="high",
            category="physical",
        )

        with patch("src.celery_app.services.intervention_service.generate_structured_output", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_intervention

            result = await generate_intervention_from_summary(test_user, summary_log)

        assert result is not None
        assert result["user"] == test_user
        assert result["intervention_type"] == "movement_reminder"
        assert result["priority"] == "high"
        assert result["category"] == "physical"


# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================

class TestFullPipelineIntegration:
    """End-to-end tests for the complete intervention pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_har_to_intervention(
        self,
        supabase_client: Client,
        test_user: str,
        sample_imu_data_factory,
        sample_upload_data_factory,
    ):
        """Test the complete pipeline from IMU data to intervention generation."""
        # ====================================================================
        # Step 1: Insert raw sensor data (simulating mobile app uploads)
        # ====================================================================

        # Insert IMU data
        imu_data = sample_imu_data_factory(test_user, count=10)
        for data in imu_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("imu").insert(d).execute()
            )

        # Insert upload/document data
        upload_data = sample_upload_data_factory(test_user, count=10)
        for data in upload_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("uploads").insert(d).execute()
            )

        await asyncio.sleep(0.5)

        # ====================================================================
        # Step 2: Process HAR (IMU -> HAR labels)
        # ====================================================================

        har_result = await process_har_for_user(test_user, client=supabase_client)
        assert har_result is not None
        assert har_result.user == test_user

        # Verify HAR in database
        har_records = await get_har_window(test_user, seconds=60, client=supabase_client)
        assert len(har_records) >= 1

        # ====================================================================
        # Step 3: Generate Atomic Activities (HAR + Uploads -> Atomic)
        # ====================================================================

        # Mock LLM calls for label generation
        with patch("src.celery_app.services.atomic_service.query_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "sitting"

            atomic_activity = await generate_all_atomic_labels(test_user, client=supabase_client)

        assert atomic_activity is not None
        assert atomic_activity.user == test_user

        # Insert to database
        inserted_atomic = await insert_atomic_activity(atomic_activity, client=supabase_client)
        assert inserted_atomic is not None

        # ====================================================================
        # Step 4: Generate Summary (Atomic -> Summary)
        # ====================================================================

        # Compress atomic activities
        compressed = await compress_atomic_activities(test_user, hours=1, client=supabase_client)
        assert compressed["total_records"] >= 1

        # Mock LLM for summary generation
        mock_summary = SummaryOutput(
            title="Work Session Summary",
            summary="You had a productive session with some activity.",
            highlights=["Good focus time", "Some movement"],
            recommendations=["Take regular breaks"],
        )

        with patch("src.celery_app.services.summary_service.generate_structured_output", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_summary

            summary_result = await generate_summary(test_user, compressed, log_type="hourly")

        assert summary_result is not None

        # Insert summary log
        summary_log_data = {
            "user": test_user,
            "log_type": "hourly",
            "summary": summary_result["summary"],
            "start_timestamp": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        inserted_summary = await insert_summary_log(summary_log_data, client=supabase_client)
        assert inserted_summary is not None

        # ====================================================================
        # Step 5: Generate Intervention (Summary -> Intervention)
        # ====================================================================

        # Get recent summaries
        summaries = await get_recent_summaries(hours=1, client=supabase_client)
        user_summaries = [s for s in summaries if s.get("user") == test_user]
        assert len(user_summaries) >= 1

        # Mock LLM for intervention generation
        mock_intervention = InterventionOutput(
            intervention_type="screen_break",
            message="Consider taking a short break from your screen.",
            priority="medium",
            category="digital_wellbeing",
        )

        with patch("src.celery_app.services.intervention_service.generate_structured_output", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_intervention

            intervention_result = await generate_intervention_from_summary(test_user, user_summaries[0])

        assert intervention_result is not None
        assert intervention_result["user"] == test_user

        # Insert intervention
        intervention_data = {
            "user": test_user,
            "intervention_content": intervention_result["message"],
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
            "end_timestamp": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        }
        inserted_intervention = await insert_intervention(intervention_data, client=supabase_client)
        assert inserted_intervention is not None

        # ====================================================================
        # Verify Complete Pipeline
        # ====================================================================

        # Verify all data exists in database
        final_har = await get_har_window(test_user, 300, client=supabase_client)
        assert len(final_har) >= 1

        # Verify atomic_activities were inserted (not uploads)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=300)
        atomic_resp = await asyncio.to_thread(
            lambda: supabase_client.table("atomic_activities")
            .select("*")
            .eq("user", test_user)
            .gte("timestamp", cutoff.isoformat())
            .execute()
        )
        assert len(atomic_resp.data or []) >= 1

        final_summaries = await get_recent_summaries(hours=2, client=supabase_client)
        user_final_summaries = [s for s in final_summaries if s.get("user") == test_user]
        assert len(user_final_summaries) >= 1

    @pytest.mark.asyncio
    async def test_pipeline_multiple_users(
        self,
        supabase_client: Client,
        supabase_admin_client: Optional[Client],
        sample_imu_data_factory,
        sample_upload_data_factory,
    ):
        """Test pipeline with multiple users to verify data isolation."""
        users = [generate_test_user_id() for _ in range(3)]

        try:
            # Create users
            for user_id in users:
                await asyncio.to_thread(
                    lambda u=user_id: supabase_client.table("user").insert({"name": u}).execute()
                )

            # Process each user through the pipeline
            for user_id in users:
                # Insert data
                imu_data = sample_imu_data_factory(user_id, count=3)
                for data in imu_data:
                    await asyncio.to_thread(
                        lambda d=data: supabase_client.table("imu").insert(d).execute()
                    )

                upload_data = sample_upload_data_factory(user_id, count=3)
                for data in upload_data:
                    await asyncio.to_thread(
                        lambda d=data: supabase_client.table("uploads").insert(d).execute()
                    )

            await asyncio.sleep(0.5)

            # Process HAR for each user
            for user_id in users:
                har_result = await process_har_for_user(user_id, client=supabase_client)
                assert har_result is not None
                assert har_result.user == user_id

            # Verify data isolation
            for user_id in users:
                har_records = await get_har_window(user_id, seconds=60, client=supabase_client)
                assert all(r["user"] == user_id for r in har_records)

        finally:
            # Cleanup all test users
            cleanup_client = supabase_admin_client if supabase_admin_client else supabase_client
            user_list = users  # Capture for closure
            for table in CLEANUP_TABLES:
                try:
                    tbl = table
                    if table == "user":
                        await asyncio.to_thread(
                            lambda t=tbl, c=cleanup_client, u=user_list: c.table(t)
                            .delete()
                            .in_("name", u)
                            .execute()
                        )
                    else:
                        await asyncio.to_thread(
                            lambda t=tbl, c=cleanup_client, u=user_list: c.table(t)
                            .delete()
                            .in_("user", u)
                            .execute()
                        )
                except Exception as e:
                    print(f"Cleanup warning for {table}: {e}")

    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(
        self,
        supabase_client: Client,
        test_user: str,
        sample_imu_data_factory,
    ):
        """Test pipeline handles errors gracefully and continues."""
        # Insert IMU data but no upload data (will cause some labels to be None)
        imu_data = sample_imu_data_factory(test_user, count=3)
        for data in imu_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("imu").insert(d).execute()
            )

        await asyncio.sleep(0.5)

        # Process HAR - should work
        har_result = await process_har_for_user(test_user, client=supabase_client)
        assert har_result is not None

        # Generate atomic labels - some should be None due to missing upload data
        with patch("src.celery_app.services.atomic_service.query_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "sitting"

            activity = await generate_all_atomic_labels(test_user, client=supabase_client)

        # Activity should still be created with available labels
        assert activity is not None
        assert activity.user == test_user
        # Some labels may be None, that's expected
        assert activity.har_label is not None  # Should have HAR from mocked LLM


# ============================================================================
# Database Constraint Tests
# ============================================================================

class TestDatabaseConstraints:
    """Tests for database constraints and data integrity."""

    @pytest.mark.asyncio
    async def test_foreign_key_constraint_user(
        self,
        supabase_client: Client,
    ):
        """Test that foreign key constraint prevents inserting data for non-existent user."""
        non_existent_user = f"nonexistent_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        with pytest.raises(Exception) as exc_info:
            await insert_har_label(
                user=non_existent_user,
                label="walking",
                confidence=0.8,
                client=supabase_client,
            )

        # Should raise a foreign key violation
        assert "foreign key" in str(exc_info.value).lower() or "violates" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_intervention_timestamp_range(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test that interventions have valid timestamp range."""
        now = datetime.now(timezone.utc)
        intervention = {
            "user": test_user,
            "intervention_content": "Test intervention",
            "start_timestamp": now.isoformat(),
            "end_timestamp": (now + timedelta(hours=1)).isoformat(),
        }

        result = await insert_intervention(intervention, client=supabase_client)

        assert result is not None
        assert result.get("start_timestamp") is not None
        assert result.get("end_timestamp") is not None

    @pytest.mark.asyncio
    async def test_summary_log_type_validation(
        self,
        supabase_client: Client,
        test_user: str,
    ):
        """Test that summary log type is correctly stored."""
        for log_type in ["hourly", "daily"]:
            summary_log = {
                "user": test_user,
                "log_type": log_type,
                "summary": f"Test {log_type} summary",
                "start_timestamp": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                "end_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            result = await insert_summary_log(summary_log, client=supabase_client)
            assert result is not None
            assert result.get("log_type") == log_type


# ============================================================================
# Performance Tests
# ============================================================================

class TestPipelinePerformance:
    """Tests for pipeline performance with larger data volumes."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_volume_atomic_activities(
        self,
        supabase_client: Client,
        test_user: str,
        sample_upload_data_factory,
    ):
        """Test processing with larger volume of data."""
        # Insert 50 upload records
        upload_data = sample_upload_data_factory(test_user, count=50)
        for data in upload_data:
            await asyncio.to_thread(
                lambda d=data: supabase_client.table("uploads").insert(d).execute()
            )

        # Insert HAR label so generate_all_atomic_labels has har data
        await insert_har_label(test_user, "sitting", 0.85, client=supabase_client)

        await asyncio.sleep(1)

        # Generate labels
        with patch("src.celery_app.services.atomic_service.query_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "sitting"

            activity = await generate_all_atomic_labels(test_user, client=supabase_client)

        assert activity is not None

        # Insert activity so compress can find it
        await insert_atomic_activity(activity, client=supabase_client)

        # Compress should handle large volume
        compressed = await compress_atomic_activities(test_user, hours=1, client=supabase_client)
        assert compressed["total_records"] >= 1