"""Tests for Celery services."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.celery_app.services.har_service import (
    get_imu_window,
    run_mock_har_model,
    insert_har_label,
    process_har_for_user,
)
from src.celery_app.services.atomic_service import (
    get_document_window,
    get_har_window,
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
# TestHarService
# ============================================================================


class TestHarService:
    """Tests for HAR service functions."""

    @pytest.mark.asyncio
    async def test_get_imu_window(self, mock_get_database, mongodb_mock):
        """Test fetching IMU data window."""
        coll = mongodb_mock["imu"]
        coll._data = [
            {
                "user": "test_user",
                "acc_X": 0.1, "acc_Y": 0.2, "acc_Z": 9.8,
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ]

        result = await get_imu_window("test_user", seconds=2)

        assert len(result) == 1
        assert result[0]["user"] == "test_user"

    @pytest.mark.asyncio
    async def test_run_mock_har_model_empty(self):
        """Test mock HAR model with empty data."""
        label, confidence = await run_mock_har_model([])
        assert label == "unknown"
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_run_mock_har_model_low_acceleration(self):
        """Test mock HAR model with low acceleration (sitting/lying)."""
        imu_data = [{"acc_X": 0.1, "acc_Y": 0.1, "acc_Z": 0.2}]
        label, confidence = await run_mock_har_model(imu_data)
        assert label in ["sitting", "lying", "standing"]
        assert 0.7 <= confidence <= 0.9

    @pytest.mark.asyncio
    async def test_run_mock_har_model_moderate_acceleration(self):
        """Test mock HAR model with moderate acceleration (walking)."""
        imu_data = [{"acc_X": 0.5, "acc_Y": 0.5, "acc_Z": 0.8}]
        label, confidence = await run_mock_har_model(imu_data)
        assert label in ["walking", "standing", "sitting"]
        assert 0.6 <= confidence <= 0.9

    @pytest.mark.asyncio
    async def test_run_mock_har_model_high_acceleration(self):
        """Test mock HAR model with high acceleration (running)."""
        imu_data = [{"acc_X": 5.0, "acc_Y": 2.0, "acc_Z": 12.0}]
        label, confidence = await run_mock_har_model(imu_data)
        assert label in ["running", "climbing stairs"]
        assert 0.6 <= confidence <= 0.9

    @pytest.mark.asyncio
    async def test_insert_har_label(self, mock_get_database):
        """Test inserting HAR label to database."""
        result = await insert_har_label("test_user", "walking", 0.8)
        assert result["har_label"] == "walking"
        assert result["confidence"] == 0.8
        assert result["user"] == "test_user"

    @pytest.mark.asyncio
    async def test_process_har_for_user_no_data(self, mock_get_database, mongodb_mock):
        """Test HAR processing with no IMU data."""
        mongodb_mock["imu"]._data = []

        result = await process_har_for_user("test_user")

        assert result is None


# ============================================================================
# TestAtomicService
# ============================================================================


class TestAtomicService:
    """Tests for Atomic Activities service functions."""

    @pytest.mark.asyncio
    async def test_get_document_window(self, mock_get_database, mongodb_mock):
        """Test fetching document data window."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "stepcount_sensor": 1000, "screen_on_ratio": 0.5}
        ]

        result = await get_document_window("test_user", seconds=10)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_har_window(self, mock_get_database, mongodb_mock):
        """Test fetching HAR data window."""
        mongodb_mock["har"]._data = [
            {"user": "test_user", "label": "walking", "confidence": 0.8}
        ]

        result = await get_har_window("test_user", seconds=2)
        assert len(result) == 1
        assert result[0]["label"] == "walking"

    @pytest.mark.asyncio
    async def test_generate_step_label_stationary(self, mock_get_database, mongodb_mock):
        """Test step label generation for stationary activity."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "stepcount_sensor": 1000},
            {"user": "test_user", "stepcount_sensor": 1000},
        ]
        result = await generate_step_label("test_user", window_seconds=10)
        assert result == "almost stationary"

    @pytest.mark.asyncio
    async def test_generate_step_label_low_activity(self, mock_get_database, mongodb_mock):
        """Test step label generation for low activity."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "stepcount_sensor": 1000},
            {"user": "test_user", "stepcount_sensor": 1010},
        ]
        result = await generate_step_label("test_user", window_seconds=10)
        assert result == "low"

    @pytest.mark.asyncio
    async def test_generate_step_label_high_activity(self, mock_get_database, mongodb_mock):
        """Test step label generation for high activity."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "stepcount_sensor": 1000},
            {"user": "test_user", "stepcount_sensor": 1020},
        ]
        result = await generate_step_label("test_user", window_seconds=10)
        assert result == "high"

    @pytest.mark.asyncio
    async def test_generate_phone_usage_heavy(self, mock_get_database, mongodb_mock):
        """Test phone usage label for heavy usage."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "screen_on_ratio": 0.9},
        ]
        result = await generate_phone_usage_label("test_user", window_seconds=10)
        assert result == "very high"

    @pytest.mark.asyncio
    async def test_generate_phone_usage_idle(self, mock_get_database, mongodb_mock):
        """Test phone usage label for idle state."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "screen_on_ratio": 0.1},
        ]
        result = await generate_phone_usage_label("test_user", window_seconds=10)
        assert result == "idle"

    @pytest.mark.asyncio
    async def test_generate_social_label_solitary(self, mock_get_database, mongodb_mock):
        """Test social label for solitary state."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "nearbyBluetoothCount": 0},
        ]
        result = await generate_social_label("test_user", window_seconds=10)
        assert result == "alone"

    @pytest.mark.asyncio
    async def test_generate_social_label_communication(self, mock_get_database, mongodb_mock):
        """Test social label for communication."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "nearbyBluetoothCount": 1, "current_app": "whatsapp"},
        ]
        result = await generate_social_label("test_user", window_seconds=10)
        assert result == "with someone"

    @pytest.mark.asyncio
    async def test_generate_movement_label_stationary(self, mock_get_database, mongodb_mock):
        """Test movement label for stationary state."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "gpsLat": 37.7749, "gpsLon": -122.4194},
            {"user": "test_user", "gpsLat": 37.7749, "gpsLon": -122.4194},
        ]
        result = await generate_movement_label("test_user", window_seconds=120)
        assert result == "stationary"

    @pytest.mark.asyncio
    async def test_generate_movement_label_walking(self, mock_get_database, mongodb_mock):
        """Test movement label for walking."""
        mongodb_mock["uploads"]._data = [
            {"user": "test_user", "gpsLat": 37.7749, "gpsLon": -122.4194},
            {"user": "test_user", "gpsLat": 37.7753, "gpsLon": -122.4194},
        ]
        result = await generate_movement_label("test_user", window_seconds=120)
        assert result in ["stationary", "slow", "medium", "fast"]


# ============================================================================
# TestSummaryService
# ============================================================================


class TestSummaryService:
    """Tests for Summary service functions."""

    @pytest.mark.asyncio
    async def test_compress_atomic_activities_empty(self, mock_get_database, mongodb_mock):
        """Test compressing empty atomic activities."""
        mongodb_mock["atomic_activities"]._data = []

        result = await compress_atomic_activities("test_user", hours=1)
        assert result["total_records"] == 0

    @pytest.mark.asyncio
    async def test_compress_atomic_activities_with_data(self, mock_get_database, mongodb_mock):
        """Test compressing atomic activities with data."""
        mongodb_mock["atomic_activities"]._data = [
            {
                "har_label": "walking", "app_category": "social_media",
                "step_count": "low_activity", "phone_usage": "moderate_usage",
                "social": "solitary", "movement": "walking", "location": "work",
            },
            {
                "har_label": "walking", "app_category": "social_media",
                "step_count": "low_activity", "phone_usage": "moderate_usage",
                "social": "solitary", "movement": "walking", "location": "work",
            },
        ]

        result = await compress_atomic_activities("test_user", hours=1)
        assert result["total_records"] == 2
        assert result["dominant"]["activity"] == "walking"
        assert result["dominant"]["app_category"] == "social_media"

    @pytest.mark.asyncio
    async def test_get_all_users_with_activities(self, mock_get_database, mongodb_mock):
        """Test getting users with recent activities."""
        mongodb_mock["atomic_activities"]._data = [
            {"user": "user1"},
            {"user": "user2"},
            {"user": "user1"},  # Duplicate
        ]

        result = await get_all_users_with_activities(hours=1)
        assert len(result) == 2
        assert "user1" in result
        assert "user2" in result

    @pytest.mark.asyncio
    async def test_get_recent_summaries(self, mock_get_database, mongodb_mock):
        """Test fetching recent summaries."""
        mongodb_mock["summary_logs"]._data = [
            {
                "id": "summary-1",
                "user": "test_user",
                "log_type": "hourly",
                "summary": "You had an active hour. Highlights: Good walking. Recommendations: Keep it up",
            }
        ]

        result = await get_recent_summaries(hours=1)
        assert len(result) == 1
        assert result[0]["user"] == "test_user"

    @pytest.mark.asyncio
    async def test_get_recent_summaries_empty(self, mock_get_database, mongodb_mock):
        """Test fetching recent summaries when none exist."""
        mongodb_mock["summary_logs"]._data = []

        result = await get_recent_summaries(hours=1)
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_intervention_from_summary_empty(self):
        """Test intervention generation with empty summary."""
        result = await generate_intervention_from_summary("test_user", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_intervention_from_summary_none(self):
        """Test intervention generation with None summary."""
        result = await generate_intervention_from_summary("test_user", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_intervention_from_summary_success(self):
        """Test successful intervention generation from summary."""
        summary_log = {
            "id": "summary-1",
            "user": "test_user",
            "summary": "You were mostly sitting during this hour.",
            "start_timestamp": datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
            "end_timestamp": datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        }

        mock_output = InterventionOutput(
            intervention_type="movement_reminder",
            message="You've been sitting for a while. Consider taking a short walk!",
            priority="medium",
            category="physical",
        )

        with patch(
            "src.celery_app.services.intervention_service.generate_structured_output",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = mock_output

            result = await generate_intervention_from_summary("test_user", summary_log)

        assert result is not None
        assert result["user"] == "test_user"
        # The function returns {user, intervention_content, start_timestamp, end_timestamp, timestamp}
        assert result["intervention_content"] == (
            "You've been sitting for a while. Consider taking a short walk!"
        )
        assert result["start_timestamp"] == summary_log["start_timestamp"]
        assert result["end_timestamp"] == summary_log["end_timestamp"]
        assert "timestamp" in result
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_intervention_from_summary_fallback(self):
        """Test intervention generation falls back on LLM error."""
        summary_log = {
            "id": "summary-1",
            "user": "test_user",
            "summary": "Good activity this hour.",
            "start_timestamp": datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
            "end_timestamp": datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        }

        with patch(
            "src.celery_app.services.intervention_service.generate_structured_output",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            result = await generate_intervention_from_summary("test_user", summary_log)

        # Fallback returns a default intervention message
        assert result is not None
        assert result["user"] == "test_user"
        assert "intervention_content" in result

    @pytest.mark.asyncio
    async def test_insert_intervention(self, mock_get_database):
        """Test inserting intervention to database."""
        intervention = {
            "user": "test_user",
            "intervention_type": "movement_reminder",
            "message": "Take a walk!",
            "priority": "medium",
            "category": "physical",
        }

        result = await insert_intervention(intervention)
        assert result["user"] == "test_user"
        assert "_id" in result

    @pytest.mark.asyncio
    async def test_insert_intervention_empty_response(self, mock_get_database):
        """Test inserting intervention returns enriched dict."""
        intervention = {
            "user": "test_user",
            "intervention_type": "movement_reminder",
        }

        result = await insert_intervention(intervention)
        # insert_intervention always returns the input dict + _id
        assert result["user"] == "test_user"
        assert "_id" in result

    @pytest.mark.asyncio
    async def test_generate_summary_no_data(self):
        """Test summary generation with no activity data."""
        compressed_data = {"total_records": 0, "summary": {}, "dominant": {}}
        result = await generate_summary("test_user", compressed_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_summary_hourly(self):
        """Test hourly summary generation."""
        compressed_data = {
            "total_records": 20,
            "period_hours": 1,
            "start_time": "2026-01-01T08:00:00+08:00",
            "end_time": "2026-01-01T09:00:00+08:00",
            "summary": {
                "har": {"walking": 15, "sitting": 5},
                "app_usage": {"productivity": 10, "social_media": 5},
            },
            "dominant": {
                "activity": "walking",
                "app_category": "productivity",
                "location": "work",
            },
        }

        mock_summary = SummaryOutput(
            title="Active Work Hour",
            summary="You had an active hour with lots of walking.",
            highlights=["15 minutes of walking", "Focused on productivity apps"],
            recommendations=["Keep up the good work!"],
        )

        with patch(
            "src.celery_app.services.summary_service.generate_structured_output",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = mock_summary

            result = await generate_summary("test_user", compressed_data, log_type="hourly")

        assert result is not None
        assert result["user"] == "test_user"
        assert result["log_type"] == "hourly"
        # Summary is now a combined string: title + summary + highlights + recommendations
        assert "Active Work Hour" in result["summary"]
        assert "active hour with lots of walking" in result["summary"]
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_generate_summary_daily(self):
        """Test daily summary generation."""
        compressed_data = {
            "total_records": 100,
            "period_hours": 24,
            "start_time": "2026-01-01T00:00:00+08:00",
            "end_time": "2026-01-02T00:00:00+08:00",
            "summary": {
                "har": {"sitting": 50, "walking": 30, "standing": 20},
            },
            "dominant": {
                "activity": "sitting",
            },
        }

        mock_summary = SummaryOutput(
            title="Productive Day",
            summary="A balanced day with good mix of activities.",
            highlights=["Good walking time", "Productive work sessions"],
            recommendations=["Try to stand more often"],
        )

        with patch(
            "src.celery_app.services.summary_service.generate_structured_output",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = mock_summary

            result = await generate_summary("test_user", compressed_data, log_type="daily")

        assert result is not None
        assert result["log_type"] == "daily"
        assert "Productive Day" in result["summary"]
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_summary_error_returns_none(self):
        """Test summary generation returns None on LLM error."""
        compressed_data = {
            "total_records": 10,
            "period_hours": 1,
            "summary": {"har": {"sitting": 10}},
            "dominant": {"activity": "sitting"},
        }

        with patch(
            "src.celery_app.services.summary_service.generate_structured_output",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            result = await generate_summary("test_user", compressed_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_insert_summary_log(self, mock_get_database):
        """Test inserting summary log to database."""
        summary_log = {
            "user": "test_user",
            "log_type": "hourly",
            "summary": "You had an active hour.",
        }

        result = await insert_summary_log(summary_log)
        assert result["user"] == "test_user"
        assert result["log_type"] == "hourly"
        assert "_id" in result

    @pytest.mark.asyncio
    async def test_insert_summary_log_empty_response(self, mock_get_database):
        """Test inserting summary log returns enriched dict."""
        summary_log = {
            "user": "test_user",
            "log_type": "daily",
        }

        result = await insert_summary_log(summary_log)
        assert result["user"] == "test_user"
        assert "_id" in result


# ============================================================================
# TestSchemas
# ============================================================================


class TestSchemas:
    """Tests for Pydantic schemas."""

    def test_har_label_schema(self):
        """Test HARLabel schema validation."""
        label = HARLabel(
            user="test_user",
            label="walking",
            confidence=0.8,
            timestamp=datetime.now(),
            source="mock_har"
        )
        assert label.user == "test_user"
        assert label.label == "walking"
        assert label.confidence == 0.8

    def test_atomic_activity_schema(self):
        """Test AtomicActivity schema validation."""
        activity = AtomicActivity(
            user="test_user",
            timestamp=datetime.now(),
            har_label="walking",
            app_category="social_media",
            step_label="low_activity",
            phone_usage="moderate_usage",
            social_label="solitary",
            movement_label="walking",
            location_label="work"
        )
        assert activity.user == "test_user"
        assert activity.har_label == "walking"
        assert activity.location_label == "work"

    def test_atomic_activity_optional_fields(self):
        """Test AtomicActivity with optional fields as None."""
        activity = AtomicActivity(
            user="test_user",
            timestamp=datetime.now()
        )
        assert activity.user == "test_user"
        assert activity.har_label is None
