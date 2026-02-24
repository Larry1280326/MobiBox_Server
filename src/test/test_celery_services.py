"""Tests for Celery services."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta

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
    generate_intervention,
    insert_intervention,
    generate_summary,
    insert_summary_log,
    InterventionOutput,
    SummaryOutput,
)
from src.celery_app.schemas.har_schemas import HARLabel
from src.celery_app.schemas.atomic_schemas import AtomicActivity


class TestHarService:
    """Tests for HAR service functions."""

    @pytest.mark.asyncio
    async def test_get_imu_window(self, mock_supabase_client):
        """Test fetching IMU data window."""
        # Setup mock response
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "acc_X": 0.1, "acc_Y": 0.2, "acc_Z": 9.8, "timestamp": "2024-01-01T00:00:00Z"}
        ]

        result = await get_imu_window("test_user", seconds=2, client=mock_supabase_client)

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
        imu_data = [
            {"acc_X": 0.1, "acc_Y": 0.1, "acc_Z": 0.2}
        ]
        label, confidence = await run_mock_har_model(imu_data)

        assert label in ["sitting", "lying_down", "standing"]
        assert 0.7 <= confidence <= 0.9

    @pytest.mark.asyncio
    async def test_run_mock_har_model_moderate_acceleration(self):
        """Test mock HAR model with moderate acceleration (walking)."""
        imu_data = [
            {"acc_X": 1.0, "acc_Y": 0.5, "acc_Z": 10.0}
        ]
        label, confidence = await run_mock_har_model(imu_data)

        assert label in ["walking", "standing", "driving", "climbing_stairs", "cycling"]
        assert 0.5 <= confidence <= 0.9

    @pytest.mark.asyncio
    async def test_run_mock_har_model_high_acceleration(self):
        """Test mock HAR model with high acceleration (running)."""
        imu_data = [
            {"acc_X": 5.0, "acc_Y": 2.0, "acc_Z": 12.0}
        ]
        label, confidence = await run_mock_har_model(imu_data)

        assert label in ["running", "climbing_stairs", "descending_stairs"]
        assert 0.6 <= confidence <= 0.9

    @pytest.mark.asyncio
    async def test_insert_har_label(self, mock_supabase_client):
        """Test inserting HAR label to database."""
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": 1, "user": "test_user", "label": "walking", "confidence": 0.8}
        ]

        result = await insert_har_label("test_user", "walking", 0.8, client=mock_supabase_client)

        assert result["label"] == "walking"
        mock_supabase_client.table.assert_called_with("har")

    @pytest.mark.asyncio
    async def test_process_har_for_user_no_data(self, mock_supabase_client):
        """Test HAR processing with no IMU data."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = []

        result = await process_har_for_user("test_user", client=mock_supabase_client)

        assert result is None


class TestAtomicService:
    """Tests for Atomic Activities service functions."""

    @pytest.mark.asyncio
    async def test_get_document_window(self, mock_supabase_client):
        """Test fetching document data window."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "stepcount_sensor": 1000, "screen_on_ratio": 0.5}
        ]

        result = await get_document_window("test_user", seconds=10, client=mock_supabase_client)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_har_window(self, mock_supabase_client):
        """Test fetching HAR data window."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "label": "walking", "confidence": 0.8}
        ]

        result = await get_har_window("test_user", seconds=2, client=mock_supabase_client)

        assert len(result) == 1
        assert result[0]["label"] == "walking"

    @pytest.mark.asyncio
    async def test_generate_step_label_stationary(self, mock_supabase_client):
        """Test step label generation for stationary activity."""
        # Two readings with no change
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "stepcount_sensor": 1000},
            {"user": "test_user", "stepcount_sensor": 1000},
        ]

        result = await generate_step_label("test_user", window_seconds=10, client=mock_supabase_client)

        assert result == "stationary"

    @pytest.mark.asyncio
    async def test_generate_step_label_low_activity(self, mock_supabase_client):
        """Test step label generation for low activity."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "stepcount_sensor": 1000},
            {"user": "test_user", "stepcount_sensor": 1010},
        ]

        result = await generate_step_label("test_user", window_seconds=10, client=mock_supabase_client)

        assert result == "low_activity"

    @pytest.mark.asyncio
    async def test_generate_step_label_high_activity(self, mock_supabase_client):
        """Test step label generation for high activity."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "stepcount_sensor": 1000},
            {"user": "test_user", "stepcount_sensor": 1200},
        ]

        result = await generate_step_label("test_user", window_seconds=10, client=mock_supabase_client)

        assert result == "high_activity"

    @pytest.mark.asyncio
    async def test_generate_phone_usage_heavy(self, mock_supabase_client):
        """Test phone usage label for heavy usage."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "screen_on_ratio": 0.9},
        ]

        result = await generate_phone_usage_label("test_user", window_seconds=10, client=mock_supabase_client)

        assert result == "heavy_usage"

    @pytest.mark.asyncio
    async def test_generate_phone_usage_idle(self, mock_supabase_client):
        """Test phone usage label for idle state."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user"},
        ]

        result = await generate_phone_usage_label("test_user", window_seconds=10, client=mock_supabase_client)

        assert result == "idle"

    @pytest.mark.asyncio
    async def test_generate_social_label_solitary(self, mock_supabase_client):
        """Test social label for solitary state."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "nearbyBluetoothCount": 0},
        ]

        result = await generate_social_label("test_user", window_seconds=10, client=mock_supabase_client)

        assert result == "solitary"

    @pytest.mark.asyncio
    async def test_generate_social_label_communication(self, mock_supabase_client):
        """Test social label for communication."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "nearbyBluetoothCount": 1, "current_app": "whatsapp"},
        ]

        result = await generate_social_label("test_user", window_seconds=10, client=mock_supabase_client)

        assert result == "direct_communication"

    @pytest.mark.asyncio
    async def test_generate_movement_label_stationary(self, mock_supabase_client):
        """Test movement label for stationary state."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "gpsLat": 37.7749, "gpsLon": -122.4194},
            {"user": "test_user", "gpsLat": 37.7749, "gpsLon": -122.4194},  # Same location
        ]

        result = await generate_movement_label("test_user", window_seconds=120, client=mock_supabase_client)

        assert result == "stationary"

    @pytest.mark.asyncio
    async def test_generate_movement_label_walking(self, mock_supabase_client):
        """Test movement label for walking."""
        # Two points ~50 meters apart over 120 seconds = ~0.4 m/s = walking
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"user": "test_user", "gpsLat": 37.7749, "gpsLon": -122.4194},
            {"user": "test_user", "gpsLat": 37.7753, "gpsLon": -122.4194},
        ]

        result = await generate_movement_label("test_user", window_seconds=120, client=mock_supabase_client)

        # Could be walking or stationary depending on distance calculation
        assert result in ["stationary", "slow_movement", "walking"]


class TestSummaryService:
    """Tests for Summary service functions."""

    @pytest.mark.asyncio
    async def test_compress_atomic_activities_empty(self, mock_supabase_client):
        """Test compressing empty atomic activities."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = []

        result = await compress_atomic_activities("test_user", hours=1, client=mock_supabase_client)

        assert result["total_records"] == 0

    @pytest.mark.asyncio
    async def test_compress_atomic_activities_with_data(self, mock_supabase_client):
        """Test compressing atomic activities with data."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            {"har_label": "walking", "app_category": "social_media", "step_label": "low_activity",
             "phone_usage": "moderate_usage", "social_label": "solitary",
             "movement_label": "walking", "location_label": "work"},
            {"har_label": "walking", "app_category": "social_media", "step_label": "low_activity",
             "phone_usage": "moderate_usage", "social_label": "solitary",
             "movement_label": "walking", "location_label": "work"},
        ]

        result = await compress_atomic_activities("test_user", hours=1, client=mock_supabase_client)

        assert result["total_records"] == 2
        assert result["dominant"]["activity"] == "walking"
        assert result["dominant"]["app_category"] == "social_media"

    @pytest.mark.asyncio
    async def test_get_all_users_with_activities(self, mock_supabase_client):
        """Test getting users with recent activities."""
        mock_supabase_client.table.return_value.select.return_value.gte.return_value.execute.return_value.data = [
            {"user": "user1"},
            {"user": "user2"},
            {"user": "user1"},  # Duplicate
        ]

        result = await get_all_users_with_activities(hours=1, client=mock_supabase_client)

        assert len(result) == 2
        assert "user1" in result
        assert "user2" in result

    @pytest.mark.asyncio
    async def test_generate_intervention_no_data(self):
        """Test intervention generation with no activity data."""
        compressed_data = {"total_records": 0, "summary": {}, "dominant": {}}

        result = await generate_intervention("test_user", compressed_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_intervention_success(self):
        """Test successful intervention generation."""
        compressed_data = {
            "total_records": 10,
            "period_hours": 1,
            "summary": {
                "har": {"sitting": 8, "walking": 2},
                "app_usage": {"social_media": 5, "productivity": 3},
                "phone_usage": {"heavy_usage": 7},
            },
            "dominant": {
                "activity": "sitting",
                "app_category": "social_media",
                "location": "home",
            },
        }

        mock_intervention = InterventionOutput(
            intervention_type="movement_reminder",
            message="You've been sitting for a while. Consider taking a short walk!",
            priority="medium",
            category="physical",
        )

        with patch("src.celery_app.services.summary_service.generate_structured_output", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_intervention

            result = await generate_intervention("test_user", compressed_data)

        assert result is not None
        assert result["user"] == "test_user"
        assert result["intervention_type"] == "movement_reminder"
        assert result["priority"] == "medium"
        assert result["category"] == "physical"
        assert "timestamp" in result
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_intervention_fallback_on_error(self):
        """Test intervention generation falls back on LLM error."""
        compressed_data = {
            "total_records": 10,
            "period_hours": 1,
            "summary": {"har": {"sitting": 10}},
            "dominant": {"activity": "sitting"},
        }

        with patch("src.celery_app.services.summary_service.generate_structured_output", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            result = await generate_intervention("test_user", compressed_data)

        assert result is not None
        assert result["intervention_type"] == "general_wellbeing"
        assert result["priority"] == "low"
        assert result["category"] == "mental"

    @pytest.mark.asyncio
    async def test_insert_intervention(self, mock_supabase_client):
        """Test inserting intervention to database."""
        intervention = {
            "user": "test_user",
            "intervention_type": "movement_reminder",
            "message": "Take a walk!",
            "priority": "medium",
            "category": "physical",
        }

        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": 1, **intervention}
        ]

        result = await insert_intervention(intervention, client=mock_supabase_client)

        assert result["id"] == 1
        assert result["user"] == "test_user"
        assert result["intervention_type"] == "movement_reminder"
        mock_supabase_client.table.assert_called_with("interventions")

    @pytest.mark.asyncio
    async def test_insert_intervention_empty_response(self, mock_supabase_client):
        """Test inserting intervention with empty database response."""
        intervention = {
            "user": "test_user",
            "intervention_type": "movement_reminder",
        }

        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = None

        result = await insert_intervention(intervention, client=mock_supabase_client)

        assert result == {}

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

        with patch("src.celery_app.services.summary_service.generate_structured_output", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_summary

            result = await generate_summary("test_user", compressed_data, log_type="hourly")

        assert result is not None
        assert result["user"] == "test_user"
        assert result["log_type"] == "hourly"
        assert result["title"] == "Active Work Hour"
        assert result["summary"] == "You had an active hour with lots of walking."
        assert len(result["highlights"]) == 2
        assert len(result["recommendations"]) == 1
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_generate_summary_daily(self):
        """Test daily summary generation."""
        compressed_data = {
            "total_records": 100,
            "period_hours": 24,
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

        with patch("src.celery_app.services.summary_service.generate_structured_output", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_summary

            result = await generate_summary("test_user", compressed_data, log_type="daily")

        assert result is not None
        assert result["log_type"] == "daily"
        assert result["title"] == "Productive Day"
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

        with patch("src.celery_app.services.summary_service.generate_structured_output", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")

            result = await generate_summary("test_user", compressed_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_insert_summary_log(self, mock_supabase_client):
        """Test inserting summary log to database."""
        summary_log = {
            "user": "test_user",
            "log_type": "hourly",
            "title": "Active Hour",
            "summary": "You had an active hour.",
            "highlights": ["Good walking"],
            "recommendations": ["Keep it up"],
        }

        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": 1, **summary_log}
        ]

        result = await insert_summary_log(summary_log, client=mock_supabase_client)

        assert result["id"] == 1
        assert result["user"] == "test_user"
        assert result["log_type"] == "hourly"
        mock_supabase_client.table.assert_called_with("summary_logs")

    @pytest.mark.asyncio
    async def test_insert_summary_log_empty_response(self, mock_supabase_client):
        """Test inserting summary log with empty database response."""
        summary_log = {
            "user": "test_user",
            "log_type": "daily",
        }

        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = None

        result = await insert_summary_log(summary_log, client=mock_supabase_client)

        assert result == {}


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
