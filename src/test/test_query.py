"""Tests for query endpoints."""

from datetime import datetime, timezone

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient


class TestGetSummaryLog:
    """Tests for POST /get_summary_log."""

    def test_get_summary_log_hourly_success(self, client: TestClient, mongodb_mock):
        """Successfully fetches hourly summary log."""
        oid = ObjectId()
        coll = mongodb_mock["summary_logs"]
        coll.find_one.side_effect = lambda filter=None, sort=None, **kw: {
            "_id": oid,
            "user": "test_user",
            "log_type": "hourly",
            "summary": "User had a productive morning with focused work sessions.",
            "start_timestamp": datetime(2026, 2, 25, 8, 0, 0, tzinfo=timezone.utc),
            "end_timestamp": datetime(2026, 2, 25, 9, 0, 0, tzinfo=timezone.utc),
            "timestamp": datetime(2026, 2, 25, 9, 0, 0, tzinfo=timezone.utc),
        }

        payload = {"user": "test_user", "log_type": "hourly"}
        response = client.post("/get_summary_log", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is not None
        assert data["data"]["id"] == str(oid)
        assert data["data"]["log_content"] == (
            "User had a productive morning with focused work sessions."
        )
        # Pydantic serializes datetime with UTC as 'Z' suffix
        assert data["data"]["start_timestamp"] == "2026-02-25T08:00:00Z"
        assert data["data"]["end_timestamp"] == "2026-02-25T09:00:00Z"
        assert data["data"]["generation_timestamp"] == "2026-02-25T09:00:00Z"

    def test_get_summary_log_daily_success(self, client: TestClient, mongodb_mock):
        """Successfully fetches daily summary log."""
        oid = ObjectId()
        coll = mongodb_mock["summary_logs"]
        coll.find_one.side_effect = lambda filter=None, sort=None, **kw: {
            "_id": oid,
            "user": "test_user",
            "log_type": "daily",
            "summary": "User had a balanced day with good activity levels.",
            "start_timestamp": datetime(2026, 2, 25, 0, 0, 0, tzinfo=timezone.utc),
            "end_timestamp": datetime(2026, 2, 25, 23, 59, 59, tzinfo=timezone.utc),
            "timestamp": datetime(2026, 2, 26, 0, 0, 0, tzinfo=timezone.utc),
        }

        payload = {"user": "test_user", "log_type": "daily"}
        response = client.post("/get_summary_log", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is not None
        assert data["data"]["id"] == str(oid)
        assert data["data"]["log_content"] == (
            "User had a balanced day with good activity levels."
        )
        # Daily logs should have a date string
        assert data["date"] == "2026-02-25"

    def test_get_summary_log_empty_result(self, client: TestClient, mongodb_mock):
        """Returns null data when no logs found."""
        coll = mongodb_mock["summary_logs"]
        coll.find_one.side_effect = lambda filter=None, sort=None, **kw: None

        payload = {"user": "nonexistent_user", "log_type": "hourly"}
        response = client.post("/get_summary_log", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is None

    def test_get_summary_log_missing_user(self, client: TestClient):
        """Missing user field returns 422 validation error."""
        payload = {"log_type": "hourly"}
        response = client.post("/get_summary_log", json=payload)
        assert response.status_code == 422

    def test_get_summary_log_missing_log_type(self, client: TestClient):
        """Missing log_type field returns 422 validation error."""
        payload = {"user": "test_user"}
        response = client.post("/get_summary_log", json=payload)
        assert response.status_code == 422

    def test_get_summary_log_invalid_log_type(self, client: TestClient):
        """Invalid log_type returns 422 validation error."""
        payload = {"user": "test_user", "log_type": "weekly"}
        response = client.post("/get_summary_log", json=payload)
        assert response.status_code == 422

    def test_get_summary_log_polling_same_id(self, client: TestClient, mongodb_mock):
        """Polling with same last_log_id returns has_new_log=False."""
        oid = ObjectId()
        coll = mongodb_mock["summary_logs"]
        coll.find_one.side_effect = lambda filter=None, sort=None, **kw: {
            "_id": oid,
            "user": "test_user",
            "log_type": "hourly",
            "summary": "No change.",
            "start_timestamp": datetime(2026, 2, 25, 8, 0, 0, tzinfo=timezone.utc),
            "end_timestamp": datetime(2026, 2, 25, 9, 0, 0, tzinfo=timezone.utc),
            "timestamp": datetime(2026, 2, 25, 9, 0, 0, tzinfo=timezone.utc),
        }

        payload = {"user": "test_user", "log_type": "hourly", "last_log_id": str(oid)}
        response = client.post("/get_summary_log", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["has_new_log"] is False
        assert data["data"] is None


class TestGetIntervention:
    """Tests for POST /get_intervention."""

    def test_get_intervention_success(self, client: TestClient, mongodb_mock):
        """Successfully fetches intervention for a user."""
        oid = ObjectId()
        coll = mongodb_mock["interventions"]
        coll.find_one.side_effect = lambda filter=None, sort=None, **kw: {
            "_id": oid,
            "user": "test_user",
            "intervention_content": "Take a short break to stretch and move around.",
            "start_timestamp": datetime(2026, 2, 25, 8, 0, 0, tzinfo=timezone.utc),
            "end_timestamp": datetime(2026, 2, 25, 9, 0, 0, tzinfo=timezone.utc),
            "timestamp": datetime(2026, 2, 25, 9, 0, 0, tzinfo=timezone.utc),
        }

        payload = {"user": "test_user"}
        response = client.post("/get_intervention", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is not None
        assert data["data"]["id"] == str(oid)
        assert data["data"]["intervention_content"] == (
            "Take a short break to stretch and move around."
        )
        assert data["data"]["start_timestamp"] == "2026-02-25T08:00:00Z"
        assert data["data"]["end_timestamp"] == "2026-02-25T09:00:00Z"
        assert data["data"]["generation_timestamp"] == "2026-02-25T09:00:00Z"

    def test_get_intervention_empty_result(self, client: TestClient, mongodb_mock):
        """Returns null data when no interventions found."""
        coll = mongodb_mock["interventions"]
        coll.find_one.side_effect = lambda filter=None, sort=None, **kw: None

        payload = {"user": "nonexistent_user"}
        response = client.post("/get_intervention", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is None

    def test_get_intervention_missing_user(self, client: TestClient):
        """Missing user field returns 422 validation error."""
        payload = {}
        response = client.post("/get_intervention", json=payload)
        assert response.status_code == 422


class TestSendFeedback:
    """Tests for feedback submission endpoints."""

    def test_send_intervention_feedback(self, client: TestClient):
        """Successfully submits intervention feedback."""
        payload = {
            "user": "test_user",
            "intervention_id": "507f1f77bcf86cd799439011",
            "feedback": "Great suggestion!",
            "mc1": "yes",
            "mc2": "no",
        }
        response = client.post("/send_intervention_feedback", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_send_log_feedback(self, client: TestClient):
        """Successfully submits summary log feedback."""
        payload = {
            "user": "test_user",
            "summary_logs_id": "507f1f77bcf86cd799439011",
            "feedback": "Accurate summary",
            "q1": "4",
            "q2": "yes",
        }
        response = client.post("/send_log_feedback", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestGetAtomicActivities:
    """Tests for atomic activities endpoints."""

    def test_get_atomic_activities_empty(self, client: TestClient, mongodb_mock):
        """Returns empty data when no atomic activities found."""
        coll = mongodb_mock["atomic_activities"]
        coll._data = []  # Empty data → cursor returns empty list

        payload = {"user": "test_user", "duration": 3600}
        response = client.post("/get_compressed_atomic_activities", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is not None
        assert data["data"]["sport"] == []
        assert data["data"]["appCategory"] == []

    def test_get_atomic_activities_with_data(self, client: TestClient, mongodb_mock):
        """Returns grouped atomic activities."""
        from datetime import datetime, timezone
        coll = mongodb_mock["atomic_activities"]
        coll._data = [
            {
                "user": "test_user",
                "har_label": "walking",
                "app_category": "social_media",
                "location": "home",
                "movement": "slow",
                "step_count": "medium",
                "phone_usage": "high",
                "timestamp": datetime(2026, 2, 25, 8, 30, 0, tzinfo=timezone.utc),
            },
            {
                "user": "test_user",
                "har_label": "sitting",
                "app_category": "productivity",
                "location": "work",
                "movement": "stationary",
                "step_count": "low",
                "phone_usage": "medium",
                "timestamp": datetime(2026, 2, 25, 8, 35, 0, tzinfo=timezone.utc),
            },
        ]

        payload = {"user": "test_user", "duration": 3600}
        response = client.post("/get_compressed_atomic_activities", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "walking" in data["data"]["sport"]
        assert "sitting" in data["data"]["sport"]
        assert "social_media" in data["data"]["appCategory"]
        assert "productivity" in data["data"]["appCategory"]
        assert data["start_timestamp"] is not None
        assert data["end_timestamp"] is not None
