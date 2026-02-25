"""Tests for query endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def mock_query_client():
    """Test client with mocked Supabase for query operations."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_execute = MagicMock()

    with patch("src.query.service.get_supabase_client", return_value=mock_client):
        mock_client.table.return_value = mock_table
        # Mock chain for summary logs: select().eq().eq().order().limit().execute()
        mock_table.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_execute
        # Mock chain for interventions: select().eq().order().limit().execute()
        mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_execute
        # Mock chain for insert operations: insert().execute()
        mock_table.insert.return_value.execute.return_value = mock_execute
        yield TestClient(app), mock_execute


class TestGetSummaryLog:
    """Tests for POST /get_summary_log."""

    def test_get_summary_log_hourly_success(self, mock_query_client):
        """Successfully fetches hourly summary log."""
        client, mock_execute = mock_query_client
        mock_execute.data = [
            {
                "id": 1,
                "user": "test_user",
                "log_type": "hourly",
                "summary": "User had a productive morning with focused work sessions.",
                "start_timestamp": "2026-02-25T08:00:00+00:00",
                "end_timestamp": "2026-02-25T09:00:00+00:00",
                "timestamp": "2026-02-25T09:00:00+00:00",
            }
        ]

        payload = {"user": "test_user", "log_type": "hourly"}
        response = client.post("/get_summary_log", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is not None
        assert data["data"]["id"] == 1
        assert data["data"]["log_content"] == "User had a productive morning with focused work sessions."
        # Pydantic serializes datetime with UTC as 'Z' suffix
        assert data["data"]["start_timestamp"] == "2026-02-25T08:00:00Z"
        assert data["data"]["end_timestamp"] == "2026-02-25T09:00:00Z"
        assert data["data"]["generation_timestamp"] == "2026-02-25T09:00:00Z"

    def test_get_summary_log_daily_success(self, mock_query_client):
        """Successfully fetches daily summary log."""
        client, mock_execute = mock_query_client
        mock_execute.data = [
            {
                "id": 2,
                "user": "test_user",
                "log_type": "daily",
                "summary": "User had a balanced day with good activity levels.",
                "start_timestamp": "2026-02-25T00:00:00+00:00",
                "end_timestamp": "2026-02-25T23:59:59+00:00",
                "timestamp": "2026-02-26T00:00:00+00:00",
            }
        ]

        payload = {"user": "test_user", "log_type": "daily"}
        response = client.post("/get_summary_log", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is not None
        assert data["data"]["id"] == 2
        assert data["data"]["log_content"] == "User had a balanced day with good activity levels."

    def test_get_summary_log_empty_result(self, mock_query_client):
        """Returns null data when no logs found."""
        client, mock_execute = mock_query_client
        mock_execute.data = []

        payload = {"user": "nonexistent_user", "log_type": "hourly"}
        response = client.post("/get_summary_log", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is None

    def test_get_summary_log_missing_user(self, mock_query_client):
        """Missing user field returns 422 validation error."""
        client, _ = mock_query_client
        payload = {"log_type": "hourly"}
        response = client.post("/get_summary_log", json=payload)
        assert response.status_code == 422

    def test_get_summary_log_missing_log_type(self, mock_query_client):
        """Missing log_type field returns 422 validation error."""
        client, _ = mock_query_client
        payload = {"user": "test_user"}
        response = client.post("/get_summary_log", json=payload)
        assert response.status_code == 422

    def test_get_summary_log_invalid_log_type(self, mock_query_client):
        """Invalid log_type returns 422 validation error."""
        client, _ = mock_query_client
        payload = {"user": "test_user", "log_type": "weekly"}
        response = client.post("/get_summary_log", json=payload)
        assert response.status_code == 422


class TestGetIntervention:
    """Tests for POST /get_intervention."""

    def test_get_intervention_success(self, mock_query_client):
        """Successfully fetches intervention for a user."""
        client, mock_execute = mock_query_client
        mock_execute.data = [
            {
                "id": 1,
                "user": "test_user",
                "message": "Take a short break to stretch and move around.",
                "intervention_type": "movement_reminder",
                "priority": "medium",
                "category": "physical",
                "start_timestamp": "2026-02-25T08:00:00+00:00",
                "end_timestamp": "2026-02-25T09:00:00+00:00",
                "timestamp": "2026-02-25T09:00:00+00:00",
            }
        ]

        payload = {"user": "test_user"}
        response = client.post("/get_intervention", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is not None
        assert data["data"]["id"] == 1
        assert data["data"]["intervention_content"] == "Take a short break to stretch and move around."
        # Pydantic serializes datetime with UTC as 'Z' suffix
        assert data["data"]["start_timestamp"] == "2026-02-25T08:00:00Z"
        assert data["data"]["end_timestamp"] == "2026-02-25T09:00:00Z"
        assert data["data"]["generation_timestamp"] == "2026-02-25T09:00:00Z"

    def test_get_intervention_empty_result(self, mock_query_client):
        """Returns null data when no interventions found."""
        client, mock_execute = mock_query_client
        mock_execute.data = []

        payload = {"user": "nonexistent_user"}
        response = client.post("/get_intervention", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] is None

    def test_get_intervention_missing_user(self, mock_query_client):
        """Missing user field returns 422 validation error."""
        client, _ = mock_query_client
        payload = {}
        response = client.post("/get_intervention", json=payload)
        assert response.status_code == 422