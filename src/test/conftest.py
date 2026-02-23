"""Pytest fixtures for upload tests."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client that returns predictable insert responses."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_execute = MagicMock()
    mock_execute.data = []
    mock_table.insert.return_value.execute.return_value = mock_execute
    mock_client.table.return_value = mock_table
    return mock_client


@pytest.fixture
def client(mock_supabase_client):
    """Test client with mocked Supabase."""
    with patch("src.upload.service.get_supabase_client", return_value=mock_supabase_client):
        yield TestClient(app)
