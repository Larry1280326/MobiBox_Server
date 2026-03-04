"""Pytest fixtures and config for tests."""

import warnings
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

# Suppress Supabase client deprecation warnings (timeout/verify moved to http client)
warnings.filterwarnings(
    "ignore",
    message="The 'timeout' parameter is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The 'verify' parameter is deprecated.*",
    category=DeprecationWarning,
)


def pytest_configure(config):
    """Register warning filters with pytest so they apply during test runs."""
    # Supabase SyncPostgrestClient passes deprecated timeout/verify; ignore in that module
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:supabase._sync.client",
    )


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
