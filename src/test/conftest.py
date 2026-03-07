"""Pytest fixtures and config for tests."""

import os
import warnings
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv

from src.main import app

# Load .env file before running tests
load_dotenv()

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
    """Register warning filters and custom markers with pytest."""
    # Supabase SyncPostgrestClient passes deprecated timeout/verify; ignore in that module
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:supabase._sync.client",
    )
    # Register integration test marker
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require real API credentials)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if LLM API credentials are not available."""
    skip_reason = "LLM API credentials not available (set OPENROUTER_API_KEY in .env)"

    # Check if OpenRouter API key is available
    has_api_key = bool(os.getenv("OPENROUTER_API_KEY"))

    if not has_api_key:
        skip_marker = pytest.mark.skip(reason=skip_reason)
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_marker)


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