"""Pytest fixtures and config for tests."""

import os
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

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
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:supabase._sync.client",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require real API credentials)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if required credentials are not available."""
    has_llm_api_key = bool(os.getenv("OPENROUTER_API_KEY"))
    has_supabase_creds = bool(
        os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )

    for item in items:
        if "integration" in item.keywords:
            if "test_llm_integration" in str(item.fspath):
                if not has_llm_api_key:
                    item.add_marker(pytest.mark.skip(
                        reason="LLM API credentials not available (set OPENROUTER_API_KEY in .env)"
                    ))


# ============================================================================
# MongoDB Mock Helpers
# ============================================================================


def _make_mock_cursor(data):
    """Create a mock MongoDB cursor that returns `data` from to_list()."""
    cursor = MagicMock()
    cursor._data = data

    async def _to_list(limit=None):
        if limit is not None:
            return cursor._data[:limit]
        return cursor._data

    cursor.to_list = AsyncMock(side_effect=_to_list)
    cursor.max_time_ms = MagicMock(return_value=cursor)
    cursor.sort = MagicMock(return_value=cursor)
    cursor.skip = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    return cursor


def _make_mock_agg_cursor(data):
    """Create a mock MongoDB aggregation cursor."""
    cursor = MagicMock()
    cursor._data = data

    async def _to_list(limit=None):
        if limit is not None:
            return cursor._data[:limit]
        return cursor._data

    cursor.to_list = AsyncMock(side_effect=_to_list)
    return cursor


def _make_collection_mock(name="unknown"):
    """Create a mock MongoDB collection supporting the Motor async API.

    Each test can configure return values via:
        coll.find_one.side_effect = ...
        coll.insert_many.return_value = ...
    """
    coll = MagicMock()
    coll._name = name
    coll._data = []

    # --- find_one(filter, sort=[...]) ---
    async def _find_one(filter=None, sort=None, projection=None, **kwargs):
        if coll._data:
            return coll._data[0]
        return None
    coll.find_one = AsyncMock(side_effect=_find_one)

    # --- insert_many(rows) ---
    async def _insert_many(rows):
        from bson import ObjectId
        ids = [ObjectId() for _ in rows]
        result = MagicMock()
        result.inserted_ids = ids
        return result
    coll.insert_many = AsyncMock(side_effect=_insert_many)

    # --- insert_one(data) ---
    async def _insert_one(data):
        from bson import ObjectId
        oid = ObjectId()
        result = MagicMock()
        result.inserted_id = oid
        return result
    coll.insert_one = AsyncMock(side_effect=_insert_one)

    # --- find(filter, projection?) → cursor ---
    def _find(*args, **kwargs):
        return _make_mock_cursor(coll._data)
    coll.find = MagicMock(side_effect=_find)

    # --- aggregate(pipeline) → cursor ---
    def _aggregate(*args, **kwargs):
        return _make_mock_agg_cursor(coll._data)
    coll.aggregate = MagicMock(side_effect=_aggregate)

    # --- create_index / create_indexes (for ensure_indexes at startup) ---
    async def _create_index(*args, **kwargs):
        return "index_created"
    coll.create_index = AsyncMock(side_effect=_create_index)

    async def _create_indexes(*args, **kwargs):
        return ["indexes_created"]
    coll.create_indexes = AsyncMock(side_effect=_create_indexes)

    async def _drop_index(*args, **kwargs):
        return None
    coll.drop_index = AsyncMock(side_effect=_drop_index)

    # --- count_documents (used in some services) ---
    async def _count_documents(filter=None, **kwargs):
        return len(coll._data)
    coll.count_documents = AsyncMock(side_effect=_count_documents)

    # --- distinct ---
    async def _distinct(key, filter=None, **kwargs):
        return list(set(d.get(key) for d in coll._data if key in d))
    coll.distinct = AsyncMock(side_effect=_distinct)

    # --- delete_many ---
    async def _delete_many(filter):
        result = MagicMock()
        result.deleted_count = len(coll._data)
        coll._data = []
        return result
    coll.delete_many = AsyncMock(side_effect=_delete_many)

    return coll


class _MongoDBMock:
    """Simulates an AsyncIOMotorDatabase with auto-creating collection mocks.

    Supports db["collection_name"] → collection mock.
    Also supports db.command("ping") and db.list_collection_names().
    """

    def __init__(self):
        self._collections = {}

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = _make_collection_mock(name)
        return self._collections[name]

    def __setitem__(self, name, coll):
        self._collections[name] = coll

    def get_collection(self, name):
        """Convenience: get a pre-configured collection mock."""
        return self[name]

    async def command(self, cmd):
        if cmd == "ping":
            return {"ok": 1}
        return {"ok": 1}

    async def list_collection_names(self):
        return list(self._collections.keys())


@pytest.fixture
def mongodb_mock():
    """Return a mock MongoDB database with collection support.

    Usage in tests:
        coll = mongodb_mock["collection_name"]
        coll.find_one.side_effect = lambda *a, **kw: {...}
        coll.insert_many.return_value = ...
        coll._data = [...]    # pre-seed data for find/find_one/aggregate

    Special collections:
        mongodb_mock["users"] — user registration
        mongodb_mock["uploads"] — sensor data
        mongodb_mock["imu"] — IMU data
        mongodb_mock["har"] — HAR labels
        mongodb_mock["atomic_activities"] — atomic activity labels
        mongodb_mock["summary_logs"] — generated summaries
        mongodb_mock["interventions"] — generated interventions
        mongodb_mock["intervention_feedbacks"] — user feedback
        mongodb_mock["summary_log_feedbacks"] — log feedback
    """
    return _MongoDBMock()


def _reset_database_globals():
    """Reset the module-level client globals in src.database.

    Without this, get_database() returns the cached client from a previous
    test, even after we patched AsyncIOMotorClient.
    """
    import src.database as db_mod
    db_mod._async_client = None
    db_mod._async_db = None
    db_mod._async_client_loop_id = None
    # Also reset sync client
    db_mod._sync_client = None
    db_mod._sync_db = None


@pytest.fixture
def client(mongodb_mock):
    """FastAPI TestClient with mocked MongoDB and Celery.

    Patches AsyncIOMotorClient so the real get_database() creates a mock
    Motor client whose __getitem__ returns our mongodb_mock collections.
    Also patches Celery task.delay to prevent RabbitMQ connection attempts.
    """
    # Create a mock Motor client that returns our mongodb_mock for any DB name
    mock_motor_client = MagicMock()
    mock_motor_client.__getitem__ = MagicMock(return_value=mongodb_mock)
    mock_motor_client.close = MagicMock()
    mock_motor_cls = MagicMock(return_value=mock_motor_client)

    # Also need to mock PyMongo's MongoClient for sync contexts (imu_dataset.py)
    mock_sync_client = MagicMock()
    mock_sync_client.__getitem__ = MagicMock(return_value=mongodb_mock)

    # Mock Celery task.delay to avoid RabbitMQ connection attempts
    mock_celery_delay = MagicMock()

    with patch("src.database.AsyncIOMotorClient", mock_motor_cls):
        with patch("src.database.MongoClient", return_value=mock_sync_client):
            # Patch Celery tasks imported in upload routes
            with patch("src.celery_app.tasks.har_tasks.process_har_batch.delay", mock_celery_delay):
                with patch("src.celery_app.tasks.atomic_tasks.process_atomic_activities_batch.delay", mock_celery_delay):
                    # Also patch the module-level celery_app connection probe in lifespan
                    with patch("src.celery_app.celery_app.celery_app.connection", return_value=MagicMock()):
                        _reset_database_globals()
                        yield TestClient(app)


@pytest.fixture
def mock_get_database(mongodb_mock):
    """Patch get_database for Celery service tests (non-HTTP tests).

    Patches AsyncIOMotorClient so service functions that call get_database()
    receive our mock DB. Also handles the case where Celery tasks create
    a new event loop and get_database() creates a fresh client.

    Usage:
        def test_something(mock_get_database, mongodb_mock):
            coll = mongodb_mock["imu"]
            coll._data = [...]
            result = await some_service_function()
    """
    mock_motor_client = MagicMock()
    mock_motor_client.__getitem__ = MagicMock(return_value=mongodb_mock)
    mock_motor_client.close = MagicMock()
    mock_motor_cls = MagicMock(return_value=mock_motor_client)

    mock_sync_client = MagicMock()
    mock_sync_client.__getitem__ = MagicMock(return_value=mongodb_mock)

    with patch("src.database.AsyncIOMotorClient", mock_motor_cls):
        with patch("src.database.MongoClient", return_value=mock_sync_client):
            _reset_database_globals()
            yield mongodb_mock
