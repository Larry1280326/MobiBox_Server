"""MongoDB database connection using Motor (async) and PyMongo (sync fallback)."""

import asyncio
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient

from src.config import get_settings

logger = logging.getLogger(__name__)

# Async client (Motor) — used by FastAPI routes and Celery services
_async_client: Optional[AsyncIOMotorClient] = None
_async_db: Optional[AsyncIOMotorDatabase] = None
_async_client_loop_id: Optional[int] = None  # Track which event loop the client is bound to

# Sync client (PyMongo) — used by imu_dataset.py (PyTorch Dataset __init__)
_sync_client: Optional[MongoClient] = None
_sync_db = None


async def get_database() -> AsyncIOMotorDatabase:
    """Get or create the async Motor MongoDB database instance.

    Recreates the Motor client if the current event loop differs from the one
    the client was originally created on.  This is essential in Celery workers
    where ``_run_async`` (or ``asyncio.run()``) creates a fresh event loop for
    every task invocation — a cached client bound to a previous (now closed)
    loop would fail with "Event loop is closed".
    """
    global _async_client, _async_db, _async_client_loop_id

    # Detect the currently-running event loop (if any)
    try:
        current_loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        current_loop_id = None

    # If the event loop has changed since the client was created, discard the
    # old client — it's bound to a loop that is now closed (or about to be).
    if (
        _async_client is not None
        and current_loop_id is not None
        and _async_client_loop_id != current_loop_id
    ):
        logger.debug(
            "Event loop changed (%s → %s); recreating Motor client",
            _async_client_loop_id, current_loop_id,
        )
        _async_client.close()
        _async_client = None
        _async_db = None
        _async_client_loop_id = None

    if _async_client is None:
        settings = get_settings()
        _async_client = AsyncIOMotorClient(
            settings.mongodb_url,
            serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
            connectTimeoutMS=settings.mongodb_connect_timeout_ms,
            maxPoolSize=settings.mongodb_max_pool_size,
            minPoolSize=settings.mongodb_min_pool_size,
        )
        _async_db = _async_client[settings.mongodb_db_name]
        if current_loop_id is not None:
            _async_client_loop_id = current_loop_id
        logger.info(
            "Connected to MongoDB: %s/%s (loop=%s)",
            settings.mongodb_url, settings.mongodb_db_name, current_loop_id,
        )
    return _async_db


async def close_database():
    """Close the async MongoDB connection."""
    global _async_client, _async_db, _async_client_loop_id
    if _async_client:
        _async_client.close()
        _async_client = None
        _async_db = None
        _async_client_loop_id = None
        logger.info("MongoDB async connection closed")


def get_sync_database():
    """Get or create a sync PyMongo database instance (for non-async contexts like PyTorch Dataset)."""
    global _sync_client, _sync_db
    if _sync_client is None:
        settings = get_settings()
        _sync_client = MongoClient(
            settings.mongodb_url,
            serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
            connectTimeoutMS=settings.mongodb_connect_timeout_ms,
        )
        _sync_db = _sync_client[settings.mongodb_db_name]
    return _sync_db


async def check_connection() -> dict:
    """Check MongoDB connection health."""
    try:
        db = await get_database()
        await db.command("ping")
        settings = get_settings()
        return {"status": "connected", "url": settings.mongodb_url, "database": settings.mongodb_db_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}
