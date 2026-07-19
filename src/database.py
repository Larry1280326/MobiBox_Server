"""MongoDB database connection using Motor (async) and PyMongo (sync fallback)."""

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient

from src.config import get_settings

logger = logging.getLogger(__name__)

# Async client (Motor) — used by FastAPI routes and Celery services
_async_client: Optional[AsyncIOMotorClient] = None
_async_db: Optional[AsyncIOMotorDatabase] = None

# Sync client (PyMongo) — used by imu_dataset.py (PyTorch Dataset __init__)
_sync_client: Optional[MongoClient] = None
_sync_db = None


async def get_database() -> AsyncIOMotorDatabase:
    """Get or create the async Motor MongoDB database instance."""
    global _async_client, _async_db
    if _async_client is None:
        settings = get_settings()
        _async_client = AsyncIOMotorClient(settings.mongodb_url)
        _async_db = _async_client[settings.mongodb_db_name]
        logger.info(f"Connected to MongoDB: {settings.mongodb_url}/{settings.mongodb_db_name}")
    return _async_db


async def close_database():
    """Close the async MongoDB connection."""
    global _async_client, _async_db
    if _async_client:
        _async_client.close()
        _async_client = None
        _async_db = None
        logger.info("MongoDB async connection closed")


def get_sync_database():
    """Get or create a sync PyMongo database instance (for non-async contexts like PyTorch Dataset)."""
    global _sync_client, _sync_db
    if _sync_client is None:
        settings = get_settings()
        _sync_client = MongoClient(settings.mongodb_url)
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
