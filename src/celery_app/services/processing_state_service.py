"""Processing state tracking service.

This module provides timestamp tracking to avoid reprocessing the same data:
- Track last processed timestamp per user per process type
- Support incremental data fetching
- Enable resume from last position after restart
"""

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from src.database import get_database

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")

# Collection name for user processing state
USER_PROCESSING_STATE_COLLECTION = "user_processing_state"


# =============================================================================
# User State Management
# =============================================================================


async def get_user_state(user: str) -> Optional[dict]:
    """Get user's processing state from database."""
    db = await get_database()
    try:
        return await db[USER_PROCESSING_STATE_COLLECTION].find_one({"_id": user})
    except Exception as e:
        logger.debug(f"Error getting user state for {user}: {e}")
    return None


async def get_last_processed(
    user: str,
    process_type: str,
) -> Optional[datetime]:
    """Get last processed timestamp for a user and process type."""
    db = await get_database()
    column_name = f"last_{process_type}_timestamp"

    try:
        doc = await db[USER_PROCESSING_STATE_COLLECTION].find_one(
            {"_id": user},
            {column_name: 1},
        )
        if doc and doc.get(column_name):
            ts = doc[column_name]
            if isinstance(ts, datetime):
                return ts
            if isinstance(ts, str):
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return ts
    except Exception as e:
        logger.debug(f"Error getting last processed timestamp for {user}/{process_type}: {e}")
    return None


async def update_last_processed(
    user: str,
    process_type: str,
    timestamp: datetime,
) -> bool:
    """Update last processed timestamp for a user and process type."""
    db = await get_database()
    column_name = f"last_{process_type}_timestamp"

    try:
        await db[USER_PROCESSING_STATE_COLLECTION].update_one(
            {"_id": user},
            {"$set": {column_name: timestamp, "updated_at": datetime.now(CHINA_TZ)}},
            upsert=True,
        )
        return True
    except Exception as e:
        logger.warning(f"Error updating last processed timestamp for {user}/{process_type}: {e}")
        return False


async def set_data_collection_start(user: str) -> datetime:
    """Set or get the data collection start time for a user."""
    db = await get_database()
    state = await get_user_state(user)
    now = datetime.now(CHINA_TZ)

    if state and state.get("data_collection_start"):
        ts = state["data_collection_start"]
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return ts

    try:
        await db[USER_PROCESSING_STATE_COLLECTION].update_one(
            {"_id": user},
            {"$setOnInsert": {
                "user": user,
                "data_collection_start": now,
                "updated_at": now,
            }},
            upsert=True,
        )
    except Exception as e:
        logger.warning(f"Error setting data collection start for {user}: {e}")
    return now


async def update_last_summary_generated(user: str) -> bool:
    """Update the last summary generated timestamp for a user."""
    db = await get_database()
    now = datetime.now(CHINA_TZ)

    try:
        await db[USER_PROCESSING_STATE_COLLECTION].update_one(
            {"_id": user},
            {"$set": {"last_summary_generated": now, "updated_at": now}},
            upsert=True,
        )
        return True
    except Exception as e:
        logger.warning(f"Error updating last summary generated for {user}: {e}")
        return False


async def get_last_summary_generated(user: str) -> Optional[datetime]:
    """Get the last summary generated timestamp for a user."""
    return await get_last_processed(user, "summary")


# =============================================================================
# Data Fetching with Timestamp Filter
# =============================================================================


async def get_imu_window_since(user: str, since: datetime) -> list[dict]:
    """Fetch IMU data for a user since a given timestamp."""
    db = await get_database()
    cursor = db["imu"].find({
        "user": user,
        "timestamp": {"$gte": since},
    }).sort("timestamp", 1)
    return await cursor.to_list(None)


async def get_documents_since(user: str, since: datetime) -> list[dict]:
    """Fetch document data for a user since a given timestamp."""
    db = await get_database()
    cursor = db["uploads"].find({
        "user": user,
        "timestamp": {"$gte": since},
    }).sort("timestamp", 1)
    return await cursor.to_list(None)


async def get_har_labels_since(user: str, since: datetime) -> list[dict]:
    """Fetch HAR labels for a user since a given timestamp."""
    db = await get_database()
    cursor = db["har"].find({
        "user": user,
        "timestamp": {"$gte": since},
    }).sort("timestamp", 1)
    return await cursor.to_list(None)
