"""Processing state tracking service.

This module provides timestamp tracking to avoid reprocessing the same data:
- Track last processed timestamp per user per process type
- Support incremental data fetching
- Enable resume from last position after restart
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from supabase import Client

from src.database import get_supabase_client

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")

# Table name for user processing state
USER_PROCESSING_STATE_TABLE = "user_processing_state"


# =============================================================================
# User State Management
# =============================================================================


async def get_user_state(
    user: str,
    client: Client | None = None,
) -> Optional[dict]:
    """Get user's processing state from database.

    Args:
        user: User identifier
        client: Optional Supabase client

    Returns:
        User state dict or None if not found
    """
    if client is None:
        client = get_supabase_client()

    try:
        response = await asyncio.to_thread(
            lambda: client.table(USER_PROCESSING_STATE_TABLE)
            .select("*")
            .eq("user", user)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]
    except Exception as e:
        logger.debug(f"Error getting user state for {user}: {e}")
    return None


async def get_last_processed(
    user: str,
    process_type: str,
    client: Client | None = None,
) -> Optional[datetime]:
    """Get last processed timestamp for a user and process type.

    Args:
        user: User identifier
        process_type: Type of process ('har', 'atomic', 'upload', 'summary')
        client: Optional Supabase client

    Returns:
        Last processed timestamp or None
    """
    if client is None:
        client = get_supabase_client()

    column_name = f"last_{process_type}_timestamp"

    try:
        response = await asyncio.to_thread(
            lambda: client.table(USER_PROCESSING_STATE_TABLE)
            .select(column_name)
            .eq("user", user)
            .limit(1)
            .execute()
        )
        if response.data and response.data[0].get(column_name):
            timestamp_str = response.data[0][column_name]
            # Parse ISO timestamp
            if isinstance(timestamp_str, str):
                return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return timestamp_str
    except Exception as e:
        logger.debug(f"Error getting last processed timestamp for {user}/{process_type}: {e}")
    return None


async def update_last_processed(
    user: str,
    process_type: str,
    timestamp: datetime,
    client: Client | None = None,
) -> bool:
    """Update last processed timestamp for a user and process type.

    Uses upsert to create or update the record.

    Args:
        user: User identifier
        process_type: Type of process ('har', 'atomic', 'upload', 'summary')
        timestamp: The timestamp to record
        client: Optional Supabase client

    Returns:
        True if successful
    """
    if client is None:
        client = get_supabase_client()

    column_name = f"last_{process_type}_timestamp"

    try:
        await asyncio.to_thread(
            lambda: client.table(USER_PROCESSING_STATE_TABLE)
            .upsert({
                "user": user,
                column_name: timestamp.isoformat(),
                "updated_at": datetime.now(CHINA_TZ).isoformat(),
            }, on_conflict="user")
            .execute()
        )
        return True
    except Exception as e:
        logger.warning(f"Error updating last processed timestamp for {user}/{process_type}: {e}")
        return False


async def set_data_collection_start(
    user: str,
    client: Client | None = None,
) -> datetime:
    """Set or get the data collection start time for a user.

    If not already set, creates a new record with current time.
    Returns the data collection start time.

    Args:
        user: User identifier
        client: Optional Supabase client

    Returns:
        Data collection start timestamp
    """
    if client is None:
        client = get_supabase_client()

    # Check if already exists
    state = await get_user_state(user, client)
    if state and state.get("data_collection_start"):
        timestamp_str = state["data_collection_start"]
        if isinstance(timestamp_str, str):
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return timestamp_str

    # Create new record with current time
    now = datetime.now(CHINA_TZ)
    try:
        await asyncio.to_thread(
            lambda: client.table(USER_PROCESSING_STATE_TABLE)
            .upsert({
                "user": user,
                "data_collection_start": now.isoformat(),
                "updated_at": now.isoformat(),
            }, on_conflict="user")
            .execute()
        )
    except Exception as e:
        logger.warning(f"Error setting data collection start for {user}: {e}")
    return now


async def update_last_summary_generated(
    user: str,
    client: Client | None = None,
) -> bool:
    """Update the last summary generated timestamp for a user.

    Args:
        user: User identifier
        client: Optional Supabase client

    Returns:
        True if successful
    """
    now = datetime.now(CHINA_TZ)
    if client is None:
        client = get_supabase_client()

    try:
        await asyncio.to_thread(
            lambda: client.table(USER_PROCESSING_STATE_TABLE)
            .upsert({
                "user": user,
                "last_summary_generated": now.isoformat(),
                "updated_at": now.isoformat(),
            }, on_conflict="user")
            .execute()
        )
        return True
    except Exception as e:
        logger.warning(f"Error updating last summary generated for {user}: {e}")
        return False


async def get_last_summary_generated(
    user: str,
    client: Client | None = None,
) -> Optional[datetime]:
    """Get the last summary generated timestamp for a user.

    Args:
        user: User identifier
        client: Optional Supabase client

    Returns:
        Last summary timestamp or None
    """
    return await get_last_processed(user, "summary", client)


# =============================================================================
# Data Fetching with Timestamp Filter
# =============================================================================


async def get_imu_window_since(
    user: str,
    since: datetime,
    client: Client | None = None,
) -> list[dict]:
    """Fetch IMU data for a user since a given timestamp.

    Used for incremental processing - only fetches new data since last processed.

    Args:
        user: User identifier
        since: Only fetch data after this timestamp
        client: Optional Supabase client

    Returns:
        List of IMU data records
    """
    if client is None:
        client = get_supabase_client()

    response = await asyncio.to_thread(
        lambda: client.table("imu")
        .select("*")
        .eq("user", user)
        .gte("timestamp", since.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    return response.data if response.data else []


async def get_documents_since(
    user: str,
    since: datetime,
    client: Client | None = None,
) -> list[dict]:
    """Fetch document data for a user since a given timestamp.

    Used for incremental processing - only fetches new data since last processed.

    Args:
        user: User identifier
        since: Only fetch data after this timestamp
        client: Optional Supabase client

    Returns:
        List of document records
    """
    if client is None:
        client = get_supabase_client()

    response = await asyncio.to_thread(
        lambda: client.table("uploads")
        .select("*")
        .eq("user", user)
        .gte("timestamp", since.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    return response.data if response.data else []


async def get_har_labels_since(
    user: str,
    since: datetime,
    client: Client | None = None,
) -> list[dict]:
    """Fetch HAR labels for a user since a given timestamp.

    Args:
        user: User identifier
        since: Only fetch data after this timestamp
        client: Optional Supabase client

    Returns:
        List of HAR records
    """
    if client is None:
        client = get_supabase_client()

    response = await asyncio.to_thread(
        lambda: client.table("har")
        .select("*")
        .eq("user", user)
        .gte("timestamp", since.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    return response.data if response.data else []