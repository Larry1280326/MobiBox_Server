"""Business logic for querying summary logs, interventions, and atomic activities."""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from supabase import Client

from src.database import get_supabase_client
from src.query.constants import (
    SUMMARY_LOGS_TABLE,
    INTERVENTIONS_TABLE,
    INTERVENTION_FEEDBACKS_TABLE,
    SUMMARY_LOG_FEEDBACKS_TABLE,
    ATOMIC_ACTIVITIES_TABLE,
)

CHINA_TZ = ZoneInfo("Asia/Shanghai")


async def get_summary_logs(
    user: str,
    log_type: str,
    last_log_id: Optional[int] = None,
    client: Client | None = None,
) -> tuple[Optional[dict], bool]:
    """
    Fetch the most recent summary log for a user from the database.

    Supports polling mechanism: if last_log_id is provided, returns None
    if the latest log ID matches (meaning no new log since last check).

    Args:
        user: User identifier
        log_type: Type of summary log (hourly or daily)
        last_log_id: Optional ID of the last received log. If provided,
                     returns (None, False) if no new log is available.
        client: Optional Supabase client

    Returns:
        Tuple of (log record or None, has_new_log boolean)
    """
    if client is None:
        client = get_supabase_client()

    # Get the most recent log
    response = await asyncio.to_thread(
        lambda: client.table(SUMMARY_LOGS_TABLE)
        .select("*")
        .eq("user", user)
        .eq("log_type", log_type)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return None, False

    latest_log = response.data[0]
    latest_id = latest_log.get("id")

    # If last_log_id provided, check if there's a newer log
    if last_log_id is not None:
        if latest_id == last_log_id:
            # No new log
            return None, False
        # There's a newer log
        return latest_log, True

    # No last_log_id provided, return latest
    return latest_log, True


async def get_interventions(
    user: str,
    client: Client | None = None,
) -> Optional[dict]:
    """
    Fetch the most recent intervention for a user from the database.

    Args:
        user: User identifier
        client: Optional Supabase client

    Returns:
        The most recent intervention record, or None if not found
    """
    if client is None:
        client = get_supabase_client()

    response = await asyncio.to_thread(
        lambda: client.table(INTERVENTIONS_TABLE)
        .select("*")
        .eq("user", user)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )

    return response.data[0] if response.data else None


def format_summary_log(record: dict) -> dict:
    """
    Format a summary log record for API response.

    Args:
        record: Raw database record

    Returns:
        Formatted record with expected field names
    """
    return {
        "id": record.get("id"),
        "log_content": record.get("summary", ""),
        "start_timestamp": record.get("start_timestamp"),
        "end_timestamp": record.get("end_timestamp"),
        "generation_timestamp": record.get("timestamp"),
    }


def format_intervention(record: dict) -> dict:
    """
    Format an intervention record for API response.

    Args:
        record: Raw database record

    Returns:
        Formatted record with expected field names
    """
    return {
        "id": record.get("id"),
        "intervention_content": record.get("intervention_content", ""),
        "start_timestamp": record.get("start_timestamp"),
        "end_timestamp": record.get("end_timestamp"),
        "generation_timestamp": record.get("timestamp"),
    }


async def submit_intervention_feedback(
    user: str,
    intervention_id: int,
    feedback: str,
    mc1: Optional[str] = None,
    mc2: Optional[str] = None,
    mc3: Optional[str] = None,
    mc4: Optional[str] = None,
    mc5: Optional[str] = None,
    mc6: Optional[str] = None,
    client: Client | None = None,
) -> dict:
    """
    Submit intervention feedback to the database.

    Args:
        user: User identifier
        intervention_id: ID of the intervention being rated
        feedback: Feedback text
        mc1-mc6: Optional multiple choice responses
        client: Optional Supabase client

    Returns:
        The inserted record
    """
    if client is None:
        client = get_supabase_client()

    data = {
        "user": user,
        "intervention_id": intervention_id,
        "feedback": feedback,
        "mc1": mc1,
        "mc2": mc2,
        "mc3": mc3,
        "mc4": mc4,
        "mc5": mc5,
        "mc6": mc6,
    }

    response = await asyncio.to_thread(
        lambda: client.table(INTERVENTION_FEEDBACKS_TABLE)
        .insert(data)
        .execute()
    )

    return response.data[0] if response.data else {}


async def submit_summary_log_feedback(
    user: int,
    summary_logs_id: int,
    feedback: Optional[str] = None,
    q1: Optional[str] = None,
    q2: Optional[str] = None,
    q3: Optional[str] = None,
    q4: Optional[str] = None,
    ground_truth: Optional[str] = None,
    suggestions: Optional[str] = None,
    client: Client | None = None,
) -> dict:
    """
    Submit summary log feedback to the database.

    Args:
        user: User ID
        summary_logs_id: ID of the summary log being rated
        feedback: Simple feedback text (for basic use cases)
        q1: Multiple choice answer for question 1
        q2: Multiple choice answer for question 2
        q3: Multiple choice answer for question 3
        q4: Multiple choice answer for question 4
        ground_truth: Standard answer provided by user
        suggestions: Optimization suggestions from user
        client: Optional Supabase client

    Returns:
        The inserted record
    """
    if client is None:
        client = get_supabase_client()

    data = {
        "user": user,
        "summary_logs_id": summary_logs_id,
        "feedback": feedback,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "ground_truth": ground_truth,
        "suggestions": suggestions,
    }

    response = await asyncio.to_thread(
        lambda: client.table(SUMMARY_LOG_FEEDBACKS_TABLE)
        .insert(data)
        .execute()
    )

    return response.data[0] if response.data else {}


# ============================================================================
# Atomic Activities
# ============================================================================


async def get_atomic_activities(
    user: str,
    duration: int,
    client: Client | None = None,
) -> dict:
    """
    Fetch atomic activities for a user within a duration.

    Args:
        user: User identifier
        duration: Duration in seconds since last fetch (0 for all)
        client: Optional Supabase client

    Returns:
        Dictionary with grouped atomic activity values
    """
    if client is None:
        client = get_supabase_client()

    # Build query
    query = client.table(ATOMIC_ACTIVITIES_TABLE).select("*").eq("user", user)

    # Apply time filter if duration > 0
    if duration > 0:
        cutoff_time = datetime.now(CHINA_TZ) - timedelta(seconds=duration)
        query = query.gte("timestamp", cutoff_time.isoformat())

    # Order by timestamp ascending
    query = query.order("timestamp", desc=False)

    response = await asyncio.to_thread(lambda: query.execute())

    if not response.data:
        return {
            "sport": [],
            "appCategory": [],
            "location": [],
            "movement": [],
            "stepCategory": [],
            "phoneCategory": [],
        }

    # Group data by field names
    sport_labels = []
    app_categories = []
    locations = []
    movements = []
    step_labels = []
    phone_usages = []

    for record in response.data:
        # Map database columns to frontend field names
        if record.get("har_label"):
            sport_labels.append(record["har_label"])
        if record.get("app_category"):
            app_categories.append(record["app_category"])
        if record.get("location"):
            locations.append(record["location"])
        if record.get("movement"):
            movements.append(record["movement"])
        if record.get("step_count"):
            step_labels.append(record["step_count"])
        if record.get("phone_usage"):
            phone_usages.append(record["phone_usage"])

    return {
        "sport": sport_labels,
        "appCategory": app_categories,
        "location": locations,
        "movement": movements,
        "stepCategory": step_labels,
        "phoneCategory": phone_usages,
    }