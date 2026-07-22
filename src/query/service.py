"""Business logic for querying summary logs, interventions, and atomic activities."""

from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from bson import ObjectId

from src.database import get_database
from src.query.constants import (
    SUMMARY_LOGS_COLLECTION,
    INTERVENTIONS_COLLECTION,
    INTERVENTION_FEEDBACKS_COLLECTION,
    SUMMARY_LOG_FEEDBACKS_COLLECTION,
    ATOMIC_ACTIVITIES_COLLECTION,
)
from src.query.atomic_encoding import encode_atomic_activities

CHINA_TZ = ZoneInfo("Asia/Shanghai")
MAX_ATOMIC_DOCS = 50_000        # Hard limit on documents returned
QUERY_MAX_TIME_MS = 15_000      # 15-second query timeout


def _serialize_doc(doc: dict) -> dict:
    """Convert MongoDB ObjectId to string for JSON serialization."""
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
    return doc


async def get_summary_logs(
    user: str,
    log_type: str,
    last_log_id: Optional[str] = None,
) -> tuple[Optional[dict], bool]:
    """
    Fetch the most recent summary log for a user.

    Supports polling mechanism: if last_log_id is provided, returns None
    if the latest log ID matches (meaning no new log since last check).

    Args:
        user: User identifier
        log_type: Type of summary log (hourly or daily)
        last_log_id: Optional ID (hex string) of the last received log.

    Returns:
        Tuple of (log record or None, has_new_log boolean)
    """
    db = await get_database()

    doc = await db[SUMMARY_LOGS_COLLECTION].find_one(
        {"user": user, "log_type": log_type},
        sort=[("timestamp", -1)],
    )

    if not doc:
        return None, False

    latest_id = str(doc["_id"])

    # If last_log_id provided, check if there's a newer log
    if last_log_id is not None:
        if latest_id == last_log_id:
            return None, False
        return _serialize_doc(doc), True

    return _serialize_doc(doc), True


async def get_interventions(user: str) -> Optional[dict]:
    """
    Fetch the most recent intervention for a user.

    Args:
        user: User identifier

    Returns:
        The most recent intervention record, or None if not found
    """
    db = await get_database()

    doc = await db[INTERVENTIONS_COLLECTION].find_one(
        {"user": user},
        sort=[("timestamp", -1)],
    )

    return _serialize_doc(doc) if doc else None


def format_summary_log(record: dict) -> dict:
    """
    Format a summary log record for API response.
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
    intervention_id: str,
    feedback: str,
    mc1: Optional[str] = None,
    mc2: Optional[str] = None,
    mc3: Optional[str] = None,
    mc4: Optional[str] = None,
    mc5: Optional[str] = None,
    mc6: Optional[str] = None,
) -> dict:
    """
    Submit intervention feedback to the database.
    """
    db = await get_database()

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
        "timestamp": datetime.now(CHINA_TZ),
    }

    result = await db[INTERVENTION_FEEDBACKS_COLLECTION].insert_one(data)
    data["_id"] = result.inserted_id
    return _serialize_doc(data)


async def submit_summary_log_feedback(
    user: str,
    summary_logs_id: str,
    feedback: Optional[str] = None,
    q1: Optional[str] = None,
    q2: Optional[str] = None,
    q2_preference: Optional[str] = None,
    ground_truth: Optional[str] = None,
    suggestions: Optional[str] = None,
) -> dict:
    """
    Submit summary log feedback to the database.
    """
    db = await get_database()

    data = {
        "user": user,
        "summary_logs_id": summary_logs_id,
        "feedback": feedback,
        "q1": q1,
        "q2": q2,
        "q2_preference": q2_preference,
        "ground_truth": ground_truth,
        "suggestions": suggestions,
        "timestamp": datetime.now(CHINA_TZ),
    }

    result = await db[SUMMARY_LOG_FEEDBACKS_COLLECTION].insert_one(data)
    data["_id"] = result.inserted_id
    return _serialize_doc(data)


# ============================================================================
# Atomic Activities
# ============================================================================


async def get_atomic_activities(
    user: str,
    duration: int,
) -> dict:
    """
    Fetch atomic activities for a user within a duration.

    Args:
        user: User identifier
        duration: Duration in seconds since last fetch (0 for all)

    Returns:
        Dictionary with grouped atomic activity values
    """
    db = await get_database()

    query = {"user": user}

    # Apply time filter if duration > 0
    if duration > 0:
        cutoff_time = datetime.now(CHINA_TZ) - timedelta(seconds=duration)
        query["timestamp"] = {"$gte": cutoff_time}

    cursor = db[ATOMIC_ACTIVITIES_COLLECTION].find(query).sort("timestamp", 1).max_time_ms(QUERY_MAX_TIME_MS)
    docs = await cursor.to_list(MAX_ATOMIC_DOCS)

    if not docs:
        return {
            "sport": [],
            "appCategory": [],
            "location": [],
            "movement": [],
            "stepCategory": [],
            "phoneCategory": [],
            "start_timestamp": None,
            "end_timestamp": None,
        }

    # Extract timestamp range from the queried documents
    start_timestamp = docs[0].get("timestamp") if docs else None
    end_timestamp = docs[-1].get("timestamp") if docs else None

    # Group data by field names
    sport_labels = []
    app_categories = []
    locations = []
    movements = []
    step_labels = []
    phone_usages = []

    for record in docs:
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
        "start_timestamp": start_timestamp.isoformat() if start_timestamp else None,
        "end_timestamp": end_timestamp.isoformat() if end_timestamp else None,
    }


async def get_atomic_activities_encoded(
    user: str,
    duration: int,
) -> dict:
    """
    Fetch atomic activities for a user and encode them into Level-1 and Level-2 formats.
    """
    db = await get_database()

    query = {"user": user}

    if duration > 0:
        cutoff_time = datetime.now(CHINA_TZ) - timedelta(seconds=duration)
        query["timestamp"] = {"$gte": cutoff_time}

    cursor = db[ATOMIC_ACTIVITIES_COLLECTION].find(query).sort("timestamp", 1).max_time_ms(QUERY_MAX_TIME_MS)
    docs = await cursor.to_list(MAX_ATOMIC_DOCS)

    if not docs:
        return encode_atomic_activities([])

    return encode_atomic_activities(docs)
