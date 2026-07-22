"""Business logic for generating summary logs."""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from src.database import get_database
from src.llm_utils.services import generate_structured_output
from src.celery_app.config import (
    MIN_ATOMIC_RECORDS_FOR_HOURLY_LOG,
    MIN_UNIQUE_LABELS_FOR_LOG,
    MIN_DATA_COLLECTION_HOURS,
    MIN_HOURS_BETWEEN_SUMMARIES,
)
from src.celery_app.services.processing_state_service import (
    get_user_state,
    set_data_collection_start,
    update_last_summary_generated,
)

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")


# ============================================================================
# Data Compression
# ============================================================================


async def compress_atomic_activities(
    user: str,
    hours: int = 1,
) -> dict:
    """Compress atomic activities for a user over a time period."""
    db = await get_database()
    cutoff_time = datetime.now(CHINA_TZ) - timedelta(hours=hours)

    cursor = db["atomic_activities"].find({
        "user": user,
        "timestamp": {"$gte": cutoff_time},
    }).sort("timestamp", 1)

    activities = await cursor.to_list(None)

    if not activities:
        return {
            "user": user,
            "period_hours": hours,
            "total_records": 0,
            "summary": {},
        }

    har_counts = {}
    app_counts = {}
    step_counts = {}
    phone_counts = {}
    social_counts = {}
    movement_counts = {}
    location_counts = {}

    for activity in activities:
        if activity.get("har_label"):
            har_counts[activity["har_label"]] = har_counts.get(activity["har_label"], 0) + 1
        if activity.get("app_category"):
            app_counts[activity["app_category"]] = app_counts.get(activity["app_category"], 0) + 1
        step_val = activity.get("step_count") or activity.get("step_label")
        if step_val:
            step_counts[step_val] = step_counts.get(step_val, 0) + 1
        if activity.get("phone_usage"):
            phone_counts[activity["phone_usage"]] = phone_counts.get(activity["phone_usage"], 0) + 1
        social_val = activity.get("social") or activity.get("social_label")
        if social_val:
            social_counts[social_val] = social_counts.get(social_val, 0) + 1
        movement_val = activity.get("movement") or activity.get("movement_label")
        if movement_val:
            movement_counts[movement_val] = movement_counts.get(movement_val, 0) + 1
        location_val = activity.get("location") or activity.get("location_label")
        if location_val:
            location_counts[location_val] = location_counts.get(location_val, 0) + 1

    def get_dominant(counts: dict) -> tuple[Optional[str], int]:
        if not counts:
            return None, 0
        max_label = max(counts, key=counts.get)
        return max_label, counts[max_label]

    return {
        "user": user,
        "period_hours": hours,
        "total_records": len(activities),
        "start_time": cutoff_time.isoformat(),
        "end_time": datetime.now(CHINA_TZ).isoformat(),
        "summary": {
            "har": dict(sorted(har_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "app_usage": dict(sorted(app_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "step_activity": dict(sorted(step_counts.items(), key=lambda x: x[1], reverse=True)[:3]),
            "phone_usage": dict(sorted(phone_counts.items(), key=lambda x: x[1], reverse=True)[:3]),
            "social": dict(sorted(social_counts.items(), key=lambda x: x[1], reverse=True)[:3]),
            "movement": dict(sorted(movement_counts.items(), key=lambda x: x[1], reverse=True)[:3]),
            "location": dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:3]),
        },
        "dominant": {
            "activity": get_dominant(har_counts)[0],
            "app_category": get_dominant(app_counts)[0],
            "location": get_dominant(location_counts)[0],
        },
    }


async def get_all_users_with_activities(hours: int = 1) -> list[str]:
    """Get list of users who have atomic activities in the last X hours."""
    db = await get_database()
    cutoff_time = datetime.now(CHINA_TZ) - timedelta(hours=hours)

    cursor = db["atomic_activities"].find(
        {"timestamp": {"$gte": cutoff_time}},
        {"user": 1},
    )

    docs = await cursor.to_list(None)
    users = list(set(d["user"] for d in docs if "user" in d))
    return users


# ============================================================================
# Summary Log Generation
# ============================================================================


class SummaryOutput(BaseModel):
    """Structured output for summary generation."""
    title: str
    summary: str
    highlights: list[str]
    recommendations: list[str]


async def generate_summary(
    user: str,
    compressed_data: dict,
    log_type: str = "hourly",
) -> Optional[dict]:
    """Generate a summary log based on compressed activity data."""
    if not compressed_data.get("total_records"):
        return None

    summary = compressed_data.get("summary", {})
    dominant = compressed_data.get("dominant", {})

    period_desc = "hour" if log_type == "hourly" else "day"

    system_prompt = f"""You are a lifestyle analyst. Create a concise, engaging summary
of the user's activities over the past {period_desc}. Include:
- A brief title summarizing the main theme
- A 2-3 sentence narrative summary
- 2-3 key highlights (interesting patterns or notable activities)
- 1-2 gentle recommendations for improvement

Return a JSON object with:
- title: a catchy title (5-8 words)
- summary: narrative description (2-3 sentences)
- highlights: list of 2-3 notable points
- recommendations: list of 1-2 suggestions"""

    user_prompt = f"""User activity summary for the past {period_desc}:

Activity patterns: {json.dumps(summary.get('har', {}))}
App usage: {json.dumps(summary.get('app_usage', {}))}
Phone usage: {json.dumps(summary.get('phone_usage', {}))}
Social context: {json.dumps(summary.get('social', {}))}
Movement: {json.dumps(summary.get('movement', {}))}
Location: {json.dumps(summary.get('location', {}))}

Dominant activity: {dominant.get('activity')}
Dominant app category: {dominant.get('app_category')}
Dominant location: {dominant.get('location')}

Total activity records: {compressed_data.get('total_records')}

Create a summary of this user's {period_desc}."""

    try:
        result = await generate_structured_output(
            system_prompt, user_prompt, SummaryOutput, temperature=0.4,
        )

        full_summary = f"{result.title}\n\n{result.summary}"
        if result.highlights:
            full_summary += f"\n\nHighlights: {', '.join(result.highlights)}"
        if result.recommendations:
            full_summary += f"\n\nRecommendations: {', '.join(result.recommendations)}"

        return {
            "user": user,
            "log_type": log_type,
            "summary": full_summary,
            "start_timestamp": compressed_data.get("start_time"),
            "end_timestamp": compressed_data.get("end_time"),
            "timestamp": datetime.now(CHINA_TZ),
        }
    except Exception as e:
        logger.error(f"Error generating summary for user {user}: {e}")
        return None


async def insert_summary_log(summary_log: dict) -> dict:
    """Insert summary log into the database."""
    db = await get_database()
    result = await db["summary_logs"].insert_one(summary_log)
    summary_log["_id"] = result.inserted_id
    return summary_log


# ============================================================================
# Threshold-Based Summary Generation
# ============================================================================


async def should_generate_summary(user: str, hours: int) -> tuple[bool, dict]:
    """Check if user has enough data to generate a meaningful summary."""
    compressed = await compress_atomic_activities(user, hours)

    total_records = compressed.get("total_records", 0)
    if total_records < MIN_ATOMIC_RECORDS_FOR_HOURLY_LOG:
        logger.debug(f"User {user} has {total_records} records, need {MIN_ATOMIC_RECORDS_FOR_HOURLY_LOG}")
        return False, compressed

    unique_activities = set()
    summary = compressed.get("summary", {})
    for label_list in summary.values():
        if isinstance(label_list, dict):
            unique_activities.update(label_list.keys())

    if len(unique_activities) < MIN_UNIQUE_LABELS_FOR_LOG:
        logger.debug(f"User {user} has {len(unique_activities)} unique activities, need {MIN_UNIQUE_LABELS_FOR_LOG}")
        return False, compressed

    return True, compressed


async def check_user_hourly_ready(user: str) -> tuple[bool, str]:
    """Check if user is ready for hourly summary generation."""
    state = await get_user_state(user)
    now = datetime.now(CHINA_TZ)

    if state and state.get("data_collection_start"):
        data_start = state["data_collection_start"]
        if isinstance(data_start, str):
            data_start = datetime.fromisoformat(data_start.replace("Z", "+00:00"))
        hours_since_start = (now - data_start).total_seconds() / 3600
        if hours_since_start < MIN_DATA_COLLECTION_HOURS:
            return False, f"Data collection started {hours_since_start:.1f}h ago, need {MIN_DATA_COLLECTION_HOURS}h"
    else:
        await set_data_collection_start(user)
        return False, "Data collection just started"

    if state and state.get("last_summary_generated"):
        last_summary = state["last_summary_generated"]
        if isinstance(last_summary, str):
            last_summary = datetime.fromisoformat(last_summary.replace("Z", "+00:00"))
        hours_since_summary = (now - last_summary).total_seconds() / 3600
        if hours_since_summary < MIN_HOURS_BETWEEN_SUMMARIES:
            return False, f"Last summary {hours_since_summary:.1f}h ago, need {MIN_HOURS_BETWEEN_SUMMARIES}h gap"

    has_enough, _ = await should_generate_summary(user, 1)
    if not has_enough:
        return False, "Insufficient data threshold"

    return True, "Ready for summary generation"


async def generate_summary_for_user(
    user: str,
    hours: int,
    log_type: str,
) -> tuple[Optional[dict], Optional[str]]:
    """Generate summary for a user with threshold checks and timestamp tracking.

    Returns:
        Tuple of (summary_log_dict, skip_reason).
        - On success: (summary_dict, None)
        - On skip: (None, reason_string)
        - On generation failure: (None, "Summary generation failed")
    """
    if log_type == "hourly":
        is_ready, reason = await check_user_hourly_ready(user)
        if not is_ready:
            logger.info(f"Skipping {user}: {reason}")
            return None, reason

    has_enough, compressed = await should_generate_summary(user, hours)
    if not has_enough:
        total_records = compressed.get("total_records", 0)
        reason = f"Insufficient data (records={total_records}, need={MIN_ATOMIC_RECORDS_FOR_HOURLY_LOG})"
        logger.info(f"Skipping {user}: {reason}")
        return None, reason

    summary_log = await generate_summary(user, compressed, log_type=log_type)

    if summary_log:
        await insert_summary_log(summary_log)
        await update_last_summary_generated(user)
        logger.info(f"Generated {log_type} summary for {user}")
        return summary_log, None

    return None, "Summary generation failed"
