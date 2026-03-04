"""Business logic for generating summary logs.

This module handles:
- Compressing atomic activities into summaries
- Creating daily/hourly summary logs
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from supabase import Client

from src.database import get_supabase_client
from src.llm_utils.services import generate_structured_output
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")


# ============================================================================
# Data Compression
# ============================================================================


async def compress_atomic_activities(
    user: str,
    hours: int = 1,
    client: Client | None = None,
) -> dict:
    """
    Compress atomic activities for a user over a time period.

    Creates a summary of activities, counts, and patterns from raw atomic data.

    Args:
        user: User identifier
        hours: Number of hours to look back
        client: Optional Supabase client

    Returns:
        Compressed activity summary dict
    """
    if client is None:
        client = get_supabase_client()

    cutoff_time = datetime.now(CHINA_TZ) - timedelta(hours=hours)

    # Fetch atomic activities
    response = await asyncio.to_thread(
        lambda: client.table("atomic_activities")
        .select("*")
        .eq("user", user)
        .gte("timestamp", cutoff_time.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    activities = response.data if response.data else []

    if not activities:
        return {
            "user": user,
            "period_hours": hours,
            "total_records": 0,
            "summary": {},
        }

    # Count occurrences of each label type
    har_counts = {}
    app_counts = {}
    step_counts = {}
    phone_counts = {}
    social_counts = {}
    movement_counts = {}
    location_counts = {}

    # DB columns: har_label, app_category, step_count, phone_usage, social, movement, location
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

    # Find dominant labels
    def get_dominant(counts: dict) -> tuple[Optional[str], int]:
        if not counts:
            return None, 0
        max_label = max(counts, key=counts.get)
        return max_label, counts[max_label]

    compressed = {
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

    return compressed


async def get_all_users_with_activities(
    hours: int = 1,
    client: Client | None = None,
) -> list[str]:
    """
    Get list of users who have atomic activities in the last X hours.

    Args:
        hours: Number of hours to look back
        client: Optional Supabase client

    Returns:
        List of user identifiers
    """
    if client is None:
        client = get_supabase_client()

    cutoff_time = datetime.now(CHINA_TZ) - timedelta(hours=hours)

    response = await asyncio.to_thread(
        lambda: client.table("atomic_activities")
        .select("user")
        .gte("timestamp", cutoff_time.isoformat())
        .execute()
    )

    if not response.data:
        return []

    # Extract unique users
    users = list(set(item["user"] for item in response.data))
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
    """
    Generate a summary log based on compressed activity data.

    Uses LLM to create a readable summary of the user's activities.

    Args:
        user: User identifier
        compressed_data: Compressed activity summary
        log_type: Type of summary ("hourly" or "daily")

    Returns:
        Summary log dict
    """
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
- recommendations: list of 1-2 suggestions
"""

    # Convert dicts to JSON and escape curly braces for ChatPromptTemplate
    def fmt(d):
        return json.dumps(d).replace("{", "{{").replace("}", "}}")

    user_prompt = f"""User activity summary for the past {period_desc}:

Activity patterns: {fmt(summary.get('har', {}))}
App usage: {fmt(summary.get('app_usage', {}))}
Phone usage: {fmt(summary.get('phone_usage', {}))}
Social context: {fmt(summary.get('social', {}))}
Movement: {fmt(summary.get('movement', {}))}
Location: {fmt(summary.get('location', {}))}

Dominant activity: {dominant.get('activity')}
Dominant app category: {dominant.get('app_category')}
Dominant location: {dominant.get('location')}

Total activity records: {compressed_data.get('total_records')}

Create a summary of this user's {period_desc}."""

    try:
        result = await generate_structured_output(
            system_prompt,
            user_prompt,
            SummaryOutput,
            temperature=0.4,
        )

        # Build full summary text including title, highlights, and recommendations
        full_summary = f"{result.title}\n\n{result.summary}"
        if result.highlights:
            full_summary += f"\n\nHighlights: {', '.join(result.highlights)}"
        if result.recommendations:
            full_summary += f"\n\nRecommendations: {', '.join(result.recommendations)}"

        # Return only fields that match the database schema
        return {
            "user": user,
            "log_type": log_type,
            "summary": full_summary,
            "start_timestamp": compressed_data.get("start_time"),
            "end_timestamp": compressed_data.get("end_time"),
        }
    except Exception as e:
        logger.error(f"Error generating summary for user {user}: {e}")
        return None


async def insert_summary_log(
    summary_log: dict,
    client: Client | None = None,
) -> dict:
    """
    Insert summary log into the database.

    Args:
        summary_log: Summary log data to insert
        client: Optional Supabase client

    Returns:
        Inserted record data
    """
    if client is None:
        client = get_supabase_client()

    response = await asyncio.to_thread(
        lambda: client.table("summary_logs").insert(summary_log).execute()
    )

    return response.data[0] if response.data else {}