"""Business logic for generating health interventions.

This module handles:
- Generating personalized health interventions via LLM
- Inserting interventions into the database
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from supabase import Client
from pydantic import BaseModel

from src.database import get_supabase_client
from src.llm_utils.services import generate_structured_output

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")


class InterventionOutput(BaseModel):
    """Structured output for intervention generation."""
    intervention_type: str
    message: str
    priority: str  # low, medium, high
    category: str  # physical, mental, social, digital_wellbeing


async def get_recent_summaries(
    hours: int = 1,
    client: Client | None = None,
) -> list[dict]:
    """
    Get recent summary logs from the last X hours.

    Args:
        hours: Number of hours to look back
        client: Optional Supabase client

    Returns:
        List of summary logs with user and summary data
    """
    if client is None:
        client = get_supabase_client()

    cutoff_time = datetime.now(CHINA_TZ) - timedelta(hours=hours)

    response = await asyncio.to_thread(
        lambda: client.table("summary_logs")
        .select("*")
        .gte("timestamp", cutoff_time.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    return response.data if response.data else []


async def generate_intervention_from_summary(
    user: str,
    summary_log: dict,
) -> Optional[dict]:
    """
    Generate a health intervention based on a summary log.

    Uses LLM to analyze the summary and suggest personalized interventions.

    Args:
        user: User identifier
        summary_log: Summary log dict from summary_logs table

    Returns:
        Intervention dict with message and metadata
    """
    if not summary_log:
        return None

    system_prompt = """You are a health and wellness advisor. Analyze the user's activity summary
and suggest a helpful, personalized intervention. Consider:
- Physical activity levels
- Screen time and phone usage
- Social interaction patterns
- Location context
- App usage patterns

Generate a specific, actionable intervention that could help improve the user's wellbeing.
The intervention should be encouraging and not judgmental.

Return a JSON object with:
- intervention_type: brief type (e.g., "movement_reminder", "screen_break", "social_encouragement")
- message: a friendly, specific message (1-2 sentences)
- priority: "low", "medium", or "high" based on urgency
- category: "physical", "mental", "social", or "digital_wellbeing"
"""

    user_prompt = f"""User activity summary:

Title: {summary_log.get('title', 'No title')}

Summary: {summary_log.get('summary', 'No summary available')}

Key Highlights: {summary_log.get('highlights', [])}

Dominant Activities: {summary_log.get('dominant_activities', {})}

Activity Counts: {summary_log.get('activity_counts', {})}

Recommendations from summary: {summary_log.get('recommendations', [])}

Period: {summary_log.get('period_hours', 1)} hour(s)

Suggest an appropriate health intervention based on this summary."""

    try:
        result = await generate_structured_output(
            system_prompt,
            user_prompt,
            InterventionOutput,
            temperature=0.3,
        )

        return {
            "user": user,
            "intervention_type": result.intervention_type,
            "message": result.message,
            "priority": result.priority,
            "category": result.category,
            "based_on_data": summary_log.get("dominant_activities", {}),
            "timestamp": datetime.now(CHINA_TZ).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error generating intervention for user {user}: {e}")
        # Fallback intervention
        return {
            "user": user,
            "intervention_type": "general_wellbeing",
            "message": "Take a moment to check in with yourself. How are you feeling?",
            "priority": "low",
            "category": "mental",
            "based_on_data": {},
            "timestamp": datetime.now(CHINA_TZ).isoformat(),
        }


async def insert_intervention(
    intervention: dict,
    client: Client | None = None,
) -> dict:
    """
    Insert intervention into the database.

    Args:
        intervention: Intervention data to insert
        client: Optional Supabase client

    Returns:
        Inserted record data
    """
    import asyncio

    if client is None:
        client = get_supabase_client()

    response = await asyncio.to_thread(
        lambda: client.table("interventions").insert(intervention).execute()
    )

    return response.data[0] if response.data else {}