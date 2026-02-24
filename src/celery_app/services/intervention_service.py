"""Business logic for generating health interventions.

This module handles:
- Generating personalized health interventions via LLM
- Inserting interventions into the database
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from supabase import Client
from pydantic import BaseModel

from src.database import get_supabase_client
from src.llm_utils.services import generate_structured_output

logger = logging.getLogger(__name__)


class InterventionOutput(BaseModel):
    """Structured output for intervention generation."""
    intervention_type: str
    message: str
    priority: str  # low, medium, high
    category: str  # physical, mental, social, digital_wellbeing


async def generate_intervention(
    user: str,
    compressed_data: dict,
) -> Optional[dict]:
    """
    Generate a health intervention based on compressed activity data.

    Uses LLM to analyze activity patterns and suggest interventions.

    Args:
        user: User identifier
        compressed_data: Compressed activity summary

    Returns:
        Intervention dict with message and metadata
    """
    if not compressed_data.get("total_records"):
        return None

    summary = compressed_data.get("summary", {})
    dominant = compressed_data.get("dominant", {})

    system_prompt = """You are a health and wellness advisor. Analyze the user's activity patterns
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

    user_prompt = f"""User activity summary for the past {compressed_data.get('period_hours', 1)} hour(s):

Activity patterns: {summary.get('har', {})}
App usage: {summary.get('app_usage', {})}
Phone usage: {summary.get('phone_usage', {})}
Social context: {summary.get('social', {})}
Movement: {summary.get('movement', {})}
Location: {summary.get('location', {})}

Dominant activity: {dominant.get('activity')}
Dominant app category: {dominant.get('app_category')}
Dominant location: {dominant.get('location')}

Total activity records: {compressed_data.get('total_records')}

Suggest an appropriate health intervention."""

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
            "based_on_data": compressed_data.get("dominant", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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