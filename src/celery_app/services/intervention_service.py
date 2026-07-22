"""Business logic for generating health interventions."""

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from src.database import get_database
from src.llm_utils.services import generate_structured_output

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")


class InterventionOutput(BaseModel):
    """Structured output for intervention generation."""
    intervention_type: str
    message: str
    priority: str  # low, medium, high
    category: str  # physical, mental, social, digital_wellbeing


async def get_recent_summaries(hours: int = 1) -> list[dict]:
    """Get recent summary logs from the last X hours."""
    db = await get_database()
    cutoff_time = datetime.now(CHINA_TZ) - timedelta(hours=hours)

    cursor = db["summary_logs"].find({
        "timestamp": {"$gte": cutoff_time},
    }).sort("timestamp", 1)

    return await cursor.to_list(None)


async def generate_intervention_from_summary(
    user: str,
    summary_log: dict,
) -> Optional[dict]:
    """Generate a health intervention based on a summary log."""
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
- category: "physical", "mental", "social", or "digital_wellbeing" """

    summary_text = summary_log.get("summary", "No summary available")

    user_prompt = f"""User activity summary:

{summary_text}

Suggest an appropriate health intervention based on this summary."""

    try:
        result = await generate_structured_output(
            system_prompt, user_prompt, InterventionOutput, temperature=0.3,
        )

        return {
            "user": user,
            "intervention_content": result.message,
            "start_timestamp": summary_log.get("start_timestamp"),
            "end_timestamp": summary_log.get("end_timestamp"),
            "timestamp": datetime.now(CHINA_TZ),
        }
    except Exception as e:
        logger.error(f"Error generating intervention for user {user}: {e}", exc_info=True)

        from src.config import get_llm_settings
        try:
            settings = get_llm_settings()
            if not settings.openrouter_api_key:
                logger.error("OPENROUTER_API_KEY is not set - LLM interventions will use fallback")
        except Exception:
            pass

        return {
            "user": user,
            "intervention_content": "Take a moment to check in with yourself. How are you feeling?",
            "start_timestamp": summary_log.get("start_timestamp"),
            "end_timestamp": summary_log.get("end_timestamp"),
            "timestamp": datetime.now(CHINA_TZ),
        }


async def insert_intervention(intervention: dict) -> dict:
    """Insert intervention into the database."""
    db = await get_database()
    result = await db["interventions"].insert_one(intervention)
    intervention["_id"] = result.inserted_id
    return intervention
