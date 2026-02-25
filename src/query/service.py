"""Business logic for querying summary logs and interventions."""

import asyncio
from datetime import datetime
from typing import Optional

from supabase import Client

from src.database import get_supabase_client
from src.query.constants import (
    SUMMARY_LOGS_TABLE,
    INTERVENTIONS_TABLE,
    INTERVENTION_FEEDBACKS_TABLE,
    SUMMARY_LOG_FEEDBACKS_TABLE,
)


async def get_summary_logs(
    user: str,
    log_type: str,
    client: Client | None = None,
) -> Optional[dict]:
    """
    Fetch the most recent summary log for a user from the database.

    Args:
        user: User identifier
        log_type: Type of summary log (hourly or daily)
        client: Optional Supabase client

    Returns:
        The most recent summary log record, or None if not found
    """
    if client is None:
        client = get_supabase_client()

    response = await asyncio.to_thread(
        lambda: client.table(SUMMARY_LOGS_TABLE)
        .select("*")
        .eq("user", user)
        .eq("log_type", log_type)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )

    return response.data[0] if response.data else None


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
        "intervention_content": record.get("message", ""),
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
    client: Client | None = None,
) -> dict:
    """
    Submit summary log feedback to the database.

    Args:
        user: User ID
        summary_logs_id: ID of the summary log being rated
        feedback: Optional feedback text
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
    }

    response = await asyncio.to_thread(
        lambda: client.table(SUMMARY_LOG_FEEDBACKS_TABLE)
        .insert(data)
        .execute()
    )

    return response.data[0] if response.data else {}