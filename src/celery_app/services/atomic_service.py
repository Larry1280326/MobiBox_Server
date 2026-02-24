"""Business logic for Atomic Activities generation.

This module implements 7 dimensions of atomic activity labeling:
1. HAR label - Human Activity Recognition via LLM
2. APP category - Application usage category via LLM
3. Steps label - Step activity via if-else rules
4. Phone usage - Phone usage pattern via if-else rules
5. Social label - Social context via if-else rules
6. Movement label - Movement pattern via if-else rules
7. Location label - Location context via LLM
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from supabase import Client

from src.database import get_supabase_client
from src.llm_utils.services import query_llm
from src.celery_app.config import (
    ATOMIC_HAR_WINDOW_SECONDS,
    ATOMIC_APP_WINDOW_SECONDS,
    ATOMIC_STEP_WINDOW_SECONDS,
    ATOMIC_PHONE_WINDOW_SECONDS,
    ATOMIC_SOCIAL_WINDOW_SECONDS,
    ATOMIC_MOVEMENT_WINDOW_SECONDS,
    ATOMIC_LOCATION_WINDOW_SECONDS,
)
from src.celery_app.schemas.atomic_schemas import AtomicActivity

logger = logging.getLogger(__name__)


# ============================================================================
# Data Fetching Utilities
# ============================================================================


async def get_document_window(
    user: str,
    seconds: int,
    client: Client | None = None,
) -> list[dict]:
    """
    Fetch document data for a user from the last X seconds.

    Args:
        user: User identifier
        seconds: Number of seconds to look back
        client: Optional Supabase client

    Returns:
        List of document records
    """
    if client is None:
        client = get_supabase_client()

    cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=seconds)

    response = await asyncio.to_thread(
        lambda: client.table("uploads")
        .select("*")
        .eq("user", user)
        .gte("timestamp", cutoff_time.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    return response.data if response.data else []


async def get_har_window(
    user: str,
    seconds: int,
    client: Client | None = None,
) -> list[dict]:
    """
    Fetch HAR labels for a user from the last X seconds.

    Args:
        user: User identifier
        seconds: Number of seconds to look back
        client: Optional Supabase client

    Returns:
        List of HAR records
    """
    if client is None:
        client = get_supabase_client()

    cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=seconds)

    response = await asyncio.to_thread(
        lambda: client.table("har")
        .select("*")
        .eq("user", user)
        .gte("timestamp", cutoff_time.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    return response.data if response.data else []


# ============================================================================
# LLM-Based Label Generation
# ============================================================================


async def generate_har_label(
    user: str,
    window_seconds: int = ATOMIC_HAR_WINDOW_SECONDS,
    client: Client | None = None,
) -> Optional[str]:
    """
    Generate HAR label using LLM based on recent HAR data.

    This uses the pre-computed HAR labels and summarizes them via LLM.

    Args:
        user: User identifier
        window_seconds: Time window to consider
        client: Optional Supabase client

    Returns:
        Activity label string or None
    """
    har_data = await get_har_window(user, window_seconds, client)

    if not har_data:
        return None

    # Extract labels with confidence
    labels = [
        f"{h.get('label', 'unknown')} (confidence: {h.get('confidence', 0.5):.2f})"
        for h in har_data
    ]

    if not labels:
        return None

    system_prompt = """You are an activity recognition expert.
Analyze the provided HAR (Human Activity Recognition) labels and determine the most likely
current activity. Consider the confidence scores when making your decision.

Return only a single activity label, one of:
- walking, running, sitting, standing, lying_down, climbing_stairs, descending_stairs, cycling, driving, unknown"""

    user_prompt = f"Recent HAR labels:\n{chr(10).join(labels)}\n\nWhat is the most likely current activity?"

    try:
        result = await query_llm(system_prompt, user_prompt, temperature=0.1)
        return result.strip().lower()
    except Exception as e:
        logger.error(f"Error generating HAR label via LLM: {e}")
        # Fallback to most common label
        if har_data:
            return har_data[0].get("label", "unknown")
        return None


async def generate_app_category(
    user: str,
    window_seconds: int = ATOMIC_APP_WINDOW_SECONDS,
    client: Client | None = None,
) -> Optional[str]:
    """
    Generate app usage category using LLM.

    Args:
        user: User identifier
        window_seconds: Time window to consider
        client: Optional Supabase client

    Returns:
        App category string or None
    """
    doc_data = await get_document_window(user, window_seconds, client)

    if not doc_data:
        return None

    # Extract app usage info
    apps = []
    for doc in doc_data:
        if doc.get("current_app"):
            apps.append(doc["current_app"])

    if not apps:
        return None

    system_prompt = """You are an app usage analyst. Categorize the user's current app usage into one of these categories:
- social_media (Facebook, Instagram, Twitter, WhatsApp, etc.)
- communication (Messenger, Telegram, Discord, etc.)
- productivity (Email, Calendar, Notes, Office apps, etc.)
- entertainment (YouTube, Netflix, Games, etc.)
- news (News apps, RSS readers, etc.)
- shopping (Amazon, eBay, food delivery, etc.)
- health_fitness (Fitness trackers, health apps, etc.)
- navigation (Maps, GPS apps, etc.)
- finance (Banking, payment apps, etc.)
- education (Learning apps, courses, etc.)
- other

Return only the category name."""

    user_prompt = f"Current apps used:\n{chr(10).join(apps[-10:])}\n\nWhat category best describes this usage?"

    try:
        result = await query_llm(system_prompt, user_prompt, temperature=0.1)
        return result.strip().lower()
    except Exception as e:
        logger.error(f"Error generating app category via LLM: {e}")
        return "other"


async def generate_location_label(
    user: str,
    window_seconds: int = ATOMIC_LOCATION_WINDOW_SECONDS,
    client: Client | None = None,
) -> Optional[str]:
    """
    Generate location context label using LLM.

    Args:
        user: User identifier
        window_seconds: Time window to consider
        client: Optional Supabase client

    Returns:
        Location label string or None
    """
    doc_data = await get_document_window(user, window_seconds, client)

    if not doc_data:
        return None

    # Extract location info
    location_info = []
    for doc in doc_data:
        info = {}
        if doc.get("gpsLat") and doc.get("gpsLon"):
            info["gps"] = f"({doc['gpsLat']}, {doc['gpsLon']})"
        if doc.get("address"):
            info["address"] = doc["address"]
        if doc.get("poi"):
            info["poi"] = doc["poi"]
        if info:
            location_info.append(info)

    if not location_info:
        return None

    system_prompt = """You are a location context analyst. Based on GPS coordinates, address, and POI (Point of Interest) data,
determine the user's current location context. Choose from:
- home
- work
- school
- shopping_mall
- restaurant
- gym
- park
- transit (bus, train, car, etc.)
- hospital
- other

Return only the location context label."""

    # Format location data
    location_descriptions = []
    for loc in location_info[-5:]:  # Last 5 locations
        desc_parts = []
        if "gps" in loc:
            desc_parts.append(f"GPS: {loc['gps']}")
        if "address" in loc:
            desc_parts.append(f"Address: {loc['address']}")
        if "poi" in loc:
            desc_parts.append(f"POI: {loc['poi']}")
        location_descriptions.append(", ".join(desc_parts))

    user_prompt = f"Location data:\n{chr(10).join(location_descriptions)}\n\nWhat is the location context?"

    try:
        result = await query_llm(system_prompt, user_prompt, temperature=0.1)
        return result.strip().lower()
    except Exception as e:
        logger.error(f"Error generating location label via LLM: {e}")
        return "other"


# ============================================================================
# Rule-Based Label Generation (If-Else Logic)
# ============================================================================


async def generate_step_label(
    user: str,
    window_seconds: int = ATOMIC_STEP_WINDOW_SECONDS,
    client: Client | None = None,
) -> Optional[str]:
    """
    Generate step activity label using if-else rules.

    Args:
        user: User identifier
        window_seconds: Time window to consider
        client: Optional Supabase client

    Returns:
        Step label string or None
    """
    doc_data = await get_document_window(user, window_seconds, client)

    if not doc_data:
        return None

    # Get step count data
    step_counts = [doc.get("stepcount_sensor") for doc in doc_data if doc.get("stepcount_sensor") is not None]

    if not step_counts:
        return None

    # Calculate step change (if we have multiple readings)
    if len(step_counts) >= 2:
        step_change = step_counts[-1] - step_counts[0]

        if step_change > 100:
            return "high_activity"
        elif step_change > 20:
            return "moderate_activity"
        elif step_change > 0:
            return "low_activity"
        else:
            return "stationary"

    # Single reading - classify by absolute value
    avg_steps = sum(step_counts) / len(step_counts)
    if avg_steps > 5000:
        return "active"
    elif avg_steps > 1000:
        return "moderate"
    else:
        return "sedentary"


async def generate_phone_usage_label(
    user: str,
    window_seconds: int = ATOMIC_PHONE_WINDOW_SECONDS,
    client: Client | None = None,
) -> Optional[str]:
    """
    Generate phone usage label using if-else rules.

    Args:
        user: User identifier
        window_seconds: Time window to consider
        client: Optional Supabase client

    Returns:
        Phone usage label string or None
    """
    doc_data = await get_document_window(user, window_seconds, client)

    if not doc_data:
        return None

    # Get screen-on ratio and current app
    screen_ratios = [doc.get("screen_on_ratio") for doc in doc_data if doc.get("screen_on_ratio") is not None]
    apps = [doc.get("current_app") for doc in doc_data if doc.get("current_app")]

    if not screen_ratios and not apps:
        return None

    avg_screen_ratio = sum(screen_ratios) / len(screen_ratios) if screen_ratios else 0

    # Classification logic
    if avg_screen_ratio > 0.8:
        return "heavy_usage"
    elif avg_screen_ratio > 0.5:
        return "moderate_usage"
    elif avg_screen_ratio > 0.2:
        return "light_usage"
    elif apps:
        # Has app data but low screen time - brief interactions
        return "intermittent_usage"
    else:
        return "idle"


async def generate_social_label(
    user: str,
    window_seconds: int = ATOMIC_SOCIAL_WINDOW_SECONDS,
    client: Client | None = None,
) -> Optional[str]:
    """
    Generate social context label using if-else rules.

    Args:
        user: User identifier
        window_seconds: Time window to consider
        client: Optional Supabase client

    Returns:
        Social label string or None
    """
    doc_data = await get_document_window(user, window_seconds, client)

    if not doc_data:
        return None

    # Get Bluetooth and app data for social context
    bt_counts = [doc.get("nearbyBluetoothCount") for doc in doc_data if doc.get("nearbyBluetoothCount") is not None]
    apps = [doc.get("current_app", "").lower() for doc in doc_data if doc.get("current_app")]

    # Social app detection
    social_apps = {"whatsapp", "facebook", "instagram", "telegram", "discord", "messenger", "twitter", "tiktok", "snapchat"}
    using_social_app = any(app in social_apps for app in apps)

    # Communication app detection
    comm_apps = {"whatsapp", "telegram", "discord", "messenger", "signal", "wechat"}
    using_comm_app = any(app in comm_apps for app in apps)

    # Bluetooth-based social detection
    avg_bt_count = sum(bt_counts) / len(bt_counts) if bt_counts else 0

    if using_comm_app:
        if avg_bt_count > 3:
            return "group_communication"
        else:
            return "direct_communication"
    elif using_social_app:
        if avg_bt_count > 5:
            return "social_gathering"
        else:
            return "social_media_browsing"
    elif avg_bt_count > 10:
        return "crowded_environment"
    elif avg_bt_count > 3:
        return "small_group"
    elif avg_bt_count > 0:
        return "few_people_nearby"
    else:
        return "solitary"


async def generate_movement_label(
    user: str,
    window_seconds: int = ATOMIC_MOVEMENT_WINDOW_SECONDS,
    client: Client | None = None,
) -> Optional[str]:
    """
    Generate movement pattern label using if-else rules.

    Uses GPS coordinates to detect movement patterns over a longer window.

    Args:
        user: User identifier
        window_seconds: Time window to consider (default 2 minutes)
        client: Optional Supabase client

    Returns:
        Movement label string or None
    """
    import math

    doc_data = await get_document_window(user, window_seconds, client)

    if not doc_data:
        return None

    # Extract GPS coordinates with timestamps
    gps_points = []
    for doc in doc_data:
        if doc.get("gpsLat") and doc.get("gpsLon"):
            gps_points.append({
                "lat": doc["gpsLat"],
                "lon": doc["gpsLon"],
                "timestamp": doc.get("timestamp"),
            })

    if len(gps_points) < 2:
        return None

    # Calculate distance between first and last point
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS points in meters."""
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # Calculate total distance
    total_distance = 0
    for i in range(1, len(gps_points)):
        total_distance += haversine(
            gps_points[i - 1]["lat"], gps_points[i - 1]["lon"],
            gps_points[i]["lat"], gps_points[i]["lon"]
        )

    # Calculate speed (m/s) - approximate
    time_diff = window_seconds
    speed = total_distance / time_diff if time_diff > 0 else 0

    # Classification based on speed
    if speed > 8:  # ~30 km/h
        return "vehicle_travel"
    elif speed > 3:  # ~10 km/h
        return "cycling"
    elif speed > 1.5:  # ~5 km/h
        return "walking"
    elif speed > 0.5:
        return "slow_movement"
    else:
        return "stationary"


# ============================================================================
# Combined Atomic Activity Generation
# ============================================================================


async def generate_all_atomic_labels(
    user: str,
    client: Client | None = None,
) -> AtomicActivity:
    """
    Generate all atomic activity labels for a user.

    Args:
        user: User identifier
        client: Optional Supabase client

    Returns:
        AtomicActivity with all labels populated
    """
    if client is None:
        client = get_supabase_client()

    # Generate all labels in parallel
    har_task = generate_har_label(user, client=client)
    app_task = generate_app_category(user, client=client)
    step_task = generate_step_label(user, client=client)
    phone_task = generate_phone_usage_label(user, client=client)
    social_task = generate_social_label(user, client=client)
    movement_task = generate_movement_label(user, client=client)
    location_task = generate_location_label(user, client=client)

    results = await asyncio.gather(
        har_task, app_task, step_task, phone_task,
        social_task, movement_task, location_task,
        return_exceptions=True,
    )

    # Handle exceptions and extract results
    def safe_result(result, default=None):
        return result if not isinstance(result, Exception) else default

    return AtomicActivity(
        user=user,
        timestamp=datetime.now(timezone.utc),
        har_label=safe_result(results[0]),
        app_category=safe_result(results[1]),
        step_label=safe_result(results[2]),
        phone_usage=safe_result(results[3]),
        social_label=safe_result(results[4]),
        movement_label=safe_result(results[5]),
        location_label=safe_result(results[6]),
    )


async def insert_atomic_activity(
    activity: AtomicActivity,
    client: Client | None = None,
) -> dict:
    """
    Insert atomic activity into the database.

    Args:
        activity: AtomicActivity to insert
        client: Optional Supabase client

    Returns:
        Inserted record data
    """
    if client is None:
        client = get_supabase_client()

    data = activity.model_dump()
    data["timestamp"] = data["timestamp"].isoformat()

    response = await asyncio.to_thread(
        lambda: client.table("atomic_activities").insert(data).execute()
    )

    return response.data[0] if response.data else {}