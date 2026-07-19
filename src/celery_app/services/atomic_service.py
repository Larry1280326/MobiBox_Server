"""Business logic for Atomic Activities generation.

This module implements 7 dimensions of atomic activity labeling:
1. HAR label - Human Activity Recognition via LLM
2. APP category - Application usage category via table lookup + LLM fallback
3. Steps label - Step activity via if-else rules
4. Phone usage - Phone usage pattern via if-else rules
5. Social label - Social context via if-else rules
6. Movement label - Movement pattern via if-else rules
7. Location label - Location context via LLM (with optional Baidu Maps integration)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.database import get_database
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
from src.celery_app.services.app_category_service import get_app_category_with_details

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")


# ============================================================================
# Data Fetching Utilities
# ============================================================================


async def get_document_window(
    user: str,
    seconds: int,
) -> list[dict]:
    """Fetch document data for a user from the last X seconds."""
    db = await get_database()
    cutoff_time = datetime.now(CHINA_TZ) - timedelta(seconds=seconds)

    cursor = db["uploads"].find({
        "user": user,
        "timestamp": {"$gte": cutoff_time},
    }).sort("timestamp", 1)

    return await cursor.to_list(None)


async def get_har_window(
    user: str,
    seconds: int,
) -> list[dict]:
    """Fetch HAR labels for a user from the last X seconds."""
    db = await get_database()
    cutoff_time = datetime.now(CHINA_TZ) - timedelta(seconds=seconds)

    cursor = db["har"].find({
        "user": user,
        "timestamp": {"$gte": cutoff_time},
    }).sort("timestamp", 1)

    return await cursor.to_list(None)


# ============================================================================
# LLM-Based Label Generation
# ============================================================================


async def generate_har_label(
    user: str,
    window_seconds: int = ATOMIC_HAR_WINDOW_SECONDS,
) -> Optional[str]:
    """Generate HAR label using LLM based on recent HAR data."""
    har_data = await get_har_window(user, window_seconds)

    if not har_data:
        return None

    labels = [h.get("har_label", h.get("label", "unknown")) for h in har_data]
    if not labels:
        return None

    labels_with_count = [f"{l} (count: {labels.count(l)})" for l in set(labels)]

    system_prompt = """You are an activity recognition expert.
Analyze the provided HAR (Human Activity Recognition) labels and determine the most likely
current activity.

Return only a single activity label, one of:
- unknown, standing, sitting, lying, walking, climbing stairs, running"""

    user_prompt = f"Recent HAR labels:\n{chr(10).join(labels_with_count)}\n\nWhat is the most likely current activity?"

    try:
        result = await query_llm(system_prompt, user_prompt, temperature=0.1)
        return result.strip().lower()
    except Exception as e:
        logger.error(f"Error generating HAR label via LLM: {e}")
        if har_data:
            return har_data[0].get("har_label", har_data[0].get("label", "unknown"))
        return None


async def generate_app_category(
    user: str,
    window_seconds: int = ATOMIC_APP_WINDOW_SECONDS,
) -> tuple[Optional[str], Optional[str]]:
    """Generate app usage category using table lookup with LLM fallback."""
    doc_data = await get_document_window(user, window_seconds)

    if not doc_data:
        return None, None

    apps = []
    for doc in doc_data:
        if doc.get("current_app"):
            apps.append(doc["current_app"])

    if not apps:
        return None, None

    from collections import Counter
    app_counts = Counter(apps)
    most_common_app = app_counts.most_common(1)[0][0]

    result = await get_app_category_with_details(most_common_app)

    if result and result.category:
        return result.category.lower(), result.app_name

    return "uncertain", most_common_app


async def generate_location_label(
    user: str,
    window_seconds: int = ATOMIC_LOCATION_WINDOW_SECONDS,
) -> Optional[str]:
    """Generate location context label using LLM with optional Baidu Maps enrichment."""
    from src.config import get_settings
    from src.services.baidu_maps import reverse_geocode

    doc_data = await get_document_window(user, window_seconds)

    if not doc_data:
        return None

    location_info = []
    settings = get_settings()

    for doc in doc_data:
        info = {}
        lat = doc.get("gpsLat")
        lon = doc.get("gpsLon")

        if lat is not None and lon is not None and settings.baidu_maps_enabled and settings.baidu_maps_api_key:
            try:
                baidu_location = await reverse_geocode(float(lat), float(lon))
                if baidu_location:
                    if baidu_location.get("address"):
                        info["address"] = baidu_location["address"]
                    if baidu_location.get("poi"):
                        info["poi"] = ", ".join(baidu_location["poi"])
                    if baidu_location.get("district"):
                        info["district"] = baidu_location["district"]
                    if baidu_location.get("business"):
                        info["business"] = baidu_location["business"]
                    info["gps"] = f"({lat}, {lon})"
            except Exception as e:
                logger.debug(f"Baidu Maps enrichment failed: {e}")
                info["gps"] = f"({lat}, {lon})"

        if doc.get("address") and "address" not in info:
            info["address"] = doc["address"]
        if doc.get("poi") and "poi" not in info:
            info["poi"] = doc["poi"]

        if info:
            location_info.append(info)

    if not location_info:
        return None

    system_prompt = """You are a location context analyst. Based on GPS coordinates, address, and POI data,
determine the user's current location context. Choose from:
- home, work, school, shopping_mall, restaurant, gym, park, transit, hospital, other
Return only the location context label."""

    location_descriptions = []
    for loc in location_info[-5:]:
        desc_parts = []
        if "gps" in loc:
            desc_parts.append(f"GPS: {loc['gps']}")
        if "address" in loc:
            desc_parts.append(f"Address: {loc['address']}")
        if "poi" in loc:
            desc_parts.append(f"POI: {loc['poi']}")
        if "district" in loc:
            desc_parts.append(f"District: {loc['district']}")
        if "business" in loc:
            desc_parts.append(f"Business: {loc['business']}")
        location_descriptions.append(", ".join(desc_parts))

    user_prompt = f"Location data:\n{chr(10).join(location_descriptions)}\n\nWhat is the location context?"

    try:
        result = await query_llm(system_prompt, user_prompt, temperature=0.1)
        return result.strip().lower()
    except Exception as e:
        logger.error(f"Error generating location label via LLM: {e}")
        return "other"


# ============================================================================
# Rule-Based Label Generation
# ============================================================================


async def generate_step_label(
    user: str,
    window_seconds: int = ATOMIC_STEP_WINDOW_SECONDS,
) -> Optional[str]:
    """Generate step activity label using delta-based logic."""
    doc_data = await get_document_window(user, window_seconds)
    if not doc_data:
        return None

    step_counts = [doc.get("stepcount_sensor") for doc in doc_data if doc.get("stepcount_sensor") is not None]
    if len(step_counts) < 2:
        return None

    total_steps_in_interval = step_counts[-1] - step_counts[0]

    if total_steps_in_interval <= 3:
        return "almost stationary"
    elif total_steps_in_interval <= 10:
        return "low"
    elif total_steps_in_interval <= 18:
        return "medium"
    elif total_steps_in_interval <= 25:
        return "high"
    else:
        return "very high"


async def generate_phone_usage_label(
    user: str,
    window_seconds: int = ATOMIC_PHONE_WINDOW_SECONDS,
) -> Optional[str]:
    """Generate phone usage label using screen ratio and network traffic."""
    doc_data = await get_document_window(user, window_seconds)
    if not doc_data:
        return None

    screen_ratios = [doc.get("screen_on_ratio") for doc in doc_data if doc.get("screen_on_ratio") is not None]
    traffic_values = [doc.get("network_traffic") for doc in doc_data if doc.get("network_traffic") is not None]

    if not screen_ratios:
        return None

    avg_screen_ratio = sum(screen_ratios) / len(screen_ratios)
    total_traffic = sum(traffic_values) if traffic_values else 0

    if avg_screen_ratio < 0.2 and total_traffic < 1 * 1024:
        return "idle"
    elif avg_screen_ratio < 0.5 and total_traffic < 10 * 1024:
        return "low"
    elif avg_screen_ratio < 0.8 and total_traffic < 50 * 1024:
        return "medium"
    elif avg_screen_ratio >= 0.8 or total_traffic >= 100 * 1024:
        return "very high"
    else:
        return "high"


async def generate_social_label(
    user: str,
    window_seconds: int = ATOMIC_SOCIAL_WINDOW_SECONDS,
) -> Optional[str]:
    """Generate social context label using hybrid Bluetooth and app detection."""
    doc_data = await get_document_window(user, window_seconds)
    if not doc_data:
        return None

    bt_devices_list = [doc.get("bluetooth_devices") for doc in doc_data if doc.get("bluetooth_devices")]
    bt_counts = [doc.get("nearbyBluetoothCount") for doc in doc_data if doc.get("nearbyBluetoothCount") is not None]
    apps = [doc.get("current_app", "").lower() for doc in doc_data if doc.get("current_app")]

    total_paired = 0
    total_unknown = 0
    for bt_devices in bt_devices_list:
        if isinstance(bt_devices, list):
            for device in bt_devices:
                if isinstance(device, dict):
                    if device.get("paired"):
                        total_paired += 1
                    else:
                        total_unknown += 1

    paired_ratio = total_paired / (total_paired + total_unknown) if (total_paired + total_unknown) > 0 else 0

    social_apps = {"whatsapp", "facebook", "instagram", "telegram", "discord", "messenger", "twitter", "tiktok", "snapchat"}
    comm_apps = {"whatsapp", "telegram", "discord", "messenger", "signal", "wechat"}
    using_social_app = any(app in social_apps for app in apps)
    using_comm_app = any(app in comm_apps for app in apps)

    avg_bt_count = sum(bt_counts) / len(bt_counts) if bt_counts else 0

    if using_comm_app:
        if paired_ratio > 0.5:
            return "in group/public space"
        else:
            return "with someone"
    elif using_social_app:
        if avg_bt_count > 5:
            return "in group/public space"
        else:
            return "alone"
    elif paired_ratio > 0.7:
        return "with someone"
    elif avg_bt_count > 10:
        return "in group/public space"
    elif avg_bt_count > 3:
        return "with someone"
    elif avg_bt_count > 0:
        return "alone or with someone"
    else:
        return "alone"


async def generate_movement_label(
    user: str,
    window_seconds: int = ATOMIC_MOVEMENT_WINDOW_SECONDS,
) -> Optional[str]:
    """Generate movement pattern label using distance-based logic."""
    import math

    doc_data = await get_document_window(user, window_seconds)
    if not doc_data:
        return None

    gps_points = []
    for doc in doc_data:
        if doc.get("gpsLat") and doc.get("gpsLon"):
            gps_points.append({"lat": doc["gpsLat"], "lon": doc["gpsLon"]})

    if len(gps_points) < 2:
        return None

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    total_distance = 0
    for i in range(1, len(gps_points)):
        total_distance += haversine(
            gps_points[i - 1]["lat"], gps_points[i - 1]["lon"],
            gps_points[i]["lat"], gps_points[i]["lon"]
        )

    if total_distance < 15:
        return "stationary"
    elif total_distance < 55:
        return "slow"
    elif total_distance < 115:
        return "medium"
    else:
        return "fast"


# ============================================================================
# Combined Atomic Activity Generation
# ============================================================================


async def generate_all_atomic_labels(user: str) -> AtomicActivity:
    """Generate all atomic activity labels for a user."""
    har_task = generate_har_label(user)
    app_task = generate_app_category(user)
    step_task = generate_step_label(user)
    phone_task = generate_phone_usage_label(user)
    social_task = generate_social_label(user)
    movement_task = generate_movement_label(user)
    location_task = generate_location_label(user)

    results = await asyncio.gather(
        har_task, app_task, step_task, phone_task,
        social_task, movement_task, location_task,
        return_exceptions=True,
    )

    def safe_result(result, default=None):
        return result if not isinstance(result, Exception) else default

    app_result = safe_result(results[1])
    if isinstance(app_result, tuple):
        app_category, app_name = app_result
    else:
        app_category, app_name = safe_result(results[1]), None

    return AtomicActivity(
        user=user,
        timestamp=datetime.now(CHINA_TZ),
        har_label=safe_result(results[0]),
        app_category=app_category,
        app_name=app_name,
        step_label=safe_result(results[2]),
        phone_usage=safe_result(results[3]),
        social_label=safe_result(results[4]),
        movement_label=safe_result(results[5]),
        location_label=safe_result(results[6]),
    )


async def insert_atomic_activity(activity: AtomicActivity) -> dict:
    """Insert atomic activity into the database."""
    db = await get_database()

    raw = activity.model_dump()
    data = {
        "user": raw["user"],
        "timestamp": raw["timestamp"],
        "har_label": raw["har_label"] or "unknown",
        "app_category": raw["app_category"] or "uncertain",
        "app_name": raw.get("app_name"),
        "step_count": raw["step_label"] or "almost stationary",
        "phone_usage": raw["phone_usage"] or "idle",
        "social": raw["social_label"] or "alone",
        "movement": raw["movement_label"] or "stationary",
        "location": raw["location_label"],
    }

    result = await db["atomic_activities"].insert_one(data)
    data["_id"] = result.inserted_id
    return data
