"""Business logic for HAR (Human Activity Recognition) processing.

Uses IMU-SelfSupEncoder-v1 as primary model with mock model fallback.
Supports incremental processing via last processed timestamp tracking.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.database import get_database
from src.celery_app.config import (
    HAR_IMU_WINDOW_SECONDS,
    HAR_DATA_DELAY_SECONDS,
    USE_SELFSUP_MODEL,
    SELFSUP_MIN_SAMPLES,
)
from src.celery_app.services.processing_state_service import (
    get_last_processed,
    update_last_processed,
    get_imu_window_since,
)

logger = logging.getLogger(__name__)
from src.celery_app.schemas.har_schemas import HARLabel

CHINA_TZ = ZoneInfo("Asia/Shanghai")


async def get_imu_window(
    user: str,
    seconds: int = HAR_IMU_WINDOW_SECONDS,
) -> list[dict]:
    """
    Fetch IMU data for a user from a delayed time window.
    """
    db = await get_database()

    delayed_end = datetime.now(CHINA_TZ) - timedelta(seconds=HAR_DATA_DELAY_SECONDS)
    delayed_start = delayed_end - timedelta(seconds=seconds)

    logger.debug(
        "Fetching IMU window for %s: %s to %s (delay=%ss, window=%ss)",
        user, delayed_start.isoformat(), delayed_end.isoformat(),
        HAR_DATA_DELAY_SECONDS, seconds,
    )

    cursor = db["imu"].find({
        "user": user,
        "timestamp": {"$gte": delayed_start, "$lte": delayed_end},
    }).sort("timestamp", 1)

    data = await cursor.to_list(None)
    logger.debug("Found %d IMU records for %s", len(data), user)
    return data


async def run_har_model(imu_data: list[dict]) -> tuple[str, float, str]:
    """
    Run HAR model on IMU data.

    Priority:
    1. SelfSupEncoder (if USE_SELFSUP_MODEL=True and model available)
    2. Mock model (fallback)

    Returns:
        Tuple of (label, confidence, source)
    """
    if len(imu_data) < 1:
        logger.warning("No IMU data provided, returning unknown")
        return "unknown", 0.5, "insufficient_data"

    # Try SelfSupEncoder
    if USE_SELFSUP_MODEL and len(imu_data) >= SELFSUP_MIN_SAMPLES:
        try:
            from .imu_selfsup_service import (
                run_selfsup_inference,
                is_selfsup_available,
            )

            selfsup_available = is_selfsup_available()
            logger.debug(
                f"SelfSupEncoder enabled, available: {selfsup_available}, "
                f"samples: {len(imu_data)}, min_required: {SELFSUP_MIN_SAMPLES}"
            )

            if selfsup_available:
                logger.info(f"Running SelfSupEncoder inference with {len(imu_data)} samples")
                label, confidence, source = run_selfsup_inference(imu_data)
                logger.info(f"SelfSupEncoder result: label={label}, confidence={confidence}")
                return label, confidence, source
            else:
                logger.warning("SelfSupEncoder not available, using mock model")
        except Exception as e:
            logger.warning(f"SelfSupEncoder failed, using mock model: {e}", exc_info=True)
    elif USE_SELFSUP_MODEL:
        logger.debug(
            f"Not enough samples for SelfSupEncoder: "
            f"{len(imu_data)} < {SELFSUP_MIN_SAMPLES}"
        )

    # Fallback to mock model
    logger.info(f"Running mock HAR model with {len(imu_data)} samples")
    label, confidence = await run_mock_har_model(imu_data)
    return label, confidence, "mock_har"


async def run_mock_har_model(imu_data: list[dict]) -> tuple[str, float]:
    """Run mock HAR model on IMU data."""
    await asyncio.sleep(0.1)

    if not imu_data:
        return "unknown", 0.5

    acc_magnitudes = []
    for sample in imu_data:
        acc_x = sample.get("acc_X", 0) or 0
        acc_y = sample.get("acc_Y", 0) or 0
        acc_z = sample.get("acc_Z", 0) or 0
        magnitude = (acc_x**2 + acc_y**2 + acc_z**2) ** 0.5
        acc_magnitudes.append(magnitude)

    avg_magnitude = sum(acc_magnitudes) / len(acc_magnitudes) if acc_magnitudes else 0

    if avg_magnitude < 0.5:
        label = random.choice(["sitting", "lying", "standing"])
        confidence = 0.7 + random.random() * 0.2
    elif avg_magnitude < 2.0:
        label = random.choice(["walking", "standing", "sitting"])
        confidence = 0.6 + random.random() * 0.3
    elif avg_magnitude < 5.0:
        label = random.choice(["walking", "climbing stairs", "running"])
        confidence = 0.5 + random.random() * 0.4
    else:
        label = random.choice(["running", "climbing stairs"])
        confidence = 0.6 + random.random() * 0.3

    return label, round(confidence, 2)


async def insert_har_label(
    user: str,
    label: str,
    confidence: float = 1.0,
    source: str = "mock_har",
) -> dict:
    """Insert HAR label into the har collection."""
    db = await get_database()

    data = {
        "user": user,
        "har_label": label,
        "confidence": round(confidence, 2),
        "source": source,
        "timestamp": datetime.now(CHINA_TZ),
    }

    result = await db["har"].insert_one(data)
    data["_id"] = result.inserted_id
    return data


async def process_har_for_user(user: str) -> HARLabel | None:
    """
    Complete HAR processing pipeline for a single user.

    1. Fetch IMU data window
    2. Run HAR model
    3. Insert result to database
    """
    imu_data = await get_imu_window(user, HAR_IMU_WINDOW_SECONDS)

    if not imu_data:
        return None

    label, confidence, source = await run_har_model(imu_data)
    await insert_har_label(user, label, confidence, source)

    return HARLabel(
        user=user,
        label=label,
        confidence=confidence,
        timestamp=datetime.now(CHINA_TZ),
        source=source,
    )


async def process_har_for_user_incremental(
    user: str,
) -> tuple[HARLabel | None, datetime | None]:
    """
    Incremental HAR processing pipeline with timestamp tracking.
    """
    last_processed = await get_last_processed(user, "har")

    if last_processed:
        imu_data = await get_imu_window_since(user, last_processed)
        logger.debug(f"Processing HAR incrementally for {user} since {last_processed}")
    else:
        imu_data = await get_imu_window(user, HAR_IMU_WINDOW_SECONDS)
        logger.debug(f"Processing HAR for {user} (first time, no last_processed)")

    if not imu_data:
        return None, None

    label, confidence, source = await run_har_model(imu_data)

    timestamps = [d.get("timestamp") for d in imu_data if d.get("timestamp")]
    latest_timestamp = None
    if timestamps:
        ts = max(timestamps)
        latest_timestamp = ts if isinstance(ts, datetime) else ts

    await insert_har_label(user, label, confidence, source)

    if latest_timestamp:
        await update_last_processed(user, "har", latest_timestamp)

    har_label = HARLabel(
        user=user,
        label=label,
        confidence=confidence,
        timestamp=datetime.now(CHINA_TZ),
        source=source,
    )

    return har_label, latest_timestamp
