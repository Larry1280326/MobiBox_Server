"""Business logic for HAR (Human Activity Recognition) processing."""

import asyncio
import random
from datetime import datetime, timedelta, timezone

from supabase import Client

from src.database import get_supabase_client
from src.celery_app.config import HAR_IMU_WINDOW_SECONDS
from src.celery_app.schemas.har_schemas import HARLabel


# Mock HAR labels for testing
MOCK_HAR_LABELS = [
    "walking",
    "running",
    "sitting",
    "standing",
    "lying_down",
    "climbing_stairs",
    "descending_stairs",
    "cycling",
    "driving",
    "unknown",
]


async def get_imu_window(
    user: str,
    seconds: int = HAR_IMU_WINDOW_SECONDS,
    client: Client | None = None,
) -> list[dict]:
    """
    Fetch IMU data for a user from the last X seconds.

    Args:
        user: User identifier
        seconds: Number of seconds to look back
        client: Optional Supabase client (creates new one if not provided)

    Returns:
        List of IMU data records
    """
    if client is None:
        client = get_supabase_client()

    cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=seconds)

    response = await asyncio.to_thread(
        lambda: client.table("imu")
        .select("*")
        .eq("user", user)
        .gte("timestamp", cutoff_time.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    return response.data if response.data else []


async def run_mock_har_model(imu_data: list[dict]) -> tuple[str, float]:
    """
    Run mock HAR model on IMU data.

    This simulates a HAR model that processes IMU data and returns
    an activity label with confidence score.

    Args:
        imu_data: List of IMU sensor readings

    Returns:
        Tuple of (label, confidence)
    """
    # Simulate processing time
    await asyncio.sleep(0.1)

    if not imu_data:
        return "unknown", 0.5

    # Calculate average acceleration magnitude
    acc_magnitudes = []
    for sample in imu_data:
        acc_x = sample.get("acc_X", 0) or 0
        acc_y = sample.get("acc_Y", 0) or 0
        acc_z = sample.get("acc_Z", 0) or 0
        magnitude = (acc_x**2 + acc_y**2 + acc_z**2) ** 0.5
        acc_magnitudes.append(magnitude)

    avg_magnitude = sum(acc_magnitudes) / len(acc_magnitudes) if acc_magnitudes else 0

    # Mock classification based on acceleration magnitude
    if avg_magnitude < 0.5:
        label = random.choice(["sitting", "lying_down", "standing"])
        confidence = 0.7 + random.random() * 0.2
    elif avg_magnitude < 2.0:
        label = random.choice(["walking", "standing", "driving"])
        confidence = 0.6 + random.random() * 0.3
    elif avg_magnitude < 5.0:
        label = random.choice(["walking", "climbing_stairs", "cycling"])
        confidence = 0.5 + random.random() * 0.4
    else:
        label = random.choice(["running", "climbing_stairs", "descending_stairs"])
        confidence = 0.6 + random.random() * 0.3

    return label, round(confidence, 2)


async def insert_har_label(
    user: str,
    label: str,
    confidence: float = 1.0,
    source: str = "mock_har",
    client: Client | None = None,
) -> dict:
    """
    Insert HAR label into the har table.

    Args:
        user: User identifier
        label: Activity label
        confidence: Confidence score (0-1)
        source: Source of the label (mock_har, ml_model, etc.)
        client: Optional Supabase client

    Returns:
        Inserted record data
    """
    if client is None:
        client = get_supabase_client()

    data = {
        "user": user,
        "label": label,
        "confidence": confidence,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    response = await asyncio.to_thread(
        lambda: client.table("har").insert(data).execute()
    )

    return response.data[0] if response.data else {}


async def process_har_for_user(user: str, client: Client | None = None) -> HARLabel | None:
    """
    Complete HAR processing pipeline for a single user.

    1. Fetch IMU data window
    2. Run mock HAR model
    3. Insert result to database

    Args:
        user: User identifier
        client: Optional Supabase client

    Returns:
        HARLabel if successful, None otherwise
    """
    if client is None:
        client = get_supabase_client()

    # Get IMU data window
    imu_data = await get_imu_window(user, HAR_IMU_WINDOW_SECONDS, client)

    if not imu_data:
        return None

    # Run mock HAR model
    label, confidence = await run_mock_har_model(imu_data)

    # Insert result
    await insert_har_label(user, label, confidence, "mock_har", client)

    return HARLabel(
        user=user,
        label=label,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
        source="mock_har",
    )