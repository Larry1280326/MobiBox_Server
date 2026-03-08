"""Business logic for HAR (Human Activity Recognition) processing.

Uses TSFM (Time Series Foundation Model) for zero-shot activity recognition
with fallback to legacy IMU transformer or mock model.

Supports incremental processing via last processed timestamp tracking.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
from supabase import Client

from src.database import get_supabase_client
from src.celery_app.config import (
    HAR_IMU_WINDOW_SECONDS,
    HAR_DATA_DELAY_SECONDS,
    HAR_IMU_WINDOW_SIZE,
    HAR_IMU_INPUT_CHANNELS,
    HAR_IMU_MODEL_CHECKPOINT,
    HAR_IMU_MODEL_CONFIG,
    USE_TSFM_MODEL,
    TSFM_MIN_SAMPLES,
)
from src.celery_app.services.processing_state_service import (
    get_last_processed,
    update_last_processed,
    get_imu_window_since,
)

logger = logging.getLogger(__name__)
from src.celery_app.schemas.har_schemas import HARLabel

# IMU model: label index -> DB enum string (from imu_labels.md)
HAR_LABEL_BY_INDEX = [
    "unknown",        # 0
    "standing",       # 1
    "sitting",        # 2
    "lying",          # 3
    "walking",        # 4
    "climbing stairs",  # 5
    "running",        # 6
]

# Column order for building model input tensor (must match HAR_IMU_INPUT_CHANNELS)
IMU_COLUMNS = [
    "acc_X", "acc_Y", "acc_Z",
    "gyro_X", "gyro_Y", "gyro_Z",
    "mag_X", "mag_Y", "mag_Z",
]

# Mock HAR labels - DB enum values (fallback when model not used)
MOCK_HAR_LABELS = [
    "walking",
    "running",
    "sitting",
    "standing",
    "lying",
    "climbing stairs",
    "unknown",
]

CHINA_TZ = ZoneInfo("Asia/Shanghai")

# Cached IMU model (lazy-loaded)
_imu_model = None
_imu_model_available = None


async def get_imu_window(
    user: str,
    seconds: int = HAR_IMU_WINDOW_SECONDS,
    client: Client | None = None,
) -> list[dict]:
    """
    Fetch IMU data for a user from a delayed time window.

    Since IMU data is uploaded in batches every 2 minutes, we apply a delay
    to ensure we're fetching data that has already been inserted.

    Args:
        user: User identifier
        seconds: Number of seconds to look back (window size)
        client: Optional Supabase client (creates new one if not provided)

    Returns:
        List of IMU data records
    """
    if client is None:
        client = get_supabase_client()

    # Apply delay to account for batch upload timing:
    # fetch data from (now - delay - seconds) to (now - delay)
    delayed_end = datetime.now(CHINA_TZ) - timedelta(seconds=HAR_DATA_DELAY_SECONDS)
    delayed_start = delayed_end - timedelta(seconds=seconds)

    logger.debug(
        "Fetching IMU window for %s: %s to %s (delay=%ss, window=%ss)",
        user,
        delayed_start.isoformat(),
        delayed_end.isoformat(),
        HAR_DATA_DELAY_SECONDS,
        seconds,
    )

    response = await asyncio.to_thread(
        lambda: client.table("imu")
        .select("*")
        .eq("user", user)
        .gte("timestamp", delayed_start.isoformat())
        .lte("timestamp", delayed_end.isoformat())
        .order("timestamp", desc=False)
        .execute()
    )

    data = response.data if response.data else []
    logger.debug("Found %d IMU records for %s", len(data), user)
    return data


def _imu_data_to_tensor(imu_data: list[dict]) -> np.ndarray:
    """
    Build a (1, window_size, input_dim) array from Supabase IMU list.
    Truncates or zero-pads to HAR_IMU_WINDOW_SIZE; uses IMU_COLUMNS order.
    """
    n = len(imu_data)
    out = np.zeros((1, HAR_IMU_WINDOW_SIZE, HAR_IMU_INPUT_CHANNELS), dtype=np.float32)
    for i in range(min(n, HAR_IMU_WINDOW_SIZE)):
        row = imu_data[i]
        for j, col in enumerate(IMU_COLUMNS):
            val = row.get(col)
            out[0, i, j] = float(val) if val is not None else 0.0
    return out


def _resolve_checkpoint_path() -> Path | None:
    """Resolve checkpoint path: try as given, then relative to imu_model_utils/ckpts."""
    if not HAR_IMU_MODEL_CHECKPOINT:
        return None
    path = Path(HAR_IMU_MODEL_CHECKPOINT)
    if path.is_file():
        return path
    # Resolve relative to this package so it works regardless of cwd (e.g. Celery worker)
    ckpts_dir = Path(__file__).resolve().parent / "imu_model_utils" / "ckpts"
    fallback = ckpts_dir / path.name
    return fallback if fallback.is_file() else None


def _get_imu_model():
    """Load and cache IMU transformer model; returns (model, available)."""
    global _imu_model, _imu_model_available
    if _imu_model_available is not None:
        return _imu_model, _imu_model_available
    path = _resolve_checkpoint_path()
    if path is None:
        _imu_model_available = False
        return None, False
    try:
        import torch
        from src.celery_app.services.imu_model_utils.imu_transformer_encoder import (
            IMUTransformerEncoder,
        )
        model = IMUTransformerEncoder(HAR_IMU_MODEL_CONFIG)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()
        _imu_model = model
        _imu_model_available = True
        return _imu_model, True
    except Exception:
        _imu_model_available = False
        return None, False


def _run_imu_model_sync(imu_tensor: np.ndarray) -> tuple[int, float]:
    """Run IMU model in sync context; returns (class_index, confidence)."""
    import torch
    model, available = _get_imu_model()
    if not available or model is None:
        raise RuntimeError("IMU model not available")
    with torch.no_grad():
        x = torch.from_numpy(imu_tensor)
        batch = {"imu": x}
        log_probs = model(batch)
        probs = torch.exp(log_probs)
        pred_idx = int(log_probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item())
    return pred_idx, confidence


async def run_har_model(imu_data: list[dict]) -> tuple[str, float, str]:
    """
    Run HAR model on IMU data.

    Priority:
    1. TSFM model (if USE_TSFM_MODEL=True and checkpoint available)
    2. Legacy IMU transformer (if checkpoint configured)
    3. Mock model (fallback)

    Returns:
        Tuple of (label, confidence, source) where source is "tsfm_model", "imu_model", or "mock_har".
    """
    # Check minimum samples
    if len(imu_data) < 1:
        logger.warning("No IMU data provided, returning unknown")
        return "unknown", 0.5, "insufficient_data"

    # Try TSFM model first (if enabled)
    if USE_TSFM_MODEL:
        try:
            from .tsfm_service import run_tsfm_inference, is_tsfm_available

            tsfm_available = is_tsfm_available()
            logger.debug(f"TSFM model enabled, available: {tsfm_available}, samples: {len(imu_data)}, min_required: {TSFM_MIN_SAMPLES}")

            if tsfm_available and len(imu_data) >= TSFM_MIN_SAMPLES:
                logger.info(f"Running TSFM inference with {len(imu_data)} samples")
                label, confidence, source = await asyncio.to_thread(
                    run_tsfm_inference, imu_data
                )
                logger.info(f"TSFM result: label={label}, confidence={confidence}")
                return label, confidence, source
            elif not tsfm_available:
                logger.warning("TSFM model not available, falling back to legacy model")
            elif len(imu_data) < TSFM_MIN_SAMPLES:
                logger.warning(f"Not enough samples for TSFM: {len(imu_data)} < {TSFM_MIN_SAMPLES}, falling back")
        except Exception as e:
            logger.warning(f"TSFM model failed, falling back to legacy: {e}", exc_info=True)
    else:
        logger.debug("TSFM model disabled (USE_TSFM_MODEL=False)")

    # Fall back to legacy IMU transformer
    model, available = _get_imu_model()
    if available and model is not None:
        logger.info(f"Running legacy IMU model inference with {len(imu_data)} samples")
        tensor = _imu_data_to_tensor(imu_data)
        pred_idx, confidence = await asyncio.to_thread(_run_imu_model_sync, tensor)
        label = HAR_LABEL_BY_INDEX[pred_idx] if pred_idx < len(HAR_LABEL_BY_INDEX) else "unknown"
        logger.info(f"Legacy IMU model result: label={label}, confidence={confidence}")
        return label, round(confidence, 2), "imu_model"
    else:
        logger.warning("Legacy IMU model not available, using mock model")

    # Final fallback to mock model
    logger.info(f"Running mock HAR model with {len(imu_data)} samples")
    label, confidence = await run_mock_har_model(imu_data)
    return label, confidence, "mock_har"


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

    # Mock classification based on acceleration magnitude (returns DB enum values)
    if avg_magnitude < 0.5:
        label = random.choice(["sitting", "lying", "standing"])
        confidence = 0.7 + random.random() * 0.2
    elif avg_magnitude < 2.0:
        label = random.choice(["walking", "standing", "sitting"])
        confidence = 0.6 + random.random() * 0.3
    elif avg_magnitude < 5.0:
        label = random.choice(["walking", "climbing stairs", "unknown"])
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
    client: Client | None = None,
) -> dict:
    """
    Insert HAR label into the har table.

    Database schema: har(id, timestamp, user, har_label, confidence, source)

    Args:
        user: User identifier
        label: Activity label (stored as har_label)
        confidence: Confidence score from model (0.0-1.0)
        source: Source of label ('tsfm_model', 'imu_model', 'mock_har', 'insufficient_data')
        client: Optional Supabase client

    Returns:
        Inserted record data
    """
    if client is None:
        client = get_supabase_client()

    data = {
        "user": user,
        "har_label": label,
        "confidence": round(confidence, 2),
        "source": source,
        "timestamp": datetime.now(CHINA_TZ).isoformat(),
    }

    response = await asyncio.to_thread(
        lambda: client.table("har").insert(data).execute()
    )

    return response.data[0] if response.data else {}


async def process_har_for_user(user: str, client: Client | None = None) -> HARLabel | None:
    """
    Complete HAR processing pipeline for a single user.

    1. Fetch IMU data window
    2. Run HAR model (IMU transformer if checkpoint set, else mock)
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

    # Run HAR model (IMU transformer if checkpoint set, else mock)
    label, confidence, source = await run_har_model(imu_data)

    # Insert result
    await insert_har_label(user, label, confidence, source, client)

    return HARLabel(
        user=user,
        label=label,
        confidence=confidence,
        timestamp=datetime.now(CHINA_TZ),
        source=source,
    )


async def process_har_for_user_incremental(
    user: str,
    client: Client | None = None,
) -> tuple[HARLabel | None, datetime | None]:
    """
    Incremental HAR processing pipeline with timestamp tracking.

    Only processes new IMU data since last processed timestamp.
    Updates the last processed timestamp after successful processing.

    Args:
        user: User identifier
        client: Optional Supabase client

    Returns:
        Tuple of (HARLabel if successful, latest timestamp processed)
    """
    if client is None:
        client = get_supabase_client()

    # Get last processed timestamp for this user
    last_processed = await get_last_processed(user, "har", client)

    if last_processed:
        # Incremental: fetch only new data since last processed
        imu_data = await get_imu_window_since(user, last_processed, client)
        logger.debug(f"Processing HAR incrementally for {user} since {last_processed}")
    else:
        # First time: use standard windowed fetch
        imu_data = await get_imu_window(user, HAR_IMU_WINDOW_SECONDS, client)
        logger.debug(f"Processing HAR for {user} (first time, no last_processed)")

    if not imu_data:
        return None, None

    # Run HAR model
    label, confidence, source = await run_har_model(imu_data)

    # Get the latest timestamp from the processed data
    timestamps = [d.get("timestamp") for d in imu_data if d.get("timestamp")]
    latest_timestamp = None
    if timestamps:
        # Parse the latest timestamp
        ts_str = max(timestamps)
        if isinstance(ts_str, str):
            latest_timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        else:
            latest_timestamp = ts_str

    # Insert result
    await insert_har_label(user, label, confidence, source, client)

    # Update last processed timestamp
    if latest_timestamp:
        await update_last_processed(user, "har", latest_timestamp, client)

    har_label = HARLabel(
        user=user,
        label=label,
        confidence=confidence,
        timestamp=datetime.now(CHINA_TZ),
        source=source,
    )

    return har_label, latest_timestamp