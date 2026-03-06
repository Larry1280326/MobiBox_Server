"""TSFM model service for human activity recognition.

This service wraps the TSFM (Time Series Foundation Model) for inference
in the MobiBox backend, providing zero-shot activity recognition from IMU data.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

# Cached TSFM model and label bank
_tsfm_model = None
_tsfm_label_bank = None
_tsfm_available = None
_tsfm_device = None

# TSFM configuration
TSFM_CHECKPOINT_PATH = Path(__file__).parent / "tsfm_model" / "ckpts" / "best.pt"
TSFM_SAMPLING_RATE = 50.0  # Match MobiBox IMU data rate (50Hz)
TSFM_PATCH_SIZE_SEC = 1.0  # 1 second patches
TSFM_TARGET_PATCH_SIZE = 64  # Fixed timesteps per patch

# Channel descriptions for MobiBox (9-channel IMU)
TSFM_CHANNEL_DESCRIPTIONS = [
    "accelerometer x-axis",
    "accelerometer y-axis",
    "accelerometer z-axis",
    "gyroscope x-axis",
    "gyroscope y-axis",
    "gyroscope z-axis",
    "magnetometer x-axis",
    "magnetometer y-axis",
    "magnetometer z-axis",
]

# Map TSFM labels to MobiBox DB enum values
# UCI-HAR training labels: walking, walking_upstairs, walking_downstairs, sitting, standing, laying
# TSFM uses label groups from label_groups.py for zero-shot generalization
TSFM_TO_MOBIBOX_LABELS = {
    # Walking
    "walking": "walking",

    # Stairs - both directions map to "climbing stairs"
    "ascending_stairs": "climbing stairs",
    "walking_upstairs": "climbing stairs",
    "climbing_stairs": "climbing stairs",
    "going_up_stairs": "climbing stairs",
    "stairs_up": "climbing stairs",
    "descending_stairs": "climbing stairs",
    "walking_downstairs": "climbing stairs",
    "going_down_stairs": "climbing stairs",
    "stairs_down": "climbing stairs",
    "stairs": "climbing stairs",

    # Running (not in UCI-HAR, but may be predicted due to semantic similarity)
    "running": "running",
    "jogging": "running",

    # Lying
    "lying": "lying",
    "laying": "lying",
    "lying_down": "lying",
    "sleeping": "lying",

    # Sitting
    "sitting": "sitting",
    "sitting_down": "sitting",

    # Standing
    "standing": "standing",
    "standing_up": "standing",

    # Default fallback
    "__default__": "unknown",
}


def _resolve_tsfm_checkpoint() -> Optional[Path]:
    """Resolve TSFM checkpoint path."""
    if TSFM_CHECKPOINT_PATH.is_file():
        return TSFM_CHECKPOINT_PATH
    return None


def _get_tsfm_model():
    """Load and cache TSFM model; returns (model, label_bank, available, device)."""
    global _tsfm_model, _tsfm_label_bank, _tsfm_available, _tsfm_device

    if _tsfm_available is not None:
        return _tsfm_model, _tsfm_label_bank, _tsfm_available, _tsfm_device

    import torch

    path = _resolve_tsfm_checkpoint()
    if path is None:
        logger.warning("TSFM checkpoint not found at %s", TSFM_CHECKPOINT_PATH)
        _tsfm_available = False
        return None, None, False, None

    try:
        from .tsfm_model.model_loading import load_model, load_label_bank

        # Use CPU to avoid MPS device issues with sentence-transformers
        # MPS can have issues with some transformer models due to placeholder storage
        device = torch.device("cpu")
        logger.info(f"Loading TSFM model on {device}...")

        model, checkpoint, hyperparams_path = load_model(str(path), device, verbose=True)
        label_bank = load_label_bank(
            checkpoint, device, hyperparams_path,
            text_encoder=model.text_encoder, verbose=True
        )

        _tsfm_model = model
        _tsfm_label_bank = label_bank
        _tsfm_available = True
        _tsfm_device = device

        logger.info(f"TSFM model loaded successfully from {path}")
        return _tsfm_model, _tsfm_label_bank, True, device

    except Exception as e:
        logger.error(f"Failed to load TSFM model: {e}", exc_info=True)
        _tsfm_available = False
        return None, None, False, None


def _imu_data_to_array(imu_data: List[dict]) -> np.ndarray:
    """Convert IMU data list to numpy array (N, 9).

    Args:
        imu_data: List of IMU records with acc_X/Y/Z, gyro_X/Y/Z, mag_X/Y/Z

    Returns:
        Numpy array of shape (N, 9) with columns:
        [acc_X, acc_Y, acc_Z, gyro_X, gyro_Y, gyro_Z, mag_X, mag_Y, mag_Z]
    """
    n = len(imu_data)
    arr = np.zeros((n, 9), dtype=np.float32)

    for i, row in enumerate(imu_data):
        arr[i, 0] = float(row.get("acc_X") or 0)
        arr[i, 1] = float(row.get("acc_Y") or 0)
        arr[i, 2] = float(row.get("acc_Z") or 0)
        arr[i, 3] = float(row.get("gyro_X") or 0)
        arr[i, 4] = float(row.get("gyro_Y") or 0)
        arr[i, 5] = float(row.get("gyro_Z") or 0)
        arr[i, 6] = float(row.get("mag_X") or 0)
        arr[i, 7] = float(row.get("mag_Y") or 0)
        arr[i, 8] = float(row.get("mag_Z") or 0)

    return arr


def _map_tsfm_label_to_mobibox(tsfm_label: str) -> str:
    """Map TSFM label to MobiBox DB enum value."""
    return TSFM_TO_MOBIBOX_LABELS.get(tsfm_label, TSFM_TO_MOBIBOX_LABELS["__default__"])


def run_tsfm_inference(imu_data: List[dict]) -> Tuple[str, float, str]:
    """
    Run TSFM inference on IMU data.

    Args:
        imu_data: List of IMU records with acc_X/Y/Z, gyro_X/Y/Z, mag_X/Y/Z

    Returns:
        Tuple of (label, confidence, source) where:
        - label: MobiBox DB enum value (walking, running, sitting, standing, lying, climbing stairs, unknown)
        - confidence: Model confidence score (0.0 to 1.0)
        - source: "tsfm_model" or "tsfm_fallback"
    """
    import torch
    from .tsfm_model.preprocessing import preprocess_imu_data
    from .tsfm_model.label_groups import LABEL_GROUPS

    model, label_bank, available, device = _get_tsfm_model()

    if not available or model is None:
        raise RuntimeError("TSFM model not available")

    # Convert to numpy array (N, 9)
    imu_array = _imu_data_to_array(imu_data)

    # Preprocess
    data_tensor = torch.from_numpy(imu_array).float()
    patches, _ = preprocess_imu_data(
        data=data_tensor,
        sampling_rate_hz=TSFM_SAMPLING_RATE,
        patch_size_sec=TSFM_PATCH_SIZE_SEC,
        target_patch_size=TSFM_TARGET_PATCH_SIZE,
        normalization_method='zscore'
    )

    # Add batch dimension: (num_patches, 64, 9) -> (1, num_patches, 64, 9)
    patches = patches.unsqueeze(0).to(device)
    num_patches = patches.shape[1]
    num_channels = patches.shape[3]

    # Create masks
    channel_mask = torch.ones(1, num_channels, dtype=torch.bool, device=device)
    patch_mask = torch.ones(1, num_patches, dtype=torch.bool, device=device)

    # Get channel descriptions (first num_channels channels)
    channel_descs = [TSFM_CHANNEL_DESCRIPTIONS[:num_channels]]

    # Run inference
    with torch.no_grad():
        # Get embedding from model
        embedding = model(
            patches,
            channel_descs,
            channel_mask=channel_mask,
            patch_mask=patch_mask
        )

        # Get all labels from LABEL_GROUPS
        all_labels = list(LABEL_GROUPS.keys())

        # Encode labels with label bank
        label_embeddings = label_bank.encode(all_labels, normalize=True)

        # Handle per-patch prediction (model outputs (batch, num_patches, dim))
        # Pool across patches by taking mean embedding
        if embedding.dim() == 3:
            # Mean pool across patches: (batch, num_patches, dim) -> (batch, dim)
            embedding = embedding.mean(dim=1)

        # Compute cosine similarity
        # embedding: (batch, dim), label_embeddings: (num_labels, dim)
        similarities = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(1),  # (batch, 1, dim)
            label_embeddings.unsqueeze(0),  # (1, num_labels, dim)
            dim=-1
        ).squeeze(0)  # (num_labels,)

        # Temperature-scaled softmax for confidence
        temperature = 10.0
        probs = torch.nn.functional.softmax(similarities * temperature, dim=0)

        # Get prediction
        pred_idx = similarities.argmax().item()
        tsfm_label = all_labels[pred_idx]
        confidence = probs[pred_idx].item()

    # Map TSFM label to MobiBox label
    mobibox_label = _map_tsfm_label_to_mobibox(tsfm_label)

    logger.debug(f"TSFM prediction: {tsfm_label} -> {mobibox_label} (confidence: {confidence:.3f})")

    return mobibox_label, round(confidence, 2), "tsfm_model"


def is_tsfm_available() -> bool:
    """Check if TSFM model is available (loaded or loadable)."""
    if _tsfm_available is not None:
        return _tsfm_available

    # Try to load
    _, _, available, _ = _get_tsfm_model()
    return available