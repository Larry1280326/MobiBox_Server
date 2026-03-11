"""Inference service for lightweight HAR models.

Provides a simple interface for running activity recognition on IMU data.
Integrates with the existing HAR service pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cached model instance
_model = None
_model_available = None
_model_config = None


def get_lightweight_har_model(
    checkpoint_path: Optional[str] = None,
    variant: str = "standard",
    device: str = "cpu",
):
    """Load and cache the lightweight HAR model.

    Args:
        checkpoint_path: Path to model checkpoint (optional, will use random weights if not provided)
        variant: Model variant ("standard", "tiny", "6ch")
        device: Device to run model on ("cpu" or "cuda")

    Returns:
        Tuple of (model, available)
    """
    global _model, _model_available, _model_config

    if _model_available is not None:
        return _model, _model_available

    try:
        import torch
        from .model import TinierHAR
        from .config import TINIER_HAR_CONFIG

        # Create model
        _model = TinierHAR(TINIER_HAR_CONFIG)
        _model_config = TINIER_HAR_CONFIG

        # Load checkpoint if provided
        if checkpoint_path:
            path = Path(checkpoint_path)
            if path.is_file():
                state = torch.load(path, map_location=device, weights_only=True)
                if isinstance(state, dict):
                    if "model_state_dict" in state:
                        state = state["model_state_dict"]
                    elif "state_dict" in state:
                        state = state["state_dict"]
                _model.load_state_dict(state, strict=True)
                logger.info(f"Loaded lightweight HAR checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}, using random weights")
        else:
            logger.info("No checkpoint provided, using random weights")

        _model.to(device)
        _model.eval()
        _model_available = True

        logger.info(f"Lightweight HAR model initialized: {_model.count_parameters():,} parameters")
        return _model, True

    except Exception as e:
        logger.error(f"Failed to load lightweight HAR model: {e}", exc_info=True)
        _model_available = False
        return None, False


def is_lightweight_har_available() -> bool:
    """Check if lightweight HAR model is available."""
    global _model_available
    if _model_available is None:
        _, _model_available = get_lightweight_har_model()
    return _model_available


def preprocess_imu_data(
    imu_data: list[dict],
    window_size: int = 50,
    input_channels: int = 9,
) -> np.ndarray:
    """Preprocess IMU data for model input.

    Converts raw IMU sensor readings to normalized tensor format.

    Args:
        imu_data: List of IMU sensor readings (from Supabase)
        window_size: Number of timesteps expected by model
        input_channels: Number of input channels (9 for acc+gyro+mag, 6 for acc+gyro)

    Returns:
        Numpy array of shape (1, input_channels, window_size)
    """
    # Column order for 9-channel input
    columns_9ch = [
        "acc_X", "acc_Y", "acc_Z",
        "gyro_X", "gyro_Y", "gyro_Z",
        "mag_X", "mag_Y", "mag_Z",
    ]

    # Column order for 6-channel input (no magnetometer)
    columns_6ch = [
        "acc_X", "acc_Y", "acc_Z",
        "gyro_X", "gyro_Y", "gyro_Z",
    ]

    columns = columns_9ch if input_channels == 9 else columns_6ch

    # Build input array
    n = len(imu_data)
    data = np.zeros((window_size, input_channels), dtype=np.float32)

    for i in range(min(n, window_size)):
        row = imu_data[i]
        for j, col in enumerate(columns):
            val = row.get(col)
            data[i, j] = float(val) if val is not None else 0.0

    # Z-score normalization per channel
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std

    # Transpose to (channels, time) and add batch dimension
    # (window_size, channels) -> (1, channels, window_size)
    data = data.T[np.newaxis, :, :]

    return data


def run_lightweight_har_inference(
    imu_data: list[dict],
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> tuple[str, float, str]:
    """Run lightweight HAR inference on IMU data.

    Args:
        imu_data: List of IMU sensor readings (from Supabase)
        checkpoint_path: Path to model checkpoint (optional)
        device: Device to run on ("cpu" or "cuda")

    Returns:
        Tuple of (label, confidence, source)
    """
    import torch

    from .config import HAR_LABELS

    # Load model
    model, available = get_lightweight_har_model(checkpoint_path, device=device)
    if not available or model is None:
        raise RuntimeError("Lightweight HAR model not available")

    # Preprocess data
    window_size = model.window_size
    input_channels = model.input_channels
    tensor = preprocess_imu_data(imu_data, window_size, input_channels)

    # Convert to torch tensor
    x = torch.from_numpy(tensor).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    # Get label
    label = HAR_LABELS[pred_idx] if pred_idx < len(HAR_LABELS) else "unknown"

    return label, round(confidence, 4), "lightweight_har"


def count_model_parameters() -> dict:
    """Get model parameter counts.

    Returns:
        Dictionary with parameter statistics
    """
    model, available = get_lightweight_har_model()
    if not available or model is None:
        return {"error": "Model not available"}

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count by layer type
    conv_params = 0
    gru_params = 0
    attention_params = 0
    classifier_params = 0

    for name, p in model.named_parameters():
        n = p.numel()
        if "stage" in name or "conv" in name:
            conv_params += n
        elif "gru" in name:
            gru_params += n
        elif "attention" in name:
            attention_params += n
        elif "classifier" in name:
            classifier_params += n

    return {
        "total": total,
        "trainable": trainable,
        "conv": conv_params,
        "gru": gru_params,
        "attention": attention_params,
        "classifier": classifier_params,
        "size_mb": total * 4 / (1024 * 1024),
    }


if __name__ == "__main__":
    # Quick test
    import json

    print("Lightweight HAR Model Statistics:")
    print(json.dumps(count_model_parameters(), indent=2))

    # Test with mock data
    mock_imu = [
        {"acc_X": 0.1, "acc_Y": 0.2, "acc_Z": 9.8, "gyro_X": 0.01, "gyro_Y": 0.02, "gyro_Z": 0.03, "mag_X": 50, "mag_Y": 10, "mag_Z": -30}
        for _ in range(50)
    ]

    label, conf, source = run_lightweight_har_inference(mock_imu)
    print(f"\nTest inference: label={label}, confidence={conf}, source={source}")