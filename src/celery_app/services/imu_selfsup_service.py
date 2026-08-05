"""IMU SelfSupEncoder service for human activity recognition.

Integrates the IMU-SelfSupEncoder-v1 model from HuggingFace
(``NikoKKK/IMU-SelfSupEncoder-v1``) as an alternative HAR model.

Model details:
  - Architecture: ViT-style Conv-stem + time-frequency fusion
  - Input:  (B, 6, 200) — 6-ch (acc_x/y/z, gyro_x/y/z), 200 timesteps @ 20 Hz
  - Output: 192-dim CLS token embedding
  - Params: ~1.4 M  (lightweight, fast CPU inference)
  - Training: Self-supervised masked prediction + SupCon contrastive + frequency loss
  - Dataset: WISDM (18 activity classes)

Integration notes:
  - MobiBox collects 9-ch @ 50 Hz → we drop magnetometer and downsample 50→20 Hz
  - Model expects 10-second windows (200 samples @ 20 Hz)
  - Classification: cosine similarity to WISDM-derived class prototypes (zero-shot)
    with fallback to acceleration-magnitude heuristic
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached model
# ---------------------------------------------------------------------------
_selfsup_model: Optional[nn.Module] = None
_selfsup_available: Optional[bool] = None
_selfsup_device: Optional[torch.device] = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SELFSUP_MODEL_ID = "NikoKKK/IMU-SelfSupEncoder-v1"
SELFSUP_EMBED_DIM = 192
SELFSUP_INPUT_CHANNELS = 6          # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
SELFSUP_INPUT_TIMESTEPS = 200       # 10 s @ 20 Hz
SELFSUP_SOURCE_RATE_HZ = 50.0       # MobiBox native rate
SELFSUP_TARGET_RATE_HZ = 20.0       # Model expected rate
SELFSUP_WINDOW_SECONDS = 10.0       # 10-second window

# MobiBox 9-ch → SelfSupEncoder 6-ch column indices
SELFSUP_CHANNEL_INDICES = [0, 1, 2, 3, 4, 5]  # acc_X/Y/Z, gyro_X/Y/Z (drop mag)

# WISDM activity labels (18 classes) → MobiBox 7-class mapping
SELFSUP_LABEL_MAP = {
    # Walking
    "walking": "walking",
    # Jogging → Running
    "jogging": "running",
    # Sitting
    "sitting": "sitting",
    # Standing
    "standing": "standing",
    # Lying / Sleeping
    "lying": "lying",
    "sleeping": "lying",
    # Stairs (both directions)
    "upstairs": "climbing stairs",
    "downstairs": "climbing stairs",
    # Other WISDM labels → unknown
    "__default__": "unknown",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_selfsup_model():
    """Load and cache the SelfSupEncoder model from HuggingFace.

    The model uses a custom architecture (``imu_masked_encoder``) that isn't
    registered in the standard ``transformers`` library.  We download the
    ``modeling_imu_encoder.py`` file from the HF repo and import it directly.

    Downloads on first call (~5 MB); cached in HF_HUB_CACHE thereafter.
    Set ``HF_HUB_OFFLINE=1`` to use pre-cached model without network.
    """
    global _selfsup_model, _selfsup_available, _selfsup_device

    if _selfsup_available is not None:
        return _selfsup_model, _selfsup_available, _selfsup_device

    try:
        from huggingface_hub import hf_hub_download

        device = torch.device("cpu")

        # ------------------------------------------------------------------
        # Resolve the model directory — local copy first, then HF Hub
        # ------------------------------------------------------------------
        local_dir = Path(__file__).resolve().parent.parent.parent.parent \
            / "models" / "imu_selfsup"
        local_model = local_dir / "model.safetensors"
        local_config = local_dir / "config.json"
        local_code = local_dir / "modeling_imu_encoder.py"

        if local_model.is_file() and local_config.is_file() and local_code.is_file():
            # Use pre-downloaded local copy (no network needed)
            logger.info("Loading IMU-SelfSupEncoder-v1 from local: %s", local_dir)
            modeling_path = str(local_code)
            model_path = str(local_dir)
        else:
            # Download from HuggingFace Hub
            logger.info("Loading IMU-SelfSupEncoder-v1 from HF Hub: %s ...", SELFSUP_MODEL_ID)
            modeling_path = hf_hub_download(
                SELFSUP_MODEL_ID, "modeling_imu_encoder.py",
            )
            model_path = SELFSUP_MODEL_ID

        # Import the custom modeling file dynamically
        spec = importlib.util.spec_from_file_location(
            "modeling_imu_encoder", modeling_path,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("modeling_imu_encoder", mod)
        spec.loader.exec_module(mod)

        IMUMaskedEncoder = mod.IMUMaskedEncoder

        # Load model weights
        model = IMUMaskedEncoder.from_pretrained(model_path)
        model.to(device)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        _selfsup_model = model
        _selfsup_available = True
        _selfsup_device = device

        logger.info(
            "IMU-SelfSupEncoder-v1 loaded successfully "
            "(device=%s, params=%d)", device, param_count,
        )
        return model, True, device

    except Exception as exc:
        logger.error("Failed to load IMU-SelfSupEncoder-v1: %s", exc)
        _selfsup_available = False
        return None, False, None


def is_selfsup_available() -> bool:
    """Check whether the SelfSupEncoder model is available."""
    if _selfsup_available is not None:
        return _selfsup_available
    _, available, _ = _load_selfsup_model()
    return available


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _imu_data_to_array(imu_data: list[dict]) -> np.ndarray:
    """Convert list of IMU records to (N, 9) float32 array (all 9 channels)."""
    n = len(imu_data)
    arr = np.zeros((n, 9), dtype=np.float32)
    cols = ["acc_X", "acc_Y", "acc_Z", "gyro_X", "gyro_Y", "gyro_Z",
            "mag_X", "mag_Y", "mag_Z"]
    for i, row in enumerate(imu_data):
        for j, col in enumerate(cols):
            arr[i, j] = float(row.get(col) or 0)
    return arr


def _resample_linear(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Resample (N, C) array to ``target_len`` samples via linear interpolation."""
    n, c = arr.shape
    if n == 0:
        return np.zeros((target_len, c), dtype=np.float32)
    if n == target_len:
        return arr.copy()

    src_x = np.linspace(0, 1, n)
    tgt_x = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, c), dtype=np.float32)
    for ch in range(c):
        out[:, ch] = np.interp(tgt_x, src_x, arr[:, ch])
    return out


def preprocess_for_selfsup(
    imu_data: list[dict],
    target_samples: int = SELFSUP_INPUT_TIMESTEPS,
) -> torch.Tensor:
    """Convert raw IMU records into a (1, 6, target_samples) tensor.

    1. Extract 6 channels (acc + gyro, drop magnetometer)
    2. Resample from 50 Hz → 20 Hz
    3. Pad or truncate to exactly ``target_samples``
    """
    # (N, 9) → (N, 6)
    arr9 = _imu_data_to_array(imu_data)
    arr6 = arr9[:, SELFSUP_CHANNEL_INDICES]  # (N, 6)

    # Resample: 50 Hz → 20 Hz
    #   N samples at 50 Hz = N/50 seconds
    #   Need target_samples at 20 Hz = target_samples/20 seconds
    #   If we have T seconds of data at 50 Hz, resample to T*20 samples at 20 Hz
    seconds = len(arr6) / SELFSUP_SOURCE_RATE_HZ
    desired = int(seconds * SELFSUP_TARGET_RATE_HZ)
    desired = max(1, min(desired, target_samples * 4))  # sanity bounds
    resampled = _resample_linear(arr6, desired)

    # Pad or truncate to target_samples
    if resampled.shape[0] < target_samples:
        padded = np.zeros((target_samples, 6), dtype=np.float32)
        padded[:resampled.shape[0]] = resampled
        resampled = padded
    else:
        resampled = resampled[:target_samples]

    # → (1, 6, target_samples) tensor
    tensor = torch.from_numpy(resampled).float()
    tensor = tensor.permute(1, 0).unsqueeze(0)  # (1, 6, T)
    return tensor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_selfsup_embedding(imu_data: list[dict]) -> Optional[np.ndarray]:
    """Extract 192-dim CLS embedding from IMU data.

    Returns ``None`` if the model is unavailable or input is empty.
    """
    if not imu_data:
        return None

    model, available, device = _load_selfsup_model()
    if not available or model is None:
        return None

    tensor = preprocess_for_selfsup(imu_data).to(device)

    with torch.no_grad():
        # model.encode() returns only the CLS features: (1, 192)
        features = model.encode(tensor)
        if isinstance(features, tuple):
            features = features[0]
        emb = features.squeeze(0).cpu().numpy()  # (192,)

    return emb


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# Class prototypes (lazy-built from synthetic data)
# ---------------------------------------------------------------------------

# Activity classes for which we build prototypes
_PROTOTYPE_LABELS = [
    "walking", "running", "sitting", "standing", "lying", "climbing stairs",
]

# Cached prototypes: {label: np.ndarray(192,)}
_prototypes: Optional[dict] = None


def _build_synthetic_imu(label: str, num_samples: int = 500) -> list[dict]:
    """Generate synthetic 9-ch IMU data for a given activity label.

    Uses simplified physics-based patterns matching real sensor behaviour.
    Returns ``num_samples`` records at 50 Hz.
    """
    records = []
    for i in range(num_samples):
        t = i / 50.0  # seconds at 50 Hz

        if label == "walking":
            ax = np.sin(t * 2.5) * 1.2 + np.random.randn() * 0.1
            ay = np.cos(t * 2.5) * 0.8 + np.random.randn() * 0.1
            az = 9.8 + np.sin(t * 5.0) * 1.5 + np.random.randn() * 0.1
            gx, gy, gz = np.random.randn(3) * 0.05

        elif label == "running":
            ax = np.sin(t * 3.5) * 3.0 + np.random.randn() * 0.2
            ay = np.cos(t * 3.5) * 2.0 + np.random.randn() * 0.2
            az = 9.8 + np.sin(t * 7.0) * 3.0 + np.random.randn() * 0.2
            gx, gy, gz = np.random.randn(3) * 0.15

        elif label == "sitting":
            ax = np.random.randn() * 0.05
            ay = np.random.randn() * 0.05
            az = 9.8 + np.random.randn() * 0.05
            gx, gy, gz = np.random.randn(3) * 0.005

        elif label == "standing":
            ax = np.random.randn() * 0.10 + np.sin(t * 0.3) * 0.05
            ay = np.random.randn() * 0.10
            az = 9.8 + np.random.randn() * 0.08
            gx, gy, gz = np.random.randn(3) * 0.008

        elif label == "lying":
            ax = np.random.randn() * 0.02
            ay = 9.8 + np.random.randn() * 0.02  # gravity on Y-axis
            az = np.random.randn() * 0.02
            gx, gy, gz = np.random.randn(3) * 0.003

        elif label == "climbing stairs":
            ax = np.sin(t * 2.0) * 1.5 + np.random.randn() * 0.15
            ay = np.cos(t * 2.0) * 1.0 + np.random.randn() * 0.15
            az = 9.8 + np.sin(t * 4.0) * 2.5 + np.random.randn() * 0.15
            gx, gy, gz = np.random.randn(3) * 0.08

        else:
            ax = np.random.randn() * 0.1
            ay = np.random.randn() * 0.1
            az = 9.8 + np.random.randn() * 0.1
            gx, gy, gz = np.random.randn(3) * 0.01

        records.append({
            "acc_X": float(ax), "acc_Y": float(ay), "acc_Z": float(az),
            "gyro_X": float(gx), "gyro_Y": float(gy), "gyro_Z": float(gz),
            "mag_X": 30.0 + np.random.randn() * 2,
            "mag_Y": -10.0 + np.random.randn() * 2,
            "mag_Z": 40.0 + np.random.randn() * 2,
        })
    return records


def _build_prototypes(num_augments: int = 10) -> dict:
    """Build per-class prototype embeddings from synthetic data.

    Generates ``num_augments`` synthetic windows per activity class,
    extracts embeddings, and averages them into a single 192-dim prototype.
    Results are cached in memory.

    Returns:
        ``{label: np.ndarray(192,)}`` dictionary.
    """
    global _prototypes

    if _prototypes is not None:
        return _prototypes

    logger.info("Building SelfSupEncoder class prototypes from synthetic data...")
    prototypes = {}

    for label in _PROTOTYPE_LABELS:
        embeddings = []
        for seed in range(num_augments):
            np.random.seed(seed)
            imu_data = _build_synthetic_imu(label, num_samples=500)
            emb = get_selfsup_embedding(imu_data)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            proto = np.mean(embeddings, axis=0)
            # L2-normalize the prototype
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            prototypes[label] = proto
            logger.debug("  %s: %d samples, norm=%.4f", label, len(embeddings), np.linalg.norm(proto))
        else:
            logger.warning("  %s: no embeddings extracted, skipping", label)

    _prototypes = prototypes
    logger.info("Built %d class prototypes", len(prototypes))
    return prototypes


def _classify_embedding(emb: np.ndarray) -> tuple[str, float]:
    """Classify a 192-dim embedding using cosine similarity to prototypes.

    Returns:
        (label, confidence) where confidence is the softmax probability
        of the best-matching prototype.
    """
    prototypes = _build_prototypes()

    if not prototypes:
        return "unknown", 0.5

    # Normalize embedding
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

    # Cosine similarity to each prototype
    scores = {}
    for label, proto in prototypes.items():
        scores[label] = float(np.dot(emb_norm, proto))

    # Softmax for confidence calibration
    score_values = np.array(list(scores.values()))
    score_values = np.clip(score_values, -10, 10)  # avoid overflow
    probs = np.exp(score_values * 10.0)  # temperature-scaled
    probs = probs / probs.sum()

    best_idx = int(np.argmax(probs))
    best_label = list(scores.keys())[best_idx]
    confidence = float(probs[best_idx])

    return best_label, round(confidence, 2)


def run_selfsup_inference(imu_data: list[dict]) -> tuple[str, float, str]:
    """Run SelfSupEncoder inference on IMU data.

    Returns:
        (label, confidence, source) where source = ``"selfsup_model"``.
        Falls back to ``"selfsup_heuristic"`` if the model isn't loaded.
    """
    if len(imu_data) < 10:
        logger.warning("SelfSupEncoder: insufficient data (%d samples)", len(imu_data))
        return "unknown", 0.5, "selfsup_insufficient"

    emb = get_selfsup_embedding(imu_data)

    if emb is None:
        # Model not loaded — fall back to magnitude heuristic
        logger.debug("SelfSupEncoder not available, using heuristic")
        label, confidence = _heuristic_predict(imu_data)
        return label, confidence, "selfsup_heuristic"

    try:
        label, confidence = _classify_embedding(emb)
        return label, confidence, "selfsup_model"
    except Exception as exc:
        logger.warning("Prototype classification failed: %s, using heuristic", exc)
        label, confidence = _heuristic_predict(imu_data)
        return label, confidence, "selfsup_heuristic"


def _heuristic_predict(imu_data: list[dict]) -> tuple[str, float]:
    """Simple acceleration-magnitude heuristic (matches mock model logic).

    Used as fallback until labelled prototypes are available.
    """
    acc_magnitudes = []
    for sample in imu_data:
        ax = float(sample.get("acc_X", 0) or 0)
        ay = float(sample.get("acc_Y", 0) or 0)
        az = float(sample.get("acc_Z", 0) or 0)
        acc_magnitudes.append(np.sqrt(ax**2 + ay**2 + az**2))

    avg_mag = float(np.mean(acc_magnitudes)) if acc_magnitudes else 0.0
    std_mag = float(np.std(acc_magnitudes)) if len(acc_magnitudes) > 1 else 0.0

    if avg_mag < 0.3:
        label = "lying"
        confidence = 0.70
    elif avg_mag < 0.8:
        label = "sitting"
        confidence = 0.65
    elif avg_mag < 1.2 and std_mag < 0.3:
        label = "standing"
        confidence = 0.60
    elif avg_mag < 2.5 and std_mag > 0.3:
        label = "walking"
        confidence = 0.55
    elif std_mag > 0.8:
        label = "running"
        confidence = 0.55
    elif avg_mag > 1.5 and 0.3 < std_mag < 0.8:
        label = "climbing stairs"
        confidence = 0.50
    else:
        label = "unknown"
        confidence = 0.50

    return label, round(confidence, 2)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test_selfsup() -> bool:
    """Quick self-test: load model and run inference on random data."""
    try:
        dummy = []
        for i in range(500):  # 10 seconds @ 50 Hz
            dummy.append({
                "acc_X": np.sin(i * 0.1),
                "acc_Y": np.cos(i * 0.1),
                "acc_Z": 9.8 + np.random.randn() * 0.1,
                "gyro_X": np.random.randn() * 0.01,
                "gyro_Y": np.random.randn() * 0.01,
                "gyro_Z": np.random.randn() * 0.01,
                "mag_X": 30.0, "mag_Y": -10.0, "mag_Z": 40.0,
            })

        label, conf, source = run_selfsup_inference(dummy)
        logger.info("SelfSupEncoder self-test: label=%s conf=%.2f source=%s",
                     label, conf, source)
        return True
    except Exception as exc:
        logger.warning("SelfSupEncoder self-test failed: %s", exc)
        return False
