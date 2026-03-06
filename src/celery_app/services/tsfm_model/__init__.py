"""TSFM (Time Series Foundation Model) for Human Activity Recognition.

This package contains the TSFM model components adapted for MobiBox:
- encoder.py: IMU Activity Recognition Encoder
- semantic_alignment.py: Semantic alignment modules
- token_text_encoder.py: Token-level text encoding
- feature_extractor.py: CNN feature extractors
- transformer.py: Transformer modules
- positional_encoding.py: Positional encoding modules
- preprocessing.py: IMU data preprocessing
- label_groups.py: Activity label groupings
- model_loading.py: Model loading utilities
"""

from .config import SMALL_CONFIG, SMALL_DEEP_CONFIG, get_config
from .encoder import IMUActivityRecognitionEncoder
from .semantic_alignment import SemanticAlignmentHead
from .token_text_encoder import TokenTextEncoder, LearnableLabelBank
from .preprocessing import preprocess_imu_data, create_patches, interpolate_patches, normalize_patches

__all__ = [
    "SMALL_CONFIG",
    "SMALL_DEEP_CONFIG",
    "get_config",
    "IMUActivityRecognitionEncoder",
    "SemanticAlignmentHead",
    "TokenTextEncoder",
    "LearnableLabelBank",
    "preprocess_imu_data",
    "create_patches",
    "interpolate_patches",
    "normalize_patches",
]