"""Configuration for lightweight HAR models."""

# TinierHAR model configuration
# Based on: https://arxiv.org/html/2507.07949v1
#
# Architecture:
# - Stage 1: Depthwise separable conv (9 -> 64 channels, 4x temporal pooling)
# - Stage 2: Depthwise separable conv (64 -> 128 channels, 2x pooling)
# - Stage 3: Bidirectional GRU (128 hidden)
# - Stage 4: Attention-based temporal aggregation
# - Head: Classification (7 classes)
#
# Total parameters: ~93K (225x smaller than TSFM, 2.4x smaller than legacy transformer)

TINIER_HAR_CONFIG = {
    # Input configuration
    "input_channels": 9,  # acc_X/Y/Z, gyro_X/Y/Z, mag_X/Y/Z
    "window_size": 50,    # 1 second at 50Hz

    # Stage 1: First depthwise separable conv block
    "stage1_channels": 64,
    "stage1_kernel": 5,
    "stage1_pool": 4,     # 50 -> 12 timesteps

    # Stage 2: Second depthwise separable conv block
    "stage2_channels": 128,
    "stage2_kernel": 3,
    "stage2_pool": 2,     # 12 -> 6 timesteps

    # Stage 3: Bidirectional GRU
    "gru_hidden": 64,     # Hidden size (output is 128 due to bidirectional)
    "gru_layers": 1,
    "gru_dropout": 0.1,

    # Stage 4: Attention aggregation
    "attention_dim": 128,

    # Classification head
    "hidden_dim": 64,
    "num_classes": 7,     # unknown, standing, sitting, lying, walking, climbing stairs, running
    "dropout": 0.1,
}

# Model variants for different use cases
TINIER_HAR_VARIANTS = {
    # Standard configuration (default)
    "standard": TINIER_HAR_CONFIG,

    # Ultra-lightweight (fewer channels)
    "tiny": {
        "input_channels": 9,
        "window_size": 50,
        "stage1_channels": 32,
        "stage1_kernel": 5,
        "stage1_pool": 4,
        "stage2_channels": 64,
        "stage2_kernel": 3,
        "stage2_pool": 2,
        "gru_hidden": 32,
        "gru_layers": 1,
        "gru_dropout": 0.1,
        "attention_dim": 64,
        "hidden_dim": 32,
        "num_classes": 7,
        "dropout": 0.1,
    },

    # For 6-channel input (acc + gyro only, no magnetometer)
    "6ch": {
        "input_channels": 6,  # acc_X/Y/Z, gyro_X/Y/Z
        "window_size": 50,
        "stage1_channels": 64,
        "stage1_kernel": 5,
        "stage1_pool": 4,
        "stage2_channels": 128,
        "stage2_kernel": 3,
        "stage2_pool": 2,
        "gru_hidden": 64,
        "gru_layers": 1,
        "gru_dropout": 0.1,
        "attention_dim": 128,
        "hidden_dim": 64,
        "num_classes": 7,
        "dropout": 0.1,
    },
}

# Label mapping (must match existing HAR labels)
HAR_LABELS = [
    "unknown",         # 0
    "standing",        # 1
    "sitting",         # 2
    "lying",           # 3
    "walking",         # 4
    "climbing stairs", # 5
    "running",         # 6
]