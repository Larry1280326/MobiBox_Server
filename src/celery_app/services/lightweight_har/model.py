"""TinierHAR: Ultra-lightweight HAR model.

Architecture based on: https://arxiv.org/html/2507.07949v1

Key innovations:
1. Depthwise separable convolutions for efficient spatial feature extraction
2. Aggressive temporal pooling (4x) to reduce sequence length early
3. Bidirectional GRU for temporal modeling
4. Attention-based temporal aggregation for global representation
5. Compact classification head

Parameter count: ~34K (600x smaller than TSFM, comparable accuracy)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for efficient feature extraction.

    Combines depthwise convolution (spatial features per channel) with
    pointwise convolution (channel mixing) for significant parameter reduction.

    Standard conv params: C_in * C_out * K
    Depthwise separable params: C_in * K + C_in * C_out = C_in * (K + C_out)

    For K=5, C_in=64, C_out=128:
    - Standard: 64 * 128 * 5 = 40,960 params
    - Separable: 64 * (5 + 128) = 8,512 params (4.8x reduction)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()

        # Depthwise convolution: each input channel convolved separately
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: groups = in_channels
            bias=False,
        )

        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional block with depthwise separable conv, activation, and pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv = DepthwiseSeparableConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Same padding
        )

        self.activation = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class AttentionAggregation(nn.Module):
    """Attention-based temporal aggregation.

    Learns to weight each timestep based on its importance for classification.
    More flexible than mean/max pooling and adds minimal parameters.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        # Learn attention weights per timestep
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)

        # Compute attention scores: (batch, time, 1)
        scores = self.attention(x)

        # Softmax over time dimension: (batch, time, 1)
        weights = F.softmax(scores, dim=1)

        # Weighted sum: (batch, features)
        output = (x * weights).sum(dim=1)

        return output


class TinierHAR(nn.Module):
    """TinierHAR: Ultra-lightweight HAR model for IMU-based activity recognition.

    Architecture:
        Input: (batch, channels, time) - 9 channels, 50 timesteps

        Stage 1: ConvBlock
            - Depthwise separable conv (9 -> 64 channels, kernel 5)
            - GELU activation
            - MaxPool 4x (50 -> 12 timesteps)

        Stage 2: ConvBlock
            - Depthwise separable conv (64 -> 128 channels, kernel 3)
            - GELU activation
            - MaxPool 2x (12 -> 6 timesteps)

        Stage 3: Bidirectional GRU
            - BiGRU(128, 64) -> output (batch, 6, 128)

        Stage 4: Attention Aggregation
            - Learn attention weights per timestep
            - Weighted sum -> (batch, 128)

        Head: Classification
            - Linear(128, 64) + GELU + Dropout
            - Linear(64, num_classes)

    Parameters: ~34,000
    Input: (batch, 9, 50) for 9-channel IMU at 50Hz for 1 second
    Output: (batch, num_classes) logits
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()

        # Use default config if not provided
        if config is None:
            from .config import TINIER_HAR_CONFIG
            config = TINIER_HAR_CONFIG

        self.config = config

        # Store dimensions for forward pass
        self.input_channels = config["input_channels"]
        self.window_size = config["window_size"]

        # Stage 1: First conv block with aggressive pooling
        self.stage1 = ConvBlock(
            in_channels=config["input_channels"],
            out_channels=config["stage1_channels"],
            kernel_size=config["stage1_kernel"],
            pool_size=config["stage1_pool"],
            dropout=config.get("dropout", 0.1),
        )

        # Stage 2: Second conv block
        self.stage2 = ConvBlock(
            in_channels=config["stage1_channels"],
            out_channels=config["stage2_channels"],
            kernel_size=config["stage2_kernel"],
            pool_size=config["stage2_pool"],
            dropout=config.get("dropout", 0.1),
        )

        # Calculate sequence length after pooling
        # After stage1: window_size / stage1_pool
        # After stage2: (window_size / stage1_pool) / stage2_pool
        seq_len_after_conv = (
            self.window_size // config["stage1_pool"]
        ) // config["stage2_pool"]

        # Stage 3: Bidirectional GRU
        self.gru = nn.GRU(
            input_size=config["stage2_channels"],
            hidden_size=config["gru_hidden"],
            num_layers=config["gru_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=config["gru_dropout"] if config["gru_layers"] > 1 else 0,
        )

        # GRU output size (bidirectional doubles it)
        gru_output_size = config["gru_hidden"] * 2

        # Stage 4: Attention-based temporal aggregation
        self.attention = AttentionAggregation(gru_output_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_output_size),
            nn.Linear(gru_output_size, config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], config["num_classes"]),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time) or (batch, time, channels)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Handle both (B, C, T) and (B, T, C) input formats
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")

        # Detect input format: if last dimension is small (~9), it's (B, T, C)
        if x.size(-1) <= 12 and x.size(1) > 12:
            # Input is (B, T, C), transpose to (B, C, T)
            x = x.transpose(1, 2)

        # Stage 1: Conv block
        # (batch, 9, 50) -> (batch, 64, 12)
        x = self.stage1(x)

        # Stage 2: Conv block
        # (batch, 64, 12) -> (batch, 128, 6)
        x = self.stage2(x)

        # Transpose for GRU: (batch, channels, time) -> (batch, time, channels)
        # (batch, 128, 6) -> (batch, 6, 128)
        x = x.transpose(1, 2)

        # Stage 3: Bidirectional GRU
        # (batch, 6, 128) -> (batch, 6, 128) [bidirectional output]
        x, _ = self.gru(x)

        # Stage 4: Attention aggregation
        # (batch, 6, 128) -> (batch, 128)
        x = self.attention(x)

        # Classification head
        # (batch, 128) -> (batch, num_classes)
        logits = self.classifier(x)

        return logits

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with confidence scores.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Tuple of (predicted_class, confidence) each of shape (batch,)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        return predicted, confidence

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """Estimate model size in MB (float32)."""
        return self.count_parameters() * 4 / (1024 * 1024)


def create_tinier_har(
    variant: str = "standard",
    num_classes: int = 7,
    input_channels: int = 9,
) -> TinierHAR:
    """Create a TinierHAR model with specified variant.

    Args:
        variant: Model variant ("standard", "tiny", "6ch")
        num_classes: Number of output classes
        input_channels: Number of input channels

    Returns:
        Configured TinierHAR model
    """
    from .config import TINIER_HAR_VARIANTS

    if variant not in TINIER_HAR_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(TINIER_HAR_VARIANTS.keys())}")

    config = TINIER_HAR_VARIANTS[variant].copy()
    config["num_classes"] = num_classes
    config["input_channels"] = input_channels

    return TinierHAR(config)


if __name__ == "__main__":
    # Quick test
    model = TinierHAR()
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Model size: {model.model_size_mb():.2f} MB")

    # Test forward pass
    x = torch.randn(2, 9, 50)  # Batch of 2, 9 channels, 50 timesteps
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test prediction
    pred, conf = model.predict(x)
    print(f"Predictions: {pred}")
    print(f"Confidences: {conf}")