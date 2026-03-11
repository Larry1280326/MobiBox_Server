"""Lightweight HAR model implementations.

This module provides ultra-lightweight deep learning models for
Human Activity Recognition (HAR) using IMU sensor data.

Models:
- TinierHAR: Depthwise separable conv + BiGRU + attention (~34K params)
"""

from .model import TinierHAR
from .config import TINIER_HAR_CONFIG

__all__ = ["TinierHAR", "TINIER_HAR_CONFIG"]