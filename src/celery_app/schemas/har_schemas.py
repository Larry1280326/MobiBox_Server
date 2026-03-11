"""Pydantic schemas for HAR (Human Activity Recognition) processing."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class HARLabel(BaseModel):
    """HAR label result from processing IMU data."""

    user: str
    label: str
    confidence: Optional[float] = None
    timestamp: datetime
    source: str = "mock_har"  # Could be "mock_har", "ml_model", etc.


class IMUWindow(BaseModel):
    """IMU data window for processing."""

    user: str
    data: list[dict]
    start_time: datetime
    end_time: datetime