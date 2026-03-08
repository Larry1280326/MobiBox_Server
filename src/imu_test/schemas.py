"""Schemas for IMU test endpoints."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# Valid activity labels for ground truth
VALID_ACTIVITY_LABELS = [
    "walking",
    "running",
    "sitting",
    "standing",
    "lying",
    "climbing stairs",
    "unknown",
]


class IMUTestItem(BaseModel):
    """Single IMU data point for testing."""

    timestamp: Optional[datetime] = None
    acc_X: float = Field(..., description="Accelerometer X-axis value")
    acc_Y: float = Field(..., description="Accelerometer Y-axis value")
    acc_Z: float = Field(..., description="Accelerometer Z-axis value")
    gyro_X: float = Field(..., description="Gyroscope X-axis value")
    gyro_Y: float = Field(..., description="Gyroscope Y-axis value")
    gyro_Z: float = Field(..., description="Gyroscope Z-axis value")
    mag_X: Optional[float] = Field(default=0.0, description="Magnetometer X-axis value")
    mag_Y: Optional[float] = Field(default=0.0, description="Magnetometer Y-axis value")
    mag_Z: Optional[float] = Field(default=0.0, description="Magnetometer Z-axis value")


class IMUTestRequest(BaseModel):
    """Request for IMU activity prediction test."""

    user: str = Field(..., description="User identifier")
    ground_truth_label: Optional[str] = Field(
        default=None,
        description="Ground truth activity label (for evaluation)",
    )
    imu_data: list[IMUTestItem] = Field(
        ...,
        min_length=50,
        description="IMU sensor data (minimum 50 samples for 1 second at 50Hz)",
    )

    def validate_ground_truth_label(self) -> Optional[str]:
        """Validate and normalize ground truth label."""
        if self.ground_truth_label is None:
            return None
        label = self.ground_truth_label.lower().strip()
        # Handle common variations
        label_mapping = {
            "stairs": "climbing stairs",
            "upstairs": "climbing stairs",
            "downstairs": "climbing stairs",
            "climbing_stairs": "climbing stairs",
            "walk": "walking",
            "run": "running",
            "jog": "running",
            "jogging": "running",
            "sit": "sitting",
            "stand": "standing",
            "lay": "lying",
            "laying": "lying",
            "lie": "lying",
        }
        return label_mapping.get(label, label) if label in VALID_ACTIVITY_LABELS or label in label_mapping else None


class IMUTestResponse(BaseModel):
    """Response for IMU activity prediction test."""

    user: str
    predicted_label: str = Field(..., description="Predicted activity label")
    confidence: float = Field(..., description="Model confidence score (0.0 to 1.0)")
    source: str = Field(..., description="Model source (tsfm_model, mock_har, etc.)")
    ground_truth_label: Optional[str] = Field(
        default=None,
        description="Ground truth label if provided",
    )
    is_correct: Optional[bool] = Field(
        default=None,
        description="Whether prediction matches ground truth (if provided)",
    )
    sample_count: int = Field(..., description="Number of IMU samples processed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class IMUTestResultRecord(BaseModel):
    """Record for storing IMU test results in database."""

    user: str
    predicted_label: str
    confidence: float
    source: str
    ground_truth_label: Optional[str]
    is_correct: Optional[bool]
    sample_count: int
    timestamp: str  # ISO format string for Supabase