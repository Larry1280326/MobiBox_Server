"""API routes for IMU test endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.imu_test.schemas import IMUTestRequest, IMUTestResponse, VALID_ACTIVITY_LABELS
from src.imu_test.service import predict_activity, save_test_result, get_test_statistics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/imu_test", tags=["imu_test"])


@router.post("/predict", response_model=IMUTestResponse)
async def imu_predict(request: IMUTestRequest):
    """
    Predict activity from IMU data and optionally compare with ground truth.

    This endpoint is for testing and evaluating the TSFM model accuracy.
    It accepts IMU sensor data and an optional ground truth label, then
    returns the predicted activity along with accuracy metrics.

    Minimum 50 samples (1 second at 50Hz) required for meaningful prediction.
    """
    try:
        # Run prediction
        result = await predict_activity(request)

        # Save result to database for evaluation
        await save_test_result(result)

        return result

    except Exception as e:
        logger.error(f"IMU test prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(
    user: Optional[str] = Query(default=None, description="Filter by user ID")
):
    """
    Get accuracy statistics from IMU test results.

    Returns overall accuracy and per-label breakdown.
    """
    try:
        stats = await get_test_statistics(user)
        return stats
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/labels")
async def get_valid_labels():
    """Get list of valid activity labels for ground truth."""
    return {"labels": VALID_ACTIVITY_LABELS}