"""Service for IMU test prediction and evaluation."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from src.database import get_supabase_client
from src.imu_test.schemas import IMUTestRequest, IMUTestResponse, IMUTestResultRecord

logger = logging.getLogger(__name__)

# Table name for storing test results
IMU_TEST_RESULTS_TABLE = "imu_test_results"


async def predict_activity(request: IMUTestRequest, timeout_seconds: float = 10.0) -> IMUTestResponse:
    """
    Predict activity from IMU data using the HAR model.

    Args:
        request: IMU test request with user, optional ground truth, and IMU data
        timeout_seconds: Maximum time to wait for inference (default: 10 seconds)

    Returns:
        IMUTestResponse with prediction and evaluation results
    """
    from src.celery_app.services.har_service import run_har_model
    from src.celery_app.services.tsfm_service import is_tsfm_available

    # Convert IMUTestItem to dict format expected by HAR service
    imu_data = [item.model_dump() for item in request.imu_data]

    logger.info(f"IMU test request: user={request.user}, samples={len(imu_data)}, ground_truth={request.ground_truth_label}")
    logger.info(f"TSFM model available: {is_tsfm_available()}")

    # Run HAR model prediction with timeout
    try:
        predicted_label, confidence, source = await asyncio.wait_for(
            run_har_model(imu_data),
            timeout=timeout_seconds
        )
        logger.info(f"Prediction result: label={predicted_label}, confidence={confidence}, source={source}")
    except asyncio.TimeoutError:
        logger.warning(f"Inference timeout after {timeout_seconds}s for user {request.user}")
        predicted_label = "unknown"
        confidence = 0.0
        source = "timeout"

    # Validate and normalize ground truth label
    ground_truth = request.validate_ground_truth_label()

    # Check if prediction is correct
    is_correct = None
    if ground_truth is not None:
        is_correct = (predicted_label == ground_truth)

    return IMUTestResponse(
        user=request.user,
        predicted_label=predicted_label,
        confidence=confidence,
        source=source,
        ground_truth_label=ground_truth,
        is_correct=is_correct,
        sample_count=len(imu_data),
        timestamp=datetime.now(),
    )


async def save_test_result(result: IMUTestResponse) -> bool:
    """
    Save IMU test result to database for evaluation.

    Args:
        result: The test result to save

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        client = get_supabase_client()

        record = IMUTestResultRecord(
            user=result.user,
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            source=result.source,
            ground_truth_label=result.ground_truth_label,
            is_correct=result.is_correct,
            sample_count=result.sample_count,
            timestamp=result.timestamp.isoformat(),
        )

        response = client.table(IMU_TEST_RESULTS_TABLE).insert(record.model_dump()).execute()

        if response.data:
            logger.info(f"Saved IMU test result for user {result.user}")
            return True
        return False

    except Exception as e:
        logger.error(f"Failed to save IMU test result: {e}")
        return False


async def get_test_statistics(user: Optional[str] = None) -> dict:
    """
    Get statistics from IMU test results.

    Args:
        user: Optional user filter

    Returns:
        Dictionary with accuracy statistics
    """
    try:
        client = get_supabase_client()

        query = client.table(IMU_TEST_RESULTS_TABLE).select("*")

        if user:
            query = query.eq("user", user)

        response = query.execute()

        if not response.data:
            return {"total": 0, "correct": 0, "accuracy": None}

        results = response.data
        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct") is True)

        # Calculate per-label accuracy
        label_stats = {}
        for r in results:
            gt = r.get("ground_truth_label")
            if gt:
                if gt not in label_stats:
                    label_stats[gt] = {"total": 0, "correct": 0}
                label_stats[gt]["total"] += 1
                if r.get("is_correct"):
                    label_stats[gt]["correct"] += 1

        # Calculate accuracy per label
        for label, stats in label_stats.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else None,
            "per_label": label_stats,
        }

    except Exception as e:
        logger.error(f"Failed to get test statistics: {e}")
        return {"error": str(e)}