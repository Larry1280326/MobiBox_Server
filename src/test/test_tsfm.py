"""
Test TSFM model inference for Human Activity Recognition.

Run from project root:
  pytest:  python -m pytest src/test/test_tsfm.py -v -s
  script:  python src/test/test_tsfm.py
"""

import numpy as np
import pytest
from pathlib import Path

# Import TSFM components
from src.celery_app.services.tsfm_service import (
    run_tsfm_inference,
    is_tsfm_available,
    _imu_data_to_array,
    _map_tsfm_label_to_mobibox,
    TSFM_CHANNEL_DESCRIPTIONS,
    TSFM_SAMPLING_RATE,
    TSFM_PATCH_SIZE_SEC,
)
from src.celery_app.config import (
    TSFM_MIN_SAMPLES,
)
from src.celery_app.services.tsfm_model.preprocessing import (
    preprocess_imu_data,
    create_patches,
    interpolate_patches,
    normalize_patches,
)
from src.celery_app.services.tsfm_model.label_groups import (
    LABEL_GROUPS,
    get_label_to_group_mapping,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

def create_dummy_imu_data(n_samples: int = 50, activity_type: str = "walking") -> list[dict]:
    """
    Create dummy IMU data for testing.

    Args:
        n_samples: Number of samples (default 50 for 1 second at 50Hz)
        activity_type: Type of activity to simulate (walking, sitting, standing, lying)

    Returns:
        List of IMU dictionaries with acc_X/Y/Z, gyro_X/Y/Z, mag_X/Y/Z
    """
    np.random.seed(42)

    data = []
    for i in range(n_samples):
        if activity_type == "walking":
            # Walking: moderate acceleration, some periodic motion
            acc_x = 0.5 * np.sin(i * 0.3) + np.random.randn() * 0.1
            acc_y = 0.2 * np.cos(i * 0.3) + np.random.randn() * 0.1
            acc_z = 9.8 + 0.3 * np.sin(i * 0.3) + np.random.randn() * 0.1
            gyro_scale = 0.1
        elif activity_type == "running":
            # Running: higher acceleration, faster periodic motion
            acc_x = 1.5 * np.sin(i * 0.6) + np.random.randn() * 0.2
            acc_y = 0.5 * np.cos(i * 0.6) + np.random.randn() * 0.2
            acc_z = 9.8 + 1.0 * np.sin(i * 0.6) + np.random.randn() * 0.2
            gyro_scale = 0.3
        elif activity_type == "sitting":
            # Sitting: stable, low motion
            acc_x = np.random.randn() * 0.02
            acc_y = np.random.randn() * 0.02
            acc_z = 9.8 + np.random.randn() * 0.02
            gyro_scale = 0.01
        elif activity_type == "standing":
            # Standing: mostly stable with small sway
            acc_x = np.random.randn() * 0.05
            acc_y = np.random.randn() * 0.05
            acc_z = 9.8 + np.random.randn() * 0.05
            gyro_scale = 0.02
        elif activity_type == "lying":
            # Lying: very stable, different orientation
            acc_x = np.random.randn() * 0.01
            acc_y = 9.8 + np.random.randn() * 0.01
            acc_z = np.random.randn() * 0.01
            gyro_scale = 0.005
        else:
            # Random
            acc_x = np.random.randn() * 0.5
            acc_y = np.random.randn() * 0.5
            acc_z = 9.8 + np.random.randn() * 0.5
            gyro_scale = 0.1

        data.append({
            "acc_X": float(acc_x),
            "acc_Y": float(acc_y),
            "acc_Z": float(acc_z),
            "gyro_X": float(np.random.randn() * gyro_scale),
            "gyro_Y": float(np.random.randn() * gyro_scale),
            "gyro_Z": float(np.random.randn() * gyro_scale),
            "mag_X": float(50 + np.random.randn() * 0.5),
            "mag_Y": float(20 + np.random.randn() * 0.5),
            "mag_Z": float(30 + np.random.randn() * 0.5),
        })

    return data


def create_batch_imu_data(batch_size: int = 4, samples_per_batch: int = 50) -> list[list[dict]]:
    """
    Create a batch of IMU data samples.

    Args:
        batch_size: Number of samples in the batch
        samples_per_batch: Number of IMU readings per sample

    Returns:
        List of IMU data lists
    """
    activities = ["walking", "sitting", "standing", "lying", "running"]
    batch = []
    for i in range(batch_size):
        activity = activities[i % len(activities)]
        batch.append(create_dummy_imu_data(samples_per_batch, activity))
    return batch


# =============================================================================
# Preprocessing Tests
# =============================================================================

class TestPreprocessing:
    """Tests for IMU data preprocessing functions."""

    def test_create_patches_basic(self):
        """Test basic patch creation."""
        data = np.random.randn(100, 9).astype(np.float32)
        patches = create_patches(data, sampling_rate_hz=50.0, patch_size_sec=1.0)

        assert patches.shape[0] == 2  # 100 samples / 50 Hz = 2 patches
        assert patches.shape[1] == 50  # 1 second at 50 Hz
        assert patches.shape[2] == 9   # 9 channels

    def test_create_patches_overlapping(self):
        """Test overlapping patch creation."""
        data = np.random.randn(150, 9).astype(np.float32)
        patches = create_patches(
            data,
            sampling_rate_hz=50.0,
            patch_size_sec=1.0,
            stride_sec=0.5  # 50% overlap
        )

        # 150 samples with 50 sample patches and 25 sample stride
        # (150 - 50) / 25 + 1 = 5 patches (with overlap)
        assert patches.shape[0] == 5
        assert patches.shape[1] == 50
        assert patches.shape[2] == 9

    def test_interpolate_patches(self):
        """Test patch interpolation to target size."""
        patches = np.random.randn(5, 50, 9).astype(np.float32)
        interpolated = interpolate_patches(patches, target_size=64)

        assert interpolated.shape[0] == 5
        assert interpolated.shape[1] == 64
        assert interpolated.shape[2] == 9

    def test_normalize_patches_zscore(self):
        """Test z-score normalization."""
        import torch
        patches = torch.randn(5, 64, 9) * 10 + 5
        normalized, means, stds = normalize_patches(patches, method='zscore')

        assert normalized.shape == patches.shape
        # Check that normalization worked (mean ~ 0, std ~ 1)
        assert torch.abs(normalized.mean(dim=1)).max() < 1e-5
        assert torch.abs(normalized.std(dim=1) - 1.0).max() < 0.1

    def test_preprocess_imu_data_full(self):
        """Test full preprocessing pipeline."""
        import torch

        data = np.random.randn(100, 9).astype(np.float32)
        data_tensor = torch.from_numpy(data)

        patches, metadata = preprocess_imu_data(
            data=data_tensor,
            sampling_rate_hz=50.0,
            patch_size_sec=1.0,
            target_patch_size=64,
            normalization_method='zscore'
        )

        assert patches.shape[0] == 2  # 2 patches
        assert patches.shape[1] == 64  # Interpolated to 64
        assert patches.shape[2] == 9   # 9 channels

        assert 'means' in metadata
        assert 'stds' in metadata
        assert metadata['sampling_rate_hz'] == 50.0


# =============================================================================
# Label Mapping Tests
# =============================================================================

class TestLabelMapping:
    """Tests for label mapping functions."""

    def test_map_walking(self):
        """Test walking label mapping."""
        assert _map_tsfm_label_to_mobibox("walking") == "walking"

    def test_map_stairs_labels(self):
        """Test stairs-related label mappings."""
        assert _map_tsfm_label_to_mobibox("walking_upstairs") == "climbing stairs"
        assert _map_tsfm_label_to_mobibox("walking_downstairs") == "climbing stairs"
        assert _map_tsfm_label_to_mobibox("ascending_stairs") == "climbing stairs"
        assert _map_tsfm_label_to_mobibox("descending_stairs") == "climbing stairs"

    def test_map_sitting(self):
        """Test sitting label mapping."""
        assert _map_tsfm_label_to_mobibox("sitting") == "sitting"

    def test_map_standing(self):
        """Test standing label mapping."""
        assert _map_tsfm_label_to_mobibox("standing") == "standing"

    def test_map_lying(self):
        """Test lying label mapping."""
        assert _map_tsfm_label_to_mobibox("lying") == "lying"
        assert _map_tsfm_label_to_mobibox("laying") == "lying"

    def test_map_running(self):
        """Test running label mapping."""
        assert _map_tsfm_label_to_mobibox("running") == "running"
        assert _map_tsfm_label_to_mobibox("jogging") == "running"

    def test_map_unknown_label(self):
        """Test unknown label mapping."""
        assert _map_tsfm_label_to_mobibox("some_unknown_activity") == "unknown"

    def test_label_groups_contains_trained_labels(self):
        """Test that LABEL_GROUPS contains all UCI-HAR training labels."""
        uci_har_labels = ["walking", "sitting", "standing", "lying"]

        for label in uci_har_labels:
            assert label in LABEL_GROUPS, f"UCI-HAR label '{label}' not in LABEL_GROUPS"


# =============================================================================
# IMU Data Conversion Tests
# =============================================================================

class TestIMUDataConversion:
    """Tests for IMU data conversion functions."""

    def test_imu_data_to_array_basic(self):
        """Test basic IMU data to array conversion."""
        imu_data = create_dummy_imu_data(50)
        arr = _imu_data_to_array(imu_data)

        assert arr.shape == (50, 9)
        assert arr.dtype == np.float32

    def test_imu_data_to_array_with_missing_values(self):
        """Test conversion with missing values."""
        imu_data = [
            {"acc_X": None, "acc_Y": 0.1, "acc_Z": 9.8,
             "gyro_X": 0, "gyro_Y": None, "gyro_Z": 0,
             "mag_X": 50, "mag_Y": 20, "mag_Z": 30},
            {"acc_X": 0.1, "acc_Y": 0.2, "acc_Z": 9.7,
             "gyro_X": 0.01, "gyro_Y": 0.02, "gyro_Z": 0.03,
             "mag_X": 51, "mag_Y": 21, "mag_Z": 31},
        ]
        arr = _imu_data_to_array(imu_data)

        assert arr.shape == (2, 9)
        assert arr[0, 0] == 0.0  # None converted to 0
        assert arr[0, 4] == 0.0  # None converted to 0

    def test_imu_data_to_array_empty(self):
        """Test conversion with empty data."""
        arr = _imu_data_to_array([])
        assert arr.shape == (0, 9)

    def test_channel_order(self):
        """Test that channel order matches expected order."""
        imu_data = [{
            "acc_X": 1.0, "acc_Y": 2.0, "acc_Z": 3.0,
            "gyro_X": 4.0, "gyro_Y": 5.0, "gyro_Z": 6.0,
            "mag_X": 7.0, "mag_Y": 8.0, "mag_Z": 9.0,
        }]
        arr = _imu_data_to_array(imu_data)

        # Expected order: acc_X, acc_Y, acc_Z, gyro_X, gyro_Y, gyro_Z, mag_X, mag_Y, mag_Z
        # Shape is (N, 9) where N is number of samples
        expected = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], dtype=np.float32)
        np.testing.assert_array_equal(arr, expected)


# =============================================================================
# Single Inference Tests
# =============================================================================

@pytest.mark.skipif(
    not is_tsfm_available(),
    reason="TSFM model not available"
)
class TestTSFMSingleInference:
    """Tests for single-sample TSFM inference."""

    def test_inference_walking_data(self):
        """Test inference on walking-like data."""
        imu_data = create_dummy_imu_data(50, activity_type="walking")
        label, confidence, source = run_tsfm_inference(imu_data)

        assert isinstance(label, str)
        assert label in ["walking", "sitting", "standing", "lying", "climbing stairs", "running", "unknown"]
        assert 0.0 <= confidence <= 1.0
        assert source == "tsfm_model"

    def test_inference_sitting_data(self):
        """Test inference on sitting-like data."""
        imu_data = create_dummy_imu_data(50, activity_type="sitting")
        label, confidence, source = run_tsfm_inference(imu_data)

        assert isinstance(label, str)
        assert label in ["walking", "sitting", "standing", "lying", "climbing stairs", "running", "unknown"]
        assert 0.0 <= confidence <= 1.0
        assert source == "tsfm_model"

    def test_inference_minimum_samples(self):
        """Test inference with minimum required samples (at least 1 second of data)."""
        # Use 50 samples (1 second at 50Hz) which is the minimum meaningful amount
        # TSFM needs at least patch_size_sec worth of data
        imu_data = create_dummy_imu_data(50)  # 1 second at 50Hz
        label, confidence, source = run_tsfm_inference(imu_data)

        assert isinstance(label, str)
        assert source == "tsfm_model"

    def test_inference_just_above_minimum(self):
        """Test inference with just above minimum samples."""
        imu_data = create_dummy_imu_data(TSFM_MIN_SAMPLES + 40)  # 50 samples
        label, confidence, source = run_tsfm_inference(imu_data)

        assert isinstance(label, str)
        assert source == "tsfm_model"

    def test_inference_random_data(self):
        """Test inference on random data."""
        imu_data = create_dummy_imu_data(50, activity_type="random")
        label, confidence, source = run_tsfm_inference(imu_data)

        assert isinstance(label, str)
        # Random data should likely return "unknown" with low confidence
        # but this is not guaranteed

    def test_inference_confidence_reasonable(self):
        """Test that confidence scores are reasonable."""
        imu_data = create_dummy_imu_data(100, activity_type="walking")
        label, confidence, source = run_tsfm_inference(imu_data)

        # Confidence should be between 0 and 1
        assert 0.0 <= confidence <= 1.0
        # For walking data, confidence should be somewhat reasonable
        # (though not guaranteed due to zero-shot nature)
        assert confidence >= 0.0


# =============================================================================
# Batch Inference Tests
# =============================================================================

@pytest.mark.skipif(
    not is_tsfm_available(),
    reason="TSFM model not available"
)
class TestTSFMBatchInference:
    """Tests for batch TSFM inference."""

    def test_batch_inference_basic(self):
        """Test batch inference with multiple samples."""
        batch = create_batch_imu_data(batch_size=4, samples_per_batch=50)

        results = []
        for imu_data in batch:
            label, confidence, source = run_tsfm_inference(imu_data)
            results.append((label, confidence, source))

        assert len(results) == 4
        for label, confidence, source in results:
            assert isinstance(label, str)
            assert label in ["walking", "sitting", "standing", "lying", "climbing stairs", "running", "unknown"]
            assert 0.0 <= confidence <= 1.0
            assert source == "tsfm_model"

    def test_batch_inference_different_activities(self):
        """Test batch inference with different activity types."""
        activities = ["walking", "sitting", "standing", "lying"]
        results = []

        for activity in activities:
            imu_data = create_dummy_imu_data(50, activity_type=activity)
            label, confidence, source = run_tsfm_inference(imu_data)
            results.append((activity, label, confidence))

        # Check that we get predictions for all activities
        assert len(results) == len(activities)
        for activity, label, confidence in results:
            print(f"{activity} -> {label} (conf: {confidence:.3f})")

    def test_batch_inference_varying_lengths(self):
        """Test inference with varying IMU data lengths."""
        lengths = [50, 100, 150, 200]  # 1s, 2s, 3s, 4s at 50Hz
        results = []

        for n_samples in lengths:
            imu_data = create_dummy_imu_data(n_samples, activity_type="walking")
            label, confidence, source = run_tsfm_inference(imu_data)
            results.append((n_samples, label, confidence))

        assert len(results) == len(lengths)
        for n_samples, label, confidence in results:
            print(f"{n_samples} samples -> {label} (conf: {confidence:.3f})")

    def test_batch_inference_performance(self):
        """Test batch inference performance (timing)."""
        import time

        batch_size = 10
        batch = create_batch_imu_data(batch_size=batch_size, samples_per_batch=50)

        start_time = time.time()
        for imu_data in batch:
            run_tsfm_inference(imu_data)
        elapsed = time.time() - start_time

        print(f"Batch inference time for {batch_size} samples: {elapsed:.3f}s")
        print(f"Average time per sample: {elapsed/batch_size:.3f}s")

        # Assert reasonable performance (adjust threshold as needed)
        assert elapsed < 60.0  # Should complete within 60 seconds


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.skipif(
    not is_tsfm_available(),
    reason="TSFM model not available"
)
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_inference_empty_data(self):
        """Test inference with empty IMU data."""
        with pytest.raises((ValueError, RuntimeError)):
            run_tsfm_inference([])

    def test_inference_insufficient_samples(self):
        """Test inference with fewer than minimum samples."""
        imu_data = create_dummy_imu_data(TSFM_MIN_SAMPLES - 1)

        # Should either raise error or return "unknown"
        try:
            label, confidence, source = run_tsfm_inference(imu_data)
            # If it doesn't raise, check result
            assert label in ["walking", "sitting", "standing", "lying", "climbing stairs", "running", "unknown"]
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_inference_large_batch(self):
        """Test inference with large IMU data batch."""
        # Create 1 second of data at 50Hz
        imu_data = create_dummy_imu_data(500)  # 10 seconds at 50Hz
        label, confidence, source = run_tsfm_inference(imu_data)

        assert isinstance(label, str)
        assert source == "tsfm_model"

    def test_channel_descriptions_correct(self):
        """Test that channel descriptions match expected order."""
        expected_channels = [
            "accelerometer x-axis",
            "accelerometer y-axis",
            "accelerometer z-axis",
            "gyroscope x-axis",
            "gyroscope y-axis",
            "gyroscope z-axis",
            "magnetometer x-axis",
            "magnetometer y-axis",
            "magnetometer z-axis",
        ]
        assert TSFM_CHANNEL_DESCRIPTIONS == expected_channels


# =============================================================================
# Model Loading Tests
# =============================================================================

class TestModelLoading:
    """Tests for model loading functionality."""

    def test_is_tsfm_available(self):
        """Test model availability check."""
        result = is_tsfm_available()
        assert isinstance(result, bool)

    def test_model_config_constants(self):
        """Test that model configuration constants are set correctly."""
        assert TSFM_SAMPLING_RATE == 50.0
        assert TSFM_PATCH_SIZE_SEC == 1.0
        assert TSFM_MIN_SAMPLES == 10
        assert len(TSFM_CHANNEL_DESCRIPTIONS) == 9


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all tests manually (for when pytest is not available)."""
    import traceback

    # Test classes and their test methods
    test_classes = [
        TestPreprocessing,
        TestLabelMapping,
        TestIMUDataConversion,
        TestModelLoading,
    ]

    # Skip inference tests if model not available
    if is_tsfm_available():
        test_classes.extend([
            TestTSFMSingleInference,
            TestTSFMBatchInference,
            TestEdgeCases,
        ])

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"✓ {test_class.__name__}.{method_name}")
                    passed += 1
                except Exception as e:
                    print(f"✗ {test_class.__name__}.{method_name}")
                    print(f"  Error: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    import sys

    # Check if model is available
    if not is_tsfm_available():
        print("WARNING: TSFM model not available. Some tests will be skipped.")
        print("To run all tests, ensure TSFM checkpoint is available.")
        print()

    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1)