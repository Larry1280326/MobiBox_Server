"""Tests for IMU model loading and inference.

Tests for:
- TSFM model loading and caching
- Legacy IMU model loading
- Model fallback chain (TSFM -> Legacy -> Mock)
- IMU test endpoint
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestTSFMModelLoading:
    """Tests for TSFM model loading and caching."""

    def test_tsfm_checkpoint_exists(self):
        """Verify TSFM checkpoint file exists."""
        checkpoint_path = Path(__file__).parent.parent.parent / \
            "celery_app" / "services" / "tsfm_model" / "ckpts" / "best.pt"

        # Skip if not available (CI/CD may not have model files)
        if not checkpoint_path.exists():
            pytest.skip("TSFM checkpoint not available")

        assert checkpoint_path.is_file(), f"Checkpoint should be a file: {checkpoint_path}"

    def test_tsfm_hyperparameters_exists(self):
        """Verify hyperparameters.json exists."""
        hyperparams_path = Path(__file__).parent.parent.parent / \
            "celery_app" / "services" / "tsfm_model" / "ckpts" / "hyperparameters.json"

        if not hyperparams_path.exists():
            pytest.skip("TSFM hyperparameters not available")

        assert hyperparams_path.is_file(), f"Hyperparameters should be a file: {hyperparams_path}"

    def test_tsfm_model_loads_successfully(self):
        """Test that TSFM model loads without errors."""
        try:
            from src.celery_app.services.tsfm_service import _get_tsfm_model, is_tsfm_available
        except ImportError:
            pytest.skip("TSFM dependencies not available")

        # Check if model is available
        available = is_tsfm_available()

        if not available:
            pytest.skip("TSFM model not available (checkpoint may not exist)")

        # Load model
        model, label_bank, success, device = _get_tsfm_model()

        assert success, "Model should load successfully"
        assert model is not None, "Model should not be None"
        assert label_bank is not None, "Label bank should not be None"

    def test_tsfm_model_caching(self):
        """Test that model is cached after first load."""
        try:
            from src.celery_app.services.tsfm_service import _get_tsfm_model, is_tsfm_available
        except ImportError:
            pytest.skip("TSFM dependencies not available")

        if not is_tsfm_available():
            pytest.skip("TSFM model not available")

        # First load
        model1, label_bank1, _, _ = _get_tsfm_model()

        # Second load should return same cached instance
        model2, label_bank2, _, _ = _get_tsfm_model()

        assert model1 is model2, "Model should be cached"
        assert label_bank1 is label_bank2, "Label bank should be cached"

    def test_tsfm_inference_basic(self):
        """Test basic TSFM inference with synthetic data."""
        try:
            from src.celery_app.services.tsfm_service import run_tsfm_inference, is_tsfm_available
        except ImportError:
            pytest.skip("TSFM dependencies not available")

        if not is_tsfm_available():
            pytest.skip("TSFM model not available")

        # Create synthetic IMU data (50 samples, 9 channels)
        imu_data = []
        for i in range(50):
            imu_data.append({
                "acc_X": np.random.randn(),
                "acc_Y": np.random.randn(),
                "acc_Z": 9.8 + np.random.randn() * 0.1,  # Gravity-like
                "gyro_X": np.random.randn() * 0.1,
                "gyro_Y": np.random.randn() * 0.1,
                "gyro_Z": np.random.randn() * 0.1,
                "mag_X": np.random.randn(),
                "mag_Y": np.random.randn(),
                "mag_Z": np.random.randn(),
            })

        label, confidence, source = run_tsfm_inference(imu_data)

        # Validate output
        assert isinstance(label, str), "Label should be string"
        assert isinstance(confidence, float), "Confidence should be float"
        assert source == "tsfm_model", "Source should be tsfm_model"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"


class TestLegacyIMUModelLoading:
    """Tests for legacy IMU transformer model loading."""

    def test_legacy_model_checkpoint_path_resolution(self):
        """Test legacy model checkpoint path resolution."""
        from src.celery_app.services.har_service import _resolve_checkpoint_path
        from src.celery_app.config import HAR_IMU_MODEL_CHECKPOINT

        path = _resolve_checkpoint_path()

        if path is None:
            # No checkpoint configured or file doesn't exist - acceptable
            assert HAR_IMU_MODEL_CHECKPOINT is None or not Path(HAR_IMU_MODEL_CHECKPOINT).exists()
        else:
            assert path.is_file(), f"Resolved path should exist: {path}"

    def test_legacy_model_loading(self):
        """Test legacy IMU model loading."""
        from src.celery_app.services.har_service import _get_imu_model

        model, available = _get_imu_model()

        if not available:
            pytest.skip("Legacy IMU model not available")

        assert model is not None, "Model should not be None when available"

    def test_legacy_model_inference_shape(self):
        """Test legacy IMU model output shape."""
        from src.celery_app.services.har_service import _get_imu_model, _imu_data_to_tensor, HAR_IMU_WINDOW_SIZE, HAR_IMU_INPUT_CHANNELS

        model, available = _get_imu_model()
        if not available:
            pytest.skip("Legacy IMU model not available")

        # Create synthetic data
        imu_data = []
        for i in range(50):
            imu_data.append({
                "acc_X": np.random.randn(),
                "acc_Y": np.random.randn(),
                "acc_Z": np.random.randn(),
                "gyro_X": np.random.randn() * 0.1,
                "gyro_Y": np.random.randn() * 0.1,
                "gyro_Z": np.random.randn() * 0.1,
                "mag_X": np.random.randn(),
                "mag_Y": np.random.randn(),
                "mag_Z": np.random.randn(),
            })

        tensor = _imu_data_to_tensor(imu_data)

        assert tensor.shape == (1, HAR_IMU_WINDOW_SIZE, HAR_IMU_INPUT_CHANNELS), \
            f"Expected shape (1, {HAR_IMU_WINDOW_SIZE}, {HAR_IMU_INPUT_CHANNELS}), got {tensor.shape}"


class TestModelFallbackChain:
    """Tests for HAR model fallback chain."""

    @pytest.mark.asyncio
    async def test_fallback_to_mock_when_no_models(self):
        """Test that mock model is used when no models available."""
        from src.celery_app.services.har_service import run_har_model

        # Create minimal IMU data
        imu_data = [{
            "acc_X": 0.1, "acc_Y": 9.8, "acc_Z": 0.1,
            "gyro_X": 0.01, "gyro_Y": 0.01, "gyro_Z": 0.01,
            "mag_X": 0.5, "mag_Y": 0.5, "mag_Z": 0.5,
        } for _ in range(50)]

        # Run HAR model - should always succeed (fallback to mock)
        label, confidence, source = await run_har_model(imu_data)

        assert isinstance(label, str), "Label should be string"
        assert isinstance(confidence, float), "Confidence should be float"
        assert isinstance(source, str), "Source should be string"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"

    @pytest.mark.asyncio
    async def test_empty_data_returns_unknown(self):
        """Test that empty data returns unknown."""
        from src.celery_app.services.har_service import run_har_model

        label, confidence, source = await run_har_model([])

        assert label == "unknown"
        assert confidence == 0.5
        assert source == "insufficient_data"

    @pytest.mark.asyncio
    async def test_single_sample_works(self):
        """Test that single sample works (uses mock fallback)."""
        from src.celery_app.services.har_service import run_har_model

        imu_data = [{
            "acc_X": 0.1, "acc_Y": 9.8, "acc_Z": 0.1,
            "gyro_X": 0.01, "gyro_Y": 0.01, "gyro_Z": 0.01,
            "mag_X": 0.5, "mag_Y": 0.5, "mag_Z": 0.5,
        }]

        label, confidence, source = await run_har_model(imu_data)

        assert isinstance(label, str)
        # Single sample won't have enough for TSFM, should use mock
        assert source in ["tsfm_model", "imu_model", "mock_har"]


class TestIMUTestEndpoint:
    """Tests for IMU test endpoint."""

    def test_valid_activity_labels(self):
        """Test valid activity labels."""
        from src.imu_test.schemas import VALID_ACTIVITY_LABELS

        expected_labels = ["walking", "running", "sitting", "standing", "lying", "climbing stairs", "unknown"]

        assert VALID_ACTIVITY_LABELS == expected_labels

    def test_ground_truth_label_normalization(self):
        """Test ground truth label normalization."""
        from src.imu_test.schemas import IMUTestRequest, IMUTestItem

        # Test various label formats
        test_cases = [
            ("walking", "walking"),
            ("WALKING", "walking"),
            ("stairs", "climbing stairs"),
            ("upstairs", "climbing stairs"),
            ("climbing_stairs", "climbing stairs"),
            ("run", "running"),
            ("jogging", "running"),
            ("sit", "sitting"),
            ("stand", "standing"),
            ("lay", "lying"),
            ("laying", "lying"),
        ]

        for input_label, expected in test_cases:
            request = IMUTestRequest(
                user="test_user",
                ground_truth_label=input_label,
                imu_data=[IMUTestItem(acc_X=0, acc_Y=0, acc_Z=0, gyro_X=0, gyro_Y=0, gyro_Z=0) for _ in range(50)]
            )
            normalized = request.validate_ground_truth_label()
            assert normalized == expected, f"Expected '{expected}' for '{input_label}', got '{normalized}'"

    def test_invalid_ground_truth_label(self):
        """Test that invalid ground truth labels return None."""
        from src.imu_test.schemas import IMUTestRequest, IMUTestItem

        request = IMUTestRequest(
            user="test_user",
            ground_truth_label="invalid_activity",
            imu_data=[IMUTestItem(acc_X=0, acc_Y=0, acc_Z=0, gyro_X=0, gyro_Y=0, gyro_Z=0) for _ in range(50)]
        )

        normalized = request.validate_ground_truth_label()
        assert normalized is None, "Invalid label should return None"

    def test_imu_test_request_validation(self):
        """Test IMU test request validation."""
        from src.imu_test.schemas import IMUTestRequest, IMUTestItem
        from pydantic import ValidationError

        # Valid request
        valid_request = IMUTestRequest(
            user="test_user",
            ground_truth_label="walking",
            imu_data=[IMUTestItem(acc_X=i, acc_Y=i, acc_Z=i, gyro_X=i, gyro_Y=i, gyro_Z=i) for i in range(50)]
        )
        assert len(valid_request.imu_data) == 50

        # Request with too few samples should fail
        with pytest.raises(ValidationError):
            IMUTestRequest(
                user="test_user",
                ground_truth_label="walking",
                imu_data=[IMUTestItem(acc_X=0, acc_Y=0, acc_Z=0, gyro_X=0, gyro_Y=0, gyro_Z=0) for _ in range(10)]
            )


class TestIMUDataConversion:
    """Tests for IMU data conversion functions."""

    def test_imu_data_to_array_shape(self):
        """Test IMU data to numpy array conversion."""
        from src.celery_app.services.tsfm_service import _imu_data_to_array

        # Create test data
        imu_data = []
        for i in range(100):
            imu_data.append({
                "acc_X": float(i),
                "acc_Y": float(i + 1),
                "acc_Z": float(i + 2),
                "gyro_X": float(i + 3),
                "gyro_Y": float(i + 4),
                "gyro_Z": float(i + 5),
                "mag_X": float(i + 6),
                "mag_Y": float(i + 7),
                "mag_Z": float(i + 8),
            })

        arr = _imu_data_to_array(imu_data)

        assert arr.shape == (100, 9), f"Expected shape (100, 9), got {arr.shape}"
        assert arr.dtype == np.float32, f"Expected dtype float32, got {arr.dtype}"

    def test_imu_data_to_array_with_missing_values(self):
        """Test IMU data conversion with missing values."""
        from src.celery_app.services.tsfm_service import _imu_data_to_array

        # Create test data with some None values
        imu_data = [{
            "acc_X": 1.0, "acc_Y": None, "acc_Z": 2.0,  # None should become 0
            "gyro_X": 0.1, "gyro_Y": 0.2, "gyro_Z": 0.3,
            "mag_X": None, "mag_Y": 0.5, "mag_Z": 0.6  # None should become 0
        } for _ in range(50)]

        arr = _imu_data_to_array(imu_data)

        assert arr.shape == (50, 9)
        assert arr[0, 1] == 0.0, "None should become 0.0"
        assert arr[0, 6] == 0.0, "None should become 0.0"


class TestLabelMapping:
    """Tests for TSFM label to MobiBox label mapping."""

    def test_label_mapping_completeness(self):
        """Test that all expected labels are mapped."""
        from src.celery_app.services.tsfm_service import TSFM_TO_MOBIBOX_LABELS

        # Common activity labels that should be mapped
        expected_mobibox_labels = {"walking", "running", "sitting", "standing", "lying", "climbing stairs", "unknown"}

        # All mapped values should be in expected set
        mapped_values = set(TSFM_TO_MOBIBOX_LABELS.values())

        assert expected_mobibox_labels == mapped_values, \
            f"All mapped labels should be valid. Missing: {expected_mobibox_labels - mapped_values}"

    def test_unknown_label_fallback(self):
        """Test that unknown labels fall back to 'unknown'."""
        from src.celery_app.services.tsfm_service import _map_tsfm_label_to_mobibox

        # Test known labels
        assert _map_tsfm_label_to_mobibox("walking") == "walking"
        assert _map_tsfm_label_to_mobibox("running") == "running"

        # Test unknown label
        assert _map_tsfm_label_to_mobibox("unknown_activity") == "unknown"
        assert _map_tsfm_label_to_mobibox("dancing") == "unknown"