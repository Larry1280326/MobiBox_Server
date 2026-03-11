#!/usr/bin/env python3
"""Test script for TinierHAR model.

Verifies:
1. Model architecture and parameter count
2. Forward pass with different input shapes
3. Inference pipeline
4. Integration with HAR service
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_model_architecture():
    """Test model architecture and parameter count."""
    print("=" * 60)
    print("Testing Model Architecture")
    print("=" * 60)

    import torch
    from src.celery_app.services.lightweight_har.model import TinierHAR
    from src.celery_app.services.lightweight_har.config import TINIER_HAR_CONFIG, TINIER_HAR_VARIANTS

    # Test default model
    model = TinierHAR()
    total_params = model.count_parameters()
    size_mb = model.model_size_mb()

    print(f"\nDefault TinierHAR Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size (float32): {size_mb:.2f} MB")
    print(f"  Expected: ~90,000 parameters (variant with larger GRU)")

    # Verify parameter count is in expected range
    assert 80_000 < total_params < 100_000, f"Parameter count {total_params} outside expected range"

    # Test different variants
    print(f"\nModel Variants:")
    for variant_name, config in TINIER_HAR_VARIANTS.items():
        from src.celery_app.services.lightweight_har.model import TinierHAR
        variant_model = TinierHAR(config)
        params = variant_model.count_parameters()
        print(f"  {variant_name}: {params:,} parameters ({variant_model.model_size_mb():.2f} MB)")

    # Test architecture breakdown
    print(f"\nArchitecture Breakdown:")
    conv_params = sum(p.numel() for n, p in model.named_parameters() if "stage" in n or "conv" in n)
    gru_params = sum(p.numel() for n, p in model.named_parameters() if "gru" in n)
    attention_params = sum(p.numel() for n, p in model.named_parameters() if "attention" in n)
    classifier_params = sum(p.numel() for n, p in model.named_parameters() if "classifier" in n)

    print(f"  Convolutional layers: {conv_params:,}")
    print(f"  GRU layer: {gru_params:,}")
    print(f"  Attention: {attention_params:,}")
    print(f"  Classifier: {classifier_params:,}")

    return True


def test_forward_pass():
    """Test forward pass with different input shapes."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    import torch
    from src.celery_app.services.lightweight_har.model import TinierHAR

    model = TinierHAR()
    model.eval()

    # Test standard input: (batch, channels, time)
    print("\nTest 1: Standard input (B, C, T)")
    x1 = torch.randn(2, 9, 50)  # Batch=2, 9 channels, 50 timesteps
    with torch.no_grad():
        out1 = model(x1)
    print(f"  Input shape: {x1.shape}")
    print(f"  Output shape: {out1.shape}")
    assert out1.shape == (2, 7), f"Expected (2, 7), got {out1.shape}"

    # Test alternative input: (batch, time, channels)
    print("\nTest 2: Alternative input (B, T, C)")
    x2 = torch.randn(4, 50, 9)  # Batch=4, 50 timesteps, 9 channels
    with torch.no_grad():
        out2 = model(x2)
    print(f"  Input shape: {x2.shape}")
    print(f"  Output shape: {out2.shape}")
    assert out2.shape == (4, 7), f"Expected (4, 7), got {out2.shape}"

    # Test prediction with confidence
    print("\nTest 3: Prediction with confidence")
    pred, conf = model.predict(x1)
    print(f"  Predictions: {pred}")
    print(f"  Confidences: {conf}")
    assert pred.shape == (2,), f"Expected (2,), got {pred.shape}"
    assert conf.shape == (2,), f"Expected (2,), got {conf.shape}"
    assert (conf >= 0).all() and (conf <= 1).all(), "Confidence should be in [0, 1]"

    # Test 6-channel variant
    print("\nTest 4: 6-channel input (acc + gyro only)")
    from src.celery_app.services.lightweight_har.config import TINIER_HAR_VARIANTS
    model_6ch = TinierHAR(TINIER_HAR_VARIANTS["6ch"])
    x3 = torch.randn(2, 6, 50)
    with torch.no_grad():
        out3 = model_6ch(x3)
    print(f"  Input shape: {x3.shape}")
    print(f"  Output shape: {out3.shape}")
    assert out3.shape == (2, 7), f"Expected (2, 7), got {out3.shape}"

    return True


def test_inference_pipeline():
    """Test inference pipeline with mock data."""
    print("\n" + "=" * 60)
    print("Testing Inference Pipeline")
    print("=" * 60)

    from src.celery_app.services.lightweight_har.inference import (
        run_lightweight_har_inference,
        preprocess_imu_data,
        count_model_parameters,
    )
    import json

    # Test parameter counting
    print("\nModel Statistics:")
    stats = count_model_parameters()
    print(json.dumps(stats, indent=2))

    # Create mock IMU data (matching Supabase format)
    print("\nTest Inference with Mock Data:")
    mock_imu = []
    for i in range(50):
        sample = {
            "acc_X": 0.1 + 0.01 * i,
            "acc_Y": 0.2 - 0.01 * i,
            "acc_Z": 9.8,
            "gyro_X": 0.01,
            "gyro_Y": 0.02,
            "gyro_Z": 0.03,
            "mag_X": 50.0,
            "mag_Y": 10.0,
            "mag_Z": -30.0,
        }
        mock_imu.append(sample)

    # Test preprocessing
    print("\n  Preprocessing...")
    tensor = preprocess_imu_data(mock_imu, window_size=50, input_channels=9)
    print(f"  Input tensor shape: {tensor.shape}")
    assert tensor.shape == (1, 9, 50), f"Expected (1, 9, 50), got {tensor.shape}"

    # Test inference
    print("\n  Running inference...")
    label, confidence, source = run_lightweight_har_inference(mock_imu)
    print(f"  Result: label={label}, confidence={confidence:.4f}, source={source}")

    # Verify output
    assert label in ["unknown", "standing", "sitting", "lying", "walking", "climbing stairs", "running"]
    assert 0 <= confidence <= 1
    assert source == "lightweight_har"

    return True


def test_har_service_integration():
    """Test integration with HAR service."""
    print("\n" + "=" * 60)
    print("Testing HAR Service Integration")
    print("=" * 60)

    # Create mock IMU data
    mock_imu = []
    for i in range(50):
        sample = {
            "acc_X": 0.5 + 0.1 * (i % 10),
            "acc_Y": 0.3,
            "acc_Z": 9.8,
            "gyro_X": 0.01 * (i % 5),
            "gyro_Y": 0.02,
            "gyro_Z": 0.03,
            "mag_X": 50.0,
            "mag_Y": 10.0,
            "mag_Z": -30.0,
        }
        mock_imu.append(sample)

    print(f"\n  Mock IMU data: {len(mock_imu)} samples")

    # Test model loading
    print("\n  Loading lightweight HAR model...")
    from src.celery_app.services.lightweight_har.inference import get_lightweight_har_model
    model, available = get_lightweight_har_model()
    print(f"  Model available: {available}")
    if available:
        print(f"  Parameters: {model.count_parameters():,}")

    return True


def compare_models():
    """Compare lightweight HAR with legacy IMU transformer."""
    print("\n" + "=" * 60)
    print("Comparing Models")
    print("=" * 60)

    import torch
    from src.celery_app.services.lightweight_har.model import TinierHAR
    from src.celery_app.services.imu_model_utils.imu_transformer_encoder import IMUTransformerEncoder
    from src.celery_app.config import HAR_IMU_MODEL_CONFIG

    # TinierHAR
    tinier = TinierHAR()
    tinier_params = tinier.count_parameters()

    # Legacy IMU Transformer
    legacy = IMUTransformerEncoder(HAR_IMU_MODEL_CONFIG)
    legacy_params = sum(p.numel() for p in legacy.parameters() if p.requires_grad)

    # TSFM (estimated from config)
    tsfm_params = 21_000_000  # ~21M from documentation

    print(f"\nModel Comparison:")
    print(f"  {'Model':<25} {'Parameters':>15} {'Size (MB)':>12}")
    print(f"  {'-'*25} {'-'*15} {'-'*12}")
    print(f"  {'TinierHAR (new)':<25} {tinier_params:>15,} {tinier.model_size_mb():>12.2f}")
    print(f"  {'Legacy IMU Transformer':<25} {legacy_params:>15,} {legacy_params * 4 / (1024*1024):>12.2f}")
    print(f"  {'TSFM':<25} {tsfm_params:>15,} {tsfm_params * 4 / (1024*1024):>12.2f}")

    # Calculate reduction
    reduction_vs_legacy = legacy_params / tinier_params
    reduction_vs_tsfm = tsfm_params / tinier_params

    print(f"\nEfficiency Gains:")
    print(f"  TinierHAR is {reduction_vs_legacy:.1f}x smaller than Legacy IMU Transformer")
    print(f"  TinierHAR is {reduction_vs_tsfm:.1f}x smaller than TSFM")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TinierHAR Model Test Suite")
    print("=" * 60)

    tests = [
        ("Model Architecture", test_model_architecture),
        ("Forward Pass", test_forward_pass),
        ("Inference Pipeline", test_inference_pipeline),
        ("HAR Service Integration", test_har_service_integration),
        ("Model Comparison", compare_models),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS", None))
            print(f"\n✓ {name}: PASSED")
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            print(f"\n✗ {name}: FAILED - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")

    for name, status, error in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}: {status}")
        if error:
            print(f"      Error: {error}")

    print(f"\nTotal: {passed}/{len(tests)} passed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)