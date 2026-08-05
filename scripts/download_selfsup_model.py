#!/usr/bin/env python3
"""Download IMU-SelfSupEncoder-v1 model files for offline deployment.

Downloads all required files from HuggingFace to a local directory so they can
be copied to servers without internet access (e.g., in mainland China).

Usage::

    # Step 1: Download on a machine WITH internet access
    python scripts/download_selfsup_model.py

    # Step 2: Copy to remote server
    scp -r models/imu_selfsup user@server:~/MobiBox_Server/models/imu_selfsup

    # Step 3: On the server, verify with offline mode
    python scripts/test_selfsup_model.py --offline
"""

import os
import shutil
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = "NikoKKK/IMU-SelfSupEncoder-v1"
REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "modeling_imu_encoder.py",
]


def download_to_local(target_dir: Path) -> bool:
    """Download all required model files to a local directory."""
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {MODEL_ID}")
    print(f"Target: {target_dir}")
    print()

    ok = True
    for fname in REQUIRED_FILES:
        print(f"Downloading {fname}...")
        try:
            path = hf_hub_download(
                MODEL_ID,
                fname,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {size_mb:.1f} MB → {path}")
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            ok = False

    return ok


def verify_local(target_dir: Path) -> bool:
    """Verify all required files are present."""
    ok = True
    total_mb = 0
    for fname in REQUIRED_FILES:
        path = target_dir / fname
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            total_mb += size_mb
            print(f"  ✓ {fname:35s} {size_mb:.1f} MB")
        else:
            print(f"  ✗ {fname:35s} MISSING")
            ok = False
    print(f"  {'─' * 45}")
    print(f"  Total: {total_mb:.1f} MB")
    return ok


def main():
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / "models" / "imu_selfsup"

    print("=" * 60)
    print("  MobiBox — IMU Model Offline Download")
    print("=" * 60)
    print()

    # Check internet
    try:
        import huggingface_hub
        print(f"  huggingface_hub {huggingface_hub.__version__}")
    except ImportError:
        print("  ERROR: pip install huggingface_hub")
        return 1

    # Download
    if not download_to_local(target_dir):
        print("\n  Some files failed to download. Check your internet connection.")
        return 1

    # Verify
    print()
    print("─" * 60)
    print("Verifying downloaded files:")
    if not verify_local(target_dir):
        return 1

    # Instructions
    print()
    print("=" * 60)
    print("  Next Steps — Deploy to Offline Server")
    print("=" * 60)
    print()
    print(f"  1. Copy to server:")
    print(f"     scp -r {target_dir.as_posix()} user@server:~/MobiBox_Server/models/imu_selfsup")
    print()
    print(f"  2. On the server, verify:")
    print(f"     python scripts/test_selfsup_model.py --offline")
    print()
    print(f"  3. The server's .env should NOT have HF_HUB_OFFLINE set —")
    print(f"     the service auto-detects the local model directory.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
