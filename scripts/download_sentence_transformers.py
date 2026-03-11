#!/usr/bin/env python3
"""
Download and cache sentence-transformers models for offline use.

Run this script on the server to download models before running the backend:
    python scripts/download_sentence_transformers.py

Or set environment variables to force offline mode:
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
"""

import os
import sys


def download_models():
    """Download required sentence-transformers models."""
    models = [
        "all-mpnet-base-v2",   # 768-dim, for label bank
        "all-MiniLM-L6-v2",    # 384-dim, for token encoding
    ]

    print("=" * 60)
    print("Downloading sentence-transformers models...")
    print("=" * 60)

    for model_name in models:
        print(f"\nDownloading {model_name}...")
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            # Warm up the model with a test encoding
            model.encode("test")

            # Get cache location
            cache_folder = model._cache_folder
            print(f"  ✓ Downloaded! Cache: {cache_folder}")

            # Verify model works
            embedding = model.encode(["test sentence"])
            print(f"  ✓ Model working! Embedding shape: {embedding.shape}")

        except Exception as e:
            print(f"  ✗ Failed to download {model_name}: {e}")
            print(f"\n  If network is unavailable, copy models from another machine:")
            print(f"  scp -r ~/.cache/huggingface/hub/models--sentence-transformers--{model_name} root@server:~/.cache/huggingface/hub/")
            return False

    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)
    print("\nTo use offline, set these environment variables:")
    print("  export HF_HUB_OFFLINE=1")
    print("  export TRANSFORMERS_OFFLINE=1")

    return True


def verify_models():
    """Verify that all required models are cached and working."""
    models = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]

    print("\nVerifying cached models...")

    for model_name in models:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            model.encode("test")
            print(f"  ✓ {model_name} - OK")
        except Exception as e:
            print(f"  ✗ {model_name} - FAILED: {e}")
            return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download sentence-transformers models")
    parser.add_argument("--verify", action="store_true", help="Verify models are cached")
    args = parser.parse_args()

    if args.verify:
        success = verify_models()
    else:
        success = download_models()

    sys.exit(0 if success else 1)