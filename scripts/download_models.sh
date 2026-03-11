#!/bin/bash
# Pre-download sentence-transformers models for offline use
# Run this on a machine with internet access, then copy the cache to your server

set -e

echo "Pre-downloading sentence-transformers models..."

python3 << 'EOF'
import os
from sentence_transformers import SentenceTransformer

# Models required by TSFM
models = [
    'all-MiniLM-L6-v2',      # 384-dim, used for token encoding
    'all-mpnet-base-v2',     # 768-dim, used for label bank
]

cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME',
                            os.path.expanduser('~/.cache/torch/sentence_transformers'))

print(f"Cache directory: {cache_dir}")

for model_name in models:
    print(f"\nDownloading {model_name}...")
    model = SentenceTransformer(model_name)
    # Encode a test string to ensure model is fully loaded
    embedding = model.encode("test")
    print(f"  - Downloaded! Embedding shape: {embedding.shape}")
    del model

print(f"\n\nAll models downloaded to: {cache_dir}")
print("Copy this directory to your server and set:")
print(f"  export SENTENCE_TRANSFORMERS_HOME={cache_dir}")
EOF

echo "Done!"