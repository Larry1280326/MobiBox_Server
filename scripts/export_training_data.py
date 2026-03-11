#!/usr/bin/env python3
"""Export training data from Supabase for HAR model training.

This script exports IMU data with ground truth labels from the `imu_test_results`
table for training the lightweight HAR model.

Usage:
    # Export all labeled data
    python scripts/export_training_data.py --output data/

    # Export data for specific users
    python scripts/export_training_data.py --output data/ --users user1,user2

    # Export with minimum samples per class
    python scripts/export_training_data.py --output data/ --min-samples-per-class 100

Output files:
    data/train_data.npy      - IMU data (N, 50, 9)
    data/train_labels.npy    - Labels (N,)
    data/val_data.npy        - Validation IMU data
    data/val_labels.npy      - Validation labels
    data/metadata.json       - Dataset statistics and info
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")

# IMU columns in order
IMU_COLUMNS = [
    "acc_X", "acc_Y", "acc_Z",
    "gyro_X", "gyro_Y", "gyro_Z",
    "mag_X", "mag_Y", "mag_Z",
]

# Label to index mapping
LABEL_TO_IDX = {
    "unknown": 0,
    "standing": 1,
    "sitting": 2,
    "lying": 3,
    "walking": 4,
    "climbing stairs": 5,
    "running": 6,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


def get_supabase_client():
    """Get Supabase client."""
    from src.database import get_supabase_client
    return get_supabase_client()


def fetch_imu_test_results(
    client,
    users: Optional[list[str]] = None,
    min_samples_per_class: int = 50,
    days_back: int = 30,
) -> list[dict]:
    """Fetch IMU test results with ground truth labels.

    Args:
        client: Supabase client
        users: Optional list of user IDs to filter
        min_samples_per_class: Minimum samples per class (for quality control)
        days_back: Number of days to look back

    Returns:
        List of test result records
    """
    start_time = datetime.now(CHINA_TZ) - timedelta(days=days_back)

    query = (
        client.table("imu_test_results")
        .select("*")
        .gte("timestamp", start_time.isoformat())
        .order("timestamp", desc=False)
    )

    if users:
        query = query.in_("user", users)

    response = query.execute()

    if not response.data:
        logger.warning("No IMU test results found")
        return []

    logger.info(f"Found {len(response.data)} IMU test results")

    # Filter by minimum samples per class
    label_counts = {}
    for record in response.data:
        label = record.get("ground_truth_label")
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1

    logger.info(f"Label distribution: {label_counts}")

    # Filter out classes with too few samples
    valid_labels = {label for label, count in label_counts.items() if count >= min_samples_per_class}
    if len(valid_labels) < len(label_counts):
        skipped = set(label_counts.keys()) - valid_labels
        logger.warning(f"Skipping classes with < {min_samples_per_class} samples: {skipped}")

    filtered_results = [
        r for r in response.data
        if r.get("ground_truth_label") in valid_labels
    ]

    logger.info(f"Filtered to {len(filtered_results)} results with valid labels")
    return filtered_results


def fetch_imu_data_for_test(
    client,
    test_result: dict,
    window_size: int = 50,
    time_buffer_seconds: float = 1.0,
) -> Optional[np.ndarray]:
    """Fetch IMU data associated with a test result.

    Args:
        client: Supabase client
        test_result: IMU test result record
        window_size: Number of samples to extract
        time_buffer_seconds: Time buffer around test timestamp

    Returns:
        IMU data array of shape (window_size, 9) or None if not enough data
    """
    test_timestamp_str = test_result.get("timestamp")
    if not test_timestamp_str:
        return None

    # Parse timestamp
    if isinstance(test_timestamp_str, str):
        test_timestamp = datetime.fromisoformat(test_timestamp_str.replace("Z", "+00:00"))
    else:
        test_timestamp = test_timestamp_str

    # Query IMU data around the test timestamp
    start_time = test_timestamp - timedelta(seconds=time_buffer_seconds)
    end_time = test_timestamp + timedelta(seconds=time_buffer_seconds)

    user = test_result.get("user")

    response = (
        client.table("imu")
        .select("*")
        .eq("user", user)
        .gte("timestamp", start_time.isoformat())
        .lte("timestamp", end_time.isoformat())
        .order("timestamp", desc=False)
        .limit(window_size * 2)  # Get extra samples for flexibility
        .execute()
    )

    if not response.data or len(response.data) < window_size:
        logger.debug(f"Not enough IMU data for test {test_result.get('id')}: {len(response.data) if response.data else 0} samples")
        return None

    # Extract IMU values
    data = np.zeros((window_size, len(IMU_COLUMNS)), dtype=np.float32)
    for i in range(min(len(response.data), window_size)):
        row = response.data[i]
        for j, col in enumerate(IMU_COLUMNS):
            val = row.get(col)
            data[i, j] = float(val) if val is not None else 0.0

    return data


def export_training_data(
    output_dir: str,
    users: Optional[list[str]] = None,
    min_samples_per_class: int = 50,
    val_split: float = 0.2,
    seed: int = 42,
    days_back: int = 30,
) -> dict:
    """Export training data from Supabase.

    Args:
        output_dir: Directory to save exported data
        users: Optional list of user IDs to filter
        min_samples_per_class: Minimum samples per class
        val_split: Validation split ratio
        seed: Random seed for reproducibility
        days_back: Number of days to look back

    Returns:
        Dictionary with export statistics
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get Supabase client
    logger.info("Connecting to Supabase...")
    client = get_supabase_client()

    # Fetch test results with ground truth
    logger.info("Fetching IMU test results...")
    test_results = fetch_imu_test_results(
        client, users, min_samples_per_class, days_back
    )

    if not test_results:
        logger.error("No valid test results found")
        return {"error": "No valid test results found"}

    # Extract IMU data for each test result
    logger.info("Fetching IMU data for each test result...")
    all_data = []
    all_labels = []
    all_users = []
    skipped = 0

    for i, result in enumerate(test_results):
        if (i + 1) % 10 == 0:
            logger.info(f"Processing {i + 1}/{len(test_results)}...")

        imu_data = fetch_imu_data_for_test(client, result)
        if imu_data is None:
            skipped += 1
            continue

        label = result.get("ground_truth_label")
        label_idx = LABEL_TO_IDX.get(label)
        if label_idx is None:
            logger.warning(f"Unknown label: {label}")
            skipped += 1
            continue

        all_data.append(imu_data)
        all_labels.append(label_idx)
        all_users.append(result.get("user"))

    logger.info(f"Extracted {len(all_data)} samples, skipped {skipped}")

    if len(all_data) == 0:
        logger.error("No valid samples extracted")
        return {"error": "No valid samples extracted"}

    # Convert to numpy arrays
    data_array = np.array(all_data, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    logger.info(f"Data shape: {data_array.shape}")
    logger.info(f"Labels shape: {labels_array.shape}")

    # Shuffle data
    indices = np.random.permutation(len(data_array))
    data_array = data_array[indices]
    labels_array = labels_array[indices]

    # Split into train/val
    val_size = int(len(data_array) * val_split)
    train_data = data_array[:-val_size]
    train_labels = labels_array[:-val_size]
    val_data = data_array[-val_size:]
    val_labels = labels_array[-val_size:]

    logger.info(f"Train: {len(train_data)} samples")
    logger.info(f"Val: {len(val_data)} samples")

    # Save arrays
    np.save(output_path / "train_data.npy", train_data)
    np.save(output_path / "train_labels.npy", train_labels)
    np.save(output_path / "val_data.npy", val_data)
    np.save(output_path / "val_labels.npy", val_labels)

    # Calculate statistics
    train_label_counts = np.bincount(train_labels, minlength=7)
    val_label_counts = np.bincount(val_labels, minlength=7)

    # Metadata
    metadata = {
        "export_timestamp": datetime.now(CHINA_TZ).isoformat(),
        "total_samples": len(data_array),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "data_shape": list(data_array.shape),
        "window_size": 50,
        "num_channels": 9,
        "num_classes": 7,
        "label_names": [IDX_TO_LABEL[i] for i in range(7)],
        "train_label_distribution": {
            IDX_TO_LABEL[i]: int(train_label_counts[i]) for i in range(7)
        },
        "val_label_distribution": {
            IDX_TO_LABEL[i]: int(val_label_counts[i]) for i in range(7)
        },
        "users": list(set(all_users)),
        "num_users": len(set(all_users)),
        "days_back": days_back,
        "seed": seed,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved training data to {output_path}")
    logger.info(f"Train label distribution: {metadata['train_label_distribution']}")
    logger.info(f"Val label distribution: {metadata['val_label_distribution']}")

    return metadata


def export_public_dataset(
    output_dir: str,
    dataset_name: str = "uci_har",
) -> dict:
    """Download and convert a public HAR dataset for pre-training.

    Supported datasets:
    - uci_har: UCI Human Activity Recognition dataset
    - wisdm: WISDM Activity Prediction dataset

    Args:
        output_dir: Directory to save exported data
        dataset_name: Name of the public dataset

    Returns:
        Dictionary with export statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset_name == "uci_har":
        return _download_uci_har(output_path)
    elif dataset_name == "wisdm":
        return _download_wisdm(output_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _download_uci_har(output_path: Path) -> dict:
    """Download and convert UCI HAR dataset.

    Note: This requires manual download from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    Place the extracted files in a 'uci_har' folder and run this script.
    """
    logger.info("UCI HAR dataset requires manual download.")
    logger.info("Download from: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones")
    logger.info("Place extracted files in 'data/uci_har/' directory")

    uci_path = output_path.parent / "uci_har"
    if not uci_path.exists():
        return {"error": "UCI HAR data not found. Please download manually."}

    # Load UCI HAR data
    # This is a placeholder - implement based on actual UCI HAR structure
    logger.warning("UCI HAR loading not yet implemented")
    return {"error": "UCI HAR loading not yet implemented"}


def _download_wisdm(output_path: Path) -> dict:
    """Download and convert WISDM dataset.

    Note: This requires manual download from:
    https://researchdata.ands.org.au/human-activity-recognition-using-smartphone-sensor/1180375
    """
    logger.info("WISDM dataset requires manual download.")
    logger.warning("WISDM loading not yet implemented")
    return {"error": "WISDM loading not yet implemented"}


def main():
    parser = argparse.ArgumentParser(description="Export training data for HAR model")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--users", type=str, default=None, help="Comma-separated user IDs")
    parser.add_argument("--min-samples-per-class", type=int, default=50, help="Minimum samples per class")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--days-back", type=int, default=30, help="Days to look back")
    parser.add_argument("--public-dataset", type=str, default=None, help="Public dataset to download (uci_har, wisdm)")

    args = parser.parse_args()

    # Parse users
    users = None
    if args.users:
        users = [u.strip() for u in args.users.split(",")]

    # Export
    if args.public_dataset:
        metadata = export_public_dataset(args.output, args.public_dataset)
    else:
        metadata = export_training_data(
            output_dir=args.output,
            users=users,
            min_samples_per_class=args.min_samples_per_class,
            val_split=args.val_split,
            seed=args.seed,
            days_back=args.days_back,
        )

    if "error" in metadata:
        logger.error(f"Export failed: {metadata['error']}")
        sys.exit(1)

    logger.info("Export complete!")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()