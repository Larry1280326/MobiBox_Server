#!/usr/bin/env python3
"""Download and prepare public HAR datasets for TinierHAR training.

Supports:
- UCI-HAR: Human Activity Recognition Using Smartphones
- WISDM: Wireless Sensor Data Mining Activity Recognition

Usage:
    # Download UCI-HAR dataset
    python scripts/download_har_dataset.py --dataset uci_har --output data/

    # Download WISDM dataset
    python scripts/download_har_dataset.py --dataset wisdm --output data/

    # List available datasets
    python scripts/download_har_dataset.py --list
"""

import argparse
import gzip
import logging
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset URLs
DATASETS = {
    "uci_har": {
        "name": "UCI Human Activity Recognition",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
        "mirror_url": "https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip",
        "description": "30 subjects, 6 activities, smartphone accelerometer/gyroscope at 50Hz",
        "activities": [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING",
        ],
        "sensors": ["accelerometer", "gyroscope"],
        "sampling_rate": 50,
        "license": "CC BY 4.0",
    },
    "wisdm": {
        "name": "WISDM Activity Prediction",
        "url": "https://www.cis.fordham.edu/wisdm/dataset.php#activityprediction",
        "direct_url": "https://www.cis.fordham.edu/wisdm/data/WISDM_ar_v1.1_raw.txt",
        "description": "Phone accelerometer, 6 activities, walking/jogging/stairs/sitting/standing",
        "activities": ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"],
        "sensors": ["accelerometer"],
        "sampling_rate": 20,
        "license": "Free for research",
    },
}


def download_file(url: str, output_path: Path, desc: str = None) -> Path:
    """Download a file with progress indication."""
    import urllib.request

    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, int(block_num * block_size * 100 / total_size))
            if block_num % 100 == 0:  # Update every 100 blocks
                logger.info(f"Downloading {desc}: {percent}%")

    logger.info(f"Downloading from {url}")
    urllib.request.urlretrieve(url, output_path, progress_hook)
    logger.info(f"Downloaded to {output_path}")
    return output_path


def download_uci_har(output_dir: Path) -> dict:
    """Download and extract UCI-HAR dataset.

    Args:
        output_dir: Directory to save dataset

    Returns:
        Dictionary with dataset info
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "UCI_HAR_Dataset"

    if dataset_dir.exists():
        logger.info(f"UCI-HAR dataset already exists at {dataset_dir}")
        return {"path": str(dataset_dir), "status": "exists"}

    zip_path = output_dir / "UCI_HAR_Dataset.zip"

    # Try primary URL, fall back to mirror
    urls_to_try = [DATASETS["uci_har"]["url"], DATASETS["uci_har"]["mirror_url"]]

    for url in urls_to_try:
        try:
            download_file(url, zip_path, "UCI-HAR")
            break
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue
    else:
        raise RuntimeError("Failed to download UCI-HAR dataset from all sources")

    # Extract
    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # The zip contains "UCI HAR Dataset" folder
    extracted_dir = output_dir / "UCI HAR Dataset"
    if extracted_dir.exists():
        extracted_dir.rename(dataset_dir)

    # Cleanup
    zip_path.unlink(missing_ok=True)

    logger.info(f"UCI-HAR dataset ready at {dataset_dir}")
    return {"path": str(dataset_dir), "status": "downloaded"}


def download_wisdm(output_dir: Path) -> dict:
    """Download WISDM activity prediction dataset.

    Args:
        output_dir: Directory to save dataset

    Returns:
        Dictionary with dataset info
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / "WISDM_ar_v1.1_raw.txt"

    if data_file.exists():
        logger.info(f"WISDM dataset already exists at {data_file}")
        return {"path": str(data_file), "status": "exists"}

    # Download
    url = DATASETS["wisdm"]["direct_url"]
    download_file(url, data_file, "WISDM")

    logger.info(f"WISDM dataset ready at {data_file}")
    return {"path": str(data_file), "status": "downloaded"}


def convert_uci_har_to_training_format(
    dataset_dir: Path,
    output_dir: Path,
    window_size: int = 50,
    stride: int = 25,
    use_body_acc: bool = True,
) -> dict:
    """Convert UCI-HAR dataset to training format for TinierHAR.

    UCI-HAR provides pre-extracted features (561 features) as well as
    raw inertial signals. For TinierHAR, we use the raw signals.

    The dataset has:
    - train/Inertial Signals/ - Training raw signals
    - test/Inertial Signals/ - Test raw signals
    - train/y_train.txt - Training labels
    - test/y_test.txt - Test labels
    - train/subject_train.txt - Training subjects
    - test/subject_test.txt - Test subjects

    Raw signals (128 timesteps per window at 50Hz = 2.56 seconds):
    - body_acc_x_train.txt, body_acc_y_train.txt, body_acc_z_train.txt
    - body_gyro_x_train.txt, body_gyro_y_train.txt, body_gyro_z_train.txt
    - total_acc_x_train.txt, total_acc_y_train.txt, total_acc_z_train.txt

    Args:
        dataset_dir: Path to UCI_HAR_Dataset
        output_dir: Output directory for numpy files
        window_size: Number of timesteps to use (UCI has 128)
        stride: Stride for sliding window
        use_body_acc: Use body acceleration (True) or total acceleration (False)

    Returns:
        Dictionary with conversion statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Activity mapping: UCI labels (1-6) to our labels (0-6)
    # UCI: 1=WALKING, 2=WALKING_UPSTAIRS, 3=WALKING_DOWNSTAIRS, 4=SITTING, 5=STANDING, 6=LAYING
    # Ours: unknown=0, standing=1, sitting=2, lying=3, walking=4, climbing_stairs=5, running=6
    UCI_TO_OUR_LABELS = {
        1: 4,  # WALKING -> walking
        2: 5,  # WALKING_UPSTAIRS -> climbing_stairs
        3: 5,  # WALKING_DOWNSTAIRS -> climbing_stairs
        4: 2,  # SITTING -> sitting
        5: 1,  # STANDING -> standing
        6: 3,  # LAYING -> lying
    }

    def load_signals(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load raw inertial signals for a split (train or test)."""
        signals_dir = dataset_dir / split / "Inertial Signals"

        # Load accelerometer signals
        if use_body_acc:
            acc_x = np.loadtxt(signals_dir / f"body_acc_x_{split}.txt")
            acc_y = np.loadtxt(signals_dir / f"body_acc_y_{split}.txt")
            acc_z = np.loadtxt(signals_dir / f"body_acc_z_{split}.txt")
        else:
            acc_x = np.loadtxt(signals_dir / f"total_acc_x_{split}.txt")
            acc_y = np.loadtxt(signals_dir / f"total_acc_y_{split}.txt")
            acc_z = np.loadtxt(signals_dir / f"total_acc_z_{split}.txt")

        # Load gyroscope signals
        gyro_x = np.loadtxt(signals_dir / f"body_gyro_x_{split}.txt")
        gyro_y = np.loadtxt(signals_dir / f"body_gyro_y_{split}.txt")
        gyro_z = np.loadtxt(signals_dir / f"body_gyro_z_{split}.txt")

        # Load labels and subjects
        labels = np.loadtxt(dataset_dir / split / f"y_{split}.txt").astype(int)
        subjects = np.loadtxt(dataset_dir / split / f"subject_{split}.txt").astype(int)

        # Stack signals: (N, 128, 6) - acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        signals = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=-1)

        return signals, labels, subjects

    logger.info("Loading UCI-HAR training data...")
    train_signals, train_labels, train_subjects = load_signals("train")
    logger.info(f"Training data shape: {train_signals.shape}")

    logger.info("Loading UCI-HAR test data...")
    test_signals, test_labels, test_subjects = load_signals("test")
    logger.info(f"Test data shape: {test_signals.shape}")

    # Combine train and test for our own split
    all_signals = np.concatenate([train_signals, test_signals], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    all_subjects = np.concatenate([train_subjects, test_subjects], axis=0)

    logger.info(f"Combined data shape: {all_signals.shape}")
    logger.info(f"Original label distribution: {np.bincount(all_labels)[1:]}")

    # Convert labels to our format
    converted_labels = np.array([UCI_TO_OUR_LABELS.get(l, 0) for l in all_labels])
    logger.info(f"Converted label distribution: {np.bincount(converted_labels)}")

    # The UCI data has 128 timesteps per window
    # Our model uses 50 timesteps
    # We can either:
    # 1. Use a subset (e.g., first 50)
    # 2. Resample
    # 3. Sliding window within each 128-sample window

    # Option 1: Use center portion of 128 samples (most stable part)
    start_idx = (128 - window_size) // 2
    end_idx = start_idx + window_size
    all_signals = all_signals[:, start_idx:end_idx, :]
    logger.info(f"Windowed data shape: {all_signals.shape}")

    # Add placeholder for magnetometer (UCI doesn't have magnetometer)
    # We'll use zeros since our model expects 9 channels
    mag_placeholder = np.zeros((all_signals.shape[0], window_size, 3), dtype=np.float32)
    all_signals = np.concatenate([all_signals, mag_placeholder], axis=-1)
    logger.info(f"Final data shape (with magnetometer placeholder): {all_signals.shape}")

    # Split by subjects for cross-subject validation
    unique_subjects = np.unique(all_subjects)
    np.random.seed(42)
    np.random.shuffle(unique_subjects)

    n_val_subjects = int(len(unique_subjects) * 0.2)
    val_subjects = unique_subjects[:n_val_subjects]
    train_subjects = unique_subjects[n_val_subjects:]

    train_mask = np.isin(all_subjects, train_subjects)
    val_mask = np.isin(all_subjects, val_subjects)

    train_data = all_signals[train_mask]
    train_labels_arr = converted_labels[train_mask]
    val_data = all_signals[val_mask]
    val_labels_arr = converted_labels[val_mask]

    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")

    # Save
    np.save(output_dir / "train_data.npy", train_data.astype(np.float32))
    np.save(output_dir / "train_labels.npy", train_labels_arr.astype(np.int64))
    np.save(output_dir / "val_data.npy", val_data.astype(np.float32))
    np.save(output_dir / "val_labels.npy", val_labels_arr.astype(np.int64))

    # Statistics
    import json
    metadata = {
        "source": "UCI_HAR",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "data_shape": list(train_data.shape),
        "window_size": window_size,
        "num_channels": 9,
        "num_classes": 7,
        "label_names": ["unknown", "standing", "sitting", "lying", "walking", "climbing_stairs", "running"],
        "train_label_distribution": {str(i): int(np.sum(train_labels_arr == i)) for i in range(7)},
        "val_label_distribution": {str(i): int(np.sum(val_labels_arr == i)) for i in range(7)},
        "train_subjects": sorted(train_subjects.tolist()),
        "val_subjects": sorted(val_subjects.tolist()),
        "notes": "Magnetometer data is placeholder zeros (UCI-HAR only has acc+gyro)",
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved training data to {output_dir}")
    logger.info(f"Train label distribution: {metadata['train_label_distribution']}")
    logger.info(f"Val label distribution: {metadata['val_label_distribution']}")

    return metadata


def convert_wisdm_to_training_format(
    data_file: Path,
    output_dir: Path,
    window_size: int = 50,
    stride: int = 25,
) -> dict:
    """Convert WISDM dataset to training format for TinierHAR.

    WISDM format: id, activity, timestamp, x, y, z
    - activity: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
    - sampling rate: 20Hz

    Args:
        data_file: Path to WISDM_ar_v1.1_raw.txt
        output_dir: Output directory for numpy files
        window_size: Number of timesteps
        stride: Stride for sliding window

    Returns:
        Dictionary with conversion statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Activity mapping
    WISDM_TO_OUR_LABELS = {
        "Walking": 4,       # walking
        "Jogging": 6,       # running
        "Upstairs": 5,      # climbing_stairs
        "Downstairs": 5,    # climbing_stairs
        "Sitting": 2,       # sitting
        "Standing": 1,      # standing
    }

    logger.info("Loading WISDM data...")

    # WISDM raw data format: id, activity, timestamp, x, y, z
    # Some lines may be malformed
    samples = []
    current_id = None
    current_activity = None
    current_samples = []

    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split(",")
                if len(parts) < 6:
                    continue

                user_id = parts[0]
                activity = parts[1]
                # timestamp = parts[2]
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5].rstrip(";"))  # Some lines end with semicolon

                if user_id != current_id or activity != current_activity:
                    # Save previous segment
                    if current_samples and current_activity in WISDM_TO_OUR_LABELS:
                        samples.append({
                            "user_id": current_id,
                            "activity": current_activity,
                            "label": WISDM_TO_OUR_LABELS[current_activity],
                            "data": np.array(current_samples),
                        })

                    current_id = user_id
                    current_activity = activity
                    current_samples = []

                current_samples.append([x, y, z])

            except Exception as e:
                continue

        # Don't forget the last segment
        if current_samples and current_activity in WISDM_TO_OUR_LABELS:
            samples.append({
                "user_id": current_id,
                "activity": current_activity,
                "label": WISDM_TO_OUR_LABELS[current_activity],
                "data": np.array(current_samples),
            })

    logger.info(f"Loaded {len(samples)} activity segments")

    # Create windows
    windows = []
    labels = []

    for segment in samples:
        data = segment["data"]
        label = segment["label"]

        # Create sliding windows
        for i in range(0, len(data) - window_size + 1, stride):
            window = data[i:i + window_size]
            windows.append(window)
            labels.append(label)

    logger.info(f"Created {len(windows)} windows")

    # Convert to numpy
    all_data = np.array(windows, dtype=np.float32)
    all_labels = np.array(labels, dtype=np.int64)

    logger.info(f"Data shape: {all_data.shape}")
    logger.info(f"Label distribution: {np.bincount(all_labels)}")

    # WISDM has 3 channels (accelerometer only)
    # Add placeholder for gyroscope and magnetometer
    gyro_placeholder = np.zeros((all_data.shape[0], window_size, 3), dtype=np.float32)
    mag_placeholder = np.zeros((all_data.shape[0], window_size, 3), dtype=np.float32)
    all_data = np.concatenate([all_data, gyro_placeholder, mag_placeholder], axis=-1)
    logger.info(f"Final data shape (with gyro/mag placeholder): {all_data.shape}")

    # Split into train/val (80/20)
    n_val = int(len(all_data) * 0.2)
    indices = np.random.permutation(len(all_data))

    train_data = all_data[indices[:-n_val]]
    train_labels_arr = all_labels[indices[:-n_val]]
    val_data = all_data[indices[-n_val:]]
    val_labels_arr = all_labels[indices[-n_val:]]

    # Save
    np.save(output_dir / "train_data.npy", train_data.astype(np.float32))
    np.save(output_dir / "train_labels.npy", train_labels_arr.astype(np.int64))
    np.save(output_dir / "val_data.npy", val_data.astype(np.float32))
    np.save(output_dir / "val_labels.npy", val_labels_arr.astype(np.int64))

    # Statistics
    import json
    metadata = {
        "source": "WISDM",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "data_shape": list(train_data.shape),
        "window_size": window_size,
        "num_channels": 9,
        "num_classes": 7,
        "label_names": ["unknown", "standing", "sitting", "lying", "walking", "climbing_stairs", "running"],
        "train_label_distribution": {str(i): int(np.sum(train_labels_arr == i)) for i in range(7)},
        "val_label_distribution": {str(i): int(np.sum(val_labels_arr == i)) for i in range(7)},
        "notes": "Gyroscope and magnetometer data are placeholder zeros (WISDM only has accelerometer)",
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved training data to {output_dir}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Download HAR datasets for training")
    parser.add_argument("--dataset", type=str, choices=["uci_har", "wisdm", "all"], required=True, help="Dataset to download")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for key, info in DATASETS.items():
            print(f"\n{key}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Activities: {info['activities']}")
            print(f"  Sensors: {info['sensors']}")
            print(f"  License: {info['license']}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "uci_har" or args.dataset == "all":
        logger.info("Downloading UCI-HAR dataset...")
        result = download_uci_har(output_dir)
        if result["status"] in ["downloaded", "exists"]:
            logger.info("Converting UCI-HAR to training format...")
            convert_uci_har_to_training_format(
                Path(result["path"]),
                output_dir / "uci_har_processed",
            )

    if args.dataset == "wisdm" or args.dataset == "all":
        logger.info("Downloading WISDM dataset...")
        result = download_wisdm(output_dir)
        if result["status"] in ["downloaded", "exists"]:
            logger.info("Converting WISDM to training format...")
            convert_wisdm_to_training_format(
                Path(result["path"]),
                output_dir / "wisdm_processed",
            )


if __name__ == "__main__":
    main()