"""PyTorch Dataset for HAR training."""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class IMUDataset(Dataset):
    """Dataset for IMU-based Human Activity Recognition.

    Expects data in format: (samples, timesteps, channels)
    Labels as integer class indices.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.01,
        scale_range: tuple[float, float] = (0.9, 1.1),
    ):
        """Initialize the dataset.

        Args:
            data: IMU data array of shape (samples, timesteps, channels)
            labels: Label array of shape (samples,)
            augment: Whether to apply data augmentation
            noise_std: Standard deviation for Gaussian noise augmentation
            scale_range: Range for random scaling augmentation
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range

        # Z-score normalization per channel
        mean = self.data.mean(dim=(0, 1), keepdim=True)
        std = self.data.std(dim=(0, 1), keepdim=True) + 1e-8
        self.data = (self.data - mean) / std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.labels[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation.

        Techniques:
        1. Gaussian noise injection
        2. Random scaling
        3. Time shifting (optional)
        """
        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Random scaling
        scale = np.random.uniform(*self.scale_range)
        x = x * scale

        return x


class IMUDatasetFromSupabase(Dataset):
    """Dataset that loads IMU data directly from Supabase.

    Loads data from the 'imu' table and labels from the 'har' table.
    """

    def __init__(
        self,
        supabase_client,
        users: Optional[list[str]] = None,
        window_size: int = 50,
        stride: int = 25,
        min_samples: int = 50,
        augment: bool = False,
    ):
        """Initialize dataset from Supabase.

        Args:
            supabase_client: Supabase client instance
            users: List of user IDs to include (None = all users)
            window_size: Number of timesteps per window
            stride: Stride for sliding window
            min_samples: Minimum samples required per window
            augment: Whether to apply data augmentation
        """
        self.window_size = window_size
        self.stride = stride
        self.augment = augment

        # Load data from Supabase
        self.windows, self.labels = self._load_data(
            supabase_client, users, min_samples
        )

        # Normalize
        if len(self.windows) > 0:
            mean = self.windows.mean(axis=(0, 1), keepdims=True)
            std = self.windows.std(axis=(0, 1), keepdims=True) + 1e-8
            self.windows = (self.windows - mean) / std

    def _load_data(
        self,
        supabase_client,
        users: Optional[list[str]],
        min_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load and window IMU data from Supabase."""
        import asyncio
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        CHINA_TZ = ZoneInfo("Asia/Shanghai")

        # Label mapping
        label_to_idx = {
            "unknown": 0,
            "standing": 1,
            "sitting": 2,
            "lying": 3,
            "walking": 4,
            "climbing stairs": 5,
            "running": 6,
        }

        columns = [
            "acc_X", "acc_Y", "acc_Z",
            "gyro_X", "gyro_Y", "gyro_Z",
            "mag_X", "mag_Y", "mag_Z",
        ]

        windows = []
        labels = []

        # Query HAR labels with associated IMU data
        # This is a simplified version - you may need to adjust based on your schema
        end_time = datetime.now(CHINA_TZ)
        start_time = end_time - timedelta(days=30)  # Last 30 days

        query = supabase_client.table("har").select(
            "*, imu!inner(*)"
        ).gte("timestamp", start_time.isoformat()).lte(
            "timestamp", end_time.isoformat()
        )

        if users:
            query = query.in_("user", users)

        response = query.execute()

        if not response.data:
            print("No data found in Supabase")
            return np.array([]), np.array([])

        # Process each HAR label and get associated IMU windows
        for record in response.data:
            label = record.get("har_label", "unknown")
            label_idx = label_to_idx.get(label, 0)

            imu_records = record.get("imu", [])
            if not isinstance(imu_records, list):
                imu_records = [imu_records] if imu_records else []

            if len(imu_records) < min_samples:
                continue

            # Create windows
            for i in range(0, len(imu_records) - self.window_size + 1, self.stride):
                window_data = []
                for j in range(self.window_size):
                    sample = imu_records[i + j]
                    window_data.append([sample.get(col, 0) or 0 for col in columns])

                windows.append(window_data)
                labels.append(label_idx)

        return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation."""
        # Gaussian noise
        noise = torch.randn_like(x) * 0.01
        x = x + noise

        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale

        return x