from torch.utils.data import Dataset
from datetime import datetime

import numpy as np

from src.database import get_supabase_client

# Column order for IMU features (must match Supabase imu table)
IMU_COLUMNS = [
    "acc_X", "acc_Y", "acc_Z",
    "gyro_X", "gyro_Y", "gyro_Z",
    "mag_X", "mag_Y", "mag_Z",
]


class IMUDataset(Dataset):
    """
    A PyTorch Dataset for IMU learning tasks. Fetches IMU data from Supabase.
    """

    def __init__(
        self,
        window_size: int,
        input_size: int,
        window_shift: int | None,
        userID: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
        client=None,
    ):
        """
        :param window_size: Number of samples per window
        :param input_size: Number of IMU channels (e.g. 6 or 9)
        :param window_shift: Step between windows; if None, uses window_size (no overlap)
        :param userID: User identifier (Supabase imu.user)
        :param start_timestamp: Start of time range (inclusive)
        :param end_timestamp: End of time range (inclusive)
        :param client: Optional Supabase client (uses get_supabase_client() if None)
        """
        super().__init__()
        if window_shift is None:
            window_shift = window_size

        if client is None:
            client = get_supabase_client()

        start_iso = start_timestamp.isoformat() if isinstance(start_timestamp, datetime) else start_timestamp
        end_iso = end_timestamp.isoformat() if isinstance(end_timestamp, datetime) else end_timestamp

        columns = ",".join(IMU_COLUMNS)
        response = (
            client.table("imu")
            .select(columns)
            .eq("user", userID)
            .gte("timestamp", start_iso)
            .lte("timestamp", end_iso)
            .order("timestamp", desc=False)
            .execute()
        )

        data = response.data if response.data else []
        # Build (N, input_size) array from list of dicts
        cols = IMU_COLUMNS[:input_size]
        n = len(data)
        self.imu = np.zeros((n, input_size), dtype=np.float32)
        for i, row in enumerate(data):
            for j, col in enumerate(cols):
                val = row.get(col)
                self.imu[i, j] = float(val) if val is not None else 0.0

        self.labels = np.zeros((n, 1), dtype=np.int64)  # dummy labels (unlabeled)
        self.start_indices = list(range(0, max(0, n - window_size + 1), window_shift))
        self.window_size = window_size

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_indices = list(range(start_index, start_index + self.window_size))
        imu = self.imu[window_indices, :]
        window_labels = self.labels[window_indices, :]
        label = int(window_labels[0, 0])
        return {"imu": imu, "label": label}
