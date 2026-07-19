from torch.utils.data import Dataset
from datetime import datetime

import numpy as np

from src.database import get_sync_database

# Column order for IMU features (must match imu collection)
IMU_COLUMNS = [
    "acc_X", "acc_Y", "acc_Z",
    "gyro_X", "gyro_Y", "gyro_Z",
    "mag_X", "mag_Y", "mag_Z",
]


class IMUDataset(Dataset):
    """
    A PyTorch Dataset for IMU learning tasks. Fetches IMU data from MongoDB.
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
        :param userID: User identifier (imu.user)
        :param start_timestamp: Start of time range (inclusive)
        :param end_timestamp: End of time range (inclusive)
        :param client: Optional PyMongo database instance
        """
        super().__init__()
        if window_shift is None:
            window_shift = window_size

        if client is None:
            db = get_sync_database()
        else:
            db = client

        # Build projection to fetch only IMU columns + sort key
        projection = {col: 1 for col in IMU_COLUMNS}
        projection["timestamp"] = 1

        cursor = db["imu"].find(
            {
                "user": userID,
                "timestamp": {"$gte": start_timestamp, "$lte": end_timestamp},
            },
            projection,
        ).sort("timestamp", 1)

        rows = list(cursor)

        self.data = np.zeros((len(rows), input_size), dtype=np.float32)
        for i, row in enumerate(rows):
            for j, col in enumerate(IMU_COLUMNS[:input_size]):
                val = row.get(col)
                self.data[i, j] = float(val) if val is not None else 0.0

        self.window_size = window_size
        self.window_shift = window_shift
        self.input_size = input_size

    def __len__(self):
        n = len(self.data) - self.window_size
        if n < 0:
            return 0
        return n // self.window_shift + 1

    def __getitem__(self, idx):
        start = idx * self.window_shift
        end = start + self.window_size
        x = self.data[start:end]
        return x
