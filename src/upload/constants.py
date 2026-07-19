"""Constants for upload module."""

# MongoDB collection names
UPLOADS_COLLECTION = "uploads"
IMU_COLLECTION = "imu"

# Optional fields for document uploads (excludes required 'user' field)
DOCUMENTS_OPTIONAL_FIELDS = [
    "timestamp",
    "volume",
    "screen_on_ratio",
    "wifi_connected",
    "wifi_ssid",
    "network_traffic",
    "Rx_traffic",
    "Tx_traffic",
    "stepcount_sensor",
    "gpsLat",
    "gpsLon",
    "battery",
    "current_app",
    "bluetooth_devices",
    "address",
    "poi",
    "nearbyBluetoothCount",
    "topBluetoothDevices",
]

# Optional fields for IMU uploads (excludes required 'user' field)
IMU_OPTIONAL_FIELDS = [
    "timestamp",
    "acc_X",
    "acc_Y",
    "acc_Z",
    "gyro_X",
    "gyro_Y",
    "gyro_Z",
    "mag_X",
    "mag_Y",
    "mag_Z",
]
