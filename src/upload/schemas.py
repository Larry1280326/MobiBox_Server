"""Pydantic schemas for upload endpoints."""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    """Single document record for bulk upload."""

    user: str
    timestamp: Optional[datetime] = None
    volume: Optional[float] = None
    screen_on_ratio: Optional[float] = None
    wifi_connected: Optional[bool] = None
    wifi_ssid: Optional[str] = None
    network_traffic: Optional[float] = None
    Rx_traffic: Optional[float] = None
    Tx_traffic: Optional[float] = None
    stepcount_sensor: Optional[int] = None
    gpsLat: Optional[float] = None
    gpsLon: Optional[float] = None
    battery: Optional[int] = None
    current_app: Optional[str] = None
    bluetooth_devices: Optional[List[str]] = None
    address: Optional[str] = None
    poi: Optional[List[str]] = None
    nearbyBluetoothCount: Optional[int] = None
    topBluetoothDevices: Optional[List[str]] = None


class DocumentUploadRequest(BaseModel):
    """Bulk request for document uploads."""

    items: list[DocumentItem] = Field(..., min_length=1, description="List of document records to insert")


class IMUItem(BaseModel):
    """Single IMU record for bulk upload."""

    user: str
    timestamp: Optional[datetime] = None
    acc_X: Optional[float] = None
    acc_Y: Optional[float] = None
    acc_Z: Optional[float] = None
    gyro_X: Optional[float] = None
    gyro_Y: Optional[float] = None
    gyro_Z: Optional[float] = None
    mag_X: Optional[float] = None
    mag_Y: Optional[float] = None
    mag_Z: Optional[float] = None


class IMUUploadRequest(BaseModel):
    """Bulk request for IMU data uploads."""

    items: list[IMUItem] = Field(..., min_length=1, description="List of IMU records to insert")
