"""Business logic for upload operations."""

from datetime import datetime

from src.database import get_database
from src.upload.constants import (
    DOCUMENTS_OPTIONAL_FIELDS,
    IMU_OPTIONAL_FIELDS,
    UPLOADS_COLLECTION,
    IMU_COLLECTION,
)


def _coerce_for_db(value) -> object:
    """Coerce values for MongoDB compatibility.
    - Floats that are whole numbers become int
    - datetime objects remain as datetime (MongoDB handles them natively)
    """
    if isinstance(value, float) and value == int(value):
        return int(value)
    return value


def _build_data_dict(item: object, optional_fields: list[str]) -> dict:
    """Build data dict from item, including only non-None optional fields."""
    data = {"user": item.user}
    for field in optional_fields:
        value = getattr(item, field)
        if value is not None:
            data[field] = _coerce_for_db(value)
    return data


async def upload_documents(request) -> dict:
    """Bulk insert document data into the uploads collection."""
    db = await get_database()
    rows = [_build_data_dict(item, DOCUMENTS_OPTIONAL_FIELDS) for item in request.items]
    result = await db[UPLOADS_COLLECTION].insert_many(rows)
    return {"status": "success", "inserted": len(result.inserted_ids)}


async def upload_imu(request) -> dict:
    """Bulk insert IMU data into the imu collection.

    This function is optimized for large payloads by inserting records in chunks
    instead of building a single huge list of rows in memory.
    """
    db = await get_database()

    total_items = len(request.items)
    chunk_size = 1_000
    inserted_count = 0

    for start in range(0, total_items, chunk_size):
        end = start + chunk_size
        chunk = request.items[start:end]
        rows = [_build_data_dict(item, IMU_OPTIONAL_FIELDS) for item in chunk]
        result = await db[IMU_COLLECTION].insert_many(rows)
        inserted_count += len(result.inserted_ids)

    return {
        "status": "success",
        "inserted": inserted_count,
    }
