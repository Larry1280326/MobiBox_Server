"""Business logic for upload operations."""

from datetime import datetime

from src.database import get_supabase_client
from src.upload.constants import (
    DOCUMENTS_OPTIONAL_FIELDS,
    IMU_OPTIONAL_FIELDS,
    UPLOADS_TABLE,
    IMU_TABLE,
)
from src.upload.schemas import DocumentUploadRequest, IMUUploadRequest


def _coerce_for_db(value) -> object:
    """Coerce values for DB compatibility.
    - Floats that are whole numbers become int (fixes smallint columns).
    - datetime objects become ISO format strings for PostgreSQL.
    """
    if isinstance(value, float) and value == int(value):
        return int(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _build_data_dict(item: object, optional_fields: list[str]) -> dict:
    """Build data dict from item, including only non-None optional fields."""
    data = {"user": item.user}
    for field in optional_fields:
        value = getattr(item, field)
        if value is not None:
            data[field] = _coerce_for_db(value)
    return data


def upload_documents(request: DocumentUploadRequest) -> dict:
    """Bulk insert document data into the uploads table."""
    client = get_supabase_client()
    rows = [_build_data_dict(item, DOCUMENTS_OPTIONAL_FIELDS) for item in request.items]
    response = client.table(UPLOADS_TABLE).insert(rows).execute()
    return {"status": "success", "inserted": len(rows), "data": response.data}


def upload_imu(request: IMUUploadRequest) -> dict:
    """Bulk insert IMU data into the imu table."""
    client = get_supabase_client()
    rows = [_build_data_dict(item, IMU_OPTIONAL_FIELDS) for item in request.items]
    response = client.table(IMU_TABLE).insert(rows).execute()
    return {"status": "success", "inserted": len(rows), "data": response.data}
