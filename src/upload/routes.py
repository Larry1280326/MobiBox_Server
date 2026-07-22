"""API routes for upload endpoints."""

import logging

from fastapi import APIRouter

from src.upload.schemas import DocumentUploadRequest, IMUUploadRequest
from src.upload.service import upload_documents as upload_documents_service
from src.upload.service import upload_imu as upload_imu_service
from src.celery_app.tasks.har_tasks import process_har_batch
from src.celery_app.tasks.atomic_tasks import process_atomic_activities_batch

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


def _try_queue_celery(task, users: list[str], task_name: str) -> None:
    """Attempt to queue a Celery task; log warning on failure (non-fatal).

    DB errors propagate to the global exception handler.
    Celery failures are non-fatal — periodic tasks will pick up the data.
    """
    try:
        task.delay(users)
    except Exception as e:
        logger.warning(
            f"Failed to queue Celery task '{task_name}' for users={users}: {e}. "
            f"Will be picked up by periodic task."
        )


@router.post("/documents")
async def upload_documents(request: DocumentUploadRequest):
    """Bulk upload document data to the uploads collection."""
    result = await upload_documents_service(request)
    users = list(set(item.user for item in request.items))
    _try_queue_celery(process_atomic_activities_batch, users, "process_atomic_activities_batch")
    return result


@router.post("/imu")
async def upload_imu(request: IMUUploadRequest):
    """Bulk upload IMU data to the imu collection."""
    result = await upload_imu_service(request)
    users = list(set(item.user for item in request.items))
    _try_queue_celery(process_har_batch, users, "process_har_batch")
    return result
