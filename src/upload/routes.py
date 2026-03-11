"""API routes for upload endpoints."""

from fastapi import APIRouter, HTTPException

from src.upload.schemas import DocumentUploadRequest, IMUUploadRequest
from src.upload.service import upload_documents as upload_documents_service
from src.upload.service import upload_imu as upload_imu_service
from src.celery_app.tasks.har_tasks import process_har_batch
from src.celery_app.tasks.atomic_tasks import process_atomic_activities_batch

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/documents")
def upload_documents(request: DocumentUploadRequest):

    """Bulk upload document data to the uploads table."""
    try:
        print(request.items[0])  # Debugging: 
        result = upload_documents_service(request)

        # Trigger atomic activities processing for affected users
        # Get unique users from the uploaded items
        users = list(set(item.user for item in request.items))

        # Queue the Celery task
        process_atomic_activities_batch.delay(users)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/imu")
def upload_imu(request: IMUUploadRequest):
    """Bulk upload IMU data to the imu table."""
    try:
        print(request.items[:3])  # Debugging: print the incoming request items

        result = upload_imu_service(request)

        # Trigger HAR processing for affected users
        # Get unique users from the uploaded items
        users = list(set(item.user for item in request.items))

        # Queue the Celery task
        process_har_batch.delay(users)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
