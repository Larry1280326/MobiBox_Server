"""API routes for upload endpoints."""

from fastapi import APIRouter, HTTPException

from src.upload.schemas import DocumentUploadRequest, IMUUploadRequest
from src.upload.service import upload_documents as upload_documents_service
from src.upload.service import upload_imu as upload_imu_service

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/documents")
def upload_documents(request: DocumentUploadRequest):
    """Bulk upload document data to the uploads table."""
    try:
        return upload_documents_service(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/imu")
def upload_imu(request: IMUUploadRequest):
    """Bulk upload IMU data to the imu table."""
    try:
        return upload_imu_service(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
