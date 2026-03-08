import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.database import get_supabase_client
from src.logging_config import setup_api_logging
from src.register import router as register_router
from src.upload import router as upload_router
from src.query import router as query_router
from src.imu_test import router as imu_test_router

# Configure rotational logging
setup_api_logging()

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Backend API for MobiBox application",
    version=settings.app_version,
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(register_router)
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(imu_test_router)


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}


@app.get("/supabase-test")
def test_supabase_connection():
    """Test Supabase connection."""
    try:
        client = get_supabase_client()
        # Simple health check - try to access the client
        return {"status": "connected", "url": settings.supabase_url}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)