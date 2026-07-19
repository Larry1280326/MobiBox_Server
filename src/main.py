import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.database import check_connection, close_database, get_database
from src.database_indexes import ensure_indexes
from src.logging_config import setup_api_logging
from src.register import router as register_router
from src.upload import router as upload_router
from src.query import router as query_router
from src.imu_test import router as imu_test_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: create indexes and verify MongoDB connection
    try:
        db = await get_database()
        await ensure_indexes(db)
        logging.info("MobiBox API started — MongoDB indexes ensured")
    except Exception as e:
        logging.warning(f"MongoDB not available during startup: {e}. "
                       "Endpoints requiring DB access will fail until MongoDB is running.")
    yield
    # Shutdown: close MongoDB connection
    try:
        await close_database()
    except Exception:
        pass
    logging.info("MobiBox API shut down")


# Configure rotational logging
setup_api_logging()

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Backend API for MobiBox application",
    version=settings.app_version,
    lifespan=lifespan,
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


@app.get("/mongodb-test")
async def test_mongodb_connection():
    """Test MongoDB connection."""
    return await check_connection()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
