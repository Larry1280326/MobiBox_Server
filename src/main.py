import logging
from contextlib import asynccontextmanager

import pymongo.errors as mongo_err
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.database import check_connection, close_database, get_database
from src.database_indexes import ensure_indexes
from src.logging_config import setup_api_logging
from src.register import router as register_router
from src.upload import router as upload_router
from src.query import router as query_router
from src.imu_test import router as imu_test_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: create indexes and verify MongoDB connection
    try:
        db = await get_database()
        await ensure_indexes(db)
        logger.info("MobiBox API started — MongoDB indexes ensured")
    except Exception as e:
        logger.warning(f"MongoDB not available during startup: {e}. "
                       "Endpoints requiring DB access will fail until MongoDB is running.")

    # Startup: Celery broker probe (best-effort)
    try:
        from src.celery_app.celery_app import celery_app
        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, interval_start=0)
        conn.release()
        logger.info("Celery broker connection verified")
    except Exception as e:
        logger.warning(f"Celery broker not available during startup: {e}. "
                       "Celery tasks will not be queued until RabbitMQ is running.")

    yield
    # Shutdown: close MongoDB connection
    try:
        await close_database()
    except Exception:
        pass
    logger.info("MobiBox API shut down")


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

# Global exception handlers — log tracebacks, return safe responses
@app.exception_handler(mongo_err.DuplicateKeyError)
async def duplicate_key_handler(request, exc):
    logger.warning(f"Duplicate key violation: {exc}")
    return JSONResponse(status_code=409, content={"detail": "Resource already exists"})


@app.exception_handler(mongo_err.ConnectionFailure)
async def mongo_connection_handler(request, exc):
    logger.error(f"MongoDB connection failure: {exc}")
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


@app.exception_handler(mongo_err.ServerSelectionTimeoutError)
async def mongo_timeout_handler(request, exc):
    logger.error(f"MongoDB server selection timeout: {exc}")
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Include routers
app.include_router(register_router)
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(imu_test_router)


@app.get("/health")
async def health_check():
    """Health check — verifies MongoDB connectivity."""
    health = {"status": "healthy", "mongodb": "up"}
    try:
        db = await get_database()
        await db.command("ping")
    except Exception as e:
        logger.error(f"Health check: MongoDB ping failed: {e}")
        health["status"] = "unhealthy"
        health["mongodb"] = "down"
        return JSONResponse(status_code=503, content=health)
    return health


@app.get("/mongodb-test")
async def test_mongodb_connection():
    """Test MongoDB connection."""
    return await check_connection()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
