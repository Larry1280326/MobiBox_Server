"""Logging configuration with rotational file handlers for MobiBox backend."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Default logs directory
DEFAULT_LOGS_DIR = Path(__file__).parent.parent / "logs"

# Log rotation settings
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
BACKUP_COUNT = 5  # Keep 5 backup files

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    logs_dir: Optional[Path] = None,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
    log_level: int = logging.INFO,
) -> None:
    """
    Set up rotational logging for the application.

    Args:
        logs_dir: Directory to store log files. Defaults to project_root/logs
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        log_level: Logging level (default: INFO)
    """
    if logs_dir is None:
        logs_dir = DEFAULT_LOGS_DIR

    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler (for development)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def get_file_handler(
    log_file: str,
    logs_dir: Optional[Path] = None,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
    log_level: int = logging.INFO,
) -> RotatingFileHandler:
    """
    Create a rotating file handler for a specific log file.

    Args:
        log_file: Name of the log file
        logs_dir: Directory to store log files
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        log_level: Logging level

    Returns:
        Configured RotatingFileHandler
    """
    if logs_dir is None:
        logs_dir = DEFAULT_LOGS_DIR

    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / log_file

    handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)

    return handler


def setup_api_logging(
    logs_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
) -> None:
    """
    Set up logging for the FastAPI application.

    Args:
        logs_dir: Directory to store log files
        log_level: Logging level
    """
    if logs_dir is None:
        logs_dir = DEFAULT_LOGS_DIR

    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger with file handler
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Create rotating file handler
    file_handler = get_file_handler("api.log", logs_dir=logs_dir, log_level=log_level)
    root_logger.addHandler(file_handler)

    # Also configure uvicorn loggers
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        # Add file handler to uvicorn loggers
        uvicorn_handler = get_file_handler(
            "api.log", logs_dir=logs_dir, log_level=log_level
        )
        logger.addHandler(uvicorn_handler)

    # Configure FastAPI/Uvicorn access log
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(log_level)


def setup_celery_logging(
    log_file: str,
    logs_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
) -> None:
    """
    Set up logging for Celery worker/beat.

    Args:
        log_file: Name of the log file
        logs_dir: Directory to store log files
        log_level: Logging level
    """
    if logs_dir is None:
        logs_dir = DEFAULT_LOGS_DIR

    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure celery logger
    celery_logger = logging.getLogger("celery")
    celery_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in celery_logger.handlers[:]:
        celery_logger.removeHandler(handler)

    # Add rotating file handler
    file_handler = get_file_handler(log_file, logs_dir=logs_dir, log_level=log_level)
    celery_logger.addHandler(file_handler)

    # Also configure root logger for any custom services
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Add file handler to root logger
    root_handler = get_file_handler(log_file, logs_dir=logs_dir, log_level=log_level)
    root_logger.addHandler(root_handler)


# Log file names for reference
LOG_FILES = {
    "api": "api.log",
    "celery_worker": "celery_worker.log",
    "celery_beat": "celery_beat.log",
}