"""Celery-specific configuration."""

from celery.schedules import crontab


# Task rate limits
HAR_TASK_RATE_LIMIT = "30/m"  # 30 HAR tasks per minute
ATOMIC_TASK_RATE_LIMIT = "10/m"  # 10 atomic activity tasks per minute

# Processing windows (seconds)
HAR_IMU_WINDOW_SECONDS = 1  # IMU data window for HAR (1s window)
HAR_DATA_DELAY_SECONDS = 126  # Delay to wait for batch IMU data upload (126s = 2min - 6s buffer)
HAR_IMU_WINDOW_SIZE = 50  # Samples per window (1s @ 50Hz, must match model)
HAR_IMU_INPUT_CHANNELS = 9  # acc_X/Y/Z, gyro_X/Y/Z, mag_X/Y/Z (must match checkpoint)
ATOMIC_HAR_WINDOW_SECONDS = 2  # Window for HAR-based atomic activity
ATOMIC_APP_WINDOW_SECONDS = 10  # Window for app category
ATOMIC_STEP_WINDOW_SECONDS = 10  # Window for step label
ATOMIC_PHONE_WINDOW_SECONDS = 10  # Window for phone usage
ATOMIC_SOCIAL_WINDOW_SECONDS = 10  # Window for social label
ATOMIC_MOVEMENT_WINDOW_SECONDS = 120  # Window for movement label (2 min)
ATOMIC_LOCATION_WINDOW_SECONDS = 120  # Window for location label (2 min)

# Debounce settings
HAR_DEBOUNCE_SECONDS = 2  # Minimum time between HAR processing per user
ATOMIC_DEBOUNCE_SECONDS = 5  # Minimum time between atomic processing per user

# =============================================================================
# SelfSupEncoder Model Configuration (IMU-SelfSupEncoder-v1)
# Lightweight (~1.6M params) self-supervised ViT for IMU HAR.
# Downloads from HuggingFace on first use (~5 MB), or load from models/imu_selfsup/
# =============================================================================
USE_SELFSUP_MODEL = True
SELFSUP_MIN_SAMPLES = 50  # Minimum IMU samples for SelfSupEncoder (1s @ 50Hz)

# Beat schedule
CELERY_BEAT_SCHEDULE = {
    "har-periodic": {
        "task": "src.celery_app.tasks.har_tasks.process_har_periodic",  # Must match the task name in @celery_app.task decorator
        "schedule": 2.0,  # Every 2 seconds
    },
    "atomic-periodic": {
        "task": "src.celery_app.tasks.atomic_tasks.process_atomic_periodic",
        "schedule": 10.0,  # Every 10 seconds
    },
    # Summary and intervention generation every 20 minutes
    "hourly-summary": {
        "task": "generate_hourly_summary",
        "schedule": 60.0,  # Every 1 minute
    },
    "hourly-interventions": {
        "task": "generate_hourly_interventions",
        "schedule": 60.0,  # Every 1 minute
    },
    "daily-summary": {
        "task": "generate_daily_summary",
        "schedule": crontab(hour=0, minute=0),  # Midnight
    },
}

# =============================================================================
# Summary Generation Thresholds
# =============================================================================

# Minimum data required before generating a summary log
MIN_ATOMIC_RECORDS_FOR_HOURLY_LOG = 60  # At least 60 atomic records
MIN_UNIQUE_LABELS_FOR_LOG = 3  # At least 3 unique activity types

# Per-user hourly timer settings
MIN_DATA_COLLECTION_HOURS = 1.0  # 1 hour of data collection minimum
MIN_HOURS_BETWEEN_SUMMARIES = 1.0  # 1 hour between summaries