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
# TSFM Model Configuration (Time Series Foundation Model)
# Zero-shot activity recognition with 87+ activity labels
# =============================================================================
USE_TSFM_MODEL = False  # Set to False to use legacy IMU transformer
TSFM_MIN_SAMPLES = 10  # Minimum IMU samples required for TSFM inference

# Legacy IMU HAR model (Transformer encoder); set to None to use mock
HAR_IMU_MODEL_CHECKPOINT = "src/celery_app/services/imu_model_utils/ckpts/run_05_06_25_14_16_final_no_cycling_7_class8_25.pth"

# IMU Transformer config (must match trained checkpoint)
HAR_IMU_MODEL_CONFIG = {
    "input_dim": 9,
    "window_size": 50,
    "num_classes": 7,
    "transformer_dim": 64,
    "nhead": 4,
    "dim_feedforward": 128,
    "num_encoder_layers": 6,
    "transformer_dropout": 0.1,
    "transformer_activation": "gelu",
    "encode_position": True,
}

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
        "schedule": 1200.0,  # Every 20 minutes
    },
    "hourly-interventions": {
        "task": "generate_hourly_interventions",
        "schedule": 1200.0,  # Every 20 minutes
    },
    "daily-summary": {
        "task": "generate_daily_summary",
        "schedule": crontab(hour=0, minute=0),  # Midnight
    },
    # Data archival - runs daily at 3 AM (low traffic time)
    "daily-archival": {
        "task": "archive_data_periodic",
        "schedule": crontab(hour=3, minute=0),  # 3 AM daily
    },
}

# =============================================================================
# Summary Generation Thresholds
# =============================================================================

# Minimum data required before generating a summary log
# TEMPORARILY REDUCED FOR TESTING - restore to original values after testing
MIN_ATOMIC_RECORDS_FOR_HOURLY_LOG = 1  # At least 1 atomic records (was: 60)
MIN_UNIQUE_LABELS_FOR_LOG = 1  # At least 1 unique activity type (was: 3)

# Per-user hourly timer settings
# TEMPORARILY REDUCED FOR TESTING - restore to original values after testing
MIN_DATA_COLLECTION_HOURS = 0.05  # ~3 minutes for testing (was: 1 hour)
MIN_HOURS_BETWEEN_SUMMARIES = 0.03  # ~2 minutes for testing (was: 1 hour)