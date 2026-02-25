"""Celery-specific configuration."""

from celery.schedules import crontab


# Task rate limits
HAR_TASK_RATE_LIMIT = "30/m"  # 30 HAR tasks per minute
ATOMIC_TASK_RATE_LIMIT = "10/m"  # 10 atomic activity tasks per minute

# Processing windows (seconds)
HAR_IMU_WINDOW_SECONDS = 2  # IMU data window for HAR
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

# Beat schedule
CELERY_BEAT_SCHEDULE = {
    "har-periodic": {
        "task": "src.celery_app.tasks.har_tasks.process_har_periodic",
        "schedule": 2.0,  # Every 2 seconds
    },
    "hourly-summary": {
        "task": "src.celery_app.tasks.summary_tasks.generate_hourly_summary",
        "schedule": crontab(minute=0),  # Every hour at minute 0
    },
    "hourly-interventions": {
        "task": "src.celery_app.tasks.summary_tasks.generate_hourly_interventions",
        "schedule": crontab(minute=5),  # 5 minutes after summary
    },
    "daily-summary": {
        "task": "src.celery_app.tasks.summary_tasks.generate_daily_summary",
        "schedule": crontab(hour=0, minute=0),  # Midnight
    },
}