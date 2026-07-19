"""MongoDB index creation — run once at application startup.

All create_index calls are idempotent (no-op if index already exists).
TTL indexes handle automatic document expiry for retention policies.
"""

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


async def ensure_indexes(db: AsyncIOMotorDatabase):
    """Create all MongoDB indexes. Idempotent — safe to call on every startup."""
    settings = None  # lazy import to avoid circular import at module level

    # ── users ──────────────────────────────────────────────
    await db["users"].create_index("name", unique=True, name="idx_users_name")

    # ── uploads ────────────────────────────────────────────
    await db["uploads"].create_index(
        [("user", 1), ("timestamp", -1)], name="idx_uploads_user_timestamp"
    )

    # ── imu ────────────────────────────────────────────────
    await db["imu"].create_index(
        [("user", 1), ("timestamp", -1)], name="idx_imu_user_timestamp"
    )
    await db["imu"].create_index(
        "timestamp", expireAfterSeconds=7 * 86400, name="idx_imu_ttl"
    )

    # ── har ────────────────────────────────────────────────
    await db["har"].create_index(
        [("user", 1), ("timestamp", -1)], name="idx_har_user_timestamp"
    )
    await db["har"].create_index("har_label", name="idx_har_label")
    await db["har"].create_index(
        "timestamp", expireAfterSeconds=30 * 86400, name="idx_har_ttl"
    )

    # ── atomic_activities ──────────────────────────────────
    await db["atomic_activities"].create_index(
        [("user", 1), ("timestamp", -1)], name="idx_atomic_user_timestamp"
    )
    await db["atomic_activities"].create_index(
        "timestamp", expireAfterSeconds=30 * 86400, name="idx_atomic_ttl"
    )

    # ── summary_logs ───────────────────────────────────────
    await db["summary_logs"].create_index(
        [("user", 1), ("log_type", 1), ("timestamp", -1)],
        name="idx_summary_user_type_timestamp",
    )
    await db["summary_logs"].create_index(
        "timestamp", expireAfterSeconds=90 * 86400, name="idx_summary_ttl"
    )

    # ── interventions ─────────────────────────────────────
    await db["interventions"].create_index(
        [("user", 1), ("timestamp", -1)], name="idx_interventions_user_timestamp"
    )
    await db["interventions"].create_index(
        "timestamp", expireAfterSeconds=90 * 86400, name="idx_interventions_ttl"
    )

    # ── intervention_feedbacks ─────────────────────────────
    await db["intervention_feedbacks"].create_index(
        "intervention_id", name="idx_feedback_intervention_id"
    )
    await db["intervention_feedbacks"].create_index(
        "user", name="idx_feedback_user"
    )

    # ── summary_log_feedbacks ──────────────────────────────
    await db["summary_log_feedbacks"].create_index(
        "summary_logs_id", name="idx_log_feedback_summary_id"
    )
    await db["summary_log_feedbacks"].create_index(
        "user", name="idx_log_feedback_user"
    )

    # ── app_categories ─────────────────────────────────────
    await db["app_categories"].create_index(
        "app_name", unique=True, name="idx_app_categories_name"
    )

    # ── user_processing_state ─────────────────────────────
    # _id is the user name (natural key, already indexed by default)

    # ── archival_logs ─────────────────────────────────────
    await db["archival_logs"].create_index(
        [("table_name", 1), ("archival_timestamp", -1)],
        name="idx_archival_table_timestamp",
    )

    # ── imu_test_results ──────────────────────────────────
    await db["imu_test_results"].create_index("user", name="idx_imu_test_user")
    await db["imu_test_results"].create_index(
        "timestamp", name="idx_imu_test_timestamp"
    )
    await db["imu_test_results"].create_index(
        "ground_truth_label", name="idx_imu_test_ground_truth"
    )
    await db["imu_test_results"].create_index(
        "is_correct", name="idx_imu_test_is_correct"
    )

    logger.info("MongoDB indexes ensured successfully")
