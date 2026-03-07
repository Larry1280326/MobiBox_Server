-- Migration: Database Improvements
-- Created: 2026-03-07
-- Description: Performance optimizations, foreign key constraints, and useful indexes

-- =============================================================================
-- 1. Foreign Key Constraints
-- =============================================================================

-- Add foreign key from user_processing_state to user table
ALTER TABLE public.user_processing_state
ADD CONSTRAINT fk_user_processing_state_user
FOREIGN KEY ("user") REFERENCES public.user(name)
ON DELETE CASCADE;

-- =============================================================================
-- 2. Performance Indexes
-- =============================================================================

-- Atomic activities: Most queries filter by user and timestamp
CREATE INDEX IF NOT EXISTS idx_atomic_activities_user_timestamp
ON public.atomic_activities("user", timestamp DESC);

-- Atomic activities: Filter by activity type
CREATE INDEX IF NOT EXISTS idx_atomic_activities_har_label
ON public.atomic_activities(har_label);

-- HAR labels: Most queries filter by user and timestamp
CREATE INDEX IF NOT EXISTS idx_har_user_timestamp
ON public.har("user", timestamp DESC);

-- HAR labels: Filter by activity type
CREATE INDEX IF NOT EXISTS idx_har_label
ON public.har(har_label);

-- IMU data: Most queries filter by user and timestamp
CREATE INDEX IF NOT EXISTS idx_imu_user_timestamp
ON public.imu("user", timestamp DESC);

-- Summary logs: Query by user and log_type with timestamp ordering
CREATE INDEX IF NOT EXISTS idx_summary_logs_user_type_timestamp
ON public.summary_logs("user", log_type, timestamp DESC);

-- Interventions: Query by user and timestamp
CREATE INDEX IF NOT EXISTS idx_interventions_user_timestamp
ON public.interventions("user", timestamp DESC);

-- Uploads: Query by user and timestamp
CREATE INDEX IF NOT EXISTS idx_uploads_user_timestamp
ON public.uploads("user", timestamp DESC);

-- User processing state: Find users ready for summary generation
CREATE INDEX IF NOT EXISTS idx_user_processing_data_start
ON public.user_processing_state(data_collection_start)
WHERE data_collection_start IS NOT NULL;

-- User processing state: Find users needing summary
CREATE INDEX IF NOT EXISTS idx_user_processing_last_summary
ON public.user_processing_state(last_summary_generated)
WHERE last_summary_generated IS NOT NULL;

-- =============================================================================
-- 3. Additional Columns for Better Tracking
-- =============================================================================

-- Add confidence and source columns to har table if not exists
DO $$
BEGIN
    -- Add confidence column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'har' AND column_name = 'confidence'
    ) THEN
        ALTER TABLE public.har ADD COLUMN confidence REAL DEFAULT 1.0;
    END IF;

    -- Add source column (tsfm_model, imu_model, mock_har)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'har' AND column_name = 'source'
    ) THEN
        ALTER TABLE public.har ADD COLUMN source VARCHAR(50) DEFAULT 'mock_har';
    END IF;
END $$;

-- Add app_name column to atomic_activities for tracking which app was used
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'atomic_activities' AND column_name = 'app_name'
    ) THEN
        ALTER TABLE public.atomic_activities ADD COLUMN app_name VARCHAR(255);
    END IF;
END $$;

-- Add source column to app_categories if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'app_categories' AND column_name = 'source'
    ) THEN
        ALTER TABLE public.app_categories ADD COLUMN source VARCHAR(50) DEFAULT 'llm';
    END IF;
END $$;

-- =============================================================================
-- 4. Updated_at Trigger Function
-- =============================================================================

-- Create a function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to user_processing_state
DROP TRIGGER IF EXISTS update_user_processing_state_updated_at ON public.user_processing_state;
CREATE TRIGGER update_user_processing_state_updated_at
    BEFORE UPDATE ON public.user_processing_state
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 5. Check Constraints for Data Integrity
-- =============================================================================

-- Ensure log_type is valid (hourly or daily)
ALTER TABLE public.summary_logs
DROP CONSTRAINT IF EXISTS check_log_type;

ALTER TABLE public.summary_logs
ADD CONSTRAINT check_log_type
CHECK (log_type IN ('hourly', 'daily'));

-- Ensure source is valid in app_categories
ALTER TABLE public.app_categories
DROP CONSTRAINT IF EXISTS check_app_category_source;

ALTER TABLE public.app_categories
ADD CONSTRAINT check_app_category_source
CHECK (source IN ('lookup', 'llm'));

-- Ensure har source is valid
ALTER TABLE public.har
DROP CONSTRAINT IF EXISTS check_har_source;

ALTER TABLE public.har
ADD CONSTRAINT check_har_source
CHECK (source IN ('tsfm_model', 'imu_model', 'mock_har', 'insufficient_data'));

-- =============================================================================
-- 6. Useful Views for Common Queries
-- =============================================================================

-- View: Recent activity summary for a user
CREATE OR REPLACE VIEW v_user_recent_activity AS
SELECT
    aa.user,
    MAX(aa.timestamp) as last_activity,
    COUNT(*) as activity_count,
    MODE() WITHIN GROUP (ORDER BY aa.har_label) as dominant_activity,
    MODE() WITHIN GROUP (ORDER BY aa.app_category) as dominant_app_category
FROM public.atomic_activities aa
GROUP BY aa.user;

-- View: Users ready for summary generation
CREATE OR REPLACE VIEW v_users_ready_for_summary AS
SELECT
    ups."user",
    ups.data_collection_start,
    ups.last_summary_generated,
    EXTRACT(EPOCH FROM (NOW() - ups.data_collection_start))/3600 as hours_since_start,
    EXTRACT(EPOCH FROM (NOW() - ups.last_summary_generated))/3600 as hours_since_last_summary
FROM public.user_processing_state ups
WHERE ups.data_collection_start IS NOT NULL
  AND EXTRACT(EPOCH FROM (NOW() - ups.data_collection_start))/3600 >= 1
  AND (
    ups.last_summary_generated IS NULL
    OR EXTRACT(EPOCH FROM (NOW() - ups.last_summary_generated))/3600 >= 1
  );

-- =============================================================================
-- 7. Comments for Documentation
-- =============================================================================

COMMENT ON TABLE public.app_categories IS 'Cache for app category classifications. Pre-populated with common apps, learns from LLM for unknown apps.';
COMMENT ON TABLE public.user_processing_state IS 'Tracks processing timestamps per user for incremental processing and per-user hourly timer';
COMMENT ON TABLE public.har IS 'Human Activity Recognition labels with confidence and source tracking';
COMMENT ON TABLE public.atomic_activities IS '7-dimensional atomic activity labels generated from sensor data';

COMMENT ON COLUMN public.har.confidence IS 'Confidence score from HAR model (0.0-1.0)';
COMMENT ON COLUMN public.har.source IS 'Source of HAR label: tsfm_model, imu_model, mock_har, or insufficient_data';
COMMENT ON COLUMN public.atomic_activities.app_name IS 'The specific app that was being used';
COMMENT ON COLUMN public.user_processing_state.data_collection_start IS 'Timestamp when user started collecting data; used for per-user hourly timer';
COMMENT ON COLUMN public.user_processing_state.last_summary_generated IS 'Timestamp of last summary log generated; used for minimum time between summaries';