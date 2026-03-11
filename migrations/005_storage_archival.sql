-- Migration: Storage Archival Setup
-- Created: 2026-03-07
-- Description: Sets up storage bucket for data archival and archival tracking
-- Format: Parquet with Snappy compression (~10-100x smaller than CSV)

-- =============================================================================
-- 1. Storage Bucket
-- =============================================================================
-- Note: Storage buckets in Supabase are typically created via the dashboard or API.
-- This migration documents the expected bucket structure.
-- Run the following SQL in Supabase SQL Editor or create via dashboard:
--
-- Create storage bucket for archival:
-- INSERT INTO storage.buckets (id, name, public)
-- VALUES ('mobibox-archive', 'mobibox-archive', false)
-- ON CONFLICT (id) DO NOTHING;

-- =============================================================================
-- 2. Archival Log Table
-- =============================================================================
-- Tracks archival operations for audit and recovery

CREATE TABLE IF NOT EXISTS archival_logs (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    records_archived INTEGER DEFAULT 0,
    records_deleted INTEGER DEFAULT 0,
    storage_path VARCHAR(500),
    file_size_bytes BIGINT,  -- Size of the Parquet file in bytes
    archival_timestamp TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'completed',  -- 'completed', 'failed', 'partial'
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying by table and timestamp
CREATE INDEX IF NOT EXISTS idx_archival_logs_table_timestamp
ON archival_logs(table_name, archival_timestamp DESC);

-- Index for querying by status
CREATE INDEX IF NOT EXISTS idx_archival_logs_status
ON archival_logs(status)
WHERE status != 'completed';

-- =============================================================================
-- 3. Comments for Documentation
-- =============================================================================

COMMENT ON TABLE archival_logs IS 'Log of data archival operations for audit trail. Archives stored in Parquet format with Snappy compression.';
COMMENT ON COLUMN archival_logs.table_name IS 'Name of the table that was archived';
COMMENT ON COLUMN archival_logs.records_archived IS 'Number of records archived to storage';
COMMENT ON COLUMN archival_logs.records_deleted IS 'Number of records deleted from database';
COMMENT ON COLUMN archival_logs.storage_path IS 'Path to the Parquet archive file in storage';
COMMENT ON COLUMN archival_logs.file_size_bytes IS 'Size of the Parquet file in bytes (for storage efficiency tracking)';
COMMENT ON COLUMN archival_logs.status IS 'Status of archival: completed, failed, or partial';

-- =============================================================================
-- 4. Storage Policy (Optional)
-- =============================================================================
-- If using Row Level Security on storage, add policies for service role:
--
-- Allow service role to upload archives:
-- CREATE POLICY "Service role can upload archives"
-- ON storage.objects FOR INSERT
-- TO service_role
-- WITH CHECK (bucket_id = 'mobibox-archive');
--
-- Allow service role to read archives:
-- CREATE POLICY "Service role can read archives"
-- ON storage.objects FOR SELECT
-- TO service_role
-- USING (bucket_id = 'mobibox-archive');

-- =============================================================================
-- 5. Useful Queries for Monitoring
-- =============================================================================

-- View: Archival history by table with storage efficiency
CREATE OR REPLACE VIEW v_archival_history AS
SELECT
    table_name,
    COUNT(*) as total_runs,
    SUM(records_archived) as total_archived,
    SUM(records_deleted) as total_deleted,
    SUM(file_size_bytes) as total_bytes,
    MAX(archival_timestamp) as last_archival,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
    ROUND(SUM(file_size_bytes)::numeric / NULLIF(SUM(records_archived), 0), 2) as avg_bytes_per_record
FROM archival_logs
GROUP BY table_name;

-- View: Recent archival failures
CREATE OR REPLACE VIEW v_recent_archival_failures AS
SELECT
    table_name,
    archival_timestamp,
    records_archived,
    file_size_bytes,
    error_message
FROM archival_logs
WHERE status = 'failed'
ORDER BY archival_timestamp DESC
LIMIT 20;

-- View: Storage efficiency over time
CREATE OR REPLACE VIEW v_archival_storage_efficiency AS
SELECT
    table_name,
    archival_timestamp,
    records_archived,
    file_size_bytes,
    ROUND(file_size_bytes::numeric / NULLIF(records_archived, 0), 2) as bytes_per_record
FROM archival_logs
WHERE status = 'completed' AND records_archived > 0
ORDER BY archival_timestamp DESC;