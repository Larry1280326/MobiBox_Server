-- Migration: Add missing features support
-- Created: 2026-03-07
-- Description: Creates tables for app categories cache and user processing state

-- =============================================================================
-- 1. App categories cache table
-- Stores cached app category classifications to avoid repeated LLM calls
-- =============================================================================

CREATE TABLE IF NOT EXISTS app_categories (
    id SERIAL PRIMARY KEY,
    app_name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    source TEXT DEFAULT 'llm',  -- 'lookup' (predefined) or 'llm' (learned from LLM)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookups by app_name
CREATE INDEX IF NOT EXISTS idx_app_categories_app_name ON app_categories(app_name);

-- Insert common app categories (seed data)
INSERT INTO app_categories (app_name, category, source) VALUES
-- Social communication
('com.whatsapp', 'social communication app', 'lookup'),
('com.facebook.katana', 'social communication app', 'lookup'),
('com.instagram.android', 'social communication app', 'lookup'),
('com.tencent.mm', 'social communication app', 'lookup'),
('com.tencent.mobileqq', 'social communication app', 'lookup'),
('com.ss.android.ugc.aweme', 'social communication app', 'lookup'),
('com.snapchat.android', 'social communication app', 'lookup'),
('com.twitter.android', 'social communication app', 'lookup'),
('org.telegram.messenger', 'social communication app', 'lookup'),
('com.discord', 'social communication app', 'lookup'),
-- Video and Music
('com.youtube.android', 'video and music app', 'lookup'),
('com.netflix.mediaclient', 'video and music app', 'lookup'),
('com.spotify.music', 'video and music app', 'lookup'),
('com.tencent.qqlive', 'video and music app', 'lookup'),
('com.netease.cloudmusic', 'video and music app', 'lookup'),
-- Games
('com.tencent.tmgp.sgame', 'games or gaming platform', 'lookup'),
('com.mojang.minecraftpe', 'games or gaming platform', 'lookup'),
('com.roblox.client', 'games or gaming platform', 'lookup'),
-- Shopping/E-commerce
('com.taobao.taobao', 'e-commerce/shopping platform', 'lookup'),
('com.jingdong.app.mall', 'e-commerce/shopping platform', 'lookup'),
('com.xunmeng.pinduoduo', 'e-commerce/shopping platform', 'lookup'),
('com.amazon.mShop.android.shopping', 'e-commerce/shopping platform', 'lookup'),
-- Office/Working
('com.microsoft.teams', 'office/working app', 'lookup'),
('com.google.android.gm', 'office/working app', 'lookup'),
('com.google.android.calendar', 'office/working app', 'lookup'),
('com.slack', 'office/working app', 'lookup'),
('com.zoom.videomeetings', 'office/working app', 'lookup'),
-- Tools
('com.google.android.apps.maps', 'tool/engineering/functional app', 'lookup'),
('com.autonavi.minimap', 'tool/engineering/functional app', 'lookup'),
('com.android.chrome', 'tool/engineering/functional app', 'lookup')
ON CONFLICT (app_name) DO NOTHING;

-- =============================================================================
-- 2. User processing state table
-- Tracks last processed timestamps to avoid reprocessing and support per-user timers
-- =============================================================================

CREATE TABLE IF NOT EXISTS user_processing_state (
    "user" TEXT PRIMARY KEY,  -- user identifier (quoted because user is reserved word)
    last_har_timestamp TIMESTAMPTZ,          -- Last HAR processing timestamp
    last_atomic_timestamp TIMESTAMPTZ,        -- Last atomic activity processing timestamp
    last_upload_timestamp TIMESTAMPTZ,        -- Last data upload timestamp
    data_collection_start TIMESTAMPTZ,        -- When user started collecting data
    last_summary_generated TIMESTAMPTZ,       -- Last summary log generation timestamp
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- 3. Index for summary_logs (optional - for polling optimization)
-- =============================================================================

-- This index is optional and can be created separately if needed:
-- CREATE INDEX IF NOT EXISTS idx_summary_logs_user_type_timestamp
-- ON summary_logs(user, log_type, timestamp);

-- =============================================================================
-- 4. Comments for documentation
-- =============================================================================

COMMENT ON TABLE app_categories IS 'Cache for app category classifications to reduce LLM calls';
COMMENT ON TABLE user_processing_state IS 'Tracks processing timestamps per user for incremental processing and per-user timing';

COMMENT ON COLUMN user_processing_state.data_collection_start IS 'Timestamp when user started collecting data; used for per-user hourly timer';
COMMENT ON COLUMN user_processing_state.last_summary_generated IS 'Timestamp of last summary log generated; used for minimum time between summaries';