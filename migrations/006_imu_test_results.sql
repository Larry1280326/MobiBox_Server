-- Migration: Create IMU test results table for evaluating TSFM model accuracy
-- Created for: IMU testing feature branch

-- Create imu_test_results table
CREATE TABLE IF NOT EXISTS imu_test_results (
    id BIGSERIAL PRIMARY KEY,
    "user" TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'tsfm_model',
    ground_truth_label TEXT,
    is_correct BOOLEAN,
    sample_count INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create index on user for faster queries
CREATE INDEX IF NOT EXISTS idx_imu_test_results_user ON imu_test_results("user");

-- Create index on timestamp for time-based queries
CREATE INDEX IF NOT EXISTS idx_imu_test_results_timestamp ON imu_test_results(timestamp);

-- Create index on ground_truth_label for accuracy calculations
CREATE INDEX IF NOT EXISTS idx_imu_test_results_ground_truth ON imu_test_results(ground_truth_label);

-- Create index on is_correct for filtering correct/incorrect predictions
CREATE INDEX IF NOT EXISTS idx_imu_test_results_is_correct ON imu_test_results(is_correct);

-- Add comment to table
COMMENT ON TABLE imu_test_results IS 'Stores IMU test results for evaluating TSFM model accuracy';