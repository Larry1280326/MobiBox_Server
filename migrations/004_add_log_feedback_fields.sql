-- Migration: Add log feedback fields
-- Created: 2026-03-07
-- Description: Add fields for structured log feedback (Q1-Q4, ground truth, suggestions)

-- =============================================================================
-- 1. Add new columns to summary_log_feedbacks table
-- =============================================================================

-- Add Q1-Q4 multiple choice answer columns
ALTER TABLE public.summary_log_feedbacks
ADD COLUMN IF NOT EXISTS q1 TEXT,
ADD COLUMN IF NOT EXISTS q2 TEXT,
ADD COLUMN IF NOT EXISTS q3 TEXT,
ADD COLUMN IF NOT EXISTS q4 TEXT;

-- Add ground truth column (standard answer from user)
ALTER TABLE public.summary_log_feedbacks
ADD COLUMN IF NOT EXISTS ground_truth TEXT;

-- Add suggestions column (optimization suggestions from user)
ALTER TABLE public.summary_log_feedbacks
ADD COLUMN IF NOT EXISTS suggestions TEXT;

-- =============================================================================
-- 2. Ensure user column can store string identifiers
-- =============================================================================

-- Alter user column to TEXT if it was INTEGER (for string user IDs like "samsumg_test")
ALTER TABLE public.summary_log_feedbacks
ALTER COLUMN "user" TYPE TEXT;

-- =============================================================================
-- 3. Add comments for documentation
-- =============================================================================

COMMENT ON COLUMN public.summary_log_feedbacks.feedback IS 'General feedback text (for simple feedback)';
COMMENT ON COLUMN public.summary_log_feedbacks.q1 IS 'Multiple choice answer for question 1';
COMMENT ON COLUMN public.summary_log_feedbacks.q2 IS 'Multiple choice answer for question 2';
COMMENT ON COLUMN public.summary_log_feedbacks.q3 IS 'Multiple choice answer for question 3';
COMMENT ON COLUMN public.summary_log_feedbacks.q4 IS 'Multiple choice answer for question 4';
COMMENT ON COLUMN public.summary_log_feedbacks.ground_truth IS 'Standard answer provided by user';
COMMENT ON COLUMN public.summary_log_feedbacks.suggestions IS 'Optimization suggestions from user';

-- =============================================================================
-- 4. Create index for querying by summary_logs_id
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_summary_log_feedbacks_summary_logs_id
ON public.summary_log_feedbacks(summary_logs_id);

-- =============================================================================
-- 5. Note: feedback column remains as optional for simple feedback use cases
-- =============================================================================

-- The feedback column is already nullable and can be used for simple text feedback
-- while q1-q4, ground_truth, suggestions are for structured feedback