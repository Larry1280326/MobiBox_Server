-- Migration: Fix log feedback user column and add structured fields
-- Created: 2026-03-07
-- Description: Fix user column type and add fields for structured log feedback (Q1-Q4, ground truth, suggestions)

-- =============================================================================
-- 1. Drop the existing foreign key constraint on user column
-- =============================================================================

ALTER TABLE public.summary_log_feedbacks
DROP CONSTRAINT IF EXISTS summary_log_feedbacks_user_fkey;

-- =============================================================================
-- 2. Change user column from bigint to varchar to match string user IDs
-- =============================================================================

-- First drop the NOT NULL constraint if it exists (user column may be nullable)
ALTER TABLE public.summary_log_feedbacks
ALTER COLUMN "user" DROP NOT NULL;

-- Then change the type to varchar to match user.name
ALTER TABLE public.summary_log_feedbacks
ALTER COLUMN "user" TYPE character varying;

-- =============================================================================
-- 3. Add new foreign key constraint referencing user.name
-- =============================================================================

ALTER TABLE public.summary_log_feedbacks
ADD CONSTRAINT summary_log_feedbacks_user_fkey
FOREIGN KEY ("user") REFERENCES public.user(name);

-- =============================================================================
-- 4. Add new columns for structured feedback
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
-- 5. Add comments for documentation
-- =============================================================================

COMMENT ON COLUMN public.summary_log_feedbacks."user" IS 'User identifier (string, references user.name)';
COMMENT ON COLUMN public.summary_log_feedbacks.feedback IS 'General feedback text (for simple feedback)';
COMMENT ON COLUMN public.summary_log_feedbacks.q1 IS 'Multiple choice answer for question 1';
COMMENT ON COLUMN public.summary_log_feedbacks.q2 IS 'Multiple choice answer for question 2';
COMMENT ON COLUMN public.summary_log_feedbacks.q3 IS 'Multiple choice answer for question 3';
COMMENT ON COLUMN public.summary_log_feedbacks.q4 IS 'Multiple choice answer for question 4';
COMMENT ON COLUMN public.summary_log_feedbacks.ground_truth IS 'Standard answer provided by user';
COMMENT ON COLUMN public.summary_log_feedbacks.suggestions IS 'Optimization suggestions from user';

-- =============================================================================
-- 6. Create index for querying by summary_logs_id
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_summary_log_feedbacks_summary_logs_id
ON public.summary_log_feedbacks(summary_logs_id);