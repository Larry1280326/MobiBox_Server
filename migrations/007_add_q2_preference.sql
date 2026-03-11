-- Migration: Add q2_preference column to summary_log_feedbacks table
-- Purpose: Store preference categories when Q2 (content preference match) is answered "no"

ALTER TABLE public.summary_log_feedbacks
ADD COLUMN IF NOT EXISTS q2_preference TEXT;

-- Comment describing the column
COMMENT ON COLUMN public.summary_log_feedbacks.q2_preference IS 'Comma-separated preference categories when Q2 is "no". Values: 学习和工作, 运动健康, 手机使用, or custom values from Other field';