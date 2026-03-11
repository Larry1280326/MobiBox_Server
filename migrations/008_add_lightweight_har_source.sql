-- Migration: Add lightweight_har to har source constraint
-- Description: Update check_har_source constraint to include 'lightweight_har' source

-- =============================================================================
-- Update har source constraint
-- =============================================================================

-- Drop the existing constraint
ALTER TABLE public.har
DROP CONSTRAINT IF EXISTS check_har_source;

-- Add updated constraint with lightweight_har source
ALTER TABLE public.har
ADD CONSTRAINT check_har_source
CHECK (source IN ('tsfm_model', 'imu_model', 'lightweight_har', 'mock_har', 'insufficient_data'));

-- =============================================================================
-- Add comment for documentation
-- =============================================================================

COMMENT ON CONSTRAINT check_har_source ON public.har IS
'Valid sources for HAR labels: tsfm_model (primary), lightweight_har (efficient fallback), imu_model (legacy), mock_har (testing), insufficient_data (not enough samples)';