"""Pydantic schemas for query endpoints."""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


# ============================================================================
# Summary Log Schemas
# ============================================================================


class SummaryLogRequest(BaseModel):
    """Request model for fetching summary logs."""

    user: str = Field(..., min_length=1, description="User identifier")
    log_type: str = Field(
        ...,
        pattern="^(hourly|daily)$",
        description="Type of summary log (hourly or daily)",
    )
    last_log_id: Optional[int] = Field(
        None,
        description="ID of the last log received. If provided and no new log exists, returns None.",
    )


class SummaryLogItem(BaseModel):
    """Single summary log item."""

    id: int = Field(..., description="Database record ID")
    log_content: str = Field(..., description="Summary log text content")
    start_timestamp: Optional[datetime] = Field(None, description="Window start timestamp")
    end_timestamp: Optional[datetime] = Field(None, description="Window end timestamp")
    generation_timestamp: datetime = Field(..., description="Timestamp when the log was generated")


class SummaryLogResponse(BaseModel):
    """Response model for summary logs."""

    status: str = "success"
    data: Optional[SummaryLogItem] = Field(None, description="The most recent summary log")
    has_new_log: bool = Field(
        True,
        description="True if there's a newer log than last_log_id, or if last_log_id wasn't provided",
    )


# ============================================================================
# Intervention Schemas
# ============================================================================


class InterventionRequest(BaseModel):
    """Request model for fetching interventions."""

    user: str = Field(..., min_length=1, description="User identifier")


class InterventionItem(BaseModel):
    """Single intervention item."""

    id: int = Field(..., description="Database record ID")
    intervention_content: str = Field(..., description="Intervention text content")
    start_timestamp: Optional[datetime] = Field(None, description="Window start timestamp")
    end_timestamp: Optional[datetime] = Field(None, description="Window end timestamp")
    generation_timestamp: datetime = Field(..., description="Timestamp when the intervention was generated")


class InterventionResponse(BaseModel):
    """Response model for interventions."""

    status: str = "success"
    data: Optional[InterventionItem] = Field(None, description="The most recent intervention")


# ============================================================================
# Intervention Feedback Schemas
# ============================================================================


class InterventionFeedbackRequest(BaseModel):
    """Request model for submitting intervention feedback."""

    user: str = Field(..., min_length=1, description="User identifier")
    intervention_id: int = Field(..., description="ID of the intervention being rated")
    mc1: Optional[str] = Field(None, description="Multiple choice response 1")
    mc2: Optional[str] = Field(None, description="Multiple choice response 2")
    mc3: Optional[str] = Field(None, description="Multiple choice response 3")
    mc4: Optional[str] = Field(None, description="Multiple choice response 4")
    mc5: Optional[str] = Field(None, description="Multiple choice response 5")
    mc6: Optional[str] = Field(None, description="Multiple choice response 6")
    feedback: str = Field(..., description="Feedback text")


class InterventionFeedbackResponse(BaseModel):
    """Response model for intervention feedback submission."""

    status: str = "success"
    message: str = "Feedback submitted successfully"


# ============================================================================
# Summary Log Feedback Schemas
# ============================================================================


class SummaryLogFeedbackRequest(BaseModel):
    """Request model for submitting summary log feedback."""

    user: int = Field(..., description="User ID")
    summary_logs_id: int = Field(..., description="ID of the summary log being rated")
    feedback: Optional[str] = Field(None, description="Feedback text")


class SummaryLogFeedbackResponse(BaseModel):
    """Response model for summary log feedback submission."""

    status: str = "success"
    message: str = "Feedback submitted successfully"


# ============================================================================
# Atomic Activities Schemas
# ============================================================================


class AtomicActivitiesRequest(BaseModel):
    """Request model for fetching compressed atomic activities."""

    user: str = Field(..., min_length=1, description="User identifier")
    duration: int = Field(
        0,
        ge=0,
        description="Duration in seconds since last fetch (0 for all available data)",
    )


class AtomicActivitiesData(BaseModel):
    """Compressed atomic activities data."""

    sport: List[str] = Field(default_factory=list, description="HAR labels (sport activities)")
    appCategory: List[str] = Field(default_factory=list, description="App category values")
    location: List[str] = Field(default_factory=list, description="Location labels")
    movement: List[str] = Field(default_factory=list, description="Movement labels")
    stepCategory: List[str] = Field(default_factory=list, description="Step count categories")
    phoneCategory: List[str] = Field(default_factory=list, description="Phone usage categories")


class AtomicActivitiesResponse(BaseModel):
    """Response model for compressed atomic activities."""

    status: str = "success"
    data: Optional[AtomicActivitiesData] = Field(None, description="Compressed atomic activities data")