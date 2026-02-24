"""Pydantic schemas for Atomic Activities processing."""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class AtomicActivity(BaseModel):
    """Atomic activity record for a single timestamp."""

    user: str
    timestamp: datetime
    har_label: Optional[str] = Field(None, description="HAR activity label from LLM")
    app_category: Optional[str] = Field(None, description="App category from LLM")
    step_label: Optional[str] = Field(None, description="Step activity label")
    phone_usage: Optional[str] = Field(None, description="Phone usage label")
    social_label: Optional[str] = Field(None, description="Social context label")
    movement_label: Optional[str] = Field(None, description="Movement pattern label")
    location_label: Optional[str] = Field(None, description="Location context from LLM")


class AtomicActivityResult(BaseModel):
    """Result of atomic activity generation for a user."""

    user: str
    success: bool
    activity: Optional[AtomicActivity] = None
    error: Optional[str] = None


class DocumentWindow(BaseModel):
    """Document data window for processing."""

    user: str
    data: list[dict]
    start_time: datetime
    end_time: datetime