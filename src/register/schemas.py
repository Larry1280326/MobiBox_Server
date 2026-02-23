"""Pydantic schemas for register endpoint."""

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """Request model for user registration."""

    name: str = Field(..., min_length=1, description="Unique user name")
