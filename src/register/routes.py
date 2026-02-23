"""API routes for user registration."""

from fastapi import APIRouter, HTTPException

from src.register.schemas import RegisterRequest
from src.register.service import register_user as register_user_service

router = APIRouter(tags=["register"])


@router.post("/register")
def register(request: RegisterRequest):
    """Register a new user. Writes to the user table."""
    try:
        return register_user_service(request)
    except Exception as e:
        error_msg = str(e).lower()
        # Handle unique constraint violation (duplicate name)
        if "unique" in error_msg or "duplicate" in error_msg or "23505" in error_msg:
            raise HTTPException(status_code=409, detail="User name already exists")
        raise HTTPException(status_code=500, detail=str(e))
