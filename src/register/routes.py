"""API routes for user registration."""

from fastapi import APIRouter

from src.register.schemas import RegisterRequest
from src.register.service import register_user as register_user_service

router = APIRouter(tags=["register"])


@router.post("/register")
async def register(request: RegisterRequest):
    """Register a new user. Writes to the users collection.

    DuplicateKeyError is caught by the global exception handler → 409.
    """
    return await register_user_service(request)
