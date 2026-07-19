"""Business logic for user registration."""

from src.database import get_database
from src.register.constants import USERS_COLLECTION


async def register_user(request) -> dict:
    """Insert a new user into the users collection."""
    db = await get_database()
    data = {"_id": request.name, "name": request.name}
    try:
        await db[USERS_COLLECTION].insert_one(data)
        return {"status": "success", "data": [data]}
    except Exception as e:
        error_msg = str(e).lower()
        if "duplicate" in error_msg or "e11000" in error_msg:
            raise
        raise
