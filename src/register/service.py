"""Business logic for user registration."""

from pymongo.errors import DuplicateKeyError

from src.database import get_database
from src.register.constants import USERS_COLLECTION


async def register_user(request) -> dict:
    """Insert a new user into the users collection.

    DuplicateKeyError is re-raised for the global exception handler → 409.
    """
    db = await get_database()
    data = {"_id": request.name, "name": request.name}
    try:
        await db[USERS_COLLECTION].insert_one(data)
        return {"status": "success", "data": [data]}
    except DuplicateKeyError:
        raise
