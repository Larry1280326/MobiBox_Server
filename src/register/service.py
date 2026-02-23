"""Business logic for user registration."""

from src.database import get_supabase_client
from src.register.constants import USER_TABLE
from src.register.schemas import RegisterRequest


def register_user(request: RegisterRequest) -> dict:
    """Insert a new user into the user table. id and timestamp are auto-generated."""
    client = get_supabase_client()
    data = {"name": request.name}
    response = client.table(USER_TABLE).insert(data).execute()
    return {"status": "success", "data": response.data}
