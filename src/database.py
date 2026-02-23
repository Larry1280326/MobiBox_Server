from supabase import create_client, Client
from src.config import get_settings


def get_supabase_client() -> Client:
    """Create and return a Supabase client instance."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_anon_key)


def get_supabase_admin_client() -> Client:
    """Create and return a Supabase admin client with service role key."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_role_key)