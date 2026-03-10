from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "MobiBox API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str = ""  # Optional, for admin operations

    # RabbitMQ / Celery
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672//"
    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "rpc://"

    # Baidu Maps API (optional, for location enrichment)
    baidu_maps_api_key: str = ""
    baidu_maps_enabled: bool = False

    # Storage Configuration
    storage_bucket: str = "mobibox-archive"  # Supabase storage bucket name

    # Data Retention Configuration (in days)
    retention_imu_days: int = 7  # IMU data retention
    retention_uploads_days: int = 30  # Uploads retention
    retention_har_days: int = 30  # HAR labels retention
    retention_atomic_days: int = 30  # Atomic activities retention
    retention_summary_logs_days: int = 90  # Summary logs retention
    retention_interventions_days: int = 90  # Interventions retention

    # Archival Configuration
    archive_enabled: bool = True  # Enable/disable archival
    archive_batch_size: int = 10000  # Records per batch


class LLMSettings(BaseSettings):
    """LLM settings for OpenRouter integration."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter API (OpenAI-compatible)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "qwen/qwen3.5-flash-02-23"  # Free model
    openrouter_site_url: str = "http://localhost:8000"  # Optional, for rankings
    openrouter_app_name: str = "MobiBox"  # Optional, for rankings
    default_temperature: float = 0.1

    # Legacy Azure settings (kept for backward compatibility)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = "https://hkust.azure-api.net"
    azure_openai_api_version: str = "2024-10-01-preview"
    azure_openai_deployment: str = "gpt-4o-mini"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Get cached LLM settings instance."""
    return LLMSettings()


# Convenience export for direct access
llm_settings = get_llm_settings()