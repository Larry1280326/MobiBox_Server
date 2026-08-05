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

    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "mobibox"
    mongodb_server_selection_timeout_ms: int = 5000
    mongodb_connect_timeout_ms: int = 5000
    mongodb_max_pool_size: int = 20
    mongodb_min_pool_size: int = 0

    # RabbitMQ / Celery
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672//"
    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "rpc://"

    # Baidu Maps API (optional, for location enrichment)
    baidu_maps_api_key: str = ""
    baidu_maps_enabled: bool = False



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
    openrouter_site_url: str = "http://localhost:8001"  # Optional, for rankings
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
