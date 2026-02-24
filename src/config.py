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

    # RabbitMQ / Celery
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672//"
    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "rpc://"


class LLMSettings(BaseSettings):
    """LLM settings for Azure OpenAI integration."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    azure_openai_api_key: str
    azure_openai_endpoint: str = "https://hkust.azure-api.net"
    azure_openai_api_version: str = "2024-10-01-preview"
    azure_openai_deployment: str = "gpt-4o"
    default_temperature: float = 0.1


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