"""Centralized, validated application configuration.

All environment variables used across the project are declared here so that
missing/invalid configuration fails fast at startup with a clear message,
instead of surfacing as a confusing error deep inside AWS or SQLAlchemy calls.
"""
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AWS / Bedrock
    aws_profile: str | None = Field(default=None, alias="AWS_PROFILE")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    bedrock_model_id: str = Field(
        default="us.anthropic.claude-haiku-4-5-20251001-v1:0", alias="BEDROCK_MODEL_ID"
    )
    embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0", alias="EMBEDDING_MODEL_ID"
    )
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")

    # Chunking
    chunk_size: int = Field(default=2000, alias="CHUNK_SIZE", gt=0)
    chunk_overlap: int = Field(default=300, alias="CHUNK_OVERLAP", ge=0)

    # Postgres / pgvector
    postgres_user: str = Field(default="rag_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="rag_password", alias="POSTGRES_PASSWORD")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="rag_db", alias="POSTGRES_DB")

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        """Ensure the configured overlap can produce forward progress."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def postgres_connection_string(self) -> str:
        """Build the psycopg-compatible SQLAlchemy connection string.

        Returns:
            str: The connection string for the configured Postgres database.

        """
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached, process-wide application settings.

    Returns:
        Settings: The validated application settings.

    """
    return Settings()
