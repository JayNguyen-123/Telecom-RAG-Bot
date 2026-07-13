# Environment validation (Pydantic settings)
import os
from typing import Literal, Optional
from pydantic import Field, SecretStr, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """
    Validates and manages configuration parameters for the Telecom RAG application.
    Properties are loaded from environment variables or a local .env file.
    """
    
    # --- System Core ---
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment target context for structural logging and safety checks."
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    PROJECT_NAME: str = "telecom-rag-bot"
    
    # --- Infrastructure API Keys (Secured via SecretStr to prevent log exposure) ---
    GEMINI_API_KEY: SecretStr = Field(
        ..., 
        description="Google Gemini GenAI authorization key."
    )
    QDRANT_API_KEY: Optional[SecretStr] = Field(
        default=None, 
        description="Authentication key for cloud hosted Qdrant vector cluster instances."
    )
    
    # --- Vector Datastore Configuration ---
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Connection URL for the target Qdrant engine instance."
    )
    QDRANT_COLLECTION_NAME: str = Field(
        default="telecom_knowledge_base"
    )
    VECTOR_DIMENSION: int = Field(
        default=768, 
        description="Dimension size for the models/text-embedding-004 vector structure."
    )
    
    # --- Relational Database Configuration ---
    DB_CONNECTION_STRING: PostgresDsn = Field(
        ...,
        description="Strictly validated connection URI for the operational PostgreSQL cluster."
    )
    DB_POOL_MIN_CONNECTIONS: int = 2
    DB_POOL_MAX_CONNECTIONS: int = 20
    
    # --- Gemini Model & Caching Metrics ---
    LLM_MODEL_NAME: str = "gemini-2.5-flash"
    EMBEDDING_MODEL_NAME: str = "models/text-embedding-004"
    CONTEXT_CACHE_TTL_SECONDS: int = Field(
        default=3600,
        description="Expiration lifespan for server-side Gemini token context pools."
    )
    MIN_TOKENS_FOR_CACHING: int = Field(
        default=32768,
        description="Gemini threshold required to initiate server-side context caching."
    )

    # --- Pydantic Engine Behavior Configuration ---
    model_config = SettingsConfigDict(
        # Standard .env file extraction
        env_file=".env",
        env_file_encoding="utf-8",
        # Allows populating settings from case-insensitive env variables
        case_sensitive=False,
        # Blocks undocumented parameters from infiltrating the application state
        extra="forbid"
    )

# Instantiate a global, read-only configuration singleton
settings = AppSettings()
