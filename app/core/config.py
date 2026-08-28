from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    # Qdrant Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "vision_rag_collection"

    # Redis & Celery Task Queue
    REDIS_URL: str = "redis://localhost:6379/0"

    # Embedding & Vision Models
    EMBEDDING_MODEL: str = "sentence-transformers/clip-ViT-B-32"
    VISION_EMBEDDING_MODEL: str = "sentence-transformers/clip-ViT-B-32"

    # LLM Engine Settings
    LLM_PROVIDER: str = "ollama"
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llava"
    LLM_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
