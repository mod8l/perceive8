"""Application configuration from environment variables."""

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Language(str, Enum):
    ENGLISH = "en"
    HEBREW = "he"
    SPANISH = "es"


class DiarizationProvider(str, Enum):
    PYANNOTE = "pyannote"
    REPLICATE = "replicate"


class TranscriptionProvider(str, Enum):
    OPENAI_WHISPER = "openai_whisper"
    REPLICATE = "replicate"


class Settings(BaseSettings):
    # Database
    database_url: str = Field(default="postgresql://localhost:5432/conversation_analyzer")
    database_public_url: str = Field(default="")

    # Storage paths
    audio_storage_path: str = Field(default="./data/audio")
    chromadb_path: str = Field(default="./data/chromadb")

    # ChromaDB remote (Railway). If chromadb_host is set, use HttpClient; else PersistentClient.
    chromadb_host: str = Field(default="")
    chromadb_port: int = Field(default=8000)
    chromadb_token: str = Field(default="")

    # API Keys
    openai_api_key: str = Field(default="")
    pyannote_api_key: str = Field(default="")
    replicate_api_token: str = Field(default="")

    # S3 Storage
    s3_endpoint: str = Field(default="")
    s3_bucket: str = Field(default="")
    s3_access_key: str = Field(default="")
    s3_secret_key: str = Field(default="")
    s3_region: str = Field(default="")

    # Default providers
    default_diarization_provider: DiarizationProvider = Field(
        default=DiarizationProvider.PYANNOTE
    )
    default_transcription_provider: TranscriptionProvider = Field(
        default=TranscriptionProvider.OPENAI_WHISPER
    )

    # Default language
    default_language: Language = Field(default=Language.ENGLISH)

    # RAG / OpenAI model settings
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_chat_model: str = Field(default="gpt-4o-mini")
    rag_top_k: int = Field(default=5)

    # Speaker matching
    speaker_match_threshold: float = Field(default=0.8)

    # Google Drive
    google_api_key: str = Field(default="")
    gdrive_max_file_size_mb: int = Field(default=500)
    gdrive_max_concurrent: int = Field(default=3)

    # App settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
