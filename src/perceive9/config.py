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

    # Storage paths
    audio_storage_path: str = Field(default="/data/audio")
    chromadb_path: str = Field(default="/data/chromadb")

    # API Keys
    openai_api_key: str = Field(default="")
    pyannote_api_key: str = Field(default="")
    replicate_api_token: str = Field(default="")

    # Default providers
    default_diarization_provider: DiarizationProvider = Field(
        default=DiarizationProvider.PYANNOTE
    )
    default_transcription_provider: TranscriptionProvider = Field(
        default=TranscriptionProvider.OPENAI_WHISPER
    )

    # Default language
    default_language: Language = Field(default=Language.ENGLISH)

    # App settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
