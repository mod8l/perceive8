"""Tests for configuration loading."""

import pytest
from unittest.mock import patch
import os

from perceive8.config import (
    Settings,
    Language,
    DiarizationProvider,
    TranscriptionProvider,
    get_settings,
)


class TestLanguageEnum:
    """Tests for Language enum."""

    def test_language_values(self):
        """Test language enum values."""
        assert Language.ENGLISH == "en"
        assert Language.HEBREW == "he"
        assert Language.SPANISH == "es"


class TestProviderEnums:
    """Tests for provider enums."""

    def test_diarization_providers(self):
        """Test diarization provider enum values."""
        assert DiarizationProvider.PYANNOTE == "pyannote"
        assert DiarizationProvider.REPLICATE == "replicate"

    def test_transcription_providers(self):
        """Test transcription provider enum values."""
        assert TranscriptionProvider.OPENAI_WHISPER == "openai_whisper"
        assert TranscriptionProvider.REPLICATE == "replicate"


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Test settings default values are applied."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            assert settings.default_language == Language.ENGLISH
            assert settings.default_diarization_provider == DiarizationProvider.PYANNOTE
            assert settings.default_transcription_provider == TranscriptionProvider.OPENAI_WHISPER
            assert settings.debug is False

    def test_custom_values(self):
        """Test settings with custom values."""
        settings = Settings(
            database_url="postgresql://custom:5432/db",
            openai_api_key="test-key",
            debug=True,
            _env_file=None,
        )
        assert settings.database_url == "postgresql://custom:5432/db"
        assert settings.openai_api_key == "test-key"
        assert settings.debug is True

    def test_environment_override(self):
        """Test environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://env:5432/envdb",
                "DEBUG": "true",
            },
        ):
            settings = Settings(_env_file=None)
            assert settings.database_url == "postgresql://env:5432/envdb"
            assert settings.debug is True


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self):
        """Test that get_settings returns a Settings instance."""
        # Clear the cache first
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
