"""Pytest fixtures for perceive8 tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from perceive8.main import app
from perceive8.config import Settings
from perceive8.providers.base import (
    DiarizationResult,
    DiarizationSegment,
    TranscriptionResult,
    TranscriptionSegment,
)


@pytest.fixture
def test_settings():
    """Create test settings with mock values."""
    return Settings(
        database_url="postgresql://test:test@localhost:5432/test_db",
        audio_storage_path="/tmp/test_audio",
        chromadb_path="/tmp/test_chromadb",
        openai_api_key="test-openai-key",
        pyannote_api_key="test-pyannote-key",
        replicate_api_token="test-replicate-token",
        debug=True,
    )


@pytest.fixture
async def async_client():
    """Create async test client for FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_diarization_result():
    """Create a mock diarization result."""
    return DiarizationResult(
        segments=[
            DiarizationSegment(
                speaker_label="SPEAKER_00",
                start_time=0.0,
                end_time=5.0,
                confidence=0.95,
            ),
            DiarizationSegment(
                speaker_label="SPEAKER_01",
                start_time=5.0,
                end_time=10.0,
                confidence=0.92,
            ),
        ],
        model_name="test-model",
        raw_response={"test": "data"},
    )


@pytest.fixture
def mock_transcription_result():
    """Create a mock transcription result."""
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start_time=0.0,
                end_time=5.0,
                text="Hello, this is a test.",
                confidence=0.98,
            ),
            TranscriptionSegment(
                start_time=5.0,
                end_time=10.0,
                text="Yes, testing audio processing.",
                confidence=0.95,
            ),
        ],
        full_text="Hello, this is a test. Yes, testing audio processing.",
        model_name="whisper-1",
        language="en",
        raw_response={"test": "data"},
    )


@pytest.fixture
def mock_diarization_provider(mock_diarization_result):
    """Create a mock diarization provider."""
    provider = MagicMock()
    provider.provider_name = "mock_diarization"
    provider.diarize = AsyncMock(return_value=mock_diarization_result)
    provider.get_speaker_embedding = AsyncMock(return_value=[0.1] * 256)
    return provider


@pytest.fixture
def mock_transcription_provider(mock_transcription_result):
    """Create a mock transcription provider."""
    provider = MagicMock()
    provider.provider_name = "mock_transcription"
    provider.transcribe = AsyncMock(return_value=mock_transcription_result)
    return provider
