"""Integration tests for real provider API calls.

These tests call actual third-party services and require API keys.
Skip by default - run with: make test-integration
"""

import os
from pathlib import Path

import pytest

from perceive8.config import Language
from perceive8.providers.openai_whisper import OpenAIWhisperProvider
from perceive8.providers.pyannote import PyannoteProvider

# Skip conditions
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")
SAMPLE_AUDIO = Path(__file__).parent / "fixtures" / "sample.wav"

skip_no_openai = pytest.mark.skipif(
    not OPENAI_API_KEY, reason="OPENAI_API_KEY not set"
)
skip_no_pyannote = pytest.mark.skipif(
    not PYANNOTE_API_KEY, reason="PYANNOTE_API_KEY not set"
)
skip_no_audio = pytest.mark.skipif(
    not SAMPLE_AUDIO.exists(), reason="tests/fixtures/sample.wav not found"
)


@pytest.mark.integration
class TestOpenAIWhisperIntegration:
    """Integration tests for OpenAI Whisper transcription."""

    @skip_no_openai
    @skip_no_audio
    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self):
        """Test that transcription returns text."""
        provider = OpenAIWhisperProvider(api_key=OPENAI_API_KEY)
        result = await provider.transcribe(
            audio_path=str(SAMPLE_AUDIO),
            language=Language.ENGLISH,
        )
        assert result.full_text
        assert len(result.segments) > 0
        assert result.model_name == "whisper-1"

    @skip_no_openai
    @skip_no_audio
    @pytest.mark.asyncio
    async def test_transcribe_with_word_timestamps(self):
        """Test that word timestamps are returned."""
        provider = OpenAIWhisperProvider(api_key=OPENAI_API_KEY)
        result = await provider.transcribe(
            audio_path=str(SAMPLE_AUDIO),
            language=Language.ENGLISH,
        )
        # Check first segment has words
        if result.segments and result.segments[0].words:
            word = result.segments[0].words[0]
            assert word.word
            assert word.start_time >= 0
            assert word.end_time >= word.start_time


@pytest.mark.integration
class TestPyannoteIntegration:
    """Integration tests for Pyannote diarization."""

    @skip_no_pyannote
    @skip_no_audio
    @pytest.mark.asyncio
    async def test_diarize_returns_segments(self):
        """Test that diarization returns speaker segments."""
        provider = PyannoteProvider(api_key=PYANNOTE_API_KEY)
        try:
            result = await provider.diarize(
                audio_path=str(SAMPLE_AUDIO),
                language=Language.ENGLISH,
            )
            assert len(result.segments) > 0
            for seg in result.segments:
                assert seg.speaker_label
                assert seg.start_time >= 0
                assert seg.end_time >= seg.start_time
        finally:
            await provider.close()

    @skip_no_pyannote
    @skip_no_audio
    @pytest.mark.asyncio
    async def test_get_speaker_embedding(self):
        """Test that embedding extraction returns a vector."""
        provider = PyannoteProvider(api_key=PYANNOTE_API_KEY)
        try:
            embedding = await provider.get_speaker_embedding(
                audio_path=str(SAMPLE_AUDIO),
            )
            assert isinstance(embedding, list)
            assert len(embedding) > 0
        finally:
            await provider.close()
