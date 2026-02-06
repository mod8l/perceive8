"""Tests for provider base classes."""

import pytest
from dataclasses import asdict

from perceive8.providers.base import (
    DiarizationSegment,
    DiarizationResult,
    WordTimestamp,
    TranscriptionSegment,
    TranscriptionResult,
    DiarizationProviderInterface,
    TranscriptionProviderInterface,
)
from perceive8.config import Language


class TestDiarizationSegment:
    """Tests for DiarizationSegment dataclass."""

    def test_create_segment(self):
        """Test creating a diarization segment."""
        segment = DiarizationSegment(
            speaker_label="SPEAKER_00",
            start_time=0.0,
            end_time=5.0,
            confidence=0.95,
        )
        assert segment.speaker_label == "SPEAKER_00"
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.confidence == 0.95

    def test_create_segment_without_confidence(self):
        """Test creating segment without optional confidence."""
        segment = DiarizationSegment(
            speaker_label="SPEAKER_01",
            start_time=5.0,
            end_time=10.0,
        )
        assert segment.confidence is None


class TestDiarizationResult:
    """Tests for DiarizationResult dataclass."""

    def test_create_result(self):
        """Test creating a diarization result."""
        segments = [
            DiarizationSegment("SPEAKER_00", 0.0, 5.0),
            DiarizationSegment("SPEAKER_01", 5.0, 10.0),
        ]
        result = DiarizationResult(
            segments=segments,
            model_name="pyannote/speaker-diarization",
        )
        assert len(result.segments) == 2
        assert result.model_name == "pyannote/speaker-diarization"
        assert result.raw_response is None


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_create_segment(self):
        """Test creating a transcription segment."""
        segment = TranscriptionSegment(
            start_time=0.0,
            end_time=5.0,
            text="Hello, world!",
            confidence=0.98,
        )
        assert segment.text == "Hello, world!"
        assert segment.words == []

    def test_segment_with_words(self):
        """Test segment with word timestamps."""
        words = [
            WordTimestamp("Hello", 0.0, 0.5, 0.99),
            WordTimestamp("world", 0.6, 1.0, 0.97),
        ]
        segment = TranscriptionSegment(
            start_time=0.0,
            end_time=1.0,
            text="Hello world",
            words=words,
        )
        assert len(segment.words) == 2
        assert segment.words[0].word == "Hello"


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_create_result(self):
        """Test creating a transcription result."""
        segments = [
            TranscriptionSegment(0.0, 5.0, "Hello"),
            TranscriptionSegment(5.0, 10.0, "World"),
        ]
        result = TranscriptionResult(
            segments=segments,
            full_text="Hello World",
            model_name="whisper-1",
            language="en",
        )
        assert result.full_text == "Hello World"
        assert result.model_name == "whisper-1"
        assert result.language == "en"


class TestProviderInterfaces:
    """Tests for provider interface classes."""

    def test_diarization_interface_is_abstract(self):
        """Test that DiarizationProviderInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            DiarizationProviderInterface()

    def test_transcription_interface_is_abstract(self):
        """Test that TranscriptionProviderInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            TranscriptionProviderInterface()
