"""Tests for the audio processing pipeline modules."""

import os
import tempfile

import pytest

from perceive8.providers.base import (
    DiarizationResult,
    DiarizationSegment,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)
from perceive8.services.enhancement import analyze_audio_quality, enhance_audio
from perceive8.services.pipeline import MergedSegment, merge_results
from perceive8.services.preprocessing import preprocess_audio
from perceive8.services.storage import save_audio_file, get_audio_path, delete_analysis_files

SAMPLE_WAV = os.path.join(os.path.dirname(__file__), "fixtures", "sample.wav")


# --- Preprocessing tests ---


@pytest.mark.asyncio
async def test_preprocess_wav_passthrough():
    """WAV input should produce a preprocessed WAV output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await preprocess_audio(SAMPLE_WAV, tmpdir)
        assert result.output_path.endswith(".wav")
        assert os.path.exists(result.output_path)
        assert result.original_format == "wav"
        assert result.sample_rate == 16000
        assert result.duration_seconds > 0


@pytest.mark.asyncio
async def test_preprocess_normalization():
    """Normalization should run without error on sample.wav."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await preprocess_audio(SAMPLE_WAV, tmpdir)
        assert result.duration_seconds > 0
        assert result.was_converted is False


# --- Enhancement tests ---


@pytest.mark.asyncio
async def test_analyze_audio_quality_returns_float():
    """SNR analysis should return a float value."""
    # Preprocess first to get a proper WAV
    with tempfile.TemporaryDirectory() as tmpdir:
        prep = await preprocess_audio(SAMPLE_WAV, tmpdir)
        quality = await analyze_audio_quality(prep.output_path)
        assert isinstance(quality.snr_db, float)
        assert isinstance(quality.needs_enhancement, bool)
        assert quality.duration_seconds > 0


@pytest.mark.asyncio
async def test_enhance_audio_on_sample():
    """Enhancement should run without error on sample.wav."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prep = await preprocess_audio(SAMPLE_WAV, tmpdir)
        result = await enhance_audio(prep.output_path, tmpdir)
        assert isinstance(result.was_enhanced, bool)
        assert isinstance(result.original_snr_db, float)


# --- Storage tests ---


@pytest.mark.asyncio
async def test_save_and_retrieve(monkeypatch):
    """Save a file and retrieve its path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr("perceive8.services.storage.get_settings", lambda: type("S", (), {"audio_storage_path": tmpdir})())
        data = b"fake audio data"
        result = await save_audio_file(data, "test.wav", "user1", "analysis1")
        assert os.path.exists(result.file_path)
        assert result.size_bytes == len(data)

        path = await get_audio_path("user1", "analysis1", "test.wav")
        assert path == result.file_path


@pytest.mark.asyncio
async def test_delete_analysis_files(monkeypatch):
    """Delete should remove the analysis directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr("perceive8.services.storage.get_settings", lambda: type("S", (), {"audio_storage_path": tmpdir})())
        await save_audio_file(b"data", "test.wav", "user1", "analysis1")
        await delete_analysis_files("user1", "analysis1")
        assert not os.path.exists(os.path.join(tmpdir, "user1", "analysis1"))


# --- Pipeline merge tests ---


def test_merge_results_basic():
    """Merge should assign speaker labels based on time overlap."""
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(speaker_label="SPEAKER_00", start_time=0.0, end_time=5.0),
            DiarizationSegment(speaker_label="SPEAKER_01", start_time=5.0, end_time=10.0),
        ],
        model_name="test",
    )
    transcription = TranscriptionResult(
        segments=[
            TranscriptionSegment(start_time=0.5, end_time=4.0, text="Hello world", confidence=0.9),
            TranscriptionSegment(start_time=5.5, end_time=9.0, text="Goodbye world", confidence=0.8),
        ],
        full_text="Hello world Goodbye world",
        model_name="test",
        language="en",
    )

    merged = merge_results(diarization, transcription)
    assert len(merged) == 2
    assert merged[0].speaker_label == "SPEAKER_00"
    assert merged[0].text == "Hello world"
    assert merged[1].speaker_label == "SPEAKER_01"
    assert merged[1].text == "Goodbye world"


def test_merge_results_overlap():
    """When a transcript segment spans two speakers, pick the one with more overlap."""
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(speaker_label="SPEAKER_00", start_time=0.0, end_time=3.0),
            DiarizationSegment(speaker_label="SPEAKER_01", start_time=3.0, end_time=10.0),
        ],
        model_name="test",
    )
    transcription = TranscriptionResult(
        segments=[
            TranscriptionSegment(start_time=2.0, end_time=6.0, text="Overlap test", confidence=0.9),
        ],
        full_text="Overlap test",
        model_name="test",
        language="en",
    )

    merged = merge_results(diarization, transcription)
    assert len(merged) == 1
    # Overlap with SPEAKER_00: 1.0s (2-3), SPEAKER_01: 3.0s (3-6)
    assert merged[0].speaker_label == "SPEAKER_01"
