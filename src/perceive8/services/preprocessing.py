"""Audio preprocessing module: format conversion, normalization, and resampling."""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when audio preprocessing fails."""


@dataclass
class PreprocessingResult:
    """Result from audio preprocessing."""

    output_path: str
    duration_seconds: float
    sample_rate: int
    original_format: str
    was_converted: bool


TARGET_DBFS = -20.0
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def _preprocess_sync(input_path: str, output_dir: str) -> PreprocessingResult:
    """Synchronous preprocessing logic."""
    from pydub import AudioSegment

    path = Path(input_path)
    if not path.exists():
        raise PreprocessingError(f"Input file not found: {input_path}")

    original_format = path.suffix.lstrip(".").lower()
    if not original_format:
        original_format = "unknown"

    try:
        if original_format == "wav":
            audio = AudioSegment.from_wav(input_path)
        elif original_format == "mp3":
            audio = AudioSegment.from_mp3(input_path)
        elif original_format in ("m4a", "mp4"):
            audio = AudioSegment.from_file(input_path, format="m4a")
        elif original_format == "ogg":
            audio = AudioSegment.from_ogg(input_path)
        elif original_format == "flac":
            audio = AudioSegment.from_file(input_path, format="flac")
        elif original_format == "webm":
            audio = AudioSegment.from_file(input_path, format="webm")
        else:
            audio = AudioSegment.from_file(input_path)
    except Exception as e:
        raise PreprocessingError(f"Failed to load audio file: {e}") from e

    # Normalize audio levels to target dBFS
    if audio.dBFS != float("-inf"):
        change_in_dbfs = TARGET_DBFS - audio.dBFS
        audio = audio.apply_gain(change_in_dbfs)

    # Convert to mono
    if audio.channels != TARGET_CHANNELS:
        audio = audio.set_channels(TARGET_CHANNELS)

    # Resample to 16kHz
    if audio.frame_rate != TARGET_SAMPLE_RATE:
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

    was_converted = original_format != "wav"

    # Export as WAV
    os.makedirs(output_dir, exist_ok=True)
    output_filename = path.stem + "_preprocessed.wav"
    output_path = os.path.join(output_dir, output_filename)

    try:
        audio.export(output_path, format="wav")
    except Exception as e:
        raise PreprocessingError(f"Failed to export preprocessed audio: {e}") from e

    duration_seconds = len(audio) / 1000.0

    logger.info(
        "Preprocessed audio: %s -> %s (%.1fs, %dHz, converted=%s)",
        input_path,
        output_path,
        duration_seconds,
        TARGET_SAMPLE_RATE,
        was_converted,
    )

    return PreprocessingResult(
        output_path=output_path,
        duration_seconds=duration_seconds,
        sample_rate=TARGET_SAMPLE_RATE,
        original_format=original_format,
        was_converted=was_converted,
    )


async def preprocess_audio(input_path: str, output_dir: str) -> PreprocessingResult:
    """
    Preprocess audio file: convert to WAV, normalize, resample to 16kHz mono.

    Args:
        input_path: Path to the input audio file.
        output_dir: Directory to write the preprocessed WAV file.

    Returns:
        PreprocessingResult with output path and metadata.

    Raises:
        PreprocessingError: If preprocessing fails.
    """
    return await asyncio.to_thread(_preprocess_sync, input_path, output_dir)
