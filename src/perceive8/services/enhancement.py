"""Audio enhancement module: quality analysis and noise reduction."""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
import noisereduce as nr
from scipy.io import wavfile

logger = logging.getLogger(__name__)


class EnhancementError(Exception):
    """Raised when audio enhancement fails."""


@dataclass
class AudioQualityInfo:
    """Audio quality analysis result."""

    snr_db: float
    needs_enhancement: bool
    duration_seconds: float


@dataclass
class EnhancementResult:
    """Result from audio enhancement."""

    output_path: str
    was_enhanced: bool
    original_snr_db: float
    enhanced_snr_db: float


SNR_THRESHOLD_DB = 20.0
NOISE_REFERENCE_SECONDS = 0.5


def _estimate_snr(samples: np.ndarray, sample_rate: int) -> float:
    """Estimate SNR using first 0.5s as noise reference."""
    samples = samples.astype(np.float64)
    noise_samples = int(NOISE_REFERENCE_SECONDS * sample_rate)
    noise_samples = min(noise_samples, len(samples))

    if noise_samples == 0 or len(samples) == 0:
        return 0.0

    noise = samples[:noise_samples]
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return 100.0  # Essentially silent noise floor

    signal_power = np.mean(samples ** 2)
    if signal_power < 1e-10:
        return 0.0

    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


def _analyze_quality_sync(audio_path: str) -> AudioQualityInfo:
    """Synchronous quality analysis."""
    try:
        sample_rate, data = wavfile.read(audio_path)
    except Exception as e:
        raise EnhancementError(f"Failed to read audio file: {e}") from e

    if data.ndim > 1:
        data = data.mean(axis=1)

    duration_seconds = len(data) / sample_rate
    snr_db = _estimate_snr(data, sample_rate)
    needs_enhancement = snr_db < SNR_THRESHOLD_DB

    logger.info(
        "Audio quality: SNR=%.1f dB, needs_enhancement=%s, duration=%.1fs",
        snr_db, needs_enhancement, duration_seconds,
    )

    return AudioQualityInfo(
        snr_db=snr_db,
        needs_enhancement=needs_enhancement,
        duration_seconds=duration_seconds,
    )


def _enhance_sync(audio_path: str, output_dir: str) -> EnhancementResult:
    """Synchronous enhancement logic."""
    import os
    from pathlib import Path

    try:
        sample_rate, data = wavfile.read(audio_path)
    except Exception as e:
        raise EnhancementError(f"Failed to read audio file: {e}") from e

    original_dtype = data.dtype
    if data.ndim > 1:
        data = data.mean(axis=1)

    data_float = data.astype(np.float64)
    original_snr = _estimate_snr(data_float, sample_rate)

    if original_snr >= SNR_THRESHOLD_DB:
        logger.info("SNR %.1f dB >= threshold, skipping enhancement", original_snr)
        return EnhancementResult(
            output_path=audio_path,
            was_enhanced=False,
            original_snr_db=original_snr,
            enhanced_snr_db=original_snr,
        )

    try:
        reduced = nr.reduce_noise(y=data_float, sr=sample_rate)
    except Exception as e:
        raise EnhancementError(f"Noise reduction failed: {e}") from e

    enhanced_snr = _estimate_snr(reduced, sample_rate)

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(audio_path).stem
    output_path = os.path.join(output_dir, f"{stem}_enhanced.wav")

    reduced_out = np.clip(reduced, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max).astype(original_dtype)
    wavfile.write(output_path, sample_rate, reduced_out)

    logger.info(
        "Enhanced audio: SNR %.1f -> %.1f dB, saved to %s",
        original_snr, enhanced_snr, output_path,
    )

    return EnhancementResult(
        output_path=output_path,
        was_enhanced=True,
        original_snr_db=original_snr,
        enhanced_snr_db=enhanced_snr,
    )


async def analyze_audio_quality(audio_path: str) -> AudioQualityInfo:
    """Analyze audio quality and estimate SNR.

    Args:
        audio_path: Path to WAV audio file.

    Returns:
        AudioQualityInfo with SNR and enhancement recommendation.
    """
    return await asyncio.to_thread(_analyze_quality_sync, audio_path)


async def enhance_audio(audio_path: str, output_dir: str) -> EnhancementResult:
    """Enhance audio by applying noise reduction if SNR is below threshold.

    Args:
        audio_path: Path to WAV audio file.
        output_dir: Directory to write enhanced file.

    Returns:
        EnhancementResult with enhancement details.
    """
    return await asyncio.to_thread(_enhance_sync, audio_path, output_dir)
