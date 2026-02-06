"""Analysis pipeline orchestrator."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from perceive8.config import Language
from perceive8.providers.base import (
    DiarizationResult,
    TranscriptionResult,
    WordTimestamp,
)
from perceive8.providers.factory import get_diarization_provider, get_transcription_provider
from perceive8.services.enhancement import (
    AudioQualityInfo,
    EnhancementResult,
    enhance_audio,
    analyze_audio_quality,
)
from perceive8.services.preprocessing import PreprocessingResult, preprocess_audio
from perceive8.services.storage import save_audio_file

logger = logging.getLogger(__name__)


@dataclass
class MergedSegment:
    """A transcript segment merged with speaker diarization."""

    speaker_label: str
    start_time: float
    end_time: float
    text: str
    confidence: Optional[float] = None
    words: List[WordTimestamp] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Full result from the analysis pipeline."""

    original_path: str
    enhanced_path: Optional[str]
    was_enhanced: bool
    preprocessing_info: PreprocessingResult
    quality_info: AudioQualityInfo
    diarization_result: Optional[DiarizationResult]
    transcription_results: List[TranscriptionResult]
    merged_segments: List[MergedSegment]


def merge_results(
    diarization_result: DiarizationResult,
    transcription_result: TranscriptionResult,
) -> List[MergedSegment]:
    """Align transcription segments with speaker labels by time overlap.

    For each transcription segment, find the diarization segment with the
    greatest time overlap and assign that speaker label.

    Args:
        diarization_result: Speaker diarization output.
        transcription_result: Transcription output.

    Returns:
        List of MergedSegment with speaker labels assigned.
    """
    merged: List[MergedSegment] = []

    for tseg in transcription_result.segments:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for dseg in diarization_result.segments:
            overlap_start = max(tseg.start_time, dseg.start_time)
            overlap_end = min(tseg.end_time, dseg.end_time)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg.speaker_label

        merged.append(
            MergedSegment(
                speaker_label=best_speaker,
                start_time=tseg.start_time,
                end_time=tseg.end_time,
                text=tseg.text,
                confidence=tseg.confidence,
                words=list(tseg.words),
            )
        )

    return merged


async def run_analysis_pipeline(
    audio_data: bytes,
    filename: str,
    user_id: str,
    analysis_id: str,
    language: Language,
    diarization_provider,
    transcription_providers: list,
) -> PipelineResult:
    """Run the full analysis pipeline.

    Steps: store original → preprocess → quality check → enhance if needed →
    store enhanced → run diarization → run transcription(s) → merge results.

    Args:
        audio_data: Raw audio bytes.
        filename: Original filename.
        user_id: User identifier.
        analysis_id: Analysis identifier.
        language: Audio language.
        diarization_provider: Diarization provider enum value (or None for default).
        transcription_providers: List of transcription provider enum values.

    Returns:
        PipelineResult with all outputs.
    """
    import os
    from pathlib import Path

    # 1. Store original
    logger.info("Pipeline [%s]: storing original file", analysis_id)
    storage_result = await save_audio_file(audio_data, filename, user_id, analysis_id)
    original_path = storage_result.file_path

    # 2. Preprocess
    logger.info("Pipeline [%s]: preprocessing", analysis_id)
    output_dir = str(Path(original_path).parent)
    preprocessing_info = await preprocess_audio(original_path, output_dir)
    audio_path = preprocessing_info.output_path

    # 3. Quality check
    logger.info("Pipeline [%s]: analyzing quality", analysis_id)
    quality_info = await analyze_audio_quality(audio_path)

    # 4. Enhance if needed
    enhanced_path: Optional[str] = None
    was_enhanced = False
    if quality_info.needs_enhancement:
        logger.info("Pipeline [%s]: enhancing audio (SNR=%.1f dB)", analysis_id, quality_info.snr_db)
        try:
            enhancement_result = await enhance_audio(audio_path, output_dir)
            was_enhanced = enhancement_result.was_enhanced
            if was_enhanced:
                enhanced_path = enhancement_result.output_path
                audio_path = enhanced_path
        except Exception:
            logger.warning("Pipeline [%s]: enhancement failed, continuing with original", analysis_id, exc_info=True)

    # 5. Run diarization
    diarization_result: Optional[DiarizationResult] = None
    try:
        logger.info("Pipeline [%s]: running diarization", analysis_id)
        provider = get_diarization_provider(diarization_provider)
        diarization_result = await provider.diarize(audio_path, language)
    except Exception:
        logger.warning("Pipeline [%s]: diarization failed", analysis_id, exc_info=True)

    # 6. Run transcription(s)
    transcription_results: List[TranscriptionResult] = []
    for tp in transcription_providers:
        try:
            logger.info("Pipeline [%s]: running transcription with %s", analysis_id, tp)
            provider = get_transcription_provider(tp)
            result = await provider.transcribe(audio_path, language)
            transcription_results.append(result)
        except Exception:
            logger.warning("Pipeline [%s]: transcription with %s failed", analysis_id, tp, exc_info=True)

    # 7. Merge results
    merged_segments: List[MergedSegment] = []
    if diarization_result and transcription_results:
        merged_segments = merge_results(diarization_result, transcription_results[0])

    return PipelineResult(
        original_path=original_path,
        enhanced_path=enhanced_path,
        was_enhanced=was_enhanced,
        preprocessing_info=preprocessing_info,
        quality_info=quality_info,
        diarization_result=diarization_result,
        transcription_results=transcription_results,
        merged_segments=merged_segments,
    )
