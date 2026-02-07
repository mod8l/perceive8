"""Analysis pipeline orchestrator."""

import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from perceive8.config import Language, get_settings
from perceive8.models.database import (
    DiarizationSegment as DiarizationSegmentModel,
    ProcessingRun,
    TranscriptSegment as TranscriptSegmentModel,
)
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

if TYPE_CHECKING:
    from perceive8.services.embedding import EmbeddingService
    from perceive8.services.query import QueryService

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
    matched_speaker_id: Optional[uuid.UUID] = None
    matched_speaker_name: Optional[str] = None


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


async def match_speakers(
    diarization_result: DiarizationResult,
    audio_path: str,
    embedding_service: "EmbeddingService",
    user_id: str,
    threshold: Optional[float] = None,
) -> dict[str, dict]:
    """Match diarized speakers against enrolled speaker embeddings.

    For each unique speaker label, find the longest segment, slice audio,
    extract embedding, and query ChromaDB.

    Returns:
        Mapping from diarization label to dict with 'name' and 'speaker_id'.
    """
    from pydub import AudioSegment

    from perceive8.providers.pyannote import PyannoteProvider

    settings = get_settings()
    if threshold is None:
        threshold = settings.speaker_match_threshold
    label_map: dict[str, dict] = {}

    # Group segments by speaker and find longest per speaker
    speaker_segments: dict[str, list] = {}
    for seg in diarization_result.segments:
        speaker_segments.setdefault(seg.speaker_label, []).append(seg)

    provider = PyannoteProvider(api_key=settings.pyannote_api_key)
    try:
        audio = AudioSegment.from_file(audio_path)

        for label, segments in speaker_segments.items():
            # Find longest segment
            longest = max(segments, key=lambda s: s.end_time - s.start_time)
            start_ms = int(longest.start_time * 1000)
            end_ms = int(longest.end_time * 1000)
            clip = audio[start_ms:end_ms]

            # Write clip to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                clip.export(tmp.name, format="wav")
                tmp_path = tmp.name

            try:
                embedding = await provider.get_speaker_embedding(tmp_path)
                matches = embedding_service.search_similar_speakers(
                    embedding=embedding,
                    user_id=user_id,
                    threshold=threshold,
                )
                if matches:
                    best = matches[0]
                    matched_name = best["metadata"].get("name", label)
                    matched_id = best["metadata"].get("speaker_id") or best.get("speaker_id")
                    label_map[label] = {
                        "name": matched_name,
                        "speaker_id": matched_id,
                    }
                    logger.info(
                        "Matched %s -> %s (similarity=%.3f)",
                        label,
                        matched_name,
                        best["similarity"],
                    )
            except Exception:
                logger.warning("Failed to match speaker %s", label, exc_info=True)
            finally:
                import os
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    finally:
        await provider.close()

    return label_map


async def persist_pipeline_results(
    db_session: AsyncSession,
    result: "PipelineResult",
    analysis_id: str,
    speaker_label_map: Optional[dict[str, dict]] = None,
) -> dict:
    """Persist pipeline results to PostgreSQL.

    Creates ProcessingRun, TranscriptSegment, and DiarizationSegment records.

    Args:
        db_session: SQLAlchemy async session.
        result: The pipeline result to persist.
        analysis_id: Analysis UUID string.
        speaker_label_map: Mapping from diarization label to matched speaker info.

    Returns:
        Dict with created record references.
    """
    analysis_uuid = uuid.UUID(analysis_id)
    now = datetime.utcnow()

    # Create ProcessingRun
    processing_run = ProcessingRun(
        analysis_id=analysis_uuid,
        run_type="full_pipeline",
        provider_name="pipeline",
        status="completed",
        completed_at=now,
    )
    db_session.add(processing_run)
    await db_session.flush()  # get processing_run.id

    # Create TranscriptSegment records from merged segments
    transcript_segments = []
    for idx, seg in enumerate(result.merged_segments):
        chromadb_id = f"{analysis_id}:{idx}"
        ts = TranscriptSegmentModel(
            processing_run_id=processing_run.id,
            speaker_id=seg.matched_speaker_id,
            start_time=seg.start_time,
            end_time=seg.end_time,
            text=seg.text,
            confidence=seg.confidence,
            word_timestamps=[
                {"word": w.word, "start": w.start_time, "end": w.end_time, "confidence": w.confidence}
                for w in seg.words
            ] if seg.words else None,
            chromadb_id=chromadb_id,
        )
        db_session.add(ts)
        transcript_segments.append(ts)

    # Create DiarizationSegment records from raw diarization results
    diarization_segments = []
    if result.diarization_result:
        label_map = speaker_label_map or {}
        for dseg in result.diarization_result.segments:
            matched_id = None
            match_info = label_map.get(dseg.speaker_label)
            if match_info:
                sid = match_info.get("speaker_id")
                if sid:
                    try:
                        matched_id = uuid.UUID(sid) if isinstance(sid, str) else sid
                    except (ValueError, TypeError):
                        pass

            ds = DiarizationSegmentModel(
                processing_run_id=processing_run.id,
                speaker_label=dseg.speaker_label,
                matched_speaker_id=matched_id,
                start_time=dseg.start_time,
                end_time=dseg.end_time,
                confidence=dseg.confidence,
            )
            db_session.add(ds)
            diarization_segments.append(ds)

    await db_session.commit()

    return {
        "processing_run": processing_run,
        "transcript_segments": transcript_segments,
        "diarization_segments": diarization_segments,
    }


async def run_analysis_pipeline(
    audio_data: bytes,
    filename: str,
    user_id: str,
    analysis_id: str,
    language: Language,
    diarization_provider,
    transcription_providers: list,
    embedding_service: Optional["EmbeddingService"] = None,
    query_service: Optional["QueryService"] = None,
    db_session: Optional[AsyncSession] = None,
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

    pipeline_start = time.monotonic()

    # Convert 'auto' language to None so providers use auto-detection
    provider_language = None if language == Language.AUTO else language

    # 1. Store original
    logger.info("Pipeline [%s]: storing original file", analysis_id)
    t0 = time.monotonic()
    storage_result = await save_audio_file(audio_data, filename, user_id, analysis_id)
    original_path = storage_result.file_path
    logger.info("Pipeline [%s]: file stored in %.2fs", analysis_id, time.monotonic() - t0)

    # 2. Preprocess
    logger.info("Pipeline [%s]: Starting preprocessing...", analysis_id)
    t0 = time.monotonic()
    output_dir = str(Path(original_path).parent)
    preprocessing_info = await preprocess_audio(original_path, output_dir)
    audio_path = preprocessing_info.output_path
    logger.info("Pipeline [%s]: Preprocessing complete in %.2fs", analysis_id, time.monotonic() - t0)

    # 3. Quality check
    logger.info("Pipeline [%s]: Starting quality analysis...", analysis_id)
    t0 = time.monotonic()
    quality_info = await analyze_audio_quality(audio_path)
    logger.info("Pipeline [%s]: Quality analysis complete in %.2fs (SNR=%.1f dB)", analysis_id, time.monotonic() - t0, quality_info.snr_db)

    # 4. Enhance if needed
    enhanced_path: Optional[str] = None
    was_enhanced = False
    if quality_info.needs_enhancement:
        logger.info("Pipeline [%s]: Starting audio enhancement (SNR=%.1f dB)...", analysis_id, quality_info.snr_db)
        t0 = time.monotonic()
        try:
            enhancement_result = await enhance_audio(audio_path, output_dir)
            was_enhanced = enhancement_result.was_enhanced
            if was_enhanced:
                enhanced_path = enhancement_result.output_path
                audio_path = enhanced_path
            logger.info("Pipeline [%s]: Enhancement complete in %.2fs (enhanced=%s)", analysis_id, time.monotonic() - t0, was_enhanced)
        except Exception:
            logger.warning("Pipeline [%s]: enhancement failed, continuing with original", analysis_id, exc_info=True)

    # 5. Run diarization
    diarization_result: Optional[DiarizationResult] = None
    try:
        logger.info("Pipeline [%s]: Starting diarization...", analysis_id)
        t0 = time.monotonic()
        provider = get_diarization_provider(diarization_provider)
        diarization_result = await provider.diarize(audio_path, provider_language)
        logger.info("Pipeline [%s]: Diarization complete in %.2fs (%d segments)", analysis_id, time.monotonic() - t0, len(diarization_result.segments) if diarization_result else 0)
    except Exception:
        logger.warning("Pipeline [%s]: diarization failed", analysis_id, exc_info=True)

    # 6. Run transcription(s)
    transcription_results: List[TranscriptionResult] = []
    for tp in transcription_providers:
        try:
            logger.info("Pipeline [%s]: Starting transcription with %s...", analysis_id, tp)
            t0 = time.monotonic()
            provider = get_transcription_provider(tp)
            result = await provider.transcribe(audio_path, provider_language)
            transcription_results.append(result)
            logger.info("Pipeline [%s]: Transcription with %s complete in %.2fs (%d segments)", analysis_id, tp, time.monotonic() - t0, len(result.segments))
        except Exception:
            logger.warning("Pipeline [%s]: transcription with %s failed", analysis_id, tp, exc_info=True)

    # 7. Match speakers against enrolled embeddings
    speaker_label_map: dict[str, str] = {}
    if diarization_result and embedding_service:
        try:
            logger.info("Pipeline [%s]: Starting speaker matching...", analysis_id)
            t0 = time.monotonic()
            speaker_label_map = await match_speakers(
                diarization_result=diarization_result,
                audio_path=audio_path,
                embedding_service=embedding_service,
                user_id=user_id,
            )
            logger.info("Pipeline [%s]: Speaker matching complete in %.2fs (%d matched)", analysis_id, time.monotonic() - t0, len(speaker_label_map))
        except Exception:
            logger.warning("Pipeline [%s]: speaker matching failed", analysis_id, exc_info=True)

    # 8. Merge results
    merged_segments: List[MergedSegment] = []
    if diarization_result and transcription_results:
        logger.info("Pipeline [%s]: Merging diarization and transcription results...", analysis_id)
        merged_segments = merge_results(diarization_result, transcription_results[0])
        logger.info("Pipeline [%s]: Merge complete — %d merged segments", analysis_id, len(merged_segments))

    # Apply speaker name mapping
    if speaker_label_map:
        for seg in merged_segments:
            if seg.speaker_label in speaker_label_map:
                match_info = speaker_label_map[seg.speaker_label]
                seg.matched_speaker_name = match_info["name"]
                speaker_id_str = match_info.get("speaker_id")
                if speaker_id_str:
                    try:
                        seg.matched_speaker_id = uuid.UUID(speaker_id_str)
                    except (ValueError, TypeError):
                        pass
                seg.speaker_label = match_info["name"]

    # 9. Embed transcript segments for RAG
    if query_service and merged_segments:
        try:
            logger.info("Pipeline [%s]: Starting embedding of %d transcript segments...", analysis_id, len(merged_segments))
            t0 = time.monotonic()
            segment_dicts = [
                {
                    "speaker": seg.speaker_label,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                }
                for seg in merged_segments
            ]
            await query_service.embed_transcript_segments(analysis_id, segment_dicts, user_id=user_id)
            logger.info("Pipeline [%s]: Embedding complete in %.2fs", analysis_id, time.monotonic() - t0)
        except Exception:
            logger.warning("Pipeline [%s]: transcript embedding failed", analysis_id, exc_info=True)

    pipeline_result = PipelineResult(
        original_path=original_path,
        enhanced_path=enhanced_path,
        was_enhanced=was_enhanced,
        preprocessing_info=preprocessing_info,
        quality_info=quality_info,
        diarization_result=diarization_result,
        transcription_results=transcription_results,
        merged_segments=merged_segments,
    )

    # 10. Persist to DB if session provided
    if db_session is not None:
        try:
            logger.info("Pipeline [%s]: Persisting results to DB...", analysis_id)
            t0 = time.monotonic()
            await persist_pipeline_results(
                db_session=db_session,
                result=pipeline_result,
                analysis_id=analysis_id,
                speaker_label_map=speaker_label_map,
            )
            logger.info("Pipeline [%s]: DB persistence complete in %.2fs", analysis_id, time.monotonic() - t0)
        except Exception:
            logger.warning("Pipeline [%s]: DB persistence failed", analysis_id, exc_info=True)

    logger.info("Pipeline [%s]: All steps complete — total time %.2fs", analysis_id, time.monotonic() - pipeline_start)
    return pipeline_result
