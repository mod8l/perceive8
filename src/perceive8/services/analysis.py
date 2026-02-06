"""Analysis service for orchestrating audio processing and storage."""

import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from perceive8.config import (
    DiarizationProvider,
    Language,
    TranscriptionProvider,
    get_settings,
)
from perceive8.models.database import (
    Analysis,
    AudioFile,
    DiarizationSegment as DBDiarizationSegment,
    ProcessingRun,
    TranscriptSegment,
    User,
)
from perceive8.providers.factory import get_diarization_provider, get_transcription_provider


settings = get_settings()


async def get_or_create_user(db: AsyncSession, external_id: str) -> User:
    """Get or create a user by external_id."""
    from sqlalchemy import select

    result = await db.execute(select(User).where(User.external_id == external_id))
    user = result.scalar_one_or_none()
    if not user:
        user = User(external_id=external_id)
        db.add(user)
        await db.flush()
    return user


async def create_analysis(
    db: AsyncSession,
    user_id: str,
    audio_data: bytes,
    filename: str,
    language: Language,
    diarization_provider: DiarizationProvider,
    transcription_providers: List[TranscriptionProvider],
    diarization_model: Optional[str] = None,
    transcription_models: Optional[List[str]] = None,
) -> Analysis:
    """
    Create an analysis: save audio, run providers, store results in DB.

    Returns the Analysis object with all processing runs.
    """
    # Get or create user
    user = await get_or_create_user(db, user_id)

    # Create analysis record
    analysis = Analysis(user_id=user.id, language=language.value)
    db.add(analysis)
    await db.flush()

    # Save audio file to temp location for processing
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, filename)
    with open(temp_audio_path, "wb") as f:
        f.write(audio_data)

    # Save to permanent storage
    storage_dir = Path(settings.audio_storage_path) / str(user.id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    permanent_path = storage_dir / f"{analysis.id}_{filename}"
    shutil.copy(temp_audio_path, permanent_path)

    # Create audio file record
    audio_file = AudioFile(
        analysis_id=analysis.id,
        original_path=str(permanent_path),
    )
    db.add(audio_file)
    await db.flush()

    try:
        # Run diarization
        try:
            await run_diarization(
                db=db,
                analysis_id=analysis.id,
                audio_path=temp_audio_path,
                language=language,
                provider=diarization_provider,
                model_name=diarization_model,
            )
        except Exception:
            pass  # Error already recorded in processing_run

        # Run transcription for each provider
        trans_models = transcription_models or [None] * len(transcription_providers)
        for trans_provider, trans_model in zip(transcription_providers, trans_models):
            try:
                await run_transcription(
                    db=db,
                    analysis_id=analysis.id,
                    audio_path=temp_audio_path,
                    language=language,
                    provider=trans_provider,
                    model_name=trans_model,
                )
            except Exception:
                pass  # Error already recorded in processing_run
    finally:
        # Cleanup temp file
        shutil.rmtree(temp_dir, ignore_errors=True)

    return analysis


async def run_diarization(
    db: AsyncSession,
    analysis_id: uuid.UUID,
    audio_path: str,
    language: Language,
    provider: DiarizationProvider,
    model_name: Optional[str] = None,
) -> ProcessingRun:
    """Run diarization and save results to database."""
    # Create processing run
    run = ProcessingRun(
        analysis_id=analysis_id,
        run_type="diarization",
        provider_name=provider.value,
        model_name=model_name,
        status="processing",
        started_at=datetime.utcnow(),
    )
    db.add(run)
    await db.flush()

    try:
        # Get provider and run diarization
        diar_provider = get_diarization_provider(provider)
        result = await diar_provider.diarize(
            audio_path=audio_path,
            language=language,
            model_name=model_name,
        )

        # Save segments
        for seg in result.segments:
            db_seg = DBDiarizationSegment(
                processing_run_id=run.id,
                speaker_label=seg.speaker_label,
                start_time=seg.start_time,
                end_time=seg.end_time,
                confidence=seg.confidence,
            )
            db.add(db_seg)

        # Update run status
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        run.model_name = result.model_name
        run.raw_response = result.raw_response
        if run.started_at:
            run.processing_time_seconds = (run.completed_at - run.started_at).total_seconds()

    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        run.completed_at = datetime.utcnow()
        raise

    await db.flush()
    return run


async def run_transcription(
    db: AsyncSession,
    analysis_id: uuid.UUID,
    audio_path: str,
    language: Language,
    provider: TranscriptionProvider,
    model_name: Optional[str] = None,
) -> ProcessingRun:
    """Run transcription and save results to database."""
    # Create processing run
    run = ProcessingRun(
        analysis_id=analysis_id,
        run_type="transcription",
        provider_name=provider.value,
        model_name=model_name,
        status="processing",
        started_at=datetime.utcnow(),
    )
    db.add(run)
    await db.flush()

    try:
        # Get provider and run transcription
        trans_provider = get_transcription_provider(provider)
        result = await trans_provider.transcribe(
            audio_path=audio_path,
            language=language,
            model_name=model_name,
        )

        # Save segments
        for seg in result.segments:
            word_timestamps = None
            if seg.words:
                word_timestamps = [
                    {"word": w.word, "start": w.start_time, "end": w.end_time, "confidence": w.confidence}
                    for w in seg.words
                ]

            db_seg = TranscriptSegment(
                processing_run_id=run.id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
                confidence=seg.confidence,
                word_timestamps=word_timestamps,
            )
            db.add(db_seg)

        # Update run status
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        run.model_name = result.model_name
        run.raw_response = result.raw_response
        if run.started_at:
            run.processing_time_seconds = (run.completed_at - run.started_at).total_seconds()

    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        run.completed_at = datetime.utcnow()
        raise

    await db.flush()
    return run
