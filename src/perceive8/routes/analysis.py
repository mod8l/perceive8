"""Analysis endpoints."""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from perceive8.config import (
    DiarizationProvider,
    Language,
    TranscriptionProvider,
    get_settings,
)
from perceive8.database import get_db
from perceive8.models.database import Analysis, AudioFile, ProcessingRun, User
from perceive8.models.schemas import AnalysisResponse
from perceive8.services import analysis as analysis_service
from perceive8.services.pipeline import run_analysis_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("")
async def create_analysis(
    user_id: str = Form(...),
    audio_file: UploadFile = File(...),
    language: Language = Form(default=None),
    diarization_provider: Optional[DiarizationProvider] = Form(default=None),
    diarization_model: Optional[str] = Form(default=None),
    transcription_providers: Optional[List[TranscriptionProvider]] = Form(default=None),
    transcription_models: Optional[List[str]] = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit audio for analysis.

    Runs diarization and transcription, saves results to database.
    Supports running multiple transcription providers/models for comparison.
    """
    # Use defaults if not specified
    lang = language or settings.default_language
    diar_provider = diarization_provider or settings.default_diarization_provider
    trans_providers = transcription_providers or [settings.default_transcription_provider]

    # Read audio data
    audio_data = await audio_file.read()

    # Run analysis
    analysis = await analysis_service.create_analysis(
        db=db,
        user_id=user_id,
        audio_data=audio_data,
        filename=audio_file.filename or "audio.wav",
        language=lang,
        diarization_provider=diar_provider,
        transcription_providers=trans_providers,
        diarization_model=diarization_model,
        transcription_models=transcription_models,
    )

    return {
        "id": str(analysis.id),
        "language": analysis.language,
        "created_at": analysis.created_at.isoformat(),
    }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    user_id: str = Form(...),
    audio_file: UploadFile = File(...),
    language: Optional[Language] = Form(default=None),
    diarization_provider: Optional[DiarizationProvider] = Form(default=None),
    transcription_providers: Optional[List[TranscriptionProvider]] = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit audio for full pipeline analysis.

    Creates Analysis and AudioFile DB records, runs the full analysis pipeline,
    and returns the analysis result.
    """
    lang = language or settings.default_language
    diar_provider = diarization_provider or settings.default_diarization_provider
    trans_providers = transcription_providers or [settings.default_transcription_provider]

    audio_data = await audio_file.read()
    filename = audio_file.filename or "audio.wav"

    logger.info(
        "POST /analysis/analyze — filename=%s, file_size=%d bytes, user_id=%s",
        filename, len(audio_data), user_id,
    )

    # Get or create user
    from perceive8.services.analysis import get_or_create_user

    user = await get_or_create_user(db, user_id)

    # Create Analysis record
    analysis = Analysis(user_id=user.id, language=lang.value)
    db.add(analysis)
    await db.flush()

    # Create AudioFile record
    audio_file_record = AudioFile(
        analysis_id=analysis.id,
        original_path=filename,
    )
    db.add(audio_file_record)
    await db.flush()

    # Get optional services from app state
    embedding_service = getattr(request.app.state, "embedding_service", None)
    query_service = getattr(request.app.state, "query_service", None)

    # Run the full pipeline
    logger.info("Starting analysis pipeline for user_id=%s, filename=%s", user_id, filename)
    try:
        await run_analysis_pipeline(
            audio_data=audio_data,
            filename=filename,
            user_id=user_id,
            analysis_id=str(analysis.id),
            language=lang,
            diarization_provider=diar_provider,
            transcription_providers=trans_providers,
            embedding_service=embedding_service,
            query_service=query_service,
            db_session=db,
        )
    except Exception as exc:
        logger.error("Pipeline failed for analysis %s: %s", analysis.id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Analysis pipeline failed: {exc}") from exc

    logger.info("Analysis pipeline completed for analysis_id=%s", analysis.id)
    return AnalysisResponse(
        id=str(analysis.id),
        language=analysis.language,
        created_at=analysis.created_at.isoformat(),
    )


@router.get("")
async def list_analyses(
    user_id: str,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List analyses for a user."""
    from perceive8.models.database import User

    # Get user
    result = await db.execute(select(User).where(User.external_id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return {"analyses": [], "total": 0}

    # Get analyses
    result = await db.execute(
        select(Analysis)
        .where(Analysis.user_id == user.id)
        .order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    analyses = result.scalars().all()

    # Get total count
    from sqlalchemy import func
    count_result = await db.execute(
        select(func.count()).select_from(Analysis).where(Analysis.user_id == user.id)
    )
    total = count_result.scalar() or 0

    return {
        "analyses": [
            {"id": str(a.id), "language": a.language, "created_at": a.created_at.isoformat()}
            for a in analyses
        ],
        "total": total,
    }


@router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: UUID,
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get analysis details including all processing runs (scoped to user_id)."""
    result = await db.execute(
        select(Analysis)
        .join(User, Analysis.user_id == User.id)
        .options(selectinload(Analysis.processing_runs))
        .where(Analysis.id == analysis_id, User.external_id == user_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "id": str(analysis.id),
        "language": analysis.language,
        "created_at": analysis.created_at.isoformat(),
        "runs": [
            {
                "id": str(r.id),
                "run_type": r.run_type,
                "provider_name": r.provider_name,
                "model_name": r.model_name,
                "status": r.status,
                "processing_time_seconds": r.processing_time_seconds,
            }
            for r in analysis.processing_runs
        ],
    }


@router.get("/{analysis_id}/runs")
async def get_processing_runs(
    analysis_id: UUID,
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get all processing runs for an analysis (scoped to user_id)."""
    # Verify analysis belongs to user
    analysis_result = await db.execute(
        select(Analysis)
        .join(User, Analysis.user_id == User.id)
        .where(Analysis.id == analysis_id, User.external_id == user_id)
    )
    if analysis_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    result = await db.execute(
        select(ProcessingRun).where(ProcessingRun.analysis_id == analysis_id)
    )
    runs = result.scalars().all()

    return {
        "runs": [
            {
                "id": str(r.id),
                "run_type": r.run_type,
                "provider_name": r.provider_name,
                "model_name": r.model_name,
                "status": r.status,
                "processing_time_seconds": r.processing_time_seconds,
            }
            for r in runs
        ]
    }


@router.get("/{analysis_id}/runs/{run_id}")
async def get_processing_run(
    analysis_id: UUID,
    run_id: UUID,
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific processing run with its segments (scoped to user_id)."""
    from perceive8.models.database import DiarizationSegment, TranscriptSegment

    # Verify analysis belongs to user
    analysis_result = await db.execute(
        select(Analysis)
        .join(User, Analysis.user_id == User.id)
        .where(Analysis.id == analysis_id, User.external_id == user_id)
    )
    if analysis_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    result = await db.execute(
        select(ProcessingRun)
        .options(
            selectinload(ProcessingRun.transcript_segments),
            selectinload(ProcessingRun.diarization_segments),
        )
        .where(ProcessingRun.id == run_id, ProcessingRun.analysis_id == analysis_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Processing run not found")

    response = {
        "id": str(run.id),
        "run_type": run.run_type,
        "provider_name": run.provider_name,
        "model_name": run.model_name,
        "status": run.status,
        "processing_time_seconds": run.processing_time_seconds,
        "error_message": run.error_message,
    }

    if run.run_type == "diarization":
        response["segments"] = [
            {
                "speaker_label": s.speaker_label,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "confidence": s.confidence,
            }
            for s in run.diarization_segments
        ]
    else:
        response["segments"] = [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
                "confidence": s.confidence,
                "word_timestamps": s.word_timestamps,
            }
            for s in run.transcript_segments
        ]

    return response
