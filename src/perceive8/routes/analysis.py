"""Analysis endpoints."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from perceive8.config import (
    DiarizationProvider,
    Language,
    TranscriptionProvider,
    get_settings,
)

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
):
    """
    Submit audio for analysis.

    Supports running multiple transcription providers/models for comparison.
    Results are stored per processing_run for later comparison.
    """
    # Use defaults if not specified
    lang = language or settings.default_language
    diar_provider = diarization_provider or settings.default_diarization_provider
    trans_providers = transcription_providers or [settings.default_transcription_provider]

    # TODO: Implement analysis pipeline
    # 1. Save audio file to storage
    # 2. Check quality and enhance if needed
    # 3. Preprocess audio (convert format, normalize)
    # 4. Run diarization (creates processing_run)
    # 5. Run transcription for each provider (creates processing_runs)
    # 6. Store results and embed transcripts in ChromaDB

    return {
        "message": "Analysis not yet implemented",
        "config": {
            "language": lang,
            "diarization_provider": diar_provider,
            "transcription_providers": trans_providers,
        },
    }


@router.get("")
async def list_analyses(
    user_id: str,
    limit: int = 100,
    offset: int = 0,
):
    """List analyses for a user."""
    # TODO: Query from PostgreSQL
    return {"analyses": [], "total": 0}


@router.get("/{analysis_id}")
async def get_analysis(analysis_id: UUID):
    """Get analysis details including all processing runs."""
    # TODO: Get from PostgreSQL with processing_runs
    raise HTTPException(status_code=404, detail="Analysis not found")


@router.get("/{analysis_id}/runs")
async def get_processing_runs(analysis_id: UUID):
    """Get all processing runs for an analysis (for model comparison)."""
    # TODO: Get processing_runs with their segments
    return {"runs": []}


@router.get("/{analysis_id}/runs/{run_id}")
async def get_processing_run(analysis_id: UUID, run_id: UUID):
    """Get a specific processing run with its segments."""
    raise HTTPException(status_code=404, detail="Processing run not found")
