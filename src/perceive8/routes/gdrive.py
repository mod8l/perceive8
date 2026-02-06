"""Google Drive analysis endpoint."""

import asyncio
import logging
import tempfile
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from perceive8.config import get_settings
from perceive8.database import get_db
from perceive8.models.database import Analysis, AudioFile
from perceive8.models.schemas import (
    GDriveAnalyzeRequest,
    GDriveAnalyzeResponse,
    GDriveFileResult,
)
from perceive8.services.gdrive import download_file, parse_file_id
from perceive8.services.pipeline import run_analysis_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


async def _process_single_file(
    file_id_or_url: str,
    language: Optional[str],
    user_id: str,
    api_key: str,
    max_size_mb: int,
    semaphore: asyncio.Semaphore,
    request: Request,
    db: AsyncSession,
) -> GDriveFileResult:
    """Process a single Google Drive file: download → pipeline."""
    file_id = ""
    filename = ""
    try:
        file_id = parse_file_id(file_id_or_url)

        async with semaphore:
            audio_data, filename = await download_file(file_id, api_key, max_size_mb)

        # Get or create user
        from perceive8.services.analysis import get_or_create_user

        user = await get_or_create_user(db, user_id)

        # Create Analysis record
        lang = language or settings.default_language.value
        analysis = Analysis(user_id=user.id, language=lang)
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

        from perceive8.config import Language

        lang_enum = Language(lang) if lang in [e.value for e in Language] else settings.default_language

        await run_analysis_pipeline(
            audio_data=audio_data,
            filename=filename,
            user_id=user_id,
            analysis_id=str(analysis.id),
            language=lang_enum,
            diarization_provider=settings.default_diarization_provider,
            transcription_providers=[settings.default_transcription_provider],
            embedding_service=embedding_service,
            query_service=query_service,
            db_session=db,
        )

        return GDriveFileResult(
            file_id=file_id,
            filename=filename,
            status="success",
            analysis_id=analysis.id,
        )

    except Exception as exc:
        logger.error("Failed to process gdrive file %s: %s", file_id_or_url, exc, exc_info=True)
        return GDriveFileResult(
            file_id=file_id or file_id_or_url,
            filename=filename or "unknown",
            status="error",
            error=str(exc),
        )


@router.post("/analyze/gdrive", response_model=GDriveAnalyzeResponse)
async def analyze_gdrive(
    body: GDriveAnalyzeRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Analyze audio files from Google Drive.

    Downloads files from Google Drive and runs the analysis pipeline on each.
    Files are processed concurrently with a configurable concurrency limit.
    Individual file failures do not fail the entire batch.
    """
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="Google API key is not configured")

    if not body.files:
        raise HTTPException(status_code=422, detail="At least one file is required")

    if len(body.files) > 20:
        raise HTTPException(status_code=422, detail="Maximum 20 files per request")

    user_id = str(body.user_id) if body.user_id else "anonymous"
    semaphore = asyncio.Semaphore(settings.gdrive_max_concurrent)

    results = await asyncio.gather(
        *[
            _process_single_file(
                file_id_or_url=f.file_id_or_url,
                language=f.language,
                user_id=user_id,
                api_key=settings.google_api_key,
                max_size_mb=settings.gdrive_max_file_size_mb,
                semaphore=semaphore,
                request=request,
                db=db,
            )
            for f in body.files
        ]
    )

    succeeded = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "error")

    return GDriveAnalyzeResponse(
        results=list(results),
        total=len(results),
        succeeded=succeeded,
        failed=failed,
    )
