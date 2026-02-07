"""Query endpoints for RAG Q&A over transcripts."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from perceive8.database import get_db
from perceive8.models.database import Analysis, ProcessingRun, TranscriptSegment, User
from perceive8.models.schemas import (
    BackfillResponse,
    QueryHistoryResponse,
    QueryRequest,
    QueryResponse,
    SourceSegment,
    TranscriptSegmentItem,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query_transcripts(
    body: QueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask questions about transcripts using RAG.

    If analysis_id is omitted, searches across all analyses.
    """
    logger.info("Query received: user_id=%s, question=%r, analysis_id=%s", body.user_id, body.question, body.analysis_id)
    # Validate analysis_id exists and belongs to user if provided
    if body.analysis_id:
        result = await db.execute(
            select(Analysis)
            .join(User, Analysis.user_id == User.id)
            .where(Analysis.id == body.analysis_id, User.external_id == body.user_id)
        )
        if result.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Analysis not found")

    query_service = getattr(request.app.state, "query_service", None)
    if query_service is None:
        raise HTTPException(status_code=503, detail="QueryService not available")

    try:
        result = await query_service.answer_question(
            question=body.question,
            user_id=body.user_id,
            analysis_id=body.analysis_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}") from exc

    sources = [
        SourceSegment(**s) for s in result.get("sources", [])
    ]
    logger.info("Query response sent: %d sources returned", len(sources))
    return QueryResponse(answer=result["answer"], sources=sources)


@router.post("/backfill", response_model=BackfillResponse)
async def backfill_transcripts(
    request: Request,
    user_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Backfill ChromaDB with transcript segments from PostgreSQL.

    If user_id is provided, only backfills segments for that user.
    """
    query_service = getattr(request.app.state, "query_service", None)
    if query_service is None:
        raise HTTPException(status_code=503, detail="QueryService not available")

    try:
        count = await query_service.backfill_from_db(db, user_id=user_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backfill failed: {exc}") from exc

    return BackfillResponse(segments_embedded=count)


@router.get("/history/{analysis_id}", response_model=QueryHistoryResponse)
async def get_query_history(
    analysis_id: str,
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return transcript segments for a given analysis (scoped to user_id)."""
    # Validate analysis exists and belongs to user
    result = await db.execute(
        select(Analysis)
        .join(User, Analysis.user_id == User.id)
        .where(Analysis.id == analysis_id, User.external_id == user_id)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Query segments via ProcessingRun, eagerly load speaker
    from sqlalchemy.orm import selectinload

    stmt = (
        select(TranscriptSegment)
        .join(ProcessingRun, TranscriptSegment.processing_run_id == ProcessingRun.id)
        .where(ProcessingRun.analysis_id == analysis_id)
        .options(selectinload(TranscriptSegment.speaker))
        .order_by(TranscriptSegment.start_time)
    )
    result = await db.execute(stmt)
    segments = result.scalars().all()

    items = [
        TranscriptSegmentItem(
            id=str(seg.id),
            text=seg.text,
            speaker=seg.speaker.name if seg.speaker else None,
            start_time=seg.start_time,
            end_time=seg.end_time,
            chromadb_id=seg.chromadb_id,
        )
        for seg in segments
    ]

    return QueryHistoryResponse(
        analysis_id=analysis_id,
        segments=items,
        total=len(items),
    )
