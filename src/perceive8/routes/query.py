"""Query endpoints for RAG Q&A over transcripts."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from perceive8.database import get_db

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    analysis_id: Optional[str] = None


class SourceSegment(BaseModel):
    analysis_id: str
    speaker_name: Optional[str] = None
    start_time: float
    end_time: float
    text: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceSegment]


class BackfillResponse(BaseModel):
    segments_embedded: int


@router.post("", response_model=QueryResponse)
async def query_transcripts(body: QueryRequest, request: Request):
    """
    Ask questions about transcripts using RAG.

    If analysis_id is omitted, searches across all analyses.
    """
    query_service = getattr(request.app.state, "query_service", None)
    if query_service is None:
        raise HTTPException(status_code=503, detail="QueryService not available")

    try:
        result = await query_service.answer_question(
            question=body.question,
            analysis_id=body.analysis_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}") from exc

    sources = [
        SourceSegment(**s) for s in result.get("sources", [])
    ]
    return QueryResponse(answer=result["answer"], sources=sources)


@router.post("/backfill", response_model=BackfillResponse)
async def backfill_transcripts(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Backfill ChromaDB with transcript segments from PostgreSQL."""
    query_service = getattr(request.app.state, "query_service", None)
    if query_service is None:
        raise HTTPException(status_code=503, detail="QueryService not available")

    try:
        count = await query_service.backfill_from_db(db)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backfill failed: {exc}") from exc

    return BackfillResponse(segments_embedded=count)
