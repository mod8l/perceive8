"""Pydantic request/response schemas for API endpoints."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


# --- Analysis schemas ---


class AnalysisRequest(BaseModel):
    """Request body for creating an analysis (used for JSON endpoint variant)."""

    user_id: str
    language: Optional[str] = None
    diarization_provider: Optional[str] = None
    diarization_model: Optional[str] = None
    transcription_providers: Optional[List[str]] = None
    transcription_models: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    """Response after creating or retrieving an analysis."""

    id: str
    language: str
    created_at: str


# --- Speaker schemas ---


class SpeakerEnrollRequest(BaseModel):
    """Request body for enrolling a speaker (used for JSON endpoint variant)."""

    user_id: str
    name: str


class SpeakerResponse(BaseModel):
    """Response for a single speaker."""

    id: str
    name: str
    user_id: Optional[str] = None
    chromadb_id: Optional[str] = None
    created_at: Optional[str] = None


class SpeakerListResponse(BaseModel):
    """Response for listing speakers."""

    speakers: List[SpeakerResponse]
    total: int


# --- Query schemas ---


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


class TranscriptSegmentItem(BaseModel):
    id: str
    text: str
    speaker: Optional[str] = None
    start_time: float
    end_time: float
    chromadb_id: Optional[str] = None


class QueryHistoryResponse(BaseModel):
    analysis_id: str
    segments: List[TranscriptSegmentItem]
    total: int
