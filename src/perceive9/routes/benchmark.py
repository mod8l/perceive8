"""Benchmark endpoints for provider comparison."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from perceive9.config import (
    DiarizationProvider,
    Language,
    TranscriptionProvider,
)

router = APIRouter()


@router.post("")
async def create_benchmark(
    user_id: str = Form(...),
    audio_file: UploadFile = File(...),
    language: Language = Form(...),
    diarization_providers: List[DiarizationProvider] = Form(...),
    transcription_providers: List[TranscriptionProvider] = Form(...),
):
    """
    Run same audio through multiple providers for comparison.

    Creates an analysis with multiple processing_runs, one per provider/model combo.
    """
    # TODO: Implement benchmark
    # 1. Create analysis record
    # 2. Run each diarization provider (parallel or sequential)
    # 3. Run each transcription provider
    # 4. Store results with timing metrics

    return {
        "message": "Benchmark not yet implemented",
        "config": {
            "language": language,
            "diarization_providers": diarization_providers,
            "transcription_providers": transcription_providers,
        },
    }


@router.get("/{benchmark_id}")
async def get_benchmark(benchmark_id: UUID):
    """Get benchmark results with comparison metrics."""
    # TODO: Aggregate results across processing_runs
    raise HTTPException(status_code=404, detail="Benchmark not found")


@router.get("/{benchmark_id}/compare")
async def compare_benchmark_results(benchmark_id: UUID):
    """
    Compare results from different providers.

    Returns side-by-side comparison of transcripts and timing metrics.
    """
    # TODO: Build comparison view
    return {"comparison": {}}
