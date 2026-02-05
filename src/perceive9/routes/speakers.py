"""Speaker management endpoints."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter()


@router.post("")
async def enroll_speaker(
    user_id: str = Form(...),
    name: str = Form(...),
    voice_sample: UploadFile = File(...),
):
    """Enroll a new speaker with a voice sample."""
    # TODO: Implement speaker enrollment
    # 1. Save voice sample to storage
    # 2. Extract embedding via pyannote.ai
    # 3. Store embedding in ChromaDB
    # 4. Create speaker record in PostgreSQL
    return {"message": "Speaker enrollment not yet implemented"}


@router.get("")
async def list_speakers(user_id: str, limit: int = 100, offset: int = 0):
    """List speakers for a user."""
    # TODO: Query speakers from PostgreSQL
    return {"speakers": [], "total": 0}


@router.get("/{speaker_id}")
async def get_speaker(speaker_id: UUID):
    """Get speaker details by ID."""
    # TODO: Get speaker from PostgreSQL
    raise HTTPException(status_code=404, detail="Speaker not found")


@router.delete("/{speaker_id}")
async def delete_speaker(speaker_id: UUID):
    """Delete a speaker and their embedding."""
    # TODO: Delete from PostgreSQL and ChromaDB
    return {"message": "Speaker deletion not yet implemented"}
