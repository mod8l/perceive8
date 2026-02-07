"""Speaker management endpoints."""

import logging
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from perceive8.config import get_settings
from perceive8.database import get_db
from perceive8.models.database import Speaker, User
from perceive8.providers.pyannote import PyannoteProvider
from perceive8.services.embedding import EmbeddingService
from perceive8.services.preprocessing import preprocess_audio

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


def _get_embedding_service(request: Request) -> EmbeddingService:
    """Extract EmbeddingService from app state."""
    svc = getattr(request.app.state, "embedding_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Embedding service not available")
    return svc


async def _get_or_create_user(db: AsyncSession, external_id: str) -> User:
    """Get or create a user by external_id."""
    result = await db.execute(select(User).where(User.external_id == external_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(external_id=external_id)
        db.add(user)
        await db.flush()
    return user


@router.post("")
async def enroll_speaker(
    request: Request,
    user_id: str = Form(...),
    name: str = Form(...),
    voice_sample: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Enroll a new speaker with a voice sample."""
    embedding_service = _get_embedding_service(request)

    # Read uploaded audio
    audio_bytes = await voice_sample.read()

    # Write to temp file for preprocessing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Preprocess audio
        import os
        tmp_dir = os.path.dirname(tmp_path)
        preprocessing_result = await preprocess_audio(tmp_path, tmp_dir)
        processed_path = preprocessing_result.output_path

        # Extract embedding
        provider = PyannoteProvider(api_key=settings.pyannote_api_key)
        try:
            embedding = await provider.get_speaker_embedding(processed_path)
        finally:
            await provider.close()

        # Generate ID upfront so ChromaDB and PG share the same key
        speaker_id = uuid.uuid4()
        speaker_chromadb_id = str(speaker_id)

        # Write to ChromaDB first
        embedding_service.add_speaker_embedding(
            speaker_id=speaker_chromadb_id,
            embedding=embedding,
            metadata={"user_id": user_id, "name": name, "speaker_id": speaker_chromadb_id},
        )

        # Then write to PostgreSQL
        try:
            user = await _get_or_create_user(db, user_id)

            speaker = Speaker(
                id=speaker_id,
                user_id=user.id,
                name=name,
                chromadb_id=speaker_chromadb_id,
            )
            db.add(speaker)
            await db.flush()
        except Exception:
            # Compensating rollback: remove from ChromaDB if PG write fails
            logger.warning("PG write failed for speaker %s, rolling back ChromaDB", speaker_chromadb_id)
            try:
                embedding_service.delete_speaker_embedding(speaker_chromadb_id)
            except Exception:
                logger.error("Failed to rollback ChromaDB for speaker %s", speaker_chromadb_id)
            raise

        return {
            "id": str(speaker.id),
            "name": speaker.name,
            "user_id": user_id,
            "created_at": speaker.created_at.isoformat() if speaker.created_at else None,
        }
    finally:
        # Clean up temp files
        import os
        for p in [tmp_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


@router.get("")
async def list_speakers(
    user_id: str,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List speakers for a user."""
    result = await db.execute(
        select(Speaker)
        .join(User, Speaker.user_id == User.id)
        .where(User.external_id == user_id)
        .limit(limit)
        .offset(offset)
    )
    speakers = result.scalars().all()

    count_result = await db.execute(
        select(Speaker.id)
        .join(User, Speaker.user_id == User.id)
        .where(User.external_id == user_id)
    )
    total = len(count_result.all())

    return {
        "speakers": [
            {
                "id": str(s.id),
                "name": s.name,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in speakers
        ],
        "total": total,
    }


@router.get("/{speaker_id}")
async def get_speaker(
    speaker_id: uuid.UUID,
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get speaker details by ID (scoped to user_id)."""
    result = await db.execute(
        select(Speaker)
        .join(User, Speaker.user_id == User.id)
        .where(Speaker.id == speaker_id, User.external_id == user_id)
    )
    speaker = result.scalar_one_or_none()
    if speaker is None:
        raise HTTPException(status_code=404, detail="Speaker not found")

    return {
        "id": str(speaker.id),
        "name": speaker.name,
        "chromadb_id": speaker.chromadb_id,
        "created_at": speaker.created_at.isoformat() if speaker.created_at else None,
    }


@router.delete("/{speaker_id}")
async def delete_speaker(
    speaker_id: uuid.UUID,
    user_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Delete a speaker and their embedding (scoped to user_id)."""
    embedding_service = _get_embedding_service(request)

    result = await db.execute(
        select(Speaker)
        .join(User, Speaker.user_id == User.id)
        .where(Speaker.id == speaker_id, User.external_id == user_id)
    )
    speaker = result.scalar_one_or_none()
    if speaker is None:
        raise HTTPException(status_code=404, detail="Speaker not found")

    # Delete from ChromaDB
    if speaker.chromadb_id:
        embedding_service.delete_speaker_embedding(speaker.chromadb_id)

    # Delete from PostgreSQL
    await db.delete(speaker)

    return {"message": "Speaker deleted", "id": str(speaker_id)}
