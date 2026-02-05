"""SQLAlchemy database models."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    speakers: Mapped[list["Speaker"]] = relationship(back_populates="user")
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="user")


class Speaker(Base):
    __tablename__ = "speakers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    chromadb_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="speakers")


class AudioFile(Base):
    __tablename__ = "audio_files"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"), index=True)
    original_path: Mapped[str] = mapped_column(String(1024))
    enhanced_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    was_enhanced: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    analysis: Mapped["Analysis"] = relationship(back_populates="audio_file")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    language: Mapped[str] = mapped_column(String(10))  # en, he, es
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="analyses")
    audio_file: Mapped["AudioFile"] = relationship(back_populates="analysis", uselist=False)
    processing_runs: Mapped[list["ProcessingRun"]] = relationship(back_populates="analysis")


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"), index=True)
    run_type: Mapped[str] = mapped_column(String(50))  # diarization, transcription
    provider_name: Mapped[str] = mapped_column(String(100))  # pyannote, replicate, openai_whisper
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending, processing, completed, failed
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_response: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    analysis: Mapped["Analysis"] = relationship(back_populates="processing_runs")
    transcript_segments: Mapped[list["TranscriptSegment"]] = relationship(back_populates="processing_run")
    diarization_segments: Mapped[list["DiarizationSegment"]] = relationship(back_populates="processing_run")


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    processing_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("processing_runs.id"), index=True)
    speaker_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("speakers.id"), nullable=True)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    word_timestamps: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    chromadb_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    processing_run: Mapped["ProcessingRun"] = relationship(back_populates="transcript_segments")
    speaker: Mapped[Optional["Speaker"]] = relationship()


class DiarizationSegment(Base):
    __tablename__ = "diarization_segments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    processing_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("processing_runs.id"), index=True)
    speaker_label: Mapped[str] = mapped_column(String(50))  # SPEAKER_00, SPEAKER_01, etc.
    matched_speaker_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("speakers.id"), nullable=True)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    processing_run: Mapped["ProcessingRun"] = relationship(back_populates="diarization_segments")
    matched_speaker: Mapped[Optional["Speaker"]] = relationship()
