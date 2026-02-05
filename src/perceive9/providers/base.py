"""Base provider interfaces for diarization and transcription."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from perceive9.config import Language


@dataclass
class DiarizationSegment:
    """A segment of speech attributed to a speaker."""

    speaker_label: str  # SPEAKER_00, SPEAKER_01, etc.
    start_time: float
    end_time: float
    confidence: Optional[float] = None


@dataclass
class DiarizationResult:
    """Result from diarization processing."""

    segments: List[DiarizationSegment]
    model_name: str
    raw_response: Optional[dict] = None


@dataclass
class WordTimestamp:
    """Word-level timestamp."""

    word: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text."""

    start_time: float
    end_time: float
    text: str
    confidence: Optional[float] = None
    words: List[WordTimestamp] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Result from transcription processing."""

    segments: List[TranscriptionSegment]
    full_text: str
    model_name: str
    language: str
    raw_response: Optional[dict] = None


class DiarizationProviderInterface(ABC):
    """Interface for diarization providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'pyannote', 'replicate')."""
        pass

    @abstractmethod
    async def diarize(
        self,
        audio_path: str,
        language: Language,
        model_name: Optional[str] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to the audio file
            language: Language of the audio
            model_name: Optional specific model to use

        Returns:
            DiarizationResult with speaker segments
        """
        pass

    @abstractmethod
    async def get_speaker_embedding(
        self,
        audio_path: str,
        model_name: Optional[str] = None,
    ) -> List[float]:
        """
        Extract speaker embedding from audio sample.

        Args:
            audio_path: Path to audio file containing single speaker
            model_name: Optional specific model to use

        Returns:
            Speaker embedding vector
        """
        pass


class TranscriptionProviderInterface(ABC):
    """Interface for transcription providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'openai_whisper', 'replicate')."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        language: Language,
        model_name: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to the audio file
            language: Language of the audio
            model_name: Optional specific model to use

        Returns:
            TranscriptionResult with text segments and word timestamps
        """
        pass
