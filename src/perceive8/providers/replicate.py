"""Replicate provider for diarization and transcription."""

from typing import List, Optional

import replicate

from perceive8.config import Language
from perceive8.providers.base import (
    DiarizationProviderInterface,
    DiarizationResult,
    DiarizationSegment,
    TranscriptionProviderInterface,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


# Language mapping for Replicate models
LANGUAGE_MAP = {
    Language.ENGLISH: "en",
    Language.HEBREW: "he",
    Language.SPANISH: "es",
}


class ReplicateDiarizationProvider(DiarizationProviderInterface):
    """Replicate provider for speaker diarization."""

    # Default models available on Replicate
    DEFAULT_MODEL = "pyannote/pyannote-audio:latest"

    def __init__(self, api_token: str):
        self.client = replicate.Client(api_token=api_token)

    @property
    def provider_name(self) -> str:
        return "replicate"

    async def diarize(
        self,
        audio_path: str,
        language: Language,
        model_name: Optional[str] = None,
    ) -> DiarizationResult:
        """
        Perform diarization using Replicate-hosted models.

        Args:
            audio_path: Path to audio file
            language: Language of audio
            model_name: Replicate model identifier

        Returns:
            DiarizationResult with speaker segments
        """
        model = model_name or self.DEFAULT_MODEL

        # Run model on Replicate
        with open(audio_path, "rb") as f:
            output = self.client.run(
                model,
                input={"audio": f},
            )

        # Parse response (format varies by model)
        segments = []
        for item in output if isinstance(output, list) else output.get("segments", []):
            segments.append(
                DiarizationSegment(
                    speaker_label=item.get("speaker", item.get("label", "SPEAKER_00")),
                    start_time=item.get("start", item.get("start_time", 0)),
                    end_time=item.get("end", item.get("end_time", 0)),
                    confidence=item.get("confidence"),
                )
            )

        return DiarizationResult(
            segments=segments,
            model_name=model,
            raw_response={"output": output},
        )

    async def get_speaker_embedding(
        self,
        audio_path: str,
        model_name: Optional[str] = None,
    ) -> List[float]:
        """
        Extract speaker embedding using Replicate.

        Args:
            audio_path: Path to audio file
            model_name: Replicate model identifier

        Returns:
            Speaker embedding vector
        """
        model = model_name or "pyannote/pyannote-audio-embedding:latest"

        with open(audio_path, "rb") as f:
            output = self.client.run(
                model,
                input={"audio": f},
            )

        return output.get("embedding", output if isinstance(output, list) else [])


class ReplicateTranscriptionProvider(TranscriptionProviderInterface):
    """Replicate provider for transcription using Whisper models."""

    # Available Whisper models on Replicate
    DEFAULT_MODEL = "openai/whisper:latest"
    MODELS = {
        "whisper-large-v3": "openai/whisper:large-v3",
        "whisper-large-v2": "openai/whisper:large-v2",
        "incredibly-fast-whisper": "vaibhavs10/incredibly-fast-whisper:latest",
    }

    def __init__(self, api_token: str):
        self.client = replicate.Client(api_token=api_token)

    @property
    def provider_name(self) -> str:
        return "replicate"

    async def transcribe(
        self,
        audio_path: str,
        language: Language,
        model_name: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Replicate-hosted Whisper models.

        Args:
            audio_path: Path to audio file
            language: Language of audio
            model_name: Model name or Replicate model identifier

        Returns:
            TranscriptionResult with text and timestamps
        """
        # Resolve model name
        if model_name in self.MODELS:
            model = self.MODELS[model_name]
        else:
            model = model_name or self.DEFAULT_MODEL

        lang_code = LANGUAGE_MAP.get(language, "en")

        with open(audio_path, "rb") as f:
            output = self.client.run(
                model,
                input={
                    "audio": f,
                    "language": lang_code,
                    "word_timestamps": True,
                },
            )

        # Parse response
        segments = []
        full_text = ""

        # Handle different output formats from various Whisper models
        if isinstance(output, dict):
            full_text = output.get("transcription", output.get("text", ""))
            raw_segments = output.get("segments", [])
        else:
            full_text = str(output)
            raw_segments = []

        for seg in raw_segments:
            words = []
            for word_data in seg.get("words", []):
                words.append(
                    WordTimestamp(
                        word=word_data.get("word", ""),
                        start_time=word_data.get("start", 0),
                        end_time=word_data.get("end", 0),
                        confidence=word_data.get("probability"),
                    )
                )

            segments.append(
                TranscriptionSegment(
                    start_time=seg.get("start", 0),
                    end_time=seg.get("end", 0),
                    text=seg.get("text", ""),
                    confidence=seg.get("avg_logprob"),
                    words=words,
                )
            )

        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            model_name=model,
            language=lang_code,
            raw_response={"output": output},
        )
