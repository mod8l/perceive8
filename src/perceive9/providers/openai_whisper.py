"""OpenAI Whisper provider for transcription."""

from typing import Optional

from openai import AsyncOpenAI

from perceive9.config import Language
from perceive9.providers.base import (
    TranscriptionProviderInterface,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


# Language code mapping for Whisper
LANGUAGE_MAP = {
    Language.ENGLISH: "en",
    Language.HEBREW: "he",
    Language.SPANISH: "es",
}


class OpenAIWhisperProvider(TranscriptionProviderInterface):
    """OpenAI Whisper API provider for transcription."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai_whisper"

    async def transcribe(
        self,
        audio_path: str,
        language: Language,
        model_name: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using OpenAI Whisper API.

        Args:
            audio_path: Path to audio file
            language: Language of audio
            model_name: Model version (default: whisper-1)

        Returns:
            TranscriptionResult with text and word timestamps
        """
        model = model_name or "whisper-1"
        lang_code = LANGUAGE_MAP.get(language, "en")

        with open(audio_path, "rb") as f:
            # Request verbose JSON to get word timestamps
            response = await self.client.audio.transcriptions.create(
                model=model,
                file=f,
                language=lang_code,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )

        # Parse response into segments
        segments = []
        for seg in response.segments or []:
            words = []
            for word_data in seg.get("words", []):
                words.append(
                    WordTimestamp(
                        word=word_data["word"],
                        start_time=word_data["start"],
                        end_time=word_data["end"],
                        confidence=word_data.get("probability"),
                    )
                )

            segments.append(
                TranscriptionSegment(
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"],
                    confidence=seg.get("avg_logprob"),
                    words=words,
                )
            )

        return TranscriptionResult(
            segments=segments,
            full_text=response.text,
            model_name=model,
            language=lang_code,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )
