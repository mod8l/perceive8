"""OpenAI Whisper provider for transcription."""

from typing import Optional

from openai import AsyncOpenAI

from perceive8.config import Language
from perceive8.providers.base import (
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
            # seg may be a Pydantic model or a dict depending on SDK version
            seg_data = seg if isinstance(seg, dict) else (seg.model_dump() if hasattr(seg, "model_dump") else seg.__dict__)
            for word_data in seg_data.get("words", []):
                if isinstance(word_data, dict):
                    wd = word_data
                else:
                    wd = word_data.model_dump() if hasattr(word_data, "model_dump") else word_data.__dict__
                words.append(
                    WordTimestamp(
                        word=wd["word"],
                        start_time=wd["start"],
                        end_time=wd["end"],
                        confidence=wd.get("probability"),
                    )
                )

            segments.append(
                TranscriptionSegment(
                    start_time=seg_data["start"],
                    end_time=seg_data["end"],
                    text=seg_data["text"],
                    confidence=seg_data.get("avg_logprob"),
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
