"""Pyannote.ai provider for diarization and speaker embeddings."""

from typing import List, Optional

import httpx

from perceive8.config import Language
from perceive8.providers.base import (
    DiarizationProviderInterface,
    DiarizationResult,
    DiarizationSegment,
)


class PyannoteProvider(DiarizationProviderInterface):
    """Pyannote.ai API provider for speaker diarization and embeddings."""

    BASE_URL = "https://api.pyannote.ai/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def provider_name(self) -> str:
        return "pyannote"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=300.0,  # 5 minutes for long audio
            )
        return self._client

    async def diarize(
        self,
        audio_path: str,
        language: Language,
        model_name: Optional[str] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization using pyannote.ai API.

        Args:
            audio_path: Path to audio file
            language: Language of audio (used for hints)
            model_name: Model version (default: latest)

        Returns:
            DiarizationResult with speaker segments
        """
        client = await self._get_client()
        model = model_name or "pyannote/speaker-diarization-3.1"

        # Upload and process audio
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            data = {"model": model}

            response = await client.post("/diarize", files=files, data=data)
            response.raise_for_status()
            result = response.json()

        # Parse response into segments
        segments = []
        for segment in result.get("output", []):
            segments.append(
                DiarizationSegment(
                    speaker_label=segment["speaker"],
                    start_time=segment["start"],
                    end_time=segment["end"],
                    confidence=segment.get("confidence"),
                )
            )

        return DiarizationResult(
            segments=segments,
            model_name=model,
            raw_response=result,
        )

    async def get_speaker_embedding(
        self,
        audio_path: str,
        model_name: Optional[str] = None,
    ) -> List[float]:
        """
        Extract speaker embedding from audio sample.

        Args:
            audio_path: Path to audio with single speaker
            model_name: Embedding model version

        Returns:
            Speaker embedding vector
        """
        client = await self._get_client()
        model = model_name or "pyannote/embedding"

        with open(audio_path, "rb") as f:
            files = {"audio": f}
            data = {"model": model}

            response = await client.post("/embed", files=files, data=data)
            response.raise_for_status()
            result = response.json()

        return result.get("embedding", [])

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
