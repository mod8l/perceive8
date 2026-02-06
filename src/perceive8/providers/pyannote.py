"""Pyannote.ai provider for diarization and speaker embeddings."""

import asyncio
import base64
import logging
import struct
import uuid
from typing import List, Optional

import httpx

from perceive8.config import Language
from perceive8.providers.base import (
    DiarizationProviderInterface,
    DiarizationResult,
    DiarizationSegment,
)

logger = logging.getLogger(__name__)


class PyannoteProvider(DiarizationProviderInterface):
    """Pyannote.ai API provider for speaker diarization and voiceprint embeddings."""

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
                timeout=300.0,
            )
        return self._client

    async def _upload_audio(self, audio_path: str) -> str:
        """Upload audio to pyannote temporary storage and return media:// URL."""
        client = await self._get_client()
        media_key = f"media://{uuid.uuid4()}.wav"

        # Step 1: Create media input to get presigned upload URL
        resp = await client.post("/media/input", json={"url": media_key})
        resp.raise_for_status()
        upload_url = resp.json()["url"]

        # Step 2: Upload file to presigned S3 URL
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        async with httpx.AsyncClient(timeout=120.0) as upload_client:
            put_resp = await upload_client.put(
                upload_url,
                content=audio_bytes,
                headers={"Content-Type": "audio/wav"},
            )
            put_resp.raise_for_status()

        return media_key

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

        # Upload audio and get media URL
        media_url = await self._upload_audio(audio_path)

        # Submit diarization job
        response = await client.post("/diarize", json={"url": media_url})
        logger.debug("Pyannote diarize response status=%s", response.status_code)

        if response.status_code in (200, 201):
            result = response.json()
            # If job is async (201), poll for completion
            job_id = result.get("jobId")
            if job_id:
                result = await self._poll_job(client, job_id)
        else:
            response.raise_for_status()
            result = {}

        # Parse response into segments
        segments = []
        output = result.get("output", [])
        # The output may contain nested structures
        if isinstance(output, dict) and "diarization" in output:
            output = output["diarization"]
        for segment in output:
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

    async def _poll_job(self, client: httpx.AsyncClient, job_id: str, max_wait: int = 300) -> dict:
        """Poll a pyannote job until completion."""
        for _ in range(max_wait // 2):
            await asyncio.sleep(2)
            resp = await client.get(f"/jobs/{job_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            logger.debug("Pyannote job %s status=%s", job_id, status)
            if status == "succeeded":
                return data
            elif status == "failed":
                raise RuntimeError(f"Pyannote job {job_id} failed: {data}")
        raise TimeoutError(f"Pyannote job {job_id} did not complete within {max_wait}s")

    async def get_speaker_embedding(
        self,
        audio_path: str,
        model_name: Optional[str] = None,
    ) -> List[float]:
        """
        Extract speaker voiceprint (embedding) using pyannote.ai /voiceprint endpoint.

        See: https://docs.pyannote.ai/tutorials/identification-with-voiceprints

        Args:
            audio_path: Path to audio with single speaker
            model_name: Unused, kept for interface compatibility

        Returns:
            Speaker embedding vector (voiceprint)
        """
        client = await self._get_client()

        media_url = await self._upload_audio(audio_path)

        # Submit voiceprint job
        response = await client.post("/voiceprint", json={"url": media_url})
        logger.debug("Pyannote voiceprint response status=%s", response.status_code)

        if response.status_code in (200, 201):
            result = response.json()
            # If async job, poll for completion
            job_id = result.get("jobId")
            if job_id:
                result = await self._poll_job(client, job_id)
        else:
            response.raise_for_status()
            result = {}

        # The voiceprint is returned in the "output" field as base64-encoded float32 array
        output = result.get("output", result)
        if isinstance(output, dict):
            voiceprint = output.get("voiceprint", output.get("embedding", ""))
        elif isinstance(output, str):
            voiceprint = output
        elif isinstance(output, list):
            # Already decoded as list of floats
            if output and isinstance(output[0], (int, float)):
                return [float(v) for v in output]
            voiceprint = output[0] if output else ""
        else:
            return []

        if isinstance(voiceprint, str) and voiceprint:
            return self._decode_base64_embedding(voiceprint)
        if isinstance(voiceprint, list):
            return [float(v) for v in voiceprint]
        return []

    @staticmethod
    def _decode_base64_embedding(b64_string: str) -> List[float]:
        """Decode a base64-encoded float32 embedding vector."""
        raw_bytes = base64.b64decode(b64_string)
        num_floats = len(raw_bytes) // 4
        return list(struct.unpack(f"<{num_floats}f", raw_bytes))

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
