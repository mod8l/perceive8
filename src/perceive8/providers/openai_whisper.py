"""OpenAI Whisper provider for transcription."""

import logging
import os
import tempfile
from typing import List, Optional, Tuple

from openai import AsyncOpenAI
from pydub import AudioSegment
from pydub.silence import detect_silence

from perceive8.config import Language
from perceive8.providers.base import (
    TranscriptionProviderInterface,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)

logger = logging.getLogger(__name__)

# 24MB limit to leave margin under OpenAI's 25MB cap
MAX_FILE_SIZE = 24 * 1024 * 1024

# Chunk duration: 10 minutes in milliseconds
MAX_CHUNK_DURATION_MS = 10 * 60 * 1000

# Overlap for fallback fixed splitting (2 seconds)
OVERLAP_MS = 2000

# Silence detection parameters
MIN_SILENCE_LEN_MS = 300
SILENCE_THRESH_DBFS = -40

# Language code mapping for Whisper
LANGUAGE_MAP = {
    Language.ENGLISH: "en",
    Language.HEBREW: "he",
    Language.SPANISH: "es",
    "auto": None,
}


def _find_silence_split_points(audio: AudioSegment) -> List[int]:
    """Find optimal split points at silence boundaries to keep chunks under ~10 minutes.

    Returns a list of split points (in ms) where the audio should be divided.
    Does NOT include 0 or the total duration — only interior split points.
    """
    total_duration_ms = len(audio)
    if total_duration_ms <= MAX_CHUNK_DURATION_MS:
        return []

    silence_ranges = detect_silence(
        audio, min_silence_len=MIN_SILENCE_LEN_MS, silence_thresh=SILENCE_THRESH_DBFS
    )
    # Each range is [start_ms, end_ms] of silence. Use midpoints as candidate split points.
    silence_midpoints = [(s + e) // 2 for s, e in silence_ranges]

    split_points: List[int] = []
    chunk_start = 0

    while chunk_start + MAX_CHUNK_DURATION_MS < total_duration_ms:
        target = chunk_start + MAX_CHUNK_DURATION_MS
        # Find the silence midpoint closest to the target but not exceeding it
        best = None
        for mid in silence_midpoints:
            if mid <= chunk_start:
                continue
            if mid > target:
                break
            best = mid

        if best is not None:
            split_points.append(best)
            chunk_start = best
        else:
            # No silence found — use None as sentinel to indicate fallback
            split_points.append(None)  # type: ignore[arg-type]
            chunk_start = target

    return split_points


def _build_chunks_from_split_points(
    split_points: List[Optional[int]], total_duration_ms: int
) -> List[Tuple[int, int, bool]]:
    """Build (start_ms, end_ms, uses_overlap) chunk tuples from split points.

    A ``None`` split point means no silence was found, so we fall back to a
    fixed 10-minute cut with 2-second overlap into the next chunk.
    """
    chunks: List[Tuple[int, int, bool]] = []
    prev = 0

    for sp in split_points:
        if sp is not None:
            # Silence-based split — no overlap needed
            chunks.append((prev, sp, False))
            prev = sp
        else:
            # Fallback fixed split
            end = prev + MAX_CHUNK_DURATION_MS
            # Extend this chunk by OVERLAP_MS so overlap region exists
            chunks.append((prev, min(end + OVERLAP_MS, total_duration_ms), True))
            prev = end  # next chunk starts at the non-overlapped boundary

    # Final chunk
    if prev < total_duration_ms:
        chunks.append((prev, total_duration_ms, False))

    return chunks


def _export_chunk(
    chunk_audio: AudioSegment, chunk_path_base: str
) -> Tuple[str, str]:
    """Export a chunk as FLAC; fall back to OGG/Opus if FLAC exceeds 24MB.

    Returns (file_path, format_used).
    """
    flac_path = chunk_path_base + ".flac"
    chunk_audio.export(flac_path, format="flac")

    if os.path.getsize(flac_path) <= MAX_FILE_SIZE:
        return flac_path, "flac"

    # FLAC too large — fall back to OGG/Opus
    os.remove(flac_path)
    ogg_path = chunk_path_base + ".ogg"
    chunk_audio.export(ogg_path, format="ogg", codec="opus", bitrate="96k")
    return ogg_path, "ogg/opus"


def _deduplicate_overlap(
    prev_segments: List[TranscriptionSegment],
    curr_segments: List[TranscriptionSegment],
    overlap_boundary_sec: float,
) -> List[TranscriptionSegment]:
    """Remove words/segments from curr_segments that fall within the previous chunk's time range.

    ``overlap_boundary_sec`` is the absolute time where the previous chunk's
    non-overlapped content ends (i.e. where the overlap region begins in absolute time).
    Words in curr_segments whose start_time < overlap_boundary_sec + OVERLAP_MS/1000
    AND whose start_time >= overlap_boundary_sec are considered duplicates from the
    overlap region and are dropped.
    """
    overlap_end_sec = overlap_boundary_sec + OVERLAP_MS / 1000.0
    cleaned: List[TranscriptionSegment] = []

    for seg in curr_segments:
        # If entire segment is before the overlap end, filter at word level
        if seg.start_time < overlap_end_sec:
            kept_words = [w for w in seg.words if w.start_time >= overlap_end_sec]
            if not kept_words:
                # All words in overlap region — drop entire segment
                continue
            # Rebuild segment from kept words
            cleaned.append(
                TranscriptionSegment(
                    start_time=kept_words[0].start_time,
                    end_time=seg.end_time,
                    text=" ".join(w.word.strip() for w in kept_words),
                    confidence=seg.confidence,
                    words=kept_words,
                )
            )
        else:
            cleaned.append(seg)

    return cleaned


class OpenAIWhisperProvider(TranscriptionProviderInterface):
    """OpenAI Whisper API provider for transcription."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai_whisper"

    async def _transcribe_single_file(
        self,
        audio_path: str,
        model: str,
        lang_code: Optional[str],
    ) -> dict:
        """Transcribe a single audio file and return parsed segments and text."""
        kwargs = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word", "segment"],
        }
        if lang_code is not None:
            kwargs["language"] = lang_code
        with open(audio_path, "rb") as f:
            kwargs["file"] = f
            response = await self.client.audio.transcriptions.create(**kwargs)

        segments = []
        for seg in response.segments or []:
            words = []
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

        return {
            "segments": segments,
            "text": response.text,
            "raw_response": response.model_dump() if hasattr(response, "model_dump") else None,
        }

    async def transcribe(
        self,
        audio_path: str,
        language: Language,
        model_name: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using OpenAI Whisper API.

        Large files (>24MB) are automatically split into chunks using
        silence detection, then transcribed separately and merged with
        adjusted timestamps.

        Args:
            audio_path: Path to audio file
            language: Language of audio
            model_name: Model version (default: whisper-1)

        Returns:
            TranscriptionResult with text and word timestamps
        """
        model = model_name or "whisper-1"
        lang_code = LANGUAGE_MAP.get(language, "en") if language else None
        logger.info(f"Language: {lang_code or 'auto-detect'}")

        file_size = os.path.getsize(audio_path)

        if file_size <= MAX_FILE_SIZE:
            result = await self._transcribe_single_file(audio_path, model, lang_code)
            return TranscriptionResult(
                segments=result["segments"],
                full_text=result["text"],
                model_name=model,
                language=lang_code,
                raw_response=result["raw_response"],
            )

        # Large file: load and split
        logger.info(
            "Audio file is %.1fMB, loading for chunked transcription...",
            file_size / (1024 * 1024),
        )
        audio = AudioSegment.from_file(audio_path)
        total_duration_ms = len(audio)

        # Find silence-based split points
        split_points = _find_silence_split_points(audio)

        if not split_points:
            # Entire audio fits in one chunk after loading (edge case: file was
            # large on disk but short duration). Transcribe directly.
            result = await self._transcribe_single_file(audio_path, model, lang_code)
            return TranscriptionResult(
                segments=result["segments"],
                full_text=result["text"],
                model_name=model,
                language=lang_code,
                raw_response=result["raw_response"],
            )

        chunks = _build_chunks_from_split_points(split_points, total_duration_ms)
        total_chunks = len(chunks)

        # Log splitting strategy
        silence_splits = sum(1 for _, _, overlap in chunks if not overlap)
        fallback_splits = sum(1 for _, _, overlap in chunks if overlap)
        logger.info(
            "Split into %d chunks (%d at silence, %d fallback with overlap)",
            total_chunks,
            silence_splits,
            fallback_splits,
        )

        all_segments: List[TranscriptionSegment] = []
        all_texts: List[str] = []

        tmpdir = tempfile.mkdtemp()
        try:
            for i, (chunk_start_ms, chunk_end_ms, uses_overlap) in enumerate(chunks):
                chunk_audio = audio[chunk_start_ms:chunk_end_ms]
                chunk_duration_sec = (chunk_end_ms - chunk_start_ms) / 1000.0

                chunk_path_base = os.path.join(tmpdir, f"chunk_{i}")
                chunk_path, fmt = _export_chunk(chunk_audio, chunk_path_base)

                logger.info(
                    "Transcribing chunk %d/%d: %.1fs, format=%s, overlap=%s",
                    i + 1,
                    total_chunks,
                    chunk_duration_sec,
                    fmt,
                    uses_overlap,
                )

                result = await self._transcribe_single_file(chunk_path, model, lang_code)

                # Adjust timestamps by chunk offset
                offset_sec = chunk_start_ms / 1000.0
                for seg in result["segments"]:
                    seg.start_time += offset_sec
                    seg.end_time += offset_sec
                    for w in seg.words:
                        w.start_time += offset_sec
                        w.end_time += offset_sec

                # Deduplicate overlap region if this chunk uses overlap
                if uses_overlap and i > 0:
                    # The overlap boundary is where the previous chunk's
                    # non-overlapped content ends = chunk_start_ms in absolute time
                    result["segments"] = _deduplicate_overlap(
                        all_segments, result["segments"], offset_sec
                    )

                all_segments.extend(result["segments"])
                all_texts.append(result["text"])

                # Clean up chunk file
                os.remove(chunk_path)
        finally:
            try:
                os.rmdir(tmpdir)
            except OSError:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Rebuild full text from segments to account for deduplication
        full_text = " ".join(seg.text.strip() for seg in all_segments)

        return TranscriptionResult(
            segments=all_segments,
            full_text=full_text,
            model_name=model,
            language=lang_code,
            raw_response=None,
        )
