"""End-to-end integration tests using real pyannote and OpenAI APIs.

Runs the full analysis pipeline against tests/fixtures/sample.wav and prints
a detailed analysis report.  Execute with: make test-integration
"""

import os
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pytest

from perceive8.config import DiarizationProvider, Language, TranscriptionProvider
from perceive8.services.pipeline import run_analysis_pipeline

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")
SAMPLE_AUDIO = Path(__file__).parent / "fixtures" / "sample.wav"

skip_no_keys = pytest.mark.skipif(
    not OPENAI_API_KEY or not PYANNOTE_API_KEY,
    reason="OPENAI_API_KEY and/or PYANNOTE_API_KEY not set",
)
skip_no_audio = pytest.mark.skipif(
    not SAMPLE_AUDIO.exists(), reason="tests/fixtures/sample.wav not found"
)


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFullPipeline:
    """Full end-to-end pipeline integration test."""

    @skip_no_keys
    @skip_no_audio
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, monkeypatch):
        """Run the complete analysis pipeline and print a detailed report."""

        # Use a temp directory for audio storage
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Monkeypatch settings so storage writes to temp dir
            from perceive8.config import get_settings

            original_settings = get_settings()

            class _PatchedSettings:
                """Proxy that overrides audio_storage_path."""

                def __getattr__(self, name):
                    if name == "audio_storage_path":
                        return tmp_dir
                    return getattr(original_settings, name)

            patched = _PatchedSettings()
            monkeypatch.setattr("perceive8.config.get_settings", lambda: patched)
            monkeypatch.setattr("perceive8.services.storage.get_settings", lambda: patched)
            monkeypatch.setattr("perceive8.providers.factory.get_settings", lambda: patched)

            audio_data = SAMPLE_AUDIO.read_bytes()
            analysis_id = str(uuid.uuid4())

            result = await run_analysis_pipeline(
                audio_data=audio_data,
                filename="sample.wav",
                user_id="test-user",
                analysis_id=analysis_id,
                language=Language.ENGLISH,
                diarization_provider=DiarizationProvider.PYANNOTE,
                transcription_providers=[TranscriptionProvider.OPENAI_WHISPER],
            )

            # ---------------------------------------------------------------
            # Assertions
            # ---------------------------------------------------------------
            assert result.original_path
            assert result.preprocessing_info is not None
            assert result.quality_info is not None
            assert result.diarization_result is not None
            assert len(result.diarization_result.segments) > 0
            assert len(result.transcription_results) > 0
            assert result.transcription_results[0].full_text
            assert len(result.merged_segments) > 0

            # ---------------------------------------------------------------
            # Detailed report
            # ---------------------------------------------------------------
            pp = result.preprocessing_info
            qi = result.quality_info

            print("\n" + "=" * 72)
            print("  DETAILED ANALYSIS REPORT")
            print("=" * 72)

            # Preprocessing
            print("\n--- Preprocessing Info ---")
            print(f"  Format        : {pp.original_format}")
            print(f"  Duration      : {pp.duration_seconds:.2f}s")
            print(f"  Sample Rate   : {pp.sample_rate} Hz")
            print(f"  Was Converted : {pp.was_converted}")

            # Audio quality
            print("\n--- Audio Quality ---")
            print(f"  SNR (dB)          : {qi.snr_db:.1f}")
            print(f"  Needs Enhancement : {qi.needs_enhancement}")

            # Enhancement
            print("\n--- Enhancement ---")
            print(f"  Was Enhanced  : {result.was_enhanced}")
            if result.was_enhanced:
                print(f"  Enhanced Path : {result.enhanced_path}")

            # Diarization
            diar = result.diarization_result
            print(f"\n--- Diarization ({len(diar.segments)} segments) ---")
            for i, seg in enumerate(diar.segments):
                print(
                    f"  [{i:3d}] {_fmt_time(seg.start_time)} -> {_fmt_time(seg.end_time)}  "
                    f"{seg.speaker_label}"
                )

            # Transcription
            for tr in result.transcription_results:
                print(f"\n--- Transcription [{tr.model_name}] ({len(tr.segments)} segments) ---")
                print(f"  Model     : {tr.model_name}")
                print(f"  Full Text : {tr.full_text[:200]}{'...' if len(tr.full_text) > 200 else ''}")
                for i, seg in enumerate(tr.segments):
                    print(
                        f"  [{i:3d}] {_fmt_time(seg.start_time)} -> {_fmt_time(seg.end_time)}  "
                        f"{seg.text}"
                    )

            # Merged results
            print(f"\n--- Merged Segments ({len(result.merged_segments)}) ---")
            for i, seg in enumerate(result.merged_segments):
                print(
                    f"  [{i:3d}] {_fmt_time(seg.start_time)} -> {_fmt_time(seg.end_time)}  "
                    f"{seg.speaker_label}: {seg.text}"
                )

            # Final conversation transcript
            print("\n" + "=" * 72)
            print("  CONVERSATION TRANSCRIPT")
            print("=" * 72 + "\n")
            for seg in result.merged_segments:
                print(f"  [{_fmt_time(seg.start_time)}] {seg.speaker_label}: {seg.text}")

            print("\n" + "=" * 72)
            print("  END OF REPORT")
            print("=" * 72 + "\n")
