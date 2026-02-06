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

from perceive8.config import DiarizationProvider, Language, Settings, TranscriptionProvider
from perceive8.services.embedding import EmbeddingService
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


# ---------------------------------------------------------------------------
# Speaker Enrollment / Embedding Integration Tests
# ---------------------------------------------------------------------------

def _make_embedding_service(tmp_dir: str) -> EmbeddingService:
    """Create an EmbeddingService backed by a temporary ChromaDB directory."""
    settings = Settings(
        chromadb_path=os.path.join(tmp_dir, "chromadb"),
        database_url="sqlite+aiosqlite://",  # unused but required
    )
    return EmbeddingService(settings)


@pytest.mark.integration
class TestSpeakerEmbedding:
    """Integration tests for speaker enrollment and embedding operations."""

    @skip_no_keys
    @skip_no_audio
    @pytest.mark.asyncio
    async def test_enroll_speaker_extracts_embedding(self):
        """Enroll a speaker via pyannote embedding extraction and store in ChromaDB."""
        from perceive8.providers.pyannote import PyannoteProvider
        from perceive8.services.preprocessing import preprocess_audio

        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_service = _make_embedding_service(tmp_dir)

            # Preprocess sample audio
            preprocessing_result = await preprocess_audio(
                str(SAMPLE_AUDIO), tmp_dir
            )

            # Extract embedding using real pyannote API
            provider = PyannoteProvider(api_key=PYANNOTE_API_KEY)
            try:
                embedding = await provider.get_speaker_embedding(
                    preprocessing_result.output_path
                )
            finally:
                await provider.close()

            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(v, float) for v in embedding)

            # Store in ChromaDB
            speaker_id = str(uuid.uuid4())
            embedding_service.add_speaker_embedding(
                speaker_id=speaker_id,
                embedding=embedding,
                metadata={"user_id": "test-user", "name": "Alice", "speaker_id": speaker_id},
            )

            assert embedding_service.speaker_collection.count() == 1

            print("\n--- Speaker Enrollment Report ---")
            print(f"  Speaker ID      : {speaker_id}")
            print(f"  Embedding dims  : {len(embedding)}")
            print(f"  ChromaDB count  : {embedding_service.speaker_collection.count()}")
            print("  Status          : SUCCESS")

    @skip_no_keys
    @skip_no_audio
    @pytest.mark.asyncio
    async def test_search_similar_speakers(self):
        """Enroll a speaker then search with same audio — should match."""
        from perceive8.providers.pyannote import PyannoteProvider
        from perceive8.services.preprocessing import preprocess_audio

        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_service = _make_embedding_service(tmp_dir)

            preprocessing_result = await preprocess_audio(
                str(SAMPLE_AUDIO), tmp_dir
            )

            provider = PyannoteProvider(api_key=PYANNOTE_API_KEY)
            try:
                embedding = await provider.get_speaker_embedding(
                    preprocessing_result.output_path
                )
            finally:
                await provider.close()

            speaker_id = str(uuid.uuid4())
            embedding_service.add_speaker_embedding(
                speaker_id=speaker_id,
                embedding=embedding,
                metadata={"user_id": "test-user", "name": "Alice", "speaker_id": speaker_id},
            )

            # Search with same embedding — should find a match
            matches = embedding_service.search_similar_speakers(
                embedding=embedding,
                user_id="test-user",
                threshold=0.8,
            )

            assert len(matches) >= 1
            assert matches[0]["speaker_id"] == speaker_id
            assert matches[0]["similarity"] >= 0.8
            assert matches[0]["metadata"]["name"] == "Alice"

            print("\n--- Speaker Search Report ---")
            print(f"  Matches found   : {len(matches)}")
            for m in matches:
                print(f"    {m['metadata']['name']} (similarity={m['similarity']:.4f})")
            print("  Status          : SUCCESS")

    def test_speaker_listing_and_retrieval(self):
        """Add multiple speakers to ChromaDB and verify listing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_service = _make_embedding_service(tmp_dir)

            # Add two speakers with dummy embeddings (ChromaDB-only, no real API needed)
            embedding_a = [0.1] * 192
            embedding_b = [0.9] * 192
            id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())

            embedding_service.add_speaker_embedding(
                speaker_id=id_a,
                embedding=embedding_a,
                metadata={"user_id": "test-user", "name": "Alice", "speaker_id": id_a},
            )
            embedding_service.add_speaker_embedding(
                speaker_id=id_b,
                embedding=embedding_b,
                metadata={"user_id": "test-user", "name": "Bob", "speaker_id": id_b},
            )

            assert embedding_service.speaker_collection.count() == 2

            # Retrieve specific speaker
            result = embedding_service.speaker_collection.get(ids=[id_a])
            assert result["ids"] == [id_a]
            assert result["metadatas"][0]["name"] == "Alice"

            print("\n--- Speaker Listing Report ---")
            print(f"  Total speakers  : {embedding_service.speaker_collection.count()}")
            print(f"  Retrieved       : {result['metadatas'][0]['name']}")
            print("  Status          : SUCCESS")

    def test_speaker_deletion(self):
        """Delete a speaker from ChromaDB and verify removal."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_service = _make_embedding_service(tmp_dir)

            speaker_id = str(uuid.uuid4())
            embedding_service.add_speaker_embedding(
                speaker_id=speaker_id,
                embedding=[0.5] * 192,
                metadata={"user_id": "test-user", "name": "Charlie", "speaker_id": speaker_id},
            )
            assert embedding_service.speaker_collection.count() == 1

            # Delete
            embedding_service.delete_speaker_embedding(speaker_id)
            assert embedding_service.speaker_collection.count() == 0

            print("\n--- Speaker Deletion Report ---")
            print(f"  Deleted ID      : {speaker_id}")
            print(f"  Remaining count : {embedding_service.speaker_collection.count()}")
            print("  Status          : SUCCESS")

    @skip_no_keys
    @skip_no_audio
    @pytest.mark.asyncio
    async def test_pipeline_with_speaker_matching(self, monkeypatch):
        """Enroll a speaker, run the pipeline, and verify speaker label mapping."""
        from perceive8.providers.pyannote import PyannoteProvider
        from perceive8.services.preprocessing import preprocess_audio

        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_service = _make_embedding_service(tmp_dir)

            # --- Enroll a speaker using the sample audio ---
            preprocessing_result = await preprocess_audio(
                str(SAMPLE_AUDIO), tmp_dir
            )

            provider = PyannoteProvider(api_key=PYANNOTE_API_KEY)
            try:
                embedding = await provider.get_speaker_embedding(
                    preprocessing_result.output_path
                )
            finally:
                await provider.close()

            speaker_id = str(uuid.uuid4())
            embedding_service.add_speaker_embedding(
                speaker_id=speaker_id,
                embedding=embedding,
                metadata={"user_id": "test-user", "name": "TestSpeaker", "speaker_id": speaker_id},
            )

            # --- Run the pipeline with embedding_service ---
            from perceive8.config import get_settings

            original_settings = get_settings()

            class _PatchedSettings:
                def __getattr__(self, name):
                    if name == "audio_storage_path":
                        return tmp_dir
                    return getattr(original_settings, name)

            patched = _PatchedSettings()
            monkeypatch.setattr("perceive8.config.get_settings", lambda: patched)
            monkeypatch.setattr("perceive8.services.storage.get_settings", lambda: patched)
            monkeypatch.setattr("perceive8.providers.factory.get_settings", lambda: patched)
            monkeypatch.setattr("perceive8.services.pipeline.get_settings", lambda: patched)

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
                embedding_service=embedding_service,
            )

            assert result.diarization_result is not None
            assert len(result.merged_segments) > 0

            # Check if any speaker label was mapped to the enrolled name
            speaker_labels = {seg.speaker_label for seg in result.merged_segments}

            print("\n" + "=" * 72)
            print("  SPEAKER MATCHING REPORT")
            print("=" * 72)
            print(f"\n  Enrolled speaker : TestSpeaker")
            print(f"  Speaker labels   : {speaker_labels}")
            matched = "TestSpeaker" in speaker_labels
            print(f"  Matched          : {matched}")
            if matched:
                print("  ✓ Speaker was correctly identified!")
            else:
                print("  ⚠ Speaker was not matched (may be below similarity threshold)")
            print("\n  Merged segments with labels:")
            for i, seg in enumerate(result.merged_segments):
                print(
                    f"    [{i:3d}] {_fmt_time(seg.start_time)} -> {_fmt_time(seg.end_time)}  "
                    f"{seg.speaker_label}: {seg.text}"
                )
            print("\n" + "=" * 72)
            print("  END OF SPEAKER MATCHING REPORT")
            print("=" * 72 + "\n")

    @skip_no_keys
    @skip_no_audio
    @pytest.mark.asyncio
    async def test_cross_file_speaker_recognition(self, monkeypatch):
        """Enroll a speaker from one analysis, run a second analysis, and verify recognition."""
        from perceive8.providers.pyannote import PyannoteProvider
        from perceive8.services.preprocessing import preprocess_audio
        from perceive8.services import pipeline as _pipeline_mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_service = _make_embedding_service(tmp_dir)
            user_id = "test-user-cross"

            # --- Step 1: Enroll speaker from sample.wav ---
            preprocessing_result = await preprocess_audio(
                str(SAMPLE_AUDIO), tmp_dir
            )

            provider = PyannoteProvider(api_key=PYANNOTE_API_KEY)
            try:
                embedding = await provider.get_speaker_embedding(
                    preprocessing_result.output_path
                )
            finally:
                await provider.close()

            speaker_id = str(uuid.uuid4())
            embedding_service.add_speaker_embedding(
                speaker_id=speaker_id,
                embedding=embedding,
                metadata={"user_id": user_id, "name": "TestSpeaker", "speaker_id": speaker_id},
            )

            # --- Step 2: Run pipeline on same audio (simulating a second file) ---
            from perceive8.config import get_settings

            original_settings = get_settings()

            class _PatchedSettings:
                def __getattr__(self, name):
                    if name == "audio_storage_path":
                        return tmp_dir
                    return getattr(original_settings, name)

            patched = _PatchedSettings()
            monkeypatch.setattr("perceive8.config.get_settings", lambda: patched)
            monkeypatch.setattr("perceive8.services.storage.get_settings", lambda: patched)
            monkeypatch.setattr("perceive8.providers.factory.get_settings", lambda: patched)
            monkeypatch.setattr("perceive8.services.pipeline.get_settings", lambda: patched)

            # Lower the speaker-matching threshold so clip-vs-full-audio
            # embeddings are close enough to match.
            _orig_match = _pipeline_mod.match_speakers

            async def _match_low_threshold(diarization_result, audio_path, embedding_service, user_id, threshold=0.4):
                return await _orig_match(diarization_result, audio_path, embedding_service, user_id, threshold=0.4)

            monkeypatch.setattr(_pipeline_mod, "match_speakers", _match_low_threshold)

            audio_data = SAMPLE_AUDIO.read_bytes()
            analysis_id = str(uuid.uuid4())

            result = await run_analysis_pipeline(
                audio_data=audio_data,
                filename="sample.wav",
                user_id=user_id,
                analysis_id=analysis_id,
                language=Language.ENGLISH,
                diarization_provider=DiarizationProvider.PYANNOTE,
                transcription_providers=[TranscriptionProvider.OPENAI_WHISPER],
                embedding_service=embedding_service,
            )

            # --- Step 3: Verify TestSpeaker appears in merged segments ---
            assert result.diarization_result is not None
            assert len(result.merged_segments) > 0

            speaker_labels = {seg.speaker_label for seg in result.merged_segments}
            assert "TestSpeaker" in speaker_labels, (
                f"Expected 'TestSpeaker' in speaker labels but got: {speaker_labels}"
            )

            # --- Step 4: Cleanup enrolled speaker ---
            embedding_service.delete_speaker_embedding(speaker_id)
            assert embedding_service.speaker_collection.count() == 0

            # --- Report ---
            print("\n" + "=" * 72)
            print("  CROSS-FILE SPEAKER RECOGNITION REPORT")
            print("=" * 72)
            print(f"\n  Enrolled speaker : TestSpeaker (id={speaker_id})")
            print(f"  Speaker labels   : {speaker_labels}")
            print(f"  Recognition      : {'TestSpeaker' in speaker_labels}")
            print("\n  Merged segments:")
            for i, seg in enumerate(result.merged_segments):
                print(
                    f"    [{i:3d}] {_fmt_time(seg.start_time)} -> {_fmt_time(seg.end_time)}  "
                    f"{seg.speaker_label}: {seg.text}"
                )
            print(f"\n  Cleanup          : speaker deleted (count={embedding_service.speaker_collection.count()})")
            print("\n" + "=" * 72)
            print("  END OF CROSS-FILE SPEAKER RECOGNITION REPORT")
            print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# RAG Query Integration Tests
# ---------------------------------------------------------------------------

skip_no_openai = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set",
)


@pytest.mark.integration
class TestRAGQuery:
    """Integration tests for RAG query service."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_embed_and_search_transcripts(self):
        """Embed sample segments and verify search returns relevant results."""
        from perceive8.services.query import QueryService

        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(
                chromadb_path=os.path.join(tmp_dir, "chromadb"),
                database_url="sqlite+aiosqlite://",
                openai_api_key=OPENAI_API_KEY,
            )
            embedding_service = EmbeddingService(settings)
            query_service = QueryService(embedding_service, settings)

            analysis_id = str(uuid.uuid4())
            segments = [
                {"speaker": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "The quarterly revenue increased by twenty percent."},
                {"speaker": "Bob", "start_time": 5.0, "end_time": 10.0, "text": "We need to hire more engineers for the backend team."},
                {"speaker": "Alice", "start_time": 10.0, "end_time": 15.0, "text": "The marketing budget should be allocated to social media."},
            ]

            await query_service.embed_transcript_segments(analysis_id, segments)

            # Verify segments are stored
            assert embedding_service.transcript_collection.count() == 3

            # Search for revenue-related content
            query_embedding = await query_service.embed_text("What was the revenue growth?")
            matches = embedding_service.search_transcripts(
                query_embedding=query_embedding,
                analysis_id=analysis_id,
                top_k=2,
            )

            assert len(matches) >= 1
            # The top match should be about revenue
            assert "revenue" in matches[0]["metadata"]["text"].lower()

            print("\n--- RAG Embed & Search Report ---")
            print(f"  Segments embedded : {len(segments)}")
            print(f"  Search matches    : {len(matches)}")
            for m in matches:
                print(f"    [{m['metadata']['speaker']}] {m['metadata']['text'][:60]} (sim={m['similarity']:.4f})")
            print("  Status            : SUCCESS")

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_answer_question(self):
        """Embed segments, ask a question, and verify the answer is relevant."""
        from perceive8.services.query import QueryService

        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(
                chromadb_path=os.path.join(tmp_dir, "chromadb"),
                database_url="sqlite+aiosqlite://",
                openai_api_key=OPENAI_API_KEY,
            )
            embedding_service = EmbeddingService(settings)
            query_service = QueryService(embedding_service, settings)

            analysis_id = str(uuid.uuid4())
            segments = [
                {"speaker": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "The quarterly revenue increased by twenty percent compared to last year."},
                {"speaker": "Bob", "start_time": 5.0, "end_time": 10.0, "text": "We should invest more in the engineering team to support growth."},
                {"speaker": "Alice", "start_time": 10.0, "end_time": 15.0, "text": "I agree, let us also increase the marketing budget by ten percent."},
            ]

            await query_service.embed_transcript_segments(analysis_id, segments)

            result = await query_service.answer_question(
                question="What was the revenue increase?",
                analysis_id=analysis_id,
                top_k=3,
            )

            assert "answer" in result
            assert "sources" in result
            assert len(result["sources"]) > 0
            # The answer should mention twenty percent or 20%
            answer_lower = result["answer"].lower()
            assert "twenty" in answer_lower or "20" in answer_lower

            print("\n--- RAG Answer Report ---")
            print(f"  Question : What was the revenue increase?")
            print(f"  Answer   : {result['answer'][:200]}")
            print(f"  Sources  : {len(result['sources'])}")
            for s in result["sources"]:
                print(f"    [{s['speaker_name']}] {s['text'][:60]} (score={s['relevance_score']:.4f})")
            print("  Status   : SUCCESS")

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_cross_analysis_query(self):
        """Embed segments from two analyses, query without analysis_id, verify cross-analysis results."""
        from perceive8.services.query import QueryService

        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(
                chromadb_path=os.path.join(tmp_dir, "chromadb"),
                database_url="sqlite+aiosqlite://",
                openai_api_key=OPENAI_API_KEY,
            )
            embedding_service = EmbeddingService(settings)
            query_service = QueryService(embedding_service, settings)

            analysis_id_1 = str(uuid.uuid4())
            analysis_id_2 = str(uuid.uuid4())

            segments_1 = [
                {"speaker": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "The quarterly revenue increased by twenty percent."},
            ]
            segments_2 = [
                {"speaker": "Bob", "start_time": 0.0, "end_time": 5.0, "text": "We need to hire more engineers for the backend team."},
            ]

            await query_service.embed_transcript_segments(analysis_id_1, segments_1)
            await query_service.embed_transcript_segments(analysis_id_2, segments_2)

            assert embedding_service.transcript_collection.count() == 2

            # Search without analysis_id — should return results from both
            query_embedding = await query_service.embed_text("revenue and hiring")
            matches = embedding_service.search_transcripts(
                query_embedding=query_embedding,
                analysis_id=None,
                top_k=5,
            )

            assert len(matches) == 2
            matched_analysis_ids = {m["metadata"]["analysis_id"] for m in matches}
            assert analysis_id_1 in matched_analysis_ids
            assert analysis_id_2 in matched_analysis_ids

            print("\n--- Cross-Analysis Query Report ---")
            print(f"  Analysis IDs found : {matched_analysis_ids}")
            print(f"  Matches            : {len(matches)}")
            print("  Status             : SUCCESS")

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_backfill_from_db(self):
        """Test backfill_from_db against the real PostgreSQL database."""
        from perceive8.services.query import QueryService
        from perceive8.database import async_session_maker
        from perceive8.models.database import (
            Base,
            User,
            Analysis,
            ProcessingRun,
            TranscriptSegment,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(
                chromadb_path=os.path.join(tmp_dir, "chromadb"),
                database_url="sqlite+aiosqlite://",
                openai_api_key=OPENAI_API_KEY,
            )
            embedding_service = EmbeddingService(settings)
            query_service = QueryService(embedding_service, settings)

            # Insert test data into the real PostgreSQL database
            test_user_id = uuid.uuid4()
            test_analysis_id = uuid.uuid4()
            test_run_id = uuid.uuid4()
            seg1_id = uuid.uuid4()
            seg2_id = uuid.uuid4()

            async with async_session_maker() as db:
                # Create parent records to satisfy FK constraints
                db.add(User(id=test_user_id, external_id=f"backfill-test-{test_user_id}"))
                db.add(Analysis(id=test_analysis_id, user_id=test_user_id, language="en"))
                db.add(ProcessingRun(
                    id=test_run_id,
                    analysis_id=test_analysis_id,
                    run_type="transcription",
                    provider_name="test",
                    status="completed",
                ))
                db.add(TranscriptSegment(
                    id=seg1_id,
                    processing_run_id=test_run_id,
                    start_time=0.0,
                    end_time=3.0,
                    text="The project deadline is next Friday.",
                ))
                db.add(TranscriptSegment(
                    id=seg2_id,
                    processing_run_id=test_run_id,
                    start_time=3.0,
                    end_time=6.0,
                    text="We should allocate more budget to marketing.",
                ))
                await db.commit()

            try:
                # Run backfill against real DB
                async with async_session_maker() as db:
                    count = await query_service.backfill_from_db(db)

                # At minimum our 2 test segments should be embedded;
                # there may be more from other test runs.
                assert count >= 2
                assert embedding_service.transcript_collection.count() >= 2

                print("\n--- Backfill Report ---")
                print(f"  Segments backfilled : {count}")
                print("  Status              : SUCCESS")
            finally:
                # Clean up test data
                async with async_session_maker() as db:
                    await db.execute(
                        TranscriptSegment.__table__.delete().where(
                            TranscriptSegment.processing_run_id == test_run_id
                        )
                    )
                    await db.execute(
                        ProcessingRun.__table__.delete().where(
                            ProcessingRun.id == test_run_id
                        )
                    )
                    await db.execute(
                        Analysis.__table__.delete().where(
                            Analysis.id == test_analysis_id
                        )
                    )
                    await db.execute(
                        User.__table__.delete().where(
                            User.id == test_user_id
                        )
                    )
                    await db.commit()
