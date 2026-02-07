"""Tests for ChromaDB embedding service and speaker matching."""

import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from perceive8.config import Settings
from perceive8.providers.base import DiarizationResult, DiarizationSegment
from perceive8.services.embedding import EmbeddingService
from perceive8.services.pipeline import match_speakers


# --- EmbeddingService unit tests ---


@pytest.fixture
def embedding_service():
    """Create an EmbeddingService backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(
            database_url="postgresql://test:test@localhost:5432/test_db",
            audio_storage_path="/tmp/test_audio",
            chromadb_path=tmpdir,
            openai_api_key="test-key",
            pyannote_api_key="test-key",
            replicate_api_token="test-token",
        )
        yield EmbeddingService(settings)


def test_add_and_search_speaker_embedding(embedding_service):
    """Add an embedding, then search with the same vector and verify match."""
    embedding = [0.1] * 256
    embedding_service.add_speaker_embedding(
        speaker_id="spk-1",
        embedding=embedding,
        metadata={"user_id": "user1", "name": "Alice"},
    )

    results = embedding_service.search_similar_speakers(
        embedding=embedding,
        user_id="user1",
        threshold=0.8,
    )
    assert len(results) >= 1
    assert results[0]["speaker_id"] == "spk-1"
    assert results[0]["similarity"] >= 0.99  # same vector -> similarity ~1.0
    assert results[0]["metadata"]["name"] == "Alice"


def test_delete_speaker_embedding(embedding_service):
    """After deletion, search should return no results."""
    embedding = [0.2] * 256
    embedding_service.add_speaker_embedding(
        speaker_id="spk-del",
        embedding=embedding,
        metadata={"user_id": "user1", "name": "Bob"},
    )

    embedding_service.delete_speaker_embedding("spk-del")

    results = embedding_service.search_similar_speakers(
        embedding=embedding,
        user_id="user1",
        threshold=0.5,
    )
    assert len(results) == 0


def test_search_with_user_id_filtering(embedding_service):
    """Searching with a different user_id should not return another user's speakers."""
    embedding = [0.3] * 256
    embedding_service.add_speaker_embedding(
        speaker_id="spk-u1",
        embedding=embedding,
        metadata={"user_id": "user1", "name": "Alice"},
    )

    results = embedding_service.search_similar_speakers(
        embedding=embedding,
        user_id="user2",
        threshold=0.5,
    )
    assert len(results) == 0


def test_threshold_filtering(embedding_service):
    """A dissimilar vector should not match above a high threshold."""
    embedding_service.add_speaker_embedding(
        speaker_id="spk-thresh",
        embedding=[1.0] + [0.0] * 255,
        metadata={"user_id": "user1", "name": "Charlie"},
    )

    # Orthogonal vector -> cosine similarity ~0
    dissimilar = [0.0] + [1.0] + [0.0] * 254
    results = embedding_service.search_similar_speakers(
        embedding=dissimilar,
        user_id="user1",
        threshold=0.8,
    )
    assert len(results) == 0


def test_add_and_search_transcript_embedding_with_user_id(embedding_service):
    """Transcript search should be scoped by user_id."""
    embedding = [0.4] * 256
    embedding_service.add_transcript_embedding(
        segment_id="seg-1",
        embedding=embedding,
        metadata={"user_id": "user1", "analysis_id": "a-1", "speaker": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "Hello"},
    )
    embedding_service.add_transcript_embedding(
        segment_id="seg-2",
        embedding=embedding,
        metadata={"user_id": "user2", "analysis_id": "a-2", "speaker": "Bob", "start_time": 0.0, "end_time": 5.0, "text": "World"},
    )

    # user1 should only see their own segment
    results = embedding_service.search_transcripts(
        query_embedding=embedding,
        user_id="user1",
    )
    assert len(results) >= 1
    assert all(r["metadata"]["user_id"] == "user1" for r in results)

    # user2 should only see their own segment
    results = embedding_service.search_transcripts(
        query_embedding=embedding,
        user_id="user2",
    )
    assert len(results) >= 1
    assert all(r["metadata"]["user_id"] == "user2" for r in results)

    # non-existent user should see nothing
    results = embedding_service.search_transcripts(
        query_embedding=embedding,
        user_id="user-nonexistent",
    )
    assert len(results) == 0


def test_search_transcripts_with_analysis_id_filter(embedding_service):
    """Transcript search with analysis_id should narrow within user scope."""
    embedding = [0.5] * 256
    embedding_service.add_transcript_embedding(
        segment_id="seg-a1",
        embedding=embedding,
        metadata={"user_id": "user1", "analysis_id": "a-1", "speaker": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "Hello"},
    )
    embedding_service.add_transcript_embedding(
        segment_id="seg-a2",
        embedding=embedding,
        metadata={"user_id": "user1", "analysis_id": "a-2", "speaker": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "World"},
    )

    results = embedding_service.search_transcripts(
        query_embedding=embedding,
        user_id="user1",
        analysis_id="a-1",
    )
    assert len(results) >= 1
    assert all(r["metadata"]["analysis_id"] == "a-1" for r in results)


# --- Speaker matching tests ---


def _make_mock_audio():
    """Create a mock AudioSegment that supports slicing and export."""
    mock_audio = MagicMock()
    mock_clip = MagicMock()
    mock_clip.export = MagicMock()
    mock_audio.__getitem__ = MagicMock(return_value=mock_clip)
    return mock_audio


@pytest.mark.asyncio
async def test_match_speakers_replaces_labels():
    """match_speakers should return a label map when matches are found."""
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(speaker_label="SPEAKER_00", start_time=0.0, end_time=5.0),
            DiarizationSegment(speaker_label="SPEAKER_01", start_time=5.0, end_time=10.0),
        ],
        model_name="test",
    )

    mock_embedding_service = MagicMock()
    mock_embedding_service.search_similar_speakers.return_value = [
        {"speaker_id": "spk-1", "similarity": 0.95, "metadata": {"name": "Alice"}},
    ]

    mock_provider = AsyncMock()
    mock_provider.get_speaker_embedding = AsyncMock(return_value=[0.1] * 256)
    mock_provider.close = AsyncMock()

    mock_provider_cls = MagicMock(return_value=mock_provider)

    with patch("perceive8.providers.pyannote.PyannoteProvider", mock_provider_cls), \
         patch("perceive8.services.pipeline.get_settings") as mock_settings, \
         patch("pydub.AudioSegment.from_file", return_value=_make_mock_audio()):

        mock_settings.return_value = MagicMock(pyannote_api_key="fake-key")

        label_map = await match_speakers(
            diarization_result=diarization,
            audio_path="/fake/audio.wav",
            embedding_service=mock_embedding_service,
            user_id="user1",
        )

    assert label_map["SPEAKER_00"]["name"] == "Alice"
    assert label_map["SPEAKER_01"]["name"] == "Alice"


@pytest.mark.asyncio
async def test_match_speakers_no_matches():
    """match_speakers should return empty map when no matches found."""
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(speaker_label="SPEAKER_00", start_time=0.0, end_time=5.0),
        ],
        model_name="test",
    )

    mock_embedding_service = MagicMock()
    mock_embedding_service.search_similar_speakers.return_value = []

    mock_provider = AsyncMock()
    mock_provider.get_speaker_embedding = AsyncMock(return_value=[0.1] * 256)
    mock_provider.close = AsyncMock()

    mock_provider_cls = MagicMock(return_value=mock_provider)

    with patch("perceive8.providers.pyannote.PyannoteProvider", mock_provider_cls), \
         patch("perceive8.services.pipeline.get_settings") as mock_settings, \
         patch("pydub.AudioSegment.from_file", return_value=_make_mock_audio()):

        mock_settings.return_value = MagicMock(pyannote_api_key="fake-key")

        label_map = await match_speakers(
            diarization_result=diarization,
            audio_path="/fake/audio.wav",
            embedding_service=mock_embedding_service,
            user_id="user1",
        )

    assert label_map == {}
