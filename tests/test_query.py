"""Unit tests for QueryService and query routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from perceive8.config import Settings
from perceive8.services.query import QueryService, _fmt_time


# ---------------------------------------------------------------------------
# Helper: build a QueryService with mocked OpenAI + EmbeddingService
# ---------------------------------------------------------------------------

def _make_query_service():
    """Return (query_service, mock_embedding_service, mock_openai_client)."""
    settings = Settings(
        database_url="postgresql://test:test@localhost:5432/test_db",
        audio_storage_path="/tmp/test_audio",
        chromadb_path="/tmp/test_chromadb",
        openai_api_key="test-key",
        openai_embedding_model="text-embedding-3-small",
        openai_chat_model="gpt-4o-mini",
        rag_top_k=3,
    )
    mock_embedding_service = MagicMock()
    service = QueryService(mock_embedding_service, settings)

    # Replace the real OpenAI async client with a mock
    mock_openai = AsyncMock()
    service._openai_client = mock_openai

    return service, mock_embedding_service, mock_openai


# ---------------------------------------------------------------------------
# _fmt_time helper
# ---------------------------------------------------------------------------

def test_fmt_time():
    """_fmt_time should format seconds as MM:SS."""
    assert _fmt_time(0) == "00:00"
    assert _fmt_time(65) == "01:05"
    assert _fmt_time(3661) == "61:01"


# ---------------------------------------------------------------------------
# QueryService.embed_text
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_text():
    """embed_text should call OpenAI embeddings and return the vector."""
    service, _, mock_openai = _make_query_service()

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_response)

    result = await service.embed_text("hello world")

    assert result == [0.1, 0.2, 0.3]
    mock_openai.embeddings.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# QueryService.embed_transcript_segments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_transcript_segments():
    """embed_transcript_segments should embed each non-empty segment and store it."""
    service, mock_emb, mock_openai = _make_query_service()

    mock_embedding_obj = MagicMock()
    mock_embedding_obj.embedding = [0.5] * 10
    mock_resp = MagicMock()
    mock_resp.data = [mock_embedding_obj]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_resp)

    segments = [
        {"speaker": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "Hello"},
        {"speaker": "Bob", "start_time": 5.0, "end_time": 10.0, "text": ""},
        {"speaker": "Alice", "start_time": 10.0, "end_time": 15.0, "text": "World"},
    ]

    await service.embed_transcript_segments("analysis-1", segments)

    # Two non-empty segments should be embedded
    assert mock_openai.embeddings.create.await_count == 2
    assert mock_emb.add_transcript_embedding.call_count == 2


@pytest.mark.asyncio
async def test_embed_transcript_segments_skips_whitespace():
    """Segments with only whitespace text should be skipped."""
    service, mock_emb, mock_openai = _make_query_service()

    segments = [
        {"speaker": "A", "start_time": 0, "end_time": 1, "text": "   "},
    ]

    await service.embed_transcript_segments("a-1", segments)

    mock_openai.embeddings.create.assert_not_awaited()
    mock_emb.add_transcript_embedding.assert_not_called()


# ---------------------------------------------------------------------------
# QueryService.answer_question
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_answer_question_no_matches():
    """When no transcript matches are found, return a canned message."""
    service, mock_emb, mock_openai = _make_query_service()

    # embed_text mock
    mock_embedding_obj = MagicMock()
    mock_embedding_obj.embedding = [0.1] * 10
    mock_resp = MagicMock()
    mock_resp.data = [mock_embedding_obj]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_resp)

    mock_emb.search_transcripts.return_value = []

    result = await service.answer_question("What happened?", analysis_id="a-1")

    assert "No relevant transcript segments found" in result["answer"]
    assert result["sources"] == []


@pytest.mark.asyncio
async def test_answer_question_with_matches():
    """When matches exist, should call chat completion and return answer + sources."""
    service, mock_emb, mock_openai = _make_query_service()

    # embed_text mock
    mock_embedding_obj = MagicMock()
    mock_embedding_obj.embedding = [0.1] * 10
    mock_resp = MagicMock()
    mock_resp.data = [mock_embedding_obj]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_resp)

    mock_emb.search_transcripts.return_value = [
        {
            "metadata": {
                "analysis_id": "a-1",
                "speaker": "Alice",
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Revenue grew 20%.",
            },
            "similarity": 0.95,
        },
    ]

    # Chat completion mock
    mock_choice = MagicMock()
    mock_choice.message.content = "Revenue increased by 20%."
    mock_chat_resp = MagicMock()
    mock_chat_resp.choices = [mock_choice]
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_chat_resp)

    result = await service.answer_question("What was the revenue?", analysis_id="a-1")

    assert result["answer"] == "Revenue increased by 20%."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["speaker_name"] == "Alice"
    assert result["sources"][0]["relevance_score"] == 0.95
    mock_openai.chat.completions.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Query route tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_route_analysis_not_found(async_client):
    """POST /query with a non-existent analysis_id should return 404."""
    response = await async_client.post(
        "/query",
        json={
            "question": "What happened?",
            "analysis_id": str(uuid4()),
        },
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_query_history_not_found(async_client):
    """GET /query/history/{analysis_id} with non-existent id should return 404."""
    response = await async_client.get(f"/query/history/{uuid4()}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
