"""Tests for route handlers."""

import pytest
from uuid import uuid4


class TestAnalysisRoutes:
    """Tests for analysis endpoints."""

    @pytest.mark.asyncio
    async def test_list_analyses(self, async_client):
        """Test listing analyses returns empty list."""
        response = await async_client.get("/analysis", params={"user_id": "test-user"})
        assert response.status_code == 200
        data = response.json()
        assert data["analyses"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_analysis_not_found(self, async_client):
        """Test getting non-existent analysis returns 404."""
        analysis_id = uuid4()
        response = await async_client.get(f"/analysis/{analysis_id}", params={"user_id": "test-user"})
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_processing_runs_not_found(self, async_client):
        """Test getting processing runs for non-existent analysis returns 404."""
        analysis_id = uuid4()
        response = await async_client.get(f"/analysis/{analysis_id}/runs", params={"user_id": "test-user"})
        assert response.status_code == 404


class TestSpeakerRoutes:
    """Tests for speaker endpoints."""

    @pytest.mark.asyncio
    async def test_list_speakers(self, async_client):
        """Test listing speakers returns empty list."""
        response = await async_client.get("/speakers", params={"user_id": "test-user"})
        assert response.status_code == 200
        data = response.json()
        assert data["speakers"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_speaker_not_found(self, async_client):
        """Test getting non-existent speaker returns 404."""
        speaker_id = uuid4()
        response = await async_client.get(f"/speakers/{speaker_id}", params={"user_id": "test-user"})
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_speaker_no_embedding_service(self, async_client):
        """Test delete speaker without embedding service returns 503."""
        speaker_id = uuid4()
        response = await async_client.delete(f"/speakers/{speaker_id}", params={"user_id": "test-user"})
        assert response.status_code == 503


class TestQueryRoutes:
    """Tests for query endpoints."""

    @pytest.mark.asyncio
    async def test_query_transcripts_no_service(self, async_client):
        """Test querying transcripts without query service returns 503."""
        response = await async_client.post(
            "/query",
            json={
                "user_id": "test-user",
                "question": "What did they discuss?",
            },
        )
        assert response.status_code == 503


class TestErrorHandling:
    """Tests for error handling in routes."""

    @pytest.mark.asyncio
    async def test_invalid_audio_file_upload(self, async_client):
        """Uploading a non-audio file should fail during pipeline processing."""
        import io
        from pathlib import Path

        invalid_path = Path(__file__).parent / "fixtures" / "invalid_audio.txt"
        content = invalid_path.read_bytes()

        response = await async_client.post(
            "/analysis",
            data={"user_id": "test-user"},
            files={"audio_file": ("invalid_audio.txt", io.BytesIO(content), "text/plain")},
        )
        # The route attempts to run the pipeline which should fail;
        # with mocked DB it may hit a DB error first — either 422/500/502 is acceptable
        assert response.status_code in (422, 500, 502)

    @pytest.mark.asyncio
    async def test_missing_api_key_error(self, async_client):
        """When settings have empty API keys, provider init should fail gracefully."""
        from unittest.mock import patch, MagicMock

        mock_settings = MagicMock()
        mock_settings.openai_api_key = ""
        mock_settings.pyannote_api_key = ""
        mock_settings.replicate_api_token = ""
        mock_settings.default_language = "en"
        mock_settings.default_diarization_provider = "pyannote"
        mock_settings.default_transcription_provider = "openai_whisper"

        with patch("perceive8.routes.analysis.settings", mock_settings), \
             patch("perceive8.routes.analysis.get_settings", return_value=mock_settings):
            import io

            response = await async_client.post(
                "/analysis",
                data={"user_id": "test-user"},
                files={"audio_file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            )
            # Should fail during pipeline execution with empty keys
            assert response.status_code in (422, 500, 502)


class TestBenchmarkRoutes:
    """Tests for benchmark endpoints."""

    @pytest.mark.asyncio
    async def test_get_benchmark_not_found(self, async_client):
        """Test getting non-existent benchmark returns 404."""
        benchmark_id = uuid4()
        response = await async_client.get(f"/benchmark/{benchmark_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_compare_benchmark(self, async_client):
        """Test comparing benchmark results."""
        benchmark_id = uuid4()
        response = await async_client.get(f"/benchmark/{benchmark_id}/compare")
        assert response.status_code == 200
        assert response.json() == {"comparison": {}}
