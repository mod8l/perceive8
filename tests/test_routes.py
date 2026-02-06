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
        response = await async_client.get(f"/analysis/{analysis_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_processing_runs(self, async_client):
        """Test getting processing runs returns empty list."""
        analysis_id = uuid4()
        response = await async_client.get(f"/analysis/{analysis_id}/runs")
        assert response.status_code == 200
        assert response.json()["runs"] == []


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
        response = await async_client.get(f"/speakers/{speaker_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_speaker(self, async_client):
        """Test delete speaker endpoint."""
        speaker_id = uuid4()
        response = await async_client.delete(f"/speakers/{speaker_id}")
        assert response.status_code == 200


class TestQueryRoutes:
    """Tests for query endpoints."""

    @pytest.mark.asyncio
    async def test_query_transcripts(self, async_client):
        """Test querying transcripts returns placeholder response."""
        response = await async_client.post(
            "/query",
            json={
                "user_id": "test-user",
                "question": "What did they discuss?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data


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
