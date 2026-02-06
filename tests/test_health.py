"""Tests for health endpoint."""

import pytest


@pytest.mark.asyncio
async def test_health_check(async_client):
    """Test that health endpoint returns healthy status."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
