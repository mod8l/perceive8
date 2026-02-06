"""Tests for Google Drive integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from perceive8.services.gdrive import (
    download_file,
    get_file_metadata,
    parse_file_id,
    validate_audio_file,
)


# --- parse_file_id tests ---


class TestParseFileId:
    def test_file_d_url(self):
        url = "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view"
        assert parse_file_id(url) == "1AbCdEfGhIjKlMnOpQrStUvWxYz"

    def test_file_d_url_with_query(self):
        url = "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing"
        assert parse_file_id(url) == "1AbCdEfGhIjKlMnOpQrStUvWxYz"

    def test_open_id_url(self):
        url = "https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQrStUvWxYz"
        assert parse_file_id(url) == "1AbCdEfGhIjKlMnOpQrStUvWxYz"

    def test_raw_id(self):
        file_id = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
        assert parse_file_id(file_id) == file_id

    def test_raw_id_with_whitespace(self):
        assert parse_file_id("  1AbCdEfGhIjKlMnOpQrStUvWxYz  ") == "1AbCdEfGhIjKlMnOpQrStUvWxYz"

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="Cannot extract"):
            parse_file_id("short")

    def test_invalid_url(self):
        with pytest.raises(ValueError, match="Cannot extract"):
            parse_file_id("https://example.com/not-a-drive-link")


# --- validate_audio_file tests ---


class TestValidateAudioFile:
    def test_valid_audio(self):
        validate_audio_file("audio/mpeg", 1024 * 1024, 500)  # 1 MB, limit 500 MB

    def test_valid_wav(self):
        validate_audio_file("audio/wav", 50 * 1024 * 1024, 500)

    def test_non_audio_mime(self):
        with pytest.raises(ValueError, match="not an audio file"):
            validate_audio_file("video/mp4", 1024, 500)

    def test_text_mime(self):
        with pytest.raises(ValueError, match="not an audio file"):
            validate_audio_file("text/plain", 1024, 500)

    def test_file_too_large(self):
        with pytest.raises(ValueError, match="exceeds limit"):
            validate_audio_file("audio/wav", 600 * 1024 * 1024, 500)  # 600 MB > 500 MB limit

    def test_exactly_at_limit(self):
        # Exactly at limit should pass
        validate_audio_file("audio/wav", 500 * 1024 * 1024, 500)


# --- download_file tests (mocked HTTP) ---


class TestDownloadFile:
    @pytest.mark.asyncio
    async def test_successful_download(self):
        metadata_response = MagicMock(spec=httpx.Response)
        metadata_response.status_code = 200
        metadata_response.json.return_value = {
            "name": "recording.wav",
            "mimeType": "audio/wav",
            "size": "1024",
        }
        metadata_response.raise_for_status = MagicMock()

        download_response = MagicMock(spec=httpx.Response)
        download_response.status_code = 200
        download_response.content = b"fake audio data"
        download_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[metadata_response, download_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("perceive8.services.gdrive.httpx.AsyncClient", return_value=mock_client):
            content, filename = await download_file("test-file-id", "fake-api-key")

        assert content == b"fake audio data"
        assert filename == "recording.wav"

    @pytest.mark.asyncio
    async def test_download_non_audio_rejected(self):
        metadata_response = MagicMock(spec=httpx.Response)
        metadata_response.status_code = 200
        metadata_response.json.return_value = {
            "name": "document.pdf",
            "mimeType": "application/pdf",
            "size": "1024",
        }
        metadata_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=metadata_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("perceive8.services.gdrive.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError, match="not an audio file"):
                await download_file("test-file-id", "fake-api-key")

    @pytest.mark.asyncio
    async def test_download_api_error(self):
        mock_client = AsyncMock()
        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 404
        error_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("Not Found", request=MagicMock(), response=error_response)
        )
        mock_client.get = AsyncMock(return_value=error_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("perceive8.services.gdrive.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await download_file("nonexistent-id", "fake-api-key")


# --- Route tests (mocked service) ---


class TestGDriveRoute:
    @pytest.mark.asyncio
    async def test_missing_api_key_returns_503(self):
        """When google_api_key is empty, endpoint should return 503."""
        from unittest.mock import patch as sync_patch

        from fastapi.testclient import TestClient

        with sync_patch("perceive8.routes.gdrive.settings") as mock_settings:
            mock_settings.google_api_key = ""
            # Need to reimport to pick up patched settings
            from perceive8.routes.gdrive import router
            from fastapi import FastAPI

            test_app = FastAPI()
            test_app.include_router(router)

            with TestClient(test_app) as client:
                resp = client.post(
                    "/analyze/gdrive",
                    json={"files": [{"file_id_or_url": "some-id-1234567890"}]},
                )
                assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_empty_files_returns_422(self):
        """Empty files list should return 422."""
        from unittest.mock import patch as sync_patch

        from fastapi.testclient import TestClient

        with sync_patch("perceive8.routes.gdrive.settings") as mock_settings:
            mock_settings.google_api_key = "test-key"
            from perceive8.routes.gdrive import router
            from fastapi import FastAPI

            test_app = FastAPI()
            test_app.include_router(router)

            with TestClient(test_app) as client:
                resp = client.post(
                    "/analyze/gdrive",
                    json={"files": []},
                )
                assert resp.status_code == 422
