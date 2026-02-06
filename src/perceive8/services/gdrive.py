"""Google Drive file download service."""

import logging
import re
from typing import Tuple

import httpx

logger = logging.getLogger(__name__)

# Patterns for extracting file IDs from Google Drive URLs
_FILE_D_PATTERN = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_OPEN_ID_PATTERN = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")
_RAW_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{10,}$")

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3/files"


def parse_file_id(url_or_id: str) -> str:
    """Extract a Google Drive file ID from a URL or pass through a raw ID.

    Supported formats:
    - https://drive.google.com/file/d/{id}/view
    - https://drive.google.com/open?id={id}
    - Raw file ID string

    Raises:
        ValueError: If the input doesn't match any known format.
    """
    url_or_id = url_or_id.strip()

    match = _FILE_D_PATTERN.search(url_or_id)
    if match:
        return match.group(1)

    match = _OPEN_ID_PATTERN.search(url_or_id)
    if match:
        return match.group(1)

    if _RAW_ID_PATTERN.match(url_or_id):
        return url_or_id

    raise ValueError(f"Cannot extract Google Drive file ID from: {url_or_id}")


async def get_file_metadata(file_id: str, api_key: str) -> dict:
    """Fetch file metadata from Google Drive API.

    Returns:
        Dict with keys: name, mimeType, size.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
    """
    url = f"{DRIVE_API_BASE}/{file_id}"
    params = {"key": api_key, "fields": "name,mimeType,size"}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def validate_audio_file(mime_type: str, size: int, max_size_mb: int) -> None:
    """Validate that a file is an audio file within size limits.

    Raises:
        ValueError: If mime type is not audio/* or file exceeds size limit.
    """
    if not mime_type.startswith("audio/"):
        raise ValueError(f"File is not an audio file (mime type: {mime_type})")

    max_bytes = max_size_mb * 1024 * 1024
    if size > max_bytes:
        raise ValueError(
            f"File size ({size / (1024*1024):.1f} MB) exceeds limit ({max_size_mb} MB)"
        )


async def download_file(
    file_id: str, api_key: str, max_size_mb: int = 500
) -> Tuple[bytes, str]:
    """Download a file from Google Drive.

    Fetches metadata first to validate, then downloads the content.

    Returns:
        Tuple of (file_bytes, filename).

    Raises:
        ValueError: If file is not audio or too large.
        httpx.HTTPStatusError: On API errors.
    """
    # Get metadata first
    metadata = await get_file_metadata(file_id, api_key)
    filename = metadata.get("name", f"{file_id}.audio")
    mime_type = metadata.get("mimeType", "")
    size = int(metadata.get("size", 0))

    validate_audio_file(mime_type, size, max_size_mb)

    # Download content
    url = f"{DRIVE_API_BASE}/{file_id}"
    params = {"alt": "media", "key": api_key}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.content, filename
