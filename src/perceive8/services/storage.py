"""File storage service for audio files."""

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from perceive8.config import get_settings

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when storage operations fail."""


@dataclass
class StorageResult:
    """Result from saving a file."""

    file_path: str
    size_bytes: int


def _get_analysis_dir(user_id: str, analysis_id: str) -> Path:
    settings = get_settings()
    return Path(settings.audio_storage_path) / user_id / analysis_id


def _save_sync(audio_data: bytes, filename: str, user_id: str, analysis_id: str) -> StorageResult:
    """Synchronous file save."""
    try:
        dir_path = _get_analysis_dir(user_id, analysis_id)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / filename
        file_path.write_bytes(audio_data)
        size_bytes = len(audio_data)
        logger.info("Saved file: %s (%d bytes)", file_path, size_bytes)
        return StorageResult(file_path=str(file_path), size_bytes=size_bytes)
    except Exception as e:
        raise StorageError(f"Failed to save file: {e}") from e


def _delete_sync(user_id: str, analysis_id: str) -> None:
    """Synchronous directory deletion."""
    dir_path = _get_analysis_dir(user_id, analysis_id)
    if dir_path.exists():
        shutil.rmtree(dir_path)
        logger.info("Deleted analysis files: %s", dir_path)


async def save_audio_file(
    audio_data: bytes, filename: str, user_id: str, analysis_id: str
) -> StorageResult:
    """Save audio data to the storage directory.

    Args:
        audio_data: Raw audio bytes.
        filename: Name for the file.
        user_id: User identifier.
        analysis_id: Analysis identifier.

    Returns:
        StorageResult with file path and size.
    """
    return await asyncio.to_thread(_save_sync, audio_data, filename, user_id, analysis_id)


async def get_audio_path(user_id: str, analysis_id: str, filename: str) -> str:
    """Get the full path for an audio file.

    Args:
        user_id: User identifier.
        analysis_id: Analysis identifier.
        filename: Name of the file.

    Returns:
        Full file path as string.

    Raises:
        StorageError: If the file does not exist.
    """
    file_path = _get_analysis_dir(user_id, analysis_id) / filename
    if not file_path.exists():
        raise StorageError(f"File not found: {file_path}")
    return str(file_path)


async def delete_analysis_files(user_id: str, analysis_id: str) -> None:
    """Delete all files for an analysis.

    Args:
        user_id: User identifier.
        analysis_id: Analysis identifier.
    """
    await asyncio.to_thread(_delete_sync, user_id, analysis_id)
