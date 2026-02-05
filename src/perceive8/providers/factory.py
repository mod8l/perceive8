"""Provider factory for creating diarization and transcription providers."""

from perceive8.config import (
    DiarizationProvider,
    TranscriptionProvider,
    get_settings,
)
from perceive8.providers.base import (
    DiarizationProviderInterface,
    TranscriptionProviderInterface,
)


def get_diarization_provider(
    provider: DiarizationProvider | None = None,
) -> DiarizationProviderInterface:
    """
    Get a diarization provider instance.

    Args:
        provider: Provider type, defaults to settings.default_diarization_provider

    Returns:
        DiarizationProviderInterface implementation
    """
    settings = get_settings()
    provider = provider or settings.default_diarization_provider

    if provider == DiarizationProvider.PYANNOTE:
        from perceive8.providers.pyannote import PyannoteProvider

        return PyannoteProvider(api_key=settings.pyannote_api_key)

    elif provider == DiarizationProvider.REPLICATE:
        from perceive8.providers.replicate import ReplicateDiarizationProvider

        return ReplicateDiarizationProvider(api_token=settings.replicate_api_token)

    raise ValueError(f"Unknown diarization provider: {provider}")


def get_transcription_provider(
    provider: TranscriptionProvider | None = None,
) -> TranscriptionProviderInterface:
    """
    Get a transcription provider instance.

    Args:
        provider: Provider type, defaults to settings.default_transcription_provider

    Returns:
        TranscriptionProviderInterface implementation
    """
    settings = get_settings()
    provider = provider or settings.default_transcription_provider

    if provider == TranscriptionProvider.OPENAI_WHISPER:
        from perceive8.providers.openai_whisper import OpenAIWhisperProvider

        return OpenAIWhisperProvider(api_key=settings.openai_api_key)

    elif provider == TranscriptionProvider.REPLICATE:
        from perceive8.providers.replicate import ReplicateTranscriptionProvider

        return ReplicateTranscriptionProvider(api_token=settings.replicate_api_token)

    raise ValueError(f"Unknown transcription provider: {provider}")
