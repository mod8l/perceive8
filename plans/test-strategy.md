# Integration Test Strategy

## Overview

This document outlines a strategy for testing the diarization/transcription providers (OpenAI Whisper, Pyannote, Replicate) with both mocked and real API calls.

## Test Categories

### Unit Tests (Current - `make test`)
- Fast, isolated tests with mocked provider responses
- No API keys required
- Run on every commit/PR

### Integration Tests (New - `make test-integration`)
- Real API calls to providers
- Requires valid API keys in environment
- Marked with `@pytest.mark.integration`
- Skipped by default in CI
- Run manually or on release branches

## Test Audio Fixture

Create `tests/fixtures/sample.wav`:
- Duration: 5-10 seconds
- 2 distinct speakers with clear dialogue
- Example: "Speaker A: Hello, how are you? Speaker B: I'm doing well, thanks."
- Format: WAV, 16kHz, mono

Consider also:
- `tests/fixtures/invalid_audio.txt` - for error handling tests
- `tests/fixtures/silence.wav` - edge case testing

## Implementation Plan

### 1. Directory Structure
```
tests/
├── fixtures/
│   └── sample.wav
├── integration/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_providers_integration.py
├── unit/
│   └── (move existing tests here)
└── conftest.py
```

### 2. Pytest Configuration

```ini
# pytest.ini
[pytest]
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
```

### 3. Integration Test Examples

```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_whisper_transcription_real():
    """Test real Whisper API transcription."""
    provider = OpenAIWhisperProvider()
    result = await provider.transcribe("tests/fixtures/sample.wav")
    assert "hello" in result.text.lower()
    assert result.segments is not None

@pytest.mark.integration
async def test_pyannote_diarization_real():
    """Test real Pyannote diarization."""
    provider = PyannoteProvider()
    result = await provider.diarize("tests/fixtures/sample.wav")
    assert len(result.speakers) >= 2
```

### 4. Makefile Updates

```makefile
test:
	pytest -m "not integration" -v

test-integration:
	pytest -m "integration" -v

test-all:
	pytest -v
```

## What to Test

| Provider | Test Case | Validation |
|----------|-----------|------------|
| Whisper | Basic transcription | Text contains expected words |
| Whisper | Timestamp accuracy | Segments have valid timestamps |
| Pyannote | Speaker count | Detects correct number of speakers |
| Pyannote | Speaker labels | Each segment has speaker assignment |
| Replicate | Model availability | API responds without error |
| All | Invalid audio | Raises appropriate exception |
| All | Missing API key | Clear error message |

## CI/CD Considerations

- Unit tests: Run on all PRs
- Integration tests: Run on `main` branch only, or manually triggered
- Store API keys as GitHub secrets
- Consider cost budgets for integration test runs
