# Perceive8

API service for analyzing audio recordings with speaker identification and transcription.

## Features

- **Speaker Diarization** - Detect who is speaking when
- **Speaker Identification** - Match speakers to enrolled profiles
- **Transcription** - Convert speech to text with word-level timestamps
- **Multi-language Support** - English, Hebrew, Spanish
- **RAG Q&A** - Ask questions about transcripts
- **Provider Flexibility** - Swap providers (pyannote.ai, Replicate, OpenAI Whisper)
- **Benchmarking** - Compare providers on same audio

## Quick Start

### Prerequisites

- Python 3.10+
- Railway CLI (for remote services)
- API keys for: OpenAI, pyannote.ai, Replicate (optional)

### Installation

```bash
# Clone and enter directory
cd perceive8

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Local Development with Railway

See [services/README.md](services/README.md) for Railway setup.

```bash
# Get database URL from Railway
export DATABASE_URL=$(railway variables get DATABASE_URL)

# Run locally
uvicorn src.perceive8.main:app --reload
```

## API Endpoints

### Analysis
- `POST /analysis` - Submit audio for analysis
- `GET /analysis/{id}` - Get analysis result
- `GET /analysis/{id}/runs` - Get processing runs for model comparison

### Speakers
- `POST /speakers` - Enroll speaker with voice sample
- `GET /speakers` - List enrolled speakers
- `DELETE /speakers/{id}` - Remove speaker

### Query (RAG)
- `POST /query` - Ask questions about transcripts

### Benchmark
- `POST /benchmark` - Run audio through multiple providers
- `GET /benchmark/{id}/compare` - Compare results

### Health
- `GET /health` - Service health check

## Example Usage

### Analyze Audio

```bash
curl -X POST http://localhost:8000/analysis \
  -F "user_id=user123" \
  -F "audio_file=@recording.wav" \
  -F "language=en" \
  -F "transcription_providers=openai_whisper"
```

### Query Transcripts

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "question": "What was discussed about pricing?"
  }'
```

## Architecture

See [plans/conversation-analyzer-plan.md](plans/conversation-analyzer-plan.md) for full architecture documentation.

## License

MIT
