# Railway Services Setup

This guide explains how to set up Railway services for local development.

## Prerequisites

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login to Railway:
   ```bash
   railway login
   ```

## Project Setup

### 1. Create Railway Project

```bash
# Create new project
railway init

# Or link to existing project
railway link
```

### 2. Add PostgreSQL Database

1. Go to Railway dashboard
2. Click "New" → "Database" → "PostgreSQL"
3. Wait for provisioning to complete

### 3. Add Volume for Audio Storage

1. In Railway dashboard, go to your service
2. Click "Settings" → "Volumes"
3. Add a new volume mounted at `/data`

## Local Development

### 1. Copy Environment Template

```bash
cp .env.example .env
```

### 2. Get Railway Variables

```bash
# View all variables
railway variables

# Get specific variable
railway variables get DATABASE_URL
```

### 3. Update .env File

Update your `.env` file with Railway database URL:

```bash
DATABASE_URL=<your-railway-postgresql-url>
```

### 4. Run Locally with Railway DB

```bash
# Option 1: Export variables
export DATABASE_URL=$(railway variables get DATABASE_URL)
uvicorn src.conversation_analyzer.main:app --reload

# Option 2: Use railway run
railway run uvicorn src.conversation_analyzer.main:app --reload
```

## Volume Configuration

For local development, create local directories:

```bash
mkdir -p data/audio data/chromadb
```

In `.env`:
```
AUDIO_STORAGE_PATH=./data/audio
CHROMADB_PATH=./data/chromadb
```

For production (Railway), volumes are mounted at `/data`:
```
AUDIO_STORAGE_PATH=/data/audio
CHROMADB_PATH=/data/chromadb
```

## Environment Variables

| Variable | Description | Local Default |
|----------|-------------|---------------|
| DATABASE_URL | PostgreSQL connection string | from Railway |
| AUDIO_STORAGE_PATH | Path to store audio files | ./data/audio |
| CHROMADB_PATH | Path for ChromaDB persistent storage | ./data/chromadb |
| OPENAI_API_KEY | OpenAI API key for Whisper & embeddings | - |
| PYANNOTE_API_KEY | Pyannote.ai API key | - |
| REPLICATE_API_TOKEN | Replicate API token | - |

## Deployment

Deploy to Railway:

```bash
railway up
```

Or connect your GitHub repo for automatic deployments.
