FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini .

# Create data directories
RUN mkdir -p /data/audio /data/chromadb

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "conversation_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]
