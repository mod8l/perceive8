"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from perceive8.config import get_settings
from perceive8.services.embedding import EmbeddingService
from perceive8.services.query import QueryService

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Initializing EmbeddingService (chromadb_path=%s)", settings.chromadb_path)
    app.state.embedding_service = EmbeddingService(settings)
    app.state.query_service = QueryService(app.state.embedding_service, settings)
    logger.info("Initialized QueryService")
    yield
    # Shutdown
    logger.info("Shutting down")


app = FastAPI(
    title="Perceive8 API",
    description="Audio analysis API with speaker identification and transcription",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# Include routers
from perceive8.routes import analysis, benchmark, health, query, speakers

app.include_router(health.router, tags=["Health"])
app.include_router(speakers.router, prefix="/speakers", tags=["Speakers"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(benchmark.router, prefix="/benchmark", tags=["Benchmark"])
app.include_router(query.router, prefix="/query", tags=["Query"])
