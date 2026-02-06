"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from perceive8.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    # TODO: Initialize database connection pool
    # TODO: Initialize ChromaDB collections
    yield
    # Shutdown
    pass


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
