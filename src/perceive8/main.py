"""FastAPI application entry point."""

from fastapi import FastAPI

from perceive8.config import get_settings
from perceive8.routes import analysis, benchmark, health, query, speakers

settings = get_settings()

app = FastAPI(
    title="Perceive8 API",
    description="Audio analysis API with speaker identification and transcription",
    version="0.1.0",
    debug=settings.debug,
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(speakers.router, prefix="/speakers", tags=["Speakers"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(benchmark.router, prefix="/benchmark", tags=["Benchmark"])
app.include_router(query.router, prefix="/query", tags=["Query"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # TODO: Initialize database connection pool
    # TODO: Initialize ChromaDB collections
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    pass
