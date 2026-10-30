"""FastAPI application factory for the LLM Inference Intelligence Engine REST API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_ii_engine.api.errors import register_error_handlers
from llm_ii_engine.api.jobs import BenchmarkJobManager
from llm_ii_engine.api.routes import (
    benchmarks_router,
    capacity_router,
    diagnostics_router,
    hardware_router,
    health_router,
    models_router,
    optimization_router,
    predictions_router,
    recommendation_router,
    serving_router,
)
from llm_ii_engine.storage.database import DatabaseManager


def create_app(
    database_url: str | None = None,
    db_manager: DatabaseManager | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure a FastAPI application instance with database and background runners.

    Args:
        database_url: Optional custom SQLite or relational database URI.
        db_manager: Optional preconfigured DatabaseManager instance.

    Returns:
        Configured FastAPI application ready to serve REST and SSE endpoints.
    """
    manager = db_manager or DatabaseManager(database_url=database_url)
    job_manager = BenchmarkJobManager(db_manager=manager)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup: initialize database tables and schema version check
        manager.init_db(check_version=True)
        yield
        # Shutdown: cancel and drain background tasks, dispose engine pool
        await app.state.job_manager.shutdown()
        manager.dispose()

    app = FastAPI(
        title="LLM Inference Intelligence Engine API",
        description=(
            "REST API foundation providing typed persistence and execution interfaces for "
            "hardware discovery, benchmarking, bottleneck diagnostics, empirical prediction, "
            "and constrained Pareto optimization."
        ),
        version="0.8.0",
        lifespan=lifespan,
    )

    app.state.db_manager = manager
    app.state.job_manager = job_manager

    # Register CORS middleware with explicit development origins
    # Note: CORS is a browser security restriction, NOT authentication.
    # Disallowing credentials prevents credentialed cross-origin token/cookie transmission.
    import os

    if cors_origins is not None:
        allowed_origins = cors_origins
    else:
        env_origins = os.getenv("IIE_CORS_ORIGINS", "")
        if env_origins.strip():
            allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()]
        else:
            allowed_origins = [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
        allow_headers=["*"],
    )

    # Register uniform error mappings
    register_error_handlers(app)

    # Mount health probe at root level and under /api/v1
    app.include_router(health_router)
    app.include_router(health_router, prefix="/api/v1")

    # Mount v1 REST API sub-routers
    app.include_router(hardware_router, prefix="/api/v1")
    app.include_router(benchmarks_router, prefix="/api/v1")
    app.include_router(diagnostics_router, prefix="/api/v1")
    app.include_router(predictions_router, prefix="/api/v1")
    app.include_router(optimization_router, prefix="/api/v1")
    app.include_router(recommendation_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(capacity_router, prefix="/api/v1")
    app.include_router(serving_router, prefix="/api/v1")

    return app
