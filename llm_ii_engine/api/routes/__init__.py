"""API routes module exports."""

from llm_ii_engine.api.routes.benchmarks import router as benchmarks_router
from llm_ii_engine.api.routes.diagnostics import router as diagnostics_router
from llm_ii_engine.api.routes.hardware import router as hardware_router
from llm_ii_engine.api.routes.health import router as health_router
from llm_ii_engine.api.routes.optimization import router as optimization_router
from llm_ii_engine.api.routes.predictions import router as predictions_router
from llm_ii_engine.api.routes.recommendations import router as recommendation_router

__all__ = [
    "benchmarks_router",
    "diagnostics_router",
    "hardware_router",
    "health_router",
    "optimization_router",
    "predictions_router",
    "recommendation_router",
]
