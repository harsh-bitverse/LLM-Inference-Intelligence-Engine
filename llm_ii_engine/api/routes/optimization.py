"""Constrained configuration optimization and Pareto frontier recommendation endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, Query

from llm_ii_engine.api.dependencies import get_repository
from llm_ii_engine.optimizer.engine import ConfigurationOptimizer
from llm_ii_engine.optimizer.models import OptimizationRequest, OptimizationResult
from llm_ii_engine.storage.repository import EngineRepository, EntityNotFoundError

router = APIRouter(prefix="/optimize", tags=["Optimization"])


@router.post(
    "/recommend",
    response_model=OptimizationResult,
    summary="Evaluate search space and compute Pareto-optimal configurations",
)
async def optimize_configuration(
    request: OptimizationRequest,
    repo: EngineRepository = Depends(get_repository),
) -> OptimizationResult:
    """Execute constrained optimization search across engine configurations and return Pareto recommendations.

    Valid scientific outcomes (including UNKNOWN_FEASIBILITY, INSUFFICIENT_EVIDENCE, or theoretical bounds)
    are returned as HTTP 200 with complete evidence attribution.
    """
    optimizer = ConfigurationOptimizer()
    result = optimizer.optimize(request)

    # Persist request and optimization result
    repo.save_optimization(request, result)
    return result


@router.get(
    "/history",
    summary="List historical optimization runs",
)
async def list_optimizations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    repo: EngineRepository = Depends(get_repository),
) -> dict[str, Any]:
    """Retrieve historical optimization runs with summary statistics."""
    items, total = repo.list_optimizations(limit=limit, offset=offset)
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/{optimization_id}",
    summary="Retrieve an optimization run and its Pareto candidates by ID",
)
async def get_optimization_by_id(
    optimization_id: str,
    repo: EngineRepository = Depends(get_repository),
) -> dict[str, Any]:
    """Retrieve full specification, candidate evaluations, and Pareto frontier for an optimization request."""
    loaded = repo.get_optimization(optimization_id)
    if not loaded:
        raise EntityNotFoundError(f"Optimization run '{optimization_id}' not found.")

    req, res = loaded
    return {
        "request": req.model_dump(),
        "result": res.model_dump(),
    }
