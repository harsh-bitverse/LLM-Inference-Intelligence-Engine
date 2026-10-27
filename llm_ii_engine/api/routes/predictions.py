"""Empirical prediction model evaluation and validation endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, Query, status

from llm_ii_engine.api.dependencies import get_repository
from llm_ii_engine.prediction.models import EvaluationResult
from llm_ii_engine.storage.repository import EngineRepository, EntityNotFoundError

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get(
    "/evaluations",
    summary="List empirical prediction model validation evaluations",
)
async def list_evaluations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    target: str | None = Query(default=None, description="Optional filter by PredictionTarget"),
    repo: EngineRepository = Depends(get_repository),
) -> dict[str, Any]:
    """Retrieve persisted prediction evaluation outcomes with error metrics and sufficiency indicators."""
    items, total = repo.list_prediction_evaluations(
        limit=limit,
        offset=offset,
        target=target,
    )
    return {
        "items": [item.model_dump() for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post(
    "/evaluations",
    status_code=status.HTTP_201_CREATED,
    summary="Record an empirical prediction evaluation outcome",
)
async def create_evaluation(
    evaluation: EvaluationResult,
    evaluation_id: str | None = Query(default=None),
    repo: EngineRepository = Depends(get_repository),
) -> dict[str, str]:
    """Store an EvaluationResult reflecting cross-validation or train/test evaluation of a predictor."""
    eval_id = repo.save_prediction_evaluation(evaluation, evaluation_id=evaluation_id)
    return {
        "evaluation_id": eval_id,
        "message": "Prediction evaluation record created successfully.",
    }


@router.get(
    "/evaluations/{evaluation_id}",
    response_model=EvaluationResult,
    summary="Retrieve an evaluation record by ID",
)
async def get_evaluation_by_id(
    evaluation_id: str,
    repo: EngineRepository = Depends(get_repository),
) -> EvaluationResult:
    """Retrieve an individual EvaluationResult by its evaluation_id."""
    evaluation = repo.get_prediction_evaluation(evaluation_id)
    if not evaluation:
        raise EntityNotFoundError(f"Prediction evaluation '{evaluation_id}' not found.")
    return evaluation
