"""FastAPI endpoints for Session 27 Serving Architecture Comparison and Recommendations."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from llm_ii_engine.serving.capabilities import ServingCapabilityInspector
from llm_ii_engine.serving.protocol import (
    ServingArchitectureRecommendation,
)
from llm_ii_engine.serving.recommender import ServingArchitectureRecommender

router = APIRouter(prefix="/serving", tags=["serving-architecture"])

SESSION_27_MANIFEST_PATH = Path("storage/experiments/session_27_serving_manifest.json")


class ArchitectureRecommendationRequest(BaseModel):
    """Payload to request an architecture-aware serving recommendation."""

    model_id: str = Field(default="qwen2.5:0.5b")
    workload_type: str = Field(default="BALANCED_MIXED")
    client_concurrency: int = Field(default=2, ge=1)
    arrival_rate_rps: float = Field(default=1.0, gt=0.0)
    latency_sla_ms: float = Field(default=2000.0, gt=0.0)
    memory_budget_mb: float | None = Field(default=None)


@router.get("/capabilities")
async def get_serving_capabilities() -> dict[str, Any]:
    """Retrieve serving capabilities matrix distinguishing advertised vs observed support."""
    return ServingCapabilityInspector.get_summary_dictionary()


@router.get("/comparison")
async def get_architecture_comparisons() -> dict[str, Any]:
    """Retrieve empirical architecture comparisons between single-slot and multi-slot continuous batching."""
    if SESSION_27_MANIFEST_PATH.exists():
        try:
            with open(SESSION_27_MANIFEST_PATH, encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "manifest_id": data.get("manifest_id"),
                    "hardware_id": data.get("hardware_id"),
                    "total_duration_s": data.get("total_benchmark_duration_s"),
                    "comparisons": data.get("comparisons", []),
                }
        except Exception:
            pass
    return {"comparisons": []}


@router.get("/hol-comparison")
async def get_hol_comparison() -> dict[str, Any]:
    """Retrieve Head-of-Line blocking comparisons showing queue wait reduction and SLA recovery."""
    if SESSION_27_MANIFEST_PATH.exists():
        try:
            with open(SESSION_27_MANIFEST_PATH, encoding="utf-8") as f:
                data = json.load(f)
                return {"hol_comparisons": data.get("hol_comparisons", [])}
        except Exception:
            pass
    return {"hol_comparisons": []}


@router.post("/recommend", response_model=ServingArchitectureRecommendation)
async def recommend_serving_architecture(
    req: ArchitectureRecommendationRequest,
) -> ServingArchitectureRecommendation:
    """Recommend optimal inference serving architecture based on workload composition and arrival rate."""
    return ServingArchitectureRecommender.recommend_architecture(
        model_id=req.model_id,
        workload_type=req.workload_type,
        client_concurrency=req.client_concurrency,
        arrival_rate_rps=req.arrival_rate_rps,
        latency_sla_ms=req.latency_sla_ms,
        memory_budget_mb=req.memory_budget_mb,
    )
