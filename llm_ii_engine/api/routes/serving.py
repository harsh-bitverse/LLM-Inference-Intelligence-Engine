"""FastAPI endpoints for Session 27 Serving Architecture Comparison and Recommendations."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from llm_ii_engine.serving.capabilities import AUTHORIZED_MODELS, ServingCapabilityInspector
from llm_ii_engine.serving.context_memory import (
    audit_model_equivalence,
    calculate_kv_cache_memory,
    get_model_architecture_spec,
)
from llm_ii_engine.serving.protocol import (
    ServingArchitectureRecommendation,
    SlotContextRecommendation,
)
from llm_ii_engine.serving.recommender import ServingArchitectureRecommender
from llm_ii_engine.serving.slot_optimizer import SlotContextOptimizer

router = APIRouter(prefix="/serving", tags=["serving-architecture"])

SESSION_27_MANIFEST_PATH = Path("storage/experiments/session_27_serving_manifest.json")
SESSION_28_MANIFEST_PATH = Path("storage/experiments/session_28_context_fair_manifest.json")


class ArchitectureRecommendationRequest(BaseModel):
    """Payload to request an architecture-aware serving recommendation."""

    model_id: str = Field(default="qwen2.5:0.5b")
    workload_type: str = Field(default="BALANCED_MIXED")
    client_concurrency: int = Field(default=2, ge=1)
    arrival_rate_rps: float = Field(default=1.0, gt=0.0)
    latency_sla_ms: float = Field(default=2000.0, gt=0.0)
    memory_budget_mb: float | None = Field(default=None)


class SlotOptimizationRequest(BaseModel):
    """Payload to request context-aware slot optimization."""

    model_id: str = Field(default="qwen2.5:0.5b")
    required_context_tokens: int = Field(default=1024, ge=128)
    expected_output_tokens: int = Field(default=64, ge=1)
    offered_concurrency: int = Field(default=2, ge=1)
    latency_sla_ms: float = Field(default=2000.0, gt=0.0)
    memory_budget_mb: float = Field(default=8192.0, gt=0.0)


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


@router.get("/kv-estimates")
async def get_kv_estimates() -> dict[str, Any]:
    """Retrieve transformer architecture specs and KV-cache scaling matrix."""
    if SESSION_28_MANIFEST_PATH.exists():
        try:
            with open(SESSION_28_MANIFEST_PATH, encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "architectures": data.get("model_architectures", []),
                    "estimates": data.get("kv_cache_estimates", []),
                }
        except Exception:
            pass

    architectures = [get_model_architecture_spec(m).model_dump() for m in AUTHORIZED_MODELS]
    estimates = []
    for m in AUTHORIZED_MODELS:
        for s in [1, 2, 4]:
            for ctx in [512, 1024, 2048, 4096]:
                estimates.append(calculate_kv_cache_memory(m, s, ctx).model_dump())
    return {
        "architectures": architectures,
        "estimates": estimates,
    }


@router.get("/model-equivalence")
async def get_model_equivalence() -> dict[str, Any]:
    """Retrieve audit records comparing model representation across Ollama and llama.cpp server."""
    if SESSION_28_MANIFEST_PATH.exists():
        try:
            with open(SESSION_28_MANIFEST_PATH, encoding="utf-8") as f:
                data = json.load(f)
                return {"records": data.get("equivalence_records", [])}
        except Exception:
            pass
    records = [r.model_dump() for r in audit_model_equivalence()]
    return {"records": records}


@router.get("/context-fair-manifest")
async def get_context_fair_manifest() -> dict[str, Any]:
    """Retrieve complete Session 28 experiment manifest."""
    if SESSION_28_MANIFEST_PATH.exists():
        try:
            with open(SESSION_28_MANIFEST_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"manifest_id": None, "status": "NOT_FOUND"}


@router.post("/optimize-slots", response_model=SlotContextRecommendation)
async def optimize_serving_slots(
    req: SlotOptimizationRequest,
) -> SlotContextRecommendation:
    """Recommend optimal slot allocation and context geometry preserving per-request context and meeting SLA."""
    return SlotContextOptimizer.optimize(
        model_id=req.model_id,
        required_context_tokens=req.required_context_tokens,
        expected_output_tokens=req.expected_output_tokens,
        offered_concurrency=req.offered_concurrency,
        latency_sla_ms=req.latency_sla_ms,
        memory_budget_mb=req.memory_budget_mb,
    )
