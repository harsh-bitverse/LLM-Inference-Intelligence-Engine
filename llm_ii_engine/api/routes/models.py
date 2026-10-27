"""Multi-model catalog, workload suite, and comparative evaluation API routes."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from llm_ii_engine.benchmark.workload_suite import WORKLOAD_SUITE, BenchmarkWorkloadTask
from llm_ii_engine.discovery.model_catalog import (
    CandidateModelSpec,
    get_model_catalog,
    get_recommended_authorization_shortlist,
)
from llm_ii_engine.recommendation.models import ModelQualityConstraintSpec
from llm_ii_engine.recommendation.multi_model_comparator import (
    MultiModelComparator,
    MultiModelComparisonReport,
)

router = APIRouter(prefix="/models", tags=["Multi-Model Comparison"])

_comparator = MultiModelComparator()


class MultiModelCompareRequest(BaseModel):
    """Request payload for multi-model comparison across candidates."""

    candidate_model_ids: list[str] = Field(
        default=["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"],
        description="Candidate model identifiers to evaluate",
    )
    prompt_tokens: int = Field(default=64, gt=0, description="Workload prompt token length")
    expected_output_tokens: int = Field(
        default=64, gt=0, description="Workload output token generation target"
    )
    latency_sla_ms: float | None = Field(
        default=None, gt=0, description="Optional hard latency constraint in ms"
    )
    quality_constraints: ModelQualityConstraintSpec = Field(
        default_factory=ModelQualityConstraintSpec,
        description="Optional quality, formatting, or memory constraints",
    )


@router.get("/catalog", response_model=list[CandidateModelSpec])
async def list_model_catalog() -> list[CandidateModelSpec]:
    """Retrieve all curated candidate models with live local installation status."""
    return get_model_catalog()


@router.get("/shortlist", response_model=list[CandidateModelSpec])
async def list_recommended_shortlist() -> list[CandidateModelSpec]:
    """Retrieve shortlisted candidate models recommended for user download authorization."""
    return get_recommended_authorization_shortlist()


@router.get("/workloads", response_model=list[BenchmarkWorkloadTask])
async def list_workload_suite() -> list[BenchmarkWorkloadTask]:
    """Retrieve deterministic, versioned benchmark workload suite tasks."""
    return WORKLOAD_SUITE


@router.get("/smoke-summary")
async def get_smoke_summary() -> dict[str, Any]:
    """Retrieve empirical summary statistics from the latest multi-model benchmark or smoke validation."""
    s21_file = Path("storage/experiments/session_21_cross_model_manifest.json")
    if s21_file.exists():
        try:
            data = json.loads(s21_file.read_text(encoding="utf-8"))
            return data.get("summary_statistics", {})
        except Exception:
            pass
    smoke_file = Path("storage/experiments/session_20_multi_model_smoke_manifest.json")
    if smoke_file.exists():
        try:
            data = json.loads(smoke_file.read_text(encoding="utf-8"))
            return data.get("summary_statistics", {})
        except Exception:
            pass
    return {}


@router.post("/compare", response_model=MultiModelComparisonReport)
async def compare_candidate_models(
    request: MultiModelCompareRequest,
) -> MultiModelComparisonReport:
    """Evaluate and rank candidate models against a target workload and constraints."""
    smoke_summary = await get_smoke_summary()
    return _comparator.compare_models(
        candidate_ids=request.candidate_model_ids,
        prompt_tokens=request.prompt_tokens,
        expected_output_tokens=request.expected_output_tokens,
        latency_sla_ms=request.latency_sla_ms,
        quality_constraints=request.quality_constraints,
        empirical_smoke_summary=smoke_summary,
    )
