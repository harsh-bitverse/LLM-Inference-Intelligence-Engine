"""Workload recommendation and prospective validation API endpoints."""

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from llm_ii_engine.recommendation.engine import WorkloadRecommendationEngine
from llm_ii_engine.recommendation.models import (
    WorkloadRecommendationRequest,
    WorkloadRecommendationResponse,
)
from llm_ii_engine.recommendation.runtime_recommender import (
    RuntimeConfigurationRecommender,
    RuntimeRecommendationRequest,
    RuntimeRecommendationResult,
)

router = APIRouter(prefix="/recommendation", tags=["Workload Recommendation"])

# In-memory recommendation engine instance initialized with dataset directory
_engine = WorkloadRecommendationEngine(dataset_path=Path(__file__).resolve().parent.parent.parent.parent)
_runtime_recommender = RuntimeConfigurationRecommender(
    manifest_path=Path(__file__).resolve().parent.parent.parent.parent / "storage" / "experiments" / "session_24_manifest.json"
)


class ValidationRequest(BaseModel):
    """Request to execute a live validation inference run against a recommended configuration."""

    recommendation_id: str
    model_id: str = "llama3.2:1b"
    prompt_tokens: int = Field(default=32, gt=0)
    expected_output_tokens: int = Field(default=16, gt=0)
    expected_latency_ms: float | None = None
    expected_decode_throughput_tok_s: float | None = None


class ValidationResponse(BaseModel):
    """Comparative results of prospective prediction vs observed live inference."""

    validation_id: str
    recommendation_id: str
    status: str
    executed_at: str
    observed_client_e2e_duration_ms: float
    observed_server_decode_duration_ms: float | None
    observed_server_prefill_duration_ms: float | None
    observed_server_load_duration_ms: float | None
    observed_server_decode_throughput_tok_s: float | None
    observed_client_e2e_output_throughput_tok_s: float | None
    predicted_client_e2e_duration_ms: float | None
    predicted_server_decode_throughput_tok_s: float | None
    duration_absolute_error_ms: float | None
    duration_relative_error_pct: float | None
    throughput_absolute_error_tok_s: float | None
    throughput_relative_error_pct: float | None
    explanation: str


@router.post(
    "/workload",
    response_model=WorkloadRecommendationResponse,
    summary="Compute evidence-backed inference configuration recommendation",
)
@router.post(
    "",
    response_model=WorkloadRecommendationResponse,
    include_in_schema=False,
)
async def recommend_workload_configuration(
    request: WorkloadRecommendationRequest,
) -> WorkloadRecommendationResponse:
    """Evaluate workload specifications, check feasibility, and return evidence-backed recommendation."""
    # Validate model availability
    if request.model_id not in ("llama3.2:1b", "llama3.2:1b-instruct-q8_0"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model_id}' is not installed or authorized on local runtime. Only 'llama3.2:1b' is supported.",
        )

    # Validate impossible prompt/output combinations
    if request.prompt_tokens <= 0 or request.expected_output_tokens <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt and output token counts must be strictly positive.",
        )

    return _engine.recommend(request)


@router.get(
    "/scenarios",
    summary="List standard product validation scenarios (A through F)",
)
async def list_validation_scenarios() -> list[dict[str, Any]]:
    """Retrieve the six standard product evaluation scenarios."""
    return [
        {
            "scenario_id": "SCENARIO_A",
            "name": "Scenario A — Short Interactive Request",
            "description": "Small prompt (32 tokens) and short response (16 tokens) prioritizing low latency.",
            "request": {
                "model_id": "llama3.2:1b",
                "prompt_tokens": 32,
                "expected_output_tokens": 16,
                "concurrency": 1,
                "expected_model_residency": "warm",
                "expected_prefix_cache": "cold_prefix",
                "latency_sla_ms": 2500.0,
                "objective": "minimize_duration",
            },
        },
        {
            "scenario_id": "SCENARIO_B",
            "name": "Scenario B — Long-Context Request",
            "description": "Large prompt (677 tokens) and moderate output (32 tokens) evaluating prefill sensitivity.",
            "request": {
                "model_id": "llama3.2:1b",
                "prompt_tokens": 677,
                "expected_output_tokens": 32,
                "concurrency": 1,
                "expected_model_residency": "warm",
                "expected_prefix_cache": "unknown",
                "latency_sla_ms": 5000.0,
                "objective": "minimize_duration",
            },
        },
        {
            "scenario_id": "SCENARIO_C",
            "name": "Scenario C — Long Generation Request",
            "description": "Moderate prompt (128 tokens) and long output (128 tokens) evaluating autoregressive decode throughput.",
            "request": {
                "model_id": "llama3.2:1b",
                "prompt_tokens": 128,
                "expected_output_tokens": 128,
                "concurrency": 1,
                "expected_model_residency": "warm",
                "latency_sla_ms": 8000.0,
                "objective": "maximize_decode_throughput",
            },
        },
        {
            "scenario_id": "SCENARIO_D",
            "name": "Scenario D — Cold-Start-Sensitive Request",
            "description": "Requires cold-model evidence with explicit 1,038 ms weight loading overhead.",
            "request": {
                "model_id": "llama3.2:1b",
                "prompt_tokens": 119,
                "expected_output_tokens": 32,
                "concurrency": 1,
                "expected_model_residency": "cold",
                "require_cold_start": True,
                "latency_sla_ms": 4000.0,
                "objective": "minimize_duration",
            },
        },
        {
            "scenario_id": "SCENARIO_E",
            "name": "Scenario E — Strict TTFT SLA Request",
            "description": "Requests an unmeasurable token-level TTFT SLA (100 ms) demonstrating unknown feasibility.",
            "request": {
                "model_id": "llama3.2:1b",
                "prompt_tokens": 46,
                "expected_output_tokens": 32,
                "concurrency": 1,
                "ttft_sla_ms": 100.0,
                "objective": "minimize_duration",
            },
        },
        {
            "scenario_id": "SCENARIO_F",
            "name": "Scenario F — Out-Of-Distribution Request",
            "description": "Workload (4000 prompt, 500 output tokens) beyond empirical envelope triggering extrapolation warnings.",
            "request": {
                "model_id": "llama3.2:1b",
                "prompt_tokens": 4000,
                "expected_output_tokens": 500,
                "concurrency": 1,
                "latency_sla_ms": 30000.0,
                "objective": "minimize_duration",
            },
        },
    ]


@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Validate recommendation with a live inference benchmark on local Ollama",
)
async def validate_recommendation(
    request: ValidationRequest,
) -> ValidationResponse:
    """Execute live inference on Ollama server and compare observed metrics against prospective prediction."""
    val_id = f"val-{uuid.uuid4().hex[:12]}"
    prompt_text = "Validation benchmark prompt: " + ("word " * max(1, request.prompt_tokens // 2))

    payload = {
        "model": request.model_id,
        "prompt": prompt_text,
        "options": {
            "num_predict": request.expected_output_tokens,
            "temperature": 0.0,
        },
        "stream": False,
    }

    t_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", trust_env=False, timeout=60.0) as client:
            resp = await client.post("/api/generate", json=payload)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Ollama server returned HTTP {resp.status_code}: {resp.text}",
                )
            data = resp.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama runtime is unavailable on http://127.0.0.1:11434. Please ensure Ollama is running.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        )
    t_end = time.perf_counter()

    obs_e2e_ms = round((t_end - t_start) * 1000.0, 2)
    eval_count = data.get("eval_count", 0)
    eval_dur_ns = data.get("eval_duration")
    prompt_dur_ns = data.get("prompt_eval_duration")
    load_dur_ns = data.get("load_duration")

    obs_decode_ms = round(eval_dur_ns / 1e6, 2) if eval_dur_ns else None
    obs_prefill_ms = round(prompt_dur_ns / 1e6, 2) if prompt_dur_ns else None
    obs_load_ms = round(load_dur_ns / 1e6, 2) if load_dur_ns else None

    obs_decode_tps = (
        round(eval_count / (eval_dur_ns / 1e9), 2)
        if (eval_count and eval_dur_ns and eval_dur_ns > 0)
        else None
    )
    obs_e2e_tps = (
        round(eval_count / (obs_e2e_ms / 1000.0), 2)
        if (eval_count and obs_e2e_ms > 0)
        else None
    )

    # Compute errors vs prospective prediction
    dur_abs_err = (
        round(abs(obs_e2e_ms - request.expected_latency_ms), 2)
        if request.expected_latency_ms is not None
        else None
    )
    dur_rel_err = (
        round((dur_abs_err / request.expected_latency_ms) * 100.0, 2)
        if (dur_abs_err is not None and request.expected_latency_ms)
        else None
    )

    tps_abs_err = (
        round(abs(obs_decode_tps - request.expected_decode_throughput_tok_s), 2)
        if (obs_decode_tps is not None and request.expected_decode_throughput_tok_s is not None)
        else None
    )
    tps_rel_err = (
        round((tps_abs_err / request.expected_decode_throughput_tok_s) * 100.0, 2)
        if (tps_abs_err is not None and request.expected_decode_throughput_tok_s)
        else None
    )

    explanation = (
        f"Validation benchmark executed successfully. Observed end-to-end duration was {obs_e2e_ms} ms "
        f"(decode duration: {obs_decode_ms} ms, prefill: {obs_prefill_ms} ms, load: {obs_load_ms} ms). "
        f"Decode throughput was {obs_decode_tps} tok/s."
    )
    if dur_rel_err is not None:
        explanation += f" Prospective latency error was {dur_rel_err}% (absolute gap: {dur_abs_err} ms)."

    return ValidationResponse(
        validation_id=val_id,
        recommendation_id=request.recommendation_id,
        status="COMPLETED",
        executed_at=datetime.now(UTC).isoformat(),
        observed_client_e2e_duration_ms=obs_e2e_ms,
        observed_server_decode_duration_ms=obs_decode_ms,
        observed_server_prefill_duration_ms=obs_prefill_ms,
        observed_server_load_duration_ms=obs_load_ms,
        observed_server_decode_throughput_tok_s=obs_decode_tps,
        observed_client_e2e_output_throughput_tok_s=obs_e2e_tps,
        predicted_client_e2e_duration_ms=request.expected_latency_ms,
        predicted_server_decode_throughput_tok_s=request.expected_decode_throughput_tok_s,
        duration_absolute_error_ms=dur_abs_err,
        duration_relative_error_pct=dur_rel_err,
        throughput_absolute_error_tok_s=tps_abs_err,
        throughput_relative_error_pct=tps_rel_err,
        explanation=explanation,
    )
# -----------------------------------------------------------------------------
# Session 24 — Hardware-Aware Runtime Recommendations & Validation
# -----------------------------------------------------------------------------


class RuntimeValidationRequest(BaseModel):
    """Request to validate a recommended runtime configuration against live Ollama."""

    recommendation_id: str
    model_id: str = "llama3.2:1b"
    num_thread: int = 8
    num_ctx: int = 4096
    prompt_tokens: int = Field(default=32, gt=0)
    expected_output_tokens: int = Field(default=16, gt=0)
    expected_latency_ms: float | None = None
    expected_decode_throughput_tok_s: float | None = None


class RuntimeValidationResponse(BaseModel):
    """Comparative results of runtime prediction vs observed live inference."""

    validation_id: str
    recommendation_id: str
    model_id: str
    num_thread: int
    num_ctx: int
    status: str
    executed_at: str
    observed_client_e2e_duration_ms: float
    observed_server_decode_duration_ms: float | None
    observed_server_prefill_duration_ms: float | None
    observed_server_load_duration_ms: float | None
    observed_server_decode_throughput_tok_s: float | None
    predicted_client_e2e_duration_ms: float | None
    predicted_server_decode_throughput_tok_s: float | None
    duration_absolute_error_ms: float | None
    duration_relative_error_pct: float | None
    throughput_absolute_error_tok_s: float | None
    throughput_relative_error_pct: float | None
    explanation: str


@router.post(
    "/runtime",
    response_model=RuntimeRecommendationResult,
    summary="Hardware-aware inference runtime configuration recommendation",
)
async def recommend_runtime_configuration(
    request: RuntimeRecommendationRequest,
) -> RuntimeRecommendationResult:
    """Evaluate workload and workstation CPU hardware to recommend optimal model, threads, and context."""
    if request.prompt_tokens <= 0 or request.expected_output_tokens <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt and output token counts must be strictly positive.",
        )
    return _runtime_recommender.recommend(request)


@router.get(
    "/runtime/scenarios",
    summary="List Session 24 runtime optimization scenarios",
)
async def list_runtime_scenarios() -> list[dict[str, Any]]:
    """Retrieve operational scenarios for runtime optimization and prospective validation."""
    return [
        {
            "scenario_id": "RUNTIME_SCENARIO_1_INTERACTIVE",
            "name": "Scenario 1 — Latency-Sensitive Interactive",
            "description": "Short query prioritizing low response time on Intel P-cores.",
            "request": {
                "prompt_tokens": 32,
                "expected_output_tokens": 16,
                "latency_sla_ms": 1200.0,
                "concurrency": 1,
                "objective": "MIN_LATENCY",
                "allow_model_selection": True,
            },
        },
        {
            "scenario_id": "RUNTIME_SCENARIO_2_SUSTAINED_GEN",
            "name": "Scenario 2 — Sustained Long Generation",
            "description": "Extended generation evaluated for conservative SLA compliance and sustained decode throughput.",
            "request": {
                "prompt_tokens": 45,
                "expected_output_tokens": 150,
                "latency_sla_ms": 6000.0,
                "concurrency": 1,
                "objective": "CONSERVATIVE_SLA",
                "allow_model_selection": True,
            },
        },
        {
            "scenario_id": "RUNTIME_SCENARIO_3_CONCURRENT_SERVING",
            "name": "Scenario 3 — Concurrent Request Serving",
            "description": "Multiple simultaneous requests testing aggregate throughput and single-slot queuing impact.",
            "request": {
                "prompt_tokens": 67,
                "expected_output_tokens": 64,
                "latency_sla_ms": 4000.0,
                "concurrency": 2,
                "objective": "MAX_THROUGHPUT",
                "allow_model_selection": True,
            },
        },
    ]


@router.post(
    "/runtime/validate",
    response_model=RuntimeValidationResponse,
    summary="Live prospective validation for runtime configuration recommendation",
)
async def validate_runtime_recommendation(
    request: RuntimeValidationRequest,
) -> RuntimeValidationResponse:
    """Execute live validation benchmark with recommended threads and context on Ollama."""
    val_id = f"val-rt-{uuid.uuid4().hex[:8]}"
    prompt_text = (
        "Respond with a brief explanation: "
        if request.prompt_tokens <= 40
        else "Explain the architectural advantages of transformer neural network inference in detail: "
    )

    payload = {
        "model": request.model_id,
        "prompt": prompt_text,
        "options": {
            "num_thread": request.num_thread,
            "num_ctx": request.num_ctx,
            "num_predict": request.expected_output_tokens,
            "temperature": 0.0,
        },
        "stream": False,
    }

    t_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:11434", trust_env=False, timeout=60.0
        ) as client:
            resp = await client.post("/api/generate", json=payload)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Ollama server returned HTTP {resp.status_code}: {resp.text}",
                )
            data = resp.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama runtime is unavailable on http://127.0.0.1:11434. Please ensure Ollama is running.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        )
    t_end = time.perf_counter()

    obs_e2e_ms = round((t_end - t_start) * 1000.0, 2)
    eval_count = data.get("eval_count", 0)
    eval_dur_ns = data.get("eval_duration")
    prompt_dur_ns = data.get("prompt_eval_duration")
    load_dur_ns = data.get("load_duration")

    obs_decode_ms = round(eval_dur_ns / 1e6, 2) if eval_dur_ns else None
    obs_prefill_ms = round(prompt_dur_ns / 1e6, 2) if prompt_dur_ns else None
    obs_load_ms = round(load_dur_ns / 1e6, 2) if load_dur_ns else None

    obs_decode_tps = (
        round(eval_count / (eval_dur_ns / 1e9), 2)
        if (eval_count and eval_dur_ns and eval_dur_ns > 0)
        else None
    )

    dur_abs_err = (
        round(abs(obs_e2e_ms - request.expected_latency_ms), 2)
        if request.expected_latency_ms is not None
        else None
    )
    dur_rel_err = (
        round((dur_abs_err / request.expected_latency_ms) * 100.0, 2)
        if (dur_abs_err is not None and request.expected_latency_ms)
        else None
    )

    tps_abs_err = (
        round(abs(obs_decode_tps - request.expected_decode_throughput_tok_s), 2)
        if (
            obs_decode_tps is not None
            and request.expected_decode_throughput_tok_s is not None
        )
        else None
    )
    tps_rel_err = (
        round((tps_abs_err / request.expected_decode_throughput_tok_s) * 100.0, 2)
        if (tps_abs_err is not None and request.expected_decode_throughput_tok_s)
        else None
    )

    explanation = (
        f"Runtime validation completed on {request.model_id} with {request.num_thread} threads. "
        f"Observed E2E duration: {obs_e2e_ms} ms (decode: {obs_decode_ms} ms, prefill: {obs_prefill_ms} ms). "
        f"Observed decode throughput: {obs_decode_tps} tok/s."
    )
    if dur_rel_err is not None:
        explanation += f" Latency error: {dur_rel_err}% (gap: {dur_abs_err} ms)."

    return RuntimeValidationResponse(
        validation_id=val_id,
        recommendation_id=request.recommendation_id,
        model_id=request.model_id,
        num_thread=request.num_thread,
        num_ctx=request.num_ctx,
        status="COMPLETED",
        executed_at=datetime.now(UTC).isoformat(),
        observed_client_e2e_duration_ms=obs_e2e_ms,
        observed_server_decode_duration_ms=obs_decode_ms,
        observed_server_prefill_duration_ms=obs_prefill_ms,
        observed_server_load_duration_ms=obs_load_ms,
        observed_server_decode_throughput_tok_s=obs_decode_tps,
        predicted_client_e2e_duration_ms=request.expected_latency_ms,
        predicted_server_decode_throughput_tok_s=request.expected_decode_throughput_tok_s,
        duration_absolute_error_ms=dur_abs_err,
        duration_relative_error_pct=dur_rel_err,
        throughput_absolute_error_tok_s=tps_abs_err,
        throughput_relative_error_pct=tps_rel_err,
        explanation=explanation,
    )
