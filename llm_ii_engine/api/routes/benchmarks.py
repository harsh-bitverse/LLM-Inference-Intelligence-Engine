"""Benchmark suite orchestration, job telemetry streaming, and historical measurement inspection."""

from typing import Any

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse

from llm_ii_engine.api.dependencies import get_job_manager, get_repository
from llm_ii_engine.api.jobs import (
    BenchmarkJobManager,
    BenchmarkSubmissionRequest,
    JobStatus,
)
from llm_ii_engine.benchmark.models import BenchmarkSuiteResult
from llm_ii_engine.storage.repository import EngineRepository, EntityNotFoundError

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks"])


@router.post(
    "/run",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an asynchronous benchmark execution job",
)
async def submit_benchmark_job(
    request: BenchmarkSubmissionRequest,
    job_manager: BenchmarkJobManager = Depends(get_job_manager),
) -> dict[str, str]:
    """Submit a benchmark workload specification for background execution."""
    job_id = await job_manager.submit_job(request)
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": "Benchmark job submitted successfully.",
    }


@router.get(
    "/jobs",
    response_model=list[JobStatus],
    summary="List active and historical benchmark jobs",
)
async def list_jobs(
    limit: int = Query(default=50, ge=1, le=100),
    job_manager: BenchmarkJobManager = Depends(get_job_manager),
) -> list[JobStatus]:
    """List in-memory benchmark job lifecycle statuses."""
    return job_manager.list_jobs(limit=limit)


@router.get(
    "/jobs/{job_id}",
    summary="Check status and outcome of a benchmark job",
)
async def get_job_status(
    job_id: str,
    job_manager: BenchmarkJobManager = Depends(get_job_manager),
) -> dict[str, Any]:
    """Retrieve execution state, timing, and results for a submitted job."""
    job_status = job_manager.get_job_status(job_id)
    if not job_status:
        raise EntityNotFoundError(f"Benchmark job '{job_id}' not found.")

    res = job_manager.get_job_result(job_id)
    return {
        "job": job_status.model_dump(),
        "result": res.model_dump() if res else None,
    }


@router.post(
    "/jobs/{job_id}/cancel",
    summary="Cancel an active or pending benchmark job",
)
async def cancel_job(
    job_id: str,
    job_manager: BenchmarkJobManager = Depends(get_job_manager),
) -> dict[str, Any]:
    """Cancel a running benchmark job."""
    cancelled = await job_manager.cancel_job(job_id)
    if not cancelled:
        status_info = job_manager.get_job_status(job_id)
        if not status_info:
            raise EntityNotFoundError(f"Benchmark job '{job_id}' not found.")
        return {
            "job_id": job_id,
            "cancelled": False,
            "message": f"Job is already in terminal state: {status_info.status}",
        }

    return {
        "job_id": job_id,
        "cancelled": True,
        "message": "Benchmark job was cancelled.",
    }


@router.get(
    "/jobs/{job_id}/stream",
    summary="Server-Sent Events (SSE) telemetry stream for a benchmark job",
)
async def stream_job_events(
    job_id: str,
    job_manager: BenchmarkJobManager = Depends(get_job_manager),
) -> StreamingResponse:
    """Stream real-time TokenStreamEvent telemetry for an active benchmark job over SSE."""
    job_status = job_manager.get_job_status(job_id)
    if not job_status:
        raise EntityNotFoundError(f"Benchmark job '{job_id}' not found.")

    return StreamingResponse(
        job_manager.subscribe_events(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "",
    summary="List persisted benchmark suite runs",
)
async def list_benchmarks(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    model_id: str | None = Query(default=None),
    engine_type: str | None = Query(default=None),
    repo: EngineRepository = Depends(get_repository),
) -> dict[str, Any]:
    """List historical benchmark suites stored in the relational database."""
    items, total = repo.list_benchmark_suites(
        limit=limit,
        offset=offset,
        model_id=model_id,
        engine_type=engine_type,
    )
    return {
        "items": [item.model_dump() for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/{suite_id}",
    response_model=BenchmarkSuiteResult,
    summary="Retrieve a persisted benchmark suite result by suite_id",
)
async def get_benchmark_by_id(
    suite_id: str,
    repo: EngineRepository = Depends(get_repository),
) -> BenchmarkSuiteResult:
    """Retrieve full request-level measurements and aggregate metrics for a benchmark suite."""
    suite = repo.get_benchmark_suite(suite_id)
    if not suite:
        raise EntityNotFoundError(f"Benchmark suite '{suite_id}' not found.")
    return suite
