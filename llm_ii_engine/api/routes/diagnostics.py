"""Evidence-based bottleneck diagnostics endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from llm_ii_engine.analysis.diagnostics import DiagnosticEngine
from llm_ii_engine.analysis.models import DiagnosticReport
from llm_ii_engine.api.dependencies import get_repository
from llm_ii_engine.core.schemas import ModelSpec
from llm_ii_engine.storage.repository import EngineRepository, EntityNotFoundError

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


class DiagnosticRunRequest(BaseModel):
    """Specification to execute evidence-based diagnostic rules on benchmark measurements."""

    suite_id: str = Field(description="Identifier of the benchmark suite to analyze")
    model: ModelSpec | None = Field(
        default=None,
        description="Optional explicit model specification if additional architectural details are known",
    )


@router.post(
    "/run",
    response_model=DiagnosticReport,
    summary="Execute evidence-based diagnostic analysis on a benchmark suite",
)
async def run_diagnostics(
    req: DiagnosticRunRequest,
    repo: EngineRepository = Depends(get_repository),
) -> DiagnosticReport:
    """Analyze a completed benchmark suite to detect memory, compute, and latency bottlenecks."""
    suite = repo.get_benchmark_suite(req.suite_id)
    if not suite:
        raise EntityNotFoundError(f"Benchmark suite '{req.suite_id}' not found.")

    model_spec = req.model or ModelSpec(
        model_id=suite.model_id,
    )
    hardware_report = repo.get_latest_hardware_report()

    report = DiagnosticEngine.analyze(
        benchmark_suite=suite,
        model_spec=model_spec,
        hardware_report=hardware_report,
    )

    repo.save_diagnostic_report(report, benchmark_suite_id=req.suite_id)
    return report


@router.get(
    "",
    summary="List persisted diagnostic reports",
)
async def list_diagnostic_reports(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    repo: EngineRepository = Depends(get_repository),
) -> dict[str, Any]:
    """List historical diagnostic reports with pagination."""
    items, total = repo.list_diagnostic_reports(limit=limit, offset=offset)
    return {
        "items": [item.model_dump() for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/{report_id}",
    response_model=DiagnosticReport,
    summary="Retrieve a diagnostic report by ID",
)
async def get_diagnostic_report(
    report_id: str,
    repo: EngineRepository = Depends(get_repository),
) -> DiagnosticReport:
    """Retrieve an evidence-backed diagnostic report with findings and roofline calculations."""
    report = repo.get_diagnostic_report(report_id)
    if not report:
        raise EntityNotFoundError(f"Diagnostic report '{report_id}' not found.")
    return report
