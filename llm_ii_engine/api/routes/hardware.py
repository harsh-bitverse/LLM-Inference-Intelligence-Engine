"""Hardware capability and discovery inspection endpoints."""

from fastapi import APIRouter, Depends, Query

from llm_ii_engine.api.dependencies import get_repository
from llm_ii_engine.discovery.catalog import list_gpu_specs
from llm_ii_engine.discovery.models import GPUSpecification, HardwareCapabilityReport
from llm_ii_engine.discovery.profiler import discover_hardware
from llm_ii_engine.storage.repository import EngineRepository, EntityNotFoundError

router = APIRouter(prefix="/hardware", tags=["Hardware"])


@router.get("", response_model=HardwareCapabilityReport, summary="Get hardware discovery report")
async def get_hardware(
    discover: bool = Query(
        default=False,
        description="If True, force a fresh hardware discovery probe rather than returning the latest stored report",
    ),
    repo: EngineRepository = Depends(get_repository),
) -> HardwareCapabilityReport:
    """Retrieve current hardware capabilities, CPU, RAM, and GPU devices."""
    if discover:
        report = discover_hardware()
        repo.save_hardware_report(report)
        return report

    latest = repo.get_latest_hardware_report()
    if latest is not None:
        return latest

    # If no stored report, execute initial discovery
    report = discover_hardware()
    repo.save_hardware_report(report)
    return report


@router.get(
    "/reports",
    summary="List historical hardware discovery reports",
)
async def list_hardware_reports(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    repo: EngineRepository = Depends(get_repository),
) -> dict[str, object]:
    """List persisted hardware discovery reports with pagination."""
    items, total = repo.list_hardware_reports(limit=limit, offset=offset)
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/catalog",
    response_model=list[GPUSpecification],
    summary="List curated reference GPU catalog specifications",
)
async def get_hardware_catalog() -> list[GPUSpecification]:
    """Retrieve authoritative hardware catalog entries documenting theoretical GPU specifications."""
    return list_gpu_specs()


@router.get(
    "/{report_id}",
    response_model=HardwareCapabilityReport,
    summary="Retrieve hardware report by ID",
)
async def get_hardware_by_id(
    report_id: str,
    repo: EngineRepository = Depends(get_repository),
) -> HardwareCapabilityReport:
    """Retrieve a specific HardwareCapabilityReport by report_id."""
    report = repo.get_hardware_report(report_id)
    if not report:
        raise EntityNotFoundError(f"Hardware report '{report_id}' not found.")
    return report
