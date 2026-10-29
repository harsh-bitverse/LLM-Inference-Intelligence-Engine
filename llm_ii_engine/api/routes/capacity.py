"""Capacity planning and SLA-aware admission API endpoints for Session 25."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_ii_engine.capacity.admission import SLAAdmissionController
from llm_ii_engine.capacity.protocol import (
    AdmissionDecision,
    AdmissionRequest,
    AdmissionResult,
    CapacityEvidenceTier,
    QueueingModelType,
    ServiceTimeDistribution,
    WorkloadCapacityReport,
)
from llm_ii_engine.capacity.queueing_model import CapacityPlanner, QueueingCalculator

router = APIRouter(prefix="/capacity", tags=["capacity"])

MANIFEST_PATH = Path("storage/experiments/session_25_capacity_manifest.json")
admission_controller = SLAAdmissionController()


class CapacityEstimateRequest(BaseModel):
    """Request payload for capacity planning report generation."""

    model_id: str = Field(default="qwen2.5:0.5b")
    num_thread: int = Field(default=8, ge=1, le=28)
    workload_id: str = Field(default="workload_v1_interactive_qa")
    latency_sla_ms: float | None = Field(default=2000.0, gt=0.0)
    queueing_model: QueueingModelType = Field(default=QueueingModelType.M_G_1_APPROX)


def load_cached_manifest() -> dict[str, Any] | None:
    """Load cached Session 25 capacity experiment manifest if present."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@router.post("/estimate", response_model=WorkloadCapacityReport)
async def estimate_capacity(req: CapacityEstimateRequest) -> WorkloadCapacityReport:
    """Generate serving capacity curve and knee analysis for a model/workload configuration."""
    manifest = load_cached_manifest()
    service_dist: ServiceTimeDistribution | None = None
    evidence_tier = CapacityEvidenceTier.EMPIRICALLY_INTERPOLATED

    if manifest:
        stg_a = manifest.get("stage_a_service_distributions", {})
        model_tasks = stg_a.get(req.model_id, {})
        task_data = model_tasks.get(req.workload_id)
        if task_data:
            try:
                service_dist = ServiceTimeDistribution(**task_data)
                evidence_tier = CapacityEvidenceTier.DIRECT_MEASURED
            except Exception:
                pass

    if service_dist is None:
        # Calibrated fallback distribution based on Session 24 measurements
        base_ms = CapacityPlanner.DEFAULT_SERVICE_BASELINES.get(
            req.model_id, {}
        ).get(req.workload_id, 800.0)
        # Adjust for thread count if different from 8
        if req.num_thread == 4:
            base_ms *= 1.25
        elif req.num_thread == 2:
            base_ms *= 1.60
        elif req.num_thread > 8:
            base_ms *= 1.10  # E-core synchronization tax

        service_dist = QueueingCalculator.calculate_distribution(
            service_times_ms=[base_ms * 0.95, base_ms, base_ms * 1.05]
        )

    report = CapacityPlanner.generate_capacity_report(
        model_id=req.model_id,
        num_thread=req.num_thread,
        workload_id=req.workload_id,
        service_distribution=service_dist,
        latency_sla_ms=req.latency_sla_ms,
        model_type=req.queueing_model,
        evidence_tier=evidence_tier,
    )
    return report


@router.post("/admit", response_model=AdmissionResult)
async def admit_request(req: AdmissionRequest) -> AdmissionResult:
    """Evaluate SLA-aware admission decision for an incoming inference request."""
    return admission_controller.evaluate_admission(req)


@router.get("/scenarios")
async def get_capacity_scenarios() -> dict[str, Any]:
    """Retrieve pre-registered operational validation scenarios and cached manifest metadata."""
    manifest = load_cached_manifest()
    has_manifest = manifest is not None
    manifest_id = manifest.get("manifest_id") if manifest else None
    obs_count = len(manifest.get("observations", [])) if manifest else 0

    scenarios = [
        {
            "scenario_id": "SCENARIO_A_LOW_TRAFFIC",
            "name": "Scenario A: Low Traffic / Under-Saturated",
            "model_id": "qwen2.5:0.5b",
            "offered_rps": 0.5,
            "target_sla_ms": 2000.0,
            "expected_decision": AdmissionDecision.ACCEPT.value,
            "description": "Safe operating regime below knee; rho < 0.20, queue delay minimal, SLA met >95%.",
        },
        {
            "scenario_id": "SCENARIO_B_NEAR_SATURATION",
            "name": "Scenario B: Near Saturation Knee",
            "model_id": "llama3.2:1b",
            "offered_rps": 1.8,
            "target_sla_ms": 2500.0,
            "expected_decision": AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT.value,
            "description": "Offered rate approaches service capacity (rho ~ 0.85); queue delays emerge, tail risk elevated.",
        },
        {
            "scenario_id": "SCENARIO_C_OVERLOAD",
            "name": "Scenario C: Overload & Backlog Accumulation",
            "model_id": "llama3.2:1b",
            "offered_rps": 3.0,
            "target_sla_ms": 2500.0,
            "expected_decision": AdmissionDecision.REJECT_OR_DEFER.value,
            "description": "Offered rate exceeds service rate (rho > 1.0); queue backlog accumulates monotonically, SLA breaches guaranteed.",
        },
    ]

    return {
        "has_cached_manifest": has_manifest,
        "manifest_id": manifest_id,
        "total_observations": obs_count,
        "scenarios": scenarios,
    }


class ScenarioValidationRequest(BaseModel):
    """Payload to validate a prospective scenario against the admission engine."""

    scenario_id: str
    current_in_flight: int = 0
    current_queue_depth: int = 0


@router.post("/validate")
async def validate_scenario(req: ScenarioValidationRequest) -> dict[str, Any]:
    """Validate an operational scenario and return live admission evaluation."""
    scenarios_meta = await get_capacity_scenarios()
    target_scenario = None
    for s in scenarios_meta["scenarios"]:
        if s["scenario_id"] == req.scenario_id:
            target_scenario = s
            break

    if not target_scenario:
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found.")

    adm_req = AdmissionRequest(
        model_id=target_scenario["model_id"],
        num_thread=8,
        prompt_tokens=20,
        expected_output_tokens=16,
        latency_sla_ms=target_scenario["target_sla_ms"],
        current_in_flight=req.current_in_flight,
        current_queue_depth=req.current_queue_depth,
    )
    result = admission_controller.evaluate_admission(adm_req)

    return {
        "scenario": target_scenario,
        "evaluation": result.model_dump(),
        "decision_matches_expectation": result.decision.value == target_scenario["expected_decision"],
    }
