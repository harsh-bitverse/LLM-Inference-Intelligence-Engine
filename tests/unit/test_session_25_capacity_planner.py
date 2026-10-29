"""Comprehensive unit and integration test suite for Session 25 Capacity Planning & Admission."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.capacity.admission import SLAAdmissionController
from llm_ii_engine.capacity.protocol import (
    AdmissionDecision,
    AdmissionRequest,
    CapacityEvidenceTier,
    CapacityExperimentManifest,
    CapacityObservation,
    QueueingModelType,
    ServiceTimeDistribution,
    WorkloadCapacityReport,
)
from llm_ii_engine.capacity.queueing_model import CapacityPlanner, QueueingCalculator


@pytest.fixture
def test_client() -> TestClient:
    """Fixture providing a FastAPI TestClient configured with capacity routes."""
    app = create_app()
    return TestClient(app)


# =========================================================================
# 1. Service Time Distribution Calculations
# =========================================================================

def test_service_distribution_calculation_valid():
    """Verify statistical service distribution calculations."""
    samples = [100.0, 110.0, 95.0, 105.0, 100.0]
    dist = QueueingCalculator.calculate_distribution(
        service_times_ms=samples,
        prefill_times_ms=[20.0, 22.0, 19.0, 21.0, 20.0],
        decode_times_ms=[80.0, 88.0, 76.0, 84.0, 80.0],
        output_tokens=[16, 16, 16, 16, 16],
    )
    assert dist.sample_count == 5
    assert dist.mean_service_ms == pytest.approx(102.0, rel=1e-2)
    assert dist.min_service_ms == 95.0
    assert dist.max_service_ms == 110.0
    assert dist.coefficient_of_variation > 0.0
    assert dist.mean_prefill_ms == pytest.approx(20.4, rel=1e-2)
    assert dist.mean_decode_ms == pytest.approx(81.6, rel=1e-2)
    assert dist.mean_output_tokens == 16.0


def test_service_distribution_empty_raises():
    """Verify empty sample list raises ValueError."""
    with pytest.raises(ValueError, match="at least one observation"):
        QueueingCalculator.calculate_distribution(service_times_ms=[])


# =========================================================================
# 2. Queueing Calculator & Pollaczek-Khinchine Formulas
# =========================================================================

def test_queue_metrics_stable_load():
    """Verify M/G/1 Pollaczek-Khinchine queue delays under stable utilization (rho < 0.85)."""
    dist = ServiceTimeDistribution(
        sample_count=10,
        mean_service_ms=100.0,
        variance_service_ms=25.0,
        std_service_ms=5.0,
        coefficient_of_variation=0.05,
        median_service_ms=100.0,
        p90_service_ms=105.0,
        p95_service_ms=108.0,
        min_service_ms=90.0,
        max_service_ms=110.0,
        mean_prefill_ms=20.0,
        mean_decode_ms=80.0,
        mean_output_tokens=16.0,
    )
    # mu = 10 req/s. At arrival rate lambda = 5 req/s, rho = 0.50
    point = QueueingCalculator.estimate_queue_metrics(
        arrival_rate_rps=5.0,
        service_dist=dist,
        model_type=QueueingModelType.M_G_1_APPROX,
        latency_sla_ms=500.0,
    )
    assert point.offered_rps == 5.0
    assert point.completed_rps == 5.0
    assert point.estimated_utilization == pytest.approx(0.50, rel=1e-2)
    assert point.queue_buildup_detected is False
    assert point.mean_queue_delay_ms > 0.0
    assert point.sla_compliance_pct >= 95.0


def test_queue_metrics_overload_backlog():
    """Verify queue behavior when arrival rate exceeds service capacity (rho >= 1.0)."""
    dist = ServiceTimeDistribution(
        sample_count=10,
        mean_service_ms=200.0,
        variance_service_ms=100.0,
        std_service_ms=10.0,
        coefficient_of_variation=0.05,
        median_service_ms=200.0,
        p90_service_ms=210.0,
        p95_service_ms=215.0,
        min_service_ms=180.0,
        max_service_ms=220.0,
        mean_prefill_ms=40.0,
        mean_decode_ms=160.0,
        mean_output_tokens=20.0,
    )
    # mu = 5.0 req/s. Offered lambda = 8.0 req/s (rho = 1.60)
    point = QueueingCalculator.estimate_queue_metrics(
        arrival_rate_rps=8.0,
        service_dist=dist,
        model_type=QueueingModelType.M_G_1_APPROX,
        latency_sla_ms=400.0,
    )
    assert point.completed_rps <= 5.0
    assert point.estimated_utilization > 1.0
    assert point.queue_buildup_detected is True
    assert point.mean_queue_delay_ms > 500.0
    assert point.sla_compliance_pct < 50.0


# =========================================================================
# 3. Capacity Planner & Serving Curves
# =========================================================================

def test_capacity_planner_report_generation():
    """Verify full capacity report and curve trajectory generation."""
    dist = ServiceTimeDistribution(
        sample_count=10,
        mean_service_ms=100.0,
        variance_service_ms=25.0,
        std_service_ms=5.0,
        coefficient_of_variation=0.05,
        median_service_ms=100.0,
        p90_service_ms=105.0,
        p95_service_ms=108.0,
        min_service_ms=90.0,
        max_service_ms=110.0,
        mean_prefill_ms=20.0,
        mean_decode_ms=80.0,
        mean_output_tokens=16.0,
    )
    report = CapacityPlanner.generate_capacity_report(
        model_id="qwen2.5:0.5b",
        num_thread=8,
        workload_id="workload_v1_interactive_qa",
        service_distribution=dist,
        latency_sla_ms=1000.0,
    )
    assert isinstance(report, WorkloadCapacityReport)
    assert report.max_sustainable_rps == 10.0
    assert report.max_sustainable_rpm == 600.0
    assert report.saturation_knee_rps == pytest.approx(8.2, rel=1e-2)
    assert report.safe_operating_rps == pytest.approx(6.15, rel=1e-2)
    assert len(report.capacity_curve) == 8
    assert len(report.assumptions) >= 4


# =========================================================================
# 4. SLA-Aware Admission Controller Verdicts
# =========================================================================

def test_admission_immediate_accept():
    """Verify ACCEPT when slot is idle and predicted latency easily meets SLA."""
    controller = SLAAdmissionController()
    req = AdmissionRequest(
        model_id="qwen2.5:0.5b",
        num_thread=8,
        prompt_tokens=20,
        expected_output_tokens=16,
        latency_sla_ms=2000.0,
        current_in_flight=0,
        current_queue_depth=0,
    )
    result = controller.evaluate_admission(req)
    assert result.decision == AdmissionDecision.ACCEPT
    assert result.predicted_queue_delay_ms == 0.0
    assert result.sla_compliance_probability >= 0.95
    assert result.unverified_code_warning is False


def test_admission_queue_with_bounded_wait():
    """Verify QUEUE_WITH_BOUNDED_WAIT when slot is busy but queue wait is within SLA."""
    controller = SLAAdmissionController()
    req = AdmissionRequest(
        model_id="llama3.2:1b",
        num_thread=8,
        prompt_tokens=20,
        expected_output_tokens=16,
        latency_sla_ms=600.0,
        current_in_flight=1,
        current_queue_depth=2,  # 3 waiting requests * ~148.8ms = ~446ms wait + 148ms service = ~595ms
    )
    result = controller.evaluate_admission(req)
    assert result.decision == AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT
    assert result.predicted_queue_delay_ms > 0.0
    assert result.predicted_total_latency_ms <= 600.0
    assert result.sla_compliance_probability >= 0.70


def test_admission_reject_or_defer_sla_violation():
    """Verify REJECT_OR_DEFER when accumulated queue wait exceeds SLA."""
    controller = SLAAdmissionController()
    req = AdmissionRequest(
        model_id="llama3.2:1b",
        num_thread=8,
        prompt_tokens=20,
        expected_output_tokens=16,
        latency_sla_ms=300.0,
        current_in_flight=1,
        current_queue_depth=4,  # 5 waiting requests * ~148.8ms = ~744ms wait > 300ms SLA
    )
    result = controller.evaluate_admission(req)
    assert result.decision == AdmissionDecision.REJECT_OR_DEFER
    assert result.predicted_total_latency_ms > 300.0
    assert "violates latency SLA" in result.rationale


def test_admission_unknown_model_insufficient_evidence():
    """Verify INSUFFICIENT_EVIDENCE when model/thread profile is uncalibrated."""
    controller = SLAAdmissionController()
    req = AdmissionRequest(
        model_id="unknown_model:70b",
        num_thread=8,
        prompt_tokens=50,
        expected_output_tokens=50,
        latency_sla_ms=2000.0,
    )
    result = controller.evaluate_admission(req)
    assert result.decision == AdmissionDecision.INSUFFICIENT_EVIDENCE
    assert result.evidence_tier == CapacityEvidenceTier.UNKNOWN_CAPACITY


def test_admission_session_23_code_warning():
    """Verify fail-closed code security warning is preserved when requested."""
    controller = SLAAdmissionController()
    req = AdmissionRequest(
        model_id="qwen2.5:0.5b",
        num_thread=8,
        prompt_tokens=20,
        expected_output_tokens=16,
        latency_sla_ms=2000.0,
        require_code_execution_pass=True,
    )
    result = controller.evaluate_admission(req)
    assert result.unverified_code_warning is True
    assert "fail-closed security mode" in result.rationale


# =========================================================================
# 5. REST API Integration Endpoints
# =========================================================================

def test_api_capacity_estimate(test_client: TestClient):
    """Verify POST /api/v1/capacity/estimate endpoint."""
    payload = {
        "model_id": "qwen2.5:0.5b",
        "num_thread": 8,
        "workload_id": "workload_v1_interactive_qa",
        "latency_sla_ms": 2000.0,
    }
    resp = test_client.post("/api/v1/capacity/estimate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "qwen2.5:0.5b"
    assert data["num_thread"] == 8
    assert data["max_sustainable_rps"] > 0.0
    assert data["saturation_knee_rps"] > 0.0
    assert len(data["capacity_curve"]) > 0


def test_api_capacity_admit(test_client: TestClient):
    """Verify POST /api/v1/capacity/admit endpoint."""
    payload = {
        "model_id": "qwen2.5:0.5b",
        "num_thread": 8,
        "prompt_tokens": 30,
        "expected_output_tokens": 16,
        "latency_sla_ms": 2000.0,
        "current_in_flight": 0,
        "current_queue_depth": 0,
    }
    resp = test_client.post("/api/v1/capacity/admit", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "ACCEPT"
    assert data["sla_compliance_probability"] >= 0.95


def test_api_capacity_scenarios(test_client: TestClient):
    """Verify GET /api/v1/capacity/scenarios endpoint."""
    resp = test_client.get("/api/v1/capacity/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert "scenarios" in data
    assert len(data["scenarios"]) == 3
    scenario_ids = [s["scenario_id"] for s in data["scenarios"]]
    assert "SCENARIO_A_LOW_TRAFFIC" in scenario_ids
    assert "SCENARIO_B_NEAR_SATURATION" in scenario_ids
    assert "SCENARIO_C_OVERLOAD" in scenario_ids


def test_api_capacity_validate_scenario(test_client: TestClient):
    """Verify POST /api/v1/capacity/validate endpoint."""
    payload = {
        "scenario_id": "SCENARIO_A_LOW_TRAFFIC",
        "current_in_flight": 0,
        "current_queue_depth": 0,
    }
    resp = test_client.post("/api/v1/capacity/validate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision_matches_expectation"] is True
    assert data["evaluation"]["decision"] == "ACCEPT"


# =========================================================================
# 6. Benchmark Data & Manifest Provenance Verification
# =========================================================================

def test_session_25_manifest_and_observations_exist():
    """Verify empirical manifest and observation records on disk."""
    manifest_path = Path("storage/experiments/session_25_capacity_manifest.json")
    obs_path = Path("storage/experiments/session_25_capacity_observations.json")
    report_path = Path("storage/experiments/session_25_queueing_analysis.md")

    assert manifest_path.exists(), "Manifest file must exist"
    assert obs_path.exists(), "Observations file must exist"
    assert report_path.exists(), "Queueing analysis report must exist"

    with open(manifest_path, encoding="utf-8") as f:
        m_data = json.load(f)
    manifest = CapacityExperimentManifest(**m_data)
    assert manifest.session_id == "session_25"
    assert len(manifest.observations) >= 60

    with open(obs_path, encoding="utf-8") as f:
        obs_data = json.load(f)
    assert len(obs_data) >= 60
    _ = CapacityObservation(**obs_data[0])


def test_session_25_prospective_validation_artifacts():
    """Verify pre-registered predictions and prospective validation report."""
    predictions_path = Path("storage/experiments/session_25_prospective_predictions.json")
    validation_report_path = Path("storage/experiments/session_25_prospective_validation_report.md")

    assert predictions_path.exists(), "Predictions lock file must exist"
    assert validation_report_path.exists(), "Validation report must exist"

    with open(predictions_path, encoding="utf-8") as f:
        pred_data = json.load(f)
    assert pred_data["status"] == "LOCKED_PRE_EXECUTION"
    assert len(pred_data["scenarios"]) == 3

    report_content = validation_report_path.read_text(encoding="utf-8")
    assert "VALIDATION_EXECUTED" in report_content
    assert "SCENARIO_A_LOW_TRAFFIC" in report_content
    assert "SCENARIO_B_NEAR_SATURATION" in report_content
    assert "SCENARIO_C_OVERLOAD" in report_content
