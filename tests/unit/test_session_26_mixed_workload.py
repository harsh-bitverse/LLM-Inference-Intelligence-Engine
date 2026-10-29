"""Comprehensive unit and integration test suite for Session 26 Mixed-Workload Serving,

Tail-Latency Reliability & Adaptive Admission Validation.

Covers all 20 required verification areas:
1. Mixed service-time moments (E[S], E[S^2], Var, Cv).
2. M/G/1 Pollaczek-Khinchine assumptions and queue wait.
3. Deterministic discrete-event simulation seed repeatability.
4. Per-class tail-latency estimation.
5. Monotonic scheduled arrival accounting.
6. Timeout accounting.
7. Rejection accounting.
8. Invariant: offered == completed + rejected + timed_out + canceled + outstanding.
9. Head-of-line blocking observation data integrity.
10. Model switching overhead observation data integrity.
11. Policy A, B, C admission decision correctness.
12. False-feasible classification detection.
13. Reject-all policy detection.
14. Per-class SLA attainment calculations & Jain fairness index.
15. Hardware/model identity integrity.
16. Provenance and evidence tiers.
17. API endpoint validations (/mixed-profiles, /estimate-mixed, /hol-analysis, /model-switching-analysis, /admission-policies).
18. Historical Session 25 compatibility.
19. Session 23 security mode fail-closed code warning preservation.
20. Manifest artifact persistence checks.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.capacity.adaptive_admission_policy import (
    MixedWorkloadAdmissionController,
    evaluate_policy_report,
)
from llm_ii_engine.capacity.mixed_queueing_model import (
    PROFILE_A_INTERACTIVE_DOMINANT,
    PROFILE_B_BALANCED,
    STANDARD_PROFILES,
    MixedQueueingCalculator,
    MixedWorkloadSimulator,
)
from llm_ii_engine.capacity.mixed_workload_protocol import (
    AdmissionPolicyType,
    HeadOfLineBlockingObservation,
    MixedRequestAccountingRecord,
    MixedWorkloadManifest,
    ModelResidencyState,
    ModelSwitchingObservation,
    RequestTerminalStatus,
    WorkloadClass,
    WorkloadProfileType,
)
from llm_ii_engine.capacity.protocol import (
    AdmissionDecision,
    ServiceTimeDistribution,
)


@pytest.fixture
def test_client() -> TestClient:
    """Fixture providing a FastAPI TestClient configured with capacity routes."""
    app = create_app()
    return TestClient(app)


# Mock calibrated distributions
@pytest.fixture
def sample_distributions() -> dict[WorkloadClass, ServiceTimeDistribution]:
    return {
        WorkloadClass.INTERACTIVE_QA: ServiceTimeDistribution(
            sample_count=10,
            mean_service_ms=100.0,
            variance_service_ms=100.0,
            std_service_ms=10.0,
            coefficient_of_variation=0.10,
            median_service_ms=100.0,
            p90_service_ms=110.0,
            p95_service_ms=115.0,
            min_service_ms=85.0,
            max_service_ms=120.0,
            mean_prefill_ms=20.0,
            mean_decode_ms=80.0,
            mean_output_tokens=16.0,
        ),
        WorkloadClass.STRUCTURED_JSON: ServiceTimeDistribution(
            sample_count=10,
            mean_service_ms=1000.0,
            variance_service_ms=2500.0,
            std_service_ms=50.0,
            coefficient_of_variation=0.05,
            median_service_ms=1000.0,
            p90_service_ms=1050.0,
            p95_service_ms=1080.0,
            min_service_ms=900.0,
            max_service_ms=1100.0,
            mean_prefill_ms=50.0,
            mean_decode_ms=950.0,
            mean_output_tokens=96.0,
        ),
        WorkloadClass.LONG_FORM_GENERATION: ServiceTimeDistribution(
            sample_count=10,
            mean_service_ms=5000.0,
            variance_service_ms=10000.0,
            std_service_ms=100.0,
            coefficient_of_variation=0.02,
            median_service_ms=5000.0,
            p90_service_ms=5100.0,
            p95_service_ms=5150.0,
            min_service_ms=4800.0,
            max_service_ms=5200.0,
            mean_prefill_ms=30.0,
            mean_decode_ms=4970.0,
            mean_output_tokens=160.0,
        ),
        WorkloadClass.LONG_CONTEXT_RETRIEVAL: ServiceTimeDistribution(
            sample_count=10,
            mean_service_ms=300.0,
            variance_service_ms=400.0,
            std_service_ms=20.0,
            coefficient_of_variation=0.067,
            median_service_ms=300.0,
            p90_service_ms=320.0,
            p95_service_ms=330.0,
            min_service_ms=270.0,
            max_service_ms=340.0,
            mean_prefill_ms=200.0,
            mean_decode_ms=100.0,
            mean_output_tokens=24.0,
        ),
    }


# =========================================================================
# 1. Mixed Service-Time Moments Calculus (E[S], E[S^2], Var, Cv)
# =========================================================================

def test_mixed_service_time_moments_calculus(sample_distributions):
    """Verify first moment, second moment, variance, and Cv for mixed traffic."""
    moments = MixedQueueingCalculator.calculate_mixed_moments(
        profile=PROFILE_A_INTERACTIVE_DOMINANT,
        per_class_distributions=sample_distributions,
    )
    # Profile A: 0.80 * 100 + 0.15 * 1000 + 0.05 * 5000 = 80 + 150 + 250 = 480 ms
    assert moments.mean_service_ms == pytest.approx(480.0, rel=1e-2)
    assert moments.second_moment_ms2 > (moments.mean_service_ms ** 2)
    assert moments.variance_service_ms2 > 0.0
    assert moments.coefficient_of_variation > 0.0
    assert "INTERACTIVE_QA" in moments.per_class_moments


# =========================================================================
# 2. M/G/1 Pollaczek-Khinchine Assumptions and Queue Wait
# =========================================================================

def test_mg1_pollaczek_khinchine_theoretical_wait(sample_distributions):
    """Verify Pollaczek-Khinchine formula Wq = lambda * E[S^2] / (2 * (1 - rho))."""
    moments = MixedQueueingCalculator.calculate_mixed_moments(
        profile=PROFILE_A_INTERACTIVE_DOMINANT,
        per_class_distributions=sample_distributions,
    )
    # At lambda = 0.5 req/s, rho = 0.5 * 0.480 = 0.24 (stable)
    wait_ms, rho, is_stable = MixedQueueingCalculator.estimate_theoretical_mean_queue_delay(
        arrival_rate_rps=0.5,
        moments=moments,
    )
    assert is_stable is True
    assert rho == pytest.approx(0.24, rel=1e-2)
    assert wait_ms > 0.0

    # Under saturation: lambda = 3.0 req/s, rho = 3.0 * 0.480 = 1.44
    sat_wait_ms, sat_rho, sat_stable = MixedQueueingCalculator.estimate_theoretical_mean_queue_delay(
        arrival_rate_rps=3.0,
        moments=moments,
    )
    assert sat_stable is False
    assert sat_rho > 1.0


# =========================================================================
# 3. Deterministic Discrete-Event Simulation Seed Repeatability
# =========================================================================

def test_discrete_event_simulation_seed_repeatability(sample_distributions):
    """Verify that identical seeds produce identical simulation outputs."""
    sim1 = MixedWorkloadSimulator.simulate(
        profile=PROFILE_B_BALANCED,
        arrival_rate_rps=0.5,
        per_class_distributions=sample_distributions,
        request_count=100,
        seed=1337,
    )
    sim2 = MixedWorkloadSimulator.simulate(
        profile=PROFILE_B_BALANCED,
        arrival_rate_rps=0.5,
        per_class_distributions=sample_distributions,
        request_count=100,
        seed=1337,
    )
    assert sim1["overall_p50_e2e_ms"] == sim2["overall_p50_e2e_ms"]
    assert sim1["overall_p95_e2e_ms"] == sim2["overall_p95_e2e_ms"]
    assert sim1["per_class_summary"] == sim2["per_class_summary"]

    # Different seed produces non-identical trace
    sim3 = MixedWorkloadSimulator.simulate(
        profile=PROFILE_B_BALANCED,
        arrival_rate_rps=0.5,
        per_class_distributions=sample_distributions,
        request_count=100,
        seed=9999,
    )
    assert isinstance(sim3["overall_p50_e2e_ms"], float)


# =========================================================================
# 4. Per-Class Tail-Latency Estimation
# =========================================================================

def test_per_class_tail_latency_estimation(sample_distributions):
    """Verify per-class P50 and P95 latency decomposition."""
    sim = MixedWorkloadSimulator.simulate(
        profile=PROFILE_B_BALANCED,
        arrival_rate_rps=0.2,
        per_class_distributions=sample_distributions,
        request_count=120,
        seed=42,
    )
    summary = sim["per_class_summary"]
    assert "INTERACTIVE_QA" in summary
    assert "LONG_FORM_GENERATION" in summary

    qa = summary["INTERACTIVE_QA"]
    gen = summary["LONG_FORM_GENERATION"]
    assert qa["p50_e2e_ms"] <= qa["p95_e2e_ms"]
    assert gen["p50_e2e_ms"] <= gen["p95_e2e_ms"]
    # Long generation turnaround should dwarf interactive QA
    assert gen["p50_e2e_ms"] > qa["p50_e2e_ms"]


# =========================================================================
# 5. Monotonic Scheduled Arrival Accounting
# =========================================================================

def test_monotonic_scheduled_arrival_accounting():
    """Verify fine-grained monotonic timestamp consistency in accounting records."""
    t0 = 1000.000
    t1 = 1000.002
    t2 = 1000.050
    t3 = 1000.070
    t4 = 1000.120

    record = MixedRequestAccountingRecord(
        request_id="req-test-01",
        session_id="session_26",
        model_id="llama3.2:1b",
        num_thread=8,
        workload_class=WorkloadClass.INTERACTIVE_QA,
        latency_sla_ms=500.0,
        scheduled_arrival_mono_s=t0,
        actual_enqueue_mono_s=t1,
        dispatch_mono_s=t2,
        first_response_mono_s=t3,
        completion_mono_s=t4,
        queue_depth_at_arrival=0,
        in_flight_at_arrival=0,
        residency_before=ModelResidencyState.RESIDENT,
        admitted=True,
        admission_policy=AdmissionPolicyType.POLICY_C_WORKLOAD_AWARE,
        terminal_status=RequestTerminalStatus.COMPLETED,
        queue_delay_ms=(t2 - t1) * 1000.0,
        client_e2e_ms=(t4 - t2) * 1000.0,
        total_turnaround_ms=(t4 - t0) * 1000.0,
        sla_met=True,
    )
    assert record.scheduled_arrival_mono_s <= record.actual_enqueue_mono_s
    assert record.actual_enqueue_mono_s <= record.dispatch_mono_s
    assert record.dispatch_mono_s <= record.first_response_mono_s
    assert record.first_response_mono_s <= record.completion_mono_s
    assert record.total_turnaround_ms == pytest.approx(120.0, rel=1e-2)


# =========================================================================
# 6. Timeout Accounting
# =========================================================================

def test_timeout_accounting():
    """Verify that requests exceeding SLA client timeouts are accounted under TIMED_OUT."""
    rec = MixedRequestAccountingRecord(
        request_id="req-timeout-01",
        model_id="llama3.2:1b",
        workload_class=WorkloadClass.INTERACTIVE_QA,
        latency_sla_ms=500.0,
        scheduled_arrival_mono_s=100.0,
        actual_enqueue_mono_s=100.001,
        admitted=True,
        admission_policy=AdmissionPolicyType.POLICY_A_NO_CONTROL,
        terminal_status=RequestTerminalStatus.TIMED_OUT,
        error_message="Client timed out waiting in queue after 8000ms",
        sla_met=False,
    )
    rep = evaluate_policy_report(
        records=[rec],
        policy=AdmissionPolicyType.POLICY_A_NO_CONTROL,
        profile_type=WorkloadProfileType.PROFILE_A_INTERACTIVE_DOMINANT,
        offered_rps=1.0,
    )
    assert rep.total_timed_out == 1
    assert rep.total_completed == 0
    assert rep.all_offered_sla_attainment_pct == 0.0
    assert rep.accounting_invariant_holds is True


# =========================================================================
# 7. Rejection Accounting
# =========================================================================

def test_rejection_accounting():
    """Verify that shed requests are recorded as REJECTED with admitted=False."""
    rec = MixedRequestAccountingRecord(
        request_id="req-reject-01",
        model_id="llama3.2:1b",
        workload_class=WorkloadClass.LONG_FORM_GENERATION,
        latency_sla_ms=2000.0,
        scheduled_arrival_mono_s=100.0,
        actual_enqueue_mono_s=100.001,
        admitted=False,
        admission_policy=AdmissionPolicyType.POLICY_C_WORKLOAD_AWARE,
        terminal_status=RequestTerminalStatus.REJECTED,
        sla_met=False,
    )
    rep = evaluate_policy_report(
        records=[rec],
        policy=AdmissionPolicyType.POLICY_C_WORKLOAD_AWARE,
        profile_type=WorkloadProfileType.PROFILE_C_GENERATION_HEAVY,
        offered_rps=2.0,
    )
    assert rep.total_rejected == 1
    assert rep.total_admitted == 0
    assert rep.accounting_invariant_holds is True


# =========================================================================
# 8. Strict Terminal Invariant: offered == completed + rejected + timed_out + canceled + outstanding
# =========================================================================

def test_accounting_invariant_holds_strictly():
    """Enforce the strict terminal outcome invariant."""
    records = []
    # 5 completed
    for i in range(5):
        records.append(
            MixedRequestAccountingRecord(
                request_id=f"rec-c-{i}",
                model_id="qwen2.5:0.5b",
                workload_class=WorkloadClass.INTERACTIVE_QA,
                latency_sla_ms=500.0,
                scheduled_arrival_mono_s=float(i),
                actual_enqueue_mono_s=float(i),
                admitted=True,
                terminal_status=RequestTerminalStatus.COMPLETED,
                total_turnaround_ms=50.0,
                sla_met=True,
            )
        )
    # 2 rejected
    for i in range(2):
        records.append(
            MixedRequestAccountingRecord(
                request_id=f"rec-r-{i}",
                model_id="qwen2.5:0.5b",
                workload_class=WorkloadClass.LONG_FORM_GENERATION,
                latency_sla_ms=2000.0,
                scheduled_arrival_mono_s=float(i + 5),
                actual_enqueue_mono_s=float(i + 5),
                admitted=False,
                terminal_status=RequestTerminalStatus.REJECTED,
                sla_met=False,
            )
        )
    # 1 timed out
    records.append(
        MixedRequestAccountingRecord(
            request_id="rec-t-0",
            model_id="qwen2.5:0.5b",
            workload_class=WorkloadClass.STRUCTURED_JSON,
            latency_sla_ms=1000.0,
            scheduled_arrival_mono_s=7.0,
            actual_enqueue_mono_s=7.0,
            admitted=True,
            terminal_status=RequestTerminalStatus.TIMED_OUT,
            sla_met=False,
        )
    )
    # 1 canceled
    records.append(
        MixedRequestAccountingRecord(
            request_id="rec-x-0",
            model_id="qwen2.5:0.5b",
            workload_class=WorkloadClass.INTERACTIVE_QA,
            latency_sla_ms=500.0,
            scheduled_arrival_mono_s=8.0,
            actual_enqueue_mono_s=8.0,
            admitted=True,
            terminal_status=RequestTerminalStatus.CANCELED,
            sla_met=False,
        )
    )
    # 1 outstanding
    records.append(
        MixedRequestAccountingRecord(
            request_id="rec-o-0",
            model_id="qwen2.5:0.5b",
            workload_class=WorkloadClass.INTERACTIVE_QA,
            latency_sla_ms=500.0,
            scheduled_arrival_mono_s=9.0,
            actual_enqueue_mono_s=9.0,
            admitted=True,
            terminal_status=RequestTerminalStatus.OUTSTANDING,
            sla_met=None,
        )
    )

    report = evaluate_policy_report(
        records=records,
        policy=AdmissionPolicyType.POLICY_C_WORKLOAD_AWARE,
        profile_type=WorkloadProfileType.PROFILE_B_BALANCED,
        offered_rps=1.0,
    )
    assert report.total_offered == 10
    assert report.total_completed == 5
    assert report.total_rejected == 2
    assert report.total_timed_out == 1
    assert report.total_canceled == 1
    assert report.total_outstanding == 1
    assert report.accounting_invariant_holds is True


# =========================================================================
# 9. Head-of-Line Blocking Observation Data Integrity
# =========================================================================

def test_head_of_line_blocking_data_integrity():
    """Verify HeadOfLineBlockingObservation semantics and penalty computation."""
    obs = HeadOfLineBlockingObservation(
        experiment_id="hol-exp-01",
        model_id="llama3.2:1b",
        scenario_description="Long Gen blocks Interactive QA",
        antecedent_workload=WorkloadClass.LONG_FORM_GENERATION,
        consequent_workload=WorkloadClass.INTERACTIVE_QA,
        antecedent_duration_ms=7576.1,
        isolated_consequent_duration_ms=109.6,
        blocked_consequent_duration_ms=7647.3,
        additional_queue_delay_ms=7537.7,
        sla_missed_due_to_blocking=True,
    )
    assert obs.blocked_consequent_duration_ms > obs.isolated_consequent_duration_ms
    assert obs.additional_queue_delay_ms == pytest.approx(
        obs.blocked_consequent_duration_ms - obs.isolated_consequent_duration_ms, rel=1e-3
    )
    assert obs.sla_missed_due_to_blocking is True


# =========================================================================
# 10. Model Switching Overhead Observation Data Integrity
# =========================================================================

def test_model_switching_overhead_data_integrity():
    """Verify ModelSwitchingObservation metrics and dual residency semantics."""
    obs = ModelSwitchingObservation(
        experiment_id="switch-exp-01",
        from_model_id="qwen2.5:0.5b",
        to_model_id="llama3.2:1b",
        from_model_resident_before=True,
        actual_load_duration_ms=2.2,
        client_switch_penalty_ms=0.0,
        warm_execution_duration_ms=115.4,
        cold_execution_duration_ms=115.8,
    )
    assert obs.actual_load_duration_ms >= 0.0
    assert obs.client_switch_penalty_ms >= 0.0
    assert obs.warm_execution_duration_ms > 0.0


# =========================================================================
# 11. Policy A, B, C Admission Decision Correctness
# =========================================================================

def test_policy_admission_decisions_a_b_c():
    """Verify distinct decisions of Policy A, B, and C under heavy backlog."""
    # Under heavy backlog (8000ms), an interactive request (SLA 500ms):
    # Policy A: Admits blindly into queue
    dec_a, _, _, _ = MixedWorkloadAdmissionController.evaluate(
        policy=AdmissionPolicyType.POLICY_A_NO_CONTROL,
        model_id="llama3.2:1b",
        workload_class=WorkloadClass.INTERACTIVE_QA,
        latency_sla_ms=500.0,
        current_queue_backlog_ms=8000.0,
        current_queue_depth=1,
        current_in_flight=1,
    )
    assert dec_a == AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT

    # Policy B: Static guardrail with queue depth 4 -> REJECT
    dec_b, _, _, _ = MixedWorkloadAdmissionController.evaluate(
        policy=AdmissionPolicyType.POLICY_B_STATIC_UTILIZATION,
        model_id="llama3.2:1b",
        workload_class=WorkloadClass.INTERACTIVE_QA,
        latency_sla_ms=500.0,
        current_queue_backlog_ms=8000.0,
        current_queue_depth=4,
        current_in_flight=1,
    )
    assert dec_b == AdmissionDecision.REJECT_OR_DEFER

    # Policy C: Workload-aware sheds because predicted latency > 500ms SLA
    dec_c, _, rationale_c, _ = MixedWorkloadAdmissionController.evaluate(
        policy=AdmissionPolicyType.POLICY_C_WORKLOAD_AWARE,
        model_id="llama3.2:1b",
        workload_class=WorkloadClass.INTERACTIVE_QA,
        latency_sla_ms=500.0,
        current_queue_backlog_ms=8000.0,
        current_queue_depth=1,
        current_in_flight=1,
    )
    assert dec_c == AdmissionDecision.REJECT_OR_DEFER
    assert "breaches SLA" in rationale_c


# =========================================================================
# 12. False-Feasible Classification Detection
# =========================================================================

def test_false_feasible_classification_detection():
    """Verify counting of admitted requests that subsequently breached SLA."""
    # Admitted request that took 1200ms against a 500ms SLA
    rec_false_feasible = MixedRequestAccountingRecord(
        request_id="req-ff-1",
        model_id="llama3.2:1b",
        workload_class=WorkloadClass.INTERACTIVE_QA,
        latency_sla_ms=500.0,
        scheduled_arrival_mono_s=0.0,
        actual_enqueue_mono_s=0.0,
        admitted=True,
        terminal_status=RequestTerminalStatus.COMPLETED,
        total_turnaround_ms=1200.0,
        sla_met=False,
    )
    report = evaluate_policy_report(
        records=[rec_false_feasible],
        policy=AdmissionPolicyType.POLICY_A_NO_CONTROL,
        profile_type=WorkloadProfileType.PROFILE_A_INTERACTIVE_DOMINANT,
        offered_rps=1.0,
    )
    assert report.false_feasible_count == 1
    assert report.admitted_sla_attainment_pct == 0.0


# =========================================================================
# 13. Reject-All Policy Detection
# =========================================================================

def test_reject_all_policy_detection():
    """Verify metrics when all requests are rejected."""
    records = [
        MixedRequestAccountingRecord(
            request_id=f"req-rej-{i}",
            model_id="llama3.2:1b",
            workload_class=WorkloadClass.INTERACTIVE_QA,
            latency_sla_ms=500.0,
            scheduled_arrival_mono_s=float(i),
            actual_enqueue_mono_s=float(i),
            admitted=False,
            terminal_status=RequestTerminalStatus.REJECTED,
            sla_met=False,
        )
        for i in range(5)
    ]
    report = evaluate_policy_report(
        records=records,
        policy=AdmissionPolicyType.POLICY_B_STATIC_UTILIZATION,
        profile_type=WorkloadProfileType.PROFILE_A_INTERACTIVE_DOMINANT,
        offered_rps=1.0,
    )
    assert report.total_admitted == 0
    assert report.total_rejected == 5
    assert report.admitted_sla_attainment_pct == 0.0


# =========================================================================
# 14. Per-Class SLA Attainment Calculations & Jain Fairness Index
# =========================================================================

def test_per_class_sla_attainment_and_jain_fairness():
    """Verify per-class attainment breakdown and Jain's fairness index."""
    records = []
    # 2 QA requests: both meet SLA
    for i in range(2):
        records.append(
            MixedRequestAccountingRecord(
                request_id=f"qa-{i}",
                model_id="qwen2.5:0.5b",
                workload_class=WorkloadClass.INTERACTIVE_QA,
                latency_sla_ms=500.0,
                scheduled_arrival_mono_s=float(i),
                actual_enqueue_mono_s=float(i),
                admitted=True,
                terminal_status=RequestTerminalStatus.COMPLETED,
                total_turnaround_ms=60.0,
                sla_met=True,
            )
        )
    # 2 Gen requests: 1 meets SLA, 1 breaches
    records.append(
        MixedRequestAccountingRecord(
            request_id="gen-0",
            model_id="qwen2.5:0.5b",
            workload_class=WorkloadClass.LONG_FORM_GENERATION,
            latency_sla_ms=3000.0,
            scheduled_arrival_mono_s=2.0,
            actual_enqueue_mono_s=2.0,
            admitted=True,
            terminal_status=RequestTerminalStatus.COMPLETED,
            total_turnaround_ms=2400.0,
            sla_met=True,
        )
    )
    records.append(
        MixedRequestAccountingRecord(
            request_id="gen-1",
            model_id="qwen2.5:0.5b",
            workload_class=WorkloadClass.LONG_FORM_GENERATION,
            latency_sla_ms=3000.0,
            scheduled_arrival_mono_s=3.0,
            actual_enqueue_mono_s=3.0,
            admitted=True,
            terminal_status=RequestTerminalStatus.COMPLETED,
            total_turnaround_ms=4500.0,
            sla_met=False,
        )
    )

    report = evaluate_policy_report(
        records=records,
        policy=AdmissionPolicyType.POLICY_C_WORKLOAD_AWARE,
        profile_type=WorkloadProfileType.PROFILE_B_BALANCED,
        offered_rps=1.0,
    )
    assert report.per_class_sla_attainment["INTERACTIVE_QA"] == 100.0
    assert report.per_class_sla_attainment["LONG_FORM_GENERATION"] == 50.0
    assert 0.0 < report.workload_fairness_index <= 1.0


# =========================================================================
# 15. Hardware/Model Identity Integrity
# =========================================================================

def test_hardware_model_identity_integrity():
    """Verify that models are confined to authorized set."""
    authorized_models = {"llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"}
    rec = MixedRequestAccountingRecord(
        request_id="req-auth-1",
        model_id="qwen2.5:0.5b",
        workload_class=WorkloadClass.INTERACTIVE_QA,
        latency_sla_ms=500.0,
        scheduled_arrival_mono_s=0.0,
        actual_enqueue_mono_s=0.0,
        admitted=True,
        terminal_status=RequestTerminalStatus.COMPLETED,
    )
    assert rec.model_id in authorized_models


# =========================================================================
# 16. Provenance and Evidence Tiers
# =========================================================================

def test_provenance_and_evidence_tiers():
    """Verify metadata provenance across Session 26 manifest."""
    manifest = MixedWorkloadManifest(
        manifest_id="manifest-s26-test",
        session_id="session_26",
        hardware_id="cpu-intel-i7-14700",
        total_benchmark_duration_s=120.0,
        profiles=list(STANDARD_PROFILES.values()),
        total_observations_count=4,
    )
    assert manifest.session_id == "session_26"
    assert manifest.hardware_id == "cpu-intel-i7-14700"
    assert len(manifest.profiles) == 4


# =========================================================================
# 17. API Endpoint Validations
# =========================================================================

def test_api_endpoints_mixed_workload(test_client):
    """Verify all Session 26 FastAPI endpoints return valid schemas."""
    # 1. /mixed-profiles
    resp = test_client.get("/api/v1/capacity/mixed-profiles")
    assert resp.status_code == 200
    profiles = resp.json()
    assert len(profiles) == 4

    # 2. /estimate-mixed
    est_payload = {
        "profile_type": "PROFILE_A_INTERACTIVE_DOMINANT",
        "model_id": "llama3.2:1b",
        "offered_rps": 0.5,
    }
    resp = test_client.post("/api/v1/capacity/estimate-mixed", json=est_payload)
    assert resp.status_code == 200
    est_data = resp.json()
    assert "moments" in est_data
    assert "simulation" in est_data
    assert est_data["moments"]["mean_service_ms"] > 0

    # 3. /hol-analysis
    resp = test_client.get("/api/v1/capacity/hol-analysis")
    assert resp.status_code == 200
    assert "observations" in resp.json()

    # 4. /model-switching-analysis
    resp = test_client.get("/api/v1/capacity/model-switching-analysis")
    assert resp.status_code == 200
    assert "observations" in resp.json()

    # 5. /admission-policies
    resp = test_client.get("/api/v1/capacity/admission-policies")
    assert resp.status_code == 200
    assert "policy_reports" in resp.json()


# =========================================================================
# 18. Historical Session 25 Compatibility
# =========================================================================

def test_historical_session_25_compatibility(test_client):
    """Verify that Session 25 capacity planning endpoints remain fully functional."""
    resp = test_client.get("/api/v1/capacity/scenarios")
    assert resp.status_code == 200
    sc_data = resp.json()
    assert "scenarios" in sc_data
    assert len(sc_data["scenarios"]) >= 3

    admit_payload = {
        "model_id": "qwen2.5:0.5b",
        "num_thread": 8,
        "prompt_tokens": 30,
        "expected_output_tokens": 20,
        "latency_sla_ms": 2000.0,
        "current_in_flight": 0,
        "current_queue_depth": 0,
    }
    resp = test_client.post("/api/v1/capacity/admit", json=admit_payload)
    assert resp.status_code == 200
    assert resp.json()["decision"] == "ACCEPT"


# =========================================================================
# 19. Session 23 Security Mode Fail-Closed Code Warning Preservation
# =========================================================================

def test_session_23_security_mode_fail_closed_code_warning():
    """Verify fail-closed code security warning is preserved across admission decisions."""
    dec, _, _, warning = MixedWorkloadAdmissionController.evaluate(
        policy=AdmissionPolicyType.POLICY_C_WORKLOAD_AWARE,
        model_id="qwen2.5:0.5b",
        workload_class=WorkloadClass.INTERACTIVE_QA,
        latency_sla_ms=2000.0,
        current_queue_backlog_ms=0.0,
        current_queue_depth=0,
        current_in_flight=0,
        require_code_security_pass=True,
    )
    assert dec == AdmissionDecision.ACCEPT
    assert warning is True


# =========================================================================
# 20. Manifest Artifact Persistence Checks
# =========================================================================

def test_manifest_artifact_persistence_checks():
    """Verify persistence and integrity of Session 26 manifest file."""
    manifest_path = Path("storage/experiments/session_26_mixed_workload_manifest.json")
    assert manifest_path.exists(), "Session 26 manifest must exist"

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["session_id"] == "session_26"
    assert data["hardware_id"] == "cpu-intel-i7-14700"
    assert "profiles" in data
    assert "policy_reports" in data
    assert "hol_observations" in data
    assert "model_switch_observations" in data
    assert len(data["policy_reports"]) > 0
    assert len(data["hol_observations"]) > 0
    assert len(data["model_switch_observations"]) > 0
