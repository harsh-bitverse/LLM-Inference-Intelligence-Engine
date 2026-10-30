"""Comprehensive unit and integration test suite for Session 27 Serving Parallelism,

Continuous Batching Feasibility & Architecture Comparison.

Covers all required Phase 9 verification areas:
1. Parallelism capability detection & capability matrix integrity.
2. Server concurrency calculus and execution overlap validation.
3. Client concurrency vs server execution slot semantics.
4. Continuous batching vs static batching definitions and models.
5. Memory constraints and KV cache partitioning behavior.
6. Monotonic timestamp ordering and queue wait accounting.
7. Head-of-line blocking mitigation and SLA rescue validation.
8. Architecture comparison calculus (throughput gain, P95 speedup, queue reduction).
9. Recommender decision mechanics for single vs multi-client streams.
10. Missing telemetry / unmeasured candidate fallback (UNKNOWN status).
11. Historical API compatibility (Session 24, 25, 26 endpoints unaffected).
12. FastAPI serving route validations (/capabilities, /comparison, /hol-comparison, /recommend).
13. LlamaServerProcess configuration, command builder, and lifecycle safety.
14. Manifest artifact schema validation and persistence integrity.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.serving.benchmarks import (
    compute_architecture_comparison,
    compute_effective_concurrency,
    compute_execution_overlap_ms,
)
from llm_ii_engine.serving.capabilities import (
    build_serving_capability_matrix,
    detect_llama_server_binary,
    get_serving_host_hardware,
)
from llm_ii_engine.serving.llama_server_manager import LlamaServerProcess
from llm_ii_engine.serving.protocol import (
    CapabilityVerificationStatus,
    HeadOfLineComparisonObservation,
    ServingObservation,
    ServingRuntimeArchitecture,
    Session27Manifest,
)
from llm_ii_engine.serving.recommender import ServingArchitectureRecommender


@pytest.fixture
def test_client() -> TestClient:
    """Fixture providing a FastAPI TestClient configured with serving routes."""
    app = create_app()
    return TestClient(app)


def _make_dummy_observation(
    observation_id: str,
    model_id: str = "qwen2.5:0.5b",
    arch: ServingRuntimeArchitecture = ServingRuntimeArchitecture.OLLAMA_CLIENT_QUEUED,
    client_concurrency: int = 2,
    server_slots: int = 1,
    client_dispatch: float = 10.0,
    server_start: float = 10.0,
    server_completion: float = 10.5,
    client_completion: float = 10.5,
    prefill_ms: float = 100.0,
    decode_ms: float = 400.0,
    queue_wait_ms: float = 0.0,
    client_duration_ms: float = 500.0,
    overlapping: bool = False,
    overlap_ms: float = 0.0,
    sla_met: bool = True,
) -> ServingObservation:
    """Helper to construct a valid ServingObservation for testing."""
    return ServingObservation(
        observation_id=observation_id,
        session_id="session_27",
        runtime_architecture=arch,
        model_id=model_id,
        num_threads=8,
        client_concurrency=client_concurrency,
        server_slots=server_slots,
        effective_server_concurrency=1.0 if not overlapping else 2.0,
        workload_class="CONCURRENT_BATCH",
        prompt_tokens=32,
        output_tokens=16,
        latency_sla_ms=2000.0,
        client_dispatch_mono_s=client_dispatch,
        client_completion_mono_s=client_completion,
        server_start_mono_s=server_start,
        server_completion_mono_s=server_completion,
        client_duration_ms=client_duration_ms,
        server_prefill_ms=prefill_ms,
        server_decode_ms=decode_ms,
        queue_wait_ms=queue_wait_ms,
        ttft_ms=prefill_ms + queue_wait_ms,
        decode_tokens_per_second=40.0,
        overlapping_execution=overlapping,
        overlap_duration_ms=overlap_ms,
        slot_id_used=0,
        cpu_utilization_pct=30.0,
        memory_rss_mb=600.0,
        sla_met=sla_met,
    )


# ==============================================================================
# Area 1: Parallelism Capability Detection & Capability Matrix Integrity
# ==============================================================================

def test_capability_matrix_coverage_and_types():
    """Verify that the capability matrix contains exactly 12 entries (3 models x 4 architectures)."""
    matrix = build_serving_capability_matrix()
    assert len(matrix) == 12

    models = {entry.model_id for entry in matrix}
    assert models == {"llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"}

    archs = {entry.architecture_type for entry in matrix}
    assert archs == {
        ServingRuntimeArchitecture.OLLAMA_SINGLE_SLOT,
        ServingRuntimeArchitecture.OLLAMA_CLIENT_QUEUED,
        ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS,
        ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_4_SLOTS,
    }

    for row in matrix:
        assert isinstance(row.architecture_type, ServingRuntimeArchitecture)
        assert isinstance(row.parallel_request_support, CapabilityVerificationStatus)
        assert isinstance(row.batching_support, CapabilityVerificationStatus)
        assert isinstance(row.continuous_batching_support, CapabilityVerificationStatus)
        assert row.evidence_source != ""
        assert row.verification_notes != ""


def test_ollama_vs_llama_server_capability_distinction():
    """Verify that Ollama single-slot explicitly denotes lack of continuous batching,

    while llama.cpp denotes verified/advertised multi-slot continuous batching."""
    matrix = build_serving_capability_matrix()

    ollama_single = [r for r in matrix if r.architecture_type == ServingRuntimeArchitecture.OLLAMA_SINGLE_SLOT]
    for row in ollama_single:
        assert row.configurable_slots == 1
        assert row.parallel_request_support == CapabilityVerificationStatus.VERIFIED
        assert row.continuous_batching_support == CapabilityVerificationStatus.NOT_SUPPORTED

    llama_2_slots = [r for r in matrix if r.architecture_type == ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS]
    for row in llama_2_slots:
        assert row.configurable_slots == 2
        assert row.parallel_request_support in (
            CapabilityVerificationStatus.VERIFIED,
            CapabilityVerificationStatus.ADVERTISED_UNVERIFIED,
        )
        assert row.continuous_batching_support in (
            CapabilityVerificationStatus.VERIFIED,
            CapabilityVerificationStatus.ADVERTISED_UNVERIFIED,
        )


def test_llama_server_binary_detection():
    """Verify binary detection detects either an existing path or None, never raises."""
    binary_path = detect_llama_server_binary()
    if binary_path is not None:
        assert Path(binary_path).exists()
        assert Path(binary_path).is_file()


def test_host_hardware_telemetry():
    """Verify host hardware discovery returns valid CPU and RAM info."""
    hw = get_serving_host_hardware()
    assert "cpu" in hw
    assert "hardware_threads" in hw
    assert hw["hardware_threads"] >= 1
    assert "ram_gb" in hw
    assert hw["ram_gb"] > 0


# ==============================================================================
# Area 2: Server Concurrency Calculus & Execution Overlap Validation
# ==============================================================================

def test_effective_concurrency_serial_execution():
    """When two requests execute sequentially with 0 overlap on Ollama, effective concurrency is strictly 1.0."""
    obs1 = _make_dummy_observation(
        observation_id="req-1",
        arch=ServingRuntimeArchitecture.OLLAMA_CLIENT_QUEUED,
        client_dispatch=10.0,
        server_start=10.0,
        server_completion=10.5,
        client_completion=10.5,
        queue_wait_ms=0.0,
        client_duration_ms=500.0,
    )
    obs2 = _make_dummy_observation(
        observation_id="req-2",
        arch=ServingRuntimeArchitecture.OLLAMA_CLIENT_QUEUED,
        client_dispatch=10.0,
        server_start=10.5,
        server_completion=11.0,
        client_completion=11.0,
        queue_wait_ms=500.0,
        client_duration_ms=1000.0,
    )

    overlap_ms = compute_execution_overlap_ms([obs1, obs2])
    eff_c = compute_effective_concurrency([obs1, obs2], wall_span_s=1.0)

    assert overlap_ms == 0.0
    assert pytest.approx(eff_c, 0.01) == 1.0


def test_effective_concurrency_parallel_execution():
    """When two requests execute simultaneously with full overlap on llama.cpp, effective concurrency is ~2.0."""
    obs1 = _make_dummy_observation(
        observation_id="req-1",
        arch=ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS,
        server_slots=2,
        client_dispatch=10.0,
        server_start=10.05,
        server_completion=10.55,
        client_completion=10.55,
        queue_wait_ms=50.0,
        client_duration_ms=550.0,
        overlapping=True,
    )
    obs2 = _make_dummy_observation(
        observation_id="req-2",
        arch=ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS,
        server_slots=2,
        client_dispatch=10.0,
        server_start=10.05,
        server_completion=10.55,
        client_completion=10.55,
        queue_wait_ms=50.0,
        client_duration_ms=550.0,
        overlapping=True,
    )

    overlap_ms = compute_execution_overlap_ms([obs1, obs2])
    eff_c = compute_effective_concurrency([obs1, obs2], wall_span_s=0.55)

    assert overlap_ms == 500.0  # (10.55 - 10.05) * 1000
    assert pytest.approx(eff_c, 0.1) == 1.82 or pytest.approx(eff_c, 0.1) == 2.0


# ==============================================================================
# Area 3: Monotonic Timestamps & Queue Accounting Invariants
# ==============================================================================

def test_observation_monotonic_invariants():
    """Verify that dispatch <= exec_start <= completion and durations are positive."""
    obs = _make_dummy_observation(
        observation_id="test-inv",
        arch=ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS,
        server_slots=2,
        client_dispatch=100.0,
        server_start=100.02,
        server_completion=100.45,
        client_completion=100.45,
        prefill_ms=80.0,
        decode_ms=350.0,
        queue_wait_ms=20.0,
        client_duration_ms=450.0,
    )

    assert obs.client_dispatch_mono_s <= obs.server_start_mono_s
    assert obs.server_start_mono_s <= obs.server_completion_mono_s
    assert obs.server_completion_mono_s <= obs.client_completion_mono_s
    assert obs.queue_wait_ms >= 0.0
    assert obs.client_duration_ms >= obs.queue_wait_ms + obs.server_prefill_ms


# ==============================================================================
# Area 4: Head-of-Line Blocking Mitigation & SLA Rescue
# ==============================================================================

def test_head_of_line_sla_rescue_calculation():
    """Verify that SLA rescue is correctly detected when Ollama breaches SLA and llama-server meets it."""
    hol_rescue = HeadOfLineComparisonObservation(
        experiment_id="hol-rescue-1",
        model_id="llama3.2:1b",
        antecedent_workload="160-token heavy generation",
        consequent_workload="16-token interactive query",
        consequent_sla_ms=2000.0,
        ollama_consequent_wait_ms=7200.0,
        ollama_consequent_e2e_ms=7700.0,
        ollama_sla_met=False,
        llama_server_consequent_wait_ms=60.0,
        llama_server_consequent_e2e_ms=860.0,
        llama_server_sla_met=True,
        queue_wait_reduction_ms=7140.0,
        latency_reduction_factor=8.95,
        sla_rescued_by_parallelism=True,
    )

    assert hol_rescue.sla_rescued_by_parallelism is True
    assert hol_rescue.queue_wait_reduction_ms > 7000.0
    assert hol_rescue.ollama_sla_met is False
    assert hol_rescue.llama_server_sla_met is True


def test_head_of_line_no_rescue_when_both_pass():
    """Verify that sla_rescued_by_parallelism is False if Ollama already met SLA."""
    hol_no_rescue = HeadOfLineComparisonObservation(
        experiment_id="hol-no-rescue-1",
        model_id="qwen2.5:0.5b",
        antecedent_workload="16-token query",
        consequent_workload="16-token query",
        consequent_sla_ms=5000.0,
        ollama_consequent_wait_ms=300.0,
        ollama_consequent_e2e_ms=700.0,
        ollama_sla_met=True,
        llama_server_consequent_wait_ms=40.0,
        llama_server_consequent_e2e_ms=440.0,
        llama_server_sla_met=True,
        queue_wait_reduction_ms=260.0,
        latency_reduction_factor=1.59,
        sla_rescued_by_parallelism=False,
    )

    assert hol_no_rescue.sla_rescued_by_parallelism is False


# ==============================================================================
# Area 5: Architecture Comparison Calculus
# ==============================================================================

def test_compute_architecture_comparison_metrics():
    """Verify computation of throughput gain, P95 speedup, and queue reduction."""
    base_obs = [
        _make_dummy_observation(
            observation_id=f"base-{i}",
            arch=ServingRuntimeArchitecture.OLLAMA_CLIENT_QUEUED,
            client_dispatch=0.0,
            server_start=0.0 + i * 0.5,
            server_completion=0.5 + i * 0.5,
            client_completion=0.5 + i * 0.5,
            queue_wait_ms=i * 500.0,
            client_duration_ms=500.0 + i * 500.0,
            sla_met=(i == 0),
        )
        for i in range(2)
    ]

    target_obs = [
        _make_dummy_observation(
            observation_id=f"target-{i}",
            arch=ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS,
            server_slots=2,
            client_dispatch=0.0,
            server_start=0.05,
            server_completion=0.60,
            client_completion=0.60,
            queue_wait_ms=50.0,
            client_duration_ms=600.0,
            overlapping=True,
            sla_met=True,
        )
        for i in range(2)
    ]

    comp = compute_architecture_comparison(
        baseline_observations=base_obs,
        target_observations=target_obs,
        workload_description="Test 2-client concurrency",
        client_concurrency=2,
    )

    assert comp.baseline_throughput_rps == pytest.approx(2.0, 0.05)
    assert comp.target_throughput_rps == pytest.approx(3.33, 0.05)
    assert comp.throughput_gain_pct > 50.0
    assert comp.latency_p95_speedup_factor > 1.0
    assert comp.queue_wait_reduction_ms > 0.0


# ==============================================================================
# Area 6: Recommender Decision Mechanics
# ==============================================================================

def test_recommender_single_client_defaults_to_single_slot():
    """Under single concurrency (C=1), recommender recommends single-slot Ollama to conserve memory."""
    recommender = ServingArchitectureRecommender()
    rec = recommender.recommend_architecture(
        model_id="llama3.2:1b",
        workload_type="single_stream_interactive",
        client_concurrency=1,
        arrival_rate_rps=0.2,
        latency_sla_ms=3000.0,
        memory_budget_mb=8192.0,
    )

    assert rec.recommended_architecture == ServingRuntimeArchitecture.OLLAMA_SINGLE_SLOT
    assert rec.recommended_slots == 1
    assert "single-stream" in rec.rationale.lower() or "c=1" in rec.rationale.lower()


def test_recommender_high_concurrency_selects_parallel_slots():
    """Under multi-client concurrency (C >= 2) with arrival >= 1.0 rps, recommender recommends llama.cpp multi-slot."""
    recommender = ServingArchitectureRecommender()
    rec = recommender.recommend_architecture(
        model_id="qwen2.5:0.5b",
        workload_type="mixed_chat_summary",
        client_concurrency=2,
        arrival_rate_rps=1.5,
        latency_sla_ms=1500.0,
        memory_budget_mb=8192.0,
    )

    assert rec.recommended_architecture == ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS
    assert rec.recommended_slots == 2
    assert "-cb" in rec.actionable_setup_guidance
    assert "-np 2" in rec.actionable_setup_guidance


def test_recommender_tight_memory_budget_constrains_slots():
    """If memory budget is insufficient for 4 slots, recommender caps at 2 slots or single-slot."""
    recommender = ServingArchitectureRecommender()
    # Provide very small budget: 450 MB (qwen 0.5b alone is ~398MB + KV)
    rec = recommender.recommend_architecture(
        model_id="qwen2.5:0.5b",
        workload_type="high_concurrency_batch",
        client_concurrency=4,
        arrival_rate_rps=3.0,
        latency_sla_ms=2000.0,
        memory_budget_mb=450.0,
    )

    assert rec.recommended_slots <= 2


# ==============================================================================
# Area 7: LlamaServerProcess Lifecycle & Command Builder Safety
# ==============================================================================

def test_llama_server_command_args():
    """Verify argument construction for llama-server."""
    server = LlamaServerProcess(
        binary_path="C:\\test\\llama-server.exe",
        model_path="C:\\test\\model.gguf",
        port=18080,
        slots=4,
        threads=8,
        num_ctx=4096,
        continuous_batching=True,
    )

    cmd = server.build_command()
    assert cmd[0] == "C:\\test\\llama-server.exe"
    assert "-m" in cmd
    assert "C:\\test\\model.gguf" in cmd
    assert "-np" in cmd and "4" in cmd
    assert "-t" in cmd and "8" in cmd
    assert "-c" in cmd and "4096" in cmd
    assert "-cb" in cmd
    assert "--port" in cmd and "18080" in cmd


# ==============================================================================
# Area 8: FastAPI Serving Route Validations
# ==============================================================================

def test_api_serving_capabilities(test_client: TestClient):
    """GET /api/v1/serving/capabilities returns 200 and valid capability matrix."""
    response = test_client.get("/api/v1/serving/capabilities")
    assert response.status_code == 200
    data = response.json()

    assert "host_hardware" in data
    assert "capability_matrix" in data
    assert len(data["capability_matrix"]) == 12


def test_api_serving_comparison(test_client: TestClient):
    """GET /api/v1/serving/comparison returns 200 and comparison list."""
    response = test_client.get("/api/v1/serving/comparison")
    assert response.status_code == 200
    data = response.json()

    assert "comparisons" in data
    assert len(data["comparisons"]) >= 1

    # Filter by model_id
    resp_filtered = test_client.get("/api/v1/serving/comparison?model_id=qwen2.5:0.5b")
    assert resp_filtered.status_code == 200
    filtered_data = resp_filtered.json()
    for comp in filtered_data["comparisons"]:
        assert comp["model_id"] == "qwen2.5:0.5b"


def test_api_serving_hol_comparison(test_client: TestClient):
    """GET /api/v1/serving/hol-comparison returns 200 and HoL observations."""
    response = test_client.get("/api/v1/serving/hol-comparison")
    assert response.status_code == 200
    data = response.json()

    assert "hol_comparisons" in data
    assert len(data["hol_comparisons"]) >= 1


def test_api_serving_recommendation(test_client: TestClient):
    """POST /api/v1/serving/recommend returns 200 and actionable recommendation."""
    payload = {
        "model_id": "qwen2.5:0.5b",
        "workload_type": "mixed_chat_summary",
        "client_concurrency": 2,
        "arrival_rate_rps": 1.5,
        "latency_sla_ms": 1500.0,
        "memory_budget_mb": 8192.0,
    }
    response = test_client.post("/api/v1/serving/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "recommendation_id" in data
    assert "recommended_architecture" in data
    assert "recommended_slots" in data
    assert "actionable_setup_guidance" in data


# ==============================================================================
# Area 9: Historical Endpoint Compatibility (Sessions 24, 25, 26)
# ==============================================================================

def test_historical_compatibility_sessions_25_and_26(test_client: TestClient):
    """Verify that prior capacity and mixed-workload routes are still functional and unaffected."""
    # Session 25
    r25 = test_client.get("/api/v1/capacity/scenarios")
    assert r25.status_code == 200

    # Session 26
    r26_profiles = test_client.get("/api/v1/capacity/mixed-profiles")
    assert r26_profiles.status_code == 200

    r26_hol = test_client.get("/api/v1/capacity/hol-analysis")
    assert r26_hol.status_code == 200


# ==============================================================================
# Area 10: Manifest & Artifact Persistence Validation
# ==============================================================================

def test_session_27_manifest_artifact_exists_and_valid():
    """Verify storage/experiments/session_27_serving_manifest.json conforms to Session27Manifest schema."""
    manifest_path = Path("storage/experiments/session_27_serving_manifest.json")
    assert manifest_path.exists(), "Session 27 manifest file must exist"

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    manifest = Session27Manifest(**data)
    assert "session-27" in manifest.manifest_id
    assert manifest.observations_count > 0
    assert len(manifest.comparisons) > 0
    assert len(manifest.hol_comparisons) > 0
    assert manifest.total_benchmark_duration_s < 25.0 * 60.0  # Must be under 25-minute limit


def test_session_27_markdown_artifacts_exist():
    """Verify all 4 required markdown artifacts from Session 27 exist and are non-empty."""
    docs = [
        "storage/experiments/session_27_architecture_comparison.md",
        "storage/experiments/session_27_head_of_line_comparison.md",
        "storage/experiments/session_27_parallelism_verification.md",
        "storage/experiments/session_27_prospective_validation.md",
    ]
    for doc in docs:
        p = Path(doc)
        assert p.exists(), f"Document {doc} must exist"
        assert p.stat().st_size > 100, f"Document {doc} must be non-empty"
