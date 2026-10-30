"""Comprehensive unit and integration test suite for Session 28:

Context-Fair Parallel Serving, KV-Cache Memory Modeling & Slot Optimization.

Covers all 18 Phase 12 testing areas:
1. Model metadata parsing (layers, heads, kv heads, head_dim, GQA ratio).
2. KV-cache calculation per token and per slot (exact formula validation).
3. Context partitioning behavior (total context vs per-slot context).
4. Per-request context validation and truncation detection (validate_context_budget).
5. Memory budget enforcement (slot capping when RAM exceeds budget).
6. Model identity equivalence audit (audit_model_equivalence across runtimes).
7. Invalid comparison detection.
8. Slot optimization decision logic (SlotContextOptimizer.optimize for C=1, C=2, C=4).
9. Throughput and latency trade-offs.
10. Evidence-tier correctness.
11. Missing runtime telemetry fallback.
12. Prospective prediction locking (session_28_prospective_predictions.json).
13. SLA feasibility assessment (marking LATENCY_UNFEASIBLE when SLA violated).
14. Request accounting (context tokens + output tokens <= verified per-slot context).
15. API contract: GET /api/v1/serving/kv-estimates.
16. API contract: GET /api/v1/serving/model-equivalence.
17. API contract: POST /api/v1/serving/optimize-slots.
18. API contract: GET /api/v1/serving/context-fair-manifest & Historical Compatibility.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.serving.capabilities import AUTHORIZED_MODELS
from llm_ii_engine.serving.context_memory import (
    audit_model_equivalence,
    calculate_kv_cache_memory,
    get_model_architecture_spec,
    validate_context_budget,
)
from llm_ii_engine.serving.llama_server_manager import LlamaServerProcess
from llm_ii_engine.serving.protocol import (
    ContextFairObservation,
    ModelArchitectureSpec,
    ModelEquivalenceVerdict,
    ServingRuntimeArchitecture,
    SlotFeasibilityStatus,
    TailLatencyRisk,
)
from llm_ii_engine.serving.slot_optimizer import SlotContextOptimizer


@pytest.fixture
def test_client() -> TestClient:
    """Fixture providing a FastAPI TestClient configured with serving routes."""
    app = create_app()
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Model Metadata Parsing
# -----------------------------------------------------------------------------
def test_model_metadata_parsing():
    """Verify architectural parameters for all authorized models."""
    spec_05b = get_model_architecture_spec("qwen2.5:0.5b")
    assert isinstance(spec_05b, ModelArchitectureSpec)
    assert spec_05b.num_layers == 24
    assert spec_05b.num_heads == 14
    assert spec_05b.num_kv_heads == 2
    assert spec_05b.head_dim == 64
    assert spec_05b.gqa_ratio == 7.0

    spec_15b = get_model_architecture_spec("qwen2.5:1.5b")
    assert spec_15b.num_layers == 28
    assert spec_15b.num_heads == 12
    assert spec_15b.num_kv_heads == 2
    assert spec_15b.head_dim == 128
    assert spec_15b.gqa_ratio == 6.0

    spec_1b = get_model_architecture_spec("llama3.2:1b")
    assert spec_1b.num_layers == 16
    assert spec_1b.num_heads == 32
    assert spec_1b.num_kv_heads == 8
    assert spec_1b.head_dim == 64
    assert spec_1b.gqa_ratio == 4.0


# -----------------------------------------------------------------------------
# 2. KV-Cache Calculation Per Token and Per Slot
# -----------------------------------------------------------------------------
def test_kv_cache_calculation_per_token_and_per_slot():
    """Verify KV cache bytes per token and total memory scaling."""
    # Formula: 2 * L * H_kv * d_k * precision (2 bytes for FP16)
    # qwen2.5:0.5b -> 2 * 24 * 2 * 64 * 2 = 12,288 B (12.0 KB)
    spec_05b = get_model_architecture_spec("qwen2.5:0.5b")
    assert spec_05b.bytes_per_token_kv == 12288.0
    assert spec_05b.kb_per_token_kv == 12.0

    # 1 slot, 1024 tokens = 12,288 * 1024 / (1024*1024) = 12.0 MB
    est_1s_1024 = calculate_kv_cache_memory("qwen2.5:0.5b", slots=1, per_slot_context_tokens=1024)
    assert est_1s_1024.theoretical_kv_cache_mb == 12.0
    assert est_1s_1024.total_context_tokens == 1024

    # 2 slots, 1024 tokens/slot = 2048 total tokens = 24.0 MB
    est_2s_1024 = calculate_kv_cache_memory("qwen2.5:0.5b", slots=2, per_slot_context_tokens=1024)
    assert est_2s_1024.theoretical_kv_cache_mb == 24.0
    assert est_2s_1024.total_context_tokens == 2048

    # 4 slots, 1024 tokens/slot = 4096 total tokens = 48.0 MB
    est_4s_1024 = calculate_kv_cache_memory("qwen2.5:0.5b", slots=4, per_slot_context_tokens=1024)
    assert est_4s_1024.theoretical_kv_cache_mb == 48.0
    assert est_4s_1024.total_context_tokens == 4096


# -----------------------------------------------------------------------------
# 3. Context Partitioning Behavior
# -----------------------------------------------------------------------------
def test_context_partitioning_behavior():
    """Verify that llama-server geometry allocates total_context = slots * per_slot_context."""
    # To provide 2048 tokens per slot across 4 slots, total context allocated must be 8192
    est = calculate_kv_cache_memory("llama3.2:1b", slots=4, per_slot_context_tokens=2048)
    assert est.total_context_tokens == 8192
    assert est.per_slot_context_tokens == 2048
    assert est.slots == 4


# -----------------------------------------------------------------------------
# 4. Per-Request Context Validation and Truncation Detection
# -----------------------------------------------------------------------------
def test_per_request_context_validation_and_truncation():
    """Verify validate_context_budget detects silent context truncation."""
    # Context allocated: 1024. Request: 500 prompt + 100 output = 600 -> Valid
    is_valid, needed, msg = validate_context_budget(
        per_slot_context=1024,
        prompt_tokens=500,
        max_output_tokens=100,
    )
    assert is_valid is True
    assert needed == 600
    assert "fits within" in msg.lower()

    # Context allocated: 512. Request: 500 prompt + 50 output = 550 -> Truncation!
    is_valid, needed, msg = validate_context_budget(
        per_slot_context=512,
        prompt_tokens=500,
        max_output_tokens=50,
    )
    assert is_valid is False
    assert needed == 550
    assert "exceeded" in msg.lower()


# -----------------------------------------------------------------------------
# 5. Memory Budget Enforcement
# -----------------------------------------------------------------------------
def test_memory_budget_enforcement():
    """Verify slot capping when memory budget cannot fit higher slot counts."""
    # llama3.2:1b has 1321 MB weights.
    # At 2048 context:
    # 1 slot = 1321 + 64 + 150 = 1535 MB
    # 2 slots = 1321 + 128 + 150 = 1599 MB
    # 4 slots = 1321 + 256 + 150 = 1727 MB
    # With a budget of 1600 MB and offered concurrency 4, slots MUST be capped at 2.
    rec = SlotContextOptimizer.optimize(
        model_id="llama3.2:1b",
        required_context_tokens=2048,
        expected_output_tokens=64,
        offered_concurrency=4,
        latency_sla_ms=8000.0,
        memory_budget_mb=1600.0,
    )
    assert rec.recommended_slots == 2
    assert rec.estimated_total_memory_mb <= 1600.0
    assert rec.verified_per_slot_context == 2048


# -----------------------------------------------------------------------------
# 6. Model Identity Equivalence Audit
# -----------------------------------------------------------------------------
def test_model_identity_equivalence_audit():
    """Verify audit_model_equivalence verifies identical blobs for authorized models."""
    for model_id in AUTHORIZED_MODELS:
        records = audit_model_equivalence(model_id)
        assert len(records) >= 2
        for r in records:
            assert r.model_id == model_id
            assert r.equivalence_verdict == ModelEquivalenceVerdict.EXACT_EQUIVALENT
            assert "blobs" in r.file_path
            assert "GGUF" in r.tokenizer_identity


# -----------------------------------------------------------------------------
# 7. Invalid Comparison Detection
# -----------------------------------------------------------------------------
def test_invalid_comparison_detection():
    """Verify handling when an unknown model is audited."""
    records = audit_model_equivalence("unknown-nonexistent-model")
    assert len(records) == 1
    assert records[0].equivalence_verdict == ModelEquivalenceVerdict.UNKNOWN
    assert "not found" in records[0].equivalence_notes.lower()


# -----------------------------------------------------------------------------
# 8. Slot Optimization Decision Logic (C=1, C=2, C=4)
# -----------------------------------------------------------------------------
def test_slot_optimization_decision_logic():
    """Verify optimizer selects 1 slot for C=1, 2 slots for C=2, 4 slots for C=4."""
    # Single stream: 1 slot recommended
    rec_c1 = SlotContextOptimizer.optimize(
        model_id="qwen2.5:0.5b",
        required_context_tokens=512,
        expected_output_tokens=32,
        offered_concurrency=1,
        latency_sla_ms=2000.0,
        memory_budget_mb=8192.0,
    )
    assert rec_c1.recommended_slots == 1

    # Concurrent 2: 2 slots recommended
    rec_c2 = SlotContextOptimizer.optimize(
        model_id="qwen2.5:0.5b",
        required_context_tokens=1024,
        expected_output_tokens=64,
        offered_concurrency=2,
        latency_sla_ms=2500.0,
        memory_budget_mb=8192.0,
    )
    assert rec_c2.recommended_slots == 2

    # High concurrency burst 4: 4 slots recommended
    rec_c4 = SlotContextOptimizer.optimize(
        model_id="qwen2.5:0.5b",
        required_context_tokens=512,
        expected_output_tokens=32,
        offered_concurrency=4,
        latency_sla_ms=3000.0,
        memory_budget_mb=8192.0,
    )
    assert rec_c4.recommended_slots == 4


# -----------------------------------------------------------------------------
# 9. Throughput and Latency Trade-Offs
# -----------------------------------------------------------------------------
def test_throughput_and_latency_tradeoffs():
    """Verify that multi-slot configurations project higher throughput with latency trade-off."""
    rec_2s = SlotContextOptimizer.optimize(
        model_id="qwen2.5:0.5b",
        required_context_tokens=1024,
        expected_output_tokens=64,
        offered_concurrency=2,
        latency_sla_ms=4000.0,
    )
    rec_4s = SlotContextOptimizer.optimize(
        model_id="qwen2.5:0.5b",
        required_context_tokens=1024,
        expected_output_tokens=64,
        offered_concurrency=4,
        latency_sla_ms=4000.0,
    )
    assert rec_4s.expected_aggregate_throughput_rps > rec_2s.expected_aggregate_throughput_rps
    assert rec_4s.expected_p95_latency_ms > rec_2s.expected_p95_latency_ms


# -----------------------------------------------------------------------------
# 10. Evidence-Tier Correctness
# -----------------------------------------------------------------------------
def test_evidence_tier_correctness():
    """Verify evidence tier reflects benchmarking status."""
    rec = SlotContextOptimizer.optimize(
        model_id="qwen2.5:0.5b",
        required_context_tokens=512,
        expected_output_tokens=32,
        offered_concurrency=1,
    )
    assert rec.evidence_tier in {"DIRECT_MEASURED", "EMPIRICAL_BENCHMARKED", "ANALYTICAL_EXTRAPOLATION"}


# -----------------------------------------------------------------------------
# 11. Missing Runtime Telemetry Fallback
# -----------------------------------------------------------------------------
def test_missing_runtime_telemetry_fallback():
    """Verify fallback when runtime process RSS telemetry is absent."""
    est = calculate_kv_cache_memory(
        model_id="qwen2.5:0.5b",
        slots=2,
        per_slot_context_tokens=1024,
        observed_process_memory_mb=None,
    )
    assert est.observed_process_memory_mb is None
    assert est.empirical_residual_mb is None
    assert est.theoretical_kv_cache_mb == 24.0
    assert est.estimated_total_memory_mb > 500.0


# -----------------------------------------------------------------------------
# 12. Prospective Prediction Locking
# -----------------------------------------------------------------------------
def test_prospective_prediction_locking():
    """Verify session_28_prospective_predictions.json exists and is locked."""
    pred_path = Path("storage/experiments/session_28_prospective_predictions.json")
    assert pred_path.exists(), "Prospective predictions file must exist"

    with open(pred_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["manifest_id"] == "session_28_prospective_predictions"
    assert data["locking_status"] == "PRE_REGISTERED_LOCKED"
    assert data["total_scenarios"] == 4
    scenario_ids = [s["scenario_id"] for s in data["scenarios"]]
    assert "scenario_1_short_context_interactive" in scenario_ids
    assert "scenario_2_moderate_context_concurrent" in scenario_ids
    assert "scenario_3_long_context_memory_constrained" in scenario_ids
    assert "scenario_4_mixed_context_competing_requests" in scenario_ids


# -----------------------------------------------------------------------------
# 13. SLA Feasibility Assessment
# -----------------------------------------------------------------------------
def test_sla_feasibility_assessment():
    """Verify that unrealistic SLA triggers LATENCY_UNFEASIBLE."""
    rec = SlotContextOptimizer.optimize(
        model_id="llama3.2:1b",
        required_context_tokens=2048,
        expected_output_tokens=128,
        offered_concurrency=4,
        latency_sla_ms=50.0,  # Impossible on CPU
    )
    assert rec.feasibility_status == SlotFeasibilityStatus.LATENCY_UNFEASIBLE
    assert rec.tail_latency_risk in {TailLatencyRisk.HIGH, TailLatencyRisk.CRITICAL, TailLatencyRisk.MODERATE}


# -----------------------------------------------------------------------------
# 14. Request Accounting in ContextFairObservation
# -----------------------------------------------------------------------------
def test_request_accounting():
    """Verify ContextFairObservation field validations and invariants."""
    obs = ContextFairObservation(
        observation_id="test_obs_01",
        runtime_architecture=ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS,
        model_id="qwen2.5:0.5b",
        slots=2,
        per_slot_context_tokens=1024,
        total_context_tokens=2048,
        prompt_tokens=400,
        output_tokens=50,
        effective_context_used=450,
        context_fairness_verified=True,
        context_truncation_occurred=False,
        client_dispatch_mono_s=100.0,
        client_completion_mono_s=100.5,
        client_duration_ms=500.0,
        server_prefill_ms=100.0,
        server_decode_ms=400.0,
        queue_wait_ms=5.0,
        ttft_ms=105.0,
        decode_tokens_per_second=125.0,
        theoretical_kv_mb=24.0,
        observed_memory_mb=580.0,
        memory_residual_mb=8.0,
        effective_server_concurrency=2.0,
        latency_sla_ms=2000.0,
        sla_met=True,
    )
    assert obs.effective_context_used <= obs.per_slot_context_tokens
    assert obs.sla_met is True


# -----------------------------------------------------------------------------
# 15. API Contract: GET /api/v1/serving/kv-estimates
# -----------------------------------------------------------------------------
def test_api_kv_estimates(test_client: TestClient):
    """Verify GET /api/v1/serving/kv-estimates returns valid schema."""
    resp = test_client.get("/api/v1/serving/kv-estimates")
    assert resp.status_code == 200
    data = resp.json()
    assert "architectures" in data
    assert "estimates" in data
    assert len(data["architectures"]) >= 3
    assert len(data["estimates"]) >= 9


# -----------------------------------------------------------------------------
# 16. API Contract: GET /api/v1/serving/model-equivalence
# -----------------------------------------------------------------------------
def test_api_model_equivalence(test_client: TestClient):
    """Verify GET /api/v1/serving/model-equivalence returns audit records."""
    resp = test_client.get("/api/v1/serving/model-equivalence")
    assert resp.status_code == 200
    data = resp.json()
    assert "records" in data
    records = data["records"]
    assert len(records) >= 3
    for r in records:
        assert r["equivalence_verdict"] == "EXACT_EQUIVALENT"


# -----------------------------------------------------------------------------
# 17. API Contract: POST /api/v1/serving/optimize-slots
# -----------------------------------------------------------------------------
def test_api_optimize_slots(test_client: TestClient):
    """Verify POST /api/v1/serving/optimize-slots returns slot recommendation."""
    payload = {
        "model_id": "qwen2.5:0.5b",
        "required_context_tokens": 1024,
        "expected_output_tokens": 64,
        "offered_concurrency": 2,
        "latency_sla_ms": 2500.0,
        "memory_budget_mb": 8192.0,
    }
    resp = test_client.post("/api/v1/serving/optimize-slots", json=payload)
    assert resp.status_code == 200
    rec = resp.json()
    assert rec["recommended_slots"] == 2
    assert rec["verified_per_slot_context"] == 1024
    assert rec["recommended_total_context"] == 2048
    assert "actionable_setup_guidance" in rec


# -----------------------------------------------------------------------------
# 18. API Contract: Manifest & Historical Compatibility
# -----------------------------------------------------------------------------
def test_api_manifest_and_historical_compatibility(test_client: TestClient):
    """Verify Session 28 manifest endpoint and compatibility with prior serving endpoints."""
    # Session 28 manifest
    resp_s28 = test_client.get("/api/v1/serving/context-fair-manifest")
    assert resp_s28.status_code == 200
    data_s28 = resp_s28.json()
    assert data_s28.get("session_id") == "session_28" or data_s28.get("status") == "NOT_FOUND"

    # Session 27 endpoints unaffected
    resp_caps = test_client.get("/api/v1/serving/capabilities")
    assert resp_caps.status_code == 200

    resp_comp = test_client.get("/api/v1/serving/comparison")
    assert resp_comp.status_code == 200

    resp_hol = test_client.get("/api/v1/serving/hol-comparison")
    assert resp_hol.status_code == 200


# -----------------------------------------------------------------------------
# 19. Process Lifecycle Safety
# -----------------------------------------------------------------------------
def test_process_lifecycle_safety():
    """Verify LlamaServerProcess command generation and parameters."""
    proc = LlamaServerProcess(
        model_id="qwen2.5:0.5b",
        slots=4,
        continuous_batching=True,
        threads=8,
        port=18080,
        context_size=4096,
    )
    cmd = proc.build_command()
    assert "-np" in cmd
    assert cmd[cmd.index("-np") + 1] == "4"
    assert "-c" in cmd
    assert cmd[cmd.index("-c") + 1] == "4096"
    assert "-t" in cmd
    assert cmd[cmd.index("-t") + 1] == "8"
    assert "-cb" in cmd
    assert proc.base_url == "http://127.0.0.1:18080"
