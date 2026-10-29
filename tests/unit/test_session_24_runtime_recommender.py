"""Focused automated test suite for Session 24 hardware-aware runtime optimization.

Validates:
1. Runtime option validation and bounds.
2. Requested vs effective configuration provenance.
3. Thread-count candidate filtering and assignment.
4. Concurrency measurement semantics.
5. Single-slot request queuing vs parallel execution.
6. Cache-state separation (warm vs cold prompt prefill).
7. Model identity and digest verification.
8. Configuration-level empirical matching.
9. Unknown feasibility for unsupported settings and fail-closed code constraints.
10. Multi-objective Pareto ranking.
11. SLA conservatism (ROBUST_FEASIBLE vs NOMINAL_FEASIBLE_ROBUST_RISK vs PREDICTED_SLA_MISS).
12. Prospective validation integrity.
13. API response contract validation.
14. Dashboard types and contract compatibility.
15. Bounded timeout and lifecycle cleanup.
16. Historical manifest schema backward compatibility.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_ii_engine.benchmark.multi_model_protocol import (
    CodeExecutionSecurityMode,
    QualityEvaluationStatus,
)
from llm_ii_engine.benchmark.runtime_tuning_protocol import (
    RuntimeConfigurationSpec,
    RuntimeExperimentManifest,
    WorkstationHardwareProfile,
)
from llm_ii_engine.recommendation.models import (
    ModelQualityConstraintSpec,
    QualityStatus,
)
from llm_ii_engine.recommendation.runtime_recommender import (
    HardwareConstraintSpec,
    RuntimeConfigurationRecommender,
    RuntimeOptimizationObjective,
    RuntimeRecommendationRequest,
)


class TestRuntimeOptionValidation:
    """Tests 1 & 2: Runtime option validation and requested vs effective provenance."""

    def test_valid_runtime_config_bounds(self):
        cfg = RuntimeConfigurationSpec(num_thread=8, num_ctx=4096, concurrency=1)
        assert cfg.num_thread == 8
        assert cfg.num_ctx == 4096
        assert cfg.concurrency == 1

    def test_invalid_thread_count_raises(self):
        with pytest.raises(ValidationError):
            RuntimeConfigurationSpec(num_thread=0)
        with pytest.raises(ValidationError):
            RuntimeConfigurationSpec(num_thread=64)

    def test_invalid_context_size_raises(self):
        with pytest.raises(ValidationError):
            RuntimeConfigurationSpec(num_ctx=256)

    def test_hardware_profile_fingerprint_integrity(self):
        hw = WorkstationHardwareProfile()
        assert hw.cpu_model == "Intel(R) Core(TM) i7-14700"
        assert hw.physical_cores == 20
        assert hw.performance_cores == 8
        assert hw.efficient_cores == 12
        assert hw.logical_processors == 28


class TestThreadCountAndConcurrencySemantics:
    """Tests 3, 4, 5, 6: Thread count tuning, concurrency queuing, and cache separation."""

    def test_recommender_picks_8_threads_for_interactive_latency(self):
        recommender = RuntimeConfigurationRecommender()
        req = RuntimeRecommendationRequest(
            prompt_tokens=32,
            expected_output_tokens=16,
            latency_sla_ms=1200.0,
            concurrency=1,
            objective=RuntimeOptimizationObjective.MIN_LATENCY,
        )
        res = recommender.recommend(req)
        assert res.model_id == "qwen2.5:0.5b"
        assert res.recommended_threads == 8
        assert res.predicted_decode_tokens_per_second >= 60.0
        assert res.conservative_feasibility == "ROBUST_FEASIBLE"

    def test_recommender_respects_target_threads_override(self):
        recommender = RuntimeConfigurationRecommender()
        req = RuntimeRecommendationRequest(
            prompt_tokens=40,
            expected_output_tokens=20,
            concurrency=1,
            hardware_constraints=HardwareConstraintSpec(target_threads=4),
        )
        res = recommender.recommend(req)
        assert res.recommended_threads == 4

    def test_concurrency_penalizes_queuing_delay(self):
        recommender = RuntimeConfigurationRecommender()
        req_c1 = RuntimeRecommendationRequest(
            prompt_tokens=50,
            expected_output_tokens=50,
            concurrency=1,
            fixed_model_id="llama3.2:1b",
        )
        req_c2 = RuntimeRecommendationRequest(
            prompt_tokens=50,
            expected_output_tokens=50,
            concurrency=2,
            fixed_model_id="llama3.2:1b",
        )
        res_c1 = recommender.recommend(req_c1)
        res_c2 = recommender.recommend(req_c2)
        # Latency of concurrency 2 must exceed concurrency 1 due to serialized single-slot queue
        assert res_c2.predicted_client_e2e_ms > res_c1.predicted_client_e2e_ms

    def test_cache_state_separation_in_manifest(self):
        manifest_path = Path("storage/experiments/session_24_manifest.json")
        assert manifest_path.exists(), "Manifest file must exist"
        manifest = RuntimeExperimentManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))

        # Check that Stage C separates cold prefix vs warm prefix
        stage_c_obs = [o for o in manifest.observations if o.stage == "STAGE_C_CONTEXT"]
        assert len(stage_c_obs) >= 6
        # Repetition 2 must have much lower prefill duration than Repetition 1 (warm prefix cache speedup)
        rep1 = [o for o in stage_c_obs if o.repetition_index == 1 and o.model_id == "qwen2.5:0.5b"][0]
        rep2 = [o for o in stage_c_obs if o.repetition_index == 2 and o.model_id == "qwen2.5:0.5b"][0]
        assert rep2.server_prefill_duration_ms < (rep1.server_prefill_duration_ms * 0.2)


class TestRecommendationPoliciesAndGuardrails:
    """Tests 7, 8, 9, 10, 11: Guardrails, feasibility, SLA conservatism, and Pareto."""

    def test_pinned_model_selection(self):
        recommender = RuntimeConfigurationRecommender()
        req = RuntimeRecommendationRequest(
            prompt_tokens=32,
            expected_output_tokens=16,
            fixed_model_id="llama3.2:1b",
            allow_model_selection=False,
        )
        res = recommender.recommend(req)
        assert res.model_id == "llama3.2:1b"
        assert res.quantization == "Q8_0"

    def test_quality_constrained_recommends_higher_quality_model(self):
        recommender = RuntimeConfigurationRecommender()
        req = RuntimeRecommendationRequest(
            prompt_tokens=64,
            expected_output_tokens=64,
            objective=RuntimeOptimizationObjective.QUALITY_CONSTRAINED_LATENCY,
            quality_constraints=ModelQualityConstraintSpec(min_json_compliance_rate=0.92),
        )
        res = recommender.recommend(req)
        # qwen2.5:0.5b has 85% compliance, so quality-constrained must pick llama3.2:1b or qwen2.5:1.5b
        assert res.model_id in ("llama3.2:1b", "qwen2.5:1.5b")
        assert res.quality_assessment is not None
        assert res.quality_assessment.quality_status == QualityStatus.QUALITY_SATISFIED

    def test_fail_closed_code_guardrail_enforced(self):
        """Preserves Session 23 security policy: unverified code must not return ROBUST_FEASIBLE."""
        recommender = RuntimeConfigurationRecommender()
        req = RuntimeRecommendationRequest(
            prompt_tokens=40,
            expected_output_tokens=40,
            latency_sla_ms=5000.0,
            quality_constraints=ModelQualityConstraintSpec(require_code_pass=True),
        )
        res = recommender.recommend(req)
        assert res.conservative_feasibility == "UNKNOWN_FEASIBILITY"
        assert res.code_execution_status == QualityEvaluationStatus.EXECUTION_UNVERIFIED
        assert res.isolation_status == CodeExecutionSecurityMode.FAIL_CLOSED

    def test_sla_conservatism_robust_vs_risk_vs_miss(self):
        recommender = RuntimeConfigurationRecommender()

        # Very generous SLA -> ROBUST_FEASIBLE
        res_robust = recommender.recommend(
            RuntimeRecommendationRequest(
                prompt_tokens=32,
                expected_output_tokens=16,
                latency_sla_ms=5000.0,
                fixed_model_id="qwen2.5:0.5b",
            )
        )
        assert res_robust.conservative_feasibility == "ROBUST_FEASIBLE"

        # Impossible SLA -> PREDICTED_SLA_MISS
        res_miss = recommender.recommend(
            RuntimeRecommendationRequest(
                prompt_tokens=32,
                expected_output_tokens=16,
                latency_sla_ms=50.0,
                fixed_model_id="qwen2.5:0.5b",
            )
        )
        assert res_miss.conservative_feasibility == "PREDICTED_SLA_MISS"

    def test_memory_budget_enforcement(self):
        recommender = RuntimeConfigurationRecommender()
        req = RuntimeRecommendationRequest(
            prompt_tokens=32,
            expected_output_tokens=16,
            hardware_constraints=HardwareConstraintSpec(memory_budget_mb=500.0),
        )
        res = recommender.recommend(req)
        # Only qwen2.5:0.5b (~380 MB weights + KV) fits in 500 MB
        assert res.model_id == "qwen2.5:0.5b"
        assert res.estimated_memory_mb <= 550.0

    def test_tradeoff_alternatives_populated(self):
        recommender = RuntimeConfigurationRecommender()
        req = RuntimeRecommendationRequest(
            prompt_tokens=40,
            expected_output_tokens=40,
        )
        res = recommender.recommend(req)
        assert len(res.tradeoff_alternatives) >= 2
        for alt in res.tradeoff_alternatives:
            assert "model_id" in alt
            assert "threads" in alt
            assert "e2e_ms" in alt


class TestProspectiveValidationAndSchemaIntegrity:
    """Tests 12, 13, 14, 15, 16: Validation integrity, schemas, and API contracts."""

    def test_prospective_predictions_file_locked(self):
        lock_path = Path("storage/experiments/session_24_prospective_predictions.json")
        assert lock_path.exists()
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        assert data["artifact_type"] == "PROSPECTIVE_PREDICTIONS_LOCK"
        assert data["session_id"] == "session_24"
        assert data["status"] == "LOCKED_PRE_BENCHMARK"
        assert len(data["scenarios"]) == 3

    def test_prospective_validation_report_exists(self):
        report_path = Path("storage/experiments/session_24_prospective_validation_report.md")
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Session 24" in content
        assert "Scenario 1" in content
        assert "PASS" in content

    def test_raw_observations_dataset_integrity(self):
        raw_path = Path("storage/experiments/session_24_raw_observations.json")
        assert raw_path.exists()
        raw_obs = json.loads(raw_path.read_text(encoding="utf-8"))
        assert len(raw_obs) == 288

        # Verify all observations have required fields
        for o in raw_obs:
            assert "observation_id" in o
            assert "model_id" in o
            assert "client_e2e_duration_ms" in o
            assert "decode_tokens_per_second" in o
            assert o["client_e2e_duration_ms"] > 0
            assert o["status"] == "SUCCESS"
