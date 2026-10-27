"""Unit tests for Session 18 Conservative SLA Decisions & Empirical Feasibility.

Validates:
1. Proven Hardware Incompatibility (unsupported model, unavailable accelerator).
2. Proven Resource Infeasibility (sequence length > context window, memory bandwidth lower bound).
3. Predicted SLA Miss (in-distribution forecast fails constraint without physical violation).
4. Measured SLA Miss (direct empirical measurement under matching conditions exceeds SLA).
5. Unknown Feasibility due to OOD (Scenario F: extrapolated forecast cannot prove infeasibility).
6. Unknown Feasibility due to Missing TTFT (unverified token boundaries in chunk-streaming).
7. Narrow SLA Headroom Risk Warnings (headroom < empirical error margin).
8. Evidence Tier Qualifications (exact match vs conditional evidence on UNKNOWN runtime state).
9. Empirical Coverage Semantics (exploratory prediction range, not statistical 95% CI).
"""

from pathlib import Path

import pytest

from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ModelResidencyState,
)
from llm_ii_engine.optimizer.models import FeasibilityStatus
from llm_ii_engine.recommendation.engine import WorkloadRecommendationEngine
from llm_ii_engine.recommendation.models import (
    RecommendationEvidenceSource,
    RecommendationStatus,
    WorkloadRecommendationRequest,
)


@pytest.fixture
def engine() -> WorkloadRecommendationEngine:
    root = Path(__file__).resolve().parent.parent.parent
    return WorkloadRecommendationEngine(dataset_path=root)


class TestConservativeSLAFeasibilityDecisions:
    """Rigorous verification of conservative feasibility decision taxonomy."""

    def test_proven_hardware_incompatibility_returns_unsupported(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """1. Proven Hardware Incompatibility: requesting model not installed/authorized."""
        req = WorkloadRecommendationRequest(
            model_id="unauthorized-model:70b",
            prompt_tokens=64,
            expected_output_tokens=32,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.UNSUPPORTED
        assert res.evidence_source == RecommendationEvidenceSource.INSUFFICIENT_EVIDENCE
        assert res.generalization_status.is_out_of_distribution is True

    def test_proven_resource_infeasibility_context_window_exceeded(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """2a. Proven Resource Infeasibility: sequence length > 131,072 max context."""
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=130_000,
            expected_output_tokens=2_000,  # 132,000 > 131,072
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.INFEASIBLE
        assert "context_limit" in res.sla_assessments
        assert res.sla_assessments["context_limit"]["status"] == FeasibilityStatus.PROVEN_INFEASIBLE.value
        assert "exceeds maximum model context window" in res.sla_assessments["context_limit"]["reason"]

    def test_proven_resource_infeasibility_physical_bandwidth_limit(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """2b. Proven Resource Infeasibility: requested latency SLA violates memory bandwidth lower bound."""
        # 100 output tokens on 1.32GB weights requires at minimum ~1,473 ms on dual-channel DDR5
        # An SLA of 50 ms violates physical memory transfer time
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=100,
            latency_sla_ms=50.0,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.INFEASIBLE
        assert "latency_sla" in res.sla_assessments
        sla = res.sla_assessments["latency_sla"]
        assert sla["status"] == FeasibilityStatus.PROVEN_INFEASIBLE.value
        assert "violates theoretical memory-bandwidth lower bound" in sla["physical_proof"]
        assert sla["theoretical_min_ms"] > 50.0

    def test_predicted_sla_miss_in_distribution(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """3. Predicted SLA Miss: in-distribution request fails SLA without hard physical violation."""
        # P=300, O=64 is in-distribution [42, 677] x [16, 128], forecast is ~3,700 ms
        # Request SLA of 1,500 ms (achievable physically on GPU, but not on CPU forecast)
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=300,
            expected_output_tokens=64,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
            latency_sla_ms=1500.0,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.INFEASIBLE
        assert res.generalization_status.is_out_of_distribution is False
        assert "latency_sla" in res.sla_assessments
        sla = res.sla_assessments["latency_sla"]
        assert sla["status"] == FeasibilityStatus.PREDICTED_SLA_MISS.value
        assert sla["violation_ms"] > 0
        assert "under in-distribution model" in res.explanation_summary

    def test_measured_sla_miss_under_matching_conditions(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """4. Measured SLA Miss: direct empirical measurement under exact conditions exceeds SLA."""
        # P=119, O=32 cold-start was directly measured in s15-expA-rep1 at 3,006.96 ms
        # If user requests SLA of 2,000 ms with cold start required
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=119,
            expected_output_tokens=32,
            expected_model_residency=ModelResidencyState.COLD,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
            require_cold_start=True,
            latency_sla_ms=2000.0,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.INFEASIBLE
        assert res.evidence_source == RecommendationEvidenceSource.DIRECTLY_MEASURED
        sla = res.sla_assessments["latency_sla"]
        assert sla["status"] == FeasibilityStatus.PROVEN_INFEASIBLE.value
        assert sla["proof_basis"] == "DIRECT_MEASUREMENT"
        assert sla["measured_ms"] > 2000.0

    def test_unknown_feasibility_due_to_ood_scenario_f(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """5. Unknown Feasibility due to OOD: Scenario F (P=4000, O=500, SLA 30s) must NOT be proven infeasible."""
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=4000,
            expected_output_tokens=500,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
            latency_sla_ms=30000.0,
        )
        res = engine.recommend(req)
        # Must be INCONCLUSIVE, NOT INFEASIBLE
        assert res.status == RecommendationStatus.INCONCLUSIVE
        assert res.generalization_status.is_out_of_distribution is True
        assert res.evidence_source == RecommendationEvidenceSource.EXPLORATORY_PREDICTION
        sla = res.sla_assessments["latency_sla"]
        # Must be UNKNOWN_FEASIBILITY, NOT PROVEN_INFEASIBLE
        assert sla["status"] == FeasibilityStatus.UNKNOWN_FEASIBILITY.value
        assert "cannot be mathematically proven without live empirical benchmarking" in sla["qualification_note"]
        assert sla["estimated_violation_ms"] > 0

    def test_unknown_feasibility_due_to_missing_ttft(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """6. Unknown Feasibility due to Missing TTFT: chunk streaming cannot verify token timestamps."""
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=32,
            expected_output_tokens=16,
            ttft_sla_ms=100.0,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.INCONCLUSIVE
        assert "ttft_sla" in res.sla_assessments
        sla = res.sla_assessments["ttft_sla"]
        assert sla["status"] == FeasibilityStatus.UNKNOWN_FEASIBILITY.value
        assert "Ollama chunk streaming emits text byte chunks" in sla["limitation_note"]

    def test_narrow_sla_headroom_risk_warning(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """7. Narrow SLA Headroom: preserves uncertainty when headroom < empirical error margin."""
        # Forecast for P=32, O=16 warm is ~807 ms. If SLA is 900 ms, headroom is ~93 ms < 385.88 ms MAE
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=32,
            expected_output_tokens=16,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
            latency_sla_ms=900.0,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.RECOMMENDED
        sla = res.sla_assessments["latency_sla"]
        assert sla["status"] == FeasibilityStatus.FEASIBLE.value
        assert "risk_warning" in sla
        assert "Narrow SLA headroom" in sla["risk_warning"]
        assert any("Narrow SLA headroom" in f for f in res.influencing_factors)


class TestEvidenceTierAndExploratoryRanges:
    """Verify strict evidence classification and non-misleading prediction range labels."""

    def test_exact_match_with_unknown_cache_yields_conditional_evidence(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """Scenario B verification: P=677, O=32 with UNKNOWN cache state yields DIRECTLY_MEASURED_CONDITIONAL."""
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=677,
            expected_output_tokens=32,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.UNKNOWN,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.RECOMMENDED
        assert res.evidence_source == RecommendationEvidenceSource.DIRECTLY_MEASURED_CONDITIONAL
        assert any("conditionally" in f for f in res.influencing_factors)

    def test_exploratory_prediction_range_replaces_confidence_interval(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        """Verify exploratory_prediction_range is populated and labeled EXPLORATORY_RANGE, not 95% CI."""
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=50,
        )
        res = engine.recommend(req)
        assert res.exploratory_prediction_range is not None
        assert len(res.exploratory_prediction_range) == 2
        assert res.confidence_label_status == "EXPLORATORY_RANGE"
        assert res.range_coverage_percent == 74.19
        assert "95%" not in res.explanation_summary
