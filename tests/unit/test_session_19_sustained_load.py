"""Unit tests for Session 19 Sustained-Load Reliability & Operating Regimes.

Verifies operating regime classification, risk-aware dual-bound predictions,
conservative SLA decision policies, worst-case latency bounding, and API serialization.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ModelResidencyState,
)
from llm_ii_engine.optimizer.models import (
    FeasibilityStatus,
    OptimizationObjective,
)
from llm_ii_engine.recommendation.engine import WorkloadRecommendationEngine
from llm_ii_engine.recommendation.models import (
    OperatingRegime,
    RecommendationStatus,
    WorkloadRecommendationRequest,
)


@pytest.fixture
def engine() -> WorkloadRecommendationEngine:
    root = Path(__file__).resolve().parent.parent.parent
    return WorkloadRecommendationEngine(dataset_path=root)


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


class TestSession19OperatingRegimes:
    """Verifies operating regime classification across distinct workload conditions."""

    def test_regime_classification_cold_start(self, engine: WorkloadRecommendationEngine) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=46,
            expected_output_tokens=16,
            require_cold_start=True,
            expected_model_residency=ModelResidencyState.COLD,
        )
        rec = engine.recommend(req)
        assert rec.operating_regime == OperatingRegime.COLD_START
        assert rec.worst_case_latency_ms is not None
        assert rec.worst_case_latency_ms > (rec.predicted_metrics.client_e2e_duration_ms or 0.0)

    def test_regime_classification_warm_short_burst(self, engine: WorkloadRecommendationEngine) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=46,
            expected_output_tokens=16,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
        )
        rec = engine.recommend(req)
        assert rec.operating_regime == OperatingRegime.WARM_SHORT_BURST

    def test_regime_classification_warm_sustained_generation(self, engine: WorkloadRecommendationEngine) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=119,
            expected_output_tokens=128,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
        )
        rec = engine.recommend(req)
        assert rec.operating_regime == OperatingRegime.WARM_SUSTAINED_GENERATION

    def test_regime_classification_prompt_prefix_reuse(self, engine: WorkloadRecommendationEngine) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=677,
            expected_output_tokens=32,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.WARM_PREFIX,
        )
        rec = engine.recommend(req)
        assert rec.operating_regime == OperatingRegime.PROMPT_PREFIX_REUSE


class TestSession19ConservativeSLAPolicy:
    """Verifies risk-aware dual-bound feasibility and empirical violation frequency reporting."""

    def test_critical_sla_nominal_feasible_robust_risk(self, engine: WorkloadRecommendationEngine) -> None:
        """Predictive case (P=100, O=128, SLA=8000ms):

        Nominal forecast (~5,900 ms) is FEASIBLE, but worst-case (~10,200 ms) violates SLA.
        Must flag NOMINAL_FEASIBLE_ROBUST_RISK and set ELEVATED_SUSTAINED_LOAD_RISK.
        """
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=128,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
            latency_sla_ms=8000.0,
            objective=OptimizationObjective.MAXIMIZE_DECODE_THROUGHPUT,
        )
        rec = engine.recommend(req)
        assert rec.status == RecommendationStatus.RECOMMENDED
        assert rec.conservative_feasibility == "NOMINAL_FEASIBLE_ROBUST_RISK"
        assert rec.sla_risk_flag == "ELEVATED_SUSTAINED_LOAD_RISK"
        assert rec.worst_case_latency_ms is not None
        assert rec.worst_case_latency_ms > 8000.0
        assert rec.historical_sla_violation_frequency_pct is not None
        assert rec.historical_sla_violation_frequency_pct > 0.0

        # In SLA assessments breakdown
        sla = rec.sla_assessments["latency_sla"]
        assert sla["conservative_feasibility"] == "NOMINAL_FEASIBLE_ROBUST_RISK"
        assert sla["sla_risk_flag"] == "ELEVATED_SUSTAINED_LOAD_RISK"
        assert "risk_warning" in sla

    def test_directly_measured_sustained_load_miss_is_proven_infeasible(self, engine: WorkloadRecommendationEngine) -> None:
        """Measured case (P=119, O=128, SLA=8000ms):

        Because Session 18 directly measured 10,282.98 ms under exact matching conditions,
        the engine identifies direct empirical infeasibility.
        """
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=119,
            expected_output_tokens=128,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
            latency_sla_ms=8000.0,
        )
        rec = engine.recommend(req)
        assert rec.status == RecommendationStatus.INFEASIBLE
        assert rec.conservative_feasibility == FeasibilityStatus.PROVEN_INFEASIBLE.value

    def test_generous_sla_is_robust_feasible(self, engine: WorkloadRecommendationEngine) -> None:
        """When SLA is generous (15,000 ms), both nominal and worst-case satisfy SLA."""
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=119,
            expected_output_tokens=128,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
            latency_sla_ms=15000.0,
        )
        rec = engine.recommend(req)
        assert rec.status == RecommendationStatus.RECOMMENDED
        assert rec.conservative_feasibility == "ROBUST_FEASIBLE"
        assert rec.sla_risk_flag is None

    def test_unachievable_sla_is_predicted_miss_or_infeasible(self, engine: WorkloadRecommendationEngine) -> None:
        """When SLA (3,000 ms) is lower than nominal forecast (~6,000 ms), must be PREDICTED_SLA_MISS."""
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=119,
            expected_output_tokens=128,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
            latency_sla_ms=3000.0,
        )
        rec = engine.recommend(req)
        assert rec.status == RecommendationStatus.INFEASIBLE
        assert rec.conservative_feasibility in (
            FeasibilityStatus.PREDICTED_SLA_MISS.value,
            FeasibilityStatus.PROVEN_INFEASIBLE.value,
        )


class TestSession19APIEndpoints:
    """Verifies that API serialization includes operating regime and risk fields."""

    def test_post_workload_recommendation_includes_regime_and_risk(self, client: TestClient) -> None:
        payload = {
            "model_id": "llama3.2:1b",
            "prompt_tokens": 100,
            "expected_output_tokens": 128,
            "concurrency": 1,
            "latency_sla_ms": 8000.0,
            "expected_model_residency": "warm",
            "expected_prefix_cache": "cold_prefix",
        }
        res = client.post("/api/v1/recommendation/workload", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert "operating_regime" in data
        assert data["operating_regime"] == "WARM_SUSTAINED_GENERATION"
        assert "worst_case_latency_ms" in data
        assert data["worst_case_latency_ms"] > 8000.0
        assert "conservative_feasibility" in data
        assert data["conservative_feasibility"] == "NOMINAL_FEASIBLE_ROBUST_RISK"
        assert "sla_risk_flag" in data
        assert data["sla_risk_flag"] == "ELEVATED_SUSTAINED_LOAD_RISK"
        assert "historical_sla_violation_frequency_pct" in data
        assert data["historical_sla_violation_frequency_pct"] > 0.0
