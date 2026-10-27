"""Unit tests for Session 17 Workload Recommendation Engine and API contracts.

Tests cover:
- Recommendation request validation and constraint consistency
- Candidate compatibility and unsupported model handling
- Evidence classification and provenance
- Unknown cache and residency state handling
- Cold-start SLA incorporation and infeasibility
- TTFT SLA unknown feasibility on chunk-streaming
- Out-of-distribution input detection
- FastAPI endpoint contracts and error mappings
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from llm_ii_engine.api.app import create_app
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ModelResidencyState,
)
from llm_ii_engine.optimizer.models import (
    FeasibilityStatus,
)
from llm_ii_engine.recommendation.engine import WorkloadRecommendationEngine
from llm_ii_engine.recommendation.models import (
    RecommendationEvidenceSource,
    RecommendationStatus,
    WorkloadRecommendationRequest,
)


class TestRecommendationRequestValidation:
    """Validate input schemas, boundary conditions, and contradiction detection."""

    def test_valid_request_creation(self) -> None:
        req = WorkloadRecommendationRequest(
            prompt_tokens=46,
            expected_output_tokens=32,
            latency_sla_ms=3000.0,
        )
        assert req.prompt_tokens == 46
        assert req.expected_output_tokens == 32
        assert req.latency_sla_ms == 3000.0
        assert req.model_id == "llama3.2:1b"
        assert req.expected_model_residency == ModelResidencyState.UNKNOWN

    def test_negative_or_zero_prompt_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadRecommendationRequest(prompt_tokens=0, expected_output_tokens=32)
        with pytest.raises(ValidationError):
            WorkloadRecommendationRequest(prompt_tokens=-10, expected_output_tokens=32)

    def test_negative_or_zero_output_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadRecommendationRequest(prompt_tokens=32, expected_output_tokens=0)
        with pytest.raises(ValidationError):
            WorkloadRecommendationRequest(prompt_tokens=32, expected_output_tokens=-5)

    def test_contradictory_cold_start_and_warm_residency_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Contradictory constraints"):
            WorkloadRecommendationRequest(
                prompt_tokens=32,
                expected_output_tokens=16,
                require_cold_start=True,
                expected_model_residency=ModelResidencyState.WARM,
            )


class TestRecommendationEngineCompatibilityAndEvidence:
    """Validate candidate selection, evidence provenance, and residency logic."""

    @pytest.fixture
    def engine(self) -> WorkloadRecommendationEngine:
        root = Path(__file__).resolve().parent.parent.parent
        return WorkloadRecommendationEngine(dataset_path=root)

    def test_unsupported_model_returns_unsupported_status(self, engine: WorkloadRecommendationEngine) -> None:
        req = WorkloadRecommendationRequest(
            model_id="unauthorized-huge-model:70b",
            prompt_tokens=32,
            expected_output_tokens=16,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.UNSUPPORTED
        assert res.selected_configuration is None
        assert res.evidence_source == RecommendationEvidenceSource.INSUFFICIENT_EVIDENCE
        assert res.generalization_status.is_out_of_distribution is True

    def test_exact_empirical_match_yields_directly_measured_evidence(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        # P=46, O=32 with matching warm residency and cold prefix yields unconditional DIRECTLY_MEASURED
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=46,
            expected_output_tokens=32,
            expected_model_residency=ModelResidencyState.WARM,
            expected_prefix_cache=CacheStateClassification.WARM_PREFIX,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.RECOMMENDED
        assert res.evidence_source == RecommendationEvidenceSource.DIRECTLY_MEASURED
        assert res.predicted_metrics.client_e2e_duration_ms is not None
        assert res.predicted_metrics.server_decode_throughput_tok_s is not None

    def test_empirical_match_with_unknown_runtime_state_yields_conditional_evidence(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        # P=46, O=32 with UNKNOWN residency/cache yields DIRECTLY_MEASURED_CONDITIONAL
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=46,
            expected_output_tokens=32,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.RECOMMENDED
        assert res.evidence_source == RecommendationEvidenceSource.DIRECTLY_MEASURED_CONDITIONAL
        assert any("conditionally" in f for f in res.influencing_factors)

    def test_in_distribution_unmeasured_workload_yields_held_out_validated_prediction(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        # P=100, O=50 is in distribution [42, 677] x [16, 128] but unmeasured
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=50,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.RECOMMENDED
        assert res.evidence_source == RecommendationEvidenceSource.HELD_OUT_VALIDATED_PREDICTION
        assert res.generalization_status.is_out_of_distribution is False
        assert len(res.ood_warnings) == 0

    def test_unknown_residency_defaults_warm_base_with_cold_penalty_disclosed(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=50,
            expected_model_residency=ModelResidencyState.UNKNOWN,
        )
        res = engine.recommend(req)
        assert res.predicted_metrics.server_load_duration_ms == 0.0
        assert "Residency UNKNOWN" in res.cold_start_assumptions or "1,038" in res.cold_start_assumptions

    def test_cold_start_requirement_incorporates_load_duration(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=50,
            require_cold_start=True,
        )
        res = engine.recommend(req)
        assert res.predicted_metrics.server_load_duration_ms == pytest.approx(1037.74, rel=1e-2)
        assert "Cold start accounted for" in res.cold_start_assumptions or "penalty" in res.cold_start_assumptions


class TestSLAFeasibilityAndExtrapolation:
    """Validate hard SLA feasibility boundaries, TTFT handling, and OOD extrapolation."""

    @pytest.fixture
    def engine(self) -> WorkloadRecommendationEngine:
        root = Path(__file__).resolve().parent.parent.parent
        return WorkloadRecommendationEngine(dataset_path=root)

    def test_latency_sla_satisfied_produces_feasible_status(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=32,
            expected_output_tokens=16,
            latency_sla_ms=5000.0,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.RECOMMENDED
        assert "latency_sla" in res.sla_assessments
        assert res.sla_assessments["latency_sla"]["status"] == FeasibilityStatus.FEASIBLE.value

    def test_impossible_latency_sla_produces_infeasible_status(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=100,
            expected_output_tokens=50,
            latency_sla_ms=100.0,  # Impossible on CPU
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.INFEASIBLE
        assert res.sla_assessments["latency_sla"]["status"] == FeasibilityStatus.PROVEN_INFEASIBLE.value
        assert res.sla_assessments["latency_sla"]["violation_ms"] > 0

    def test_ttft_sla_produces_unknown_feasibility_and_inconclusive_status(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=46,
            expected_output_tokens=32,
            ttft_sla_ms=100.0,
        )
        res = engine.recommend(req)
        assert res.status == RecommendationStatus.INCONCLUSIVE
        assert "ttft_sla" in res.sla_assessments
        assert res.sla_assessments["ttft_sla"]["status"] == FeasibilityStatus.UNKNOWN_FEASIBILITY.value
        assert "Ollama chunk streaming emits text byte chunks" in res.sla_assessments["ttft_sla"]["limitation_note"]

    def test_out_of_distribution_request_triggers_exploratory_warnings(
        self, engine: WorkloadRecommendationEngine
    ) -> None:
        req = WorkloadRecommendationRequest(
            model_id="llama3.2:1b",
            prompt_tokens=4000,
            expected_output_tokens=500,
        )
        res = engine.recommend(req)
        assert res.evidence_source == RecommendationEvidenceSource.EXPLORATORY_PREDICTION
        assert res.generalization_status.is_out_of_distribution is True
        assert len(res.ood_warnings) >= 2
        assert any("Prompt tokens (4000)" in w for w in res.ood_warnings)
        assert any("Output tokens (500)" in w for w in res.ood_warnings)


class TestRecommendationAPIEndpoints:
    """Validate FastAPI routes, error status codes, and serialization contracts."""

    @pytest.fixture
    def client(self) -> TestClient:
        app = create_app("sqlite:///:memory:")
        return TestClient(app)

    def test_get_scenarios_returns_six_definitions(self, client: TestClient) -> None:
        resp = client.get("/api/v1/recommendation/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 6
        scenario_ids = [s["scenario_id"] for s in data]
        assert "SCENARIO_A" in scenario_ids
        assert "SCENARIO_E" in scenario_ids
        assert "SCENARIO_F" in scenario_ids

    def test_post_workload_recommendation_success(self, client: TestClient) -> None:
        payload = {
            "model_id": "llama3.2:1b",
            "prompt_tokens": 32,
            "expected_output_tokens": 16,
            "latency_sla_ms": 2500.0,
            "objective": "minimize_duration",
        }
        resp = client.post("/api/v1/recommendation/workload", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("RECOMMENDED", "INCONCLUSIVE", "INFEASIBLE")
        assert "predicted_metrics" in data
        assert data["predicted_metrics"]["client_e2e_duration_ms"] > 0
        assert data["predicted_metrics"]["server_decode_throughput_tok_s"] > 0

    def test_post_workload_unsupported_model_returns_404(self, client: TestClient) -> None:
        payload = {
            "model_id": "unknown-nonexistent-model",
            "prompt_tokens": 32,
            "expected_output_tokens": 16,
        }
        resp = client.post("/api/v1/recommendation/workload", json=payload)
        assert resp.status_code == 404
        assert "not installed or authorized" in resp.json()["message"]

    def test_post_workload_invalid_tokens_returns_422(self, client: TestClient) -> None:
        payload = {
            "model_id": "llama3.2:1b",
            "prompt_tokens": -5,
            "expected_output_tokens": 16,
        }
        resp = client.post("/api/v1/recommendation/workload", json=payload)
        assert resp.status_code == 422
