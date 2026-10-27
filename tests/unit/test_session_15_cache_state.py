"""Unit tests for Session 15: Cache-State-Controlled Benchmarking & Reproducible Performance Modeling.

Verifies:
1. Enum semantics and string serialization for ModelResidencyState, CacheStateClassification, and PrefixRelationshipType.
2. Full backward compatibility when deserializing historical Session 12B and Session 13 observation records.
3. Predecessor request tracking, cache telemetry extraction, and schema stability.
4. Prospective feature safety in RidgeRegressionPredictor (forbids post-execution telemetry).
5. Optimizer evidence classification rules (rejection of warm-cache evidence for cold-start SLAs).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_ii_engine.benchmark.dataset_export import EmpiricalObservationRecord
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    EngineType,
    ExecutionMode,
    ModelResidencyState,
    PrefixRelationshipType,
    ProvenanceType,
    QuantizationType,
    TokenCountSource,
)
from llm_ii_engine.core.schemas import ModelSpec, WorkloadSpec
from llm_ii_engine.discovery.models import ComputeCapabilitySpec, GPUSpecification
from llm_ii_engine.optimizer.engine import ConfigurationOptimizer
from llm_ii_engine.optimizer.models import (
    EvidenceStatus,
    FeasibilityStatus,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    SearchSpace,
)
from llm_ii_engine.prediction.baselines import RidgeRegressionPredictor
from llm_ii_engine.prediction.dataset import DatasetBuilder
from llm_ii_engine.prediction.models import PredictionRecord, PredictionTarget


class TestCacheStateEnums:
    """Verifies typed runtime-state and cache-state enum classifications."""

    def test_model_residency_enum_values(self) -> None:
        assert ModelResidencyState.COLD.value == "cold"
        assert ModelResidencyState.WARM.value == "warm"
        assert ModelResidencyState.UNKNOWN.value == "unknown"

    def test_cache_state_classification_values(self) -> None:
        assert CacheStateClassification.COLD_PREFIX.value == "cold_prefix"
        assert CacheStateClassification.WARM_PREFIX.value == "warm_prefix"
        assert CacheStateClassification.PARTIAL_PREFIX.value == "partial_prefix"
        assert CacheStateClassification.UNKNOWN.value == "unknown"

    def test_prefix_relationship_values(self) -> None:
        assert PrefixRelationshipType.IDENTICAL.value == "identical"
        assert PrefixRelationshipType.SHARED_PREFIX.value == "shared_prefix"
        assert PrefixRelationshipType.UNRELATED.value == "unrelated"
        assert PrefixRelationshipType.NONE.value == "none"


class TestHistoricalBackwardCompatibility:
    """Verifies that Session 12B and Session 13 historical datasets load without errors."""

    def test_historical_pilot_jsonl_loads_with_default_none_cache_fields(self) -> None:
        pilot_path = Path("storage/datasets/pilot_llama3_2_1b.jsonl")
        assert pilot_path.exists(), "pilot_llama3_2_1b.jsonl must exist"

        lines = [line.strip() for line in pilot_path.read_text(encoding="utf-8").split("\n") if line.strip()]
        assert len(lines) == 9

        for line in lines:
            data = json.loads(line)
            # Historical records do not have Session 15 fields
            rec = EmpiricalObservationRecord.model_validate(data)
            assert rec.model_residency_state is None
            assert rec.cache_state is None
            assert rec.predecessor_request_id is None
            assert rec.server_prompt_eval_cached_count is None
            assert rec.prefix_relationship is None

    def test_historical_expansion_jsonl_loads_with_default_none_cache_fields(self) -> None:
        exp_path = Path("storage/datasets/expansion_llama3_2_1b.jsonl")
        assert exp_path.exists(), "expansion_llama3_2_1b.jsonl must exist"

        lines = [line.strip() for line in exp_path.read_text(encoding="utf-8").split("\n") if line.strip()]
        assert len(lines) == 21

        for line in lines:
            data = json.loads(line)
            rec = EmpiricalObservationRecord.model_validate(data)
            assert rec.model_residency_state is None
            assert rec.cache_state is None
            assert rec.predecessor_request_id is None
            assert rec.server_prompt_eval_cached_count is None
            assert rec.prefix_relationship is None

    def test_dataset_builder_maps_historical_records_safely(self) -> None:
        pilot_path = Path("storage/datasets/pilot_llama3_2_1b.jsonl")
        lines = [line.strip() for line in pilot_path.read_text(encoding="utf-8").split("\n") if line.strip()]
        obs_records = [EmpiricalObservationRecord.model_validate(json.loads(line)) for line in lines]

        model_spec = ModelSpec(
            model_id="llama3.2:1b",
            family="llama",
            parameter_count_billion=1.23,
            context_window=131072,
            num_layers=16,
            num_attention_heads=16,
            num_kv_heads=8,
            hidden_dimension=2048,
            quantization=QuantizationType.GGUF_Q8_0,
        )

        pred_dataset = DatasetBuilder.from_empirical_observations(
            observations=obs_records,
            model_spec=model_spec,
        )

        assert len(pred_dataset.records) == 9
        for r in pred_dataset.records:
            assert r.model_residency_state is None
            assert r.cache_state is None
            assert r.predecessor_request_id is None
            assert r.server_prompt_eval_cached_count is None
            assert r.prefix_relationship is None


class TestProspectiveFeatureSafety:
    """Verifies that post-execution empirical telemetry cannot leak into prospective predictors."""

    def test_ridge_regression_rejects_cached_token_count_feature(self) -> None:
        with pytest.raises(ValueError, match="Cannot use post-execution empirical telemetry fields"):
            RidgeRegressionPredictor(
                target=PredictionTarget.DURATION_MS,
                feature_names=["prompt_tokens", "server_prompt_eval_cached_count"],
            )

    def test_ridge_regression_rejects_actual_output_tokens_feature(self) -> None:
        with pytest.raises(ValueError, match="Cannot use post-execution empirical telemetry fields"):
            RidgeRegressionPredictor(
                target=PredictionTarget.DURATION_MS,
                feature_names=["prompt_tokens", "actual_output_tokens"],
            )

    def test_ridge_regression_default_features_contain_no_post_execution_fields(self) -> None:
        for target, features in RidgeRegressionPredictor.DEFAULT_FEATURES_BY_TARGET.items():
            for f in features:
                assert f not in RidgeRegressionPredictor.POST_EXECUTION_FIELDS, (
                    f"Target {target.value} contains forbidden post-execution feature {f}"
                )


def _make_dummy_gpu(catalog_id: str = "catalog-workstation-cpu", name: str = "Host Hardware", vram_gb: float = 32.0) -> GPUSpecification:
    return GPUSpecification(
        catalog_id=catalog_id,
        name=name,
        architecture="x86_64",
        form_factor="Host",
        memory_capacity_gb=vram_gb,
        memory_type="DDR5",
        theoretical_memory_bandwidth_gbps=89.6,
        compute=ComputeCapabilitySpec(
            fp32_tflops=1.0,
            fp16_dense_tflops=1.0,
            fp16_sparse_tflops=1.0,
            bf16_dense_tflops=1.0,
            bf16_sparse_tflops=1.0,
            int8_dense_tops=1.0,
            int8_sparse_tops=1.0,
        ),
        tdp_watts=125,
        source_reference="Host Specification",
    )


class TestOptimizerColdStartEvidenceEnforcement:
    """Verifies that the optimizer prevents warm-cache contamination of cold-start SLAs."""

    @pytest.fixture
    def warm_cache_observation(self) -> PredictionRecord:
        return PredictionRecord(
            record_id="rec-warm-cache-001",
            experiment_id="exp-warm-test",
            suite_id="suite-warm",
            request_id="req-warm-1",
            model_id="llama3.2:1b",
            model_digest="baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            is_warmup=False,
            parameter_count_billion=1.23,
            num_layers=16,
            num_attention_heads=16,
            num_kv_heads=8,
            hidden_dimension=2048,
            quantization=QuantizationType.GGUF_Q8_0,
            hardware_catalog_id="catalog-workstation-cpu",
            concurrency=1,
            prompt_tokens=119,
            prompt_token_count_source=TokenCountSource.SERVER_REPORTED,
            requested_output_tokens=32,
            actual_output_tokens=32,
            output_tokens=32,
            token_count_source=TokenCountSource.SERVER_REPORTED,
            ttft_ms=50.0,  # Fast warm-cache TTFT
            duration_ms=1200.0,
            server_prompt_eval_duration_ms=45.0,
            server_eval_duration_ms=1150.0,
            model_residency_state=ModelResidencyState.WARM,
            cache_state=CacheStateClassification.WARM_PREFIX,
            predecessor_request_id="req-prev",
            server_prompt_eval_cached_count=118,
            prefix_relationship=PrefixRelationshipType.IDENTICAL,
            is_eligible_for_ttft=True,
            is_eligible_for_duration=True,
        )

    def test_optimizer_accepts_warm_evidence_when_cold_start_not_required(
        self, warm_cache_observation: PredictionRecord
    ) -> None:
        optimizer = ConfigurationOptimizer(
            empirical_observations=[warm_cache_observation],
        )

        model_spec = ModelSpec(
            model_id="llama3.2:1b",
            family="llama",
            parameter_count_billion=1.23,
            context_window=131072,
            num_layers=16,
            num_attention_heads=16,
            num_kv_heads=8,
            hidden_dimension=2048,
            quantization=QuantizationType.GGUF_Q8_0,
        )

        gpu = _make_dummy_gpu()
        request = OptimizationRequest(
            search_space=SearchSpace(
                models=[model_spec],
                quantizations=[QuantizationType.GGUF_Q8_0],
                concurrencies=[1],
                engines=[EngineType.OLLAMA],
            ),
            hardware_spec=gpu,
            workload=WorkloadSpec(prompt_tokens=119, output_tokens=32),
            objectives=[OptimizationObjective.MINIMIZE_TTFT],
            constraints=OptimizationConstraints(
                max_ttft_ms=100.0,
                require_cold_start_feasibility=False,  # Cold-start not required
            ),
        )

        result = optimizer.optimize(request)
        cand = result.all_evaluations[0]
        # Should match and evaluate as MEASURED_EVALUATED and FEASIBLE
        assert cand.evidence_status[OptimizationObjective.MINIMIZE_TTFT] == EvidenceStatus.MEASURED_EVALUATED
        assert cand.feasibility_status == FeasibilityStatus.FEASIBLE

    def test_optimizer_rejects_warm_evidence_when_cold_start_required(
        self, warm_cache_observation: PredictionRecord
    ) -> None:
        optimizer = ConfigurationOptimizer(
            empirical_observations=[warm_cache_observation],
        )

        model_spec = ModelSpec(
            model_id="llama3.2:1b",
            family="llama",
            parameter_count_billion=1.23,
            context_window=131072,
            num_layers=16,
            num_attention_heads=16,
            num_kv_heads=8,
            hidden_dimension=2048,
            quantization=QuantizationType.GGUF_Q8_0,
        )

        gpu = _make_dummy_gpu()
        request = OptimizationRequest(
            search_space=SearchSpace(
                models=[model_spec],
                quantizations=[QuantizationType.GGUF_Q8_0],
                concurrencies=[1],
                engines=[EngineType.OLLAMA],
            ),
            hardware_spec=gpu,
            workload=WorkloadSpec(prompt_tokens=119, output_tokens=32),
            objectives=[OptimizationObjective.MINIMIZE_TTFT],
            constraints=OptimizationConstraints(
                max_ttft_ms=100.0,
                require_cold_start_feasibility=True,  # Cold-start explicitly required!
            ),
        )

        result = optimizer.optimize(request)
        cand = result.all_evaluations[0]
        # Must NOT claim MEASURED_EVALUATED; must be demoted to INSUFFICIENT_EVIDENCE
        assert cand.evidence_status[OptimizationObjective.MINIMIZE_TTFT] == EvidenceStatus.INSUFFICIENT_EVIDENCE
        # Must result in UNKNOWN_FEASIBILITY rather than asserting FEASIBLE
        assert cand.feasibility_status == FeasibilityStatus.UNKNOWN_FEASIBILITY
        assert any("insufficient evidence" in r.lower() for r in cand.rejection_reasons)
