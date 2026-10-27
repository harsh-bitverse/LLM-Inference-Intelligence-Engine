"""Unit tests for constrained configuration optimization, Pareto search, and feasibility checks.

Tests:
1. Pareto dominance with minimization and maximization objectives.
2. Multi-trade-off Pareto frontier extraction and dominated candidate elimination.
3. Equal objective vectors and deterministic duplicate handling.
4. Hard-constraint feasibility (context window, engine-quantization, VRAM, and SLAs).
5. Unknown feasibility handling when hardware specifications are missing.
6. Search space budget guardrails and early pruning.
7. Evidence status and provenance preservation (MEASURED vs PREDICTED vs HEURISTIC).
8. Single-candidate selection policies (PRIORITIZE_OBJECTIVE, BALANCED_KNEE_POINT, LOWEST_COST).
9. Zero feasible candidate edge cases without crashing.
10. Strict rejection of unsupported cost optimization without cost inputs.
11. End-to-end optimization workflow with explainable recommendations.
"""

import pytest

from llm_ii_engine.analysis.resource_estimator import ModelResourceEstimator
from llm_ii_engine.core.enums import (
    EngineType,
    ExecutionMode,
    ProvenanceType,
    QuantizationType,
)
from llm_ii_engine.core.schemas import ModelSpec, WorkloadSpec
from llm_ii_engine.discovery.models import (
    ComputeCapabilitySpec,
    GPUSpecification,
)
from llm_ii_engine.optimizer.engine import ConfigurationOptimizer
from llm_ii_engine.optimizer.feasibility import FeasibilityChecker
from llm_ii_engine.optimizer.models import (
    CandidateConfig,
    CandidateEvaluation,
    EvidenceStatus,
    FeasibilityStatus,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
    SearchSpace,
    SelectionPolicy,
)
from llm_ii_engine.optimizer.pareto import ParetoEngine
from llm_ii_engine.optimizer.search import SearchSpaceGenerator
from llm_ii_engine.prediction.baselines import RidgeRegressionPredictor
from llm_ii_engine.prediction.models import (
    PredictionRecord,
    PredictionTarget,
)


def _make_dummy_gpu(name: str = "Test-GPU", vram_gb: float = 24.0) -> GPUSpecification:
    """Helper to construct a valid GPUSpecification fixture."""
    return GPUSpecification(
        catalog_id="test-gpu-24gb",
        name=name,
        architecture="Ada Lovelace",
        form_factor="PCIe",
        memory_capacity_gb=vram_gb,
        memory_type="GDDR6X",
        theoretical_memory_bandwidth_gbps=1000.0,
        compute=ComputeCapabilitySpec(
            fp32_tflops=35.0,
            fp16_dense_tflops=165.0,
            fp16_sparse_tflops=330.0,
            bf16_dense_tflops=165.0,
            bf16_sparse_tflops=330.0,
            int8_dense_tops=330.0,
            int8_sparse_tops=660.0,
        ),
        tdp_watts=450,
        source_reference="Test Reference",
    )


def _make_candidate_eval(
    config_id: str,
    ttft_ms: float = 20.0,
    tpot_ms: float = 15.0,
    throughput_tok_s: float = 65.0,
    vram_mb: float = 8000.0,
    feasibility: FeasibilityStatus = FeasibilityStatus.FEASIBLE,
    reasons: list[str] | None = None,
    evidence: EvidenceStatus = EvidenceStatus.THEORETICAL_ANALYTICAL,
    cost_per_1m: float | None = None,
) -> CandidateEvaluation:
    """Helper to build a CandidateEvaluation for Pareto testing."""
    config = CandidateConfig(
        model_spec=ModelSpec(
            model_id=config_id,
            parameter_count_billion=7.0,
            context_window=8192,
        ),
        quantization=QuantizationType.FP16,
        concurrency=1,
        engine_type=EngineType.MOCK,
    )
    values = {
        OptimizationObjective.MINIMIZE_TTFT: ttft_ms,
        OptimizationObjective.MINIMIZE_TPOT: tpot_ms,
        OptimizationObjective.MAXIMIZE_THROUGHPUT: throughput_tok_s,
        OptimizationObjective.MINIMIZE_VRAM: vram_mb,
    }
    units = {
        OptimizationObjective.MINIMIZE_TTFT: "ms",
        OptimizationObjective.MINIMIZE_TPOT: "ms",
        OptimizationObjective.MAXIMIZE_THROUGHPUT: "tok/s",
        OptimizationObjective.MINIMIZE_VRAM: "MiB",
    }
    if cost_per_1m is not None:
        values[OptimizationObjective.MINIMIZE_COST] = cost_per_1m
        units[OptimizationObjective.MINIMIZE_COST] = "$/1M tokens"

    ev_status = {obj: evidence for obj in values}
    prov = {obj: ProvenanceType.HEURISTIC for obj in values}

    return CandidateEvaluation(
        config=config,
        feasibility_status=feasibility,
        rejection_reasons=reasons or [],
        objective_values=values,
        objective_units=units,
        evidence_status=ev_status,
        provenance=prov,
    )


class TestParetoDominanceAndFrontier:
    """Test suite for multi-objective Pareto dominance engine."""

    def test_pareto_dominance_minimization_and_maximization(self) -> None:
        """Candidate A dominates B when at least as good on all objectives and strictly better on one."""
        objectives = [
            OptimizationObjective.MINIMIZE_TTFT,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
        ]

        # Candidate A: TTFT=10ms, Throughput=100 tok/s
        cand_a = _make_candidate_eval("cand-A", ttft_ms=10.0, throughput_tok_s=100.0)
        # Candidate B: TTFT=20ms, Throughput=50 tok/s (strictly worse on both)
        cand_b = _make_candidate_eval("cand-B", ttft_ms=20.0, throughput_tok_s=50.0)

        assert ParetoEngine.dominates(cand_a, cand_b, objectives) is True
        assert ParetoEngine.dominates(cand_b, cand_a, objectives) is False

    def test_trade_off_candidates_do_not_dominate_each_other(self) -> None:
        """Candidates with trade-offs (better on one objective, worse on another) must not dominate."""
        objectives = [
            OptimizationObjective.MINIMIZE_TTFT,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
        ]

        # Candidate FastPrefill: TTFT=5ms (better), Throughput=40 tok/s (worse)
        cand_fast = _make_candidate_eval("fast-prefill", ttft_ms=5.0, throughput_tok_s=40.0)
        # Candidate HighThroughput: TTFT=30ms (worse), Throughput=150 tok/s (better)
        cand_thru = _make_candidate_eval("high-thru", ttft_ms=30.0, throughput_tok_s=150.0)

        assert ParetoEngine.dominates(cand_fast, cand_thru, objectives) is False
        assert ParetoEngine.dominates(cand_thru, cand_fast, objectives) is False

        frontier = ParetoEngine.extract_pareto_frontier([cand_fast, cand_thru], objectives)
        assert len(frontier) == 2

    def test_dominated_candidate_excluded_from_frontier(self) -> None:
        """Frontier must include all non-dominated candidates and exclude strictly dominated ones."""
        objectives = [
            OptimizationObjective.MINIMIZE_TTFT,
            OptimizationObjective.MINIMIZE_VRAM,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
        ]

        c1 = _make_candidate_eval("c1", ttft_ms=10.0, vram_mb=4000.0, throughput_tok_s=80.0)
        c2 = _make_candidate_eval("c2", ttft_ms=25.0, vram_mb=2000.0, throughput_tok_s=120.0)
        # Dominated by c1: worse TTFT (20 > 10), worse VRAM (5000 > 4000), worse Throughput (60 < 80)
        c_dominated = _make_candidate_eval(
            "c-dom", ttft_ms=20.0, vram_mb=5000.0, throughput_tok_s=60.0
        )

        frontier = ParetoEngine.extract_pareto_frontier([c1, c2, c_dominated], objectives)
        frontier_ids = [c.config.model_spec.model_id for c in frontier]

        assert "c1" in frontier_ids
        assert "c2" in frontier_ids
        assert "c-dom" not in frontier_ids
        assert len(frontier) == 2

    def test_infeasible_candidates_cannot_enter_frontier(self) -> None:
        """Infeasible candidates must be strictly excluded from the Pareto frontier."""
        objectives = [OptimizationObjective.MINIMIZE_TTFT]
        cand_infeasible = _make_candidate_eval(
            "infeasible-cand",
            ttft_ms=1.0,  # Super fast, but infeasible!
            feasibility=FeasibilityStatus.PROVEN_INFEASIBLE,
            reasons=["Exceeds VRAM capacity"],
        )
        cand_feasible = _make_candidate_eval("feasible-cand", ttft_ms=20.0)

        frontier = ParetoEngine.extract_pareto_frontier(
            [cand_infeasible, cand_feasible], objectives
        )
        assert len(frontier) == 1
        assert frontier[0].config.model_spec.model_id == "feasible-cand"

    def test_equal_objective_vectors_deterministic_handling(self) -> None:
        """Duplicate objective vectors must be handled deterministically without crashes."""
        objectives = [OptimizationObjective.MINIMIZE_TTFT]
        cand1 = _make_candidate_eval("cand-1", ttft_ms=15.0)
        cand2 = _make_candidate_eval("cand-2", ttft_ms=15.0)

        frontier = ParetoEngine.extract_pareto_frontier([cand1, cand2], objectives)
        # Both represent distinct recipes with equal objective scores; both preserved
        assert len(frontier) == 2


class TestSelectionPolicies:
    """Test suite for documented single-candidate selection policies."""

    def test_prioritize_objective_selection(self) -> None:
        """PRIORITIZE_OBJECTIVE picks candidate with optimal value for designated target."""
        c_low_ttft = _make_candidate_eval("low-ttft", ttft_ms=5.0, throughput_tok_s=50.0)
        c_high_thru = _make_candidate_eval("high-thru", ttft_ms=25.0, throughput_tok_s=200.0)
        frontier = [c_low_ttft, c_high_thru]

        objectives = [
            OptimizationObjective.MINIMIZE_TTFT,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
        ]

        # Prioritize TTFT
        best_ttft, rat_ttft = ParetoEngine.select_candidate(
            frontier=frontier,
            objectives=objectives,
            policy=SelectionPolicy.PRIORITIZE_OBJECTIVE,
            primary_objective=OptimizationObjective.MINIMIZE_TTFT,
        )
        assert best_ttft is not None
        assert best_ttft.config.model_spec.model_id == "low-ttft"
        assert "optimal value of 5.00 ms" in rat_ttft

        # Prioritize Throughput
        best_thru, rat_thru = ParetoEngine.select_candidate(
            frontier=frontier,
            objectives=objectives,
            policy=SelectionPolicy.PRIORITIZE_OBJECTIVE,
            primary_objective=OptimizationObjective.MAXIMIZE_THROUGHPUT,
        )
        assert best_thru is not None
        assert best_thru.config.model_spec.model_id == "high-thru"
        assert "optimal value of 200.00 tok/s" in rat_thru

    def test_balanced_knee_point_selection(self) -> None:
        """BALANCED_KNEE_POINT selects candidate closest to normalized utopia point."""
        c1 = _make_candidate_eval("c-extreme-ttft", ttft_ms=10.0, throughput_tok_s=10.0)
        c2 = _make_candidate_eval("c-balanced", ttft_ms=20.0, throughput_tok_s=90.0)
        c3 = _make_candidate_eval("c-extreme-thru", ttft_ms=100.0, throughput_tok_s=100.0)
        frontier = [c1, c2, c3]

        objectives = [
            OptimizationObjective.MINIMIZE_TTFT,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
        ]
        best_cand, rationale = ParetoEngine.select_candidate(
            frontier=frontier,
            objectives=objectives,
            policy=SelectionPolicy.BALANCED_KNEE_POINT,
        )
        assert best_cand is not None
        assert best_cand.config.model_spec.model_id == "c-balanced"
        assert "BALANCED_KNEE_POINT" in rationale


class TestFeasibilityAndHardConstraints:
    """Test suite for feasibility evaluations and hard SLA constraint checks."""

    def test_context_window_exceeded_proven_infeasible(self) -> None:
        """Workload sequence exceeding model context length must be PROVEN_INFEASIBLE."""
        config = CandidateConfig(
            model_spec=ModelSpec(model_id="tiny-model", context_window=1024),
            quantization=QuantizationType.FP16,
            engine_type=EngineType.MOCK,
        )
        workload = WorkloadSpec(prompt_tokens=800, output_tokens=300)  # 1100 > 1024
        constraints = OptimizationConstraints()

        status, reasons = FeasibilityChecker.evaluate(config, workload, constraints)
        assert status == FeasibilityStatus.PROVEN_INFEASIBLE
        assert any("exceeds model context length" in r for r in reasons)

    def test_engine_quantization_incompatibility_proven_infeasible(self) -> None:
        """Engine that cannot run a quantization format must be PROVEN_INFEASIBLE."""
        # llama.cpp does not run AWQ
        config = CandidateConfig(
            model_spec=ModelSpec(model_id="llama-7b"),
            quantization=QuantizationType.AWQ,
            engine_type=EngineType.LLAMACPP,
        )
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=64)
        constraints = OptimizationConstraints()

        status, reasons = FeasibilityChecker.evaluate(config, workload, constraints)
        assert status == FeasibilityStatus.PROVEN_INFEASIBLE
        assert any("does not support quantization 'awq'" in r for r in reasons)

    def test_memory_exceeding_gpu_capacity_estimated_infeasible(self) -> None:
        """Model requiring more VRAM than target GPU capacity must be ESTIMATED_INFEASIBLE."""
        config = CandidateConfig(
            model_spec=ModelSpec(model_id="llama-70b", parameter_count_billion=70.0),
            quantization=QuantizationType.FP16,  # 70B * 2 bytes = 140 GB weights
            engine_type=EngineType.MOCK,
        )
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=64)
        constraints = OptimizationConstraints()
        gpu_24gb = _make_dummy_gpu("RTX-4090", vram_gb=24.0)

        # Estimate resources for 70B model (~160 GB VRAM)
        res_est = ModelResourceEstimator.estimate_model_resources(
            model_spec=config.model_spec,
            prompt_tokens=128,
            output_tokens=64,
            weight_precision=config.quantization,
        )

        status, reasons = FeasibilityChecker.evaluate(
            config, workload, constraints, gpu_spec=gpu_24gb, resource_estimate=res_est
        )
        assert status == FeasibilityStatus.ESTIMATED_INFEASIBLE
        assert any("exceeds available device capacity" in r for r in reasons)

    def test_missing_gpu_spec_triggers_unknown_feasibility(self) -> None:
        """When hardware capacity is missing, memory feasibility must be UNKNOWN_FEASIBILITY."""
        config = CandidateConfig(
            model_spec=ModelSpec(model_id="llama-7b", parameter_count_billion=7.0),
            quantization=QuantizationType.FP16,
            engine_type=EngineType.MOCK,
        )
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=64)
        constraints = OptimizationConstraints()
        res_est = ModelResourceEstimator.estimate_model_resources(
            model_spec=config.model_spec, prompt_tokens=128, output_tokens=64
        )

        status, reasons = FeasibilityChecker.evaluate(
            config, workload, constraints, gpu_spec=None, resource_estimate=res_est
        )
        assert status == FeasibilityStatus.UNKNOWN_FEASIBILITY
        assert any("Target GPU hardware specification is missing" in r for r in reasons)

    def test_user_sla_latency_and_throughput_constraints(self) -> None:
        """Violating user max_ttft_ms or min_throughput_tok_s triggers ESTIMATED_INFEASIBLE."""
        config = CandidateConfig(
            model_spec=ModelSpec(model_id="llama-7b"),
            quantization=QuantizationType.FP16,
            engine_type=EngineType.MOCK,
        )
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=64)
        constraints = OptimizationConstraints(
            max_ttft_ms=15.0,  # SLA: TTFT <= 15 ms
            min_throughput_tok_s=100.0,  # SLA: Throughput >= 100 tok/s
        )
        # Evaluated values that violate both SLAs
        evaluated_metrics = {
            OptimizationObjective.MINIMIZE_TTFT: 25.0,  # Violates (25 > 15)
            OptimizationObjective.MAXIMIZE_THROUGHPUT: 40.0,  # Violates (40 < 100)
        }

        status, reasons = FeasibilityChecker.evaluate(
            config, workload, constraints, evaluated_metrics=evaluated_metrics
        )
        assert status == FeasibilityStatus.ESTIMATED_INFEASIBLE
        assert any("exceeds SLA maximum of 15.00 ms" in r for r in reasons)
        assert any("below SLA minimum of 100.00 tok/s" in r for r in reasons)


class TestSearchSpaceGenerationAndGuardrails:
    """Test suite for candidate enumeration, duplicates pruning, and budget safety."""

    def test_search_space_generation_and_deterministic_order(self) -> None:
        """Candidate generator enumerates Cartesian product in deterministic order."""
        space = SearchSpace(
            models=[
                ModelSpec(model_id="model-B", parameter_count_billion=7.0),
                ModelSpec(model_id="model-A", parameter_count_billion=3.0),
            ],
            quantizations=[QuantizationType.FP16, QuantizationType.INT8],
            concurrencies=[1, 4],
            engines=[EngineType.MOCK],
            tensor_parallel_sizes=[1],
        )

        candidates, rejections = SearchSpaceGenerator.generate(space)
        # Expected: 2 models * 2 quants * 2 concs * 1 engine * 1 tp = 8 candidates
        assert len(candidates) == 8
        # Deterministic sorting: model-A should precede model-B
        assert candidates[0].model_spec.model_id == "model-A"
        assert candidates[-1].model_spec.model_id == "model-B"

    def test_search_space_budget_guardrail_prevents_combinatorial_explosion(self) -> None:
        """Generating more candidates than max_candidates budget must raise ValueError."""
        space = SearchSpace(
            models=[ModelSpec(model_id=f"m-{i}") for i in range(20)],
            quantizations=[QuantizationType.FP16, QuantizationType.INT8, QuantizationType.AWQ],
            concurrencies=list(range(1, 21)),  # 20 concurrencies
            engines=[EngineType.MOCK, EngineType.VLLM],
            max_candidates=100,  # 20 * 3 * 20 * 2 = 2400 > 100
        )

        with pytest.raises(ValueError, match="exceeds the maximum safety budget"):
            SearchSpaceGenerator.generate(space)


class TestCostObjectiveValidation:
    """Test suite ensuring cost objective is strictly verified without fabricated inputs."""

    def test_cost_objective_without_cost_per_hour_raises_value_error(self) -> None:
        """Requesting MINIMIZE_COST without cost_per_gpu_hour must raise ValueError."""
        space = SearchSpace(models=[ModelSpec(model_id="llama-7b")])
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=64)

        with pytest.raises(
            ValueError, match="requires 'cost_per_gpu_hour' to be explicitly supplied"
        ):
            OptimizationRequest(
                workload=workload,
                search_space=space,
                objectives=[OptimizationObjective.MINIMIZE_COST],
                cost_per_gpu_hour=None,  # Missing cost!
            )


class TestEndToEndConfigurationOptimization:
    """End-to-end integration tests using analytical models and empirical predictor fixtures."""

    def test_end_to_end_analytical_optimization_run(self) -> None:
        """Full optimizer run with analytical Roofline resource models."""
        gpu_24gb = _make_dummy_gpu("RTX-4090", vram_gb=24.0)
        space = SearchSpace(
            models=[
                ModelSpec(model_id="llama-3b", parameter_count_billion=3.0, context_window=8192),
                ModelSpec(model_id="llama-7b", parameter_count_billion=7.0, context_window=8192),
            ],
            quantizations=[QuantizationType.FP16, QuantizationType.INT8],
            concurrencies=[1, 2],
            engines=[EngineType.MOCK],
        )
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=128)
        request = OptimizationRequest(
            workload=workload,
            search_space=space,
            hardware_spec=gpu_24gb,
            objectives=[
                OptimizationObjective.MINIMIZE_TTFT,
                OptimizationObjective.MAXIMIZE_THROUGHPUT,
                OptimizationObjective.MINIMIZE_VRAM,
            ],
            constraints=OptimizationConstraints(max_vram_mb=20000.0),
            selection_policy=SelectionPolicy.BALANCED_KNEE_POINT,
        )

        optimizer = ConfigurationOptimizer()
        result = optimizer.optimize(request)

        assert isinstance(result, OptimizationResult)
        assert result.total_candidates_evaluated == 8  # 2 * 2 * 2 * 1
        assert result.feasible_candidates_count > 0
        assert len(result.pareto_frontier) >= 1
        assert result.recommended_candidate is not None
        assert "BALANCED_KNEE_POINT" in result.selection_rationale
        assert "NOTICE: All candidate performance figures derive strictly" in result.summary

        # Check provenance
        best = result.recommended_candidate
        assert (
            best.evidence_status[OptimizationObjective.MINIMIZE_TTFT]
            == EvidenceStatus.THEORETICAL_ANALYTICAL
        )
        assert best.provenance[OptimizationObjective.MINIMIZE_TTFT] == ProvenanceType.HEURISTIC
        assert best.trade_off_summary != ""

    def test_end_to_end_with_empirical_predictor_and_measured_evidence(self) -> None:
        """Optimizer correctly differentiates MEASURED, PREDICTED, and THEORETICAL evidence."""
        # 1. Fit empirical Ridge regression predictor on prompt scaling
        train_records = [
            PredictionRecord(
                record_id=f"rec-{p}",
                suite_id="s1",
                model_id="llama-7b",
                engine_type=EngineType.MOCK,
                execution_mode=ExecutionMode.REAL,
                provenance=ProvenanceType.MEASURED,
                parameter_count_billion=7.0,
                quantization=QuantizationType.FP16,
                hardware_catalog_id="test-gpu-24gb",
                concurrency=1,
                prompt_tokens=p,
                requested_output_tokens=64,
                actual_output_tokens=64,
                planned_sequence_length=p + 64,
                total_sequence_length=p + 64,
                ttft_ms=10.0 + 0.05 * p,
                is_eligible_for_ttft=True,
            )
            for p in [32, 64, 128, 256, 512]
        ]
        ridge_ttft = RidgeRegressionPredictor(
            target=PredictionTarget.TTFT_MS,
            feature_names=["prompt_tokens"],
        ).fit(train_records)

        # 2. Historical measured observation for llama-7b FP16 c=1
        measured_obs = PredictionRecord(
            record_id="rec-measured-exact",
            suite_id="s-real",
            model_id="llama-7b",
            engine_type=EngineType.MOCK,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            parameter_count_billion=7.0,
            quantization=QuantizationType.FP16,
            hardware_catalog_id="test-gpu-24gb",
            concurrency=1,
            prompt_tokens=128,
            requested_output_tokens=64,
            actual_output_tokens=64,
            planned_sequence_length=192,
            total_sequence_length=192,
            output_tokens_per_second=92.5,  # Measured throughput
            is_eligible_for_throughput=True,
        )

        space = SearchSpace(
            models=[ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)],
            quantizations=[QuantizationType.FP16],
            concurrencies=[1],
            engines=[EngineType.MOCK],
        )
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=64)
        gpu = _make_dummy_gpu()

        request = OptimizationRequest(
            workload=workload,
            search_space=space,
            hardware_spec=gpu,
            objectives=[
                OptimizationObjective.MINIMIZE_TTFT,
                OptimizationObjective.MAXIMIZE_THROUGHPUT,
            ],
        )

        optimizer = ConfigurationOptimizer(
            predictors={PredictionTarget.TTFT_MS: ridge_ttft},
            empirical_observations=[measured_obs],
        )
        result = optimizer.optimize(request)

        assert len(result.pareto_frontier) == 1
        cand = result.pareto_frontier[0]

        # TTFT came from empirical Ridge regression predictor
        assert (
            cand.evidence_status[OptimizationObjective.MINIMIZE_TTFT]
            == EvidenceStatus.PREDICTED_EMPIRICAL
        )
        assert cand.provenance[OptimizationObjective.MINIMIZE_TTFT] == ProvenanceType.PREDICTED

        # Throughput came from direct MEASURED benchmark observation
        assert (
            cand.evidence_status[OptimizationObjective.MAXIMIZE_THROUGHPUT]
            == EvidenceStatus.MEASURED_EVALUATED
        )
        assert cand.provenance[OptimizationObjective.MAXIMIZE_THROUGHPUT] == ProvenanceType.MEASURED
        assert cand.objective_values[OptimizationObjective.MAXIMIZE_THROUGHPUT] == 92.5


class TestSession071OptimizerAudit:
    """Rigorous audit test cases for Session 07.1 addressing edge cases and requirements."""

    def test_tensor_parallelism_sharding_and_kv_head_replication(self) -> None:
        """Verify per-device memory calculation shards parameters by TP and respects KV head replication."""
        model = ModelSpec(
            model_id="gqa-model-8b",
            parameter_count_billion=8.0,
            num_attention_heads=32,
            num_kv_heads=2,  # Only 2 KV heads!
            hidden_dimension=4096,
            num_layers=32,
        )
        workload = WorkloadSpec(prompt_tokens=512, output_tokens=128)
        res_est = ModelResourceEstimator.estimate_model_resources(
            model_spec=model,
            prompt_tokens=512,
            output_tokens=128,
            batch_size=1,
            weight_precision=QuantizationType.FP16,
        )

        gpu = _make_dummy_gpu(name="NVIDIA A100", vram_gb=40.0)

        # Config with TP=4
        config_tp4 = CandidateConfig(
            model_spec=model,
            quantization=QuantizationType.FP16,
            concurrency=1,
            engine_type=EngineType.MOCK,
            tensor_parallel_size=4,
        )

        constraints = OptimizationConstraints(
            available_gpus=4,
            vram_safety_margin=1.15,
        )

        status, reasons = FeasibilityChecker.evaluate(
            config=config_tp4,
            workload=workload,
            constraints=constraints,
            gpu_spec=gpu,
            resource_estimate=res_est,
        )

        assert status == FeasibilityStatus.FEASIBLE

        # Now test that if TP=4 but model has only 2 KV heads, effective_kv_shards is 2
        tp_factor = 4
        param_per_rank = res_est.parameter_storage_mb / tp_factor
        assert res_est.kv_cache_total_mb is not None
        kv_per_rank = res_est.kv_cache_total_mb / 2  # min(4, 2) = 2
        expected_per_rank_vram = ((param_per_rank + kv_per_rank) * 1.15) + 512.0

        # Run through optimizer to check MINIMIZE_VRAM evaluation
        opt = ConfigurationOptimizer()
        req = OptimizationRequest(
            workload=workload,
            search_space=SearchSpace(
                models=[model],
                quantizations=[QuantizationType.FP16],
                concurrencies=[1],
                engines=[EngineType.MOCK],
                tensor_parallel_sizes=[4],
            ),
            constraints=constraints,
            hardware_spec=gpu,
            objectives=[OptimizationObjective.MINIMIZE_VRAM],
        )
        result = opt.optimize(req)
        assert len(result.pareto_frontier) == 1
        cand = result.pareto_frontier[0]
        assert cand.objective_values[OptimizationObjective.MINIMIZE_VRAM] == round(
            expected_per_rank_vram, 2
        )

    def test_available_gpus_and_unknown_topology_handling(self) -> None:
        """Verify that multi-GPU without available_gpus returns UNKNOWN_FEASIBILITY, and exceeding available_gpus is PROVEN_INFEASIBLE."""
        model = ModelSpec(model_id="test-7b", parameter_count_billion=7.0, num_attention_heads=32)
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=32)
        gpu = _make_dummy_gpu()

        config_tp2 = CandidateConfig(
            model_spec=model,
            quantization=QuantizationType.FP16,
            concurrency=1,
            engine_type=EngineType.MOCK,
            tensor_parallel_size=2,
        )

        # 1. available_gpus is None -> UNKNOWN_FEASIBILITY
        constraints_unspecified = OptimizationConstraints(available_gpus=None)
        status, reasons = FeasibilityChecker.evaluate(
            config=config_tp2,
            workload=workload,
            constraints=constraints_unspecified,
            gpu_spec=gpu,
        )
        assert status == FeasibilityStatus.UNKNOWN_FEASIBILITY
        assert any("available_gpus" in r for r in reasons)

        # 2. available_gpus is 1, but TP is 2 -> PROVEN_INFEASIBLE
        constraints_too_few = OptimizationConstraints(available_gpus=1)
        status, reasons = FeasibilityChecker.evaluate(
            config=config_tp2,
            workload=workload,
            constraints=constraints_too_few,
            gpu_spec=gpu,
        )
        assert status == FeasibilityStatus.PROVEN_INFEASIBLE
        assert any("exceeds available GPUs" in r for r in reasons)

    def test_unsupported_engine_tp_and_head_divisibility(self) -> None:
        """Verify that Ollama/llama.cpp with TP > 1 and non-divisible heads are PROVEN_INFEASIBLE."""
        model = ModelSpec(model_id="test-7b", parameter_count_billion=7.0, num_attention_heads=32)
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=32)

        # Ollama with TP=2 -> PROVEN_INFEASIBLE
        config_ollama_tp2 = CandidateConfig(
            model_spec=model,
            quantization=QuantizationType.INT4,
            concurrency=1,
            engine_type=EngineType.OLLAMA,
            tensor_parallel_size=2,
        )
        status, reasons = FeasibilityChecker.evaluate(
            config=config_ollama_tp2,
            workload=workload,
            constraints=OptimizationConstraints(available_gpus=4),
        )
        assert status == FeasibilityStatus.PROVEN_INFEASIBLE
        assert any("does not support tensor parallelism" in r for r in reasons)

        # llama.cpp with TP=2 -> PROVEN_INFEASIBLE
        config_llamacpp_tp2 = CandidateConfig(
            model_spec=model,
            quantization=QuantizationType.GGUF_Q4_K_M,
            concurrency=1,
            engine_type=EngineType.LLAMACPP,
            tensor_parallel_size=2,
        )
        status, reasons = FeasibilityChecker.evaluate(
            config=config_llamacpp_tp2,
            workload=workload,
            constraints=OptimizationConstraints(available_gpus=4),
        )
        assert status == FeasibilityStatus.PROVEN_INFEASIBLE
        assert any("does not support tensor parallelism" in r for r in reasons)

        # num_attention_heads (32) not divisible by TP (3) -> PROVEN_INFEASIBLE
        config_indivisible = CandidateConfig(
            model_spec=model,
            quantization=QuantizationType.FP16,
            concurrency=1,
            engine_type=EngineType.MOCK,
            tensor_parallel_size=3,
        )
        status, reasons = FeasibilityChecker.evaluate(
            config=config_indivisible,
            workload=workload,
            constraints=OptimizationConstraints(available_gpus=4),
        )
        assert status == FeasibilityStatus.PROVEN_INFEASIBLE
        assert any("does not evenly divide model attention heads" in r for r in reasons)

    def test_sla_missing_metric_produces_unknown_feasibility(self) -> None:
        """Missing SLA metric must produce UNKNOWN_FEASIBILITY, never FEASIBLE."""
        model = ModelSpec(model_id="test-unknown-sla")
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=32)

        config = CandidateConfig(
            model_spec=model,
            quantization=QuantizationType.FP16,
            concurrency=1,
            engine_type=EngineType.MOCK,
        )

        status, reasons = FeasibilityChecker.evaluate(
            config=config,
            workload=workload,
            constraints=OptimizationConstraints(max_ttft_ms=50.0),
            evaluated_metrics={},
        )
        assert status == FeasibilityStatus.UNKNOWN_FEASIBILITY
        assert any("max_ttft_ms" in r and "cannot be verified" in r for r in reasons)

    def test_sla_insufficient_evidence_produces_unknown_feasibility(self) -> None:
        """When an SLA metric has INSUFFICIENT_EVIDENCE (OOD), feasibility is UNKNOWN."""
        model = ModelSpec(model_id="test-7b", parameter_count_billion=7.0)
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=32)
        config = CandidateConfig(
            model_spec=model,
            quantization=QuantizationType.FP16,
            concurrency=1,
            engine_type=EngineType.MOCK,
        )

        status, reasons = FeasibilityChecker.evaluate(
            config=config,
            workload=workload,
            constraints=OptimizationConstraints(max_ttft_ms=50.0),
            evaluated_metrics={OptimizationObjective.MINIMIZE_TTFT: 45.0},
            evidence_status={
                OptimizationObjective.MINIMIZE_TTFT: EvidenceStatus.INSUFFICIENT_EVIDENCE
            },
        )
        assert status == FeasibilityStatus.UNKNOWN_FEASIBILITY
        assert any("insufficient evidence" in r for r in reasons)

    def test_sla_measured_vs_predicted_violations(self) -> None:
        """Measured SLA violations are PROVEN_INFEASIBLE, predicted/analytical are ESTIMATED_INFEASIBLE."""
        model = ModelSpec(model_id="test-7b", parameter_count_billion=7.0)
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=32)
        config = CandidateConfig(
            model_spec=model,
            quantization=QuantizationType.FP16,
            concurrency=1,
            engine_type=EngineType.MOCK,
        )

        # Measured violation -> PROVEN_INFEASIBLE
        status_meas, reasons_meas = FeasibilityChecker.evaluate(
            config=config,
            workload=workload,
            constraints=OptimizationConstraints(max_ttft_ms=50.0),
            evaluated_metrics={OptimizationObjective.MINIMIZE_TTFT: 75.0},
            evidence_status={
                OptimizationObjective.MINIMIZE_TTFT: EvidenceStatus.MEASURED_EVALUATED
            },
        )
        assert status_meas == FeasibilityStatus.PROVEN_INFEASIBLE
        assert any("Measured TTFT" in r for r in reasons_meas)

        # Predicted violation -> ESTIMATED_INFEASIBLE
        status_pred, reasons_pred = FeasibilityChecker.evaluate(
            config=config,
            workload=workload,
            constraints=OptimizationConstraints(max_ttft_ms=50.0),
            evaluated_metrics={OptimizationObjective.MINIMIZE_TTFT: 75.0},
            evidence_status={
                OptimizationObjective.MINIMIZE_TTFT: EvidenceStatus.PREDICTED_EMPIRICAL
            },
        )
        assert status_pred == FeasibilityStatus.ESTIMATED_INFEASIBLE
        assert any("Estimated/predicted TTFT" in r for r in reasons_pred)

    def test_cost_calculation_without_explicit_throughput_objective(self) -> None:
        """Cost is correctly computed even when MAXIMIZE_THROUGHPUT is not in request.objectives."""
        model = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)
        workload = WorkloadSpec(prompt_tokens=128, output_tokens=64)
        gpu = _make_dummy_gpu()

        measured_obs = PredictionRecord(
            record_id="rec-cost",
            suite_id="s1",
            model_id="llama-7b",
            engine_type=EngineType.MOCK,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            parameter_count_billion=7.0,
            quantization=QuantizationType.FP16,
            hardware_catalog_id="test-gpu-24gb",
            concurrency=1,
            prompt_tokens=128,
            requested_output_tokens=64,
            actual_output_tokens=64,
            planned_sequence_length=192,
            total_sequence_length=192,
            output_tokens_per_second=100.0,
            is_eligible_for_throughput=True,
        )

        opt = ConfigurationOptimizer(empirical_observations=[measured_obs])
        # Only request MINIMIZE_COST
        req = OptimizationRequest(
            workload=workload,
            search_space=SearchSpace(
                models=[model],
                quantizations=[QuantizationType.FP16],
                concurrencies=[1],
                engines=[EngineType.MOCK],
            ),
            hardware_spec=gpu,
            cost_per_gpu_hour=2.50,
            objectives=[OptimizationObjective.MINIMIZE_COST],
        )
        result = opt.optimize(req)
        assert len(result.pareto_frontier) == 1
        cand = result.pareto_frontier[0]

        # 100 tok/s -> 360,000 tok/hr
        # $2.50 / 360,000 tokens * 1,000,000 = $6.9444 / 1M tokens
        expected_cost = round((2.50 / (100.0 * 3600.0)) * 1_000_000.0, 4)
        assert cand.objective_values[OptimizationObjective.MINIMIZE_COST] == expected_cost

    def test_exact_benchmark_matching_prevents_hardware_and_workload_leakage(self) -> None:
        """Benchmark matching must not match mismatched hardware, prompts, or tensor parallelism."""
        obs_a100 = PredictionRecord(
            record_id="rec-a100",
            suite_id="s1",
            model_id="llama-7b",
            engine_type=EngineType.MOCK,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            parameter_count_billion=7.0,
            quantization=QuantizationType.FP16,
            hardware_catalog_id="a100-80gb",  # A100!
            concurrency=1,
            prompt_tokens=128,
            requested_output_tokens=64,
            actual_output_tokens=64,
            planned_sequence_length=192,
            total_sequence_length=192,
            output_tokens_per_second=200.0,
            is_eligible_for_throughput=True,
        )

        opt = ConfigurationOptimizer(empirical_observations=[obs_a100])
        # Target hardware is test-gpu-24gb (NOT A100)
        req = OptimizationRequest(
            workload=WorkloadSpec(prompt_tokens=128, output_tokens=64),
            search_space=SearchSpace(
                models=[ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)],
                quantizations=[QuantizationType.FP16],
                concurrencies=[1],
                engines=[EngineType.MOCK],
            ),
            hardware_spec=_make_dummy_gpu(),  # test-gpu-24gb
            objectives=[OptimizationObjective.MAXIMIZE_THROUGHPUT],
        )
        result = opt.optimize(req)
        cand = result.pareto_frontier[0]
        # Must NOT use the 200.0 tok/s from A100!
        assert (
            cand.evidence_status[OptimizationObjective.MAXIMIZE_THROUGHPUT]
            != EvidenceStatus.MEASURED_EVALUATED
        )
        assert cand.objective_values[OptimizationObjective.MAXIMIZE_THROUGHPUT] != 200.0

    def test_mixed_and_incomparable_evidence_detection(self) -> None:
        """Expose mixed evidence on candidates and incomparable evidence on the frontier."""
        obs_measured = PredictionRecord(
            record_id="rec-meas",
            suite_id="s1",
            model_id="llama-7b",
            engine_type=EngineType.MOCK,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            parameter_count_billion=7.0,
            quantization=QuantizationType.FP16,
            hardware_catalog_id="test-gpu-24gb",
            concurrency=1,
            prompt_tokens=128,
            requested_output_tokens=64,
            actual_output_tokens=64,
            planned_sequence_length=192,
            total_sequence_length=192,
            output_tokens_per_second=95.0,  # Throughput is measured, but TTFT is not!
            is_eligible_for_throughput=True,
        )

        opt = ConfigurationOptimizer(empirical_observations=[obs_measured])
        req = OptimizationRequest(
            workload=WorkloadSpec(prompt_tokens=128, output_tokens=64),
            search_space=SearchSpace(
                models=[ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)],
                quantizations=[QuantizationType.FP16],
                concurrencies=[1],
                engines=[EngineType.MOCK],
            ),
            hardware_spec=_make_dummy_gpu(),
            objectives=[
                OptimizationObjective.MINIMIZE_TTFT,  # Theoretical
                OptimizationObjective.MAXIMIZE_THROUGHPUT,  # Measured
            ],
        )
        result = opt.optimize(req)
        assert len(result.pareto_frontier) == 1
        cand = result.pareto_frontier[0]

        # Candidate has mixed evidence: TTFT is theoretical, throughput is measured
        assert cand.has_mixed_evidence is True
        assert cand.comparability_warning is not None
        assert "mixed evidence tiers" in cand.comparability_warning

        # Result detects incomparable evidence and sets caveats
        assert result.has_incomparable_evidence is True
        assert len(result.evidence_caveats) > 0
        assert "NOTICE" in result.summary
