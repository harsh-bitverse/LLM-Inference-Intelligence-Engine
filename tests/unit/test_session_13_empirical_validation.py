"""Unit tests for Session 13 controlled benchmark expansion and empirical predictor validation.

Verifies:
1. Workload expansion suite calibration, deterministic prompts, and manifest metadata.
2. Leakage-resistant train/test splitting (grouping by configuration/workload_config).
3. Target eligibility enforcement for chunked streaming without per-token timestamps.
4. Predictor evaluation (ConstantMedian vs RidgeRegression) under held-out configuration groups.
5. ConfigurationOptimizer evidence classification (MEASURED_EVALUATED vs PREDICTED_EMPIRICAL).
6. SLA safety for unmeasured latency metrics (preventing false SLA compliance via UNKNOWN_FEASIBILITY).
7. Incompatible engine-quantization early rejection.
"""


from llm_ii_engine.benchmark.dataset_export import (
    EmpiricalObservationRecord,
)
from llm_ii_engine.benchmark.expansion import (
    EXPANSION_SUITE,
    PROMPT_TEXT_32,
    PROMPT_TEXT_128,
    PROMPT_TEXT_384,
    PROMPT_TEXT_768,
    create_expansion_manifest,
)
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    FailurePolicy,
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
    EvidenceStatus,
    FeasibilityStatus,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    SearchSpace,
)
from llm_ii_engine.prediction.baselines import (
    ConstantMedianPredictor,
    RidgeRegressionPredictor,
)
from llm_ii_engine.prediction.dataset import DatasetBuilder
from llm_ii_engine.prediction.evaluation import DataSplitter, PredictionEvaluator
from llm_ii_engine.prediction.models import (
    EvaluationResult,
    PredictionRecord,
    PredictionTarget,
)

# ============================================================================
# Helpers & Fixtures
# ============================================================================

def _make_dummy_gpu(catalog_id: str = "catalog-workstation-cpu", name: str = "Host Hardware", vram_gb: float = 32.0) -> GPUSpecification:
    """Helper to construct a valid GPUSpecification fixture."""
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


_MODEL_SPEC = ModelSpec(
    model_id="llama3.2:1b",
    parameter_count_billion=1.235,
    quantization=QuantizationType.GGUF_Q8_0,
)


def _create_mock_expansion_records() -> list[EmpiricalObservationRecord]:
    """Create synthetic empirical observations matching the 7 Session 13 configurations."""
    configs = [
        (46, 32, 21.75, 1474.0),
        (46, 64, 22.96, 2788.0),
        (119, 64, 22.88, 2797.0),
        (119, 128, 22.35, 5727.0),
        (300, 64, 22.61, 2831.0),
        (677, 64, 22.35, 2864.0),
        (677, 128, 21.96, 5828.0),
    ]
    records: list[EmpiricalObservationRecord] = []
    req_idx = 1
    for prompt_tok, out_tok, thru, dur in configs:
        # Warmup (repetition 0)
        records.append(
            EmpiricalObservationRecord(
                record_id=f"rec-warm-{req_idx:04d}",
                experiment_id="exp-test-s13",
                suite_id="suite-s13",
                request_id=f"req-warm-{req_idx:04d}",
                repetition_index=0,
                is_warmup=True,
                model_id="llama3.2:1b",
                runtime_name="ollama",
                adapter_version="1.0.0",
                device_name="CPU",
                prompt_tokens=prompt_tok,
                prompt_token_count_source="server_reported",
                requested_output_tokens=out_tok,
                actual_output_tokens=out_tok,
                token_count_source="server_reported",
                duration_ms=dur * 1.5,
                ttfc_ms=150.0,
                chunk_count=out_tok,
                chunk_latency_p50_ms=45.0,
                output_tokens_per_second=thru * 0.7,
                execution_mode="REAL",
                clock_source="MONOTONIC_REAL",
                provenance="MEASURED",
                success=True,
            )
        )
        req_idx += 1
        # 3 measured repetitions
        for rep in range(1, 4):
            records.append(
                EmpiricalObservationRecord(
                    record_id=f"rec-meas-{req_idx:04d}",
                    experiment_id="exp-test-s13",
                    suite_id="suite-s13",
                    request_id=f"req-meas-{req_idx:04d}",
                    repetition_index=rep,
                    is_warmup=False,
                    model_id="llama3.2:1b",
                    runtime_name="ollama",
                    adapter_version="1.0.0",
                    device_name="CPU",
                    prompt_tokens=prompt_tok,
                    prompt_token_count_source="server_reported",
                    requested_output_tokens=out_tok,
                    actual_output_tokens=out_tok,
                    token_count_source="server_reported",
                    duration_ms=dur + (rep * 5.0),
                    ttfc_ms=80.0 + (rep * 2.0),
                    chunk_count=out_tok,
                chunk_latency_p50_ms=45.0,
                    output_tokens_per_second=thru - (rep * 0.1),
                    execution_mode="REAL",
                    clock_source="MONOTONIC_REAL",
                    provenance="MEASURED",
                    success=True,
                )
            )
            req_idx += 1
    return records


# ============================================================================
# 1. Workload Expansion Suite & Calibration Tests
# ============================================================================

class TestWorkloadExpansionSuite:
    """Tests for the 7 controlled expansion workloads and manifest generation."""

    def test_expansion_suite_composition(self) -> None:
        """Suite must contain exactly 7 systematically varied workload configurations."""
        assert len(EXPANSION_SUITE) == 7

        prompt_targets = [w.prompt_tokens for w in EXPANSION_SUITE]
        output_targets = [w.output_tokens for w in EXPANSION_SUITE]

        assert set(prompt_targets) == {32, 128, 384, 768}
        assert set(output_targets) == {32, 64, 128}

        # Deterministic seeds must be strictly unique
        seeds = [w.seed for w in EXPANSION_SUITE]
        assert len(set(seeds)) == 7

    def test_prompt_text_lengths_and_determinism(self) -> None:
        """Prompt texts must be non-empty, non-sensitive, and monotonically increasing in length."""
        assert len(PROMPT_TEXT_32) < len(PROMPT_TEXT_128)
        assert len(PROMPT_TEXT_128) < len(PROMPT_TEXT_384)
        assert len(PROMPT_TEXT_384) < len(PROMPT_TEXT_768)

        # Prompts are technical discussions of cache locality, intensity, and parallelism
        assert "cache locality" in PROMPT_TEXT_32.lower()
        assert "multi-level hierarchical caches" in PROMPT_TEXT_128.lower()
        assert "operational intensity" in PROMPT_TEXT_384.lower()
        assert "tensor parallelism" in PROMPT_TEXT_768.lower()

    def test_expansion_manifest_contract(self) -> None:
        """Manifest generation must set explicit scientific execution and provenance parameters."""
        manifest = create_expansion_manifest(
            hardware_report_id="hw-test-123",
            cpu_model="Intel Core i7-14700",
            host_ram_gb=32.0,
        )

        assert manifest.experiment_id == "exp-expansion-llama3-2-1b-s13"
        assert manifest.runtime_name == "ollama"
        assert manifest.runtime_version == "0.40.11"
        assert manifest.model_id == "llama3.2:1b"
        assert manifest.execution_mode == ExecutionMode.REAL
        assert manifest.clock_source == ClockSource.MONOTONIC_REAL
        assert manifest.metric_provenance == ProvenanceType.MEASURED
        assert manifest.warmup_repetitions == 1
        assert manifest.measured_repetitions == 3
        assert manifest.concurrency == 1
        assert manifest.sampling_temperature == 0.0
        assert manifest.sampling_top_p == 1.0
        assert manifest.failure_policy == FailurePolicy.CONTINUE_ON_ERROR
        assert len(manifest.workload_suite) == 7


# ============================================================================
# 2. Leakage-Resistant Train/Test Splitting Tests
# ============================================================================

class TestDataSplitterGroupedHoldout:
    """Tests verifying zero intra-configuration data leakage in DataSplitter."""

    def test_grouped_split_by_configuration_string(self) -> None:
        """Grouped split on 'configuration' must keep all repetitions in either train or test."""
        records = _create_mock_expansion_records()
        measured_recs = [r for r in records if not r.is_warmup]
        dataset = DatasetBuilder.from_empirical_observations(measured_recs, model_spec=_MODEL_SPEC)

        # 7 distinct configs, 3 reps each = 21 records
        assert len(dataset.records) == 21

        train_records, test_records = DataSplitter.grouped_split(
            dataset.records,
            group_key="configuration",
            test_ratio=0.3,
            seed=42,
        )

        train_configs = {f"{r.prompt_tokens}x{r.requested_output_tokens or r.actual_output_tokens}" for r in train_records}
        test_configs = {f"{r.prompt_tokens}x{r.requested_output_tokens or r.actual_output_tokens}" for r in test_records}

        # Zero intersection between train configurations and test configurations
        assert train_configs.isdisjoint(test_configs)
        assert len(train_records) + len(test_records) == 21

    def test_grouped_split_with_callable_extractor(self) -> None:
        """Grouped split must support arbitrary callable group extractors."""
        records = _create_mock_expansion_records()
        measured_recs = [r for r in records if not r.is_warmup]
        dataset = DatasetBuilder.from_empirical_observations(measured_recs, model_spec=_MODEL_SPEC)

        # Group by output token target (32, 64, 128)
        train_records, test_records = DataSplitter.grouped_split(
            dataset.records,
            group_key=lambda r: str(r.requested_output_tokens),
            test_ratio=0.3,
            seed=42,
        )

        train_outputs = {r.requested_output_tokens for r in train_records}
        test_outputs = {r.requested_output_tokens for r in test_records}

        assert train_outputs.isdisjoint(test_outputs)
        assert len(train_records) + len(test_records) == 21


# ============================================================================
# 3. Measurement Semantics & Target Eligibility Tests
# ============================================================================

class TestMeasurementTargetEligibility:
    """Tests verifying target eligibility semantics on chunked streaming records."""

    def test_chunked_streaming_marks_ttft_and_tpot_ineligible(self) -> None:
        """When individually timed tokens are not observed, TTFT and TPOT must be ineligible."""
        records = _create_mock_expansion_records()
        measured_recs = [r for r in records if not r.is_warmup]
        dataset = DatasetBuilder.from_empirical_observations(measured_recs, model_spec=_MODEL_SPEC)

        ttft_eligible = [r for r in dataset.records if r.is_eligible_for_ttft]
        tpot_eligible = [r for r in dataset.records if r.is_eligible_for_tpot]
        thru_eligible = [r for r in dataset.records if r.is_eligible_for_throughput]
        chunk_eligible = [r for r in dataset.records if r.is_eligible_for_chunk_latency]

        assert len(ttft_eligible) == 0
        assert len(tpot_eligible) == 0
        assert len(thru_eligible) == 21
        assert len(chunk_eligible) == 21

        # Check ineligibility reasons
        for r in dataset.records:
            reasons_str = " ".join(r.ineligibility_reasons.get("ttft_ms", []))
            assert "missing" in reasons_str.lower() or "unmeasured" in reasons_str.lower() or "chunked" in reasons_str.lower()


# ============================================================================
# 4. Predictor Evaluation on Held-Out Configuration Groups
# ============================================================================

class TestPredictorEvaluationGroupedHoldout:
    """Tests comparing ConstantMedianPredictor vs RidgeRegressionPredictor on configuration holdouts."""

    def test_constant_median_and_ridge_on_held_out_configs(self) -> None:
        """Predictors must train on subset of configs and evaluate without data leakage on held-out configs."""
        records = _create_mock_expansion_records()
        measured_recs = [r for r in records if not r.is_warmup]
        dataset = DatasetBuilder.from_empirical_observations(measured_recs, model_spec=_MODEL_SPEC)

        train_records, test_records = DataSplitter.grouped_split(
            dataset.records,
            group_key="configuration",
            test_ratio=0.3,
            seed=42,
        )

        # Fit ConstantMedianPredictor
        median_predictor = ConstantMedianPredictor(target=PredictionTarget.OUTPUT_THROUGHPUT_TOK_S)
        median_predictor.fit(train_records)

        # Fit RidgeRegressionPredictor
        ridge_predictor = RidgeRegressionPredictor(
            target=PredictionTarget.OUTPUT_THROUGHPUT_TOK_S,
            feature_names=["prompt_tokens", "requested_output_tokens"],
        )
        ridge_predictor.fit(train_records)

        # Evaluate both
        eval_median = PredictionEvaluator.evaluate(
            predictor=median_predictor,
            test_records=test_records,
            split_strategy="LEAKAGE_RESISTANT_CONFIG_GROUP",
        )

        eval_ridge = PredictionEvaluator.evaluate(
            predictor=ridge_predictor,
            test_records=test_records,
            baseline_predictor=median_predictor,
            split_strategy="LEAKAGE_RESISTANT_CONFIG_GROUP",
        )

        assert isinstance(eval_median, EvaluationResult)
        assert isinstance(eval_ridge, EvaluationResult)

        # Metrics must be finite non-negative numbers
        assert eval_median.mae >= 0.0
        assert eval_median.rmse >= 0.0
        assert eval_median.mape_percent is not None and eval_median.mape_percent >= 0.0

        assert eval_ridge.mae >= 0.0
        assert eval_ridge.rmse >= 0.0
        assert eval_ridge.mape_percent is not None and eval_ridge.mape_percent >= 0.0

        # Small sample size warning (< 10 test samples)
        assert not eval_median.is_statistically_sufficient
        assert not eval_ridge.is_statistically_sufficient


# ============================================================================
# 5. ConfigurationOptimizer Empirical Validation & SLA Safety Tests
# ============================================================================

class TestConfigurationOptimizerEmpiricalValidation:
    """Tests verifying ConfigurationOptimizer evidence status, SLA guardrails, and quantization compatibility."""

    def test_exact_empirical_match_classified_as_measured(self) -> None:
        """Historical empirical observation with matching hardware and config yields MEASURED_EVALUATED."""
        obs = PredictionRecord(
            record_id="rec-llama-1b-meas",
            suite_id="s-s13",
            model_id="llama3.2:1b",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            parameter_count_billion=1.235,
            quantization=QuantizationType.GGUF_Q8_0,
            hardware_catalog_id="catalog-workstation-cpu",
            concurrency=1,
            prompt_tokens=119,
            requested_output_tokens=64,
            actual_output_tokens=64,
            planned_sequence_length=183,
            total_sequence_length=183,
            output_tokens_per_second=22.88,
            is_eligible_for_throughput=True,
        )

        space = SearchSpace(
            models=[ModelSpec(model_id="llama3.2:1b", parameter_count_billion=1.235)],
            quantizations=[QuantizationType.GGUF_Q8_0],
            concurrencies=[1],
            engines=[EngineType.OLLAMA],
        )
        workload = WorkloadSpec(prompt_tokens=119, output_tokens=64)
        gpu = _make_dummy_gpu(catalog_id="catalog-workstation-cpu")

        request = OptimizationRequest(
            workload=workload,
            search_space=space,
            hardware_spec=gpu,
            objectives=[OptimizationObjective.MAXIMIZE_THROUGHPUT],
        )

        optimizer = ConfigurationOptimizer(empirical_observations=[obs])
        result = optimizer.optimize(request)

        assert len(result.pareto_frontier) == 1
        cand = result.pareto_frontier[0]

        assert cand.evidence_status[OptimizationObjective.MAXIMIZE_THROUGHPUT] == EvidenceStatus.MEASURED_EVALUATED
        assert cand.provenance[OptimizationObjective.MAXIMIZE_THROUGHPUT] == ProvenanceType.MEASURED
        assert cand.objective_values[OptimizationObjective.MAXIMIZE_THROUGHPUT] == 22.88

    def test_unmeasured_ttft_sla_triggers_unknown_feasibility(self) -> None:
        """When TTFT is unobserved/missing, SLA constraint on TTFT must NOT assert false compliance."""
        config = CandidateConfig(
            model_spec=ModelSpec(model_id="llama3.2:1b", parameter_count_billion=1.235),
            quantization=QuantizationType.GGUF_Q8_0,
            engine_type=EngineType.OLLAMA,
        )
        workload = WorkloadSpec(prompt_tokens=119, output_tokens=64)
        # SLA: TTFT <= 50 ms
        constraints = OptimizationConstraints(max_ttft_ms=50.0)

        # Evaluated metrics has throughput, but TTFT is completely unmeasured/missing
        evaluated_metrics = {
            OptimizationObjective.MAXIMIZE_THROUGHPUT: 22.88,
        }

        status, reasons = FeasibilityChecker.evaluate(
            config,
            workload,
            constraints,
            evaluated_metrics=evaluated_metrics,
        )

        # Must report UNKNOWN_FEASIBILITY, never FEASIBLE!
        assert status == FeasibilityStatus.UNKNOWN_FEASIBILITY
        assert any("cannot be verified" in r.lower() for r in reasons)

    def test_incompatible_engine_quantization_early_rejection(self) -> None:
        """Ollama with AWQ quantization must be PROVEN_INFEASIBLE early."""
        config = CandidateConfig(
            model_spec=ModelSpec(model_id="llama3.2:1b", parameter_count_billion=1.235),
            quantization=QuantizationType.AWQ,
            engine_type=EngineType.OLLAMA,
        )
        workload = WorkloadSpec(prompt_tokens=119, output_tokens=64)
        constraints = OptimizationConstraints()

        status, reasons = FeasibilityChecker.evaluate(config, workload, constraints)
        assert status == FeasibilityStatus.PROVEN_INFEASIBLE
        assert any("Engine 'ollama' does not support quantization 'awq'" in r for r in reasons)
