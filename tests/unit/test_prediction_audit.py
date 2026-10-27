"""Focused correctness audit tests for the empirical prediction pipeline (Session 06.1).

Validates:
1. Feature availability, pre-execution feature selection, and output token leakage prevention.
2. Data splitting determinism, grouped intra-suite leakage prevention, and holdout exclusion.
3. Regression mathematics, Gaussian elimination, non-finite handling, and ill-conditioning detection.
4. Evaluation metric correctness, near-zero MAPE handling, and EvaluationResult field integrity.
5. Analytical baseline provenance and strict rejection of missing inputs without silent fabrication.
"""

import math

import pytest

from llm_ii_engine.benchmark.models import (
    BenchmarkSuiteResult,
    RequestBenchmarkResult,
)
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    ProvenanceType,
)
from llm_ii_engine.core.schemas import ModelSpec
from llm_ii_engine.prediction.baselines import (
    AnalyticalBaselinePredictor,
    RidgeRegressionPredictor,
    _solve_linear_system,
)
from llm_ii_engine.prediction.dataset import DatasetBuilder
from llm_ii_engine.prediction.evaluation import DataSplitter, PredictionEvaluator
from llm_ii_engine.prediction.models import (
    EvaluationResult,
    PredictionRecord,
    PredictionTarget,
)


def _make_audit_record(
    suite_id: str = "suite-1",
    request_id: str = "req-1",
    model_id: str = "llama-7b",
    prompt_tokens: int = 128,
    requested_output_tokens: int = 128,
    actual_output_tokens: int = 64,
    parameter_count_billion: float = 7.0,
    concurrency: int = 1,
    bytes_per_parameter: float = 2.0,
    hardware_catalog_id: str = "nvidia-h100-sxm5-80gb",
    execution_mode: ExecutionMode = ExecutionMode.REAL,
    target_val: float = 25.0,
) -> PredictionRecord:
    """Helper to build an authoritative PredictionRecord with full provenance."""
    return PredictionRecord(
        record_id=f"rec-{suite_id}-{request_id}",
        suite_id=suite_id,
        request_id=request_id,
        model_id=model_id,
        engine_type=EngineType.MOCK,
        execution_mode=execution_mode,
        provenance=ProvenanceType.MEASURED,
        parameter_count_billion=parameter_count_billion,
        bytes_per_parameter=bytes_per_parameter,
        hardware_catalog_id=hardware_catalog_id,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        requested_output_tokens=requested_output_tokens,
        actual_output_tokens=actual_output_tokens,
        output_tokens=actual_output_tokens,
        planned_sequence_length=prompt_tokens + requested_output_tokens,
        total_sequence_length=prompt_tokens + actual_output_tokens,
        ttft_ms=target_val,
        tpot_ms=target_val,
        output_tokens_per_second=target_val,
        peak_vram_mb=target_val,
        is_eligible_for_ttft=True,
        is_eligible_for_tpot=True,
        is_eligible_for_throughput=True,
        is_eligible_for_vram=True,
    )


class TestFeatureAvailabilityAndLeakageAudit:
    """Audit tests ensuring pre-execution features do not leak post-execution observations."""

    def test_record_distinguishes_requested_from_actual_output_tokens(self) -> None:
        """PredictionRecord must cleanly separate requested limit from actual generated tokens."""
        record = _make_audit_record(
            prompt_tokens=100,
            requested_output_tokens=256,
            actual_output_tokens=15,  # Early stop on EOS
        )
        assert record.requested_output_tokens == 256
        assert record.actual_output_tokens == 15
        assert record.output_tokens == 15  # Backward-compatible alias
        assert record.planned_sequence_length == 356
        assert record.total_sequence_length == 115

    def test_default_features_by_target_strictly_pre_execution(self) -> None:
        """Pre-execution predictor default features must not contain actual generated tokens."""
        for target, feats in RidgeRegressionPredictor.DEFAULT_FEATURES_BY_TARGET.items():
            # Actual output tokens and total sequence length (derived from actual tokens) must never be pre-execution features
            assert "actual_output_tokens" not in feats, (
                f"Leakage detected: {target} uses actual_output_tokens"
            )
            assert "output_tokens" not in feats, (
                f"Leakage detected: {target} uses output_tokens (actual count)"
            )
            assert "total_sequence_length" not in feats, (
                f"Leakage detected: {target} uses total_sequence_length (actual sequence)"
            )

    def test_dataset_builder_extracts_requested_and_actual_tokens(self) -> None:
        """DatasetBuilder.create_record maps requested_output_tokens and actual output_tokens."""
        req = RequestBenchmarkResult(
            request_id="req-leakage-audit",
            success=True,
            prompt_tokens=128,
            requested_output_tokens=256,
            output_tokens=42,  # Stopped early
            ttft_ms=18.5,
            tpot_ms=12.0,
            duration_ms=500.0,
            output_tokens_per_second=84.0,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
        )
        suite = BenchmarkSuiteResult(
            suite_id="suite-audit-01",
            status="COMPLETED",
            model_id="llama-7b",
            engine_type=EngineType.MOCK,
            concurrency=1,
            total_requests=1,
            warmup_requests=0,
            successful_requests=1,
            failed_requests=0,
            wall_time_seconds=0.5,
            execution_mode=ExecutionMode.REAL,
            clock_source=ClockSource.MONOTONIC_REAL,
            provenance=ProvenanceType.MEASURED,
            request_results=[req],
        )
        spec = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)

        record = DatasetBuilder.create_record(req, suite, spec)
        assert record.requested_output_tokens == 256
        assert record.actual_output_tokens == 42
        assert record.planned_sequence_length == 128 + 256
        assert record.total_sequence_length == 128 + 42


class TestDataSplittingAdversarialAudit:
    """Audit tests verifying absence of data leakage across partitions."""

    def test_grouped_split_adversarial_intra_suite_leakage_prevented(self) -> None:
        """Repeated requests sharing the same suite_id must stay 100% on one side of partition."""
        # 4 suites with 5 records each (total 20 records)
        records: list[PredictionRecord] = []
        for s_idx in range(4):
            for r_idx in range(5):
                records.append(
                    _make_audit_record(
                        suite_id=f"suite-{s_idx}",
                        request_id=f"req-{r_idx}",
                        prompt_tokens=64 * (r_idx + 1),
                        target_val=10.0 + r_idx,
                    )
                )

        train_set, test_set = DataSplitter.grouped_split(
            records, group_key="suite_id", test_ratio=0.25, seed=99
        )

        train_suites = {r.suite_id for r in train_set}
        test_suites = {r.suite_id for r in test_set}

        # Zero intersection: no suite appears in both training and test
        assert len(train_suites.intersection(test_suites)) == 0
        assert len(train_set) > 0
        assert len(test_set) > 0
        assert len(train_set) + len(test_set) == len(records)

    def test_grouped_split_single_group_raises_value_error(self) -> None:
        """Grouped split on a single suite must raise ValueError instead of silently leaking."""
        records = [
            _make_audit_record(suite_id="single-suite", request_id=f"req-{i}") for i in range(10)
        ]

        with pytest.raises(ValueError, match="fewer than 2 distinct groups"):
            DataSplitter.grouped_split(records, group_key="suite_id", test_ratio=0.2)

    def test_holdout_split_strictly_isolates_designated_models(self) -> None:
        """Holdout split guarantees 100% exclusion of designated models from training."""
        records = [
            _make_audit_record(model_id="llama-7b", suite_id="s1", request_id="1"),
            _make_audit_record(model_id="llama-7b", suite_id="s2", request_id="2"),
            _make_audit_record(model_id="mistral-7b", suite_id="s3", request_id="3"),
            _make_audit_record(model_id="mistral-7b", suite_id="s4", request_id="4"),
        ]

        train_set, test_set = DataSplitter.holdout_split(
            records, holdout_key="model_id", holdout_values=["mistral-7b"]
        )

        assert all(r.model_id != "mistral-7b" for r in train_set)
        assert all(r.model_id == "mistral-7b" for r in test_set)
        assert len(train_set) == 2
        assert len(test_set) == 2

    def test_holdout_split_empty_partitions_raise_value_error(self) -> None:
        """Holdout split with empty train or test set must raise ValueError."""
        records = [
            _make_audit_record(model_id="llama-7b", suite_id="s1", request_id="1"),
            _make_audit_record(model_id="llama-7b", suite_id="s2", request_id="2"),
        ]

        # Case 1: All records match holdout -> empty train set
        with pytest.raises(ValueError, match="empty training set"):
            DataSplitter.holdout_split(records, holdout_key="model_id", holdout_values=["llama-7b"])

        # Case 2: No records match holdout -> empty test set
        with pytest.raises(ValueError, match="empty test set"):
            DataSplitter.holdout_split(
                records, holdout_key="model_id", holdout_values=["unseen-model"]
            )

        # Case 3: Empty holdout_values
        with pytest.raises(ValueError, match="requires non-empty holdout_values"):
            DataSplitter.holdout_split(records, holdout_key="model_id", holdout_values=[])


class TestRegressionMathematicsAudit:
    """Audit tests for linear algebra solver, feature encoding, and regularization."""

    def test_solve_linear_system_reference_calculations(self) -> None:
        """Validate Gaussian elimination against closed-form 2x2 and 3x3 reference systems."""
        # System 1: 2x2
        # 2*x0 + 3*x1 = 7
        # 4*x0 + 9*x1 = 17
        # Solution: x0 = 2.0, x1 = 1.0
        a2 = [[2.0, 3.0], [4.0, 9.0]]
        b2 = [7.0, 17.0]
        x2 = _solve_linear_system(a2, b2)
        assert round(x2[0], 6) == 2.0
        assert round(x2[1], 6) == 1.0

        # System 2: 3x3
        # 1*x0 + 1*x1 + 1*x2 = 6
        # 0*x0 + 2*x1 + 5*x2 = -4
        # 2*x0 + 5*x1 - 1*x2 = 27
        # Solution: x0 = 5.0, x1 = 3.0, x2 = -2.0
        a3 = [[1.0, 1.0, 1.0], [0.0, 2.0, 5.0], [2.0, 5.0, -1.0]]
        b3 = [6.0, -4.0, 27.0]
        x3 = _solve_linear_system(a3, b3)
        assert round(x3[0], 6) == 5.0
        assert round(x3[1], 6) == 3.0
        assert round(x3[2], 6) == -2.0

    def test_solve_linear_system_detects_non_finite_values(self) -> None:
        """Matrix or vector containing NaN or Inf must raise ValueError."""
        a_nan = [[1.0, float("nan")], [0.0, 1.0]]
        b = [1.0, 2.0]
        with pytest.raises(ValueError, match="non-finite"):
            _solve_linear_system(a_nan, b)

        a_inf = [[float("inf"), 1.0], [0.0, 1.0]]
        with pytest.raises(ValueError, match="non-finite"):
            _solve_linear_system(a_inf, b)

        a_valid = [[1.0, 0.0], [0.0, 1.0]]
        b_nan = [1.0, float("nan")]
        with pytest.raises(ValueError, match="non-finite"):
            _solve_linear_system(a_valid, b_nan)

    def test_solve_linear_system_detects_singular_matrix(self) -> None:
        """Singular matrix must be detected and rejected."""
        a_sing = [[1.0, 2.0], [2.0, 4.0]]
        b = [3.0, 6.0]
        with pytest.raises(ValueError, match="Singular matrix"):
            _solve_linear_system(a_sing, b)

    def test_standardization_strictly_isolated_to_training_partition(self) -> None:
        """Feature means and standard deviations must derive solely from training records."""
        train_records = [
            _make_audit_record(prompt_tokens=100, target_val=10.0),
            _make_audit_record(prompt_tokens=200, target_val=20.0),
            _make_audit_record(prompt_tokens=300, target_val=30.0),
        ]
        predictor = RidgeRegressionPredictor(
            target=PredictionTarget.TTFT_MS,
            feature_names=["prompt_tokens"],
        ).fit(train_records)

        # Expected training mean = (100+200+300)/3 = 200.0
        assert round(predictor.means[0], 2) == 200.0
        # Expected training std = sqrt(((100-200)^2 + 0 + (300-200)^2)/3) = sqrt(20000/3) = 81.65
        assert round(predictor.stds[0], 2) == round(math.sqrt(20000 / 3), 2)


class TestEvaluationSemanticsAudit:
    """Audit tests for metric accuracy, near-zero MAPE protection, and EvaluationResult fields."""

    def test_hand_calculated_error_metrics(self) -> None:
        """Verify MAE, RMSE, and MAPE against known closed-form values."""
        actuals = [100.0, 200.0]
        preds = [110.0, 180.0]
        mae, rmse, mape = PredictionEvaluator.calculate_metrics(actuals, preds)

        assert mae == 15.0  # (|100-110| + |200-180|) / 2 = 15.0
        assert rmse == 15.81  # sqrt((100 + 400) / 2) = sqrt(250) = 15.81
        assert mape == 10.0  # ((10/100 + 20/200) / 2) * 100 = 10.0%

    def test_mape_returns_none_when_any_target_is_near_zero(self) -> None:
        """MAPE must return None if any target is near zero (< 1e-3) to prevent misleading percentage claims."""
        actuals_with_zero = [0.0, 50.0, 100.0]
        preds = [1.0, 52.0, 98.0]
        _, _, mape = PredictionEvaluator.calculate_metrics(actuals_with_zero, preds)
        assert mape is None

        actuals_near_zero = [0.0001, 10.0]
        _, _, mape_near = PredictionEvaluator.calculate_metrics(actuals_near_zero, [0.01, 10.0])
        assert mape_near is None

    def test_evaluation_result_schema_completeness(self) -> None:
        """EvaluationResult communicates train/test counts, execution mode, is_synthetic, and sufficiency."""
        train_records = [
            _make_audit_record(suite_id="s1", request_id=str(i), target_val=10.0 + i)
            for i in range(5)
        ]
        test_records = [
            _make_audit_record(suite_id="s2", request_id=str(i), target_val=15.0 + i)
            for i in range(3)
        ]

        ridge = RidgeRegressionPredictor(
            target=PredictionTarget.TTFT_MS,
            feature_names=["prompt_tokens"],
        ).fit(train_records)

        result = PredictionEvaluator.evaluate(
            predictor=ridge,
            test_records=test_records,
            split_strategy="GROUPED_BY_SUITE",
        )

        assert isinstance(result, EvaluationResult)
        assert result.train_sample_count == 5
        assert result.test_sample_count == 3
        assert result.sample_count == 3
        assert result.is_synthetic is False
        assert result.execution_mode == ExecutionMode.REAL
        assert result.sufficiency_threshold == 10
        assert result.is_statistically_sufficient is False  # 3 < 10


class TestAnalyticalBaselineProvenanceAudit:
    """Audit tests ensuring analytical baseline never silently fabricates missing inputs."""

    def test_missing_parameter_count_raises_value_error(self) -> None:
        """Analytical baseline must reject missing parameter_count_billion rather than fabricating 7.0B."""
        predictor = AnalyticalBaselinePredictor(target=PredictionTarget.TTFT_MS).fit([])

        rec = PredictionRecord(
            record_id="rec-missing-pcount",
            suite_id="s1",
            model_id="unknown-model",
            engine_type=EngineType.MOCK,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.HEURISTIC,
            parameter_count_billion=None,  # Missing!
            hardware_catalog_id="nvidia-h100-sxm5-80gb",
            concurrency=1,
            prompt_tokens=128,
            requested_output_tokens=128,
            actual_output_tokens=64,
            planned_sequence_length=256,
            total_sequence_length=192,
        )

        with pytest.raises(ValueError, match="parameter_count_billion"):
            predictor.predict(rec)

    def test_missing_hardware_catalog_id_raises_value_error(self) -> None:
        """Analytical roofline baseline must reject missing hardware spec rather than fabricating nominal constants."""
        predictor = AnalyticalBaselinePredictor(target=PredictionTarget.TPOT_MS).fit([])

        rec = PredictionRecord(
            record_id="rec-missing-gpu",
            suite_id="s1",
            model_id="llama-7b",
            engine_type=EngineType.MOCK,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.HEURISTIC,
            parameter_count_billion=7.0,
            hardware_catalog_id=None,  # Missing GPU catalog ID!
            concurrency=1,
            prompt_tokens=128,
            requested_output_tokens=128,
            actual_output_tokens=64,
            planned_sequence_length=256,
            total_sequence_length=192,
        )

        with pytest.raises(ValueError, match="resolved GPUSpecification"):
            predictor.predict(rec)

    def test_chunk_latency_rejected_with_architectural_explanation(self) -> None:
        """Analytical baseline must reject chunk latency because chunks are transport framing artifacts."""
        predictor = AnalyticalBaselinePredictor(target=PredictionTarget.CHUNK_LATENCY_MS).fit([])

        rec = _make_audit_record(target_val=50.0)
        with pytest.raises(ValueError, match="does not support chunk latency"):
            predictor.predict(rec)
