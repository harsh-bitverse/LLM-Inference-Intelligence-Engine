"""Unit tests for data partitioning strategies, evaluation metrics, and baseline comparisons.

Tests random, grouped (by suite), and holdout splits, MAE/RMSE/MAPE hand calculations,
zero-target handling, and statistical sufficiency checks.
"""

import pytest

from llm_ii_engine.core.enums import (
    EngineType,
    ExecutionMode,
    ProvenanceType,
)
from llm_ii_engine.prediction.baselines import ConstantMedianPredictor, RidgeRegressionPredictor
from llm_ii_engine.prediction.evaluation import DataSplitter, PredictionEvaluator
from llm_ii_engine.prediction.models import EvaluationResult, PredictionRecord, PredictionTarget


def _make_eval_record(
    suite_id: str,
    target_val: float,
    model_id: str = "llama-7b",
    prompt_tokens: int = 128,
) -> PredictionRecord:
    """Helper to build a PredictionRecord with suite and model identifiers."""
    return PredictionRecord(
        record_id=f"rec-{suite_id}-{target_val}",
        suite_id=suite_id,
        model_id=model_id,
        engine_type=EngineType.MOCK,
        execution_mode=ExecutionMode.REAL,
        provenance=ProvenanceType.MEASURED,
        parameter_count_billion=7.0,
        concurrency=1,
        prompt_tokens=prompt_tokens,
        output_tokens=64,
        total_sequence_length=prompt_tokens + 64,
        ttft_ms=target_val,
        tpot_ms=target_val,
        output_tokens_per_second=target_val,
        is_eligible_for_ttft=True,
        is_eligible_for_tpot=True,
        is_eligible_for_throughput=True,
    )


class TestDataSplitter:
    """Test suite for data partitioning strategies and leakage prevention."""

    def test_random_split_proportions_and_reproducibility(self) -> None:
        """Random split respects test ratio and is deterministic with seed."""
        records = [_make_eval_record(f"suite-{i}", float(i)) for i in range(20)]

        train_a, test_a = DataSplitter.random_split(records, test_ratio=0.2, seed=42)
        train_b, test_b = DataSplitter.random_split(records, test_ratio=0.2, seed=42)

        assert len(test_a) == 4  # 20 * 0.2 = 4
        assert len(train_a) == 16
        # Deterministic reproducibility
        assert [r.record_id for r in test_a] == [r.record_id for r in test_b]

    def test_grouped_split_prevents_intra_suite_leakage(self) -> None:
        """Grouped split guarantees that all records from the same suite stay together."""
        # 3 suites: suite-A (5 records), suite-B (5 records), suite-C (5 records)
        records: list[PredictionRecord] = []
        for s in ["suite-A", "suite-B", "suite-C"]:
            for i in range(5):
                records.append(_make_eval_record(s, float(i)))

        train_set, test_set = DataSplitter.grouped_split(
            records, group_key="suite_id", test_ratio=0.33, seed=42
        )

        train_suites = {r.suite_id for r in train_set}
        test_suites = {r.suite_id for r in test_set}

        # Crucial: zero intersection between train and test suite IDs!
        assert len(train_suites.intersection(test_suites)) == 0
        assert len(train_set) + len(test_set) == 15

    def test_holdout_split_by_model(self) -> None:
        """Holdout split reserves specified model variants strictly for evaluation."""
        records = [
            _make_eval_record("suite-1", 10.0, model_id="llama-7b"),
            _make_eval_record("suite-2", 15.0, model_id="llama-7b"),
            _make_eval_record("suite-3", 25.0, model_id="mistral-7b"),
            _make_eval_record("suite-4", 30.0, model_id="mistral-7b"),
        ]

        train_set, test_set = DataSplitter.holdout_split(
            records, holdout_key="model_id", holdout_values=["mistral-7b"]
        )

        assert all(r.model_id == "llama-7b" for r in train_set)
        assert all(r.model_id == "mistral-7b" for r in test_set)
        assert len(test_set) == 2
        assert len(train_set) == 2

    def test_invalid_split_ratios_rejected(self) -> None:
        """Split ratios <= 0 or >= 1 must raise ValueError."""
        with pytest.raises(ValueError):
            DataSplitter.random_split([_make_eval_record("s", 1.0)], test_ratio=0.0)
        with pytest.raises(ValueError):
            DataSplitter.random_split([_make_eval_record("s", 1.0)], test_ratio=1.0)


class TestPredictionEvaluatorMetrics:
    """Test suite for error metrics and baseline comparisons."""

    def test_hand_calculated_mae_rmse_mape(self) -> None:
        """Verify metric calculations against known hand-derived values.

        Actuals: [10.0, 20.0, 30.0]
        Predictions: [12.0, 18.0, 33.0]
        Absolute Errors: [2.0, 2.0, 3.0] -> MAE = 7 / 3 = 2.33
        Squared Errors: [4.0, 4.0, 9.0] = 17 -> RMSE = sqrt(17/3) = 2.38
        Relative Errors: [2/10, 2/20, 3/30] = [0.2, 0.1, 0.1] = 0.4 -> MAPE = 13.33%
        """
        actuals = [10.0, 20.0, 30.0]
        preds = [12.0, 18.0, 33.0]

        mae, rmse, mape = PredictionEvaluator.calculate_metrics(actuals, preds)
        assert mae == 2.33
        assert rmse == 2.38
        assert mape == 13.33

    def test_zero_target_metrics_handled_without_division_by_zero(self) -> None:
        """All-zero target list must not raise ZeroDivisionError."""
        actuals = [0.0, 0.0]
        preds = [0.0, 0.0]

        mae, rmse, mape = PredictionEvaluator.calculate_metrics(actuals, preds)
        assert mae == 0.0
        assert rmse == 0.0
        assert mape is None  # Handled safely without error

    def test_evaluator_generates_complete_evaluation_result(self) -> None:
        """PredictionEvaluator produces typed EvaluationResult with baseline comparison."""
        train_records = [
            _make_eval_record("suite-1", 10.0, prompt_tokens=32),
            _make_eval_record("suite-1", 20.0, prompt_tokens=64),
            _make_eval_record("suite-1", 30.0, prompt_tokens=128),
            _make_eval_record("suite-1", 40.0, prompt_tokens=256),
        ]
        test_records = [
            _make_eval_record("suite-2", 15.0, prompt_tokens=48),
            _make_eval_record("suite-2", 35.0, prompt_tokens=200),
        ]

        ridge = RidgeRegressionPredictor(
            target=PredictionTarget.TTFT_MS,
            feature_names=["prompt_tokens"],
        ).fit(train_records)

        median_baseline = ConstantMedianPredictor(target=PredictionTarget.TTFT_MS).fit(
            train_records
        )

        result = PredictionEvaluator.evaluate(
            predictor=ridge,
            test_records=test_records,
            split_strategy="GROUPED_BY_SUITE",
            baseline_predictor=median_baseline,
        )

        assert isinstance(result, EvaluationResult)
        assert result.target == PredictionTarget.TTFT_MS
        assert result.sample_count == 2
        assert result.mae >= 0.0
        assert result.rmse >= 0.0
        assert result.baseline_mae is not None
        assert result.is_statistically_sufficient is False  # N=2 < 10
        assert "exploratory" in result.notes.lower()

    def test_end_to_end_pipeline_integration(self) -> None:
        """End-to-end integration: dataset -> grouped split -> fit -> evaluate."""
        # 4 suites with 3 records each (N=12 total)
        records: list[PredictionRecord] = []
        for s_idx in range(4):
            for r_idx in range(3):
                p_len = 32 * (s_idx + 1) + 10 * r_idx
                ttft = 10.0 + 0.1 * p_len
                records.append(_make_eval_record(f"suite-{s_idx}", ttft, prompt_tokens=p_len))

        train_recs, test_recs = DataSplitter.grouped_split(
            records, group_key="suite_id", test_ratio=0.25, seed=123
        )

        ridge = RidgeRegressionPredictor(
            target=PredictionTarget.TTFT_MS,
            feature_names=["prompt_tokens"],
            alpha=0.1,
        ).fit(train_recs)

        median = ConstantMedianPredictor(target=PredictionTarget.TTFT_MS).fit(train_recs)

        result = PredictionEvaluator.evaluate(
            predictor=ridge,
            test_records=test_recs,
            split_strategy="GROUPED_BY_SUITE",
            baseline_predictor=median,
        )

        assert result.sample_count == len(test_recs)
        assert result.mae >= 0.0
        # Ridge regression should beat constant median baseline on linear data
        if result.baseline_mae is not None:
            assert result.mae <= result.baseline_mae
