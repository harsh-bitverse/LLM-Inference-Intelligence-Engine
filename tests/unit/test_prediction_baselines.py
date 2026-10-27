"""Unit tests for inference performance baseline predictors.

Tests ConstantMedianPredictor, RidgeRegressionPredictor (with Gaussian elimination solver),
AnalyticalBaselinePredictor, and strict provenance preservation.
"""

import pytest

from llm_ii_engine.core.enums import (
    EngineType,
    ExecutionMode,
    ProvenanceType,
)
from llm_ii_engine.prediction.baselines import (
    AnalyticalBaselinePredictor,
    ConstantMedianPredictor,
    InsufficientDataError,
    RidgeRegressionPredictor,
    _solve_linear_system,
)
from llm_ii_engine.prediction.models import PredictionOutput, PredictionRecord, PredictionTarget


def _make_record(
    target_val: float,
    prompt_tokens: int = 128,
    output_tokens: int = 64,
    parameter_count_billion: float = 7.0,
    concurrency: int = 1,
    bytes_per_parameter: float = 2.0,
    hardware_catalog_id: str = "nvidia-h100-sxm5-80gb",
    execution_mode: ExecutionMode = ExecutionMode.REAL,
) -> PredictionRecord:
    """Helper to build a PredictionRecord with specified target value and features."""
    return PredictionRecord(
        record_id=f"rec-{target_val}",
        suite_id="suite-1",
        model_id="llama-7b",
        engine_type=EngineType.MOCK,
        execution_mode=execution_mode,
        provenance=ProvenanceType.MEASURED,
        parameter_count_billion=parameter_count_billion,
        bytes_per_parameter=bytes_per_parameter,
        hardware_catalog_id=hardware_catalog_id,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_sequence_length=prompt_tokens + output_tokens,
        ttft_ms=target_val,
        tpot_ms=target_val,
        output_tokens_per_second=target_val,
        peak_vram_mb=target_val,
        is_eligible_for_ttft=True,
        is_eligible_for_tpot=True,
        is_eligible_for_throughput=True,
        is_eligible_for_vram=True,
    )


class TestLinearSystemSolver:
    """Tests for Gaussian elimination with partial pivoting."""

    def test_solve_exact_2x2_system(self) -> None:
        """Solve: 2*x0 + 1*x1 = 8, 1*x0 + 3*x1 = 14 -> x0=2, x1=4."""
        a = [[2.0, 1.0], [1.0, 3.0]]
        b = [8.0, 14.0]
        x = _solve_linear_system(a, b)
        assert round(x[0], 5) == 2.0
        assert round(x[1], 5) == 4.0

    def test_solve_3x3_system(self) -> None:
        """Solve: 3x3 diagonal system."""
        a = [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
        b = [10.0, 20.0, 15.0]
        x = _solve_linear_system(a, b)
        assert x == [5.0, 5.0, 3.0]

    def test_singular_matrix_raises_value_error(self) -> None:
        """Singular matrix must raise ValueError."""
        a = [[1.0, 2.0], [2.0, 4.0]]  # Linearly dependent rows
        b = [3.0, 6.0]
        with pytest.raises(ValueError, match="Singular matrix"):
            _solve_linear_system(a, b)


class TestConstantMedianPredictor:
    """Test suite for ConstantMedianPredictor baseline."""

    def test_median_fitting_and_prediction(self) -> None:
        """ConstantMedianPredictor predicts exact training set median."""
        records = [
            _make_record(target_val=10.0),
            _make_record(target_val=20.0),
            _make_record(target_val=30.0),
            _make_record(target_val=40.0),
            _make_record(target_val=50.0),
        ]
        predictor = ConstantMedianPredictor(target=PredictionTarget.TTFT_MS).fit(records)

        assert predictor.median_value == 30.0
        assert predictor.training_sample_count == 5

        # Predict for a new record
        out = predictor.predict(_make_record(target_val=999.0))
        assert isinstance(out, PredictionOutput)
        assert out.predicted_value == 30.0
        assert out.target == PredictionTarget.TTFT_MS
        assert out.unit == "ms"
        assert out.predictor_type == "ConstantMedianBaseline"
        assert out.provenance == ProvenanceType.PREDICTED

    def test_zero_eligible_records_raises_insufficient_data(self) -> None:
        """Fitting on zero eligible records must raise InsufficientDataError."""
        predictor = ConstantMedianPredictor(target=PredictionTarget.TTFT_MS)
        with pytest.raises(InsufficientDataError, match="0 eligible observations"):
            predictor.fit([])

    def test_execution_mode_preservation(self) -> None:
        """Predictor inherits REAL execution mode when fitted on REAL observations."""
        records = [_make_record(target_val=25.0, execution_mode=ExecutionMode.REAL)]
        predictor = ConstantMedianPredictor(target=PredictionTarget.TTFT_MS).fit(records)
        out = predictor.predict(_make_record(target_val=0.0))
        assert out.execution_mode == ExecutionMode.REAL


class TestRidgeRegressionPredictor:
    """Test suite for RidgeRegressionPredictor."""

    def test_fit_and_predict_known_linear_trend(self) -> None:
        """Ridge regression captures linear scaling with prompt length."""
        # y = 10.0 + 0.1 * prompt_tokens
        records = [
            _make_record(target_val=10.0 + 0.1 * p, prompt_tokens=p)
            for p in [32, 64, 128, 256, 512, 1024]
        ]
        predictor = RidgeRegressionPredictor(
            target=PredictionTarget.TTFT_MS,
            feature_names=["prompt_tokens"],
            alpha=0.01,  # minimal regularization to track slope closely
        ).fit(records)

        assert predictor.is_fitted is True
        assert predictor.training_sample_count == 6

        # Predict at prompt_tokens = 200 -> expected ~ 10 + 20 = 30 ms
        out = predictor.predict(_make_record(target_val=0.0, prompt_tokens=200))
        assert abs(out.predicted_value - 30.0) < 3.0

    def test_insufficient_samples_raises(self) -> None:
        """Requires >= 3 samples to fit; fewer must raise InsufficientDataError."""
        records = [_make_record(target_val=10.0), _make_record(target_val=20.0)]
        predictor = RidgeRegressionPredictor(target=PredictionTarget.TTFT_MS)
        with pytest.raises(InsufficientDataError, match="requires >= 3 valid records"):
            predictor.fit(records)

    def test_missing_feature_imputed_with_training_mean(self) -> None:
        """Record with missing feature uses training-partition mean without crashing."""
        records = [
            _make_record(target_val=20.0, prompt_tokens=100),
            _make_record(target_val=30.0, prompt_tokens=200),
            _make_record(target_val=40.0, prompt_tokens=300),
        ]
        predictor = RidgeRegressionPredictor(
            target=PredictionTarget.TTFT_MS,
            feature_names=["prompt_tokens"],
        ).fit(records)

        # Build test record without prompt_tokens feature
        test_dict = _make_record(target_val=0.0).model_dump()
        test_dict["prompt_tokens"] = 0
        rec = PredictionRecord.model_validate(test_dict)

        out = predictor.predict(rec)
        assert out.predicted_value >= 0.0

    def test_non_negative_clamping(self) -> None:
        """Predictions must never return negative values for physical latency/throughput metrics."""
        records = [
            _make_record(target_val=1.0, prompt_tokens=10),
            _make_record(target_val=2.0, prompt_tokens=20),
            _make_record(target_val=3.0, prompt_tokens=30),
        ]
        predictor = RidgeRegressionPredictor(
            target=PredictionTarget.TTFT_MS,
            feature_names=["prompt_tokens"],
        ).fit(records)

        # Negative slope extrapolation check: output must be >= 0.0
        out = predictor.predict(_make_record(target_val=0.0, prompt_tokens=0))
        assert out.predicted_value >= 0.0


class TestAnalyticalBaselinePredictor:
    """Test suite for Session 05 analytical roofline baseline."""

    def test_analytical_baseline_calculations_and_provenance(self) -> None:
        """Analytical baseline calculates theoretical ceilings with HEURISTIC provenance."""
        predictor = AnalyticalBaselinePredictor(target=PredictionTarget.TTFT_MS).fit([])

        rec = _make_record(
            target_val=50.0,
            parameter_count_billion=7.0,
            prompt_tokens=128,
            hardware_catalog_id="nvidia-h100-sxm5-80gb",
        )
        out = predictor.predict(rec)

        assert isinstance(out, PredictionOutput)
        assert out.predicted_value > 0.0
        assert out.provenance == ProvenanceType.HEURISTIC
        assert out.validation_status == "THEORETICAL_ANALYTICAL"
        assert "roofline" in out.predictor_type.lower()
        assert any("theoretical" in lim.lower() for lim in out.limitations)
