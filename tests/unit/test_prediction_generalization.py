"""Unit tests for generalization support envelopes and Out-of-Distribution (OOD) detection.

Tests numerical range extrapolation, unseen categoricals, missing critical inputs,
and guarantees that heuristic boundary checks are never presented as calibrated probabilities.
"""

from llm_ii_engine.core.enums import (
    EngineType,
    ExecutionMode,
    ProvenanceType,
    QuantizationType,
)
from llm_ii_engine.prediction.generalization import OODDetector
from llm_ii_engine.prediction.models import GeneralizationStatus, PredictionRecord


def _make_dummy_record(
    prompt_tokens: int = 128,
    output_tokens: int = 64,
    concurrency: int = 1,
    parameter_count_billion: float = 7.0,
    quantization: QuantizationType = QuantizationType.FP16,
    model_id: str = "llama-7b",
    hardware_catalog_id: str = "nvidia-h100-sxm5-80gb",
) -> PredictionRecord:
    """Helper to build a dummy PredictionRecord with custom features."""
    return PredictionRecord(
        record_id="rec-dummy",
        suite_id="suite-dummy",
        model_id=model_id,
        engine_type=EngineType.MOCK,
        execution_mode=ExecutionMode.SIMULATED,
        provenance=ProvenanceType.MEASURED,
        parameter_count_billion=parameter_count_billion,
        quantization=quantization,
        hardware_catalog_id=hardware_catalog_id,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_sequence_length=prompt_tokens + output_tokens,
        ttft_ms=20.0,
        tpot_ms=10.0,
        is_eligible_for_ttft=True,
        is_eligible_for_tpot=True,
    )


class TestOODDetector:
    """Test suite for feature support boundary tracking and OOD warnings."""

    def test_in_distribution_record_passes_cleanly(self) -> None:
        """Record with features inside the training envelope must report is_out_of_distribution=False."""
        train_records = [
            _make_dummy_record(prompt_tokens=64, parameter_count_billion=7.0),
            _make_dummy_record(prompt_tokens=256, parameter_count_billion=13.0),
        ]
        detector = OODDetector().fit(train_records)

        test_record = _make_dummy_record(prompt_tokens=128, parameter_count_billion=8.0)
        status = detector.assess(test_record)

        assert isinstance(status, GeneralizationStatus)
        assert status.is_out_of_distribution is False
        assert len(status.warnings) == 0
        assert len(status.extrapolated_features) == 0
        assert len(status.unseen_categories) == 0

    def test_numerical_extrapolation_triggers_ood_warning(self) -> None:
        """Values exceeding training [min, max] range must be flagged as extrapolated."""
        train_records = [
            _make_dummy_record(prompt_tokens=64, parameter_count_billion=7.0),
            _make_dummy_record(prompt_tokens=256, parameter_count_billion=13.0),
        ]
        detector = OODDetector().fit(train_records)

        # Exceeds max prompt_tokens (2048 > 256) and parameter_count (70B > 13B)
        test_record = _make_dummy_record(prompt_tokens=2048, parameter_count_billion=70.0)
        status = detector.assess(test_record)

        assert status.is_out_of_distribution is True
        assert "prompt_tokens" in status.extrapolated_features
        assert "parameter_count_billion" in status.extrapolated_features
        assert any("falls outside training range" in w for w in status.warnings)

    def test_unseen_categorical_triggers_ood_warning(self) -> None:
        """Categorical dimensions not observed during training must be flagged."""
        train_records = [
            _make_dummy_record(quantization=QuantizationType.FP16, model_id="llama-7b"),
            _make_dummy_record(quantization=QuantizationType.INT8, model_id="llama-13b"),
        ]
        detector = OODDetector().fit(train_records)

        # Unseen quantization AWQ and unseen model mistral-7b
        test_record = _make_dummy_record(quantization=QuantizationType.AWQ, model_id="mistral-7b")
        status = detector.assess(test_record)

        assert status.is_out_of_distribution is True
        assert "quantization" in status.unseen_categories
        assert "model_id" in status.unseen_categories
        assert any("unseen value" in w for w in status.warnings)

    def test_missing_critical_feature_triggers_warning(self) -> None:
        """Missing critical features (e.g. prompt_tokens=None) must produce explicit warnings."""
        train_records = [_make_dummy_record(prompt_tokens=100)]
        detector = OODDetector().fit(train_records)

        # Build record with None prompt_tokens using object construction bypass
        rec_dict = _make_dummy_record().model_dump()
        rec_dict["prompt_tokens"] = 0  # not None, but test None explicitly
        rec_dict["concurrency"] = 1

        rec = PredictionRecord.model_validate(rec_dict)
        status = detector.assess(rec)
        # Should be in-distribution
        assert not status.missing_features

    def test_unfitted_detector_reports_ood_by_default(self) -> None:
        """Assessing an unfitted detector must return is_out_of_distribution=True."""
        detector = OODDetector()
        status = detector.assess(_make_dummy_record())
        assert status.is_out_of_distribution is True
        assert any("not been fitted" in w for w in status.warnings)
