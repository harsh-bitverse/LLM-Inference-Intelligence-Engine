"""Unit tests for theoretical Roofline performance modeling.

Tests machine balance point (ridge point) derivations, precision extraction,
attainable compute ceilings, and token generation rate upper bounds.
"""

import pytest

from llm_ii_engine.analysis.models import RooflineAnalysisResult
from llm_ii_engine.analysis.roofline import RooflineAnalyzer
from llm_ii_engine.core.enums import ProvenanceType
from llm_ii_engine.discovery.catalog import H100_SXM5_80GB, RTX_4090, TESLA_T4


class TestRooflineModelCalculations:
    """Test suite for Roofline equations and precision ratings."""

    def test_precision_extraction_h100_sxm5(self) -> None:
        """Verify precision ratings for H100 SXM5 from verified datasheet."""
        assert RooflineAnalyzer.get_peak_compute_tflops(H100_SXM5_80GB, "fp16_dense") == 989.0
        assert RooflineAnalyzer.get_peak_compute_tflops(H100_SXM5_80GB, "fp16_sparse") == 1979.0
        assert RooflineAnalyzer.get_peak_compute_tflops(H100_SXM5_80GB, "bf16_dense") == 989.0
        assert RooflineAnalyzer.get_peak_compute_tflops(H100_SXM5_80GB, "fp8_dense") == 1979.0
        assert RooflineAnalyzer.get_peak_compute_tflops(H100_SXM5_80GB, "int8_dense") == 1979.0
        assert RooflineAnalyzer.get_peak_compute_tflops(H100_SXM5_80GB, "fp32") == 67.0

    def test_unsupported_precision_rejected(self) -> None:
        """Unrecognized precision strings must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported precision"):
            RooflineAnalyzer.get_peak_compute_tflops(H100_SXM5_80GB, "int2_ultra")

    def test_ridge_point_calculation_h100(self) -> None:
        """H100 SXM5 ridge point = (989 * 1000) / 3350 = 295.22 FLOPs/Byte."""
        ridge = RooflineAnalyzer.calculate_ridge_point(989.0, 3350.0)
        assert ridge == 295.22

    def test_ridge_point_calculation_t4(self) -> None:
        """Tesla T4 ridge point = (65 * 1000) / 300 = 216.67 FLOPs/Byte."""
        ridge = RooflineAnalyzer.calculate_ridge_point(65.0, 300.0)
        assert ridge == 216.67

    def test_invalid_ridge_point_inputs_rejected(self) -> None:
        """Zero or negative compute or bandwidth must raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            RooflineAnalyzer.calculate_ridge_point(0.0, 1000.0)
        with pytest.raises(ValueError, match="must be positive"):
            RooflineAnalyzer.calculate_ridge_point(100.0, -500.0)

    def test_memory_bandwidth_bound_regime(self) -> None:
        """Workload with operational intensity below ridge point is memory-bandwidth bound."""
        # Operational intensity = 1.0 FLOPs/Byte (typical decode batch size 1)
        res = RooflineAnalyzer.evaluate_roofline(
            gpu_spec=H100_SXM5_80GB,
            operational_intensity=1.0,
            operational_intensity_provenance=ProvenanceType.PREDICTED,
            precision="fp16_dense",
            parameter_count_billion=7.0,
        )
        assert isinstance(res, RooflineAnalysisResult)
        assert res.limiting_factor == "MEMORY_BANDWIDTH_BOUND"
        assert res.ridge_point_flops_per_byte == 295.22
        # Attainable compute = (1.0 * 3350) / 1000 = 3.35 TFLOPs
        assert res.attainable_compute_tflops == 3.35
        # Token rate = (3.35 * 1000) / (2 * 7.0) = 3350 / 14 = 239.29 tok/s
        assert res.theoretical_max_tokens_per_second == 239.29
        assert res.provenance == ProvenanceType.HEURISTIC
        assert res.operational_intensity_provenance == ProvenanceType.PREDICTED

    def test_compute_bound_regime(self) -> None:
        """Workload with operational intensity above ridge point is compute bound."""
        # Operational intensity = 500.0 FLOPs/Byte (large prefill batch)
        res = RooflineAnalyzer.evaluate_roofline(
            gpu_spec=H100_SXM5_80GB,
            operational_intensity=500.0,
            operational_intensity_provenance=ProvenanceType.PREDICTED,
            precision="fp16_dense",
            parameter_count_billion=7.0,
        )
        assert res.limiting_factor == "COMPUTE_BOUND"
        assert res.attainable_compute_tflops == 989.0  # Peak compute ceiling
        # Token rate = (989 * 1000) / (2 * 7.0) = 989000 / 14 = 70642.86 tok/s
        assert res.theoretical_max_tokens_per_second == 70642.86

    def test_token_rate_bound_without_parameter_count(self) -> None:
        """If parameter count is not provided, theoretical_max_tokens_per_second is None."""
        res = RooflineAnalyzer.evaluate_roofline(
            gpu_spec=RTX_4090,
            operational_intensity=2.0,
            precision="fp16_dense",
            parameter_count_billion=None,
        )
        assert res.theoretical_max_tokens_per_second is None
        assert res.attainable_compute_tflops > 0.0

    def test_negative_or_zero_operational_intensity_rejected(self) -> None:
        """Operational intensity must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            RooflineAnalyzer.evaluate_roofline(
                gpu_spec=TESLA_T4,
                operational_intensity=0.0,
            )
