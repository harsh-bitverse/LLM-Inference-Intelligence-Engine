"""Theoretical Roofline performance modeling.

Derives machine balance points (ridge points), attainable compute throughput ceilings,
and theoretical token generation limits based on hardware catalog specifications and
arithmetic operational intensity.
"""

from llm_ii_engine.analysis.models import RooflineAnalysisResult
from llm_ii_engine.core.enums import ProvenanceType
from llm_ii_engine.discovery.models import GPUSpecification


class RooflineAnalyzer:
    """Evaluates theoretical compute and memory bandwidth ceilings under the Roofline Model."""

    SUPPORTED_PRECISIONS: set[str] = {
        "fp16_dense",
        "fp16_sparse",
        "bf16_dense",
        "bf16_sparse",
        "fp8_dense",
        "fp8_sparse",
        "int8_dense",
        "int8_sparse",
        "fp32",
    }

    @classmethod
    def get_peak_compute_tflops(
        cls,
        gpu_spec: GPUSpecification,
        precision: str = "fp16_dense",
    ) -> float:
        """Extract theoretical peak compute throughput in TFLOPs for a precision.

        Args:
            gpu_spec: GPUSpecification catalog entry.
            precision: Precision string (e.g. 'fp16_dense', 'bf16_dense', 'int8_dense', 'fp32').

        Returns:
            Peak compute in TFLOPs.

        Raises:
            ValueError: If precision is unrecognized or not documented for this GPU.
        """
        if precision not in cls.SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision '{precision}'. Must be one of: {sorted(cls.SUPPORTED_PRECISIONS)}"
            )

        compute_spec = gpu_spec.compute
        value: float | None = None
        if precision == "fp16_dense":
            value = compute_spec.fp16_dense_tflops
        elif precision == "fp16_sparse":
            value = compute_spec.fp16_sparse_tflops
        elif precision == "bf16_dense":
            value = compute_spec.bf16_dense_tflops
        elif precision == "bf16_sparse":
            value = compute_spec.bf16_sparse_tflops
        elif precision == "fp8_dense":
            value = compute_spec.fp8_dense_tflops
        elif precision == "fp8_sparse":
            value = compute_spec.fp8_sparse_tflops
        elif precision == "int8_dense":
            value = compute_spec.int8_dense_tops
        elif precision == "int8_sparse":
            value = compute_spec.int8_sparse_tops
        elif precision == "fp32":
            value = compute_spec.fp32_tflops

        if value is None:
            raise ValueError(
                f"GPU '{gpu_spec.name}' has no documented compute specification for precision '{precision}'"
            )

        return float(value)

    @classmethod
    def calculate_ridge_point(
        cls,
        peak_compute_tflops: float,
        peak_memory_bandwidth_gbps: float,
    ) -> float:
        """Calculate the machine balance point (ridge point) in FLOPs/Byte.

        Formula:
            I_ridge = (peak_compute_tflops * 1e12 FLOP/s) / (peak_memory_bandwidth_gbps * 1e9 Byte/s)
                    = (peak_compute_tflops * 1000) / peak_memory_bandwidth_gbps

        Args:
            peak_compute_tflops: Peak compute ceiling in TFLOPs.
            peak_memory_bandwidth_gbps: Peak memory bandwidth ceiling in GB/s.

        Returns:
            Ridge point in FLOPs/Byte.
        """
        if peak_compute_tflops <= 0:
            raise ValueError(f"peak_compute_tflops must be positive, got {peak_compute_tflops}")
        if peak_memory_bandwidth_gbps <= 0:
            raise ValueError(
                f"peak_memory_bandwidth_gbps must be positive, got {peak_memory_bandwidth_gbps}"
            )

        ridge_point = (peak_compute_tflops * 1000.0) / peak_memory_bandwidth_gbps
        return round(ridge_point, 2)

    @classmethod
    def evaluate_roofline(
        cls,
        gpu_spec: GPUSpecification,
        operational_intensity: float,
        operational_intensity_provenance: ProvenanceType = ProvenanceType.PREDICTED,
        precision: str = "fp16_dense",
        parameter_count_billion: float | None = None,
    ) -> RooflineAnalysisResult:
        """Evaluate theoretical attainable performance under the Roofline Model.

        Args:
            gpu_spec: Catalog specification of the target GPU.
            operational_intensity: Arithmetic/Operational intensity (FLOPs/Byte).
            operational_intensity_provenance: PREDICTED (analytical) or MEASURED.
            precision: Target compute precision rating to evaluate.
            parameter_count_billion: Optional model parameter count (B) to bound token generation rate.

        Returns:
            Validated RooflineAnalysisResult.
        """
        if operational_intensity <= 0:
            raise ValueError(f"operational_intensity must be positive, got {operational_intensity}")

        peak_compute = cls.get_peak_compute_tflops(gpu_spec, precision)
        peak_bandwidth = gpu_spec.theoretical_memory_bandwidth_gbps
        ridge_point = cls.calculate_ridge_point(peak_compute, peak_bandwidth)

        # Attainable compute: min(Peak Compute, AI * Peak Bandwidth / 1000)
        bandwidth_constrained_tflops = (operational_intensity * peak_bandwidth) / 1000.0

        if operational_intensity < ridge_point:
            attainable_tflops = min(peak_compute, bandwidth_constrained_tflops)
            limiting_factor = "MEMORY_BANDWIDTH_BOUND"
        elif operational_intensity > ridge_point:
            attainable_tflops = peak_compute
            limiting_factor = "COMPUTE_BOUND"
        else:
            attainable_tflops = peak_compute
            limiting_factor = "AT_RIDGE_POINT"

        # Theoretical upper bound on token generation rate (tok/s):
        # Tokens/s = (Attainable FLOP/s) / (2 * P * 1e9 FLOP/tok) = (Attainable TFLOPs * 1000) / (2 * P)
        max_tok_per_sec: float | None = None
        if parameter_count_billion is not None and parameter_count_billion > 0:
            max_tok_per_sec = round(
                (attainable_tflops * 1000.0) / (2.0 * parameter_count_billion), 2
            )

        assumptions = [
            f"Hardware ceiling: {gpu_spec.name} ({gpu_spec.form_factor}) with theoretical peak {peak_compute:.1f} TFLOPs ({precision}) and {peak_bandwidth:.1f} GB/s bandwidth.",
            f"Ridge point is {ridge_point:.2f} FLOPs/Byte. Workloads below this intensity are memory bandwidth bound; above are compute bound.",
            f"Evaluated operational intensity: {operational_intensity:.2f} FLOPs/Byte ({operational_intensity_provenance.value}).",
        ]
        if max_tok_per_sec is not None:
            assumptions.append(
                f"Theoretical token rate upper bound assumes 2 FLOPs per parameter per token ({parameter_count_billion:.2f}B parameters)."
            )

        return RooflineAnalysisResult(
            hardware_name=gpu_spec.name,
            form_factor=gpu_spec.form_factor,
            precision=precision,
            peak_compute_tflops=peak_compute,
            peak_memory_bandwidth_gbps=peak_bandwidth,
            ridge_point_flops_per_byte=ridge_point,
            operational_intensity=round(operational_intensity, 2),
            operational_intensity_provenance=operational_intensity_provenance,
            attainable_compute_tflops=round(attainable_tflops, 2),
            theoretical_max_tokens_per_second=max_tok_per_sec,
            limiting_factor=limiting_factor,
            provenance=ProvenanceType.HEURISTIC,
            assumptions=assumptions,
            notes=(
                "Roofline calculation represents theoretical upper bounds from manufacturer specs, "
                "not empirical hardware counter measurements."
            ),
        )
