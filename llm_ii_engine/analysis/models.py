"""Domain models for analytical resource estimation, Roofline analysis, and diagnostics.

Enforces strict data provenance and separates theoretical bounds from observed measurements.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.core.enums import (
    DiagnosticCategory,
    DiagnosticConfidence,
    DiagnosticSeverity,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    QuantizationType,
)
from llm_ii_engine.core.schemas import Observation


class ModelResourceEstimate(BaseModel):
    """Analytical resource estimation for an LLM model and workload configuration.

    Computes logical parameter storage, attention KV-cache memory, and theoretical
    computational FLOPs based on documented transformer architectural equations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1, description="Target model identifier")
    parameter_count_billion: float | None = Field(
        default=None, gt=0.0, description="Total model parameter count in billions"
    )

    # Parameter storage
    parameter_storage_mb: float = Field(
        ge=0.0, description="Logical parameter weight storage in MiB"
    )
    weight_precision: QuantizationType = Field(
        description="Weight precision or quantization scheme"
    )
    bytes_per_parameter: float = Field(
        gt=0.0, description="Effective bytes required per model parameter"
    )
    is_quantization_exact: bool = Field(
        description="True if weight byte size is mathematically exact; False if metadata/packing is estimated"
    )
    quantization_notes: str | None = Field(
        default=None, description="Caveats regarding quantization scales, packing, or group size"
    )

    # KV-cache memory
    attention_architecture: str = Field(
        description="Attention variant: 'MHA' (Multi-Head), 'GQA' (Grouped-Query), 'MQA' (Multi-Query), or 'UNKNOWN'"
    )
    kv_cache_per_token_bytes: float | None = Field(
        default=None, ge=0.0, description="KV-cache bytes per token across all layers"
    )
    kv_cache_total_mb: float | None = Field(
        default=None, ge=0.0, description="Total KV-cache payload memory in MiB"
    )
    kv_cache_precision: QuantizationType = Field(
        default=QuantizationType.FP16, description="Precision of cached KV tensors"
    )
    batch_size: int = Field(ge=1, description="Active batch size (concurrency)")
    sequence_length: int = Field(
        ge=0, description="Evaluated context sequence length (prompt + output)"
    )

    # Aggregates & Runtime Memory
    total_payload_memory_mb: float = Field(
        ge=0.0, description="Combined theoretical payload memory (parameters + KV cache) in MiB"
    )
    estimated_runtime_vram_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated actual runtime VRAM in MiB (payload + allocator margin + context overhead)",
    )

    # Computational Work (FLOPs)
    prefill_flops: float | None = Field(
        default=None, ge=0.0, description="Estimated computational work for prefill phase in FLOPs"
    )
    decode_flops: float | None = Field(
        default=None, ge=0.0, description="Estimated computational work for decode phase in FLOPs"
    )

    # Operational Intensity (FLOPs / Byte)
    operational_intensity_prefill: float | None = Field(
        default=None,
        ge=0.0,
        description="Analytically estimated operational intensity for prefill (FLOPs/Byte)",
    )
    operational_intensity_decode: float | None = Field(
        default=None,
        ge=0.0,
        description="Analytically estimated operational intensity for decode (FLOPs/Byte)",
    )

    provenance: ProvenanceType = Field(
        default=ProvenanceType.PREDICTED,
        description="Analytical estimates are PREDICTED, not directly MEASURED",
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Documented architectural and mathematical assumptions"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Documented limitations and unknown architectural fields"
    )


class RooflineAnalysisResult(BaseModel):
    """Evaluation of achievable performance ceilings using the theoretical Roofline Model.

    Combines hardware catalog specifications with operational intensity to derive
    the machine balance point (ridge point) and attainable throughput upper bounds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hardware_name: str = Field(min_length=1, description="Target GPU/accelerator product name")
    form_factor: str = Field(description="Hardware form factor (e.g. 'SXM5', 'PCIe')")
    precision: str = Field(description="Compute precision evaluating ceiling (e.g. 'fp16_dense')")
    peak_compute_tflops: float = Field(
        gt=0.0, description="Theoretical peak compute ceiling in TFLOPs"
    )
    peak_memory_bandwidth_gbps: float = Field(
        gt=0.0, description="Theoretical peak memory bandwidth in GB/s"
    )
    ridge_point_flops_per_byte: float = Field(
        gt=0.0,
        description="Machine balance point: peak compute FLOPs / peak memory bandwidth bytes",
    )

    operational_intensity: float = Field(
        gt=0.0, description="Operational intensity (Arithmetic Intensity) in FLOPs/Byte"
    )
    operational_intensity_provenance: ProvenanceType = Field(
        description="Provenance of operational intensity: PREDICTED (analytical) or MEASURED"
    )
    attainable_compute_tflops: float = Field(
        gt=0.0, description="Theoretical upper bound on attainable compute throughput in TFLOPs"
    )
    theoretical_max_tokens_per_second: float | None = Field(
        default=None,
        ge=0.0,
        description="Theoretical upper bound on token generation rate (tok/s) under this roofline",
    )
    limiting_factor: str = Field(
        description="'MEMORY_BANDWIDTH_BOUND', 'COMPUTE_BOUND', or 'AT_RIDGE_POINT'"
    )

    provenance: ProvenanceType = Field(
        default=ProvenanceType.HEURISTIC,
        description="Catalog ceilings and theoretical rooflines are HEURISTIC",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Documented assumptions underlying this roofline evaluation",
    )
    notes: str = Field(
        default="Roofline calculation represents theoretical upper bounds, not measured hardware counters.",
        description="Contextual note",
    )


class DiagnosticFinding(BaseModel):
    """An individual evidence-based diagnostic finding with explicit provenance and confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str = Field(
        min_length=1, description="Unique identifier for this diagnostic finding"
    )
    category: DiagnosticCategory = Field(description="Diagnostic categorization")
    severity: DiagnosticSeverity = Field(description="Severity: INFO, WARNING, or CRITICAL")
    title: str = Field(min_length=1, description="Brief summary title of finding")
    message: str = Field(min_length=1, description="Detailed diagnostic explanation and reasoning")

    confidence: DiagnosticConfidence = Field(
        description="Confidence in finding: HIGH, MEDIUM, LOW, or UNKNOWN"
    )
    confidence_rationale: str = Field(
        min_length=1, description="Justification for assigned confidence based on evidence quality"
    )

    supporting_observations: list[Observation[Any]] = Field(
        default_factory=list,
        description="Typed observations and data points backing this diagnosis",
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Assumptions made by the diagnostic rule"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Data or telemetry gaps preventing a stronger conclusion"
    )
    suggested_experiments: list[str] = Field(
        default_factory=list,
        description="Actionable follow-up benchmark or configuration experiments",
    )


class DiagnosticReport(BaseModel):
    """Comprehensive performance and bottleneck diagnostic report for an experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(min_length=1, description="Unique diagnostic report identifier")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when diagnostic evaluation occurred",
    )
    model_id: str = Field(min_length=1, description="Target model evaluated")
    engine_type: EngineType = Field(description="Inference engine runtime evaluated")
    execution_mode: ExecutionMode = Field(
        description="REAL vs SIMULATED execution mode of evidence"
    )
    hardware_catalog_id: str | None = Field(
        default=None, description="Matched hardware catalog identifier if GPU was resolved"
    )

    findings: list[DiagnosticFinding] = Field(
        default_factory=list, description="List of evidence-backed diagnostic findings"
    )
    resource_estimate: ModelResourceEstimate | None = Field(
        default=None, description="Analytical model resource estimates"
    )
    roofline_analysis: RooflineAnalysisResult | None = Field(
        default=None, description="Roofline ceiling evaluation if hardware was resolved"
    )
    summary: str = Field(min_length=1, description="Executive summary of diagnostic conclusions")
