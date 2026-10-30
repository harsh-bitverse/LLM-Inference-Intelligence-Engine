"""Protocol schemas and domain models for Session 27 Serving Parallelism & Architecture Comparison.

Defines schemas for:
- Serving runtime architectures (Ollama single-slot, Ollama serial queue, llama.cpp parallel continuous batching).
- Serving capability matrix distinguishing advertised vs observed functionality.
- Fine-grained serving observations with monotonic timestamps, execution overlap, and slot telemetry.
- Architecture comparison reports (throughput gain, tail latency reduction, HoL blocking reduction).
- Architecture-aware serving recommendations.
- Versioned experiment manifest.
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ServingRuntimeArchitecture(str, Enum):
    """Categorization of inference serving runtime architectures."""

    OLLAMA_SINGLE_SLOT = "OLLAMA_SINGLE_SLOT"
    OLLAMA_CLIENT_QUEUED = "OLLAMA_CLIENT_QUEUED"
    LLAMA_CPP_PARALLEL_2_SLOTS = "LLAMA_CPP_PARALLEL_2_SLOTS"
    LLAMA_CPP_PARALLEL_4_SLOTS = "LLAMA_CPP_PARALLEL_4_SLOTS"


class CapabilityVerificationStatus(str, Enum):
    """Verification status of a serving capability."""

    VERIFIED = "VERIFIED"
    ADVERTISED_UNVERIFIED = "ADVERTISED_UNVERIFIED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNKNOWN = "UNKNOWN"


class ServingCapabilityMatrix(BaseModel):
    """Versioned specification of serving capabilities per runtime and model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_name: str = Field(description="Name of the serving runtime (e.g. Ollama, llama.cpp server)")
    runtime_version: str = Field(description="Version string of the runtime")
    architecture_type: ServingRuntimeArchitecture
    model_id: str
    quantization: str
    available_cpu_threads: int = Field(default=8)
    configurable_slots: int = Field(description="Number of parallel execution slots configured")
    parallel_request_support: CapabilityVerificationStatus
    batching_support: CapabilityVerificationStatus
    continuous_batching_support: CapabilityVerificationStatus
    kv_cache_memory_behavior: str = Field(description="Context and KV cache allocation mechanics")
    model_residency_behavior: str = Field(description="RAM residency and model swapping mechanics")
    evidence_source: str = Field(description="Evidence provenance (e.g. DIRECT_MEASURED, ADVERTISED)")
    verification_notes: str = Field(description="Detailed notes explaining observed behavior vs documentation")


class ServingObservation(BaseModel):
    """Empirical measurement of an inference request under a specific serving architecture."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str
    session_id: str = Field(default="session_27")
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    runtime_architecture: ServingRuntimeArchitecture
    model_id: str
    num_threads: int = Field(default=8)

    # Concurrency and Slots
    client_concurrency: int = Field(ge=1, description="Number of concurrent client requests dispatched")
    server_slots: int = Field(ge=1, description="Configured parallel execution slots on server")
    effective_server_concurrency: float = Field(
        ge=0.0, description="Measured server execution overlap ratio (1.0 = full overlap, 0.0 = completely serialized)"
    )

    # Workload parameters
    workload_class: str
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    latency_sla_ms: float = Field(gt=0.0)

    # Monotonic timestamps (seconds)
    client_dispatch_mono_s: float
    client_completion_mono_s: float
    server_start_mono_s: float | None = None
    server_completion_mono_s: float | None = None

    # Latency breakdown (ms)
    client_duration_ms: float = Field(ge=0.0)
    server_prefill_ms: float = Field(ge=0.0, default=0.0)
    server_decode_ms: float = Field(ge=0.0, default=0.0)
    queue_wait_ms: float = Field(ge=0.0, default=0.0)
    ttft_ms: float = Field(ge=0.0, default=0.0)
    decode_tokens_per_second: float = Field(ge=0.0, default=0.0)

    # Parallelism Telemetry
    overlapping_execution: bool = Field(description="Whether request overlapped in wall-clock time with another request")
    overlap_duration_ms: float = Field(ge=0.0, default=0.0)
    slot_id_used: int | None = Field(default=None)

    # Resource Telemetry
    cpu_utilization_pct: float = Field(ge=0.0, default=0.0)
    memory_rss_mb: float = Field(ge=0.0, default=0.0)

    # SLA evaluation
    sla_met: bool


class ServingArchitectureComparison(BaseModel):
    """Direct comparison between baseline (Ollama single-slot) and target (parallel continuous batching)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str
    model_id: str
    workload_description: str
    client_concurrency: int

    baseline_architecture: ServingRuntimeArchitecture
    target_architecture: ServingRuntimeArchitecture

    # Latency percentiles (ms)
    baseline_p50_ms: float
    baseline_p90_ms: float
    baseline_p95_ms: float
    target_p50_ms: float
    target_p90_ms: float
    target_p95_ms: float
    latency_p95_speedup_factor: float = Field(description="baseline_p95 / target_p95")

    # Throughput (req/s)
    baseline_throughput_rps: float
    target_throughput_rps: float
    throughput_gain_pct: float

    # Queue wait & HoL
    baseline_mean_queue_wait_ms: float
    target_mean_queue_wait_ms: float
    queue_wait_reduction_ms: float
    hol_blocking_eliminated: bool

    # SLA attainment
    baseline_sla_compliance_pct: float
    target_sla_compliance_pct: float

    # Memory trade-off
    baseline_memory_mb: float
    target_memory_mb: float
    memory_overhead_mb: float

    scientific_provenance_summary: str


class ServingArchitectureRecommendation(BaseModel):
    """Architecture recommendation based on workload, SLA, arrival rate, and hardware constraints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recommendation_id: str
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    recommended_architecture: ServingRuntimeArchitecture
    recommended_slots: int
    recommended_runtime_name: str
    expected_throughput_rps: float
    expected_p95_latency_ms: float
    expected_sla_compliance_pct: float
    evidence_tier: str
    rationale: str
    tradeoffs: list[str]
    actionable_setup_guidance: str


class HeadOfLineComparisonObservation(BaseModel):
    """Matched comparison of Head-of-Line blocking under Ollama vs llama-server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    model_id: str
    antecedent_workload: str
    consequent_workload: str
    consequent_sla_ms: float

    # Ollama Single-Slot Baseline
    ollama_consequent_wait_ms: float
    ollama_consequent_e2e_ms: float
    ollama_sla_met: bool

    # llama-server Continuous Batching
    llama_server_consequent_wait_ms: float
    llama_server_consequent_e2e_ms: float
    llama_server_sla_met: bool

    # Deltas
    queue_wait_reduction_ms: float
    latency_reduction_factor: float
    sla_rescued_by_parallelism: bool


class Session27Manifest(BaseModel):
    """Complete versioned experiment manifest for Session 27."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_id: str = Field(default="manifest-session-27-serving-comparison")
    session_id: str = Field(default="session_27")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    hardware_id: str = Field(default="cpu-intel-i7-14700")
    total_benchmark_duration_s: float
    capability_matrix: list[ServingCapabilityMatrix]
    observations_count: int
    comparisons: list[ServingArchitectureComparison]
    hol_comparisons: list[HeadOfLineComparisonObservation]
    raw_observations_file: str
