"""Capacity planning, queueing models, and adaptive request admission protocol schemas.

Defines schemas for service time distributions, queue observations, capacity curves,
queueing model parameters, and SLA-aware admission decisions for Session 25.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArrivalProcess(str, Enum):
    """Arrival process pattern for capacity and queueing benchmarks."""

    CLOSED_LOOP = "CLOSED_LOOP"
    OPEN_LOOP_SCHEDULED = "OPEN_LOOP_SCHEDULED"
    POISSON = "POISSON"
    BURST = "BURST"


class CapacityEvidenceTier(str, Enum):
    """Scientific evidence tier supporting capacity and queue forecasts."""

    DIRECT_MEASURED = "DIRECT_MEASURED"
    EMPIRICALLY_INTERPOLATED = "EMPIRICALLY_INTERPOLATED"
    EXPLORATORY_EXTRAPOLATION = "EXPLORATORY_EXTRAPOLATION"
    UNKNOWN_CAPACITY = "UNKNOWN_CAPACITY"


class AdmissionDecision(str, Enum):
    """SLA-aware request admission decision."""

    ACCEPT = "ACCEPT"
    QUEUE_WITH_BOUNDED_WAIT = "QUEUE_WITH_BOUNDED_WAIT"
    REJECT_OR_DEFER = "REJECT_OR_DEFER"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class QueueingModelType(str, Enum):
    """Analytical or empirical queueing model formulation."""

    EMPIRICAL_INTERPOLATION = "EMPIRICAL_INTERPOLATION"
    M_M_1_APPROX = "M_M_1_APPROX"
    M_G_1_APPROX = "M_G_1_APPROX"


class ServiceTimeDistribution(BaseModel):
    """Statistical distribution of isolated service times for a model/workload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_count: int = Field(ge=1, description="Sample count of measured requests")
    mean_service_ms: float = Field(ge=0.0, description="Mean total service time E[S] in ms")
    variance_service_ms: float = Field(ge=0.0, description="Variance of service time Var[S]")
    std_service_ms: float = Field(ge=0.0, description="Standard deviation of service time in ms")
    coefficient_of_variation: float = Field(
        ge=0.0, description="Coefficient of variation Cv = std / mean"
    )
    median_service_ms: float = Field(ge=0.0, description="Median service time in ms")
    p90_service_ms: float = Field(ge=0.0, description="90th percentile service time in ms")
    p95_service_ms: float = Field(ge=0.0, description="95th percentile service time in ms")
    min_service_ms: float = Field(ge=0.0, description="Minimum observed service time in ms")
    max_service_ms: float = Field(ge=0.0, description="Maximum observed service time in ms")
    mean_prefill_ms: float = Field(ge=0.0, description="Mean server prompt evaluation time in ms")
    mean_decode_ms: float = Field(ge=0.0, description="Mean server token generation time in ms")
    mean_output_tokens: float = Field(ge=0.0, description="Mean evaluated output token count")


class CapacityObservation(BaseModel):
    """Per-request observation during load and queueing benchmarks."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(description="Unique observation ID")
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_id: str = Field(description="Model identifier")
    num_thread: int = Field(description="CPU thread allocation")
    workload_id: str = Field(description="Workload identifier")
    arrival_process: ArrivalProcess = Field(description="Arrival pattern")
    scheduled_arrival_ms: float = Field(description="Monotonic scheduled arrival timestamp (ms)")
    actual_dispatch_ms: float = Field(description="Monotonic actual dispatch timestamp (ms)")
    dispatch_delay_ms: float = Field(ge=0.0, description="Client queue slip (actual - scheduled)")
    client_e2e_ms: float = Field(ge=0.0, description="Total client wall-clock latency (ms)")
    server_prefill_ms: float = Field(default=0.0, ge=0.0)
    server_decode_ms: float = Field(default=0.0, ge=0.0)
    estimated_queue_wait_ms: float = Field(
        default=0.0, ge=0.0, description="Estimated server queue waiting duration"
    )
    output_tokens: int = Field(ge=0)
    concurrency_level: int = Field(ge=1)
    in_flight_count: int = Field(ge=0)
    status: str = Field(default="SUCCESS")
    error_message: str | None = Field(default=None)


class CapacityCurvePoint(BaseModel):
    """Performance and queue metrics measured or forecasted at an offered arrival rate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    offered_rps: float = Field(ge=0.0, description="Offered arrival rate lambda (req/s)")
    completed_rps: float = Field(ge=0.0, description="Completed throughput (req/s)")
    estimated_utilization: float = Field(
        ge=0.0, description="Server utilization rho = lambda * E[S]"
    )
    median_e2e_ms: float = Field(ge=0.0, description="Median client end-to-end latency")
    p90_e2e_ms: float = Field(ge=0.0, description="90th percentile latency")
    p95_e2e_ms: float = Field(ge=0.0, description="95th percentile latency")
    mean_queue_delay_ms: float = Field(ge=0.0, description="Mean waiting time before execution")
    sla_compliance_pct: float = Field(
        ge=0.0, le=100.0, description="Percentage of requests meeting SLA"
    )
    queue_buildup_detected: bool = Field(description="Whether backlog grows monotonically")
    failure_count: int = Field(ge=0, description="Number of timeout or error requests")


class WorkloadCapacityReport(BaseModel):
    """Comprehensive capacity plan for a specific model and workload configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(description="Unique capacity report ID")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_id: str = Field(description="Model identifier")
    num_thread: int = Field(description="CPU thread count")
    workload_id: str = Field(description="Workload task identifier")
    service_distribution: ServiceTimeDistribution = Field(
        description="Isolated service time statistics"
    )
    saturation_knee_rps: float = Field(
        ge=0.0, description="Offered arrival rate where queue begins rapid divergence"
    )
    max_sustainable_rps: float = Field(
        ge=0.0, description="Maximum sustainable completed throughput (1 / E[S])"
    )
    max_sustainable_rpm: float = Field(ge=0.0, description="Maximum requests per minute")
    safe_operating_rps: float = Field(
        ge=0.0, description="Conservative safe offered load target (typically ~70-80% knee)"
    )
    capacity_curve: list[CapacityCurvePoint] = Field(
        default_factory=list, description="Performance trajectory across arrival rates"
    )
    evidence_tier: CapacityEvidenceTier = Field(description="Evidence provenance tier")
    queueing_model_type: QueueingModelType = Field(description="Underlying queue formulation")
    assumptions: list[str] = Field(
        default_factory=list, description="Explicit modeling assumptions and caveats"
    )


class AdmissionRequest(BaseModel):
    """Incoming request evaluated for admission against current capacity and queue depth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(default="qwen2.5:0.5b")
    num_thread: int = Field(default=8)
    prompt_tokens: int = Field(gt=0)
    expected_output_tokens: int = Field(gt=0)
    latency_sla_ms: float | None = Field(default=None, gt=0.0)
    current_in_flight: int = Field(default=0, ge=0)
    current_queue_depth: int = Field(default=0, ge=0)
    require_code_execution_pass: bool = Field(default=False)


class AdmissionResult(BaseModel):
    """SLA-aware admission recommendation with queue delay forecast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: AdmissionDecision = Field(description="Admission outcome decision")
    model_id: str = Field(description="Target model")
    num_thread: int = Field(description="Configured threads")
    predicted_queue_delay_ms: float = Field(ge=0.0, description="Forecasted wait time in queue")
    predicted_service_time_ms: float = Field(ge=0.0, description="Forecasted isolated service time")
    predicted_total_latency_ms: float = Field(ge=0.0, description="Total predicted response time")
    sla_compliance_probability: float = Field(
        ge=0.0, le=1.0, description="Estimated probability of meeting SLA"
    )
    current_server_utilization: float = Field(
        ge=0.0, description="Current estimated server utilization rho"
    )
    capacity_headroom_pct: float = Field(description="Headroom percentage before saturation")
    rationale: str = Field(description="Human-readable decision rationale")
    evidence_tier: CapacityEvidenceTier = Field(description="Evidence provenance tier")
    unverified_code_warning: bool = Field(
        default=False, description="Flagged if code correctness is unverified"
    )


class CapacityExperimentManifest(BaseModel):
    """Versioned experiment manifest for Session 25 capacity benchmarks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_id: str = Field(description="Unique capacity manifest ID")
    session_id: str = Field(default="session_25")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    hardware_catalog_id: str = Field(default="cpu-intel-i7-14700")
    total_active_seconds: float = Field(ge=0.0)
    stage_a_service_distributions: dict[str, Any] = Field(default_factory=dict)
    stage_b_rate_curves: dict[str, Any] = Field(default_factory=dict)
    stage_c_concurrency_curves: dict[str, Any] = Field(default_factory=dict)
    observations: list[CapacityObservation] = Field(default_factory=list)
