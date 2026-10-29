"""Protocol schemas and domain models for Session 26 mixed-workload capacity and adaptive admission.

Defines schemas for:
- Per-request arrival accounting with monotonic timestamps and terminal states.
- Mixed workload profiles (Interactive Dominant, Balanced, Generation Heavy, Bursty).
- Multi-class service time moments (first and second moments, mixed variance, Cv).
- Head-of-line (HoL) blocking and model-switching metrics.
- Adaptive admission policies (No Control, Static Utilization, Workload-Aware).
- Policy evaluation metrics (admitted vs. all-offered SLA attainment, trade-off utility, fairness).
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RequestTerminalStatus(str, Enum):
    """Mutually exclusive terminal outcomes for an offered request."""

    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"
    CANCELED = "CANCELED"
    OUTSTANDING = "OUTSTANDING"


class WorkloadClass(str, Enum):
    """Classification of inference workload task types."""

    INTERACTIVE_QA = "INTERACTIVE_QA"
    STRUCTURED_JSON = "STRUCTURED_JSON"
    LONG_FORM_GENERATION = "LONG_FORM_GENERATION"
    LONG_CONTEXT_RETRIEVAL = "LONG_CONTEXT_RETRIEVAL"


class WorkloadProfileType(str, Enum):
    """Predefined versioned mixed-workload traffic compositions."""

    PROFILE_A_INTERACTIVE_DOMINANT = "PROFILE_A_INTERACTIVE_DOMINANT"
    PROFILE_B_BALANCED = "PROFILE_B_BALANCED"
    PROFILE_C_GENERATION_HEAVY = "PROFILE_C_GENERATION_HEAVY"
    PROFILE_D_BURSTY = "PROFILE_D_BURSTY"


class AdmissionPolicyType(str, Enum):
    """Inference request admission control policy."""

    POLICY_A_NO_CONTROL = "POLICY_A_NO_CONTROL"
    POLICY_B_STATIC_UTILIZATION = "POLICY_B_STATIC_UTILIZATION"
    POLICY_C_WORKLOAD_AWARE = "POLICY_C_WORKLOAD_AWARE"


class ModelResidencyState(str, Enum):
    """Model memory residency state prior to request execution."""

    RESIDENT = "RESIDENT"
    COLD_SWITCH = "COLD_SWITCH"
    UNKNOWN = "UNKNOWN"


class MixedWorkloadItem(BaseModel):
    """Individual workload specification within a mixed-workload profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workload_class: WorkloadClass
    weight: float = Field(ge=0.0, le=1.0, description="Fraction of traffic composition")
    prompt_tokens: int = Field(gt=0)
    expected_output_tokens: int = Field(gt=0)
    latency_sla_ms: float = Field(gt=0.0, description="Workload-specific latency SLA")


class MixedWorkloadProfile(BaseModel):
    """Versioned composition profile representing heterogeneous traffic streams."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str
    name: str
    profile_type: WorkloadProfileType
    description: str
    items: list[MixedWorkloadItem]
    burst_idle_interval_s: float | None = Field(
        default=None, description="Idle interval between bursts for bursty profiles"
    )
    burst_batch_size: int | None = Field(
        default=None, description="Number of requests dispatched in a burst"
    )


class MixedRequestAccountingRecord(BaseModel):
    """Fine-grained, monotonic timestamp record for every offered request."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(description="Unique request identifier")
    session_id: str = Field(default="session_26")
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_id: str = Field(description="Target model identifier")
    num_thread: int = Field(default=8)
    workload_class: WorkloadClass
    latency_sla_ms: float = Field(gt=0.0)

    # Monotonic timestamps for accurate elapsed-time calculations
    scheduled_arrival_mono_s: float = Field(description="Scheduled monotonic arrival time")
    actual_enqueue_mono_s: float = Field(description="Monotonic time request entered queue")
    dispatch_mono_s: float | None = Field(default=None, description="Monotonic dispatch time")
    first_response_mono_s: float | None = Field(default=None, description="Monotonic TTFT time")
    completion_mono_s: float | None = Field(default=None, description="Monotonic completion time")

    # Queue state at arrival
    queue_depth_at_arrival: int = Field(ge=0, default=0)
    in_flight_at_arrival: int = Field(ge=0, default=0)
    residency_before: ModelResidencyState = Field(default=ModelResidencyState.UNKNOWN)

    # Terminal accounting
    admitted: bool = Field(description="Whether admission controller admitted request")
    admission_policy: AdmissionPolicyType = Field(default=AdmissionPolicyType.POLICY_A_NO_CONTROL)
    terminal_status: RequestTerminalStatus = Field(description="Terminal accounting state")
    error_message: str | None = Field(default=None)

    # Timing metrics (ms)
    queue_delay_ms: float = Field(ge=0.0, default=0.0, description="Actual wait before execution")
    client_e2e_ms: float | None = Field(default=None, ge=0.0, description="Execution turnaround")
    total_turnaround_ms: float | None = Field(
        default=None, ge=0.0, description="Total client duration from arrival to completion"
    )
    ttft_ms: float | None = Field(default=None, ge=0.0)
    server_load_ms: float = Field(default=0.0, ge=0.0)
    server_prefill_ms: float = Field(default=0.0, ge=0.0)
    server_decode_ms: float = Field(default=0.0, ge=0.0)
    output_tokens: int = Field(default=0, ge=0)

    # SLA evaluation
    sla_met: bool | None = Field(default=None, description="Whether SLA was satisfied")


class MixedServiceMoments(BaseModel):
    """Analytical statistical moments for a heterogeneous workload mixture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean_service_ms: float = Field(ge=0.0, description="First moment E[S_mix]")
    second_moment_ms2: float = Field(ge=0.0, description="Second moment E[S_mix^2]")
    variance_service_ms2: float = Field(ge=0.0, description="Mixed variance Var[S_mix]")
    std_service_ms: float = Field(ge=0.0, description="Mixed standard deviation")
    coefficient_of_variation: float = Field(ge=0.0, description="Mixed Cv = std / mean")
    per_class_moments: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="First and second moments per workload class"
    )


class HeadOfLineBlockingObservation(BaseModel):
    """Observation quantifying head-of-line blocking penalty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_id: str
    scenario_description: str
    antecedent_workload: WorkloadClass = Field(description="Preceding long workload in queue")
    consequent_workload: WorkloadClass = Field(description="Succeeding short workload in queue")
    antecedent_duration_ms: float
    isolated_consequent_duration_ms: float
    blocked_consequent_duration_ms: float
    additional_queue_delay_ms: float = Field(
        description="Extra wait time imposed by antecedent request"
    )
    sla_missed_due_to_blocking: bool


class ModelSwitchingObservation(BaseModel):
    """Observation quantifying cold model-switching latency and memory overhead."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    from_model_id: str
    to_model_id: str
    from_model_resident_before: bool
    actual_load_duration_ms: float = Field(
        description="Cold load / weight paging duration reported by server"
    )
    client_switch_penalty_ms: float = Field(
        description="Turnaround time delta compared to warm execution"
    )
    warm_execution_duration_ms: float
    cold_execution_duration_ms: float


class PolicyEvaluationReport(BaseModel):
    """Comprehensive policy comparison metrics for mixed-workload admission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_type: AdmissionPolicyType
    profile_type: WorkloadProfileType
    offered_rps: float
    total_offered: int
    total_admitted: int
    total_completed: int
    total_rejected: int
    total_timed_out: int
    total_canceled: int
    total_outstanding: int

    # Accounting Invariant Check: offered == completed + rejected + timed_out + canceled + outstanding
    accounting_invariant_holds: bool

    # SLA Metrics
    all_offered_sla_attainment_pct: float = Field(
        ge=0.0, le=100.0, description="SLA compliance over all offered requests"
    )
    admitted_sla_attainment_pct: float = Field(
        ge=0.0, le=100.0, description="SLA compliance among admitted requests"
    )
    false_feasible_count: int = Field(
        ge=0, description="Admitted requests that failed SLA (false feasible)"
    )
    unnecessarily_rejected_count: int = Field(
        ge=0, description="Rejected requests that could have met SLA"
    )

    # Latency & Queue Statistics (completed requests only)
    completed_p50_e2e_ms: float
    completed_p90_e2e_ms: float
    completed_p95_e2e_ms: float
    mean_queue_delay_ms: float
    max_queue_depth: int

    # Throughput & Operational Metrics
    completed_throughput_rps: float
    model_switch_count: int
    per_class_sla_attainment: dict[str, float] = Field(
        default_factory=dict, description="SLA compliance percentage per workload class"
    )
    workload_fairness_index: float = Field(
        ge=0.0, le=1.0, description="Jain's fairness index across workload classes"
    )
    tradeoff_utility_score: float = Field(
        description="Net utility balancing admitted completions against SLA breaches and rejections"
    )


class MixedWorkloadManifest(BaseModel):
    """Complete versioned experiment manifest for Session 26."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_id: str
    session_id: str = Field(default="session_26")
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    hardware_id: str = Field(default="cpu-intel-i7-14700")
    total_benchmark_duration_s: float
    profiles: list[MixedWorkloadProfile]
    policy_reports: list[PolicyEvaluationReport] = Field(default_factory=list)
    hol_observations: list[HeadOfLineBlockingObservation] = Field(default_factory=list)
    model_switch_observations: list[ModelSwitchingObservation] = Field(default_factory=list)
    total_observations_count: int
