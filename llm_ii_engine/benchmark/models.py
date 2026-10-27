"""Domain schemas for benchmark metrics, memory watermarks, and execution results."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ClockSource,
    EngineType,
    ExecutionMode,
    ModelResidencyState,
    PrefixRelationshipType,
    ProvenanceType,
    TokenCountSource,
)
from llm_ii_engine.core.schemas import ServerReportedStats


class MemoryWatermarkResult(BaseModel):
    """Memory consumption sampled during benchmark execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    peak_process_rss_mb: float | None = Field(
        default=None, description="Peak resident set size of the benchmark process in MiB"
    )
    peak_gpu_vram_mb: float | None = Field(
        default=None, description="Peak GPU VRAM allocated across sampled devices in MiB"
    )
    sample_count: int = Field(
        default=0, ge=0, description="Total number of memory samples recorded"
    )
    sample_interval_ms: float = Field(
        default=100.0, gt=0.0, description="Nominal sampling period in milliseconds"
    )
    telemetry_source: str = Field(
        description="Telemetry provider (e.g. 'mock', 'adapter', 'unavailable')"
    )
    is_telemetry_available: bool = Field(
        description="Whether memory telemetry was actively available during the benchmark"
    )
    notes: str | None = Field(
        default=None, description="Diagnostic notes or reason if telemetry unavailable"
    )
    device_index: int | None = Field(
        default=None, description="Specific GPU device index sampled, if applicable"
    )
    device_name: str | None = Field(
        default=None, description="Specific GPU device name sampled, if applicable"
    )


class RequestBenchmarkResult(BaseModel):
    """Comprehensive measurement outcomes for an individual inference request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, description="Unique identifier of the request")
    success: bool = Field(
        description="True if request finished successfully without unhandled errors"
    )
    error_message: str | None = Field(default=None, description="Error details if success is False")
    prompt_tokens: int | None = Field(
        default=None, ge=0, description="Prompt token count. None if unknown or unobserved."
    )
    requested_output_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Requested output token limit (e.g. max_tokens / num_predict from WorkloadSpec)",
    )
    output_tokens: int = Field(ge=0, description="Actual output tokens decoded and received")

    # Timing and latencies (milliseconds)
    ttft_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time-to-First-Token in ms. Populated ONLY when true discrete token events are observed; None if token boundaries unavailable.",
    )
    ttfc_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time-to-First-Chunk in ms. Populated when streaming text chunks. Distinct from true token-level TTFT.",
    )
    tpot_ms: float | None = Field(
        default=None, ge=0.0, description="Time-Per-Output-Token in ms. None if output_tokens < 2"
    )
    duration_ms: float = Field(
        ge=0.0, description="Total elapsed generation duration in milliseconds"
    )

    # Per-request throughput
    output_tokens_per_second: float | None = Field(
        default=None, ge=0.0, description="Output tokens / total generation duration (seconds)"
    )

    # Inter-token latency series and percentiles (ms)
    inter_token_latencies_ms: list[float] = Field(
        default_factory=list, description="Array of discrete ITL deltas in milliseconds"
    )
    itl_p50_ms: float | None = Field(default=None, ge=0.0, description="Median ITL (p50)")
    itl_p90_ms: float | None = Field(default=None, ge=0.0, description="90th percentile ITL")
    itl_p95_ms: float | None = Field(default=None, ge=0.0, description="95th percentile ITL")
    itl_p99_ms: float | None = Field(default=None, ge=0.0, description="99th percentile ITL")

    # Text-chunk streaming metrics (when individual tokens are not exposed by the engine)
    chunk_count: int = Field(
        default=0, ge=0, description="Total number of text chunks received from stream"
    )
    chunk_latencies_ms: list[float] = Field(
        default_factory=list, description="Array of discrete chunk inter-arrival latencies in ms"
    )
    chunk_latency_p50_ms: float | None = Field(
        default=None, ge=0.0, description="Median chunk inter-arrival latency in ms"
    )
    chunk_latency_p90_ms: float | None = Field(
        default=None, ge=0.0, description="90th percentile chunk inter-arrival latency in ms"
    )
    chunk_latency_p95_ms: float | None = Field(
        default=None, ge=0.0, description="95th percentile chunk inter-arrival latency in ms"
    )
    chunk_latency_p99_ms: float | None = Field(
        default=None, ge=0.0, description="99th percentile chunk inter-arrival latency in ms"
    )

    # Server-reported statistics and token count attribution
    server_stats: ServerReportedStats | None = Field(
        default=None, description="Engine/server-reported timing and token statistics"
    )
    token_count_source: TokenCountSource = Field(
        default=TokenCountSource.TOKEN_EVENTS,
        description="Source of output token count (token_events, server_reported, requested, unknown)",
    )
    prompt_token_count_source: TokenCountSource = Field(
        default=TokenCountSource.REQUESTED,
        description="Source of prompt token count (token_events, server_reported, requested, unknown)",
    )

    # Provenance and metadata
    clock_source: ClockSource = Field(description="Monotonic real clock vs synthetic virtual clock")
    execution_mode: ExecutionMode = Field(
        description="REAL hardware execution vs SIMULATED mock run"
    )
    provenance: ProvenanceType = Field(description="Data provenance: MEASURED or HEURISTIC")
    is_warmup: bool = Field(
        default=False, description="True if request was executed during warm-up phase"
    )
    incomplete_reason: str | None = Field(
        default=None, description="Reason if stream terminated prematurely without completion"
    )

    # Cache State & Model Residency Telemetry (Session 15)
    model_residency_state: ModelResidencyState | None = Field(
        default=None, description="Model memory residency state: cold, warm, or unknown"
    )
    cache_state: CacheStateClassification | None = Field(
        default=None, description="Prompt KV cache state classification: cold_prefix, warm_prefix, partial_prefix, or unknown"
    )
    predecessor_request_id: str | None = Field(
        default=None, description="Request ID of immediately preceding request on the inference engine"
    )
    prefix_relationship: PrefixRelationshipType | None = Field(
        default=None, description="Prefix relationship to immediately preceding request"
    )

    @property
    def client_e2e_duration_ms(self) -> float:
        """Client elapsed monotonic duration in milliseconds."""
        return self.duration_ms

    @property
    def client_e2e_output_throughput_tok_s(self) -> float | None:
        """Client end-to-end output throughput (output tokens / client duration)."""
        return self.output_tokens_per_second

    @property
    def server_total_duration_ms(self) -> float | None:
        """Total execution duration reported by server in ms."""
        return self.server_stats.total_duration_ms if self.server_stats else None

    @property
    def server_load_duration_ms(self) -> float | None:
        """Model loading duration reported by server in ms."""
        return self.server_stats.load_duration_ms if self.server_stats else None

    @property
    def server_prefill_duration_ms(self) -> float | None:
        """Prefill / prompt evaluation duration reported by server in ms."""
        return self.server_stats.prompt_duration_ms if self.server_stats else None

    @property
    def server_decode_duration_ms(self) -> float | None:
        """Decode / token generation duration reported by server in ms."""
        return self.server_stats.eval_duration_ms if self.server_stats else None

    @property
    def server_decode_throughput_tok_s(self) -> float | None:
        """Server decode throughput (output tokens / server decode duration) in tok/s."""
        return self.server_stats.eval_tokens_per_second if self.server_stats else None


class BenchmarkSuiteResult(BaseModel):
    """Aggregate benchmark results and operational summary for a completed experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite_id: str = Field(
        min_length=1, description="Unique identifier for the benchmark experiment"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when benchmark execution initiated",
    )
    status: str = Field(
        description="Execution outcome: 'COMPLETED', 'FAILED', 'CANCELLED', 'PARTIAL'"
    )
    model_id: str = Field(min_length=1, description="Model evaluated in the benchmark")
    engine_type: EngineType = Field(description="Inference engine adapter tested")
    concurrency: int = Field(ge=1, description="Maximum concurrent client requests permitted")

    # Counts
    total_requests: int = Field(ge=0, description="Number of measured requests scheduled")
    warmup_requests: int = Field(ge=0, description="Number of unmeasured warm-up requests executed")
    successful_requests: int = Field(
        ge=0, description="Number of measured requests completed successfully"
    )
    failed_requests: int = Field(
        ge=0, description="Number of measured requests that encountered errors"
    )

    # Wall clock duration
    wall_time_seconds: float = Field(
        ge=0.0, description="Total elapsed benchmark execution wall time in seconds"
    )

    # Aggregate Serving Throughputs
    aggregate_tokens_per_second: float | None = Field(
        default=None,
        ge=0.0,
        description="Total (prompt + output) tokens across all requests / wall time",
    )
    aggregate_output_tokens_per_second: float | None = Field(
        default=None, ge=0.0, description="Total output tokens across all requests / wall time"
    )
    request_throughput_per_second: float | None = Field(
        default=None, ge=0.0, description="Successful requests completed / wall time"
    )

    # Aggregate Latency Distributions (ms) - True Token-Level
    ttft_p50_ms: float | None = Field(default=None, ge=0.0)
    ttft_p90_ms: float | None = Field(default=None, ge=0.0)
    ttft_p95_ms: float | None = Field(default=None, ge=0.0)
    ttft_p99_ms: float | None = Field(default=None, ge=0.0)

    # Aggregate Text-Chunk Latency Distributions (ms) - Stream Chunk Level
    ttfc_p50_ms: float | None = Field(
        default=None, ge=0.0, description="Median Time-to-First-Chunk (p50) in ms"
    )
    ttfc_p90_ms: float | None = Field(
        default=None, ge=0.0, description="90th percentile Time-to-First-Chunk in ms"
    )
    ttfc_p95_ms: float | None = Field(
        default=None, ge=0.0, description="95th percentile Time-to-First-Chunk in ms"
    )
    ttfc_p99_ms: float | None = Field(
        default=None, ge=0.0, description="99th percentile Time-to-First-Chunk in ms"
    )
    chunk_latency_p50_ms: float | None = Field(
        default=None, ge=0.0, description="Median chunk inter-arrival latency in ms"
    )

    tpot_mean_ms: float | None = Field(default=None, ge=0.0)
    itl_p50_ms: float | None = Field(default=None, ge=0.0)
    itl_p90_ms: float | None = Field(default=None, ge=0.0)
    itl_p95_ms: float | None = Field(default=None, ge=0.0)
    itl_p99_ms: float | None = Field(default=None, ge=0.0)

    # Memory and Telemetry
    memory_watermark: MemoryWatermarkResult | None = Field(
        default=None, description="Peak memory consumption during benchmark execution"
    )

    # Provenance
    execution_mode: ExecutionMode = Field(description="REAL vs SIMULATED execution mode")
    clock_source: ClockSource = Field(description="Timing source used for elapsed calculations")
    provenance: ProvenanceType = Field(
        default=ProvenanceType.MEASURED, description="Provenance classification"
    )

    # Results collection
    request_results: list[RequestBenchmarkResult] = Field(
        default_factory=list, description="Per-request results for measured iterations"
    )
    warmup_results: list[RequestBenchmarkResult] = Field(
        default_factory=list, description="Per-request results for warm-up iterations"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Operational warnings encountered during run"
    )
