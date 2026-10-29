"""Hardware-aware runtime optimization protocol and schema specifications for Session 24.

Defines schemas for CPU thread tuning, concurrency scaling, context controls,
and empirical measurement provenance without confusing text-chunk timing with discrete token TTFT.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkstationHardwareProfile(BaseModel):
    """Verified hardware specifications for Intel i7-14700 workstation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_model: str = Field(
        default="Intel(R) Core(TM) i7-14700",
        description="CPU processor model name",
    )
    physical_cores: int = Field(
        default=20,
        description="Total physical cores (8 P-cores + 12 E-cores)",
    )
    performance_cores: int = Field(
        default=8,
        description="Performance Cores with Hyper-Threading (16 logical threads)",
    )
    efficient_cores: int = Field(
        default=12,
        description="Efficient Cores without Hyper-Threading (12 logical threads)",
    )
    logical_processors: int = Field(
        default=28,
        description="Total logical CPU processors exposed to OS scheduler",
    )
    total_ram_gb: float = Field(
        default=31.69,
        description="Total visible physical RAM in gigabytes",
    )
    os_caption: str = Field(
        default="Microsoft Windows 11 Pro",
        description="Host operating system name and edition",
    )
    execution_backend: str = Field(
        default="cpu_ggml_avx2",
        description="Ollama/llama.cpp CPU compute backend",
    )


class RuntimeConfigurationSpec(BaseModel):
    """Requested runtime options for Ollama inference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_thread: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Requested CPU thread allocation for inference",
    )
    num_ctx: int = Field(
        default=4096,
        ge=512,
        le=131072,
        description="Requested context window size in tokens",
    )
    concurrency: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Client concurrent request dispatch level",
    )
    num_batch: int = Field(
        default=512,
        ge=1,
        description="Prompt batching chunk size in tokens",
    )
    keep_alive: str = Field(
        default="5m",
        description="Model memory residency duration window",
    )


class RuntimeObservation(BaseModel):
    """Empirical observation capturing client wall-clock, server timings, and provenance."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(description="Unique observation identifier")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Observation timestamp in ISO 8601 UTC",
    )
    model_id: str = Field(
        description="Target model identifier, e.g. 'qwen2.5:0.5b'"
    )
    model_digest: str = Field(
        description="SHA256 weight digest from Ollama tags"
    )
    quantization: str = Field(
        description="Quantization format, e.g. 'Q4_K_M', 'Q8_0'"
    )
    hardware: WorkstationHardwareProfile = Field(
        default_factory=WorkstationHardwareProfile,
        description="Verified workstation hardware fingerprint",
    )
    requested_config: RuntimeConfigurationSpec = Field(
        description="Requested runtime configuration options"
    )
    task_id: str = Field(description="Workload task identifier")
    task_name: str = Field(description="Workload task descriptive name")
    stage: str = Field(
        description="Benchmark stage: STAGE_A_THREADS, STAGE_B_CONCURRENCY, STAGE_C_CONTEXT, PROSPECTIVE_VALIDATION"
    )

    # Token counts
    prompt_tokens: int = Field(
        ge=0, description="Server-evaluated prompt token count"
    )
    generated_tokens: int = Field(
        ge=0, description="Server-evaluated generated token count"
    )

    # Durations (in milliseconds)
    client_e2e_duration_ms: float = Field(
        ge=0.0, description="Client monotonic wall-clock latency"
    )
    time_to_first_chunk_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Client monotonic latency to first streamed HTTP text chunk (NOT true discrete token TTFT)",
    )
    server_load_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Server model loading duration in ms",
    )
    server_prefill_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Server prompt evaluation duration in ms",
    )
    server_decode_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Server token generation duration in ms",
    )

    # Derived throughputs
    decode_tokens_per_second: float = Field(
        ge=0.0,
        description="Server decode throughput: generated_tokens / (server_decode_duration_ms / 1000)",
    )
    client_tokens_per_second: float = Field(
        ge=0.0,
        description="End-to-end client throughput: generated_tokens / (client_e2e_duration_ms / 1000)",
    )
    aggregate_tokens_per_second: float | None = Field(
        default=None,
        description="Aggregate tokens per second during concurrent batch runs",
    )
    queuing_delay_estimate_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated queuing time before server execution began",
    )

    # Execution conditions
    model_residency: str = Field(
        default="WARM", description="'WARM' or 'COLD' model residency"
    )
    prefix_cache_state: str = Field(
        default="COLD_PREFIX", description="'COLD_PREFIX' or 'WARM_PREFIX'"
    )
    sequence_index: int = Field(
        ge=0, description="Sequential dispatch index in session"
    )
    repetition_index: int = Field(
        ge=1, description="Repetition index for identical configuration"
    )
    status: str = Field(
        default="SUCCESS", description="'SUCCESS', 'TIMEOUT', or 'ERROR'"
    )
    error_message: str | None = Field(
        default=None, description="Error details if status != SUCCESS"
    )
    provenance_disclaimer: str = Field(
        default="Direct empirical measurement on Intel Core i7-14700 under Windows 11 with Ollama v0.40.11",
        description="Hardware and runtime provenance assertion",
    )


class RuntimeExperimentManifest(BaseModel):
    """Comprehensive manifest holding all Session 24 runtime benchmark observations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_id: str = Field(description="Unique experiment manifest ID")
    session_id: str = Field(default="session_24")
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    hardware_profile: WorkstationHardwareProfile = Field(
        default_factory=WorkstationHardwareProfile
    )
    total_active_duration_seconds: float = Field(
        ge=0.0, description="Total active benchmark execution duration in seconds"
    )
    observations: list[RuntimeObservation] = Field(default_factory=list)
    summary_by_thread_count: dict[str, Any] = Field(default_factory=dict)
    summary_by_concurrency: dict[str, Any] = Field(default_factory=dict)
    summary_by_context: dict[str, Any] = Field(default_factory=dict)
