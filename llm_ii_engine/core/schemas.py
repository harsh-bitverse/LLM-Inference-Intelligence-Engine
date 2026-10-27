"""Core validated domain schemas for the LLM Inference Intelligence Engine.

All schemas enforce strict type validation, explicit data provenance,
and a clear separation between real and simulated execution.
"""

from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    QuantizationType,
    StreamTerminationReason,
    TokenEventType,
)


class ModelSpec(BaseModel):
    """Specification of an LLM architecture and structural parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(
        min_length=1, description="Unique model identifier or HuggingFace repo ID"
    )
    family: str = Field(
        default="unknown", description="Model architecture family (e.g. llama, mistral)"
    )
    parameter_count_billion: float | None = Field(
        default=None, gt=0.0, description="Total parameter count in billions"
    )
    context_window: int = Field(
        default=4096,
        gt=0,
        description="Maximum context length supported by architecture",
    )
    num_layers: int | None = Field(default=None, gt=0, description="Number of transformer layers")
    num_attention_heads: int | None = Field(
        default=None, gt=0, description="Number of query attention heads"
    )
    num_kv_heads: int | None = Field(
        default=None, gt=0, description="Number of key/value heads (GQA/MQA)"
    )
    hidden_dimension: int | None = Field(
        default=None, gt=0, description="Hidden embedding dimension"
    )
    head_dimension: int | None = Field(
        default=None,
        gt=0,
        description="Explicit per-head dimension; defaults to hidden_dimension // num_attention_heads if omitted",
    )
    vocab_size: int | None = Field(default=None, gt=0, description="Vocabulary size")
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE,
        description="Quantization scheme applied to weights",
    )

    @field_validator("model_id")
    @classmethod
    def validate_model_id_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("model_id cannot be blank or whitespace only")
        return stripped


class EngineConfig(BaseModel):
    """Configuration applied to an inference engine instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_type: EngineType = Field(description="Inference engine runtime identifier")
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE,
        description="Quantization scheme applied to weights",
    )
    max_model_len: int | None = Field(
        default=None, gt=0, description="Server-enforced maximum sequence length"
    )
    tensor_parallel_size: int = Field(
        default=1, ge=1, description="Number of GPUs used for tensor parallelism"
    )
    gpu_memory_utilization: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Fraction of GPU VRAM allocated for weights and KV cache",
    )
    extra_args: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific parameter overrides"
    )


class ServerReportedStats(BaseModel):
    """Metadata and execution statistics reported directly by the inference server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int | None = Field(
        default=None, ge=0, description="Prompt tokens evaluated reported by server"
    )
    output_tokens: int | None = Field(
        default=None, ge=0, description="Output tokens generated reported by server"
    )
    prompt_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Prefill/prompt evaluation time in ms"
    )
    eval_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Decode evaluation time in ms"
    )
    total_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Total generation time in ms reported by server"
    )
    eval_tokens_per_second: float | None = Field(
        default=None, ge=0.0, description="Server-reported decode throughput (tok/s)"
    )
    prompt_eval_cached_count: int | None = Field(
        default=None, ge=0, description="Cached prompt tokens reused as reported by server"
    )
    load_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Model load duration into memory in ms as reported by server"
    )
    raw_stats: dict[str, Any] = Field(
        default_factory=dict, description="Raw metadata dictionary from server response"
    )

    @property
    def server_prefill_duration_ms(self) -> float | None:
        """Prefill / prompt evaluation duration reported by server in ms."""
        return self.prompt_duration_ms

    @property
    def server_decode_duration_ms(self) -> float | None:
        """Decode / token generation duration reported by server in ms."""
        return self.eval_duration_ms

    @property
    def server_load_duration_ms(self) -> float | None:
        """Model loading duration reported by server in ms."""
        return self.load_duration_ms

    @property
    def server_total_duration_ms(self) -> float | None:
        """Total execution duration reported by server in ms."""
        return self.total_duration_ms

    @property
    def server_decode_throughput_tok_s(self) -> float | None:
        """Server-reported decode throughput in tok/s."""
        return self.eval_tokens_per_second

    @property
    def server_overhead_ms(self) -> float | None:
        """Unaccounted server execution time (total - (load + prefill + decode)) in ms."""
        if self.total_duration_ms is None:
            return None
        accounted = (
            (self.load_duration_ms or 0.0)
            + (self.prompt_duration_ms or 0.0)
            + (self.eval_duration_ms or 0.0)
        )
        return max(0.0, round(self.total_duration_ms - accounted, 3))


class WorkloadSpec(BaseModel):
    """Defines a discrete generation workload passed to an inference engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Number of tokens in the prompt. 0 is valid for empty prompt. None if prompt_text is used without tokenizer.",
    )
    output_tokens: int = Field(
        default=128,
        ge=0,
        description="Number of output tokens requested (e.g. num_predict or max_tokens). 0 enables prefill-only benchmarking.",
    )
    prompt_text: str | None = Field(
        default=None, description="Raw text prompt if text generation is used"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Sampling temperature; 0.0 denotes greedy decoding",
    )
    top_p: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold",
    )
    stop_sequences: list[str] = Field(
        default_factory=list, description="Sequences that halt generation early"
    )
    seed: int | None = Field(
        default=None, description="Deterministic sampling seed if supported by engine"
    )

    @model_validator(mode="after")
    def validate_prompt_input(self) -> "WorkloadSpec":
        if self.prompt_tokens is None and self.prompt_text is None:
            raise ValueError("WorkloadSpec requires either prompt_tokens or prompt_text (or both)")
        return self


class RequestMetadata(BaseModel):
    """Metadata tracking request context, provenance, and timestamps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, description="Unique client-assigned request identifier")
    model_id: str = Field(min_length=1, description="Model ID targeted by the request")
    engine_type: EngineType = Field(description="Engine type servicing this request")
    execution_mode: ExecutionMode = Field(
        description="REAL hardware execution vs SIMULATED mock execution"
    )
    created_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="ISO 8601 UTC timestamp of request submission",
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Arbitrary user or session tags")


class TokenStreamEvent(BaseModel):
    """An atomic event emitted by an asynchronous token stream during inference.

    Timing semantics:
    - `timestamp_ns`: Nanosecond monotonic timestamp indicating event emission.
    - `delta_ns`: Elapsed nanoseconds since the preceding event (or start of request).
    - `clock_source`: Explicitly documents whether timing originates from a physical
      monotonic timer (MONOTONIC_REAL) or synthetic advancement (SYNTHETIC_VIRTUAL).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, description="Identifier of the originating request")
    sequence_index: int = Field(
        ge=0, description="0-indexed event sequence number within the stream"
    )
    event_type: TokenEventType = Field(description="Lifecycle category of this event")
    token_index: int | None = Field(
        default=None,
        ge=0,
        description="0-indexed position of the generated token. None for non-token events.",
    )
    token_text: str | None = Field(
        default=None,
        description="Decoded string fragment of the token, or empty for terminal/control events.",
    )
    token_id: int | None = Field(
        default=None, description="Integer token ID if exposed by tokenizer"
    )
    timestamp_ns: int = Field(ge=0, description="Monotonic timestamp in nanoseconds")
    delta_ns: int | None = Field(
        default=None,
        ge=0,
        description="Nanoseconds elapsed since preceding event, or None for initial event",
    )
    clock_source: ClockSource = Field(description="Origin and validity of nanosecond timestamps")
    execution_mode: ExecutionMode = Field(description="REAL vs SIMULATED execution indicator")
    is_first_token: bool = Field(
        default=False,
        description="True strictly on the very first decoded output token",
    )
    is_terminal: bool = Field(
        default=False, description="True on completion or unrecoverable error events"
    )
    termination_reason: StreamTerminationReason | None = Field(
        default=None,
        description="Reason stream concluded. Populated on terminal events.",
    )
    error_message: str | None = Field(
        default=None, description="Error details if event_type is ERROR"
    )
    server_stats: ServerReportedStats | None = Field(
        default=None,
        description="Server-reported execution metrics and token counts on completion",
    )

    @model_validator(mode="after")
    def validate_event_contract(self) -> "TokenStreamEvent":
        # 1. Validate clock source and execution mode consistency
        if (
            self.execution_mode == ExecutionMode.REAL
            and self.clock_source != ClockSource.MONOTONIC_REAL
        ):
            raise ValueError(
                f"ExecutionMode.REAL requires ClockSource.MONOTONIC_REAL, got {self.clock_source}"
            )
        if (
            self.clock_source == ClockSource.SYNTHETIC_VIRTUAL
            and self.execution_mode != ExecutionMode.SIMULATED
        ):
            raise ValueError(
                f"ClockSource.SYNTHETIC_VIRTUAL is only allowed with ExecutionMode.SIMULATED, got {self.execution_mode}"
            )

        # 2. Lifecycle event contracts
        if self.event_type == TokenEventType.START:
            if self.token_index is not None:
                raise ValueError("START event must not contain token_index")
            if self.is_first_token:
                raise ValueError("START event cannot have is_first_token=True")
            if self.is_terminal:
                raise ValueError("START event cannot have is_terminal=True")
            if self.termination_reason is not None:
                raise ValueError("START event cannot have termination_reason")
            if self.error_message is not None:
                raise ValueError("START event cannot have error_message")

        elif self.event_type == TokenEventType.TOKEN:
            if self.token_index is None:
                raise ValueError("TOKEN event must have a non-null token_index")
            expected_first = self.token_index == 0
            if self.is_first_token != expected_first:
                raise ValueError(
                    f"TOKEN event with token_index={self.token_index} must have is_first_token={expected_first}"
                )
            if self.is_terminal:
                raise ValueError("TOKEN event cannot have is_terminal=True")
            if self.termination_reason is not None:
                raise ValueError("TOKEN event cannot have termination_reason")
            if self.error_message is not None:
                raise ValueError("TOKEN event cannot have error_message")

        elif self.event_type == TokenEventType.CHUNK:
            if self.is_terminal:
                raise ValueError("CHUNK event cannot have is_terminal=True")
            if self.is_first_token:
                raise ValueError(
                    "CHUNK event cannot have is_first_token=True (use TOKEN for verified tokens)"
                )
            if self.termination_reason is not None:
                raise ValueError("CHUNK event cannot have termination_reason")
            if self.error_message is not None:
                raise ValueError("CHUNK event cannot have error_message")

        elif self.event_type == TokenEventType.COMPLETION:
            if not self.is_terminal:
                raise ValueError("COMPLETION event must have is_terminal=True")
            if self.termination_reason is None:
                raise ValueError("COMPLETION event must specify a termination_reason")
            if self.termination_reason == StreamTerminationReason.ERROR:
                raise ValueError(
                    "COMPLETION event cannot have termination_reason=ERROR; use TokenEventType.ERROR"
                )
            if self.error_message is not None:
                raise ValueError("COMPLETION event cannot have error_message")

        elif self.event_type == TokenEventType.ERROR:
            if not self.is_terminal:
                raise ValueError("ERROR event must have is_terminal=True")
            if self.termination_reason != StreamTerminationReason.ERROR:
                raise ValueError(
                    "ERROR event must have termination_reason=StreamTerminationReason.ERROR"
                )
            if not self.error_message or not self.error_message.strip():
                raise ValueError("ERROR event must include a non-empty error_message")

        if self.event_type != TokenEventType.COMPLETION and self.server_stats is not None:
            raise ValueError(
                f"{self.event_type} event cannot contain server_stats; only COMPLETION events may include server_stats"
            )

        return self


class GenerationRequest(BaseModel):
    """Unified request payload submitted to an inference engine adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, description="Unique identifier for this generation run")
    model_spec: ModelSpec = Field(description="Architecture specifications of the target model")
    engine_config: EngineConfig = Field(description="Runtime engine configuration")
    workload: WorkloadSpec = Field(
        description="Workload parameters (prompt, output tokens, sampling)"
    )
    metadata: RequestMetadata | None = Field(
        default=None, description="Tracing and session metadata"
    )


class EngineCapabilities(BaseModel):
    """Static and dynamic capabilities advertised by an inference engine adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_type: EngineType = Field(description="Inference engine type")
    execution_mode: ExecutionMode = Field(
        description="Real hardware vs simulated mock execution mode"
    )
    supported_quantizations: list[QuantizationType] = Field(
        default_factory=list,
        description="List of quantization formats supported by this adapter",
    )
    supports_streaming: bool = Field(
        default=True, description="Whether adapter yields incremental token events"
    )
    supports_gpu_memory_query: bool = Field(
        default=False,
        description="Whether engine can query live VRAM allocation directly",
    )
    supports_token_ids: bool = Field(
        default=False,
        description="Whether engine exposes integer token IDs in token events",
    )
    supports_true_token_itl: bool = Field(
        default=False,
        description="Whether engine streams individual tokens (True) vs arbitrary text chunks (False)",
    )
    supports_server_metrics: bool = Field(
        default=False,
        description="Whether engine returns server-side eval durations and token counts",
    )
    max_context_window: int | None = Field(
        default=None,
        gt=0,
        description="Maximum total sequence length supported by adapter",
    )


T = TypeVar("T")


class Observation[T](BaseModel):
    """Generic container for an observed, predicted, or theoretical data point.

    Guarantees strict labeling of data origin, provenance, and execution mode.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: T = Field(description="The underlying metric or observation payload")
    provenance: ProvenanceType = Field(
        description="Data provenance: MEASURED, PREDICTED, or HEURISTIC"
    )
    execution_mode: ExecutionMode = Field(
        description="REAL hardware execution vs SIMULATED mock execution"
    )
    clock_source: ClockSource | None = Field(
        default=None,
        description="Timing origin if observation is a temporal measurement",
    )
    unit: str | None = Field(
        default=None,
        description="Physical or statistical unit (e.g. 'ms', 'MB', 'tok/s')",
    )
    description: str | None = Field(default=None, description="Contextual note or documentation")

    @model_validator(mode="after")
    def validate_provenance_integrity(self) -> "Observation[T]":
        if (
            self.execution_mode == ExecutionMode.SIMULATED
            and self.provenance == ProvenanceType.MEASURED
            and self.clock_source == ClockSource.MONOTONIC_REAL
        ):
            # This is a real-time observation of a simulation (e.g. wall clock test duration),
            # which is valid, but must remain labeled SIMULATED.
            pass
        return self
