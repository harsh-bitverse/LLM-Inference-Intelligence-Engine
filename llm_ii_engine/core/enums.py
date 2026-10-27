"""Core domain enumerations for LLM Inference Intelligence Engine.

Establishes strict categorization for data provenance, execution mode,
clock sources, engine types, quantization schemes, and token stream events.
"""

from enum import Enum


class ProvenanceType(str, Enum):
    """Data provenance classification ensuring zero fabrication of measurements.

    Attributes:
        MEASURED: Directly recorded from hardware counters or physical monotonic timers.
        PREDICTED: Estimated by an analytical or empirical machine learning model.
        HEURISTIC: Derived from hardware catalog specifications or theoretical equations.
    """

    MEASURED = "MEASURED"
    PREDICTED = "PREDICTED"
    HEURISTIC = "HEURISTIC"


class ExecutionMode(str, Enum):
    """Distinguishes actual model execution from simulated or mocked execution.

    A synthetic event or benchmark must NEVER be presented as real hardware evidence.
    Execution mode and data provenance are strictly independent concepts.

    Attributes:
        REAL: Executed against actual engine runtime, model weights, and compute hardware.
        SIMULATED: Produced by a synthetic adapter, mock engine, or virtual simulation.
    """

    REAL = "REAL"
    SIMULATED = "SIMULATED"


class ClockSource(str, Enum):
    """Source and semantics of nanosecond timestamps.

    Attributes:
        MONOTONIC_REAL: Nanosecond timestamps from physical monotonic timer (e.g. time.perf_counter_ns).
        SYNTHETIC_VIRTUAL: Deterministically advanced virtual nanoseconds; non-physical, zero sleep.
    """

    MONOTONIC_REAL = "MONOTONIC_REAL"
    SYNTHETIC_VIRTUAL = "SYNTHETIC_VIRTUAL"


class EngineType(str, Enum):
    """Supported or pluggable inference engine types."""

    MOCK = "mock"
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    TRANSFORMERS = "transformers"
    CUSTOM = "custom"


class QuantizationType(str, Enum):
    """Weight and activation quantization formats."""

    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    AWQ = "awq"
    GPTQ = "gptq"
    GGUF_Q4_K_M = "q4_k_m"
    GGUF_Q8_0 = "q8_0"


class TokenEventType(str, Enum):
    """Lifecycle event types within an asynchronous token stream."""

    START = "start"
    TOKEN = "token"
    CHUNK = "chunk"
    COMPLETION = "completion"
    ERROR = "error"


class TokenCountSource(str, Enum):
    """Attribution source for prompt and output token counts.

    Separates verified tokenizer-derived counts from server reports, explicit requests,
    heuristic word-count estimates, and unverified/unknown sources.
    """

    TOKENIZER = "tokenizer"
    SERVER_REPORTED = "server_reported"
    REQUESTED = "requested"
    TOKEN_EVENTS = "token_events"
    HEURISTIC_WORD_COUNT = "heuristic_word_count"
    UNKNOWN = "unknown"


class StreamTerminationReason(str, Enum):
    """Termination condition for an asynchronous token stream."""

    FINISHED = "finished"
    MAX_TOKENS = "max_tokens"
    CANCELLED = "cancelled"
    ERROR = "error"


class MetricType(str, Enum):
    """Core LLM inference performance metric categories."""

    TTFT = "ttft"
    ITL = "itl"
    TPOT = "tpot"
    CHUNK_LATENCY = "chunk_latency"
    THROUGHPUT_TOK_S = "throughput_tok_s"
    MEMORY_PEAK_MB = "memory_peak_mb"


class DiagnosticSeverity(str, Enum):
    """Severity classification for diagnostic findings."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DiagnosticCategory(str, Enum):
    """Categorization of engine performance and architectural diagnostics."""

    MEMORY_CAPACITY = "memory_capacity"
    CONFIGURATION_VIABILITY = "configuration_viability"
    LATENCY_OVERHEAD = "latency_overhead"
    EFFICIENCY_BOTTLENECK = "efficiency_bottleneck"
    TELEMETRY_GAP = "telemetry_gap"


class DiagnosticConfidence(str, Enum):
    """Confidence level of a diagnostic finding based on evidence strength."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class FailurePolicy(str, Enum):
    """Policy governing benchmark execution when a request encounters an error."""

    CONTINUE_ON_ERROR = "continue_on_error"
    ABORT_IMMEDIATELY = "abort_immediately"
    ABORT_ON_CONSECUTIVE = "abort_on_consecutive"


class ModelResidencyState(str, Enum):
    """Model memory residency state prior to request execution."""

    COLD = "cold"
    WARM = "warm"
    UNKNOWN = "unknown"


class CacheStateClassification(str, Enum):
    """Prompt KV cache state classification based on empirical runtime telemetry."""

    COLD_PREFIX = "cold_prefix"
    WARM_PREFIX = "warm_prefix"
    PARTIAL_PREFIX = "partial_prefix"
    UNKNOWN = "unknown"


class PrefixRelationshipType(str, Enum):
    """Relationship of prompt prefix to immediately preceding request."""

    IDENTICAL = "identical"
    SHARED_PREFIX = "shared_prefix"
    UNRELATED = "unrelated"
    NONE = "none"


