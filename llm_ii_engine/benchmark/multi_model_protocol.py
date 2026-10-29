"""Multi-model benchmarking protocol, normalized metrics, and comparison contracts.

Enforces scientifically rigorous comparison across different model families, parameter counts,
and tokenizers, distinguishing true task completion time from tokenizer-dependent token counts.
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ModelResidencyState,
    ProvenanceType,
)


class ModelInstallationStatus(str, Enum):
    """Local runtime installation status of a model candidate."""

    INSTALLED = "INSTALLED"
    UNINSTALLED_CANDIDATE = "UNINSTALLED_CANDIDATE"
    CORRUPTED = "CORRUPTED"
    UNSUPPORTED = "UNSUPPORTED"


class CodeExecutionSecurityMode(str, Enum):
    """Security isolation mode for model code evaluation."""

    FAIL_CLOSED = "FAIL_CLOSED"
    OPT_IN_HARDENED_JOB_OBJECT = "OPT_IN_HARDENED_JOB_OBJECT"


class QualityEvaluationStatus(str, Enum):
    """Granular verification status for benchmark quality evaluation."""

    SYNTAX_VALID = "SYNTAX_VALID"
    SYNTAX_INVALID = "SYNTAX_INVALID"
    STATIC_CHECK_PASSED = "STATIC_CHECK_PASSED"
    STATIC_CHECK_FAILED = "STATIC_CHECK_FAILED"
    EXECUTION_TEST_PASSED = "EXECUTION_TEST_PASSED"
    EXECUTION_TEST_FAILED = "EXECUTION_TEST_FAILED"
    EXECUTION_UNVERIFIED = "EXECUTION_UNVERIFIED"
    EVALUATOR_ERROR = "EVALUATOR_ERROR"
    EVALUATION_TIMEOUT = "EVALUATION_TIMEOUT"


class TaskEvaluationType(str, Enum):
    """Evaluation category for benchmarking workload tasks."""

    INTERACTIVE_QA = "INTERACTIVE_QA"
    LONG_CONTEXT_RETRIEVAL = "LONG_CONTEXT_RETRIEVAL"
    LONG_FORM_GENERATION = "LONG_FORM_GENERATION"
    STRUCTURED_JSON = "STRUCTURED_JSON"
    CODE_GENERATION = "CODE_GENERATION"
    REASONING_NUMERICAL = "REASONING_NUMERICAL"


class ModelBenchmarkMetadata(BaseModel):
    """Architectural, runtime, and quantization metadata for a benchmarked model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Model identifier, e.g. 'llama3.2:1b' or 'qwen2.5:1.5b'")
    model_digest: str | None = Field(default=None, description="SHA256 digest of weights or manifest")
    model_family: str = Field(description="Architecture family, e.g. 'llama', 'qwen2', 'phi'")
    parameter_count_billion: float = Field(gt=0, description="Parameter count in billions")
    quantization: str = Field(description="Quantization level, e.g. 'Q8_0', 'Q4_K_M'")
    model_size_bytes: int = Field(gt=0, description="On-disk size of model weights in bytes")
    context_window: int = Field(gt=0, description="Maximum supported context length")
    tokenizer_family: str = Field(description="Tokenizer family, e.g. 'tiktoken_llama3', 'bpe_qwen2'")
    vocabulary_size: int = Field(gt=0, description="Vocabulary size")
    runtime_name: str = Field(default="ollama", description="Serving runtime name")
    runtime_version: str = Field(description="Runtime version string, e.g. '0.40.11'")
    hardware_catalog_id: str = Field(
        default="cpu-intel-i7-14700", description="Hardware catalog identifier"
    )
    execution_backend: str = Field(
        default="cpu_ggml_avx2", description="Execution backend engine, e.g. 'cpu_ggml_avx2'"
    )
    license: str = Field(description="Model license identifier or name")


class BenchmarkSamplingConfig(BaseModel):
    """Deterministic sampling parameters for multi-model evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=1, ge=1, description="Top-k sampling cutoff")
    seed: int | None = Field(default=42, description="Random seed for reproducibility")
    stop_sequences: list[str] = Field(default_factory=list, description="Stop sequence strings")


class BenchmarkExecutionCondition(BaseModel):
    """Environmental and runtime execution state during benchmarking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concurrency: int = Field(default=1, ge=1, description="Concurrent request count")
    model_residency: ModelResidencyState = Field(
        default=ModelResidencyState.WARM, description="Memory residency state"
    )
    prompt_cache_state: CacheStateClassification = Field(
        default=CacheStateClassification.COLD_PREFIX, description="KV cache prefix condition"
    )
    warmup_policy: str = Field(
        default="1_warmup_excluded", description="Warmup exclusion description"
    )
    request_index: int = Field(default=1, ge=1, description="Request sequence index")
    repetition_index: int = Field(default=1, ge=1, description="Repetition sequence index (1-based)")


class TaskNormalizedMetrics(BaseModel):
    """Performance measurements with explicit distinction between raw tokens and normalized metrics.

    Token throughput is not directly comparable across models with different tokenizers.
    Task elapsed time and character/word rates provide normalized comparison metrics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_e2e_duration_ms: float = Field(
        ge=0.0, description="Total monotonic client wall-clock duration in milliseconds"
    )
    server_total_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server-reported total execution duration in milliseconds"
    )
    server_load_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server-reported model load duration in milliseconds"
    )
    server_prefill_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server-reported prompt evaluation duration in milliseconds"
    )
    server_decode_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server-reported token evaluation duration in milliseconds"
    )

    # Tokenizer-dependent raw metrics
    raw_prompt_tokens: int = Field(ge=0, description="Actual prompt tokens according to model's tokenizer")
    raw_output_tokens: int = Field(ge=0, description="Actual output tokens according to model's tokenizer")
    raw_decode_tokens_per_second: float | None = Field(
        default=None,
        ge=0.0,
        description="Raw server decode tokens/second. (NOTE: Tokenizer-dependent; cannot be directly compared across different vocabularies).",
    )
    raw_client_tokens_per_second: float | None = Field(
        default=None,
        ge=0.0,
        description="Raw client E2E tokens/second. (NOTE: Tokenizer-dependent).",
    )

    # Task-normalized metrics (comparable across different tokenizers)
    task_elapsed_ms: float = Field(
        ge=0.0, description="Primary task completion time in milliseconds (comparable across models)"
    )
    characters_generated: int = Field(
        ge=0, description="Character count of generated output text"
    )
    words_generated: int = Field(
        ge=0, description="Whitespace-delimited word count of generated output text"
    )
    characters_per_second: float = Field(
        ge=0.0, description="Normalized output throughput in characters/second"
    )
    words_per_second: float = Field(
        ge=0.0, description="Normalized output throughput in words/second"
    )

    # Comparability flag and advisory
    cross_tokenizer_comparable: bool = Field(
        default=False,
        description="Whether raw token throughput can be directly compared. False across different model families.",
    )
    tokenizer_comparability_advisory: str = Field(
        default="Raw tokens/second cannot be directly compared across differing tokenizers. Compare task_elapsed_ms or characters_per_second for equivalent workloads.",
        description="Caveat explaining why token counts differ for the same textual prompt.",
    )


class QualityEvaluationResult(BaseModel):
    """Objective validation outcomes for benchmark task output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(description="Unique identifier of the evaluated task")
    task_type: TaskEvaluationType = Field(description="Evaluation category")
    format_compliant: bool | None = Field(
        default=None, description="Whether output adhered to requested structure (e.g. valid JSON)"
    )
    task_success: bool | None = Field(
        default=None, description="Whether task met objective correctness criteria"
    )
    correctness_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Deterministic score between 0.0 and 1.0"
    )
    evaluation_method: str = Field(
        description="Verification method: 'JSON_SCHEMA_VALIDATOR', 'REGEX_EXTRACTOR', 'EXACT_MATCH', 'AST_SYNTAX_CHECK', 'FAIL_CLOSED_AST_STATIC_VERIFICATION'"
    )
    evaluation_notes: str = Field(default="", description="Diagnostic details or error explanation")
    response_snippet: str = Field(default="", description="Snippet of evaluated model response")
    evaluation_status: QualityEvaluationStatus | None = Field(
        default=None, description="Granular verification taxonomy status"
    )
    security_mode: CodeExecutionSecurityMode | None = Field(
        default=None, description="Execution isolation mode used for code tasks"
    )
    isolation_guarantees: list[str] = Field(
        default_factory=list, description="List of active OS/process isolation guarantees"
    )


class MultiModelBenchmarkObservation(BaseModel):
    """Complete, self-contained record of a single benchmark observation in a multi-model experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str = Field(description="Unique observation identifier")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC timestamp",
    )
    experiment_id: str = Field(description="Experiment run identifier")
    model_metadata: ModelBenchmarkMetadata = Field(
        description="Architecture, runtime, and quantization details"
    )
    sampling_config: BenchmarkSamplingConfig = Field(description="Sampling parameters")
    execution_condition: BenchmarkExecutionCondition = Field(description="Runtime state and cache")
    performance_metrics: TaskNormalizedMetrics = Field(
        description="Raw and normalized performance outcomes"
    )
    quality_result: QualityEvaluationResult = Field(
        description="Automated quality assessment outcome"
    )
    process_rss_mb: float | None = Field(
        default=None, description="Sampled process working set memory in MiB"
    )
    provenance: ProvenanceType = Field(
        default=ProvenanceType.MEASURED,
        description="Data provenance (MEASURED, PREDICTED, or HEURISTIC)",
    )
