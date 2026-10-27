"""Domain models for empirical performance prediction and generalization evaluation.

Defines typed prediction records, datasets, baseline outputs, and evaluation metrics,
enforcing strict separation between empirical measurements and theoretical estimates.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_ii_engine.core.enums import (
    CacheStateClassification,
    EngineType,
    ExecutionMode,
    ModelResidencyState,
    PrefixRelationshipType,
    ProvenanceType,
    QuantizationType,
    TokenCountSource,
)


class PredictionTarget(str, Enum):
    """Target inference performance metric to predict."""

    TTFT_MS = "ttft_ms"
    TPOT_MS = "tpot_ms"
    CHUNK_LATENCY_MS = "chunk_latency_ms"
    OUTPUT_THROUGHPUT_TOK_S = "output_throughput_tok_s"
    PEAK_VRAM_MB = "peak_vram_mb"
    DURATION_MS = "duration_ms"
    TTFC_MS = "ttfc_ms"
    SERVER_PROMPT_EVAL_MS = "server_prompt_eval_duration_ms"
    SERVER_EVAL_MS = "server_eval_duration_ms"

    # Distinct Semantic Metrics (Session 16)
    CLIENT_E2E_DURATION_MS = "client_e2e_duration_ms"
    CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S = "client_e2e_output_throughput_tok_s"
    SERVER_TOTAL_DURATION_MS = "server_total_duration_ms"
    SERVER_LOAD_DURATION_MS = "server_load_duration_ms"
    SERVER_PREFILL_DURATION_MS = "server_prefill_duration_ms"
    SERVER_DECODE_DURATION_MS = "server_decode_duration_ms"
    SERVER_DECODE_THROUGHPUT_TOK_S = "server_decode_throughput_tok_s"


class GeneralizationStatus(BaseModel):
    """Analysis of whether a prediction request falls within the training support envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_out_of_distribution: bool = Field(
        default=False,
        description="True if input features fall outside the training support envelope",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Explicit warnings regarding extrapolated features or unseen categories",
    )
    extrapolated_features: list[str] = Field(
        default_factory=list,
        description="Numerical features outside the [min, max] range of training support",
    )
    unseen_categories: list[str] = Field(
        default_factory=list,
        description="Categorical features with values not present in training data",
    )
    missing_features: list[str] = Field(
        default_factory=list,
        description="Critical features that were omitted or unavailable in the input",
    )


class PredictionRecord(BaseModel):
    """A single tabular record extracted from benchmark evidence for prediction and evaluation.

    Preserves missing values as None without fabricating observations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Identifiers & Provenance (metadata only; never used directly as regression features)
    record_id: str = Field(min_length=1, description="Unique record identifier")
    experiment_id: str | None = Field(
        default=None, description="Benchmark experiment manifest identifier (for repetition grouping)"
    )
    suite_id: str = Field(
        min_length=1, description="Originating benchmark suite ID (used for grouped splits)"
    )
    request_id: str | None = Field(
        default=None, description="Request ID if extracted from a per-request result"
    )
    model_id: str = Field(min_length=1, description="Model architecture identifier")
    model_digest: str | None = Field(
        default=None, description="Artifact hash or digest of the model weights"
    )
    engine_type: EngineType = Field(description="Inference engine runtime")
    execution_mode: ExecutionMode = Field(
        description="REAL hardware execution vs SIMULATED mock run"
    )
    provenance: ProvenanceType = Field(description="MEASURED or HEURISTIC provenance")
    is_warmup: bool = Field(
        default=False, description="True if observation occurred during warmup phase"
    )

    # Model & Architecture Features
    parameter_count_billion: float | None = Field(default=None, gt=0.0)
    num_layers: int | None = Field(default=None, gt=0)
    num_attention_heads: int | None = Field(default=None, gt=0)
    num_kv_heads: int | None = Field(default=None, gt=0)
    hidden_dimension: int | None = Field(default=None, gt=0)
    head_dimension: int | None = Field(default=None, gt=0)
    quantization: QuantizationType = Field(default=QuantizationType.NONE)
    bytes_per_parameter: float | None = Field(default=None, gt=0.0)

    # Hardware Features
    hardware_catalog_id: str | None = Field(default=None)
    gpu_memory_capacity_gb: float | None = Field(default=None, gt=0.0)
    gpu_memory_bandwidth_gbps: float | None = Field(default=None, gt=0.0)
    gpu_peak_compute_tflops: float | None = Field(default=None, gt=0.0)
    host_ram_total_gb: float | None = Field(default=None, gt=0.0)

    # Workload & Concurrency Features
    concurrency: int = Field(default=1, ge=1)
    prompt_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Prompt token length (authoritative count when available, None if unknown)",
    )
    prompt_token_count_source: TokenCountSource = Field(default=TokenCountSource.REQUESTED)
    requested_output_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Output token limit requested before execution (known prior to inference)",
    )
    actual_output_tokens: int = Field(
        ge=0,
        description="Actual decoded output tokens generated and received (known only after execution)",
    )
    output_tokens: int = Field(
        ge=0,
        description="Actual decoded output tokens (maintained for backward compatibility with actual_output_tokens)",
    )
    token_count_source: TokenCountSource = Field(default=TokenCountSource.TOKEN_EVENTS)
    planned_sequence_length: int | None = Field(
        default=None,
        ge=0,
        description="Pre-execution planned sequence length (prompt_tokens + requested_output_tokens)",
    )
    total_sequence_length: int | None = Field(
        default=None,
        ge=0,
        description="Actual combined sequence length (prompt_tokens + actual_output_tokens)",
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_token_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Synchronize output_tokens and actual_output_tokens
            if "actual_output_tokens" not in data and "output_tokens" in data:
                data["actual_output_tokens"] = data["output_tokens"]
            elif "output_tokens" not in data and "actual_output_tokens" in data:
                data["output_tokens"] = data["actual_output_tokens"]

            p_tok = data.get("prompt_tokens")
            req_tok = data.get("requested_output_tokens")
            if req_tok is not None and "planned_sequence_length" not in data:
                data["planned_sequence_length"] = (p_tok + req_tok) if p_tok is not None else None

            act_tok = data.get("actual_output_tokens", 0) or 0
            if "total_sequence_length" not in data:
                data["total_sequence_length"] = (p_tok + act_tok) if p_tok is not None else None

            # Synchronize client e2e duration
            if data.get("client_e2e_duration_ms") is None and data.get("duration_ms") is not None:
                data["client_e2e_duration_ms"] = data.get("duration_ms")
            elif data.get("duration_ms") is None and data.get("client_e2e_duration_ms") is not None:
                data["duration_ms"] = data.get("client_e2e_duration_ms")

            # Synchronize client e2e throughput
            if data.get("client_e2e_output_throughput_tok_s") is None and data.get("output_tokens_per_second") is not None:
                data["client_e2e_output_throughput_tok_s"] = data.get("output_tokens_per_second")
            elif data.get("output_tokens_per_second") is None and data.get("client_e2e_output_throughput_tok_s") is not None:
                data["output_tokens_per_second"] = data.get("client_e2e_output_throughput_tok_s")

            # Synchronize server prefill duration
            if data.get("server_prefill_duration_ms") is None and data.get("server_prompt_eval_duration_ms") is not None:
                data["server_prefill_duration_ms"] = data.get("server_prompt_eval_duration_ms")
            elif data.get("server_prompt_eval_duration_ms") is None and data.get("server_prefill_duration_ms") is not None:
                data["server_prompt_eval_duration_ms"] = data.get("server_prefill_duration_ms")

            # Synchronize server decode duration
            if data.get("server_decode_duration_ms") is None and data.get("server_eval_duration_ms") is not None:
                data["server_decode_duration_ms"] = data.get("server_eval_duration_ms")
            elif data.get("server_eval_duration_ms") is None and data.get("server_decode_duration_ms") is not None:
                data["server_eval_duration_ms"] = data.get("server_decode_duration_ms")

            # Compute server decode throughput
            if data.get("server_decode_throughput_tok_s") is None:
                dec_ms = data.get("server_decode_duration_ms") or data.get("server_eval_duration_ms")
                if dec_ms and dec_ms > 0 and act_tok and act_tok > 0:
                    data["server_decode_throughput_tok_s"] = round(act_tok / (dec_ms / 1000.0), 2)
        return data

    # Target Measurements (None if unavailable, unmeasured, or ineligible)
    ttft_ms: float | None = Field(default=None, ge=0.0)
    ttfc_ms: float | None = Field(default=None, ge=0.0)
    tpot_ms: float | None = Field(default=None, ge=0.0)
    chunk_latency_ms: float | None = Field(default=None, ge=0.0)
    output_tokens_per_second: float | None = Field(default=None, ge=0.0)
    peak_vram_mb: float | None = Field(default=None, ge=0.0)
    duration_ms: float | None = Field(default=None, ge=0.0)
    server_prompt_eval_duration_ms: float | None = Field(default=None, ge=0.0)
    server_eval_duration_ms: float | None = Field(default=None, ge=0.0)

    # Distinct Semantic Metrics (Session 16)
    client_e2e_duration_ms: float | None = Field(default=None, ge=0.0)
    client_e2e_output_throughput_tok_s: float | None = Field(default=None, ge=0.0)
    server_total_duration_ms: float | None = Field(default=None, ge=0.0)
    server_load_duration_ms: float | None = Field(default=None, ge=0.0)
    server_prefill_duration_ms: float | None = Field(default=None, ge=0.0)
    server_decode_duration_ms: float | None = Field(default=None, ge=0.0)
    server_decode_throughput_tok_s: float | None = Field(default=None, ge=0.0)

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
    server_prompt_eval_cached_count: int | None = Field(
        default=None, ge=0, description="Cached prompt tokens reused as reported by server"
    )
    prefix_relationship: PrefixRelationshipType | None = Field(
        default=None, description="Prefix relationship to immediately preceding request"
    )

    # Eligibility Flags & Audit Notes
    is_eligible_for_ttft: bool = Field(default=False)
    is_eligible_for_tpot: bool = Field(default=False)
    is_eligible_for_chunk_latency: bool = Field(default=False)
    is_eligible_for_throughput: bool = Field(default=False)
    is_eligible_for_vram: bool = Field(default=False)
    is_eligible_for_duration: bool = Field(default=False)
    is_eligible_for_ttfc: bool = Field(default=False)
    is_eligible_for_prefill: bool = Field(default=False)
    is_eligible_for_decode: bool = Field(default=False)
    ineligibility_reasons: dict[PredictionTarget, list[str]] = Field(default_factory=dict)

    def is_eligible_for_target(self, target: PredictionTarget) -> bool:
        """Return whether record is eligible for a given prediction target."""
        if target == PredictionTarget.TTFT_MS:
            return self.is_eligible_for_ttft
        elif target == PredictionTarget.TPOT_MS:
            return self.is_eligible_for_tpot
        elif target == PredictionTarget.CHUNK_LATENCY_MS:
            return self.is_eligible_for_chunk_latency
        elif target in (
            PredictionTarget.OUTPUT_THROUGHPUT_TOK_S,
            PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S,
        ):
            return self.is_eligible_for_throughput
        elif target == PredictionTarget.PEAK_VRAM_MB:
            return self.is_eligible_for_vram
        elif target in (
            PredictionTarget.DURATION_MS,
            PredictionTarget.CLIENT_E2E_DURATION_MS,
        ):
            return self.is_eligible_for_duration
        elif target == PredictionTarget.TTFC_MS:
            return self.is_eligible_for_ttfc
        elif target in (
            PredictionTarget.SERVER_PROMPT_EVAL_MS,
            PredictionTarget.SERVER_PREFILL_DURATION_MS,
        ):
            return self.is_eligible_for_prefill
        elif target in (
            PredictionTarget.SERVER_EVAL_MS,
            PredictionTarget.SERVER_DECODE_DURATION_MS,
        ):
            return self.is_eligible_for_decode
        elif target == PredictionTarget.SERVER_TOTAL_DURATION_MS:
            return self.server_total_duration_ms is not None and self.server_total_duration_ms > 0
        elif target == PredictionTarget.SERVER_LOAD_DURATION_MS:
            return self.server_load_duration_ms is not None and self.server_load_duration_ms >= 0
        elif target == PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S:
            return (
                self.is_eligible_for_decode
                and self.actual_output_tokens > 0
                and self.server_decode_throughput_tok_s is not None
                and self.server_decode_throughput_tok_s > 0
            )
        return False


class PredictionDataset(BaseModel):
    """Collection of typed prediction records with query and filtering capabilities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1, description="Dataset identifier")
    created_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Dataset creation timestamp",
    )
    records: list[PredictionRecord] = Field(default_factory=list)
    description: str | None = Field(default=None)

    def __len__(self) -> int:
        return len(self.records)

    def get_eligible_records(self, target: PredictionTarget) -> list[PredictionRecord]:
        """Return subset of records eligible for the specified prediction target."""
        if target == PredictionTarget.TTFT_MS:
            return [r for r in self.records if r.is_eligible_for_ttft]
        elif target == PredictionTarget.TPOT_MS:
            return [r for r in self.records if r.is_eligible_for_tpot]
        elif target == PredictionTarget.CHUNK_LATENCY_MS:
            return [r for r in self.records if r.is_eligible_for_chunk_latency]
        elif target == PredictionTarget.OUTPUT_THROUGHPUT_TOK_S:
            return [r for r in self.records if r.is_eligible_for_throughput]
        elif target == PredictionTarget.PEAK_VRAM_MB:
            return [r for r in self.records if r.is_eligible_for_vram]
        elif target == PredictionTarget.DURATION_MS:
            return [r for r in self.records if r.is_eligible_for_duration]
        elif target == PredictionTarget.TTFC_MS:
            return [r for r in self.records if r.is_eligible_for_ttfc]
        elif target == PredictionTarget.SERVER_PROMPT_EVAL_MS:
            return [r for r in self.records if r.is_eligible_for_prefill]
        elif target == PredictionTarget.SERVER_EVAL_MS:
            return [r for r in self.records if r.is_eligible_for_decode]
        return []


class PredictionOutput(BaseModel):
    """Result of an inference performance prediction, with provenance and uncertainty metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_id: str = Field(min_length=1)
    target: PredictionTarget = Field(description="Target metric predicted")
    predicted_value: float = Field(description="Predicted point estimate")
    unit: str = Field(description="Physical unit (e.g. 'ms', 'tok/s', 'MiB')")

    predictor_id: str = Field(
        description="Identifier of model or baseline that produced prediction"
    )
    predictor_type: str = Field(
        description="Predictor classification ('ConstantMedian', 'RidgeRegression', 'AnalyticalRoofline')"
    )
    provenance: ProvenanceType = Field(
        default=ProvenanceType.PREDICTED,
        description="Prediction provenance is PREDICTED",
    )
    execution_mode: ExecutionMode = Field(
        description="Execution mode of the underlying evidence used to fit the predictor",
    )

    generalization_status: GeneralizationStatus = Field(
        description="Out-of-distribution and feature coverage assessment",
    )
    training_sample_count: int = Field(
        ge=0, description="Number of valid samples used to train predictor"
    )
    validation_status: str = Field(
        description="'VALIDATED_EMPIRICAL', 'EXPLORATORY_INSUFFICIENT_DATA', or 'THEORETICAL_ANALYTICAL'",
    )
    prediction_interval: tuple[float, float] | None = Field(
        default=None, description="Calibrated prediction interval (lower, upper) if sample size is sufficient"
    )
    interval_status: str | None = Field(
        default=None,
        description="Validation status of prediction interval ('CALIBRATED', 'EXPLORATORY_INSUFFICIENT_SAMPLE', or 'UNAVAILABLE')",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Documented limitations and assumptions",
    )
    notes: str = Field(default="", description="Contextual notes")


class EvaluationResult(BaseModel):
    """Evaluation metrics for a trained predictor on a designated validation split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: PredictionTarget = Field(description="Target metric evaluated")
    predictor_id: str = Field(description="Identifier of evaluated predictor")
    predictor_type: str = Field(description="Predictor architecture classification")
    split_strategy: str = Field(
        description="Partitioning strategy ('RANDOM', 'GROUPED_BY_SUITE', 'GROUPED_BY_MODEL', 'IN_SAMPLE')",
    )
    sample_count: int = Field(ge=0, description="Number of test records evaluated")
    test_sample_count: int = Field(
        default=0, ge=0, description="Number of test records evaluated (synonym for sample_count)"
    )
    train_sample_count: int = Field(
        default=0, ge=0, description="Number of training records used to fit predictor"
    )
    unit: str = Field(description="Target metric unit")

    mae: float = Field(ge=0.0, description="Mean Absolute Error")
    rmse: float = Field(ge=0.0, description="Root Mean Squared Error")
    mape_percent: float | None = Field(
        default=None,
        ge=0.0,
        description="Mean Absolute Percentage Error (%) if target is non-zero",
    )

    baseline_mae: float | None = Field(
        default=None, ge=0.0, description="MAE of median baseline on same split"
    )
    baseline_rmse: float | None = Field(
        default=None, ge=0.0, description="RMSE of median baseline on same split"
    )

    is_synthetic: bool = Field(
        default=False,
        description="True if evaluation dataset was generated from simulated/synthetic execution",
    )
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.SIMULATED,
        description="REAL hardware execution vs SIMULATED mock run",
    )
    sufficiency_threshold: int = Field(
        default=10,
        ge=1,
        description="Minimum sample count required for statistical sufficiency",
    )
    is_statistically_sufficient: bool = Field(
        description="False if test set size or diversity is inadequate for strong accuracy claims",
    )
    notes: str = Field(default="", description="Evaluation observations and data limitations")

    @model_validator(mode="before")
    @classmethod
    def _sync_counts(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "sample_count" in data and "test_sample_count" not in data:
                data["test_sample_count"] = data["sample_count"]
            elif "test_sample_count" in data and "sample_count" not in data:
                data["sample_count"] = data["test_sample_count"]
        return data
