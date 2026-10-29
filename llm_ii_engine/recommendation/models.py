"""Domain schemas for workload-aware inference configuration recommendation.

Defines typed request, response, metric predictions, evidence classification,
and SLA feasibility assessments for evidence-backed configuration recommendations.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ModelResidencyState,
    TokenCountSource,
)
from llm_ii_engine.optimizer.models import OptimizationObjective
from llm_ii_engine.prediction.models import GeneralizationStatus


class RecommendationStatus(str, Enum):
    """Overall status of a workload configuration recommendation."""

    RECOMMENDED = "RECOMMENDED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INFEASIBLE = "INFEASIBLE"
    UNSUPPORTED = "UNSUPPORTED"
    REJECTED = "REJECTED"


class OperatingRegime(str, Enum):
    """Identified physical operating regime governing inference performance."""

    COLD_START = "COLD_START"
    WARM_SHORT_BURST = "WARM_SHORT_BURST"
    WARM_SUSTAINED_GENERATION = "WARM_SUSTAINED_GENERATION"
    PROMPT_PREFIX_REUSE = "PROMPT_PREFIX_REUSE"
    CONSECUTIVE_SATURATED = "CONSECUTIVE_SATURATED"


class RecommendationEvidenceSource(str, Enum):
    """Evidence category supporting the recommended configuration and predictions.

    Maintains rigorous separation between measured historical facts and predictions.
    """

    DIRECTLY_MEASURED = "DIRECTLY_MEASURED"
    DIRECTLY_MEASURED_CONDITIONAL = "DIRECTLY_MEASURED_CONDITIONAL"
    HELD_OUT_VALIDATED_PREDICTION = "HELD_OUT_VALIDATED_PREDICTION"
    EXPLORATORY_PREDICTION = "EXPLORATORY_PREDICTION"
    ANALYTICAL_ESTIMATE = "ANALYTICAL_ESTIMATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class QualityStatus(str, Enum):
    """Evaluation status of model output quality and correctness constraints."""

    QUALITY_SATISFIED = "QUALITY_SATISFIED"
    QUALITY_VIOLATED = "QUALITY_VIOLATED"
    UNKNOWN_QUALITY_NO_EVALUATION = "UNKNOWN_QUALITY_NO_EVALUATION"


class ModelCompatibilityStatus(str, Enum):
    """Compatibility status of a model candidate with the host runtime and hardware."""

    COMPATIBLE = "COMPATIBLE"
    UNINSTALLED = "UNINSTALLED"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    INCOMPATIBLE_RUNTIME = "INCOMPATIBLE_RUNTIME"


class ModelQualityConstraintSpec(BaseModel):
    """Specification of quality, compliance, and memory constraints for candidate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    require_json_compliance: bool = Field(
        default=False, description="Require valid JSON parseability for structured tasks"
    )
    require_code_pass: bool = Field(
        default=False, description="Require unit test pass rate >= 80% on code generation tasks"
    )
    min_task_correctness: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum overall correctness score (0.0 to 1.0)"
    )
    min_json_compliance_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum JSON compliance rate (0.0 to 1.0)"
    )
    min_code_pass_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum code unit test pass rate (0.0 to 1.0)"
    )
    min_reasoning_accuracy: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum numerical reasoning accuracy (0.0 to 1.0)"
    )
    require_json_semantic_correctness: bool = Field(
        default=False, description="Require verified semantic field correctness on structured JSON tasks"
    )
    min_retrieval_accuracy: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum context retrieval accuracy (0.0 to 1.0)"
    )
    min_instruction_compliance: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum long-form instruction compliance score (0.0 to 1.0)"
    )
    target_category: str | None = Field(
        default=None, description="Optional target category identifier"
    )
    max_memory_budget_mb: float | None = Field(
        default=None, gt=0.0, description="Maximum allowed memory footprint in MiB"
    )
    preferred_model_family: str | None = Field(
        default=None, description="Optional filter for preferred model architecture family"
    )


class QualityAssessmentResult(BaseModel):
    """Quality and correctness evaluation outcome for a recommended model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    quality_status: QualityStatus = Field(
        default=QualityStatus.UNKNOWN_QUALITY_NO_EVALUATION,
        description="Whether quality constraints were met, violated, or unassessed",
    )
    json_compliance_rate: float | None = Field(
        default=None, description="Observed JSON compliance rate"
    )
    code_pass_rate: float | None = Field(
        default=None, description="Observed code pass rate"
    )
    reasoning_accuracy: float | None = Field(
        default=None, description="Observed numerical reasoning accuracy"
    )
    task_correctness_mean: float | None = Field(
        default=None, description="Mean correctness score across evaluated tasks"
    )
    evaluation_notes: list[str] = Field(
        default_factory=list, description="Diagnostic notes detailing quality evaluation"
    )


class WorkloadRecommendationRequest(BaseModel):
    """Typed request describing user workload and constraints for configuration recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(
        default="llama3.2:1b",
        description="Target model identifier (e.g. 'llama3.2:1b').",
    )
    hardware_catalog_id: str | None = Field(
        default="cpu-intel-i7-14700",
        description="Hardware platform identifier. Defaults to host Intel Core i7-14700 CPU.",
    )
    runtime: str = Field(
        default="ollama",
        description="Serving runtime engine (e.g. 'ollama').",
    )
    prompt_tokens: int = Field(
        ...,
        gt=0,
        description="Prompt length in tokens (must be greater than 0).",
    )
    prompt_tokens_provenance: TokenCountSource = Field(
        default=TokenCountSource.REQUESTED,
        description="Provenance of the prompt token count.",
    )
    expected_output_tokens: int = Field(
        ...,
        gt=0,
        description="Expected or maximum output generation length in tokens (must be greater than 0).",
    )
    concurrency: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Concurrent request count.",
    )
    expected_model_residency: ModelResidencyState = Field(
        default=ModelResidencyState.UNKNOWN,
        description="Expected model weight residency state in memory (WARM, COLD, or UNKNOWN).",
    )
    expected_prefix_cache: CacheStateClassification = Field(
        default=CacheStateClassification.UNKNOWN,
        description="Expected KV cache state for prompt prefix (COLD_PREFIX, WARM_PREFIX, UNKNOWN).",
    )
    latency_sla_ms: float | None = Field(
        default=None,
        gt=0,
        description="Optional hard client end-to-end latency constraint in milliseconds.",
    )
    decode_throughput_sla_tok_s: float | None = Field(
        default=None,
        gt=0,
        description="Optional minimum autoregressive decode throughput target in tokens/sec.",
    )
    ttft_sla_ms: float | None = Field(
        default=None,
        gt=0,
        description="Optional token-level Time-To-First-Token SLA. (Produces UNKNOWN feasibility on chunk-streaming).",
    )
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MINIMIZE_DURATION,
        description="Primary optimization objective (e.g. MINIMIZE_DURATION, MAXIMIZE_DECODE_THROUGHPUT).",
    )
    require_cold_start: bool = Field(
        default=False,
        description="Whether cold-start model weight loading must be explicitly accounted for.",
    )
    user_preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional user preferences or soft priorities.",
    )
    quality_constraints: ModelQualityConstraintSpec = Field(
        default_factory=ModelQualityConstraintSpec,
        description="Optional model quality, correctness, and memory constraints.",
    )

    @model_validator(mode="after")
    def validate_constraints(self) -> "WorkloadRecommendationRequest":
        """Validate logical consistency between requested constraints."""
        if self.require_cold_start and self.expected_model_residency == ModelResidencyState.WARM:
            raise ValueError(
                "Contradictory constraints: 'require_cold_start' is True, but 'expected_model_residency' is set to 'WARM'."
            )
        return self


class PredictedMetrics(BaseModel):
    """Decomposed performance forecasts supporting the recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_e2e_duration_ms: float | None = Field(
        default=None,
        description="Forecasted total client wall-clock duration in milliseconds.",
    )
    server_decode_throughput_tok_s: float | None = Field(
        default=None,
        description="Forecasted autoregressive generation throughput in tokens/second.",
    )
    client_e2e_output_throughput_tok_s: float | None = Field(
        default=None,
        description="Forecasted end-to-end user-perceived delivery rate in tokens/second.",
    )
    server_prefill_duration_ms: float | None = Field(
        default=None,
        description="Forecasted prompt evaluation duration in milliseconds.",
    )
    server_decode_duration_ms: float | None = Field(
        default=None,
        description="Forecasted autoregressive token evaluation duration in milliseconds.",
    )
    server_load_duration_ms: float | None = Field(
        default=None,
        description="Forecasted model weight loading duration in milliseconds.",
    )


class WorkloadRecommendationResponse(BaseModel):
    """Structured response containing evidence-backed configuration recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recommendation_id: str = Field(description="Unique identifier for this recommendation.")
    created_at: str = Field(description="ISO-8601 creation timestamp.")
    status: RecommendationStatus = Field(description="Recommendation outcome status.")
    workload_summary: dict[str, Any] = Field(description="Summary of the evaluated workload.")
    selected_configuration: dict[str, Any] | None = Field(
        default=None,
        description="Recommended inference engine and runtime configuration.",
    )
    predicted_metrics: PredictedMetrics = Field(
        description="Decomposed performance forecasts for the selected configuration.",
    )
    evidence_source: RecommendationEvidenceSource = Field(
        description="Classification of the scientific evidence supporting the recommendation.",
    )
    applicable_dataset: str = Field(
        description="Identifier of empirical dataset used for verification or baseline modeling.",
    )
    prediction_uncertainty_interval: list[float] | None = Field(
        default=None,
        description="Empirical uncertainty interval [lower, upper] for client_e2e_duration_ms (uncalibrated exploratory range).",
    )
    exploratory_prediction_range: list[float] | None = Field(
        default=None,
        description="Uncalibrated exploratory prediction range [lower, upper] in milliseconds.",
    )
    confidence_label_status: str = Field(
        default="EXPLORATORY_RANGE",
        description="Scientific qualification of the prediction range (e.g. 'EXPLORATORY_RANGE' or 'EMPIRICAL_ERROR_MARGIN').",
    )
    range_coverage_percent: float | None = Field(
        default=74.19,
        description="Empirically measured coverage percentage across historical held-out LOCO evaluations (~74.19% for 1x MAE).",
    )
    generalization_status: GeneralizationStatus = Field(
        description="Whether workload falls within empirical training distribution.",
    )
    ood_warnings: list[str] = Field(
        default_factory=list,
        description="Extrapolation or out-of-distribution warnings if applicable.",
    )
    cold_start_assumptions: str = Field(
        description="Assumptions regarding model weight residency and cold load latency.",
    )
    prefix_cache_assumptions: str = Field(
        description="Assumptions regarding prompt KV cache reuse.",
    )
    sla_assessments: dict[str, Any] = Field(
        description="Detailed feasibility assessments against all requested SLAs.",
    )
    relevant_alternatives: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative configurations evaluated and trade-off comparisons.",
    )
    explanation_summary: str = Field(
        description="Clear, user-facing rationale explaining why this configuration was recommended.",
    )
    influencing_factors: list[str] = Field(
        default_factory=list,
        description="Key factors and physical trade-offs influencing the recommendation.",
    )
    validation_benchmark_payload: dict[str, Any] | None = Field(
        default=None,
        description="Structured payload to execute an empirical benchmark validating this recommendation.",
    )
    operating_regime: OperatingRegime = Field(
        default=OperatingRegime.WARM_SHORT_BURST,
        description="Identified operating performance regime.",
    )
    worst_case_latency_ms: float | None = Field(
        default=None,
        description="Worst-case latency observed in matching empirical regime / workload.",
    )
    historical_sla_violation_frequency_pct: float | None = Field(
        default=None,
        description="Empirical frequency (%) of SLA violations for similar workloads under sustained or contention conditions.",
    )
    conservative_feasibility: str | None = Field(
        default=None,
        description="Risk-calibrated feasibility: ROBUST_FEASIBLE, NOMINAL_FEASIBLE_ROBUST_RISK, PREDICTED_SLA_MISS, PROVEN_INFEASIBLE, or UNKNOWN_FEASIBILITY.",
    )
    sla_risk_flag: str | None = Field(
        default=None,
        description="SLA risk flag if operating regime exhibits elevated variance or sustained slowdown.",
    )
    quality_assessment: QualityAssessmentResult | None = Field(
        default=None,
        description="Evaluation of model output quality, compliance, and correctness if quality constraints were specified.",
    )
    model_compatibility: ModelCompatibilityStatus = Field(
        default=ModelCompatibilityStatus.COMPATIBLE,
        description="Compatibility status of model with local runtime and hardware.",
    )
