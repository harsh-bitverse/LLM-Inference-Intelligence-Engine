"""Domain schemas for constrained configuration optimization and Pareto search.

Defines optimization objectives, candidate configurations, feasibility statuses,
evidence provenance ratings, constraints, search spaces, and Pareto recommendations.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_ii_engine.analysis.models import ModelResourceEstimate, RooflineAnalysisResult
from llm_ii_engine.core.enums import (
    EngineType,
    ProvenanceType,
    QuantizationType,
)
from llm_ii_engine.core.schemas import ModelSpec, WorkloadSpec
from llm_ii_engine.discovery.models import GPUSpecification
from llm_ii_engine.prediction.models import GeneralizationStatus


class OptimizationObjective(str, Enum):
    """Target performance, memory, or cost metrics to optimize."""

    MINIMIZE_TTFT = "minimize_ttft"
    MINIMIZE_TPOT = "minimize_tpot"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_VRAM = "minimize_vram"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_DECODE_THROUGHPUT = "maximize_decode_throughput"
    MINIMIZE_DURATION = "minimize_duration"
    MINIMIZE_CLIENT_DURATION = "minimize_client_duration"


class ObjectiveDirection(str, Enum):
    """Optimization direction for an objective metric."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class FeasibilityStatus(str, Enum):
    """Classification of whether a candidate configuration satisfies hard constraints.

    Distinguishes provable feasibility/infeasibility from analytical estimates
    and missing information.
    """

    FEASIBLE = "FEASIBLE"
    PROVABLY_INFEASIBLE = "PROVABLY_INFEASIBLE"
    PROVEN_INFEASIBLE = "PROVEN_INFEASIBLE"
    PREDICTED_SLA_MISS = "PREDICTED_SLA_MISS"
    ESTIMATED_INFEASIBLE = "ESTIMATED_INFEASIBLE"
    UNKNOWN_FEASIBILITY = "UNKNOWN_FEASIBILITY"


class EvidenceStatus(str, Enum):
    """Scientific confidence and provenance rating for an evaluated candidate metric."""

    MEASURED_EVALUATED = "MEASURED_EVALUATED"
    PREDICTED_EMPIRICAL = "PREDICTED_EMPIRICAL"
    THEORETICAL_ANALYTICAL = "THEORETICAL_ANALYTICAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SelectionPolicy(str, Enum):
    """Explicit policy for selecting a single candidate from the Pareto frontier."""

    NONE = "none"
    PRIORITIZE_OBJECTIVE = "prioritize_objective"
    BALANCED_KNEE_POINT = "balanced_knee_point"
    LOWEST_COST = "lowest_cost"


class CandidateConfig(BaseModel):
    """A discrete inference runtime configuration candidate for evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_spec: ModelSpec = Field(description="Target model architecture specification")
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE, description="Weight and activation quantization scheme"
    )
    concurrency: int = Field(default=1, ge=1, description="Maximum concurrent requests serviced")
    engine_type: EngineType = Field(
        default=EngineType.MOCK, description="Inference engine adapter runtime"
    )
    tensor_parallel_size: int = Field(
        default=1, ge=1, description="Number of GPUs used for tensor parallelism"
    )
    gpu_memory_utilization: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Target GPU VRAM allocation fraction (e.g. for vLLM)",
    )
    extra_args: dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific parameter overrides"
    )

    @property
    def config_id(self) -> str:
        """Deterministic unique configuration identifier."""
        return (
            f"{self.model_spec.model_id}:{self.quantization.value}:"
            f"c{self.concurrency}:{self.engine_type.value}:tp{self.tensor_parallel_size}"
        )


class OptimizationConstraints(BaseModel):
    """Hard operational and SLA constraints applied before Pareto ranking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_ttft_ms: float | None = Field(
        default=None, ge=0.0, description="Maximum acceptable Time-To-First-Token in milliseconds"
    )
    max_tpot_ms: float | None = Field(
        default=None, ge=0.0, description="Maximum acceptable Time-Per-Output-Token in milliseconds"
    )
    min_throughput_tok_s: float | None = Field(
        default=None, ge=0.0, description="Minimum acceptable output throughput in tokens/second"
    )
    max_vram_mb: float | None = Field(
        default=None, ge=0.0, description="Maximum device memory consumption limit in MiB"
    )
    max_cost_per_million_tokens: float | None = Field(
        default=None, ge=0.0, description="Maximum acceptable monetary cost per 1M tokens"
    )
    allowed_quantizations: list[QuantizationType] | None = Field(
        default=None, description="Whitelist of acceptable quantization formats"
    )
    allowed_engines: list[EngineType] | None = Field(
        default=None, description="Whitelist of acceptable engine runtimes"
    )
    max_concurrency: int | None = Field(
        default=None, ge=1, description="Maximum concurrency allowed by system deployment"
    )
    available_gpus: int | None = Field(
        default=None,
        ge=1,
        description="Number of available GPUs in host/cluster deployment for multi-GPU parallelism",
    )
    vram_safety_margin: float = Field(
        default=1.15,
        ge=1.0,
        description="Multiplier for memory safety headroom (default: 1.15 = 15% margin)",
    )
    require_cold_start_feasibility: bool = Field(
        default=False,
        description="If True, SLA constraints must be verified with cold-model/cold-cache evidence; warm-only evidence returns UNKNOWN_FEASIBILITY",
    )


class SearchSpace(BaseModel):
    """Discrete multi-dimensional search space defining candidate options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    models: list[ModelSpec] = Field(
        min_length=1, description="List of candidate model architectures"
    )
    quantizations: list[QuantizationType] = Field(
        default_factory=lambda: [QuantizationType.FP16],
        description="Quantization formats to evaluate",
    )
    concurrencies: list[int] = Field(
        default_factory=lambda: [1], description="Concurrency levels to evaluate"
    )
    engines: list[EngineType] = Field(
        default_factory=lambda: [EngineType.MOCK], description="Inference engines to evaluate"
    )
    tensor_parallel_sizes: list[int] = Field(
        default_factory=lambda: [1], description="Tensor parallel degrees to evaluate"
    )
    max_candidates: int = Field(
        default=1000,
        ge=1,
        description="Upper safety budget on generated candidates to prevent combinatorial explosion",
    )

    @model_validator(mode="after")
    def validate_search_dimensions(self) -> "SearchSpace":
        if any(c < 1 for c in self.concurrencies):
            raise ValueError("All concurrency values in search space must be >= 1")
        if any(tp < 1 for tp in self.tensor_parallel_sizes):
            raise ValueError("All tensor parallel sizes in search space must be >= 1")
        return self


class OptimizationRequest(BaseModel):
    """User specification for a constrained configuration optimization job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workload: WorkloadSpec = Field(
        description="Workload specification containing prompt and requested output limits"
    )
    search_space: SearchSpace = Field(description="Bounded candidate search space")
    objectives: list[OptimizationObjective] = Field(
        min_length=1, description="List of performance and resource objectives to optimize"
    )
    constraints: OptimizationConstraints = Field(
        default_factory=OptimizationConstraints,
        description="Hard feasibility limits and SLA thresholds",
    )
    hardware_catalog_id: str | None = Field(
        default=None, description="Hardware catalog identifier (e.g. 'nvidia-rtx-4090')"
    )
    hardware_spec: GPUSpecification | None = Field(
        default=None, description="Explicit target GPU hardware specification"
    )
    cost_per_gpu_hour: float | None = Field(
        default=None, ge=0.0, description="Nominal hardware rental or amortized cost per GPU hour"
    )
    selection_policy: SelectionPolicy = Field(
        default=SelectionPolicy.NONE,
        description="Policy for selecting a single recommended candidate from the Pareto frontier",
    )
    primary_objective: OptimizationObjective | None = Field(
        default=None,
        description="Primary objective used when selection_policy is PRIORITIZE_OBJECTIVE",
    )

    @model_validator(mode="after")
    def validate_cost_objective_requirements(self) -> "OptimizationRequest":
        if (
            OptimizationObjective.MINIMIZE_COST in self.objectives
            and self.cost_per_gpu_hour is None
        ):
            raise ValueError(
                "OptimizationObjective.MINIMIZE_COST requires 'cost_per_gpu_hour' to be explicitly supplied. "
                "Cost cannot be optimized from fabricated financial values."
            )
        if (
            self.selection_policy == SelectionPolicy.PRIORITIZE_OBJECTIVE
            and self.primary_objective is None
        ):
            raise ValueError(
                "selection_policy=PRIORITIZE_OBJECTIVE requires 'primary_objective' to be designated."
            )
        return self


class CandidateEvaluation(BaseModel):
    """Comprehensive evaluation outcomes for a candidate configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: CandidateConfig = Field(description="Configuration evaluated")
    feasibility_status: FeasibilityStatus = Field(description="Hard constraint feasibility outcome")
    rejection_reasons: list[str] = Field(
        default_factory=list, description="Documented reasons if candidate is infeasible or unknown"
    )

    # Physical Objective Values (raw measurements or predictions in standard units)
    objective_values: dict[OptimizationObjective, float] = Field(
        default_factory=dict, description="Calculated objective values in physical units"
    )
    objective_units: dict[OptimizationObjective, str] = Field(
        default_factory=dict, description="Physical units for each objective value"
    )
    evidence_status: dict[OptimizationObjective, EvidenceStatus] = Field(
        default_factory=dict, description="Scientific provenance rating for each objective"
    )
    provenance: dict[OptimizationObjective, ProvenanceType] = Field(
        default_factory=dict,
        description="Core provenance classification (MEASURED/PREDICTED/HEURISTIC)",
    )

    # Analytical and Diagnostic Evidence
    resource_estimate: ModelResourceEstimate | None = Field(
        default=None, description="Session 05 model resource and memory calculations"
    )
    roofline_result: RooflineAnalysisResult | None = Field(
        default=None, description="Session 05 Roofline operational intensity and bounds"
    )
    generalization_status: GeneralizationStatus | None = Field(
        default=None, description="Generalization and out-of-distribution assessment if predicted"
    )

    # Rationale and Explanations
    has_mixed_evidence: bool = Field(
        default=False,
        description="True if candidate objective metrics derive from different evidence tiers",
    )
    comparability_warning: str | None = Field(
        default=None,
        description="Warning emitted when candidate metrics combine measured and theoretical/uncalibrated tiers",
    )
    trade_off_summary: str = Field(
        default="", description="Human-readable trade-off rationale explaining frontier presence"
    )
    notes: list[str] = Field(default_factory=list, description="Diagnostic notes and assumptions")


class OptimizationResult(BaseModel):
    """Complete results of a constrained configuration search and Pareto ranking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, description="Unique optimization request identifier")
    summary: str = Field(description="High-level executive summary of optimization results")

    # Search & Candidate Statistics
    total_candidates_generated: int = Field(
        ge=0, description="Total candidate configurations generated"
    )
    total_candidates_evaluated: int = Field(ge=0, description="Total candidates evaluated")
    feasible_candidates_count: int = Field(ge=0, description="Number of fully feasible candidates")
    infeasible_candidates_count: int = Field(
        ge=0, description="Number of rejected infeasible candidates"
    )
    unknown_feasibility_count: int = Field(
        ge=0,
        description="Number of candidates with unknown feasibility due to missing hardware data",
    )

    # Pareto Optimal Recommendations
    pareto_frontier: list[CandidateEvaluation] = Field(
        default_factory=list, description="Non-dominated feasible candidate configurations"
    )
    recommended_candidate: CandidateEvaluation | None = Field(
        default=None, description="Single selected candidate if a selection policy was requested"
    )
    selection_rationale: str | None = Field(
        default=None, description="Explanation for single candidate selection"
    )

    # Rejection Diagnostics
    has_incomparable_evidence: bool = Field(
        default=False,
        description="True if the Pareto frontier combines candidates with materially distinct evidence tiers",
    )
    evidence_caveats: list[str] = Field(
        default_factory=list,
        description="Explicit caveats regarding evidence comparability across candidates",
    )
    rejections_by_reason: dict[str, int] = Field(
        default_factory=dict, description="Frequency counts of specific constraint violations"
    )
    all_evaluations: list[CandidateEvaluation] = Field(
        default_factory=list, description="Complete evaluation outcomes for all candidates"
    )
