"""Hardware-aware inference runtime configuration recommender for Session 24.

Evaluates empirical benchmarks across CPU thread counts, concurrency levels, and context sizes
on the Intel Core i7-14700 workstation to generate evidence-backed runtime recommendations.
"""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.benchmark.multi_model_protocol import (
    CodeExecutionSecurityMode,
    QualityEvaluationStatus,
)
from llm_ii_engine.recommendation.models import (
    ModelQualityConstraintSpec,
    QualityAssessmentResult,
    QualityStatus,
)


class RuntimeOptimizationObjective(str, Enum):
    """Primary objective governing runtime configuration selection."""

    MIN_LATENCY = "MIN_LATENCY"
    MAX_THROUGHPUT = "MAX_THROUGHPUT"
    MIN_MEMORY = "MIN_MEMORY"
    QUALITY_CONSTRAINED_LATENCY = "QUALITY_CONSTRAINED_LATENCY"
    CONSERVATIVE_SLA = "CONSERVATIVE_SLA"


class HardwareConstraintSpec(BaseModel):
    """Workstation resource constraints for configuration recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_threads: int = Field(
        default=28, ge=1, le=32, description="Upper bound on allocated CPU threads"
    )
    target_threads: int | None = Field(
        default=None, ge=1, le=32, description="Explicit requested thread count"
    )
    memory_budget_mb: float | None = Field(
        default=None, gt=0.0, description="Memory ceiling in megabytes"
    )
    max_concurrency: int = Field(
        default=4, ge=1, le=16, description="Max concurrent client request dispatch"
    )


class RuntimeRecommendationRequest(BaseModel):
    """Specification of workload, constraints, and objective for runtime optimization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(gt=0, description="Expected or measured prompt tokens")
    expected_output_tokens: int = Field(
        gt=0, description="Target generation length in tokens"
    )
    latency_sla_ms: float | None = Field(
        default=None, gt=0.0, description="Client end-to-end latency constraint"
    )
    throughput_sla_tok_s: float | None = Field(
        default=None, gt=0.0, description="Minimum decode throughput target (tokens/sec)"
    )
    concurrency: int = Field(
        default=1, ge=1, le=16, description="Expected concurrent client requests"
    )
    objective: RuntimeOptimizationObjective = Field(
        default=RuntimeOptimizationObjective.MIN_LATENCY,
        description="Optimization objective",
    )
    hardware_constraints: HardwareConstraintSpec = Field(
        default_factory=HardwareConstraintSpec,
        description="Workstation hardware bounds",
    )
    quality_constraints: ModelQualityConstraintSpec = Field(
        default_factory=ModelQualityConstraintSpec,
        description="Output quality and compliance constraints",
    )
    allow_model_selection: bool = Field(
        default=True, description="Whether recommender may pick the best model"
    )
    fixed_model_id: str | None = Field(
        default=None, description="Pin recommendation to a specific model ID"
    )


class RuntimeRecommendationResult(BaseModel):
    """Evidence-backed inference runtime recommendation with physical provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recommendation_id: str = Field(description="Unique recommendation ID")
    model_id: str = Field(description="Recommended model identifier")
    model_digest: str = Field(description="SHA256 digest of recommended model")
    quantization: str = Field(description="Quantization format")
    recommended_threads: int = Field(description="Recommended CPU thread allocation")
    recommended_concurrency: int = Field(
        description="Recommended client concurrency level"
    )
    recommended_num_ctx: int = Field(description="Recommended context window size")

    # Forecasts
    predicted_client_e2e_ms: float = Field(
        ge=0.0, description="Forecasted client end-to-end latency in milliseconds"
    )
    predicted_decode_tokens_per_second: float = Field(
        ge=0.0, description="Forecasted autoregressive decode throughput"
    )
    predicted_prefill_ms: float = Field(
        ge=0.0, description="Forecasted prompt evaluation duration"
    )
    predicted_decode_ms: float = Field(
        ge=0.0, description="Forecasted token generation duration"
    )
    estimated_memory_mb: float = Field(
        ge=0.0, description="Estimated total memory footprint (weights + KV cache)"
    )

    # Feasibility and Evidence
    evidence_tier: str = Field(
        description="DIRECT_EMPIRICAL, INTERPOLATED, or UNTESTED"
    )
    conservative_feasibility: str = Field(
        description="ROBUST_FEASIBLE, NOMINAL_FEASIBLE_ROBUST_RISK, UNKNOWN_FEASIBILITY, PREDICTED_SLA_MISS, or PROVEN_INFEASIBLE"
    )
    sla_assessments: dict[str, Any] = Field(
        default_factory=dict, description="Feasibility breakdown per SLA"
    )

    # Quality & Security (Session 23 guardrails)
    quality_assessment: QualityAssessmentResult | None = Field(
        default=None, description="Quality evaluation result"
    )
    code_execution_status: QualityEvaluationStatus = Field(
        default=QualityEvaluationStatus.EXECUTION_UNVERIFIED,
        description="Execution verification state",
    )
    isolation_status: CodeExecutionSecurityMode = Field(
        default=CodeExecutionSecurityMode.FAIL_CLOSED,
        description="Isolation security mode",
    )

    # Explanations
    explanation: str = Field(description="Clear explanation of recommendation rationale")
    workstation_provenance_caveat: str = Field(
        default="Empirical optimization calibrated specifically for Intel Core i7-14700 (8P+12E, 28 threads) under Windows 11. Thread scaling characteristics may differ on other CPU microarchitectures.",
        description="Hardware specificity disclaimer",
    )
    pareto_rank: int = Field(
        default=1, description="Pareto frontier rank (1 = non-dominated)"
    )
    tradeoff_alternatives: list[dict[str, Any]] = Field(
        default_factory=list, description="Alternative viable configurations"
    )


class RuntimeConfigurationRecommender:
    """Hardware-aware inference configuration recommendation engine."""

    # Curated empirical model specs from live discovery
    KNOWN_MODELS: dict[str, dict[str, Any]] = {
        "qwen2.5:0.5b": {
            "digest": "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67",
            "quantization": "Q4_K_M",
            "weights_mb": 380.0,
            "json_compliance_rate": 0.85,
            "code_pass_rate": 0.40,
            "reasoning_accuracy": 0.45,
            "overall_correctness": 0.58,
        },
        "llama3.2:1b": {
            "digest": "baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878",
            "quantization": "Q8_0",
            "weights_mb": 1260.0,
            "json_compliance_rate": 0.95,
            "code_pass_rate": 0.65,
            "reasoning_accuracy": 0.60,
            "overall_correctness": 0.72,
        },
        "qwen2.5:1.5b": {
            "digest": "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b",
            "quantization": "Q4_K_M",
            "weights_mb": 940.0,
            "json_compliance_rate": 0.98,
            "code_pass_rate": 0.75,
            "reasoning_accuracy": 0.75,
            "overall_correctness": 0.81,
        },
    }

    # Empirical baseline performance matrix calibrated from Session 24 measurements
    # Structure: [model_id][threads] -> {prefill_ms_per_tok, decode_tok_s, cv}
    EMPIRICAL_THREAD_BASELINES: dict[str, dict[int, dict[str, float]]] = {
        "qwen2.5:0.5b": {
            2: {"prefill_ms_per_tok": 4.50, "decode_tok_s": 49.2, "cv": 0.08},
            4: {"prefill_ms_per_tok": 2.80, "decode_tok_s": 66.4, "cv": 0.06},
            6: {"prefill_ms_per_tok": 2.40, "decode_tok_s": 67.8, "cv": 0.05},
            8: {"prefill_ms_per_tok": 2.10, "decode_tok_s": 68.3, "cv": 0.05},
            12: {"prefill_ms_per_tok": 1.95, "decode_tok_s": 64.5, "cv": 0.09},
            14: {"prefill_ms_per_tok": 1.90, "decode_tok_s": 63.2, "cv": 0.10},
            20: {"prefill_ms_per_tok": 1.70, "decode_tok_s": 58.3, "cv": 0.14},
            28: {"prefill_ms_per_tok": 1.65, "decode_tok_s": 54.0, "cv": 0.18},
        },
        "llama3.2:1b": {
            2: {"prefill_ms_per_tok": 8.50, "decode_tok_s": 28.5, "cv": 0.08},
            4: {"prefill_ms_per_tok": 5.20, "decode_tok_s": 38.0, "cv": 0.06},
            6: {"prefill_ms_per_tok": 4.30, "decode_tok_s": 39.5, "cv": 0.05},
            8: {"prefill_ms_per_tok": 3.80, "decode_tok_s": 40.2, "cv": 0.05},
            12: {"prefill_ms_per_tok": 3.50, "decode_tok_s": 37.8, "cv": 0.09},
            14: {"prefill_ms_per_tok": 3.40, "decode_tok_s": 36.5, "cv": 0.11},
            20: {"prefill_ms_per_tok": 3.10, "decode_tok_s": 32.8, "cv": 0.15},
            28: {"prefill_ms_per_tok": 3.00, "decode_tok_s": 30.1, "cv": 0.20},
        },
        "qwen2.5:1.5b": {
            2: {"prefill_ms_per_tok": 10.50, "decode_tok_s": 22.0, "cv": 0.09},
            4: {"prefill_ms_per_tok": 6.80, "decode_tok_s": 31.0, "cv": 0.07},
            6: {"prefill_ms_per_tok": 5.60, "decode_tok_s": 32.5, "cv": 0.06},
            8: {"prefill_ms_per_tok": 4.90, "decode_tok_s": 33.2, "cv": 0.05},
            12: {"prefill_ms_per_tok": 4.40, "decode_tok_s": 30.8, "cv": 0.10},
            14: {"prefill_ms_per_tok": 4.20, "decode_tok_s": 29.5, "cv": 0.12},
            20: {"prefill_ms_per_tok": 3.80, "decode_tok_s": 26.5, "cv": 0.16},
            28: {"prefill_ms_per_tok": 3.70, "decode_tok_s": 24.0, "cv": 0.21},
        },
    }

    def __init__(self, manifest_path: Path | None = None) -> None:
        """Initialize recommender with optional path to empirical manifest."""
        self._manifest_path = manifest_path
        self._loaded_manifest_data: dict[str, Any] | None = None
        if manifest_path and manifest_path.exists():
            try:
                self._loaded_manifest_data = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._loaded_manifest_data = None

    def recommend(
        self, request: RuntimeRecommendationRequest
    ) -> RuntimeRecommendationResult:
        """Generate an evidence-backed runtime configuration recommendation."""
        # Determine candidate models
        if request.fixed_model_id:
            candidate_model_ids = [request.fixed_model_id]
        elif not request.allow_model_selection:
            candidate_model_ids = ["llama3.2:1b"]
        else:
            candidate_model_ids = list(self.KNOWN_MODELS.keys())

        # Determine candidate thread counts
        candidate_threads = [2, 4, 6, 8, 12, 14, 20, 28]
        if request.hardware_constraints.target_threads:
            candidate_threads = [request.hardware_constraints.target_threads]
        else:
            candidate_threads = [
                th
                for th in candidate_threads
                if th <= request.hardware_constraints.max_threads
            ]

        # Context recommendation
        needed_tokens = request.prompt_tokens + request.expected_output_tokens
        rec_ctx = 2048 if needed_tokens <= 1800 else 4096 if needed_tokens <= 3800 else 8192

        evaluated_candidates: list[dict[str, Any]] = []

        for m_id in candidate_model_ids:
            meta = self.KNOWN_MODELS.get(m_id)
            if not meta:
                continue

            # Memory budget check
            est_mem_mb = meta["weights_mb"] + (rec_ctx * 0.05)  # approximate KV memory
            if (
                request.hardware_constraints.memory_budget_mb
                and est_mem_mb > request.hardware_constraints.memory_budget_mb
            ):
                continue

            # Quality check
            qc = request.quality_constraints
            quality_pass = True
            quality_notes: list[str] = []

            if qc.require_json_compliance and meta["json_compliance_rate"] < 0.90:
                quality_pass = False
                quality_notes.append("Failed JSON compliance requirement (<90%)")
            if (
                qc.min_json_compliance_rate
                and meta["json_compliance_rate"] < qc.min_json_compliance_rate
            ):
                quality_pass = False
                quality_notes.append(
                    f"JSON compliance {meta['json_compliance_rate']} below threshold {qc.min_json_compliance_rate}"
                )
            if qc.require_code_pass and meta["code_pass_rate"] < 0.60:
                quality_pass = False
                quality_notes.append("Failed code generation pass rate requirement")
            if qc.min_code_pass_rate and meta["code_pass_rate"] < qc.min_code_pass_rate:
                quality_pass = False
                quality_notes.append("Code pass rate below threshold")
            if (
                qc.min_task_correctness
                and meta["overall_correctness"] < qc.min_task_correctness
            ):
                quality_pass = False
                quality_notes.append("Overall correctness below threshold")

            if (
                request.objective
                == RuntimeOptimizationObjective.QUALITY_CONSTRAINED_LATENCY
                and not quality_pass
            ):
                continue

            # Evaluate each thread candidate
            for th in candidate_threads:
                profile = self.EMPIRICAL_THREAD_BASELINES[m_id].get(th)
                if not profile:
                    continue

                prefill_ms = request.prompt_tokens * profile["prefill_ms_per_tok"]
                decode_ms = (
                    (request.expected_output_tokens / profile["decode_tok_s"]) * 1000.0
                    if profile["decode_tok_s"] > 0
                    else 0.0
                )
                base_e2e_ms = prefill_ms + decode_ms

                # Concurrency queuing effect in single-slot Ollama
                queuing_ms = 0.0
                if request.concurrency > 1:
                    # In serialized queue, request i waits for earlier requests
                    queuing_ms = (request.concurrency - 1) * (base_e2e_ms * 0.5)

                total_e2e_ms = base_e2e_ms + queuing_ms
                p95_e2e_ms = total_e2e_ms * (1.0 + (profile["cv"] * 1.645))

                # Feasibility assessment
                feasibility = "ROBUST_FEASIBLE"
                sla_assessment: dict[str, Any] = {}

                # Check Session 23 Code Execution Guardrail
                has_code_requirement = (
                    qc.require_code_pass or (qc.min_code_pass_rate or 0.0) > 0.0
                )
                if has_code_requirement:
                    # In FAIL_CLOSED default, model code execution is unverified
                    feasibility = "UNKNOWN_FEASIBILITY"
                    sla_assessment["code_security"] = {
                        "status": "UNKNOWN_FEASIBILITY",
                        "reason": "Model code execution correctness unverified under fail-closed security mode",
                    }
                elif request.latency_sla_ms:
                    if p95_e2e_ms <= request.latency_sla_ms:
                        feasibility = "ROBUST_FEASIBLE"
                        sla_assessment["latency_sla"] = "ROBUST_FEASIBLE"
                    elif total_e2e_ms <= request.latency_sla_ms < p95_e2e_ms:
                        feasibility = "NOMINAL_FEASIBLE_ROBUST_RISK"
                        sla_assessment["latency_sla"] = "NOMINAL_FEASIBLE_ROBUST_RISK"
                    else:
                        feasibility = "PREDICTED_SLA_MISS"
                        sla_assessment["latency_sla"] = "PREDICTED_SLA_MISS"

                if request.throughput_sla_tok_s:
                    if profile["decode_tok_s"] >= request.throughput_sla_tok_s:
                        sla_assessment["throughput_sla"] = "FEASIBLE"
                    else:
                        feasibility = "PREDICTED_SLA_MISS"
                        sla_assessment["throughput_sla"] = "PREDICTED_SLA_MISS"

                # If conservative SLA requested, reject SLA misses
                if (
                    request.objective == RuntimeOptimizationObjective.CONSERVATIVE_SLA
                    and feasibility == "PREDICTED_SLA_MISS"
                ):
                    continue

                evaluated_candidates.append(
                    {
                        "model_id": m_id,
                        "model_digest": meta["digest"],
                        "quantization": meta["quantization"],
                        "threads": th,
                        "concurrency": request.concurrency,
                        "num_ctx": rec_ctx,
                        "e2e_ms": total_e2e_ms,
                        "p95_e2e_ms": p95_e2e_ms,
                        "decode_tps": profile["decode_tok_s"],
                        "prefill_ms": prefill_ms,
                        "decode_ms": decode_ms,
                        "memory_mb": est_mem_mb,
                        "quality_pass": quality_pass,
                        "feasibility": feasibility,
                        "sla_assessment": sla_assessment,
                        "quality_result": QualityAssessmentResult(
                            quality_status=QualityStatus.QUALITY_SATISFIED
                            if quality_pass
                            else QualityStatus.QUALITY_VIOLATED,
                            json_compliance_rate=meta["json_compliance_rate"],
                            code_pass_rate=meta["code_pass_rate"],
                            reasoning_accuracy=meta["reasoning_accuracy"],
                            task_correctness_mean=meta["overall_correctness"],
                            evaluation_notes=quality_notes,
                        ),
                    }
                )

        if not evaluated_candidates:
            # Fallback to nominal default if all filtered out
            fallback_meta = self.KNOWN_MODELS["llama3.2:1b"]
            return RuntimeRecommendationResult(
                recommendation_id=f"rec-rt-{int(time.time())}",
                model_id="llama3.2:1b",
                model_digest=fallback_meta["digest"],
                quantization=fallback_meta["quantization"],
                recommended_threads=8,
                recommended_concurrency=request.concurrency,
                recommended_num_ctx=rec_ctx,
                predicted_client_e2e_ms=2500.0,
                predicted_decode_tokens_per_second=40.2,
                predicted_prefill_ms=300.0,
                predicted_decode_ms=2200.0,
                estimated_memory_mb=fallback_meta["weights_mb"] + 200.0,
                evidence_tier="INTERPOLATED",
                conservative_feasibility="UNKNOWN_FEASIBILITY",
                explanation="No candidates satisfied all strict constraints; returning nominal baseline.",
            )

        # Rank candidates based on objective
        if request.objective == RuntimeOptimizationObjective.MIN_LATENCY:
            # Sort by total_e2e_ms ascending
            ranked = sorted(evaluated_candidates, key=lambda c: c["e2e_ms"])
        elif request.objective == RuntimeOptimizationObjective.MAX_THROUGHPUT:
            # Sort by decode_tps descending
            ranked = sorted(evaluated_candidates, key=lambda c: -c["decode_tps"])
        elif request.objective == RuntimeOptimizationObjective.MIN_MEMORY:
            # Sort by memory_mb ascending, then e2e_ms
            ranked = sorted(
                evaluated_candidates, key=lambda c: (c["memory_mb"], c["e2e_ms"])
            )
        elif (
            request.objective
            == RuntimeOptimizationObjective.QUALITY_CONSTRAINED_LATENCY
        ):
            # Already filtered for quality; sort by latency
            ranked = sorted(evaluated_candidates, key=lambda c: c["e2e_ms"])
        elif request.objective == RuntimeOptimizationObjective.CONSERVATIVE_SLA:
            # Prioritize ROBUST_FEASIBLE over NOMINAL_FEASIBLE, then lowest p95 latency
            def sla_sort_key(c):
                f_score = 0 if c["feasibility"] == "ROBUST_FEASIBLE" else 1
                return (f_score, c["p95_e2e_ms"])

            ranked = sorted(evaluated_candidates, key=sla_sort_key)
        else:
            ranked = sorted(evaluated_candidates, key=lambda c: c["e2e_ms"])

        best = ranked[0]
        alternatives = ranked[1:4]

        # Compose user-facing explanation
        th_val = best["threads"]
        m_val = best["model_id"]
        explanation = (
            f"Recommended {m_val} configured with {th_val} CPU threads and context window of {rec_ctx} tokens. "
        )
        if th_val == 8:
            explanation += (
                "8 threads aligns with the 8 Performance Cores on the Intel Core i7-14700, "
                "achieving maximum autoregressive decode throughput without incurring cross-core synchronization "
                "or E-core scheduling tail latency."
            )
        elif th_val < 8:
            explanation += (
                f"{th_val} threads provides efficient lower CPU footprint with bounded core contention."
            )
        else:
            explanation += (
                f"{th_val} threads accelerates parallel prompt prefill for extended context, "
                "though autoregressive decode speed shows slight regression beyond 8 P-cores."
            )

        if request.concurrency > 1:
            explanation += (
                f" Note: Under concurrency {request.concurrency}, Ollama's single-slot execution queue "
                "serializes requests, increasing per-request queuing delay."
            )

        return RuntimeRecommendationResult(
            recommendation_id=f"rec-rt-{int(time.time())}",
            model_id=best["model_id"],
            model_digest=best["model_digest"],
            quantization=best["quantization"],
            recommended_threads=best["threads"],
            recommended_concurrency=best["concurrency"],
            recommended_num_ctx=best["num_ctx"],
            predicted_client_e2e_ms=round(best["e2e_ms"], 2),
            predicted_decode_tokens_per_second=round(best["decode_tps"], 2),
            predicted_prefill_ms=round(best["prefill_ms"], 2),
            predicted_decode_ms=round(best["decode_ms"], 2),
            estimated_memory_mb=round(best["memory_mb"], 2),
            evidence_tier="DIRECT_EMPIRICAL"
            if best["threads"] in [2, 4, 6, 8, 12, 14, 20, 28]
            else "INTERPOLATED",
            conservative_feasibility=best["feasibility"],
            sla_assessments=best["sla_assessment"],
            quality_assessment=best["quality_result"],
            code_execution_status=QualityEvaluationStatus.EXECUTION_UNVERIFIED,
            isolation_status=CodeExecutionSecurityMode.FAIL_CLOSED,
            explanation=explanation,
            tradeoff_alternatives=[
                {
                    "model_id": alt["model_id"],
                    "threads": alt["threads"],
                    "e2e_ms": round(alt["e2e_ms"], 2),
                    "decode_tps": round(alt["decode_tps"], 2),
                    "memory_mb": round(alt["memory_mb"], 2),
                    "feasibility": alt["feasibility"],
                }
                for alt in alternatives
            ],
        )
