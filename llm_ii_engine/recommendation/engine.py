"""Workload-aware inference configuration recommendation engine.

Integrates empirical historical benchmarks, physical workload-aware stage predictors,
out-of-distribution detection, SLA feasibility checks, and transparent explanations
into an evidence-backed configuration recommendation pipeline.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_ii_engine.benchmark.dataset_export import EmpiricalObservationRecord
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ExecutionMode,
    ModelResidencyState,
    ProvenanceType,
)
from llm_ii_engine.optimizer.models import (
    FeasibilityStatus,
)
from llm_ii_engine.prediction.models import GeneralizationStatus
from llm_ii_engine.recommendation.models import (
    OperatingRegime,
    PredictedMetrics,
    RecommendationEvidenceSource,
    RecommendationStatus,
    WorkloadRecommendationRequest,
    WorkloadRecommendationResponse,
)


class WorkloadRecommendationEngine:
    """Evaluates workloads, computes evidence-backed predictions, and recommends inference configurations."""

    # Physical parameters calibrated from empirical observations on Intel Core i7-14700 CPU
    PREFILL_MS_PER_TOKEN_COLD = 3.026  # ~330 tok/s prefill on CPU
    PREFILL_MS_PER_TOKEN_WARM = 0.042  # ~23,800 tok/s cache-replay
    DECODE_MS_PER_TOKEN = 44.012       # ~22.72 tok/s autoregressive decode
    COLD_START_LOAD_MS = 1037.74       # GGUF weight mapping from disk into RAM
    WARM_START_LOAD_MS = 0.0           # Resident in host RAM
    OVERHEAD_LOOPBACK_MS = 6.25        # HTTP framing & local loopback socket

    # Empirical dataset bounds
    PROMPT_MIN_TOKENS = 42
    PROMPT_MAX_TOKENS = 677
    OUTPUT_MIN_TOKENS = 16
    OUTPUT_MAX_TOKENS = 128

    def __init__(
        self,
        observations: list[EmpiricalObservationRecord] | None = None,
        dataset_path: Path | None = None,
    ) -> None:
        """Initialize engine with optional in-memory observations or dataset directory."""
        self.observations = observations or []
        if not self.observations and dataset_path:
            self._load_datasets(dataset_path)

    def _load_datasets(self, root: Path) -> None:
        """Load historical empirical datasets into memory for exact observation matching."""
        dataset_files = [
            root / "storage" / "datasets" / "expansion_llama3_2_1b_v2.jsonl",
            root / "storage" / "datasets" / "session_19_sustained_load_measurements.jsonl",
            root / "storage" / "datasets" / "session_15_cache_controlled_measurements.jsonl",
            root / "storage" / "datasets" / "pilot_llama3_2_1b.jsonl",
        ]
        for fpath in dataset_files:
            if fpath.exists():
                try:
                    with open(fpath, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                self.observations.append(
                                    EmpiricalObservationRecord.model_validate_json(line)
                                )
                except Exception:
                    pass

        # Also load Session 18 controlled validation measurements if available
        s18_path = root / "storage" / "experiments" / "session_18_controlled_validation.json"
        if s18_path.exists():
            try:
                import json
                with open(s18_path, encoding="utf-8") as f:
                    s18_data = json.load(f)
                    for wl in s18_data.get("live_validation_results", []):
                        w_id = wl["workload_id"]
                        for rep in wl["repetitions"]:
                            self.observations.append(
                                EmpiricalObservationRecord(
                                    record_id=f"obs-s18-{w_id}-{rep['repetition']}",
                                    experiment_id="exp-session-18-controlled-validation",
                                    suite_id="suite-session-18-validation",
                                    request_id=f"req-s18-{w_id}-{rep['repetition']}",
                                    repetition_index=rep["repetition"],
                                    is_warmup=False,
                                    model_id="llama3.2:1b",
                                    model_digest="baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878",
                                    runtime_name="ollama",
                                    runtime_version="0.40.11",
                                    adapter_version="1.0.0",
                                    device_name="CPU",
                                    cpu_model="Intel(R) Core(TM) i7-14700",
                                    host_ram_gb=31.69,
                                    prompt_tokens=rep.get("actual_prompt_tokens", 119),
                                    prompt_token_count_source="server_reported",
                                    requested_output_tokens=rep.get("actual_output_tokens", 128),
                                    actual_output_tokens=rep.get("actual_output_tokens", 128),
                                    token_count_source="server_reported",
                                    duration_ms=rep["client_e2e_duration_ms"],
                                    chunk_count=rep.get("actual_output_tokens", 128),
                                    client_e2e_duration_ms=rep["client_e2e_duration_ms"],
                                    server_decode_duration_ms=rep.get("server_decode_duration_ms"),
                                    server_prefill_duration_ms=rep.get("server_prefill_duration_ms"),
                                    server_load_duration_ms=rep.get("server_load_duration_ms", 0.0),
                                    server_decode_throughput_tok_s=rep.get("server_decode_throughput_tok_s"),
                                    model_residency_state="warm",
                                    cache_state="warm_prefix" if "WARM" in w_id else "cold_prefix",
                                    execution_mode="REAL",
                                    clock_source="MONOTONIC_REAL",
                                    provenance="MEASURED",
                                    success=True,
                                )
                            )
            except Exception:
                pass

    def _find_exact_empirical_match(
        self,
        request: WorkloadRecommendationRequest,
    ) -> tuple[EmpiricalObservationRecord | None, bool]:
        """Search historical benchmark records for an exact match on workload configuration.

        Returns:
            Tuple of (matched_record, is_conditional).
            is_conditional is True if runtime residency or cache state was requested
            as UNKNOWN, meaning the empirical match is conditional on the benchmark's
            measured residency/cache conditions rather than an unconditional exact match.
        """
        matching = []
        for obs in self.observations:
            if obs.is_warmup:
                continue
            if obs.execution_mode != ExecutionMode.REAL or obs.provenance != ProvenanceType.MEASURED:
                continue
            if obs.model_id != request.model_id:
                continue
            if obs.prompt_tokens != request.prompt_tokens:
                continue
            act_or_req_out = obs.requested_output_tokens or obs.actual_output_tokens
            if act_or_req_out != request.expected_output_tokens:
                continue
            if request.concurrency > 1:
                continue

            # Residency matching
            if (
                request.require_cold_start
                or request.expected_model_residency == ModelResidencyState.COLD
            ) and obs.model_residency_state != ModelResidencyState.COLD.value:
                continue
            if (
                request.expected_model_residency == ModelResidencyState.WARM
                and obs.model_residency_state != ModelResidencyState.WARM.value
            ):
                continue

            # Prefix cache matching
            if request.expected_prefix_cache in (
                CacheStateClassification.WARM_PREFIX,
                CacheStateClassification.COLD_PREFIX,
            ) and obs.cache_state and obs.cache_state != request.expected_prefix_cache.value:
                continue

            matching.append(obs)

        if not matching:
            return None, False

        # If residency or cache state is UNKNOWN, the match is conditional
        is_conditional = (
            request.expected_model_residency == ModelResidencyState.UNKNOWN
            or request.expected_prefix_cache == CacheStateClassification.UNKNOWN
        )

        # Select median observation across matching repetitions to represent nominal performance
        matching_sorted = sorted(
            matching,
            key=lambda o: (o.client_e2e_duration_ms or o.duration_ms or 0.0),
        )
        median_obs = matching_sorted[len(matching_sorted) // 2]
        return median_obs, is_conditional

    def recommend(
        self,
        request: WorkloadRecommendationRequest,
    ) -> WorkloadRecommendationResponse:
        """Evaluate workload request and produce structured, evidence-backed recommendation."""
        rec_id = f"rec-{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(UTC).isoformat()
        ood_warnings: list[str] = []
        sla_assessments: dict[str, Any] = {}
        influencing_factors: list[str] = []

        # 1. Compatibility verification
        if request.model_id not in ("llama3.2:1b", "llama3.2:1b-instruct-q8_0"):
            return WorkloadRecommendationResponse(
                recommendation_id=rec_id,
                created_at=now_iso,
                status=RecommendationStatus.UNSUPPORTED,
                workload_summary={
                    "model_id": request.model_id,
                    "prompt_tokens": request.prompt_tokens,
                    "expected_output_tokens": request.expected_output_tokens,
                },
                selected_configuration=None,
                predicted_metrics=PredictedMetrics(),
                evidence_source=RecommendationEvidenceSource.INSUFFICIENT_EVIDENCE,
                applicable_dataset="none",
                generalization_status=GeneralizationStatus(
                    is_out_of_distribution=True,
                    warnings=[f"Model '{request.model_id}' is not authorized or installed."],
                ),
                ood_warnings=[f"Model '{request.model_id}' is not authorized or installed."],
                cold_start_assumptions="Unverified model weights.",
                prefix_cache_assumptions="Unverified.",
                sla_assessments={},
                relevant_alternatives=[],
                explanation_summary=f"Configuration request rejected: model '{request.model_id}' is unsupported.",
                influencing_factors=["Only authorized local model 'llama3.2:1b' is supported."],
                validation_benchmark_payload=None,
            )

        # 2. Check for exact empirical benchmark match
        exact_match, is_conditional = self._find_exact_empirical_match(request)

        # 3. Detect Out-of-Distribution status
        is_ood = False
        if request.prompt_tokens > self.PROMPT_MAX_TOKENS:
            is_ood = True
            ood_warnings.append(
                f"Prompt tokens ({request.prompt_tokens}) exceed empirical benchmark maximum ({self.PROMPT_MAX_TOKENS}). Latency is extrapolated linearly."
            )
        elif request.prompt_tokens < self.PROMPT_MIN_TOKENS:
            ood_warnings.append(
                f"Prompt tokens ({request.prompt_tokens}) are below benchmark minimum ({self.PROMPT_MIN_TOKENS})."
            )

        if request.expected_output_tokens > self.OUTPUT_MAX_TOKENS:
            is_ood = True
            ood_warnings.append(
                f"Output tokens ({request.expected_output_tokens}) exceed empirical benchmark maximum ({self.OUTPUT_MAX_TOKENS})."
            )

        if request.concurrency > 1:
            is_ood = True
            ood_warnings.append(
                f"Concurrency ({request.concurrency}) exceeds single-client empirical verification range (1)."
            )

        extrapolated: list[str] = []
        if request.prompt_tokens > self.PROMPT_MAX_TOKENS or request.prompt_tokens < self.PROMPT_MIN_TOKENS:
            extrapolated.append("prompt_tokens")
        if request.expected_output_tokens > self.OUTPUT_MAX_TOKENS:
            extrapolated.append("output_tokens")
        if request.concurrency > 1:
            extrapolated.append("concurrency")

        gen_status = GeneralizationStatus(
            is_out_of_distribution=is_ood,
            warnings=list(ood_warnings),
            extrapolated_features=extrapolated,
        )

        # 4. Predict or assign metrics
        if exact_match is not None:
            if is_conditional:
                evidence_source = RecommendationEvidenceSource.DIRECTLY_MEASURED_CONDITIONAL
                influencing_factors.append(
                    "Direct empirical observation matched conditionally; runtime residency or cache state "
                    "was requested as UNKNOWN and assumed from benchmark conditions."
                )
            else:
                evidence_source = RecommendationEvidenceSource.DIRECTLY_MEASURED
                influencing_factors.append(
                    f"Direct empirical observation matched from experiment suite '{exact_match.suite_id or 'expansion_llama3_2_1b_v2'}'."
                )
            pred_e2e_ms = exact_match.client_e2e_duration_ms or exact_match.duration_ms
            pred_decode_tps = (
                exact_match.server_decode_throughput_tok_s
                or 22.72
            )
            pred_e2e_tps = (
                exact_match.client_e2e_output_throughput_tok_s
                or exact_match.output_tokens_per_second
            )
            pred_prefill_ms = exact_match.server_prefill_duration_ms or (
                request.prompt_tokens * self.PREFILL_MS_PER_TOKEN_COLD
            )
            pred_decode_ms = exact_match.server_decode_duration_ms or (
                request.expected_output_tokens * self.DECODE_MS_PER_TOKEN
            )
            pred_load_ms = exact_match.server_load_duration_ms or (
                self.COLD_START_LOAD_MS if exact_match.model_residency_state == "cold" else 0.0
            )
            applicable_ds = exact_match.suite_id or "expansion_llama3_2_1b_v2"
            uncertainty_interval = [round(pred_e2e_ms * 0.97, 1), round(pred_e2e_ms * 1.03, 1)]
        else:
            if is_ood:
                evidence_source = RecommendationEvidenceSource.EXPLORATORY_PREDICTION
            else:
                evidence_source = RecommendationEvidenceSource.HELD_OUT_VALIDATED_PREDICTION

            applicable_ds = "expansion_llama3_2_1b_v2"

            # Residency modeling
            if request.require_cold_start or request.expected_model_residency == ModelResidencyState.COLD:
                pred_load_ms = self.COLD_START_LOAD_MS
                cold_assump = "Model weight loading from disk to RAM required (+1,037.74 ms penalty)."
                influencing_factors.append("Cold start model load penalty included (+1,037.74 ms).")
            elif request.expected_model_residency == ModelResidencyState.WARM:
                pred_load_ms = self.WARM_START_LOAD_MS
                cold_assump = "Model weights assumed pre-loaded in host RAM (0.0 ms load penalty)."
                influencing_factors.append("Warm memory residency assumes zero model loading delay.")
            else:
                pred_load_ms = self.WARM_START_LOAD_MS
                cold_assump = (
                    "Model residency is UNKNOWN. Base forecast assumes warm residency; "
                    "a cold start will add +1,037.74 ms to end-to-end duration."
                )
                influencing_factors.append(
                    "Unknown model residency: baseline modeled warm, cold start penalty flagged."
                )

            # Prefix cache modeling
            if request.expected_prefix_cache == CacheStateClassification.WARM_PREFIX:
                pred_prefill_ms = round(request.prompt_tokens * self.PREFILL_MS_PER_TOKEN_WARM, 2)
                cache_assump = "Prompt prefix assumed resident in KV cache (~0.042 ms/token prefill)."
                influencing_factors.append("KV cache prefix hit accelerates prompt prefill by ~98%.")
            else:
                pred_prefill_ms = round(request.prompt_tokens * self.PREFILL_MS_PER_TOKEN_COLD, 2)
                cache_assump = "Cold prompt prefill evaluated at ~3.026 ms/token on CPU."
                influencing_factors.append("Full prompt prefill on CPU requires ~3.026 ms/token.")

            # Decode modeling
            pred_decode_ms = round(request.expected_output_tokens * self.DECODE_MS_PER_TOKEN, 2)
            pred_decode_tps = 22.72
            influencing_factors.append(
                "Autoregressive token generation modeled at stable ~22.72 tok/s on Intel i7-14700."
            )

            # End-to-end duration
            pred_e2e_ms = round(
                pred_load_ms + pred_prefill_ms + pred_decode_ms + self.OVERHEAD_LOOPBACK_MS,
                2,
            )
            pred_e2e_tps = round(
                request.expected_output_tokens / (pred_e2e_ms / 1000.0),
                2,
            ) if pred_e2e_ms > 0 else None

            # Uncertainty interval based on LOCO cross-validation MAE
            mae_margin = 385.88 if not is_ood else 385.88 * 2.0
            lower = max(round(pred_load_ms + 10.0, 1), round(pred_e2e_ms - mae_margin, 1))
            upper = round(pred_e2e_ms + mae_margin, 1)
            uncertainty_interval = [lower, upper]

        cold_assump = (
            "Cold start accounted for." if (request.require_cold_start or request.expected_model_residency == ModelResidencyState.COLD)
            else "Warm residency assumed." if request.expected_model_residency == ModelResidencyState.WARM
            else "Residency UNKNOWN (cold start would add ~1,038 ms)."
        )
        cache_assump = (
            "Warm prefix KV cache hit." if request.expected_prefix_cache == CacheStateClassification.WARM_PREFIX
            else "Cold prefix evaluation."
        )

        # 5. Operating Regime Classification & Risk Characterization (Session 19)
        if request.require_cold_start or request.expected_model_residency == ModelResidencyState.COLD:
            operating_regime = OperatingRegime.COLD_START
        elif request.expected_prefix_cache == CacheStateClassification.WARM_PREFIX:
            operating_regime = OperatingRegime.PROMPT_PREFIX_REUSE
        elif request.expected_output_tokens <= 32:
            operating_regime = OperatingRegime.WARM_SHORT_BURST
        else:
            operating_regime = OperatingRegime.WARM_SUSTAINED_GENERATION

        # Worst-case empirical latency bound for regime
        if operating_regime == OperatingRegime.COLD_START:
            worst_case_latency_ms = round(1200.0 + pred_prefill_ms + (request.expected_output_tokens * 48.0) + self.OVERHEAD_LOOPBACK_MS, 1)
        elif operating_regime == OperatingRegime.WARM_SHORT_BURST:
            worst_case_latency_ms = round(pred_e2e_ms * 1.35, 1)
        elif operating_regime == OperatingRegime.PROMPT_PREFIX_REUSE:
            if request.expected_output_tokens >= 64:
                worst_case_latency_ms = round(55.0 + (request.expected_output_tokens * 79.6) + self.OVERHEAD_LOOPBACK_MS, 1)
            else:
                worst_case_latency_ms = round(pred_e2e_ms * 1.30, 1)
        else:  # WARM_SUSTAINED_GENERATION
            # Saturated decode degrades from ~22.7 tok/s to 12.56 tok/s (79.6 ms/tok)
            worst_case_latency_ms = round(pred_prefill_ms + (request.expected_output_tokens * 79.6) + self.OVERHEAD_LOOPBACK_MS, 1)

        # 6. Hard Physical Constraints & Conservative SLA Feasibility Assessment
        overall_status = RecommendationStatus.RECOMMENDED
        feasibility_reasons: list[str] = []
        conservative_feasibility: str | None = None
        sla_risk_flag: str | None = None
        historical_violation_pct: float | None = None

        # 6.1 Independent Hard Physical Constraints
        MAX_CONTEXT_TOKENS = 131072
        total_tokens = request.prompt_tokens + request.expected_output_tokens
        if total_tokens > MAX_CONTEXT_TOKENS:
            sla_assessments["context_limit"] = {
                "constraint_type": "HARD_PHYSICAL",
                "max_tokens": MAX_CONTEXT_TOKENS,
                "requested_tokens": total_tokens,
                "status": FeasibilityStatus.PROVEN_INFEASIBLE.value,
                "reason": f"Requested sequence length ({total_tokens}) exceeds maximum model context window ({MAX_CONTEXT_TOKENS} tokens).",
            }
            overall_status = RecommendationStatus.INFEASIBLE
            feasibility_reasons.append(
                f"Requested total tokens ({total_tokens}) exceed model context window ({MAX_CONTEXT_TOKENS})."
            )

        # Theoretical hardware minimum decode latency (Memory-bandwidth bound on DDR5-5600 ~89.6 GB/s)
        MIN_DECODE_MS_PER_TOK = 14.73
        theoretical_min_ms = request.expected_output_tokens * MIN_DECODE_MS_PER_TOK

        # 6.2 Latency SLA Assessment
        if request.latency_sla_ms is not None:
            # Check historical SLA violation frequency on similar observed workloads
            matching_obs = [
                obs for obs in self.observations
                if not obs.is_warmup
                and obs.model_id == request.model_id
                and abs((obs.actual_output_tokens or 0) - request.expected_output_tokens) <= 16
            ]
            if matching_obs:
                viols = sum(
                    1 for obs in matching_obs
                    if (obs.client_e2e_duration_ms or obs.duration_ms) > request.latency_sla_ms
                )
                historical_violation_pct = round(100.0 * viols / len(matching_obs), 1)
            else:
                if request.latency_sla_ms < pred_e2e_ms:
                    historical_violation_pct = 100.0
                elif request.latency_sla_ms < worst_case_latency_ms:
                    historical_violation_pct = 50.0
                else:
                    historical_violation_pct = 0.0

            if request.latency_sla_ms < theoretical_min_ms:
                violation = round(theoretical_min_ms - request.latency_sla_ms, 1)
                conservative_feasibility = FeasibilityStatus.PROVEN_INFEASIBLE.value
                sla_assessments["latency_sla"] = {
                    "constraint_type": "HARD",
                    "target_ms": request.latency_sla_ms,
                    "theoretical_min_ms": round(theoretical_min_ms, 1),
                    "forecast_ms": pred_e2e_ms,
                    "worst_case_forecast_ms": worst_case_latency_ms,
                    "status": FeasibilityStatus.PROVEN_INFEASIBLE.value,
                    "conservative_feasibility": conservative_feasibility,
                    "historical_sla_violation_frequency_pct": historical_violation_pct,
                    "violation_ms": violation,
                    "physical_proof": (
                        f"Latency SLA ({request.latency_sla_ms} ms) violates theoretical memory-bandwidth "
                        f"lower bound ({theoretical_min_ms:.1f} ms for {request.expected_output_tokens} tokens)."
                    ),
                }
                overall_status = RecommendationStatus.INFEASIBLE
                feasibility_reasons.append(
                    f"Requested latency SLA ({request.latency_sla_ms} ms) is physically impossible on host memory bus (minimum {theoretical_min_ms:.1f} ms)."
                )
            elif (
                exact_match is not None
                and not is_conditional
                and pred_e2e_ms > request.latency_sla_ms
            ):
                meas_dur = exact_match.client_e2e_duration_ms or exact_match.duration_ms
                gap = round(meas_dur - request.latency_sla_ms, 1)
                conservative_feasibility = FeasibilityStatus.PROVEN_INFEASIBLE.value
                sla_assessments["latency_sla"] = {
                    "constraint_type": "HARD",
                    "target_ms": request.latency_sla_ms,
                    "measured_ms": meas_dur,
                    "forecast_ms": pred_e2e_ms,
                    "worst_case_forecast_ms": worst_case_latency_ms,
                    "status": FeasibilityStatus.PROVEN_INFEASIBLE.value,
                    "conservative_feasibility": conservative_feasibility,
                    "historical_sla_violation_frequency_pct": historical_violation_pct,
                    "violation_ms": gap,
                    "proof_basis": "DIRECT_MEASUREMENT",
                }
                overall_status = RecommendationStatus.INFEASIBLE
                feasibility_reasons.append(
                    f"Directly measured latency ({meas_dur} ms) exceeds requested SLA ({request.latency_sla_ms} ms) by {gap} ms."
                )
            elif pred_e2e_ms <= request.latency_sla_ms:
                headroom = round(request.latency_sla_ms - pred_e2e_ms, 1)
                mae_margin = 385.88 if not is_ood else 771.76

                # Check if worst-case latency exceeds SLA
                if worst_case_latency_ms > request.latency_sla_ms:
                    conservative_feasibility = "NOMINAL_FEASIBLE_ROBUST_RISK"
                    sla_risk_flag = "ELEVATED_SUSTAINED_LOAD_RISK"
                    influencing_factors.append(
                        f"Elevated sustained-load risk: worst-case latency ({worst_case_latency_ms} ms) exceeds SLA ({request.latency_sla_ms} ms)."
                    )
                else:
                    conservative_feasibility = "ROBUST_FEASIBLE"

                assessment = {
                    "constraint_type": "HARD",
                    "target_ms": request.latency_sla_ms,
                    "forecast_ms": pred_e2e_ms,
                    "worst_case_forecast_ms": worst_case_latency_ms,
                    "status": FeasibilityStatus.FEASIBLE.value,
                    "conservative_feasibility": conservative_feasibility,
                    "historical_sla_violation_frequency_pct": historical_violation_pct,
                    "headroom_ms": headroom,
                }
                risk_warnings: list[str] = []
                if sla_risk_flag is not None:
                    assessment["sla_risk_flag"] = sla_risk_flag
                    risk_warnings.append(
                        f"SLA target ({request.latency_sla_ms} ms) is met by nominal forecast ({pred_e2e_ms} ms), "
                        f"but exceeded by worst-case sustained-load latency ({worst_case_latency_ms} ms). "
                        f"Under consecutive multi-request bursts or CPU contention, decode throughput degrades to ~12.56 tok/s."
                    )
                if headroom < mae_margin:
                    risk_warnings.append(
                        f"Narrow SLA headroom ({headroom} ms) is within empirical error margin ({mae_margin} ms); "
                        "satisfaction is sensitive to runtime variance."
                    )
                    influencing_factors.append("Narrow SLA headroom: prospective forecast is close to constraint threshold.")

                if risk_warnings:
                    assessment["risk_warning"] = " | ".join(risk_warnings)

                sla_assessments["latency_sla"] = assessment
            else:
                gap = round(pred_e2e_ms - request.latency_sla_ms, 1)
                if is_ood:
                    conservative_feasibility = FeasibilityStatus.UNKNOWN_FEASIBILITY.value
                    sla_assessments["latency_sla"] = {
                        "constraint_type": "HARD",
                        "target_ms": request.latency_sla_ms,
                        "forecast_ms": pred_e2e_ms,
                        "worst_case_forecast_ms": worst_case_latency_ms,
                        "status": FeasibilityStatus.UNKNOWN_FEASIBILITY.value,
                        "conservative_feasibility": conservative_feasibility,
                        "historical_sla_violation_frequency_pct": historical_violation_pct,
                        "estimated_violation_ms": gap,
                        "qualification_note": (
                            f"Workload is out-of-distribution (P={request.prompt_tokens}, O={request.expected_output_tokens}); "
                            f"extrapolated latency ({pred_e2e_ms} ms) exceeds SLA ({request.latency_sla_ms} ms) by {gap} ms, "
                            "but SLA miss cannot be mathematically proven without live empirical benchmarking."
                        ),
                    }
                    if overall_status == RecommendationStatus.RECOMMENDED:
                        overall_status = RecommendationStatus.INCONCLUSIVE
                    feasibility_reasons.append(
                        f"Workload is out-of-distribution; extrapolated SLA miss ({gap} ms) cannot be proven."
                    )
                else:
                    conservative_feasibility = FeasibilityStatus.PREDICTED_SLA_MISS.value
                    sla_assessments["latency_sla"] = {
                        "constraint_type": "HARD",
                        "target_ms": request.latency_sla_ms,
                        "forecast_ms": pred_e2e_ms,
                        "worst_case_forecast_ms": worst_case_latency_ms,
                        "status": FeasibilityStatus.PREDICTED_SLA_MISS.value,
                        "conservative_feasibility": conservative_feasibility,
                        "historical_sla_violation_frequency_pct": historical_violation_pct,
                        "violation_ms": gap,
                    }
                    overall_status = RecommendationStatus.INFEASIBLE
                    feasibility_reasons.append(
                        f"Forecasted latency ({pred_e2e_ms} ms) exceeds requested SLA ({request.latency_sla_ms} ms) by {gap} ms under in-distribution model."
                    )

        # 5.3 Decode Throughput SLA
        if request.decode_throughput_sla_tok_s is not None:
            if (
                exact_match is not None
                and not is_conditional
                and exact_match.server_decode_throughput_tok_s
                and exact_match.server_decode_throughput_tok_s < request.decode_throughput_sla_tok_s
            ):
                meas_tps = exact_match.server_decode_throughput_tok_s
                gap = round(request.decode_throughput_sla_tok_s - meas_tps, 1)
                sla_assessments["decode_throughput_sla"] = {
                    "constraint_type": "HARD",
                    "target_tok_s": request.decode_throughput_sla_tok_s,
                    "measured_tok_s": meas_tps,
                    "status": FeasibilityStatus.PROVEN_INFEASIBLE.value,
                    "deficit_tok_s": gap,
                    "proof_basis": "DIRECT_MEASUREMENT",
                }
                overall_status = RecommendationStatus.INFEASIBLE
                feasibility_reasons.append(
                    f"Directly measured decode throughput ({meas_tps} tok/s) fails target ({request.decode_throughput_sla_tok_s} tok/s) by {gap} tok/s."
                )
            elif pred_decode_tps >= request.decode_throughput_sla_tok_s:
                sla_assessments["decode_throughput_sla"] = {
                    "constraint_type": "HARD",
                    "target_tok_s": request.decode_throughput_sla_tok_s,
                    "forecast_tok_s": pred_decode_tps,
                    "status": FeasibilityStatus.FEASIBLE.value,
                }
            else:
                gap = round(request.decode_throughput_sla_tok_s - pred_decode_tps, 1)
                if is_ood:
                    sla_assessments["decode_throughput_sla"] = {
                        "constraint_type": "HARD",
                        "target_tok_s": request.decode_throughput_sla_tok_s,
                        "forecast_tok_s": pred_decode_tps,
                        "status": FeasibilityStatus.UNKNOWN_FEASIBILITY.value,
                        "estimated_deficit_tok_s": gap,
                        "qualification_note": "Workload is out-of-distribution; throughput deficit cannot be proven.",
                    }
                    if overall_status == RecommendationStatus.RECOMMENDED:
                        overall_status = RecommendationStatus.INCONCLUSIVE
                    feasibility_reasons.append(
                        f"Workload is out-of-distribution; forecasted decode throughput deficit ({gap} tok/s) cannot be proven."
                    )
                else:
                    sla_assessments["decode_throughput_sla"] = {
                        "constraint_type": "HARD",
                        "target_tok_s": request.decode_throughput_sla_tok_s,
                        "forecast_tok_s": pred_decode_tps,
                        "status": FeasibilityStatus.PREDICTED_SLA_MISS.value,
                        "deficit_tok_s": gap,
                    }
                    overall_status = RecommendationStatus.INFEASIBLE
                    feasibility_reasons.append(
                        f"Forecasted decode throughput ({pred_decode_tps} tok/s) fails target ({request.decode_throughput_sla_tok_s} tok/s) by {gap} tok/s."
                    )

        # 5.4 TTFT SLA (Ollama text chunk streaming does not provide verified token boundaries)
        if request.ttft_sla_ms is not None:
            sla_assessments["ttft_sla"] = {
                "constraint_type": "HARD",
                "target_ms": request.ttft_sla_ms,
                "forecast_ms": None,
                "status": FeasibilityStatus.UNKNOWN_FEASIBILITY.value,
                "limitation_note": (
                    "Ollama chunk streaming emits text byte chunks without verified discrete token timestamps. "
                    "True token-level TTFT cannot be verified; client-observed TTFC is recorded instead."
                ),
            }
            if overall_status == RecommendationStatus.RECOMMENDED:
                overall_status = RecommendationStatus.INCONCLUSIVE
                feasibility_reasons.append(
                    "Token-level TTFT SLA cannot be verified on Ollama streaming protocol (emits text chunks without discrete token timestamps)."
                )

        # 5.5 Severe OOD with tight SLA
        if is_ood and overall_status == RecommendationStatus.RECOMMENDED:
            overall_status = RecommendationStatus.INCONCLUSIVE
            feasibility_reasons.append(
                "Workload is out-of-distribution; available empirical evidence is insufficient for SLA guarantee without live benchmarking."
            )

        # 6. Selected Configuration Specification
        selected_config = {
            "model_id": request.model_id,
            "quantization": "q8_0",
            "runtime": "ollama",
            "runtime_version": "0.40.11",
            "device": "CPU",
            "cpu_model": "Intel(R) Core(TM) i7-14700",
            "concurrency": request.concurrency,
            "options": {
                "num_predict": request.expected_output_tokens,
                "temperature": 0.0,
                "num_threads": 20,
            },
        }

        # 7. Alternatives & Trade-offs
        alternatives = [
            {
                "name": "Warm-Residency Baseline",
                "residency": "WARM",
                "expected_duration_ms": round(
                    pred_prefill_ms + pred_decode_ms + self.OVERHEAD_LOOPBACK_MS, 2
                ),
                "trade_off": "Requires model to be kept resident in RAM; eliminates 1,038 ms cold load delay.",
            },
            {
                "name": "Cold-Start Baseline",
                "residency": "COLD",
                "expected_duration_ms": round(
                    self.COLD_START_LOAD_MS + pred_prefill_ms + pred_decode_ms + self.OVERHEAD_LOOPBACK_MS, 2
                ),
                "trade_off": "Loads model from disk on every dispatch; adds 1,038 ms cold start latency.",
            },
            {
                "name": "Prefix-Cached Configuration",
                "cache_state": "WARM_PREFIX",
                "expected_duration_ms": round(
                    (request.prompt_tokens * self.PREFILL_MS_PER_TOKEN_WARM) + pred_decode_ms + self.OVERHEAD_LOOPBACK_MS, 2
                ),
                "trade_off": "Requires re-using identical prompt prefix; reduces prefill duration to ~0.04 ms/token.",
            },
        ]

        # 8. Explanation Summary
        if overall_status == RecommendationStatus.RECOMMENDED:
            explanation = (
                f"Recommended Ollama with '{request.model_id}' (Q8_0) on Intel Core i7-14700 CPU. "
                f"Workload ({request.prompt_tokens} prompt tokens, {request.expected_output_tokens} output tokens) "
                f"yields a forecasted client end-to-end duration of {pred_e2e_ms:.1f} ms "
                f"and stable autoregressive decode throughput of {pred_decode_tps:.2f} tok/s. "
                f"All requested SLAs are satisfied with positive feasibility margins."
            )
        elif overall_status == RecommendationStatus.INCONCLUSIVE:
            explanation = (
                f"Recommendation is inconclusive: {'; '.join(feasibility_reasons)}. "
                f"Forecasted operational metrics: {pred_e2e_ms:.1f} ms duration, {pred_decode_tps:.2f} tok/s decode throughput. "
                f"A live validation benchmark is required to verify real-world behavior."
            )
        elif overall_status == RecommendationStatus.INFEASIBLE:
            explanation = (
                f"Configuration is infeasible for the requested SLA: {'; '.join(feasibility_reasons)}. "
                f"The CPU-bound execution on Intel Core i7-14700 cannot satisfy the required constraint."
            )
        else:
            explanation = f"Configuration status: {overall_status.value}."

        # 9. Validation Benchmark Payload
        val_payload = {
            "model_id": request.model_id,
            "engine": "ollama",
            "prompt_tokens": request.prompt_tokens,
            "output_tokens": request.expected_output_tokens,
            "concurrency": request.concurrency,
            "repetition_count": 3,
            "warmup_count": 1,
            "temperature": 0.0,
            "provenance_target": "REAL",
        }

        return WorkloadRecommendationResponse(
            recommendation_id=rec_id,
            created_at=now_iso,
            status=overall_status,
            workload_summary={
                "model_id": request.model_id,
                "prompt_tokens": request.prompt_tokens,
                "expected_output_tokens": request.expected_output_tokens,
                "concurrency": request.concurrency,
                "objective": request.objective.value,
            },
            selected_configuration=selected_config,
            predicted_metrics=PredictedMetrics(
                client_e2e_duration_ms=pred_e2e_ms,
                server_decode_throughput_tok_s=pred_decode_tps,
                client_e2e_output_throughput_tok_s=pred_e2e_tps,
                server_prefill_duration_ms=pred_prefill_ms,
                server_decode_duration_ms=pred_decode_ms,
                server_load_duration_ms=pred_load_ms,
            ),
            evidence_source=evidence_source,
            applicable_dataset=applicable_ds,
            prediction_uncertainty_interval=uncertainty_interval,
            exploratory_prediction_range=uncertainty_interval,
            confidence_label_status="EXPLORATORY_RANGE",
            range_coverage_percent=74.19 if not is_ood else 90.32,
            generalization_status=gen_status,
            ood_warnings=ood_warnings,
            cold_start_assumptions=cold_assump,
            prefix_cache_assumptions=cache_assump,
            sla_assessments=sla_assessments,
            relevant_alternatives=alternatives,
            explanation_summary=explanation,
            influencing_factors=influencing_factors,
            validation_benchmark_payload=val_payload,
            operating_regime=operating_regime,
            worst_case_latency_ms=worst_case_latency_ms,
            historical_sla_violation_frequency_pct=historical_violation_pct,
            conservative_feasibility=conservative_feasibility,
            sla_risk_flag=sla_risk_flag,
        )
