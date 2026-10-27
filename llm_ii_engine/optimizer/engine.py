"""Constrained configuration optimization and multi-objective Pareto search orchestrator.

Integrates bounded candidate generation, Session 05 analytical resource estimation,
Roofline performance ceilings, Session 06 empirical ML predictions, hard feasibility
checks, and Pareto frontier ranking into an explainable recommendation pipeline.
"""

import statistics
import uuid

from llm_ii_engine.analysis.resource_estimator import ModelResourceEstimator
from llm_ii_engine.analysis.roofline import RooflineAnalyzer
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ExecutionMode,
    ModelResidencyState,
    ProvenanceType,
    QuantizationType,
    TokenCountSource,
)
from llm_ii_engine.discovery.catalog import match_gpu_name
from llm_ii_engine.discovery.models import GPUSpecification
from llm_ii_engine.optimizer.feasibility import FeasibilityChecker
from llm_ii_engine.optimizer.models import (
    CandidateConfig,
    CandidateEvaluation,
    EvidenceStatus,
    FeasibilityStatus,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
)
from llm_ii_engine.optimizer.pareto import ParetoEngine
from llm_ii_engine.optimizer.search import SearchSpaceGenerator
from llm_ii_engine.prediction.baselines import BasePredictor
from llm_ii_engine.prediction.models import (
    GeneralizationStatus,
    PredictionRecord,
    PredictionTarget,
)


class ConfigurationOptimizer:
    """Orchestrates candidate generation, evaluation, constraint checking, and Pareto optimization."""

    def __init__(
        self,
        predictors: dict[PredictionTarget, BasePredictor] | None = None,
        empirical_observations: list[PredictionRecord] | None = None,
    ) -> None:
        """Initialize optimizer with optional fitted empirical predictors and benchmark records.

        Args:
            predictors: Optional dict mapping prediction targets to fitted BasePredictor models.
            empirical_observations: Optional historical benchmark records for exact matching.
        """
        self.predictors = predictors or {}
        self.empirical_observations = empirical_observations or []

    def _resolve_gpu_specification(
        self,
        request: OptimizationRequest,
    ) -> GPUSpecification | None:
        """Resolve the target GPU specification from explicit spec or catalog ID."""
        if request.hardware_spec is not None:
            return request.hardware_spec
        if request.hardware_catalog_id:
            return match_gpu_name(request.hardware_catalog_id)
        return None

    def _match_empirical_observation(
        self,
        config: CandidateConfig,
        request: OptimizationRequest,
        gpu_spec: GPUSpecification | None,
    ) -> PredictionRecord | None:
        """Find exact historical measured observations for this configuration and aggregate across repetitions."""
        prompt_len = request.workload.prompt_tokens
        requested_out = request.workload.output_tokens
        target_hw = gpu_spec.catalog_id if gpu_spec else request.hardware_catalog_id
        target_digest = config.extra_args.get("model_digest")

        matching_recs: list[PredictionRecord] = []

        for rec in self.empirical_observations:
            # Warmup runs must never be matched as steady-state evidence
            if rec.is_warmup:
                continue

            if (
                rec.execution_mode != ExecutionMode.REAL
                or rec.provenance != ProvenanceType.MEASURED
            ):
                continue

            # Model and architecture match
            if rec.model_id != config.model_spec.model_id:
                continue
            if (
                config.model_spec.parameter_count_billion is not None
                and rec.parameter_count_billion is not None
                and abs(rec.parameter_count_billion - config.model_spec.parameter_count_billion)
                > 1e-4
            ):
                continue

            # Model digest match when specified
            if target_digest and rec.model_digest and rec.model_digest != target_digest:
                continue

            # Quantization, engine, and concurrency match
            if rec.quantization != config.quantization:
                continue
            if rec.concurrency != config.concurrency:
                continue
            if rec.engine_type != config.engine_type:
                continue

            # Workload prompt exact match (no loose delta tolerance)
            if prompt_len is not None and rec.prompt_tokens != prompt_len or prompt_len is None and rec.prompt_tokens is not None:
                continue

            # Workload output tokens match
            if requested_out > 0:
                rec_out = (
                    rec.requested_output_tokens
                    if rec.requested_output_tokens is not None
                    else rec.actual_output_tokens
                )
                if rec_out != requested_out:
                    continue

            # Hardware platform match
            if target_hw is not None:
                if rec.hardware_catalog_id != target_hw:
                    continue
            elif rec.hardware_catalog_id is not None:
                continue

            # Tensor parallelism: single-GPU benchmark does not match multi-GPU
            if config.tensor_parallel_size > 1:
                continue

            matching_recs.append(rec)

        if not matching_recs:
            return None

        if len(matching_recs) == 1:
            return matching_recs[0]

        # Aggregate multiple measured repetitions using median
        rep0 = matching_recs[0]

        def _med(vals: list[float | None]) -> float | None:
            valid = [v for v in vals if v is not None]
            return float(statistics.median(valid)) if valid else None

        thru = _med([r.output_tokens_per_second for r in matching_recs])
        dur = _med([r.duration_ms for r in matching_recs])
        ttfc = _med([r.ttfc_ms for r in matching_recs])
        ttft = _med([r.ttft_ms for r in matching_recs])
        tpot = _med([r.tpot_ms for r in matching_recs])
        chunk_lat = _med([r.chunk_latency_ms for r in matching_recs])
        vram = _med([r.peak_vram_mb for r in matching_recs])
        s_prefill = _med([r.server_prompt_eval_duration_ms for r in matching_recs])
        s_decode = _med([r.server_eval_duration_ms for r in matching_recs])

        act_toks = [
            r.actual_output_tokens for r in matching_recs if r.actual_output_tokens is not None
        ]
        med_act_tok = int(statistics.median(act_toks)) if act_toks else rep0.actual_output_tokens

        return PredictionRecord(
            record_id=f"agg-median-{len(matching_recs)}-reps",
            experiment_id=rep0.experiment_id,
            suite_id=rep0.suite_id,
            request_id=None,
            model_id=rep0.model_id,
            model_digest=rep0.model_digest,
            engine_type=rep0.engine_type,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            is_warmup=False,
            parameter_count_billion=rep0.parameter_count_billion,
            num_layers=rep0.num_layers,
            num_attention_heads=rep0.num_attention_heads,
            num_kv_heads=rep0.num_kv_heads,
            hidden_dimension=rep0.hidden_dimension,
            head_dimension=rep0.head_dimension,
            quantization=rep0.quantization,
            bytes_per_parameter=rep0.bytes_per_parameter,
            hardware_catalog_id=rep0.hardware_catalog_id,
            gpu_memory_capacity_gb=rep0.gpu_memory_capacity_gb,
            gpu_memory_bandwidth_gbps=rep0.gpu_memory_bandwidth_gbps,
            gpu_peak_compute_tflops=rep0.gpu_peak_compute_tflops,
            host_ram_total_gb=rep0.host_ram_total_gb,
            concurrency=rep0.concurrency,
            prompt_tokens=rep0.prompt_tokens,
            prompt_token_count_source=rep0.prompt_token_count_source,
            requested_output_tokens=rep0.requested_output_tokens,
            actual_output_tokens=med_act_tok,
            output_tokens=med_act_tok,
            token_count_source=rep0.token_count_source,
            planned_sequence_length=rep0.planned_sequence_length,
            total_sequence_length=(
                (rep0.prompt_tokens + med_act_tok) if rep0.prompt_tokens is not None else None
            ),
            ttft_ms=ttft,
            ttfc_ms=ttfc,
            tpot_ms=tpot,
            chunk_latency_ms=chunk_lat,
            output_tokens_per_second=thru,
            peak_vram_mb=vram,
            duration_ms=dur,
            server_prompt_eval_duration_ms=s_prefill,
            server_eval_duration_ms=s_decode,
            is_eligible_for_ttft=rep0.is_eligible_for_ttft and ttft is not None,
            is_eligible_for_tpot=rep0.is_eligible_for_tpot and tpot is not None,
            is_eligible_for_chunk_latency=rep0.is_eligible_for_chunk_latency and chunk_lat is not None,
            is_eligible_for_throughput=rep0.is_eligible_for_throughput and thru is not None,
            is_eligible_for_vram=rep0.is_eligible_for_vram and vram is not None,
            is_eligible_for_duration=rep0.is_eligible_for_duration and dur is not None,
            is_eligible_for_ttfc=rep0.is_eligible_for_ttfc and ttfc is not None,
            is_eligible_for_prefill=rep0.is_eligible_for_prefill and s_prefill is not None,
            is_eligible_for_decode=rep0.is_eligible_for_decode and s_decode is not None,
        )

    def _evaluate_candidate(
        self,
        config: CandidateConfig,
        request: OptimizationRequest,
        gpu_spec: GPUSpecification | None,
    ) -> CandidateEvaluation:
        """Evaluate resource footprints, performance metrics, and hard constraints for a single candidate."""
        prompt_len = request.workload.prompt_tokens or 128
        requested_out = request.workload.output_tokens
        notes: list[str] = []

        # 1. Session 05 Analytical Resource Estimation
        res_est = None
        if config.model_spec.parameter_count_billion is not None:
            res_est = ModelResourceEstimator.estimate_model_resources(
                model_spec=config.model_spec,
                prompt_tokens=prompt_len,
                output_tokens=requested_out,
                batch_size=config.concurrency,
                weight_precision=config.quantization,
            )
            notes.extend(res_est.assumptions)
        else:
            notes.append(
                "Model parameter_count_billion is unspecified; analytical resource estimation unavailable."
            )

        # 2. Session 05 Roofline Ceilings
        roofline = None
        if (
            gpu_spec is not None
            and config.model_spec.parameter_count_billion is not None
            and res_est is not None
        ):
            prec = (
                "int8_dense"
                if config.quantization in (QuantizationType.INT8, QuantizationType.GGUF_Q8_0)
                else "fp16_dense"
            )
            oi = res_est.operational_intensity_decode or 1.0
            try:
                roofline = RooflineAnalyzer.evaluate_roofline(
                    gpu_spec=gpu_spec,
                    operational_intensity=oi,
                    precision=prec,
                    parameter_count_billion=config.model_spec.parameter_count_billion,
                )
            except ValueError as err:
                notes.append(f"Roofline ceiling could not be calculated: {err}")

        # 3. Check for exact empirical benchmark match
        matched_obs = self._match_empirical_observation(config, request, gpu_spec)

        # 4. Evaluate requested objective metrics
        obj_values: dict[OptimizationObjective, float] = {}
        obj_units: dict[OptimizationObjective, str] = {}
        evidence_status: dict[OptimizationObjective, EvidenceStatus] = {}
        provenance: dict[OptimizationObjective, ProvenanceType] = {}
        gen_status: GeneralizationStatus | None = None

        # Build candidate record for empirical predictors
        cand_record = PredictionRecord(
            record_id=f"eval-{uuid.uuid4().hex[:8]}",
            suite_id="opt-session",
            model_id=config.model_spec.model_id,
            engine_type=config.engine_type,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.HEURISTIC,
            parameter_count_billion=config.model_spec.parameter_count_billion,
            num_layers=config.model_spec.num_layers,
            num_attention_heads=config.model_spec.num_attention_heads,
            num_kv_heads=config.model_spec.num_kv_heads,
            hidden_dimension=config.model_spec.hidden_dimension,
            head_dimension=config.model_spec.head_dimension,
            quantization=config.quantization,
            hardware_catalog_id=gpu_spec.catalog_id if gpu_spec else None,
            gpu_memory_capacity_gb=gpu_spec.memory_capacity_gb if gpu_spec else None,
            gpu_memory_bandwidth_gbps=gpu_spec.theoretical_memory_bandwidth_gbps
            if gpu_spec
            else None,
            gpu_peak_compute_tflops=gpu_spec.compute.fp16_dense_tflops if gpu_spec else None,
            concurrency=config.concurrency,
            prompt_tokens=prompt_len,
            prompt_token_count_source=TokenCountSource.REQUESTED,
            requested_output_tokens=requested_out,
            actual_output_tokens=requested_out,
            planned_sequence_length=prompt_len + requested_out,
            total_sequence_length=prompt_len + requested_out,
        )

        # Determine all objectives that need evaluation (requested objectives + any objectives required by constraints or cost)
        objectives_to_eval = list(request.objectives)
        if (
            request.constraints.max_ttft_ms is not None
            and OptimizationObjective.MINIMIZE_TTFT not in objectives_to_eval
        ):
            objectives_to_eval.append(OptimizationObjective.MINIMIZE_TTFT)
        if (
            request.constraints.max_tpot_ms is not None
            and OptimizationObjective.MINIMIZE_TPOT not in objectives_to_eval
        ):
            objectives_to_eval.append(OptimizationObjective.MINIMIZE_TPOT)
        if (
            request.constraints.min_throughput_tok_s is not None
            and OptimizationObjective.MAXIMIZE_THROUGHPUT not in objectives_to_eval
        ):
            objectives_to_eval.append(OptimizationObjective.MAXIMIZE_THROUGHPUT)
        if (
            request.constraints.max_vram_mb is not None
            and OptimizationObjective.MINIMIZE_VRAM not in objectives_to_eval
        ):
            objectives_to_eval.append(OptimizationObjective.MINIMIZE_VRAM)
        if (
            OptimizationObjective.MINIMIZE_COST in objectives_to_eval
            or request.cost_per_gpu_hour is not None
        ) and OptimizationObjective.MAXIMIZE_THROUGHPUT not in objectives_to_eval:
            objectives_to_eval.append(OptimizationObjective.MAXIMIZE_THROUGHPUT)

        # Enforce evaluation order: evaluate Throughput, TTFT, TPOT, VRAM before Cost
        eval_order = [
            OptimizationObjective.MINIMIZE_TTFT,
            OptimizationObjective.MINIMIZE_TPOT,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
            OptimizationObjective.MINIMIZE_VRAM,
            OptimizationObjective.MINIMIZE_COST,
        ]
        sorted_objectives = [obj for obj in eval_order if obj in objectives_to_eval]
        for obj in objectives_to_eval:
            if obj not in sorted_objectives:
                sorted_objectives.append(obj)

        for obj in sorted_objectives:
            # Objective: TTFT (ms)
            if obj == OptimizationObjective.MINIMIZE_TTFT:
                obj_units[obj] = "ms"
                if matched_obs and matched_obs.ttft_ms is not None:
                    obj_values[obj] = matched_obs.ttft_ms
                    if request.constraints.require_cold_start_feasibility and (
                        matched_obs.model_residency_state == ModelResidencyState.WARM
                        or matched_obs.cache_state
                        in (
                            CacheStateClassification.WARM_PREFIX,
                            CacheStateClassification.PARTIAL_PREFIX,
                        )
                    ):
                        evidence_status[obj] = EvidenceStatus.INSUFFICIENT_EVIDENCE
                        notes.append(
                            "Matched real observation is from warm-residency/warm-prefix execution; "
                            "insufficient evidence for cold-start TTFT SLA."
                        )
                    else:
                        evidence_status[obj] = EvidenceStatus.MEASURED_EVALUATED
                        provenance[obj] = ProvenanceType.MEASURED
                elif (
                    matched_obs
                    and matched_obs.ttfc_ms is not None
                    and PredictionTarget.TTFT_MS not in self.predictors
                    and not (roofline and res_est and res_est.prefill_flops and roofline.peak_compute_tflops > 0)
                ):
                    notes.append(
                        f"Matched real benchmark observation has measured TTFC={matched_obs.ttfc_ms:.2f} ms, "
                        "but token-level TTFT is unmeasured in streaming chunk adapter; TTFT SLA cannot be verified."
                    )
                elif PredictionTarget.TTFT_MS in self.predictors:
                    pred = self.predictors[PredictionTarget.TTFT_MS]
                    out = pred.predict(cand_record)
                    obj_values[obj] = out.predicted_value
                    gen_status = out.generalization_status
                    evidence_status[obj] = (
                        EvidenceStatus.INSUFFICIENT_EVIDENCE
                        if out.generalization_status.is_out_of_distribution
                        else EvidenceStatus.PREDICTED_EMPIRICAL
                    )
                    provenance[obj] = ProvenanceType.PREDICTED
                elif (
                    roofline
                    and res_est
                    and res_est.prefill_flops
                    and roofline.peak_compute_tflops > 0
                ):
                    val = (res_est.prefill_flops / (roofline.peak_compute_tflops * 1e12)) * 1000.0
                    obj_values[obj] = round(val, 2)
                    evidence_status[obj] = EvidenceStatus.THEORETICAL_ANALYTICAL
                    provenance[obj] = ProvenanceType.HEURISTIC

            # Objective: TPOT (ms)
            elif obj == OptimizationObjective.MINIMIZE_TPOT:
                obj_units[obj] = "ms"
                if matched_obs and matched_obs.tpot_ms is not None:
                    obj_values[obj] = matched_obs.tpot_ms
                    evidence_status[obj] = EvidenceStatus.MEASURED_EVALUATED
                    provenance[obj] = ProvenanceType.MEASURED
                elif (
                    matched_obs
                    and matched_obs.chunk_latency_ms is not None
                    and PredictionTarget.TPOT_MS not in self.predictors
                    and not (roofline and roofline.theoretical_max_tokens_per_second)
                ):
                    notes.append(
                        f"Matched real benchmark observation has measured chunk latency={matched_obs.chunk_latency_ms:.2f} ms, "
                        "but token-level TPOT is unmeasured; TPOT SLA cannot be verified."
                    )
                elif PredictionTarget.TPOT_MS in self.predictors:
                    pred = self.predictors[PredictionTarget.TPOT_MS]
                    out = pred.predict(cand_record)
                    obj_values[obj] = out.predicted_value
                    gen_status = out.generalization_status
                    evidence_status[obj] = (
                        EvidenceStatus.INSUFFICIENT_EVIDENCE
                        if out.generalization_status.is_out_of_distribution
                        else EvidenceStatus.PREDICTED_EMPIRICAL
                    )
                    provenance[obj] = ProvenanceType.PREDICTED
                elif roofline and roofline.theoretical_max_tokens_per_second:
                    val = 1000.0 / roofline.theoretical_max_tokens_per_second
                    obj_values[obj] = round(val, 2)
                    evidence_status[obj] = EvidenceStatus.THEORETICAL_ANALYTICAL
                    provenance[obj] = ProvenanceType.HEURISTIC

            # Objective: Throughput (tok/s)
            elif obj == OptimizationObjective.MAXIMIZE_THROUGHPUT:
                obj_units[obj] = "tok/s"
                if matched_obs and matched_obs.output_tokens_per_second is not None:
                    obj_values[obj] = matched_obs.output_tokens_per_second
                    evidence_status[obj] = EvidenceStatus.MEASURED_EVALUATED
                    provenance[obj] = ProvenanceType.MEASURED
                elif PredictionTarget.OUTPUT_THROUGHPUT_TOK_S in self.predictors:
                    pred = self.predictors[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S]
                    out = pred.predict(cand_record)
                    obj_values[obj] = out.predicted_value
                    gen_status = out.generalization_status
                    evidence_status[obj] = (
                        EvidenceStatus.INSUFFICIENT_EVIDENCE
                        if out.generalization_status.is_out_of_distribution
                        else EvidenceStatus.PREDICTED_EMPIRICAL
                    )
                    provenance[obj] = ProvenanceType.PREDICTED
                elif roofline and roofline.theoretical_max_tokens_per_second:
                    # Concurrency-scaled theoretical throughput ceiling
                    val = roofline.theoretical_max_tokens_per_second * config.concurrency
                    obj_values[obj] = round(val, 2)
                    evidence_status[obj] = EvidenceStatus.THEORETICAL_ANALYTICAL
                    provenance[obj] = ProvenanceType.HEURISTIC

            # Objective: VRAM (MiB)
            elif obj == OptimizationObjective.MINIMIZE_VRAM:
                obj_units[obj] = "MiB"
                if matched_obs and matched_obs.peak_vram_mb is not None:
                    obj_values[obj] = matched_obs.peak_vram_mb
                    evidence_status[obj] = EvidenceStatus.MEASURED_EVALUATED
                    provenance[obj] = ProvenanceType.MEASURED
                elif PredictionTarget.PEAK_VRAM_MB in self.predictors:
                    pred = self.predictors[PredictionTarget.PEAK_VRAM_MB]
                    out = pred.predict(cand_record)
                    obj_values[obj] = out.predicted_value
                    gen_status = out.generalization_status
                    evidence_status[obj] = (
                        EvidenceStatus.INSUFFICIENT_EVIDENCE
                        if out.generalization_status.is_out_of_distribution
                        else EvidenceStatus.PREDICTED_EMPIRICAL
                    )
                    provenance[obj] = ProvenanceType.PREDICTED
                elif res_est:
                    tp_factor = max(1, config.tensor_parallel_size)
                    param_per_rank = res_est.parameter_storage_mb / tp_factor
                    kv_total = res_est.kv_cache_total_mb or 0.0
                    if (
                        config.model_spec.num_kv_heads is not None
                        and tp_factor > config.model_spec.num_kv_heads
                    ):
                        effective_kv_shards = max(1, config.model_spec.num_kv_heads)
                    else:
                        effective_kv_shards = tp_factor
                    kv_per_rank = kv_total / effective_kv_shards
                    payload_per_rank = param_per_rank + kv_per_rank
                    val = (payload_per_rank * request.constraints.vram_safety_margin) + 512.0
                    obj_values[obj] = round(val, 2)
                    evidence_status[obj] = EvidenceStatus.THEORETICAL_ANALYTICAL
                    provenance[obj] = ProvenanceType.HEURISTIC

            # Objective: Cost ($/1M tokens)
            elif obj == OptimizationObjective.MINIMIZE_COST:
                obj_units[obj] = "$/1M tokens"
                if request.cost_per_gpu_hour is not None:
                    # Resolve throughput
                    thru = obj_values.get(OptimizationObjective.MAXIMIZE_THROUGHPUT)
                    if thru is not None and thru > 0:
                        total_cost_per_hour = (
                            request.cost_per_gpu_hour * config.tensor_parallel_size
                        )
                        tokens_per_hour = thru * 3600.0
                        cost_per_1m = (total_cost_per_hour / tokens_per_hour) * 1_000_000.0
                        obj_values[obj] = round(cost_per_1m, 4)
                        evidence_status[obj] = evidence_status.get(
                            OptimizationObjective.MAXIMIZE_THROUGHPUT,
                            EvidenceStatus.THEORETICAL_ANALYTICAL,
                        )
                        provenance[obj] = ProvenanceType.HEURISTIC
                    else:
                        notes.append("Throughput was missing; cost could not be calculated.")

            # Objective: Decode Throughput (tok/s)
            elif obj == OptimizationObjective.MAXIMIZE_DECODE_THROUGHPUT:
                obj_units[obj] = "tok/s"
                dec_thru = getattr(matched_obs, "server_decode_throughput_tok_s", None)
                if matched_obs and dec_thru is not None:
                    obj_values[obj] = dec_thru
                    evidence_status[obj] = EvidenceStatus.MEASURED_EVALUATED
                    provenance[obj] = ProvenanceType.MEASURED
                elif PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S in self.predictors:
                    pred = self.predictors[PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S]
                    out = pred.predict(cand_record)
                    obj_values[obj] = out.predicted_value
                    gen_status = out.generalization_status
                    evidence_status[obj] = (
                        EvidenceStatus.INSUFFICIENT_EVIDENCE
                        if out.generalization_status.is_out_of_distribution
                        else EvidenceStatus.PREDICTED_EMPIRICAL
                    )
                    provenance[obj] = ProvenanceType.PREDICTED
                elif roofline and roofline.theoretical_max_tokens_per_second:
                    val = roofline.theoretical_max_tokens_per_second
                    obj_values[obj] = round(val, 2)
                    evidence_status[obj] = EvidenceStatus.THEORETICAL_ANALYTICAL
                    provenance[obj] = ProvenanceType.HEURISTIC

            # Objective: End-to-End Latency / Duration (ms)
            elif obj in (OptimizationObjective.MINIMIZE_DURATION, OptimizationObjective.MINIMIZE_CLIENT_DURATION):
                obj_units[obj] = "ms"
                dur = getattr(matched_obs, "client_e2e_duration_ms", None) or (matched_obs.duration_ms if matched_obs else None)
                if matched_obs and dur is not None:
                    obj_values[obj] = dur
                    evidence_status[obj] = EvidenceStatus.MEASURED_EVALUATED
                    provenance[obj] = ProvenanceType.MEASURED
                elif PredictionTarget.CLIENT_E2E_DURATION_MS in self.predictors:
                    pred = self.predictors[PredictionTarget.CLIENT_E2E_DURATION_MS]
                    out = pred.predict(cand_record)
                    obj_values[obj] = out.predicted_value
                    gen_status = out.generalization_status
                    evidence_status[obj] = (
                        EvidenceStatus.INSUFFICIENT_EVIDENCE
                        if out.generalization_status.is_out_of_distribution
                        else EvidenceStatus.PREDICTED_EMPIRICAL
                    )
                    provenance[obj] = ProvenanceType.PREDICTED
                elif PredictionTarget.DURATION_MS in self.predictors:
                    pred = self.predictors[PredictionTarget.DURATION_MS]
                    out = pred.predict(cand_record)
                    obj_values[obj] = out.predicted_value
                    gen_status = out.generalization_status
                    evidence_status[obj] = (
                        EvidenceStatus.INSUFFICIENT_EVIDENCE
                        if out.generalization_status.is_out_of_distribution
                        else EvidenceStatus.PREDICTED_EMPIRICAL
                    )
                    provenance[obj] = ProvenanceType.PREDICTED

        # 5. Evaluate hard constraints and feasibility
        feasibility, rejection_reasons = FeasibilityChecker.evaluate(
            config=config,
            workload=request.workload,
            constraints=request.constraints,
            gpu_spec=gpu_spec,
            resource_estimate=res_est,
            evaluated_metrics=obj_values,
            evidence_status=evidence_status,
        )

        # 6. Check for mixed evidence tiers across requested objectives
        requested_ev_statuses = {
            obj: evidence_status[obj] for obj in request.objectives if obj in evidence_status
        }
        distinct_tiers = set(requested_ev_statuses.values())
        has_mixed = len(distinct_tiers) > 1
        comp_warning = None
        if has_mixed:
            tiers_str = ", ".join(
                f"{obj.value}: {st.value}" for obj, st in requested_ev_statuses.items()
            )
            comp_warning = (
                f"Candidate evaluated with mixed evidence tiers across objectives ({tiers_str}). "
                "Trade-offs between these objectives should be interpreted with caution."
            )

        return CandidateEvaluation(
            config=config,
            feasibility_status=feasibility,
            rejection_reasons=rejection_reasons,
            objective_values=obj_values,
            objective_units=obj_units,
            evidence_status=evidence_status,
            provenance=provenance,
            resource_estimate=res_est,
            roofline_result=roofline,
            generalization_status=gen_status,
            has_mixed_evidence=has_mixed,
            comparability_warning=comp_warning,
            notes=notes,
        )

    def optimize(
        self,
        request: OptimizationRequest,
    ) -> OptimizationResult:
        """Execute the constrained configuration search and compute the Pareto frontier.

        Args:
            request: The complete optimization request specification.

        Returns:
            OptimizationResult containing candidate evaluations, Pareto frontier, and recommendations.
        """
        request_id = f"opt-{uuid.uuid4().hex[:10]}"
        gpu_spec = self._resolve_gpu_specification(request)

        # 1. Generate search space candidates with early pruning
        candidates, early_rejections = SearchSpaceGenerator.generate(
            search_space=request.search_space,
            constraints=request.constraints,
        )

        # 2. Evaluate all generated candidates
        evaluations: list[CandidateEvaluation] = []
        rejection_counts: dict[str, int] = dict(early_rejections)

        feasible_count = 0
        infeasible_count = 0
        unknown_count = 0

        for cand in candidates:
            evaluation = self._evaluate_candidate(cand, request, gpu_spec)
            evaluations.append(evaluation)

            if evaluation.feasibility_status == FeasibilityStatus.FEASIBLE:
                feasible_count += 1
            elif evaluation.feasibility_status == FeasibilityStatus.UNKNOWN_FEASIBILITY:
                unknown_count += 1
                for r in evaluation.rejection_reasons:
                    rejection_counts[r] = rejection_counts.get(r, 0) + 1
            else:
                infeasible_count += 1
                for r in evaluation.rejection_reasons:
                    rejection_counts[r] = rejection_counts.get(r, 0) + 1

        # 3. Extract non-dominated Pareto frontier
        frontier = ParetoEngine.extract_pareto_frontier(
            candidates=evaluations,
            objectives=request.objectives,
        )

        # 4. Apply single-candidate selection policy if requested
        selected_candidate, selection_rationale = ParetoEngine.select_candidate(
            frontier=frontier,
            objectives=request.objectives,
            policy=request.selection_policy,
            primary_objective=request.primary_objective,
        )

        # Check for mixed or incomparable evidence across the Pareto frontier
        has_incomparable = False
        evidence_caveats: list[str] = []

        if frontier:
            for c in frontier:
                if c.has_mixed_evidence and c.comparability_warning:
                    has_incomparable = True
                    evidence_caveats.append(
                        f"Config '{c.config.config_id}': {c.comparability_warning}"
                    )

            for obj in request.objectives:
                tiers_for_obj = {
                    c.evidence_status.get(obj) for c in frontier if obj in c.evidence_status
                }
                if len(tiers_for_obj) > 1:
                    has_incomparable = True
                    tiers_list = sorted(t.value for t in tiers_for_obj if t is not None)
                    evidence_caveats.append(
                        f"Objective '{obj.value}' has mixed evidence tiers across frontier candidates: {tiers_list}. "
                        "Trade-offs between these candidates may reflect modeling discrepancies rather than true performance differences."
                    )

        # 5. Formulate human-readable executive summary
        summary = self._build_executive_summary(
            total_gen=len(candidates) + sum(early_rejections.values()),
            feasible_cnt=feasible_count,
            infeasible_cnt=infeasible_count,
            unknown_cnt=unknown_count,
            frontier_cnt=len(frontier),
            frontier=frontier,
            selected=selected_candidate,
            gpu_spec=gpu_spec,
            has_incomparable=has_incomparable,
        )

        return OptimizationResult(
            request_id=request_id,
            summary=summary,
            total_candidates_generated=len(candidates) + sum(early_rejections.values()),
            total_candidates_evaluated=len(evaluations),
            feasible_candidates_count=feasible_count,
            infeasible_candidates_count=infeasible_count,
            unknown_feasibility_count=unknown_count,
            pareto_frontier=frontier,
            recommended_candidate=selected_candidate,
            selection_rationale=selection_rationale,
            rejections_by_reason=rejection_counts,
            all_evaluations=evaluations,
            has_incomparable_evidence=has_incomparable,
            evidence_caveats=evidence_caveats,
        )

    def _build_executive_summary(
        self,
        total_gen: int,
        feasible_cnt: int,
        infeasible_cnt: int,
        unknown_cnt: int,
        frontier_cnt: int,
        frontier: list[CandidateEvaluation],
        selected: CandidateEvaluation | None,
        gpu_spec: GPUSpecification | None,
        has_incomparable: bool = False,
    ) -> str:
        """Formulate a scientifically honest summary of optimization outcomes."""
        lines: list[str] = []
        gpu_label = (
            f"{gpu_spec.name} ({gpu_spec.memory_capacity_gb} GB VRAM)"
            if gpu_spec
            else "Unspecified / Host Hardware"
        )
        lines.append(f"Evaluated {total_gen} configurations for target: {gpu_label}.")
        lines.append(
            f"Feasibility breakdown: {feasible_cnt} feasible, {infeasible_cnt} infeasible, "
            f"{unknown_cnt} unknown feasibility."
        )

        if frontier_cnt == 0:
            lines.append(
                "No configuration satisfied all hard feasibility constraints. Zero Pareto candidates returned."
            )
        else:
            lines.append(
                f"Identified {frontier_cnt} non-dominated Pareto-optimal configuration(s) across requested objectives."
            )
            if selected is not None:
                lines.append(f"Selected candidate: {selected.config.config_id}.")

            # Incomparable / mixed evidence notice
            if has_incomparable:
                lines.append(
                    "NOTICE: Pareto frontier contains candidates evaluated using disparate or mixed evidence tiers "
                    "(e.g., measured vs predicted vs analytical). Direct trade-off comparisons across these tiers "
                    "should be verified with hardware benchmarking."
                )

            # Evidence status transparency notice
            has_measured = any(
                any(ev == EvidenceStatus.MEASURED_EVALUATED for ev in c.evidence_status.values())
                for c in frontier
            )
            has_analytical_only = all(
                all(
                    ev == EvidenceStatus.THEORETICAL_ANALYTICAL for ev in c.evidence_status.values()
                )
                for c in frontier
            )

            if has_analytical_only:
                lines.append(
                    "NOTICE: All candidate performance figures derive strictly from Session 05 analytical resource models "
                    "and Roofline hardware ceilings. These represent theoretical upper bounds, not measured hardware observations."
                )
            elif not has_measured:
                lines.append(
                    "NOTICE: Recommendations derive from empirical model predictions. Actual hardware runtimes may vary "
                    "with framework scheduling overheads."
                )

        return " ".join(lines)
