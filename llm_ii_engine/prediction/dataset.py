"""Dataset construction and target eligibility evaluation for prediction tasks.

Transforms benchmark suite results and hardware reports into typed PredictionRecord
instances, strictly enforcing data provenance, warmup exclusion, and chunk-vs-token separation.
"""

import uuid

from llm_ii_engine.analysis.resource_estimator import ModelResourceEstimator
from llm_ii_engine.benchmark.dataset_export import EmpiricalObservationRecord
from llm_ii_engine.benchmark.models import BenchmarkSuiteResult, RequestBenchmarkResult
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    EngineType,
    ExecutionMode,
    ModelResidencyState,
    PrefixRelationshipType,
    ProvenanceType,
    TokenCountSource,
)
from llm_ii_engine.core.schemas import ModelSpec
from llm_ii_engine.discovery.models import HardwareCapabilityReport
from llm_ii_engine.prediction.models import (
    PredictionDataset,
    PredictionRecord,
    PredictionTarget,
)


class DatasetBuilder:
    """Builds typed PredictionDataset instances from benchmark results and metadata."""

    @classmethod
    def evaluate_record_eligibility(
        cls,
        req: RequestBenchmarkResult,
        suite: BenchmarkSuiteResult,
    ) -> tuple[dict[PredictionTarget, bool], dict[PredictionTarget, list[str]]]:
        """Determine eligibility for each prediction target and record audit reasons.

        Returns:
            Tuple of (eligibility_map, ineligibility_reasons_map).
        """
        eligibility: dict[PredictionTarget, bool] = {target: False for target in PredictionTarget}
        reasons: dict[PredictionTarget, list[str]] = {target: [] for target in PredictionTarget}

        # 1. Warmup exclusion: warmup observations must never be used as training targets
        if req.is_warmup or suite.status == "WARMUP":
            for target in PredictionTarget:
                reasons[target].append(
                    "Observation occurred during warmup phase; excluded from targets"
                )
            return eligibility, reasons

        # 2. Failure & incomplete execution exclusion
        if not req.success or req.incomplete_reason is not None:
            for target in PredictionTarget:
                reasons[target].append(
                    f"Request unsuccessful or incomplete: {req.incomplete_reason or req.error_message or 'unhandled error'}"
                )
            return eligibility, reasons

        # 3. TTFT Eligibility
        # Requires verified prompt tokens, at least 1 decoded output token, and valid non-negative ttft measurement
        if req.prompt_tokens is None or req.prompt_token_count_source in (
            TokenCountSource.HEURISTIC_WORD_COUNT,
            TokenCountSource.UNKNOWN,
        ):
            reasons[PredictionTarget.TTFT_MS].append(
                f"Prompt token count provenance is {req.prompt_token_count_source.value}; "
                "verified prompt tokens required for TTFT training"
            )
        elif req.output_tokens == 0:
            reasons[PredictionTarget.TTFT_MS].append("Zero output tokens produced; TTFT unmeasured")
        elif req.ttft_ms is None or req.ttft_ms < 0:
            reasons[PredictionTarget.TTFT_MS].append("TTFT timing metric is missing or negative")
        else:
            eligibility[PredictionTarget.TTFT_MS] = True

        # 4. Token-level TPOT Eligibility
        # Must have verified discrete token events, >= 2 output tokens, and valid positive tpot measurement
        if req.token_count_source != TokenCountSource.TOKEN_EVENTS:
            reasons[PredictionTarget.TPOT_MS].append(
                f"Token-level TPOT requires verified discrete token events (got {req.token_count_source.value}). "
                "Chunks must not be treated as tokens for TPOT."
            )
        elif req.output_tokens < 2:
            reasons[PredictionTarget.TPOT_MS].append(
                f"Requires >= 2 output tokens for inter-token latency deltas (got {req.output_tokens})"
            )
        elif req.tpot_ms is None or req.tpot_ms <= 0:
            reasons[PredictionTarget.TPOT_MS].append("Token-level TPOT timing is missing or zero")
        else:
            eligibility[PredictionTarget.TPOT_MS] = True

        # 5. Chunk Inter-Arrival Latency Eligibility
        if req.chunk_count < 2:
            reasons[PredictionTarget.CHUNK_LATENCY_MS].append(
                f"Requires >= 2 chunks for chunk latency deltas (got {req.chunk_count})"
            )
        elif req.chunk_latency_p50_ms is None or req.chunk_latency_p50_ms <= 0:
            reasons[PredictionTarget.CHUNK_LATENCY_MS].append(
                "Chunk latency p50 is missing or zero"
            )
        else:
            eligibility[PredictionTarget.CHUNK_LATENCY_MS] = True

        # 6. Throughput Eligibility
        if req.token_count_source == TokenCountSource.UNKNOWN:
            reasons[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S].append(
                "Output token count source is unknown; cannot verify throughput"
            )
        elif req.output_tokens < 1:
            reasons[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S].append("Zero output tokens produced")
        elif req.output_tokens_per_second is None or req.output_tokens_per_second <= 0:
            reasons[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S].append(
                "Output throughput is missing or zero"
            )
        else:
            eligibility[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S] = True

        # 7. Memory Peak VRAM Eligibility
        mem = suite.memory_watermark
        if mem is None or not mem.is_telemetry_available:
            reasons[PredictionTarget.PEAK_VRAM_MB].append(
                "Memory telemetry was unavailable during benchmark run"
            )
        elif mem.peak_gpu_vram_mb is None or mem.peak_gpu_vram_mb <= 0:
            reasons[PredictionTarget.PEAK_VRAM_MB].append("Peak GPU VRAM is missing or zero")
        else:
            eligibility[PredictionTarget.PEAK_VRAM_MB] = True

        # 8. Request Duration Eligibility (End-to-End Latency)
        if not req.success:
            reasons[PredictionTarget.DURATION_MS].append("Request did not succeed; duration invalid")
        elif req.duration_ms is None or req.duration_ms <= 0:
            reasons[PredictionTarget.DURATION_MS].append("Request duration is missing or non-positive")
        else:
            eligibility[PredictionTarget.DURATION_MS] = True

        # 9. TTFC Eligibility
        if req.ttfc_ms is None or req.ttfc_ms <= 0:
            reasons[PredictionTarget.TTFC_MS].append("TTFC is missing or non-positive")
        elif req.chunk_count < 1:
            reasons[PredictionTarget.TTFC_MS].append("No chunks received; TTFC unmeasured")
        else:
            eligibility[PredictionTarget.TTFC_MS] = True

        # 10. Server Prompt Eval Duration Eligibility
        s_stats = getattr(req, "server_stats", None)
        if s_stats is None or s_stats.prompt_duration_ms is None or s_stats.prompt_duration_ms <= 0:
            reasons[PredictionTarget.SERVER_PROMPT_EVAL_MS].append(
                "Server prompt eval duration is missing or unobserved"
            )
        else:
            eligibility[PredictionTarget.SERVER_PROMPT_EVAL_MS] = True

        # 11. Server Eval Duration Eligibility
        if s_stats is None or s_stats.eval_duration_ms is None or s_stats.eval_duration_ms <= 0:
            reasons[PredictionTarget.SERVER_EVAL_MS].append(
                "Server eval duration is missing or unobserved"
            )
        else:
            eligibility[PredictionTarget.SERVER_EVAL_MS] = True

        # 12. Distinct Semantic Targets (Session 16)
        eligibility[PredictionTarget.CLIENT_E2E_DURATION_MS] = eligibility[PredictionTarget.DURATION_MS]
        reasons[PredictionTarget.CLIENT_E2E_DURATION_MS] = list(reasons[PredictionTarget.DURATION_MS])

        eligibility[PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S] = eligibility[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S]
        reasons[PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S] = list(reasons[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S])

        eligibility[PredictionTarget.SERVER_PREFILL_DURATION_MS] = eligibility[PredictionTarget.SERVER_PROMPT_EVAL_MS]
        reasons[PredictionTarget.SERVER_PREFILL_DURATION_MS] = list(reasons[PredictionTarget.SERVER_PROMPT_EVAL_MS])

        eligibility[PredictionTarget.SERVER_DECODE_DURATION_MS] = eligibility[PredictionTarget.SERVER_EVAL_MS]
        reasons[PredictionTarget.SERVER_DECODE_DURATION_MS] = list(reasons[PredictionTarget.SERVER_EVAL_MS])

        # Server Total Duration Eligibility
        if s_stats is None or s_stats.total_duration_ms is None or s_stats.total_duration_ms <= 0:
            reasons[PredictionTarget.SERVER_TOTAL_DURATION_MS].append(
                "Server total duration is missing or unobserved"
            )
        else:
            eligibility[PredictionTarget.SERVER_TOTAL_DURATION_MS] = True

        # Server Load Duration Eligibility
        if s_stats is None or s_stats.load_duration_ms is None or s_stats.load_duration_ms < 0:
            reasons[PredictionTarget.SERVER_LOAD_DURATION_MS].append(
                "Server load duration is missing or unobserved"
            )
        else:
            eligibility[PredictionTarget.SERVER_LOAD_DURATION_MS] = True

        # Server Decode Throughput Eligibility
        if not eligibility[PredictionTarget.SERVER_DECODE_DURATION_MS]:
            reasons[PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S].append(
                "Server decode duration is missing or unobserved"
            )
        elif req.output_tokens < 1:
            reasons[PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S].append(
                "Zero output tokens produced"
            )
        else:
            eligibility[PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S] = True

        return eligibility, reasons

    @classmethod
    def create_record(
        cls,
        req: RequestBenchmarkResult,
        suite: BenchmarkSuiteResult,
        model_spec: ModelSpec,
        hardware_report: HardwareCapabilityReport | None = None,
        experiment_id: str | None = None,
    ) -> PredictionRecord:
        """Construct a single PredictionRecord from benchmark request evidence."""
        eligibility, reasons = cls.evaluate_record_eligibility(req, suite)

        # Architectural features
        b_param: float | None = None
        if model_spec.quantization is not None:
            b_param, _, _ = ModelResourceEstimator.get_quantization_bytes(model_spec.quantization)

        head_dim = model_spec.head_dimension
        if (
            head_dim is None
            and model_spec.hidden_dimension
            and model_spec.num_attention_heads
            and model_spec.hidden_dimension % model_spec.num_attention_heads == 0
        ):
            head_dim = model_spec.hidden_dimension // model_spec.num_attention_heads

        # Hardware features (preserve None if unavailable)
        hw_catalog_id = None
        gpu_cap_gb = None
        gpu_bw_gbps = None
        gpu_compute = None
        host_ram_gb = None

        if hardware_report is not None:
            host_ram_gb = hardware_report.memory.total_gb
            if hardware_report.matched_catalog_specs:
                gpu_spec = hardware_report.matched_catalog_specs[0]
                hw_catalog_id = gpu_spec.catalog_id
                gpu_cap_gb = gpu_spec.memory_capacity_gb
                gpu_bw_gbps = gpu_spec.theoretical_memory_bandwidth_gbps
                gpu_compute = gpu_spec.compute.fp16_dense_tflops

        peak_vram = None
        if suite.memory_watermark and suite.memory_watermark.is_telemetry_available:
            peak_vram = suite.memory_watermark.peak_gpu_vram_mb

        s_stats = getattr(req, "server_stats", None)
        server_prompt_ms = s_stats.prompt_duration_ms if s_stats else None
        server_eval_ms = s_stats.eval_duration_ms if s_stats else None
        server_total_ms = s_stats.total_duration_ms if s_stats else None
        server_load_ms = s_stats.load_duration_ms if s_stats else None
        server_decode_thru = s_stats.eval_tokens_per_second if s_stats else None

        return PredictionRecord(
            record_id=f"rec-{uuid.uuid4().hex[:10]}",
            experiment_id=experiment_id or suite.suite_id,
            suite_id=suite.suite_id,
            request_id=req.request_id,
            model_id=model_spec.model_id,
            model_digest=getattr(suite, "model_digest", None) or getattr(model_spec, "model_digest", None),
            engine_type=suite.engine_type,
            execution_mode=req.execution_mode,
            provenance=req.provenance,
            is_warmup=req.is_warmup,
            parameter_count_billion=model_spec.parameter_count_billion,
            num_layers=model_spec.num_layers,
            num_attention_heads=model_spec.num_attention_heads,
            num_kv_heads=model_spec.num_kv_heads,
            hidden_dimension=model_spec.hidden_dimension,
            head_dimension=head_dim,
            quantization=model_spec.quantization,
            bytes_per_parameter=b_param,
            hardware_catalog_id=hw_catalog_id,
            gpu_memory_capacity_gb=gpu_cap_gb,
            gpu_memory_bandwidth_gbps=gpu_bw_gbps,
            gpu_peak_compute_tflops=gpu_compute,
            host_ram_total_gb=host_ram_gb,
            concurrency=suite.concurrency,
            prompt_tokens=req.prompt_tokens,
            prompt_token_count_source=req.prompt_token_count_source,
            requested_output_tokens=req.requested_output_tokens,
            actual_output_tokens=req.output_tokens,
            output_tokens=req.output_tokens,
            token_count_source=req.token_count_source,
            planned_sequence_length=(
                (req.prompt_tokens + req.requested_output_tokens)
                if (req.prompt_tokens is not None and req.requested_output_tokens is not None)
                else None
            ),
            total_sequence_length=(
                (req.prompt_tokens + req.output_tokens)
                if req.prompt_tokens is not None
                else None
            ),
            ttft_ms=req.ttft_ms,
            ttfc_ms=req.ttfc_ms,
            tpot_ms=req.tpot_ms,
            chunk_latency_ms=req.chunk_latency_p50_ms,
            output_tokens_per_second=req.output_tokens_per_second,
            client_e2e_duration_ms=req.duration_ms,
            client_e2e_output_throughput_tok_s=req.output_tokens_per_second,
            server_total_duration_ms=server_total_ms,
            server_load_duration_ms=server_load_ms,
            server_prefill_duration_ms=server_prompt_ms,
            server_decode_duration_ms=server_eval_ms,
            server_decode_throughput_tok_s=server_decode_thru,
            peak_vram_mb=peak_vram,
            duration_ms=req.duration_ms,
            server_prompt_eval_duration_ms=server_prompt_ms,
            server_eval_duration_ms=server_eval_ms,
            model_residency_state=getattr(req, "model_residency_state", None),
            cache_state=getattr(req, "cache_state", None),
            predecessor_request_id=getattr(req, "predecessor_request_id", None),
            server_prompt_eval_cached_count=(
                s_stats.prompt_eval_cached_count
                if s_stats
                else None
            ),
            prefix_relationship=getattr(req, "prefix_relationship", None),
            is_eligible_for_ttft=eligibility[PredictionTarget.TTFT_MS],
            is_eligible_for_tpot=eligibility[PredictionTarget.TPOT_MS],
            is_eligible_for_chunk_latency=eligibility[PredictionTarget.CHUNK_LATENCY_MS],
            is_eligible_for_throughput=eligibility[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S],
            is_eligible_for_vram=eligibility[PredictionTarget.PEAK_VRAM_MB],
            is_eligible_for_duration=eligibility[PredictionTarget.DURATION_MS],
            is_eligible_for_ttfc=eligibility[PredictionTarget.TTFC_MS],
            is_eligible_for_prefill=eligibility[PredictionTarget.SERVER_PROMPT_EVAL_MS],
            is_eligible_for_decode=eligibility[PredictionTarget.SERVER_EVAL_MS],
            ineligibility_reasons={k: v for k, v in reasons.items() if v},
        )

    @classmethod
    def from_benchmark_suite(
        cls,
        suite: BenchmarkSuiteResult,
        model_spec: ModelSpec,
        hardware_report: HardwareCapabilityReport | None = None,
        dataset_id: str | None = None,
    ) -> PredictionDataset:
        """Build a PredictionDataset from a completed BenchmarkSuiteResult."""
        records: list[PredictionRecord] = []
        for req in suite.request_results:
            records.append(
                cls.create_record(
                    req=req,
                    suite=suite,
                    model_spec=model_spec,
                    hardware_report=hardware_report,
                    experiment_id=suite.suite_id,
                )
            )
        for req in suite.warmup_results:
            records.append(
                cls.create_record(
                    req=req,
                    suite=suite,
                    model_spec=model_spec,
                    hardware_report=hardware_report,
                    experiment_id=suite.suite_id,
                )
            )

        return PredictionDataset(
            dataset_id=dataset_id or f"ds-{uuid.uuid4().hex[:8]}",
            records=records,
            description=f"Extracted from suite '{suite.suite_id}' ({suite.model_id}, {suite.engine_type.value})",
        )

    @classmethod
    def from_empirical_observations(
        cls,
        observations: list[EmpiricalObservationRecord],
        model_spec: ModelSpec,
        hardware_report: HardwareCapabilityReport | None = None,
        dataset_id: str | None = None,
    ) -> PredictionDataset:
        """Construct a typed PredictionDataset from exported EmpiricalObservationRecord instances."""
        records: list[PredictionRecord] = []

        # Architectural features
        b_param: float | None = None
        if model_spec.quantization is not None:
            b_param, _, _ = ModelResourceEstimator.get_quantization_bytes(model_spec.quantization)

        head_dim = model_spec.head_dimension
        if (
            head_dim is None
            and model_spec.hidden_dimension
            and model_spec.num_attention_heads
            and model_spec.hidden_dimension % model_spec.num_attention_heads == 0
        ):
            head_dim = model_spec.hidden_dimension // model_spec.num_attention_heads

        for obs in observations:
            # Eligibility rules matching evaluate_record_eligibility
            eligibility = {target: False for target in PredictionTarget}
            reasons: dict[PredictionTarget, list[str]] = {target: [] for target in PredictionTarget}

            if obs.is_warmup:
                for target in PredictionTarget:
                    reasons[target].append("Warmup observation excluded from prediction training")
            elif not obs.success:
                for target in PredictionTarget:
                    reasons[target].append(f"Unsuccessful request: {obs.error_message or 'error'}")
            else:
                # TTFT eligibility
                prompt_src = TokenCountSource(obs.prompt_token_count_source)
                if obs.prompt_tokens is None or prompt_src in (
                    TokenCountSource.HEURISTIC_WORD_COUNT,
                    TokenCountSource.UNKNOWN,
                ):
                    reasons[PredictionTarget.TTFT_MS].append(
                        f"Prompt count source is {prompt_src.value}; verified prompt tokens required"
                    )
                elif obs.actual_output_tokens == 0:
                    reasons[PredictionTarget.TTFT_MS].append("Zero output tokens produced")
                elif obs.ttft_ms is None or obs.ttft_ms < 0:
                    reasons[PredictionTarget.TTFT_MS].append("TTFT timing metric is missing or negative")
                else:
                    eligibility[PredictionTarget.TTFT_MS] = True

                # TPOT eligibility
                tok_src = TokenCountSource(obs.token_count_source)
                if tok_src != TokenCountSource.TOKEN_EVENTS:
                    reasons[PredictionTarget.TPOT_MS].append(
                        f"TPOT requires verified discrete TOKEN events (got {tok_src.value})"
                    )
                elif obs.actual_output_tokens < 2:
                    reasons[PredictionTarget.TPOT_MS].append("Requires >= 2 output tokens")
                elif obs.tpot_ms is None or obs.tpot_ms <= 0:
                    reasons[PredictionTarget.TPOT_MS].append("TPOT timing missing or zero")
                else:
                    eligibility[PredictionTarget.TPOT_MS] = True

                # Chunk Latency eligibility
                if obs.chunk_count < 2:
                    reasons[PredictionTarget.CHUNK_LATENCY_MS].append("Requires >= 2 chunks")
                elif obs.chunk_latency_p50_ms is None or obs.chunk_latency_p50_ms <= 0:
                    reasons[PredictionTarget.CHUNK_LATENCY_MS].append("Chunk latency p50 missing or zero")
                else:
                    eligibility[PredictionTarget.CHUNK_LATENCY_MS] = True

                # Throughput eligibility
                if tok_src == TokenCountSource.UNKNOWN:
                    reasons[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S].append("Output token source unknown")
                elif obs.actual_output_tokens < 1:
                    reasons[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S].append("Zero output tokens")
                elif obs.output_tokens_per_second is None or obs.output_tokens_per_second <= 0:
                    reasons[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S].append("Throughput missing or zero")
                else:
                    eligibility[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S] = True

                # Peak VRAM eligibility
                if obs.peak_vram_mb is None or obs.peak_vram_mb <= 0:
                    reasons[PredictionTarget.PEAK_VRAM_MB].append("Peak VRAM missing or zero")
                else:
                    eligibility[PredictionTarget.PEAK_VRAM_MB] = True

                # Duration eligibility (end-to-end latency)
                if not obs.success:
                    reasons[PredictionTarget.DURATION_MS].append("Request unsuccessful; duration invalid")
                elif obs.duration_ms is None or obs.duration_ms <= 0:
                    reasons[PredictionTarget.DURATION_MS].append("Duration missing or non-positive")
                else:
                    eligibility[PredictionTarget.DURATION_MS] = True

                # TTFC eligibility
                if obs.ttfc_ms is None or obs.ttfc_ms <= 0:
                    reasons[PredictionTarget.TTFC_MS].append("TTFC missing or non-positive")
                elif obs.chunk_count < 1:
                    reasons[PredictionTarget.TTFC_MS].append("Zero chunks received")
                else:
                    eligibility[PredictionTarget.TTFC_MS] = True

                # Server prompt eval duration eligibility
                if obs.server_prompt_eval_duration_ms is None or obs.server_prompt_eval_duration_ms <= 0:
                    reasons[PredictionTarget.SERVER_PROMPT_EVAL_MS].append("Server prompt eval duration missing")
                else:
                    eligibility[PredictionTarget.SERVER_PROMPT_EVAL_MS] = True

                # Server eval duration eligibility
                if obs.server_eval_duration_ms is None or obs.server_eval_duration_ms <= 0:
                    reasons[PredictionTarget.SERVER_EVAL_MS].append("Server eval duration missing")
                else:
                    eligibility[PredictionTarget.SERVER_EVAL_MS] = True

                # Distinct Semantic Targets (Session 16)
                eligibility[PredictionTarget.CLIENT_E2E_DURATION_MS] = eligibility[PredictionTarget.DURATION_MS]
                reasons[PredictionTarget.CLIENT_E2E_DURATION_MS] = list(reasons[PredictionTarget.DURATION_MS])

                eligibility[PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S] = eligibility[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S]
                reasons[PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S] = list(reasons[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S])

                eligibility[PredictionTarget.SERVER_PREFILL_DURATION_MS] = eligibility[PredictionTarget.SERVER_PROMPT_EVAL_MS]
                reasons[PredictionTarget.SERVER_PREFILL_DURATION_MS] = list(reasons[PredictionTarget.SERVER_PROMPT_EVAL_MS])

                eligibility[PredictionTarget.SERVER_DECODE_DURATION_MS] = eligibility[PredictionTarget.SERVER_EVAL_MS]
                reasons[PredictionTarget.SERVER_DECODE_DURATION_MS] = list(reasons[PredictionTarget.SERVER_EVAL_MS])

                # Server total duration eligibility
                s_tot = getattr(obs, "server_total_duration_ms", None)
                if s_tot is None or s_tot <= 0:
                    reasons[PredictionTarget.SERVER_TOTAL_DURATION_MS].append("Server total duration missing")
                else:
                    eligibility[PredictionTarget.SERVER_TOTAL_DURATION_MS] = True

                # Server load duration eligibility
                s_load = getattr(obs, "server_load_duration_ms", None)
                if s_load is None or s_load < 0:
                    reasons[PredictionTarget.SERVER_LOAD_DURATION_MS].append("Server load duration missing")
                else:
                    eligibility[PredictionTarget.SERVER_LOAD_DURATION_MS] = True

                # Server decode throughput eligibility
                if not eligibility[PredictionTarget.SERVER_DECODE_DURATION_MS]:
                    reasons[PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S].append("Server decode duration missing")
                elif obs.actual_output_tokens < 1:
                    reasons[PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S].append("Zero output tokens")
                else:
                    eligibility[PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S] = True

            try:
                eng_type = EngineType(obs.runtime_name)
            except ValueError:
                eng_type = EngineType.CUSTOM

            exec_mode = (
                ExecutionMode.REAL
                if obs.execution_mode == ExecutionMode.REAL.value
                else ExecutionMode.SIMULATED
            )
            prov = (
                ProvenanceType.MEASURED
                if obs.provenance == ProvenanceType.MEASURED.value
                else ProvenanceType.HEURISTIC
            )

            p_tok = obs.prompt_tokens
            req_tok = obs.requested_output_tokens
            act_tok = obs.actual_output_tokens

            residency = None
            if getattr(obs, "model_residency_state", None):
                try:
                    residency = ModelResidencyState(obs.model_residency_state)
                except ValueError:
                    residency = ModelResidencyState.UNKNOWN

            c_state = None
            if getattr(obs, "cache_state", None):
                try:
                    c_state = CacheStateClassification(obs.cache_state)
                except ValueError:
                    c_state = CacheStateClassification.UNKNOWN

            prefix_rel = None
            if getattr(obs, "prefix_relationship", None):
                try:
                    prefix_rel = PrefixRelationshipType(obs.prefix_relationship)
                except ValueError:
                    prefix_rel = PrefixRelationshipType.NONE

            records.append(
                PredictionRecord(
                    record_id=f"pred-{obs.record_id}",
                    experiment_id=obs.experiment_id,
                    suite_id=obs.suite_id,
                    request_id=obs.request_id,
                    model_id=model_spec.model_id,
                    model_digest=obs.model_digest,
                    engine_type=eng_type,
                    execution_mode=exec_mode,
                    provenance=prov,
                    is_warmup=obs.is_warmup,
                    parameter_count_billion=model_spec.parameter_count_billion,
                    num_layers=model_spec.num_layers,
                    num_attention_heads=model_spec.num_attention_heads,
                    num_kv_heads=model_spec.num_kv_heads,
                    hidden_dimension=model_spec.hidden_dimension,
                    head_dimension=head_dim,
                    quantization=model_spec.quantization,
                    bytes_per_parameter=b_param,
                    hardware_catalog_id=obs.hardware_id,
                    gpu_memory_capacity_gb=obs.gpu_vram_gb,
                    host_ram_total_gb=obs.host_ram_gb,
                    concurrency=1,
                    prompt_tokens=p_tok,
                    prompt_token_count_source=TokenCountSource(obs.prompt_token_count_source),
                    requested_output_tokens=req_tok,
                    actual_output_tokens=act_tok,
                    output_tokens=act_tok,
                    token_count_source=TokenCountSource(obs.token_count_source),
                    planned_sequence_length=(
                        (p_tok + req_tok) if (p_tok is not None and req_tok is not None) else None
                    ),
                    total_sequence_length=(p_tok + act_tok) if p_tok is not None else None,
                    ttft_ms=obs.ttft_ms,
                    ttfc_ms=obs.ttfc_ms,
                    tpot_ms=obs.tpot_ms,
                    chunk_latency_ms=obs.chunk_latency_p50_ms,
                    output_tokens_per_second=obs.output_tokens_per_second,
                    client_e2e_duration_ms=getattr(obs, "client_e2e_duration_ms", obs.duration_ms),
                    client_e2e_output_throughput_tok_s=getattr(obs, "client_e2e_output_throughput_tok_s", obs.output_tokens_per_second),
                    server_total_duration_ms=getattr(obs, "server_total_duration_ms", None),
                    server_load_duration_ms=getattr(obs, "server_load_duration_ms", None),
                    server_prefill_duration_ms=getattr(obs, "server_prefill_duration_ms", obs.server_prompt_eval_duration_ms),
                    server_decode_duration_ms=getattr(obs, "server_decode_duration_ms", obs.server_eval_duration_ms),
                    server_decode_throughput_tok_s=getattr(obs, "server_decode_throughput_tok_s", None),
                    peak_vram_mb=obs.peak_vram_mb,
                    duration_ms=obs.duration_ms,
                    server_prompt_eval_duration_ms=obs.server_prompt_eval_duration_ms,
                    server_eval_duration_ms=obs.server_eval_duration_ms,
                    model_residency_state=residency,
                    cache_state=c_state,
                    predecessor_request_id=obs.predecessor_request_id,
                    server_prompt_eval_cached_count=obs.server_prompt_eval_cached_count,
                    prefix_relationship=prefix_rel,
                    is_eligible_for_ttft=eligibility[PredictionTarget.TTFT_MS],
                    is_eligible_for_tpot=eligibility[PredictionTarget.TPOT_MS],
                    is_eligible_for_chunk_latency=eligibility[PredictionTarget.CHUNK_LATENCY_MS],
                    is_eligible_for_throughput=eligibility[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S],
                    is_eligible_for_vram=eligibility[PredictionTarget.PEAK_VRAM_MB],
                    is_eligible_for_duration=eligibility[PredictionTarget.DURATION_MS],
                    is_eligible_for_ttfc=eligibility[PredictionTarget.TTFC_MS],
                    is_eligible_for_prefill=eligibility[PredictionTarget.SERVER_PROMPT_EVAL_MS],
                    is_eligible_for_decode=eligibility[PredictionTarget.SERVER_EVAL_MS],
                    ineligibility_reasons={k: v for k, v in reasons.items() if v},
                )
            )

        return PredictionDataset(
            dataset_id=dataset_id or f"ds-{uuid.uuid4().hex[:8]}",
            records=records,
            description=f"Constructed from {len(observations)} empirical observations ({model_spec.model_id})",
        )
