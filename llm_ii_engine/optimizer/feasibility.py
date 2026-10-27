"""Hard constraint evaluation and feasibility assessment for candidate configurations.

Distinguishes provable infeasibility, estimated analytical infeasibility, unknown
feasibility due to missing specifications, and estimated feasibility.
"""

from llm_ii_engine.analysis.models import ModelResourceEstimate
from llm_ii_engine.core.enums import EngineType, QuantizationType
from llm_ii_engine.core.schemas import WorkloadSpec
from llm_ii_engine.discovery.models import GPUSpecification
from llm_ii_engine.optimizer.models import (
    CandidateConfig,
    EvidenceStatus,
    FeasibilityStatus,
    OptimizationConstraints,
    OptimizationObjective,
)


class FeasibilityChecker:
    """Evaluates hard constraints and resource feasibility for candidate configurations."""

    # Engine-quantization compatibility matrix
    SUPPORTED_QUANTIZATIONS_BY_ENGINE: dict[EngineType, set[QuantizationType]] = {
        EngineType.MOCK: {
            QuantizationType.NONE,
            QuantizationType.FP16,
            QuantizationType.BF16,
            QuantizationType.INT8,
            QuantizationType.INT4,
            QuantizationType.AWQ,
            QuantizationType.GPTQ,
            QuantizationType.GGUF_Q4_K_M,
            QuantizationType.GGUF_Q8_0,
        },
        EngineType.LLAMACPP: {
            QuantizationType.NONE,
            QuantizationType.FP16,
            QuantizationType.GGUF_Q4_K_M,
            QuantizationType.GGUF_Q8_0,
        },
        EngineType.OLLAMA: {
            QuantizationType.NONE,
            QuantizationType.FP16,
            QuantizationType.GGUF_Q4_K_M,
            QuantizationType.GGUF_Q8_0,
        },
        EngineType.VLLM: {
            QuantizationType.NONE,
            QuantizationType.FP16,
            QuantizationType.BF16,
            QuantizationType.INT8,
            QuantizationType.INT4,
            QuantizationType.AWQ,
            QuantizationType.GPTQ,
        },
        EngineType.TRANSFORMERS: {
            QuantizationType.NONE,
            QuantizationType.FP16,
            QuantizationType.BF16,
            QuantizationType.INT8,
            QuantizationType.INT4,
            QuantizationType.AWQ,
            QuantizationType.GPTQ,
        },
        EngineType.CUSTOM: {
            QuantizationType.NONE,
            QuantizationType.FP16,
            QuantizationType.BF16,
            QuantizationType.INT8,
            QuantizationType.INT4,
            QuantizationType.AWQ,
            QuantizationType.GPTQ,
            QuantizationType.GGUF_Q4_K_M,
            QuantizationType.GGUF_Q8_0,
        },
    }

    @classmethod
    def evaluate(
        cls,
        config: CandidateConfig,
        workload: WorkloadSpec,
        constraints: OptimizationConstraints,
        gpu_spec: GPUSpecification | None = None,
        resource_estimate: ModelResourceEstimate | None = None,
        evaluated_metrics: dict[OptimizationObjective, float] | None = None,
        evidence_status: dict[OptimizationObjective, EvidenceStatus] | None = None,
    ) -> tuple[FeasibilityStatus, list[str]]:
        """Evaluate hard constraints for a candidate configuration.

        Args:
            config: Candidate deployment configuration.
            workload: Target workload with prompt and output limits.
            constraints: Optimization hard constraints and SLA limits.
            gpu_spec: Target GPU specification if available.
            resource_estimate: Calculated model resource estimate.
            evaluated_metrics: Dict of evaluated objective values.
            evidence_status: Dict mapping evaluated objectives to their evidence tiers.

        Returns:
            Tuple of (FeasibilityStatus, list of rejection reasons).
        """
        proven_reasons: list[str] = []
        estimated_reasons: list[str] = []
        unknown_reasons: list[str] = []

        metrics = evaluated_metrics or {}
        ev_status = evidence_status or {}

        # 1. Structural / Whitelist Constraints (Provable)
        if (
            constraints.allowed_quantizations is not None
            and config.quantization not in constraints.allowed_quantizations
        ):
            proven_reasons.append(
                f"Quantization '{config.quantization.value}' is not in allowed quantizations: "
                f"{[q.value for q in constraints.allowed_quantizations]}"
            )

        if (
            constraints.allowed_engines is not None
            and config.engine_type not in constraints.allowed_engines
        ):
            proven_reasons.append(
                f"Engine '{config.engine_type.value}' is not in allowed engines: "
                f"{[e.value for e in constraints.allowed_engines]}"
            )

        if (
            constraints.max_concurrency is not None
            and config.concurrency > constraints.max_concurrency
        ):
            proven_reasons.append(
                f"Concurrency {config.concurrency} exceeds max allowed concurrency of {constraints.max_concurrency}"
            )

        # 2. Engine, Quantization, and Tensor Parallelism Compatibility (Provable)
        supported_q = cls.SUPPORTED_QUANTIZATIONS_BY_ENGINE.get(config.engine_type, set())
        if config.quantization not in supported_q:
            proven_reasons.append(
                f"Engine '{config.engine_type.value}' does not support quantization '{config.quantization.value}'. "
                f"Supported quantizations: {[q.value for q in sorted(supported_q, key=lambda x: x.value)]}"
            )

        if config.tensor_parallel_size > 1 and config.engine_type in (
            EngineType.OLLAMA,
            EngineType.LLAMACPP,
        ):
            proven_reasons.append(
                f"Engine '{config.engine_type.value}' does not support tensor parallelism (TP={config.tensor_parallel_size})"
            )

        # 3. Model Architecture and Attention Head Divisibility (Provable)
        if (
            config.model_spec.num_attention_heads is not None
            and config.tensor_parallel_size > 1
            and config.model_spec.num_attention_heads % config.tensor_parallel_size != 0
        ):
            proven_reasons.append(
                f"Tensor parallel size {config.tensor_parallel_size} does not evenly divide model "
                f"attention heads ({config.model_spec.num_attention_heads})"
            )

        # 4. Topology and Available GPUs Verification
        if constraints.available_gpus is not None:
            if config.tensor_parallel_size > constraints.available_gpus:
                proven_reasons.append(
                    f"Candidate tensor_parallel_size {config.tensor_parallel_size} exceeds available GPUs "
                    f"({constraints.available_gpus})"
                )
        elif config.tensor_parallel_size > 1:
            unknown_reasons.append(
                f"Multi-GPU topology (TP={config.tensor_parallel_size}) cannot be verified: 'available_gpus' "
                "is unspecified in constraints"
            )

        # 5. Context Window Limits (Provable)
        prompt_len = workload.prompt_tokens or 0
        requested_out = workload.output_tokens
        total_seq = prompt_len + requested_out

        if (
            config.model_spec.context_window is not None
            and total_seq > config.model_spec.context_window
        ):
            proven_reasons.append(
                f"Planned sequence length {total_seq} (prompt {prompt_len} + requested output {requested_out}) "
                f"exceeds model context length limit of {config.model_spec.context_window} tokens"
            )

        # 6. Memory Feasibility (Estimated or Measured)
        if (
            metrics.get(OptimizationObjective.MINIMIZE_VRAM) is not None
            and ev_status.get(OptimizationObjective.MINIMIZE_VRAM)
            == EvidenceStatus.MEASURED_EVALUATED
        ):
            per_rank_vram_mb = metrics[OptimizationObjective.MINIMIZE_VRAM]
        elif resource_estimate is not None:
            tp_factor = max(1, config.tensor_parallel_size)
            # Parameter memory is evenly sharded across TP ranks
            param_storage_per_rank = resource_estimate.parameter_storage_mb / tp_factor

            # KV cache sharding: cannot shard across more ranks than there are KV heads
            kv_total = resource_estimate.kv_cache_total_mb or 0.0
            if (
                config.model_spec.num_kv_heads is not None
                and tp_factor > config.model_spec.num_kv_heads
            ):
                effective_kv_shards = max(1, config.model_spec.num_kv_heads)
            else:
                effective_kv_shards = tp_factor
            kv_cache_per_rank = kv_total / effective_kv_shards

            # Apply user vram_safety_margin to per-rank payload and add per-device driver/runtime context
            per_rank_payload = param_storage_per_rank + kv_cache_per_rank
            baseline_driver_context_mb = 512.0  # Allocated per physical GPU process
            per_rank_vram_mb = (
                per_rank_payload * constraints.vram_safety_margin
            ) + baseline_driver_context_mb
        else:
            per_rank_vram_mb = None

        if per_rank_vram_mb is not None:
            # User max VRAM limit check
            if constraints.max_vram_mb is not None and per_rank_vram_mb > constraints.max_vram_mb:
                if (
                    ev_status.get(OptimizationObjective.MINIMIZE_VRAM)
                    == EvidenceStatus.MEASURED_EVALUATED
                ):
                    proven_reasons.append(
                        f"Measured runtime VRAM {per_rank_vram_mb:.1f} MiB per GPU exceeds user max_vram_mb limit "
                        f"of {constraints.max_vram_mb:.1f} MiB"
                    )
                else:
                    estimated_reasons.append(
                        f"Estimated runtime VRAM {per_rank_vram_mb:.1f} MiB per GPU exceeds user max_vram_mb limit "
                        f"of {constraints.max_vram_mb:.1f} MiB"
                    )

            # Hardware device capacity check
            if gpu_spec is not None:
                device_vram_mb = gpu_spec.memory_capacity_gb * 1024.0
                if per_rank_vram_mb > device_vram_mb:
                    if (
                        ev_status.get(OptimizationObjective.MINIMIZE_VRAM)
                        == EvidenceStatus.MEASURED_EVALUATED
                    ):
                        proven_reasons.append(
                            f"Measured memory {per_rank_vram_mb:.1f} MiB per GPU exceeds available device capacity "
                            f"{device_vram_mb:.1f} MiB ({gpu_spec.name}, {gpu_spec.memory_capacity_gb} GB)"
                        )
                    else:
                        estimated_reasons.append(
                            f"Estimated memory {per_rank_vram_mb:.1f} MiB per GPU exceeds available device capacity "
                            f"{device_vram_mb:.1f} MiB ({gpu_spec.name}, {gpu_spec.memory_capacity_gb} GB)"
                        )
            else:
                unknown_reasons.append(
                    "Target GPU hardware specification is missing; device memory capacity cannot be verified"
                )
        else:
            unknown_reasons.append(
                "Model resource estimate was not computed; memory feasibility is unknown"
            )

        # 7. User SLA Performance Constraints (Distinguish Missing, Insufficient, Measured, and Predicted)
        if constraints.max_ttft_ms is not None:
            ttft = metrics.get(OptimizationObjective.MINIMIZE_TTFT)
            if (
                ev_status.get(OptimizationObjective.MINIMIZE_TTFT)
                == EvidenceStatus.INSUFFICIENT_EVIDENCE
            ):
                unknown_reasons.append(
                    f"SLA max_ttft_ms ({constraints.max_ttft_ms:.2f} ms) cannot be verified: TTFT has insufficient evidence (cold-start SLA required or out-of-distribution)"
                )
            elif ttft is None:
                unknown_reasons.append(
                    f"SLA max_ttft_ms ({constraints.max_ttft_ms:.2f} ms) cannot be verified because TTFT was not evaluated"
                )
            elif ttft > constraints.max_ttft_ms:
                if (
                    ev_status.get(OptimizationObjective.MINIMIZE_TTFT)
                    == EvidenceStatus.MEASURED_EVALUATED
                ):
                    proven_reasons.append(
                        f"Measured TTFT {ttft:.2f} ms exceeds SLA maximum of {constraints.max_ttft_ms:.2f} ms"
                    )
                else:
                    estimated_reasons.append(
                        f"Estimated/predicted TTFT {ttft:.2f} ms exceeds SLA maximum of {constraints.max_ttft_ms:.2f} ms"
                    )

        if constraints.max_tpot_ms is not None:
            tpot = metrics.get(OptimizationObjective.MINIMIZE_TPOT)
            if tpot is None:
                unknown_reasons.append(
                    f"SLA max_tpot_ms ({constraints.max_tpot_ms:.2f} ms) cannot be verified because TPOT was not evaluated"
                )
            elif (
                ev_status.get(OptimizationObjective.MINIMIZE_TPOT)
                == EvidenceStatus.INSUFFICIENT_EVIDENCE
            ):
                unknown_reasons.append(
                    f"SLA max_tpot_ms ({constraints.max_tpot_ms:.2f} ms) cannot be verified: TPOT prediction has insufficient evidence (out-of-distribution)"
                )
            elif tpot > constraints.max_tpot_ms:
                if (
                    ev_status.get(OptimizationObjective.MINIMIZE_TPOT)
                    == EvidenceStatus.MEASURED_EVALUATED
                ):
                    proven_reasons.append(
                        f"Measured TPOT {tpot:.2f} ms exceeds SLA maximum of {constraints.max_tpot_ms:.2f} ms"
                    )
                else:
                    estimated_reasons.append(
                        f"Estimated/predicted TPOT {tpot:.2f} ms exceeds SLA maximum of {constraints.max_tpot_ms:.2f} ms"
                    )

        if constraints.min_throughput_tok_s is not None:
            thru = metrics.get(OptimizationObjective.MAXIMIZE_THROUGHPUT)
            if thru is None:
                unknown_reasons.append(
                    f"SLA min_throughput_tok_s ({constraints.min_throughput_tok_s:.2f} tok/s) cannot be verified because throughput was not evaluated"
                )
            elif (
                ev_status.get(OptimizationObjective.MAXIMIZE_THROUGHPUT)
                == EvidenceStatus.INSUFFICIENT_EVIDENCE
            ):
                unknown_reasons.append(
                    f"SLA min_throughput_tok_s ({constraints.min_throughput_tok_s:.2f} tok/s) cannot be verified: throughput prediction has insufficient evidence (out-of-distribution)"
                )
            elif thru < constraints.min_throughput_tok_s:
                if (
                    ev_status.get(OptimizationObjective.MAXIMIZE_THROUGHPUT)
                    == EvidenceStatus.MEASURED_EVALUATED
                ):
                    proven_reasons.append(
                        f"Measured throughput {thru:.2f} tok/s is below SLA minimum of "
                        f"{constraints.min_throughput_tok_s:.2f} tok/s"
                    )
                else:
                    estimated_reasons.append(
                        f"Estimated/predicted throughput {thru:.2f} tok/s is below SLA minimum of "
                        f"{constraints.min_throughput_tok_s:.2f} tok/s"
                    )

        if constraints.max_cost_per_million_tokens is not None:
            cost = metrics.get(OptimizationObjective.MINIMIZE_COST)
            if cost is None:
                unknown_reasons.append(
                    f"SLA max_cost_per_million_tokens (${constraints.max_cost_per_million_tokens:.4f}/1M) cannot be verified because cost was not evaluated"
                )
            elif cost > constraints.max_cost_per_million_tokens:
                estimated_reasons.append(
                    f"Evaluated cost ${cost:.4f}/1M tokens exceeds SLA maximum of "
                    f"${constraints.max_cost_per_million_tokens:.4f}/1M tokens"
                )

        # Determine overall feasibility status by precedence:
        # PROVEN_INFEASIBLE > ESTIMATED_INFEASIBLE > UNKNOWN_FEASIBILITY > FEASIBLE
        if proven_reasons:
            return FeasibilityStatus.PROVEN_INFEASIBLE, proven_reasons
        if estimated_reasons:
            return FeasibilityStatus.ESTIMATED_INFEASIBLE, estimated_reasons
        if unknown_reasons:
            return FeasibilityStatus.UNKNOWN_FEASIBILITY, unknown_reasons

        return FeasibilityStatus.FEASIBLE, []
