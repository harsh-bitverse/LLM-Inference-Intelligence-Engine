"""Analytical resource estimation for LLM architectures and inference workloads.

Calculates parameter storage, attention KV-cache memory (MHA/GQA/MQA), computational FLOPs,
and theoretical operational intensity with documented mathematical assumptions.
"""

from llm_ii_engine.analysis.models import ModelResourceEstimate
from llm_ii_engine.core.enums import ProvenanceType, QuantizationType
from llm_ii_engine.core.schemas import ModelSpec


class ModelResourceEstimator:
    """Estimates parameter memory, KV-cache footprint, and computational work for LLMs."""

    # Bytes per parameter mapping for supported quantization formats
    # Format: (bytes_per_param, is_exact, notes)
    QUANTIZATION_SPEC: dict[QuantizationType, tuple[float, bool, str]] = {
        QuantizationType.NONE: (
            2.0,
            True,
            "Baseline assumed FP16 (16 bits/param = 2.0 bytes) for unquantized models",
        ),
        QuantizationType.FP16: (2.0, True, "Standard IEEE 754 half-precision float (16 bits)"),
        QuantizationType.BF16: (2.0, True, "Bfloat16 precision format (16 bits)"),
        QuantizationType.INT8: (
            1.0,
            True,
            "8-bit integer quantization (8 bits/param = 1.0 byte)",
        ),
        QuantizationType.INT4: (
            0.5,
            True,
            "Raw 4-bit integer packing (4 bits/param = 0.5 bytes, excluding scaling metadata)",
        ),
        QuantizationType.AWQ: (
            0.53125,
            False,
            "Activation-aware Weight Quantization (4-bit weights + FP16 scales/zeros with group size 128 = ~4.25 bits/param)",
        ),
        QuantizationType.GPTQ: (
            0.53125,
            False,
            "GPTQ 4-bit matrix quantization with group size 128 = ~4.25 bits/param",
        ),
        QuantizationType.GGUF_Q4_K_M: (
            0.5625,
            False,
            "GGUF Q4_K_M medium quantization: mixed 4-bit and 6-bit quantization with block scales (~4.5 bits/param)",
        ),
        QuantizationType.GGUF_Q8_0: (
            1.0625,
            True,
            "GGUF Q8_0 quantization: 32-element INT8 blocks with 16-bit FP16 block scale = (32 + 2)/32 = 1.0625 bytes/param",
        ),
    }

    @classmethod
    def get_quantization_bytes(cls, quantization: QuantizationType) -> tuple[float, bool, str]:
        """Return (bytes_per_param, is_exact, notes) for a quantization format."""
        return cls.QUANTIZATION_SPEC.get(
            quantization,
            (
                2.0,
                False,
                f"Unknown quantization '{quantization}'; defaulting to 2.0 bytes/parameter",
            ),
        )

    @classmethod
    def estimate_parameter_storage_mb(
        cls,
        parameter_count_billion: float,
        quantization: QuantizationType = QuantizationType.NONE,
    ) -> tuple[float, float, bool, str]:
        """Calculate logical parameter memory in MiB.

        Args:
            parameter_count_billion: Model parameter count in billions.
            quantization: Quantization format.

        Returns:
            Tuple of (storage_mb, bytes_per_param, is_exact, notes).
        """
        if parameter_count_billion <= 0:
            raise ValueError(
                f"parameter_count_billion must be positive, got {parameter_count_billion}"
            )

        bytes_per_param, is_exact, notes = cls.get_quantization_bytes(quantization)
        total_bytes = parameter_count_billion * 1e9 * bytes_per_param
        storage_mb = total_bytes / (1024.0 * 1024.0)
        return round(storage_mb, 2), bytes_per_param, is_exact, notes

    @classmethod
    def estimate_kv_cache_memory(
        cls,
        model_spec: ModelSpec,
        batch_size: int,
        sequence_length: int,
        kv_precision: QuantizationType = QuantizationType.FP16,
    ) -> tuple[float, float, str, list[str]]:
        """Calculate attention KV-cache memory in MiB and bytes-per-token.

        Formula for decoder-only transformer:
            bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
            total_bytes = bytes_per_token * batch_size * sequence_length

        Supports:
            - Multi-Head Attention (MHA): num_kv_heads == num_attention_heads
            - Grouped-Query Attention (GQA): 1 < num_kv_heads < num_attention_heads
            - Multi-Query Attention (MQA): num_kv_heads == 1

        Returns:
            Tuple of (kv_cache_per_token_bytes, total_kv_cache_mb, attention_architecture, notes).
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if sequence_length < 0:
            raise ValueError(f"sequence_length must be >= 0, got {sequence_length}")

        notes: list[str] = []

        if model_spec.num_layers is None or model_spec.num_attention_heads is None:
            raise ValueError(
                "ModelSpec must specify 'num_layers' and 'num_attention_heads' to calculate KV-cache memory"
            )

        num_layers = model_spec.num_layers
        num_q_heads = model_spec.num_attention_heads

        # Determine num_kv_heads
        if model_spec.num_kv_heads is not None:
            num_kv_heads = model_spec.num_kv_heads
            if num_kv_heads > num_q_heads:
                raise ValueError(
                    f"num_kv_heads ({num_kv_heads}) cannot exceed num_attention_heads ({num_q_heads})"
                )
        else:
            num_kv_heads = num_q_heads
            notes.append(
                "num_kv_heads not specified; assumed standard Multi-Head Attention (num_kv_heads == num_attention_heads)"
            )

        # Classify attention architecture
        if num_kv_heads == num_q_heads:
            arch = "MHA"
        elif num_kv_heads == 1:
            arch = "MQA"
        else:
            arch = "GQA"

        # Determine head_dimension
        if model_spec.head_dimension is not None:
            head_dim = model_spec.head_dimension
        elif model_spec.hidden_dimension is not None:
            if model_spec.hidden_dimension % num_q_heads != 0:
                raise ValueError(
                    f"hidden_dimension ({model_spec.hidden_dimension}) must be divisible by "
                    f"num_attention_heads ({num_q_heads})"
                )
            head_dim = model_spec.hidden_dimension // num_q_heads
        else:
            raise ValueError(
                "ModelSpec must specify either 'head_dimension' or 'hidden_dimension' to calculate KV-cache memory"
            )

        bytes_per_elem, _, _ = cls.get_quantization_bytes(kv_precision)

        # Factor 2 accounts for separate Key and Value tensor allocations
        bytes_per_token = 2.0 * num_layers * num_kv_heads * head_dim * bytes_per_elem
        total_bytes = bytes_per_token * batch_size * sequence_length
        total_mb = total_bytes / (1024.0 * 1024.0)

        notes.append(
            f"KV cache computed for {arch} ({num_layers} layers, {num_kv_heads} KV heads, {head_dim} head dim, "
            f"{bytes_per_elem} bytes/elem). Factor 2 accounts for Key and Value tensors."
        )

        return round(bytes_per_token, 2), round(total_mb, 2), arch, notes

    @classmethod
    def estimate_computational_work_flops(
        cls,
        parameter_count_billion: float,
        prompt_tokens: int,
        output_tokens: int,
        model_spec: ModelSpec | None = None,
    ) -> tuple[float, float, list[str]]:
        """Calculate estimated computational work in FLOPs for prefill and decode phases.

        Formulas:
            Prefill FLOPs = 2 * P * S_prompt + 4 * num_layers * d_hidden * (S_prompt)^2
            Decode FLOPs = 2 * P * S_decode + sum_{i=1}^{S_decode} [4 * num_layers * d_hidden * (S_prompt + i)]

        Where 2 * P represents linear projections (1 multiply + 1 add per parameter per token),
        and the secondary term models quadratic attention score computation and softmax application.

        Returns:
            Tuple of (prefill_flops, decode_flops, assumptions).
        """
        assumptions: list[str] = [
            "2 FLOPs per parameter per token for forward pass matrix multiplications (GEMMs)",
        ]

        p_params = parameter_count_billion * 1e9

        # 1. Base GEMM FLOPs
        prefill_gemm_flops = 2.0 * p_params * prompt_tokens
        decode_gemm_flops = 2.0 * p_params * output_tokens

        # 2. Attention FLOPs if architecture dimensions are available
        has_attn_dims = (
            model_spec is not None
            and model_spec.num_layers is not None
            and model_spec.hidden_dimension is not None
        )

        if has_attn_dims:
            num_layers_val = model_spec.num_layers
            d_hidden = model_spec.hidden_dimension
            # Attention FLOPs for prefill: 2 * (QK^T) + 2 * (softmax * V) = 4 * l * d_hidden * s^2
            prefill_attn_flops = 4.0 * num_layers_val * d_hidden * (prompt_tokens**2)

            # Attention FLOPs for decode: query token attending to growing context
            decode_attn_flops = 0.0
            for step in range(1, output_tokens + 1):
                ctx_len = prompt_tokens + step
                decode_attn_flops += 4.0 * num_layers_val * d_hidden * ctx_len

            total_prefill_flops = prefill_gemm_flops + prefill_attn_flops
            total_decode_flops = decode_gemm_flops + decode_attn_flops
            assumptions.append(
                f"Attention FLOPs included using 4 * num_layers ({num_layers_val}) * d_hidden ({d_hidden}) * S^2 for prefill, "
                "and incremental causal attention projections for decode."
            )
        else:
            total_prefill_flops = prefill_gemm_flops
            total_decode_flops = decode_gemm_flops
            assumptions.append(
                "Attention self-interaction FLOPs omitted because 'num_layers' or 'hidden_dimension' is unspecified. "
                "GEMM 2*P*S dominates (>95% of total compute for moderate contexts)."
            )

        return round(total_prefill_flops, 2), round(total_decode_flops, 2), assumptions

    @classmethod
    def estimate_model_resources(
        cls,
        model_spec: ModelSpec,
        prompt_tokens: int = 128,
        output_tokens: int = 128,
        batch_size: int = 1,
        weight_precision: QuantizationType = QuantizationType.NONE,
        kv_precision: QuantizationType = QuantizationType.FP16,
    ) -> ModelResourceEstimate:
        """Compile a comprehensive ModelResourceEstimate for a model and workload.

        Args:
            model_spec: Model architecture specifications.
            prompt_tokens: Prompt token length.
            output_tokens: Generated output token length.
            batch_size: Concurrency / batch size.
            weight_precision: Weight storage precision.
            kv_precision: KV-cache storage precision.

        Returns:
            Validated ModelResourceEstimate with logical storage, KV-cache, FLOPs, and AI.
        """
        assumptions: list[str] = []
        limitations: list[str] = []

        # 1. Parameter storage
        if model_spec.parameter_count_billion is not None:
            param_storage_mb, bytes_per_param, is_exact, q_notes = (
                cls.estimate_parameter_storage_mb(
                    model_spec.parameter_count_billion, weight_precision
                )
            )
            assumptions.append(
                f"Parameter weights: {model_spec.parameter_count_billion:.2f}B parameters at {bytes_per_param} bytes/param"
            )
            if q_notes:
                assumptions.append(q_notes)
        else:
            param_storage_mb = 0.0
            bytes_per_param = 2.0
            is_exact = False
            q_notes = None
            limitations.append(
                "model_spec.parameter_count_billion is missing; parameter storage and FLOPs cannot be calculated"
            )

        # 2. KV cache memory
        total_seq_len = prompt_tokens + output_tokens
        kv_per_token_bytes: float | None = None
        kv_total_mb: float | None = None
        attn_arch = "UNKNOWN"

        try:
            kv_per_token_bytes, kv_total_mb, attn_arch, kv_notes = cls.estimate_kv_cache_memory(
                model_spec=model_spec,
                batch_size=batch_size,
                sequence_length=total_seq_len,
                kv_precision=kv_precision,
            )
            assumptions.extend(kv_notes)
        except ValueError as err:
            limitations.append(f"KV-cache calculation incomplete: {err}")

        # 3. Payload and estimated runtime VRAM
        total_payload_mb = param_storage_mb + (kv_total_mb or 0.0)
        # Runtime overhead assumption: 15% allocator & workspace margin + 512 MiB CUDA/runtime context
        estimated_runtime_vram_mb = (
            round((total_payload_mb * 1.15) + 512.0, 2) if param_storage_mb > 0 else None
        )
        if estimated_runtime_vram_mb is not None:
            assumptions.append(
                "Estimated runtime VRAM includes 15% allocator/activation margin and 512 MiB baseline driver/runtime context overhead."
            )

        # 4. Computational work (FLOPs)
        prefill_flops: float | None = None
        decode_flops: float | None = None
        oi_prefill: float | None = None
        oi_decode: float | None = None

        if model_spec.parameter_count_billion is not None:
            prefill_flops, decode_flops, flop_assumptions = cls.estimate_computational_work_flops(
                parameter_count_billion=model_spec.parameter_count_billion,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                model_spec=model_spec,
            )
            assumptions.extend(flop_assumptions)

            # 5. Operational Intensity (FLOPs / Byte)
            # Prefill: FLOPs / (Weight Bytes transferred + KV Cache Written)
            weight_bytes = model_spec.parameter_count_billion * 1e9 * bytes_per_param
            if prompt_tokens > 0 and prefill_flops is not None:
                kv_prefill_bytes = (kv_per_token_bytes or 0.0) * batch_size * prompt_tokens
                prefill_traffic_bytes = weight_bytes + kv_prefill_bytes
                if prefill_traffic_bytes > 0:
                    oi_prefill = round(prefill_flops / prefill_traffic_bytes, 2)
                    assumptions.append(
                        f"Prefill operational intensity ({oi_prefill} FLOPs/Byte) assumes weights are streamed once "
                        "and KV cache is written to VRAM for prompt tokens."
                    )

            # Decode: In each decode step, the entire weight tensor is read once per batch,
            # plus reading the active KV cache for prompt+generated tokens.
            if output_tokens > 0 and decode_flops is not None:
                avg_ctx = prompt_tokens + (output_tokens / 2.0)
                kv_decode_read_bytes = (
                    (kv_per_token_bytes or 0.0) * batch_size * avg_ctx * output_tokens
                )
                # Weights are read from VRAM for each output step (batch size amortizes weight reads across batch)
                total_decode_weight_bytes = (weight_bytes / max(1, batch_size)) * output_tokens
                decode_traffic_bytes = total_decode_weight_bytes + kv_decode_read_bytes
                if decode_traffic_bytes > 0:
                    oi_decode = round(decode_flops / decode_traffic_bytes, 2)
                    assumptions.append(
                        f"Decode operational intensity ({oi_decode} FLOPs/Byte) models weight transfer per decode step "
                        f"with batch size {batch_size} and KV cache read operations."
                    )

        return ModelResourceEstimate(
            model_id=model_spec.model_id,
            parameter_count_billion=model_spec.parameter_count_billion,
            parameter_storage_mb=param_storage_mb,
            weight_precision=weight_precision,
            bytes_per_parameter=bytes_per_param,
            is_quantization_exact=is_exact,
            quantization_notes=q_notes,
            attention_architecture=attn_arch,
            kv_cache_per_token_bytes=kv_per_token_bytes,
            kv_cache_total_mb=kv_total_mb,
            kv_cache_precision=kv_precision,
            batch_size=batch_size,
            sequence_length=total_seq_len,
            total_payload_memory_mb=total_payload_mb,
            estimated_runtime_vram_mb=estimated_runtime_vram_mb,
            prefill_flops=prefill_flops,
            decode_flops=decode_flops,
            operational_intensity_prefill=oi_prefill,
            operational_intensity_decode=oi_decode,
            provenance=ProvenanceType.PREDICTED,
            assumptions=assumptions,
            limitations=limitations,
        )
