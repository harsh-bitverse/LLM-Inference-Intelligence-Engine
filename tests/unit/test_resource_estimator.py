"""Unit tests for analytical model resource estimation.

Tests parameter storage calculations (including quantization metadata), attention KV-cache
memory across MHA/GQA/MQA architectures, computational FLOPs, and operational intensity.
"""

import pytest

from llm_ii_engine.analysis.models import ModelResourceEstimate
from llm_ii_engine.analysis.resource_estimator import ModelResourceEstimator
from llm_ii_engine.core.enums import ProvenanceType, QuantizationType
from llm_ii_engine.core.schemas import ModelSpec


class TestParameterStorageEstimation:
    """Test suite for parameter memory storage formulas and quantization overhead."""

    def test_fp16_parameter_storage(self) -> None:
        """7B model at FP16 (2.0 bytes/param) requires ~13,351.44 MiB."""
        mb, bytes_per_param, is_exact, notes = ModelResourceEstimator.estimate_parameter_storage_mb(
            parameter_count_billion=7.0,
            quantization=QuantizationType.FP16,
        )
        # 7e9 * 2 / (1024^2) = 14,000,000,000 / 1,048,576 = 13,351.44 MiB
        expected_mb = round((7.0 * 1e9 * 2.0) / (1024.0 * 1024.0), 2)
        assert mb == expected_mb
        assert bytes_per_param == 2.0
        assert is_exact is True

    def test_int8_parameter_storage(self) -> None:
        """7B model at INT8 (1.0 byte/param) requires ~6,675.72 MiB."""
        mb, bytes_per_param, is_exact, _ = ModelResourceEstimator.estimate_parameter_storage_mb(
            parameter_count_billion=7.0,
            quantization=QuantizationType.INT8,
        )
        expected_mb = round((7.0 * 1e9 * 1.0) / (1024.0 * 1024.0), 2)
        assert mb == expected_mb
        assert bytes_per_param == 1.0
        assert is_exact is True

    def test_int4_raw_parameter_storage(self) -> None:
        """7B model at raw INT4 (0.5 bytes/param) requires ~3,337.86 MiB."""
        mb, bytes_per_param, is_exact, _ = ModelResourceEstimator.estimate_parameter_storage_mb(
            parameter_count_billion=7.0,
            quantization=QuantizationType.INT4,
        )
        expected_mb = round((7.0 * 1e9 * 0.5) / (1024.0 * 1024.0), 2)
        assert mb == expected_mb
        assert bytes_per_param == 0.5
        assert is_exact is True

    def test_gguf_q8_0_quantization_with_block_scales(self) -> None:
        """GGUF Q8_0 uses 32-element INT8 blocks with FP16 scale = 1.0625 bytes/param."""
        mb, bytes_per_param, is_exact, notes = ModelResourceEstimator.estimate_parameter_storage_mb(
            parameter_count_billion=7.0,
            quantization=QuantizationType.GGUF_Q8_0,
        )
        assert bytes_per_param == 1.0625
        assert is_exact is True
        expected_mb = round((7.0 * 1e9 * 1.0625) / (1024.0 * 1024.0), 2)
        assert mb == expected_mb
        assert "Q8_0" in notes

    def test_gguf_q4_k_m_quantization(self) -> None:
        """GGUF Q4_K_M uses ~4.5 bits/param = 0.5625 bytes/param with block scales."""
        mb, bytes_per_param, is_exact, notes = ModelResourceEstimator.estimate_parameter_storage_mb(
            parameter_count_billion=7.0,
            quantization=QuantizationType.GGUF_Q4_K_M,
        )
        assert bytes_per_param == 0.5625
        assert is_exact is False
        assert "Q4_K_M" in notes

    def test_awq_and_gptq_group_128_quantization(self) -> None:
        """AWQ and GPTQ with group size 128 use ~4.25 bits/param = 0.53125 bytes/param."""
        mb_awq, b_awq, exact_awq, _ = ModelResourceEstimator.estimate_parameter_storage_mb(
            parameter_count_billion=7.0,
            quantization=QuantizationType.AWQ,
        )
        mb_gptq, b_gptq, exact_gptq, _ = ModelResourceEstimator.estimate_parameter_storage_mb(
            parameter_count_billion=7.0,
            quantization=QuantizationType.GPTQ,
        )
        assert b_awq == 0.53125
        assert b_gptq == 0.53125
        assert exact_awq is False
        assert mb_awq == mb_gptq

    def test_zero_or_negative_parameter_count_raises(self) -> None:
        """Zero or negative parameter count must raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ModelResourceEstimator.estimate_parameter_storage_mb(0.0)
        with pytest.raises(ValueError, match="must be positive"):
            ModelResourceEstimator.estimate_parameter_storage_mb(-7.0)


class TestKVCacheMemoryEstimation:
    """Test suite for attention KV-cache memory formulas across MHA, GQA, and MQA."""

    def test_mha_kv_cache_calculation(self) -> None:
        """Standard MHA: 32 layers, 32 Q heads, 32 KV heads, head_dim 128, FP16.

        bytes_per_token = 2 * 32 * 32 * 128 * 2 = 524,288 bytes/token.
        For batch size 1 and sequence 2048:
        total = 524,288 * 2048 = 1,073,741,824 bytes = exactly 1024.0 MiB.
        """
        spec = ModelSpec(
            model_id="test-mha-7b",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=32,
            head_dimension=128,
        )
        bytes_per_token, total_mb, arch, notes = ModelResourceEstimator.estimate_kv_cache_memory(
            model_spec=spec,
            batch_size=1,
            sequence_length=2048,
            kv_precision=QuantizationType.FP16,
        )
        assert arch == "MHA"
        assert bytes_per_token == 524288.0
        assert total_mb == 1024.0

    def test_gqa_kv_cache_calculation(self) -> None:
        """Grouped-Query Attention (GQA): 32 layers, 32 Q heads, 8 KV heads, head_dim 128, FP16.

        bytes_per_token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes/token.
        For batch size 1 and sequence 2048:
        total = 131,072 * 2048 = 268,435,456 bytes = exactly 256.0 MiB (4x smaller than MHA).
        """
        spec = ModelSpec(
            model_id="test-gqa-7b",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dimension=128,
        )
        bytes_per_token, total_mb, arch, notes = ModelResourceEstimator.estimate_kv_cache_memory(
            model_spec=spec,
            batch_size=1,
            sequence_length=2048,
            kv_precision=QuantizationType.FP16,
        )
        assert arch == "GQA"
        assert bytes_per_token == 131072.0
        assert total_mb == 256.0

    def test_mqa_kv_cache_calculation(self) -> None:
        """Multi-Query Attention (MQA): 32 layers, 32 Q heads, 1 KV head, head_dim 128, FP16.

        bytes_per_token = 2 * 32 * 1 * 128 * 2 = 16,384 bytes/token.
        For batch size 1 and sequence 2048:
        total = 16,384 * 2048 = 33,554,432 bytes = exactly 32.0 MiB.
        """
        spec = ModelSpec(
            model_id="test-mqa-7b",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=1,
            head_dimension=128,
        )
        bytes_per_token, total_mb, arch, notes = ModelResourceEstimator.estimate_kv_cache_memory(
            model_spec=spec,
            batch_size=1,
            sequence_length=2048,
            kv_precision=QuantizationType.FP16,
        )
        assert arch == "MQA"
        assert bytes_per_token == 16384.0
        assert total_mb == 32.0

    def test_kv_heads_exceeding_q_heads_rejected(self) -> None:
        """num_kv_heads > num_attention_heads is architecturally invalid and must raise."""
        spec = ModelSpec(
            model_id="invalid-heads",
            num_layers=32,
            num_attention_heads=16,
            num_kv_heads=32,
            head_dimension=128,
        )
        with pytest.raises(ValueError, match="cannot exceed"):
            ModelResourceEstimator.estimate_kv_cache_memory(
                model_spec=spec,
                batch_size=1,
                sequence_length=128,
            )

    def test_hidden_dim_derived_head_dim(self) -> None:
        """If head_dimension is None, derive from hidden_dimension // num_attention_heads."""
        spec = ModelSpec(
            model_id="derived-dim",
            num_layers=32,
            num_attention_heads=32,
            hidden_dimension=4096,  # 4096 // 32 = 128
        )
        bytes_per_token, total_mb, arch, _ = ModelResourceEstimator.estimate_kv_cache_memory(
            model_spec=spec,
            batch_size=1,
            sequence_length=1024,
        )
        assert arch == "MHA"
        assert bytes_per_token == 524288.0
        assert total_mb == 512.0

    def test_non_divisible_hidden_dim_rejected(self) -> None:
        """Indivisible hidden_dimension without head_dimension must raise ValueError."""
        spec = ModelSpec(
            model_id="indivisible-dim",
            num_layers=32,
            num_attention_heads=32,
            hidden_dimension=4097,
        )
        with pytest.raises(ValueError, match="must be divisible"):
            ModelResourceEstimator.estimate_kv_cache_memory(
                model_spec=spec,
                batch_size=1,
                sequence_length=128,
            )


class TestComputationalWorkAndResourceEstimate:
    """Test suite for FLOP calculations and compiled ModelResourceEstimate."""

    def test_flop_estimation_with_and_without_attention_dims(self) -> None:
        """Verify GEMM FLOPs and quadratic attention FLOPs."""
        # 1. Without attention dims: GEMM only (2 * P * S)
        prefill_f, decode_f, _ = ModelResourceEstimator.estimate_computational_work_flops(
            parameter_count_billion=7.0,
            prompt_tokens=100,
            output_tokens=50,
            model_spec=None,
        )
        assert prefill_f == 2.0 * 7.0 * 1e9 * 100
        assert decode_f == 2.0 * 7.0 * 1e9 * 50

        # 2. With attention dims: includes quadratic attention
        spec = ModelSpec(
            model_id="llama-7b",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            hidden_dimension=4096,
        )
        prefill_f_attn, decode_f_attn, notes = (
            ModelResourceEstimator.estimate_computational_work_flops(
                parameter_count_billion=7.0,
                prompt_tokens=100,
                output_tokens=50,
                model_spec=spec,
            )
        )
        assert prefill_f_attn > prefill_f
        assert decode_f_attn > decode_f
        assert any("Attention FLOPs included" in n for n in notes)

    def test_full_model_resource_estimate_compilation(self) -> None:
        """Validate comprehensive ModelResourceEstimate compilation and provenance."""
        spec = ModelSpec(
            model_id="meta-llama/Llama-3-8B-Instruct",
            parameter_count_billion=8.0,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dimension=128,
            context_window=8192,
        )
        estimate = ModelResourceEstimator.estimate_model_resources(
            model_spec=spec,
            prompt_tokens=512,
            output_tokens=256,
            batch_size=2,
            weight_precision=QuantizationType.FP16,
            kv_precision=QuantizationType.FP16,
        )
        assert isinstance(estimate, ModelResourceEstimate)
        assert estimate.model_id == "meta-llama/Llama-3-8B-Instruct"
        assert estimate.parameter_count_billion == 8.0
        assert estimate.attention_architecture == "GQA"
        assert estimate.provenance == ProvenanceType.PREDICTED
        assert estimate.sequence_length == 768
        assert estimate.batch_size == 2
        assert estimate.total_payload_memory_mb > 0
        assert estimate.estimated_runtime_vram_mb is not None
        assert estimate.estimated_runtime_vram_mb > estimate.total_payload_memory_mb
        assert estimate.operational_intensity_prefill is not None
        assert estimate.operational_intensity_decode is not None
        assert len(estimate.assumptions) > 0
        assert len(estimate.limitations) == 0
