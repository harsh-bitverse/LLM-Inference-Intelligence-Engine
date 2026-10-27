"""Unit tests for domain schemas and contracts."""

import pytest
from pydantic import ValidationError

from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    QuantizationType,
    StreamTerminationReason,
    TokenEventType,
)
from llm_ii_engine.core.schemas import (
    EngineCapabilities,
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    Observation,
    RequestMetadata,
    TokenStreamEvent,
    WorkloadSpec,
)


class TestModelSpec:
    """Tests for ModelSpec schema validation and constraints."""

    def test_valid_model_spec(self) -> None:
        spec = ModelSpec(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            family="llama",
            parameter_count_billion=8.03,
            context_window=8192,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            hidden_dimension=4096,
            vocab_size=128256,
        )
        assert spec.model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert spec.parameter_count_billion == 8.03
        assert spec.context_window == 8192

    def test_empty_or_whitespace_model_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelSpec(model_id="")
        with pytest.raises(ValidationError):
            ModelSpec(model_id="   ")

    def test_invalid_parameter_counts(self) -> None:
        with pytest.raises(ValidationError):
            ModelSpec(model_id="llama3", parameter_count_billion=0.0)
        with pytest.raises(ValidationError):
            ModelSpec(model_id="llama3", parameter_count_billion=-7.0)

    def test_invalid_context_window(self) -> None:
        with pytest.raises(ValidationError):
            ModelSpec(model_id="llama3", context_window=0)
        with pytest.raises(ValidationError):
            ModelSpec(model_id="llama3", context_window=-2048)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ModelSpec(model_id="llama3", unexpected_param="bad")  # type: ignore[call-arg]


class TestEngineConfig:
    """Tests for EngineConfig schema validation."""

    def test_valid_engine_config(self) -> None:
        cfg = EngineConfig(
            engine_type=EngineType.MOCK,
            quantization=QuantizationType.INT4,
            max_model_len=4096,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            extra_args={"kv_cache_dtype": "fp8"},
        )
        assert cfg.engine_type == EngineType.MOCK
        assert cfg.quantization == QuantizationType.INT4
        assert cfg.gpu_memory_utilization == 0.85

    def test_invalid_gpu_memory_utilization(self) -> None:
        with pytest.raises(ValidationError):
            EngineConfig(engine_type=EngineType.VLLM, gpu_memory_utilization=0.0)
        with pytest.raises(ValidationError):
            EngineConfig(engine_type=EngineType.VLLM, gpu_memory_utilization=1.5)

    def test_invalid_tensor_parallel_size(self) -> None:
        with pytest.raises(ValidationError):
            EngineConfig(engine_type=EngineType.VLLM, tensor_parallel_size=0)


class TestWorkloadSpec:
    """Tests for WorkloadSpec schema validation."""

    def test_valid_workload_spec(self) -> None:
        wl = WorkloadSpec(
            prompt_tokens=512,
            output_tokens=128,
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["\n\n", "</s>"],
            seed=42,
        )
        assert wl.prompt_tokens == 512
        assert wl.output_tokens == 128
        assert wl.temperature == 0.7

    def test_zero_output_tokens_allowed(self) -> None:
        # 0 output tokens is valid for prefill-only measurement
        wl = WorkloadSpec(prompt_tokens=1024, output_tokens=0)
        assert wl.output_tokens == 0

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadSpec(prompt_tokens=-1, output_tokens=50)
        with pytest.raises(ValidationError):
            WorkloadSpec(prompt_tokens=50, output_tokens=-1)

    def test_invalid_sampling_parameters(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadSpec(prompt_tokens=10, output_tokens=10, temperature=-0.1)
        with pytest.raises(ValidationError):
            WorkloadSpec(prompt_tokens=10, output_tokens=10, top_p=0.0)
        with pytest.raises(ValidationError):
            WorkloadSpec(prompt_tokens=10, output_tokens=10, top_p=1.2)


class TestTokenStreamEvent:
    """Tests for TokenStreamEvent lifecycle contracts, timing, and provenance."""

    def test_valid_start_event(self) -> None:
        event = TokenStreamEvent(
            request_id="req-001",
            sequence_index=0,
            event_type=TokenEventType.START,
            timestamp_ns=1_000_000_000,
            clock_source=ClockSource.SYNTHETIC_VIRTUAL,
            execution_mode=ExecutionMode.SIMULATED,
        )
        assert event.event_type == TokenEventType.START
        assert not event.is_first_token
        assert not event.is_terminal
        assert event.token_index is None

    def test_start_event_with_token_index_rejected(self) -> None:
        with pytest.raises(ValidationError, match="START event must not contain token_index"):
            TokenStreamEvent(
                request_id="req-001",
                sequence_index=0,
                event_type=TokenEventType.START,
                token_index=0,
                timestamp_ns=1_000_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            )

    def test_valid_first_token_event(self) -> None:
        event = TokenStreamEvent(
            request_id="req-001",
            sequence_index=1,
            event_type=TokenEventType.TOKEN,
            token_index=0,
            token_text="Hello",
            token_id=101,
            timestamp_ns=1_050_000_000,
            delta_ns=50_000_000,
            clock_source=ClockSource.SYNTHETIC_VIRTUAL,
            execution_mode=ExecutionMode.SIMULATED,
            is_first_token=True,
        )
        assert event.is_first_token
        assert event.token_index == 0
        assert not event.is_terminal

    def test_mismatched_first_token_flags_rejected(self) -> None:
        # token_index=0 must have is_first_token=True
        with pytest.raises(ValidationError, match="must have is_first_token=True"):
            TokenStreamEvent(
                request_id="req-001",
                sequence_index=1,
                event_type=TokenEventType.TOKEN,
                token_index=0,
                token_text="Hello",
                timestamp_ns=1_050_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=False,
            )

        # token_index=1 must have is_first_token=False
        with pytest.raises(ValidationError, match="must have is_first_token=False"):
            TokenStreamEvent(
                request_id="req-001",
                sequence_index=2,
                event_type=TokenEventType.TOKEN,
                token_index=1,
                token_text=" world",
                timestamp_ns=1_060_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=True,
            )

    def test_token_event_marked_terminal_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot have is_terminal=True"):
            TokenStreamEvent(
                request_id="req-001",
                sequence_index=1,
                event_type=TokenEventType.TOKEN,
                token_index=0,
                token_text="Hello",
                timestamp_ns=1_050_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=True,
                is_terminal=True,
            )

    def test_valid_completion_event(self) -> None:
        event = TokenStreamEvent(
            request_id="req-001",
            sequence_index=5,
            event_type=TokenEventType.COMPLETION,
            token_index=4,
            token_text="",
            timestamp_ns=1_090_000_000,
            delta_ns=1_000_000,
            clock_source=ClockSource.SYNTHETIC_VIRTUAL,
            execution_mode=ExecutionMode.SIMULATED,
            is_terminal=True,
            termination_reason=StreamTerminationReason.FINISHED,
        )
        assert event.is_terminal
        assert event.termination_reason == StreamTerminationReason.FINISHED

    def test_completion_event_not_terminal_rejected(self) -> None:
        with pytest.raises(ValidationError, match="COMPLETION event must have is_terminal=True"):
            TokenStreamEvent(
                request_id="req-001",
                sequence_index=5,
                event_type=TokenEventType.COMPLETION,
                timestamp_ns=1_090_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_terminal=False,
                termination_reason=StreamTerminationReason.FINISHED,
            )

    def test_valid_error_event(self) -> None:
        event = TokenStreamEvent(
            request_id="req-001",
            sequence_index=2,
            event_type=TokenEventType.ERROR,
            timestamp_ns=1_060_000_000,
            clock_source=ClockSource.SYNTHETIC_VIRTUAL,
            execution_mode=ExecutionMode.SIMULATED,
            is_terminal=True,
            termination_reason=StreamTerminationReason.ERROR,
            error_message="CUDA out of memory",
        )
        assert event.is_terminal
        assert event.termination_reason == StreamTerminationReason.ERROR
        assert event.error_message == "CUDA out of memory"

    def test_error_event_without_message_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must include a non-empty error_message"):
            TokenStreamEvent(
                request_id="req-001",
                sequence_index=2,
                event_type=TokenEventType.ERROR,
                timestamp_ns=1_060_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_terminal=True,
                termination_reason=StreamTerminationReason.ERROR,
                error_message="",
            )

    def test_synthetic_virtual_clock_incompatible_with_real_execution(self) -> None:
        with pytest.raises(
            ValidationError,
            match="ExecutionMode.REAL requires ClockSource.MONOTONIC_REAL",
        ):
            TokenStreamEvent(
                request_id="req-001",
                sequence_index=0,
                event_type=TokenEventType.START,
                timestamp_ns=1_000_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.REAL,
            )

    def test_valid_clock_source_and_execution_mode_combinations(self) -> None:
        # 1. Monotonic real clock with real execution
        ev_real = TokenStreamEvent(
            request_id="req-001",
            sequence_index=0,
            event_type=TokenEventType.START,
            timestamp_ns=1_000_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
        )
        assert ev_real.execution_mode == ExecutionMode.REAL
        assert ev_real.clock_source == ClockSource.MONOTONIC_REAL

        # 2. Monotonic real clock with simulated execution (e.g. real-time sleep demo)
        ev_sim_real_clock = TokenStreamEvent(
            request_id="req-002",
            sequence_index=0,
            event_type=TokenEventType.START,
            timestamp_ns=1_000_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
        )
        assert ev_sim_real_clock.execution_mode == ExecutionMode.SIMULATED
        assert ev_sim_real_clock.clock_source == ClockSource.MONOTONIC_REAL


class TestObservation:
    """Tests for Observation container and provenance tagging."""

    def test_measured_real_observation(self) -> None:
        obs = Observation[float](
            value=23.4,
            provenance=ProvenanceType.MEASURED,
            execution_mode=ExecutionMode.REAL,
            clock_source=ClockSource.MONOTONIC_REAL,
            unit="ms",
            description="TTFT prefill latency",
        )
        assert obs.provenance == ProvenanceType.MEASURED
        assert obs.execution_mode == ExecutionMode.REAL
        assert obs.value == 23.4

    def test_predicted_observation(self) -> None:
        obs = Observation[float](
            value=25.0,
            provenance=ProvenanceType.PREDICTED,
            execution_mode=ExecutionMode.SIMULATED,
            unit="ms",
            description="Analytical TTFT estimation",
        )
        assert obs.provenance == ProvenanceType.PREDICTED
        assert obs.execution_mode == ExecutionMode.SIMULATED


class TestGenerationRequest:
    """Tests for full GenerationRequest creation."""

    def test_valid_generation_request(self) -> None:
        model = ModelSpec(model_id="llama3-8b", parameter_count_billion=8.0)
        engine = EngineConfig(engine_type=EngineType.MOCK)
        workload = WorkloadSpec(prompt_tokens=100, output_tokens=20)
        meta = RequestMetadata(
            request_id="r-1",
            model_id="llama3-8b",
            engine_type=EngineType.MOCK,
            execution_mode=ExecutionMode.SIMULATED,
        )

        req = GenerationRequest(
            request_id="r-1",
            model_spec=model,
            engine_config=engine,
            workload=workload,
            metadata=meta,
        )
        assert req.request_id == "r-1"
        assert req.model_spec.model_id == "llama3-8b"
        assert req.workload.output_tokens == 20


class TestSchemaEdgeCasesAndImports:
    """Additional edge cases for schema validation and package imports."""

    def test_heuristic_observation(self) -> None:
        obs = Observation[float](
            value=1008.0,
            provenance=ProvenanceType.HEURISTIC,
            execution_mode=ExecutionMode.SIMULATED,
            unit="GB/s",
            description="Theoretical peak HBM bandwidth for NVIDIA H100 SXM",
        )
        assert obs.provenance == ProvenanceType.HEURISTIC
        assert obs.value == 1008.0

    def test_negative_timing_values_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TokenStreamEvent(
                request_id="req-001",
                sequence_index=0,
                event_type=TokenEventType.START,
                timestamp_ns=-1,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            )

        with pytest.raises(ValidationError):
            TokenStreamEvent(
                request_id="req-001",
                sequence_index=1,
                event_type=TokenEventType.TOKEN,
                token_index=0,
                token_text="test",
                timestamp_ns=1_000,
                delta_ns=-10,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=True,
            )

    def test_engine_capabilities_validation(self) -> None:
        caps = EngineCapabilities(
            engine_type=EngineType.VLLM,
            execution_mode=ExecutionMode.REAL,
            supported_quantizations=[QuantizationType.AWQ, QuantizationType.GPTQ],
            supports_streaming=True,
            supports_gpu_memory_query=True,
            max_context_window=65536,
        )
        assert caps.engine_type == EngineType.VLLM
        assert caps.supports_gpu_memory_query is True
        assert caps.max_context_window == 65536

    def test_top_level_package_exports(self) -> None:
        import llm_ii_engine
        import llm_ii_engine.adapters
        import llm_ii_engine.core

        assert hasattr(llm_ii_engine, "__version__")
        assert hasattr(llm_ii_engine, "discovery")
        assert hasattr(llm_ii_engine, "optimizer")
        assert hasattr(llm_ii_engine.core, "TokenStreamEvent")
        assert hasattr(llm_ii_engine.core, "ProvenanceType")
        assert hasattr(llm_ii_engine.adapters, "MockEngineAdapter")
        assert hasattr(llm_ii_engine.adapters, "InferenceEngineAdapter")
        assert hasattr(llm_ii_engine.adapters, "OllamaAdapter")
        assert hasattr(llm_ii_engine.adapters, "LlamaCppAdapter")
