"""Unit tests for MockEngineAdapter and adapter protocol conformance."""

import asyncio

import pytest

from llm_ii_engine.adapters.base import InferenceEngineAdapter
from llm_ii_engine.adapters.mock import MockAdapterConfig, MockEngineAdapter
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    QuantizationType,
    StreamTerminationReason,
    TokenEventType,
)
from llm_ii_engine.core.exceptions import (
    EngineConnectionError,
    EngineExecutionError,
    StreamCancelledError,
)
from llm_ii_engine.core.schemas import (
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    WorkloadSpec,
)


@pytest.fixture
def default_model_spec() -> ModelSpec:
    return ModelSpec(model_id="mock-model", parameter_count_billion=7.0)


@pytest.fixture
def default_engine_config() -> EngineConfig:
    return EngineConfig(engine_type=EngineType.MOCK, quantization=QuantizationType.FP16)


def create_request(
    output_tokens: int = 5,
    prompt_tokens: int = 128,
    seed: int = 42,
) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-unit-test",
        model_spec=ModelSpec(model_id="mock-model"),
        engine_config=EngineConfig(engine_type=EngineType.MOCK),
        workload=WorkloadSpec(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            seed=seed,
        ),
    )


class TestMockEngineAdapterProtocol:
    """Verify MockEngineAdapter satisfies the InferenceEngineAdapter protocol."""

    def test_protocol_conformance(self) -> None:
        adapter = MockEngineAdapter()
        assert isinstance(adapter, InferenceEngineAdapter)
        assert adapter.engine_type == EngineType.MOCK
        assert adapter.execution_mode == ExecutionMode.SIMULATED

    def test_capabilities(self) -> None:
        adapter = MockEngineAdapter()
        caps = adapter.get_capabilities()
        assert caps.engine_type == EngineType.MOCK
        assert caps.execution_mode == ExecutionMode.SIMULATED
        assert caps.supports_streaming is True
        assert caps.supports_gpu_memory_query is True
        assert QuantizationType.FP16 in caps.supported_quantizations

    @pytest.mark.asyncio
    async def test_lifecycle_and_health_check(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        adapter = MockEngineAdapter()
        assert not adapter.is_initialized
        assert await adapter.health_check() is True

        init_ok = await adapter.initialize(default_model_spec, default_engine_config)
        assert init_ok is True
        assert adapter.is_initialized is True

        mem_mb = await adapter.get_memory_usage_mb()
        assert mem_mb == 1024.0

        await adapter.teardown()
        assert adapter.is_initialized is False

    @pytest.mark.asyncio
    async def test_simulated_initialization_failure(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        config = MockAdapterConfig(fail_on_initialize=True)
        adapter = MockEngineAdapter(config=config)
        with pytest.raises(EngineConnectionError, match="Simulated initialization failure"):
            await adapter.initialize(default_model_spec, default_engine_config)

    @pytest.mark.asyncio
    async def test_simulated_health_check_failure(self) -> None:
        config = MockAdapterConfig(fail_on_health_check=True)
        adapter = MockEngineAdapter(config=config)
        assert await adapter.health_check() is False


class TestMockEngineAdapterStreaming:
    """Verify deterministic streaming generation, timing, and edge cases."""

    @pytest.mark.asyncio
    async def test_standard_stream_sequence_and_ordering(self) -> None:
        config = MockAdapterConfig(
            prefill_delay_ns=40_000_000,
            inter_token_delay_ns=10_000_000,
            jitter_ratio=0.0,
            virtual_time_mode=True,
            initial_timestamp_ns=1_000_000_000,
            token_prefix="test_tok_",
        )
        adapter = MockEngineAdapter(config=config)
        request = create_request(output_tokens=4)

        events = []
        async for event in adapter.generate_stream(request):
            events.append(event)

        # Expected: 1 START + 4 TOKEN + 1 COMPLETION = 6 events
        assert len(events) == 6

        # Check sequence indices
        for idx, event in enumerate(events):
            assert event.sequence_index == idx
            assert event.request_id == "req-unit-test"
            assert event.execution_mode == ExecutionMode.SIMULATED
            assert event.clock_source == ClockSource.SYNTHETIC_VIRTUAL

        # Event 0: START
        assert events[0].event_type == TokenEventType.START
        assert events[0].timestamp_ns == 1_000_000_000
        assert events[0].token_index is None

        # Event 1: First Token (TTFT = 40ms)
        assert events[1].event_type == TokenEventType.TOKEN
        assert events[1].token_index == 0
        assert events[1].is_first_token is True
        assert events[1].token_text == "test_tok_0"
        assert events[1].delta_ns == 40_000_000
        assert events[1].timestamp_ns == 1_040_000_000

        # Event 2: Second Token (ITL = 10ms)
        assert events[2].event_type == TokenEventType.TOKEN
        assert events[2].token_index == 1
        assert events[2].is_first_token is False
        assert events[2].token_text == "test_tok_1"
        assert events[2].delta_ns == 10_000_000
        assert events[2].timestamp_ns == 1_050_000_000

        # Event 3: Third Token
        assert events[3].token_index == 2
        assert events[3].timestamp_ns == 1_060_000_000

        # Event 4: Fourth Token
        assert events[4].token_index == 3
        assert events[4].timestamp_ns == 1_070_000_000

        # Event 5: COMPLETION
        assert events[5].event_type == TokenEventType.COMPLETION
        assert events[5].is_terminal is True
        assert events[5].termination_reason == StreamTerminationReason.FINISHED
        assert events[5].timestamp_ns > events[4].timestamp_ns

    @pytest.mark.asyncio
    async def test_strictly_monotonic_timestamps(self) -> None:
        config = MockAdapterConfig(jitter_ratio=0.2, seed=123, virtual_time_mode=True)
        adapter = MockEngineAdapter(config=config)
        request = create_request(output_tokens=10)

        events = [ev async for ev in adapter.generate_stream(request)]
        for i in range(1, len(events)):
            assert events[i].timestamp_ns > events[i - 1].timestamp_ns, (
                f"Timestamp not monotonically increasing at event {i}: "
                f"{events[i].timestamp_ns} <= {events[i - 1].timestamp_ns}"
            )

    @pytest.mark.asyncio
    async def test_reproducibility_with_identical_seeds(self) -> None:
        config = MockAdapterConfig(jitter_ratio=0.5, seed=999, virtual_time_mode=True)
        adapter1 = MockEngineAdapter(config=config)
        adapter2 = MockEngineAdapter(config=config)

        request1 = create_request(output_tokens=5, seed=999)
        request2 = create_request(output_tokens=5, seed=999)

        events1 = [ev async for ev in adapter1.generate_stream(request1)]
        events2 = [ev async for ev in adapter2.generate_stream(request2)]

        assert len(events1) == len(events2)
        for ev1, ev2 in zip(events1, events2):
            assert ev1.timestamp_ns == ev2.timestamp_ns
            assert ev1.delta_ns == ev2.delta_ns
            assert ev1.token_text == ev2.token_text

    @pytest.mark.asyncio
    async def test_different_seeds_produce_different_jitter(self) -> None:
        config1 = MockAdapterConfig(jitter_ratio=0.5, seed=100, virtual_time_mode=True)
        config2 = MockAdapterConfig(jitter_ratio=0.5, seed=200, virtual_time_mode=True)
        adapter1 = MockEngineAdapter(config=config1)
        adapter2 = MockEngineAdapter(config=config2)

        request1 = create_request(output_tokens=5, seed=100)
        request2 = create_request(output_tokens=5, seed=200)

        events1 = [ev async for ev in adapter1.generate_stream(request1)]
        events2 = [ev async for ev in adapter2.generate_stream(request2)]

        timestamps1 = [ev.timestamp_ns for ev in events1]
        timestamps2 = [ev.timestamp_ns for ev in events2]
        assert timestamps1 != timestamps2

    @pytest.mark.asyncio
    async def test_zero_token_output_edge_case(self) -> None:
        # Workload requesting 0 output tokens (prefill-only benchmark)
        adapter = MockEngineAdapter()
        request = create_request(output_tokens=0)

        events = [ev async for ev in adapter.generate_stream(request)]
        # Expected: START + COMPLETION (0 token events)
        assert len(events) == 2
        assert events[0].event_type == TokenEventType.START
        assert events[1].event_type == TokenEventType.COMPLETION
        assert events[1].is_terminal is True
        assert events[1].termination_reason == StreamTerminationReason.MAX_TOKENS

    @pytest.mark.asyncio
    async def test_single_token_output_edge_case(self) -> None:
        adapter = MockEngineAdapter()
        request = create_request(output_tokens=1)

        events = [ev async for ev in adapter.generate_stream(request)]
        # Expected: START + 1 TOKEN + COMPLETION
        assert len(events) == 3
        assert events[0].event_type == TokenEventType.START
        assert events[1].event_type == TokenEventType.TOKEN
        assert events[1].token_index == 0
        assert events[1].is_first_token is True
        assert events[2].event_type == TokenEventType.COMPLETION
        assert events[2].is_terminal is True

    @pytest.mark.asyncio
    async def test_error_injection_as_terminal_event(self) -> None:
        config = MockAdapterConfig(
            error_at_token_index=2,
            error_delivery_mode="event",
            virtual_time_mode=True,
        )
        adapter = MockEngineAdapter(config=config)
        request = create_request(output_tokens=5)

        events = [ev async for ev in adapter.generate_stream(request)]
        # Sequence: START, TOKEN 0, TOKEN 1, ERROR (at index 2)
        assert len(events) == 4
        assert events[0].event_type == TokenEventType.START
        assert events[1].event_type == TokenEventType.TOKEN
        assert events[2].event_type == TokenEventType.TOKEN
        assert events[3].event_type == TokenEventType.ERROR
        assert events[3].is_terminal is True
        assert events[3].termination_reason == StreamTerminationReason.ERROR
        assert "Simulated injected failure" in (events[3].error_message or "")

    @pytest.mark.asyncio
    async def test_error_injection_as_exception(self) -> None:
        config = MockAdapterConfig(
            error_at_token_index=1,
            error_delivery_mode="exception",
            virtual_time_mode=True,
        )
        adapter = MockEngineAdapter(config=config)
        request = create_request(output_tokens=5)

        with pytest.raises(EngineExecutionError, match="Simulated injected exception"):
            async for _ in adapter.generate_stream(request):
                pass

    @pytest.mark.asyncio
    async def test_cancellation_handling(self) -> None:
        config = MockAdapterConfig(
            inter_token_delay_ns=10_000_000,
            virtual_time_mode=False,  # Uses real sleep so task can be cancelled mid-stream
        )
        adapter = MockEngineAdapter(config=config)
        request = create_request(output_tokens=50)

        async def run_consumer() -> None:
            async for _ in adapter.generate_stream(request):
                await asyncio.sleep(0.01)

        task = asyncio.create_task(run_consumer())
        await asyncio.sleep(0.02)
        task.cancel()

        with pytest.raises(StreamCancelledError):
            await task

    @pytest.mark.asyncio
    async def test_custom_token_prefix_generation(self) -> None:
        config = MockAdapterConfig(token_prefix="custom_token_")
        adapter = MockEngineAdapter(config=config)
        request = create_request(output_tokens=2)

        events = [ev async for ev in adapter.generate_stream(request)]
        token_events = [ev for ev in events if ev.event_type == TokenEventType.TOKEN]
        assert len(token_events) == 2
        assert token_events[0].token_text == "custom_token_0"
        assert token_events[1].token_text == "custom_token_1"

    def test_base_adapter_cannot_be_instantiated_directly(self) -> None:
        from llm_ii_engine.adapters.base import BaseInferenceAdapter

        with pytest.raises(TypeError):
            BaseInferenceAdapter(  # type: ignore[abstract]
                name="test",
                engine_type=EngineType.MOCK,
                execution_mode=ExecutionMode.SIMULATED,
            )

    def test_mock_config_forbids_extra_fields(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MockAdapterConfig(unknown_setting=123)  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_pure_offline_operation(self) -> None:
        # Verify complete operation without network, GPU, or model weights
        adapter = MockEngineAdapter()
        request = create_request(output_tokens=3)
        events = [ev async for ev in adapter.generate_stream(request)]
        assert len(events) == 5  # START + 3 TOKENS + COMPLETION
