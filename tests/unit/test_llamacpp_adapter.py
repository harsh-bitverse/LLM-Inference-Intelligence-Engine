"""Unit and offline integration tests for llama.cpp server HTTP streaming adapter.

Uses httpx.MockTransport to test SSE streaming, token-id vs chunk discrimination,
chunk-boundary splitting, timeouts, cancellation, and metric collection without a live llama.cpp binary.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from llm_ii_engine.adapters.llamacpp import LlamaCppAdapter
from llm_ii_engine.benchmark.collector import MetricCollector
from llm_ii_engine.benchmark.runner import BenchmarkRunner, BenchmarkRunnerConfig
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    StreamTerminationReason,
    TokenCountSource,
    TokenEventType,
)
from llm_ii_engine.core.exceptions import (
    EngineConnectionError,
    EngineTimeoutError,
    StreamCancelledError,
)
from llm_ii_engine.core.schemas import (
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    WorkloadSpec,
)


class ChunkedByteStream(httpx.AsyncByteStream):
    """Custom byte stream yielding raw byte chunks to simulate network packet boundaries."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def default_model_spec() -> ModelSpec:
    return ModelSpec(model_id="llama-2-7b.Q4_K_M.gguf", parameter_count_billion=7.0)


@pytest.fixture
def default_engine_config() -> EngineConfig:
    return EngineConfig(engine_type=EngineType.LLAMACPP)


class TestLlamaCppAdapterProtocolAndLifecycle:
    """Tests for llama.cpp protocol conformance, capabilities, and lifecycle checks."""

    def test_capabilities(self) -> None:
        adapter = LlamaCppAdapter(base_url="http://localhost:8080")
        caps = adapter.get_capabilities()
        assert caps.engine_type == EngineType.LLAMACPP
        assert caps.execution_mode == ExecutionMode.REAL
        assert caps.supports_streaming is True
        assert caps.supports_token_ids is True
        assert caps.supports_true_token_itl is False  # By default text chunks
        assert caps.supports_server_metrics is True

    @pytest.mark.asyncio
    async def test_health_check_ok(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        assert await adapter.health_check() is True
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_health_check_loading_model(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "loading model"})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        assert await adapter.health_check() is False
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_initialize_success_with_props(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            if request.url.path == "/props":
                return httpx.Response(
                    200,
                    json={
                        "default_generation_settings": {
                            "n_ctx": 4096,
                            "model": "llama-2-7b.Q4_K_M.gguf",
                        }
                    },
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        success = await adapter.initialize(default_model_spec, default_engine_config)
        assert success is True
        assert adapter.is_initialized is True
        caps = adapter.get_capabilities()
        assert caps.max_context_window == 4096
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_initialize_loading_model_raises_connection_error(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "loading model"})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)

        with pytest.raises(EngineConnectionError, match="still loading a model"):
            await adapter.initialize(default_model_spec, default_engine_config)

        await adapter.teardown()


class TestLlamaCppAdapterStreamingAndSSE:
    """Tests for SSE streaming, text chunk parsing vs token ids, and metric fidelity."""

    @pytest.mark.asyncio
    async def test_standard_sse_text_chunk_streaming(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            assert request.url.path == "/completion"
            body = json.loads(request.content.decode("utf-8"))
            assert "prompt" in body
            assert body["stream"] is True

            sse_lines = [
                b": ping\n\n",
                b'data: {"content":"High","id_slot":0,"stop":false}\n\n',
                b'data: {"content":" throughput","id_slot":0,"stop":false}\n\n',
                b'data: {"content":"","id_slot":0,"stop":true,"stopped_eos":true,'
                b'"timings":{"prompt_n":12,"prompt_ms":30.0,"predicted_n":2,'
                b'"predicted_ms":60.0,"predicted_per_second":33.33}}\n\n',
            ]
            return httpx.Response(200, stream=ChunkedByteStream(sse_lines))

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        await adapter.initialize(default_model_spec, default_engine_config)

        request = GenerationRequest(
            request_id="llamacpp-req-1",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=12, output_tokens=16),
        )

        events = [e async for e in adapter.generate_stream(request)]

        assert len(events) == 4  # START + 2 CHUNKs + COMPLETION
        assert events[0].event_type == TokenEventType.START
        assert events[1].event_type == TokenEventType.CHUNK
        assert events[1].token_text == "High"
        assert events[2].event_type == TokenEventType.CHUNK
        assert events[2].token_text == " throughput"

        comp_event = events[3]
        assert comp_event.event_type == TokenEventType.COMPLETION
        assert comp_event.is_terminal is True
        assert comp_event.termination_reason == StreamTerminationReason.FINISHED
        assert comp_event.server_stats is not None
        assert comp_event.server_stats.output_tokens == 2
        assert comp_event.server_stats.prompt_tokens == 12
        assert comp_event.server_stats.eval_duration_ms == 60.0

        # Pass through collector
        res = MetricCollector.collect_request(events, prompt_tokens=12)
        assert res.success is True
        assert res.output_tokens == 2
        assert res.token_count_source == TokenCountSource.SERVER_REPORTED
        assert res.prompt_tokens == 12
        assert res.prompt_token_count_source == TokenCountSource.SERVER_REPORTED
        # Verify TTFC is captured, but true token-level TTFT, ITL, and TPOT are None for chunk streams
        assert res.ttfc_ms is not None
        assert res.ttft_ms is None
        assert res.inter_token_latencies_ms == []
        assert res.itl_p50_ms is None
        assert res.tpot_ms is None

        # Verify chunk-level latencies are tracked
        assert res.chunk_count == 2
        assert len(res.chunk_latencies_ms) == 1

        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_sse_streaming_with_discrete_token_ids(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            sse_lines = [
                b'data: {"content":"First","token_id":1001,"stop":false}\n\n',
                b'data: {"content":"Second","token_id":1002,"stop":false}\n\n',
                b'data: {"content":"Third","token_id":1003,"stop":false}\n\n',
                b'data: {"content":"","id_slot":0,"stop":true,"timings":{"prompt_n":10,"predicted_n":3}}\n\n',
            ]
            return httpx.Response(200, stream=ChunkedByteStream(sse_lines))

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)

        request = GenerationRequest(
            request_id="llamacpp-tokens-1",
            model_spec=default_model_spec,
            engine_config=EngineConfig(
                engine_type=EngineType.LLAMACPP,
                extra_args={"skip_model_check": True, "verified_token_stream": True},
            ),
            workload=WorkloadSpec(prompt_tokens=10, output_tokens=10),
        )

        events = [e async for e in adapter.generate_stream(request)]

        assert len(events) == 5  # START + 3 TOKEN events + COMPLETION
        assert events[0].event_type == TokenEventType.START

        # When token_id is exposed, events must be true TOKEN events
        assert events[1].event_type == TokenEventType.TOKEN
        assert events[1].token_index == 0
        assert events[1].is_first_token is True
        assert events[1].token_id == 1001

        assert events[2].event_type == TokenEventType.TOKEN
        assert events[2].token_index == 1
        assert events[2].is_first_token is False
        assert events[2].token_id == 1002

        assert events[3].event_type == TokenEventType.TOKEN
        assert events[3].token_index == 2
        assert events[3].token_id == 1003

        assert events[4].event_type == TokenEventType.COMPLETION

        # In this verified token mode, true token ITL and TPOT are calculated!
        res = MetricCollector.collect_request(events, prompt_tokens=10)
        assert res.success is True
        assert res.output_tokens == 3
        assert res.token_count_source == TokenCountSource.TOKEN_EVENTS
        assert len(res.inter_token_latencies_ms) == 2
        assert res.itl_p50_ms is not None
        assert res.tpot_ms is not None
        assert res.ttft_ms is not None
        assert res.ttfc_ms is None

        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_sse_chunk_boundary_splitting_across_network_packets(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            chunks = [
                b'data: {"content":"split ',
                b'across chunks","stop":false}\n\n',
                b"data: [DONE]\n\n",
            ]
            return httpx.Response(200, stream=ChunkedByteStream(chunks))

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        request = GenerationRequest(
            request_id="llamacpp-split-1",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_text="Testing splitting", output_tokens=10),
        )

        events = [e async for e in adapter.generate_stream(request)]
        assert len(events) == 3  # START, CHUNK, COMPLETION
        assert events[1].event_type == TokenEventType.CHUNK
        assert events[1].token_text == "split across chunks"
        assert events[2].event_type == TokenEventType.COMPLETION
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_prefill_only_zero_output_tokens(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            sse_lines = [
                b'data: {"content":"","stop":true,"timings":{"prompt_n":64,"prompt_ms":20.0,"predicted_n":0}}\n\n'
            ]
            return httpx.Response(200, stream=ChunkedByteStream(sse_lines))

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        request = GenerationRequest(
            request_id="llamacpp-prefill-0",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=64, output_tokens=0),
        )

        events = [e async for e in adapter.generate_stream(request)]
        assert len(events) == 2  # START + COMPLETION
        assert events[1].server_stats.output_tokens == 0

        res = MetricCollector.collect_request(events, prompt_tokens=64)
        assert res.success is True
        assert res.output_tokens == 0
        assert res.ttft_ms is None
        assert res.tpot_ms is None
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_stream_error_in_sse_payload(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            sse_lines = [
                b'data: {"content":"Partial","stop":false}\n\n',
                b'data: {"error":"Slot allocation limit reached"}\n\n',
            ]
            return httpx.Response(200, stream=ChunkedByteStream(sse_lines))

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        request = GenerationRequest(
            request_id="llamacpp-err-1",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=10, output_tokens=50),
        )

        events = [e async for e in adapter.generate_stream(request)]
        assert len(events) == 3  # START, CHUNK, ERROR
        assert events[2].event_type == TokenEventType.ERROR
        assert "Slot allocation limit" in events[2].error_message

        res = MetricCollector.collect_request(events, prompt_tokens=10)
        assert res.success is False
        assert "Slot allocation limit" in res.error_message
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_malformed_sse_json_emits_error(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            sse_lines = [b"data: {bad json\n\n"]
            return httpx.Response(200, stream=ChunkedByteStream(sse_lines))

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        request = GenerationRequest(
            request_id="llamacpp-bad-sse",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=10, output_tokens=50),
        )

        events = [e async for e in adapter.generate_stream(request)]
        assert len(events) == 2  # START, ERROR
        assert events[1].event_type == TokenEventType.ERROR
        assert "Malformed SSE JSON" in events[1].error_message
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_read_timeout_maps_to_engine_timeout(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Mocked read timeout on completion stream")

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport, timeout_seconds=0.05)
        request = GenerationRequest(
            request_id="llamacpp-timeout",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=10, output_tokens=50),
        )

        with pytest.raises(EngineTimeoutError, match="read timeout"):
            async for _ in adapter.generate_stream(request):
                pass
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_stream_cancellation_handling(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield b'data: {"content":"slow1","stop":false}\n\n'
                await asyncio.sleep(10.0)
                yield b'data: {"content":"slow2","stop":true}\n\n'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=SlowStream())

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        request = GenerationRequest(
            request_id="llamacpp-cancel",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=10, output_tokens=50),
        )

        async def consume() -> None:
            async for _ in adapter.generate_stream(request):
                await asyncio.sleep(0.01)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.02)
        task.cancel()

        with pytest.raises((StreamCancelledError, asyncio.CancelledError)):
            await task

        await adapter.teardown()


class TestLlamaCppBenchmarkRunnerIntegration:
    """Tests executing end-to-end benchmarks using BenchmarkRunner with LlamaCppAdapter."""

    @pytest.mark.asyncio
    async def test_llamacpp_benchmark_runner_suite_execution(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            if request.url.path == "/props":
                return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 4096}})
            if request.url.path == "/completion":
                lines = [
                    b'data: {"content":"llama ","id_slot":0,"stop":false}\n\n',
                    b'data: {"content":"token","id_slot":0,"stop":false}\n\n',
                    b'data: {"content":"","id_slot":0,"stop":true,"timings":{"prompt_n":16,"prompt_ms":10.0,"predicted_n":2,"predicted_ms":30.0}}\n\n',
                ]
                return httpx.Response(200, stream=ChunkedByteStream(lines))
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = LlamaCppAdapter(transport=transport)
        await adapter.initialize(default_model_spec, default_engine_config)

        runner = BenchmarkRunner(
            adapter=adapter,
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            runner_config=BenchmarkRunnerConfig(
                concurrency=2,
                warmup_requests=1,
                request_timeout_seconds=5.0,
            ),
        )

        workloads = [
            WorkloadSpec(prompt_tokens=16, output_tokens=8),
            WorkloadSpec(prompt_tokens=16, output_tokens=8),
        ]

        result = await runner.run(workloads)

        assert result.status == "COMPLETED"
        assert result.engine_type == EngineType.LLAMACPP
        assert result.execution_mode == ExecutionMode.REAL
        assert result.clock_source == ClockSource.MONOTONIC_REAL
        assert result.successful_requests == 2
        assert len(result.warmup_results) == 1
        assert len(result.request_results) == 2

        req_res = result.request_results[0]
        assert req_res.output_tokens == 2
        assert req_res.token_count_source == TokenCountSource.SERVER_REPORTED
        assert req_res.chunk_count == 2
        assert req_res.server_stats is not None
        assert req_res.server_stats.eval_duration_ms == 30.0
        assert req_res.tpot_ms is None  # Not fabricated from chunks!

        await adapter.teardown()
