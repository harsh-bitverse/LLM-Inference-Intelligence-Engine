"""Unit and offline integration tests for Ollama HTTP streaming adapter.

Uses httpx.MockTransport to test JSON-line streaming, chunk-boundary splitting,
error mapping, timeouts, cancellation, and metric collection without a live Ollama daemon.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from llm_ii_engine.adapters.ollama import OllamaAdapter
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
    EngineTimeoutError,
    ModelNotFoundError,
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
    return ModelSpec(model_id="llama3:latest", parameter_count_billion=8.0)


@pytest.fixture
def default_engine_config() -> EngineConfig:
    return EngineConfig(engine_type=EngineType.OLLAMA)


class TestOllamaAdapterProtocolAndLifecycle:
    """Tests for protocol conformance, capabilities, and lifecycle checks."""

    def test_capabilities(self) -> None:
        adapter = OllamaAdapter(base_url="http://localhost:11434")
        caps = adapter.get_capabilities()
        assert caps.engine_type == EngineType.OLLAMA
        assert caps.execution_mode == ExecutionMode.REAL
        assert caps.supports_streaming is True
        assert caps.supports_true_token_itl is False
        assert caps.supports_server_metrics is True

    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.1.32"})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        assert await adapter.health_check() is True
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_health_check_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        assert await adapter.health_check() is False
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_initialize_model_present(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.1.32"})
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "llama3:latest", "size": 4000000000},
                            {"name": "mistral:7b", "size": 4000000000},
                        ]
                    },
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        success = await adapter.initialize(default_model_spec, default_engine_config)
        assert success is True
        assert adapter.is_initialized is True
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_initialize_model_missing_rejects_without_pull(
        self, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.1.32"})
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "qwen2:7b"}]})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        missing_spec = ModelSpec(model_id="llama3:latest")

        with pytest.raises(ModelNotFoundError, match="is not installed on Ollama server"):
            await adapter.initialize(missing_spec, default_engine_config)

        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_initialize_skip_model_check_override(
        self, default_model_spec: ModelSpec
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.1.32"})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        config = EngineConfig(engine_type=EngineType.OLLAMA, extra_args={"skip_model_check": True})
        assert await adapter.initialize(default_model_spec, config) is True
        await adapter.teardown()


class TestOllamaAdapterStreamingAndMetrics:
    """Tests for streaming generation, text chunk parsing, and metric semantics."""

    @pytest.mark.asyncio
    async def test_standard_streaming_sequence_and_provenance(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.1.32"})
            assert request.url.path == "/api/generate"
            body = json.loads(request.content.decode("utf-8"))
            assert body["model"] == "llama3:latest"
            assert "prompt" in body
            assert body["stream"] is True

            lines = [
                b'{"model":"llama3:latest","response":"Fast","done":false}\n',
                b'{"model":"llama3:latest","response":" inference","done":false}\n',
                b'{"model":"llama3:latest","response":" engine","done":false}\n',
                b'{"model":"llama3:latest","response":"","done":true,"done_reason":"stop",'
                b'"prompt_eval_count":15,"eval_count":3,"prompt_eval_duration":45000000,'
                b'"eval_duration":90000000,"total_duration":135000000}\n',
            ]
            return httpx.Response(200, stream=ChunkedByteStream(lines))

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        await adapter.initialize(
            default_model_spec,
            EngineConfig(engine_type=EngineType.OLLAMA, extra_args={"skip_model_check": True}),
        )

        request = GenerationRequest(
            request_id="ollama-req-1",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=15, output_tokens=32),
        )

        events = [e async for e in adapter.generate_stream(request)]

        # Verify event sequence
        assert len(events) == 5  # START + 3 CHUNKs + COMPLETION
        assert events[0].event_type == TokenEventType.START
        assert events[0].clock_source == ClockSource.MONOTONIC_REAL
        assert events[0].execution_mode == ExecutionMode.REAL

        for i in (1, 2, 3):
            assert events[i].event_type == TokenEventType.CHUNK
            assert events[i].is_first_token is False
            assert events[i].token_text is not None
            assert events[i].clock_source == ClockSource.MONOTONIC_REAL

        assert events[1].token_text == "Fast"
        assert events[2].token_text == " inference"
        assert events[3].token_text == " engine"

        comp_event = events[4]
        assert comp_event.event_type == TokenEventType.COMPLETION
        assert comp_event.is_terminal is True
        assert comp_event.termination_reason == StreamTerminationReason.FINISHED
        assert comp_event.server_stats is not None
        assert comp_event.server_stats.output_tokens == 3
        assert comp_event.server_stats.prompt_tokens == 15
        assert comp_event.server_stats.eval_duration_ms == 90.0

        # Pass through MetricCollector and verify measurement semantics
        res = MetricCollector.collect_request(events, prompt_tokens=15)
        assert res.success is True
        assert res.output_tokens == 3
        assert res.token_count_source == TokenCountSource.SERVER_REPORTED
        assert res.prompt_tokens == 15
        assert res.prompt_token_count_source == TokenCountSource.SERVER_REPORTED
        # CRITICAL VERIFICATION: TTFC is captured, but true token-level TTFT, ITL and TPOT are None!
        assert res.ttfc_ms is not None
        assert res.ttfc_ms >= 0.0
        assert res.ttft_ms is None
        assert res.inter_token_latencies_ms == []
        assert res.itl_p50_ms is None
        assert res.tpot_ms is None

        # Verify chunk-level latencies ARE captured
        assert res.chunk_count == 3
        assert len(res.chunk_latencies_ms) == 2
        assert res.chunk_latency_p50_ms is not None

        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_chunk_boundary_splitting_across_network_packets(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # Split a single JSON line across two network packets
            chunks = [
                b'{"model":"llama3:latest","res',
                b'ponse":"split text","done":false}\n',
                b'{"model":"llama3:latest","response":"","done":true,"eval_count":1}\n',
            ]
            return httpx.Response(200, stream=ChunkedByteStream(chunks))

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        request = GenerationRequest(
            request_id="ollama-split-1",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_text="Hello world", output_tokens=10),
        )

        events = [e async for e in adapter.generate_stream(request)]
        assert len(events) == 3  # START, CHUNK, COMPLETION
        assert events[1].event_type == TokenEventType.CHUNK
        assert events[1].token_text == "split text"
        assert events[2].event_type == TokenEventType.COMPLETION
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_prefill_only_zero_output_tokens(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # Ollama prefill only returns immediate done with empty response
            lines = [
                b'{"model":"llama3:latest","response":"","done":true,"done_reason":"stop",'
                b'"prompt_eval_count":128,"eval_count":0,"prompt_eval_duration":50000000}\n'
            ]
            return httpx.Response(200, stream=ChunkedByteStream(lines))

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        request = GenerationRequest(
            request_id="ollama-prefill-0",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=128, output_tokens=0),
        )

        events = [e async for e in adapter.generate_stream(request)]
        assert len(events) == 2  # START + COMPLETION
        assert events[0].event_type == TokenEventType.START
        assert events[1].event_type == TokenEventType.COMPLETION
        assert events[1].server_stats.output_tokens == 0

        res = MetricCollector.collect_request(events, prompt_tokens=128)
        assert res.success is True
        assert res.output_tokens == 0
        assert res.ttft_ms is None
        assert res.tpot_ms is None
        assert res.chunk_count == 0
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_stream_error_in_json_payload(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            lines = [
                b'{"model":"llama3:latest","response":"initial","done":false}\n',
                b'{"error":"CUDA out of memory"}\n',
            ]
            return httpx.Response(200, stream=ChunkedByteStream(lines))

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        request = GenerationRequest(
            request_id="ollama-err-1",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=10, output_tokens=50),
        )

        events = [e async for e in adapter.generate_stream(request)]
        assert len(events) == 3  # START, CHUNK, ERROR
        assert events[2].event_type == TokenEventType.ERROR
        assert events[2].is_terminal is True
        assert "CUDA out of memory" in events[2].error_message

        res = MetricCollector.collect_request(events, prompt_tokens=10)
        assert res.success is False
        assert "CUDA out of memory" in res.error_message
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_malformed_json_emits_error_event(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            lines = [b"NOT VALID JSON\n"]
            return httpx.Response(200, stream=ChunkedByteStream(lines))

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        request = GenerationRequest(
            request_id="ollama-bad-json",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=10, output_tokens=50),
        )

        events = [e async for e in adapter.generate_stream(request)]
        assert len(events) == 2  # START, ERROR
        assert events[1].event_type == TokenEventType.ERROR
        assert "Malformed JSON" in events[1].error_message
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_http_404_model_not_found(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text='{"error":"model \'llama3\' not found"}')

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        request = GenerationRequest(
            request_id="ollama-404",
            model_spec=default_model_spec,
            engine_config=default_engine_config,
            workload=WorkloadSpec(prompt_tokens=10, output_tokens=50),
        )

        with pytest.raises(ModelNotFoundError, match="not found"):
            async for _ in adapter.generate_stream(request):
                pass
        await adapter.teardown()

    @pytest.mark.asyncio
    async def test_read_timeout_maps_to_engine_timeout(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Mocked read timeout during stream")

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport, timeout_seconds=0.05)
        request = GenerationRequest(
            request_id="ollama-timeout",
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
                yield b'{"response":"1","done":false}\n'
                await asyncio.sleep(10.0)
                yield b'{"response":"2","done":true}\n'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=SlowStream())

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
        request = GenerationRequest(
            request_id="ollama-cancel",
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


class TestOllamaBenchmarkRunnerIntegration:
    """Tests executing end-to-end benchmarks using BenchmarkRunner with OllamaAdapter."""

    @pytest.mark.asyncio
    async def test_ollama_benchmark_runner_suite_execution(
        self, default_model_spec: ModelSpec, default_engine_config: EngineConfig
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.1.32"})
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})
            if request.url.path == "/api/generate":
                lines = [
                    b'{"response":"Chunk 1","done":false}\n',
                    b'{"response":" Chunk 2","done":false}\n',
                    b'{"response":"","done":true,"prompt_eval_count":20,"eval_count":2,"eval_duration":40000000}\n',
                ]
                return httpx.Response(200, stream=ChunkedByteStream(lines))
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        adapter = OllamaAdapter(transport=transport)
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
            WorkloadSpec(prompt_tokens=20, output_tokens=10),
            WorkloadSpec(prompt_tokens=20, output_tokens=10),
        ]

        result = await runner.run(workloads)

        assert result.status == "COMPLETED"
        assert result.engine_type == EngineType.OLLAMA
        assert result.execution_mode == ExecutionMode.REAL
        assert result.clock_source == ClockSource.MONOTONIC_REAL
        assert result.successful_requests == 2
        assert len(result.warmup_results) == 1
        assert len(result.request_results) == 2

        # Verify each request captured server reported stats and chunk latencies
        req_res = result.request_results[0]
        assert req_res.output_tokens == 2
        assert req_res.token_count_source == TokenCountSource.SERVER_REPORTED
        assert req_res.chunk_count == 2
        assert req_res.server_stats is not None
        assert req_res.server_stats.eval_duration_ms == 40.0
        assert req_res.tpot_ms is None  # Not fabricated from chunks!

        await adapter.teardown()
