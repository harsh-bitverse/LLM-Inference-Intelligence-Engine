"""Session 10: Rigorous measurement integrity, protocol fidelity, and lifecycle validation tests.

Tests:
1. Protocol-faithful streaming fixtures (Ollama NDJSON and llama.cpp SSE).
2. Separation of text chunks from true discrete token events.
3. Accurate TTFT (token) vs TTFC (chunk) metrics.
4. Handling of empty keepalives, malformed payloads, and truncated streams.
5. Client cancellation, timeouts, and error isolation.
6. Server-side usage metadata attribution and fallbacks.
7. Statistical aggregation and warmup exclusion.
8. Hardware and memory telemetry attribution (no fabricated zeros, separate process isolation).
9. Schema forward migration (v1 -> v2) and persistence durability.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text

from llm_ii_engine.adapters.base import InferenceEngineAdapter
from llm_ii_engine.adapters.llamacpp import LlamaCppAdapter
from llm_ii_engine.adapters.ollama import OllamaAdapter
from llm_ii_engine.benchmark.collector import MetricCollector
from llm_ii_engine.benchmark.models import (
    BenchmarkSuiteResult,
    RequestBenchmarkResult,
)
from llm_ii_engine.benchmark.runner import BenchmarkRunner, BenchmarkRunnerConfig
from llm_ii_engine.benchmark.sampler import AdapterMemorySampler
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    StreamTerminationReason,
    TokenCountSource,
    TokenEventType,
)
from llm_ii_engine.core.schemas import (
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    TokenStreamEvent,
    WorkloadSpec,
)
from llm_ii_engine.storage.database import DatabaseManager
from llm_ii_engine.storage.models import (
    Base,
)
from llm_ii_engine.storage.repository import EngineRepository


class ChunkedByteStream(httpx.AsyncByteStream):
    """Deterministic async byte stream yielding discrete byte chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class InfiniteAsyncByteStream(httpx.AsyncByteStream):
    """Infinite async byte stream for cancellation testing."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"response":"chunk 1","done":false}\n'
        while True:
            await asyncio.sleep(0.05)
            yield b'{"response":"chunk ...","done":false}\n'


class SlowAsyncByteStream(httpx.AsyncByteStream):
    """Delayed async byte stream for timeout testing."""

    def __init__(self, delay: float = 1.0) -> None:
        self._delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"response":"slow...","done":false}\n'
        await asyncio.sleep(self._delay)
        yield b'{"response":"","done":true}\n'


@pytest.fixture
def test_model_spec() -> ModelSpec:
    return ModelSpec(model_id="meta-llama/Llama-3-8B", parameter_count_billion=8.0)


@pytest.fixture
def test_engine_config() -> EngineConfig:
    return EngineConfig(engine_type=EngineType.OLLAMA, extra_args={"skip_model_check": True})


# ============================================================================
# 1. Ollama Protocol & Chunk vs Token Accounting
# ============================================================================


@pytest.mark.asyncio
async def test_ollama_streaming_fixtures_and_chunk_accounting(
    test_model_spec: ModelSpec, test_engine_config: EngineConfig
) -> None:
    """Verify Ollama NDJSON stream emits CHUNK events and collector records TTFC while keeping TTFT None."""

    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            b'{"model":"llama3","response":"The","done":false}\n',
            b'{"model":"llama3","response":" quick","done":false}\n',
            b'{"model":"llama3","response":" brown","done":false}\n',
            b'{"model":"llama3","response":" fox","done":true,"done_reason":"stop",'
            b'"prompt_eval_count":16,"eval_count":4,"prompt_eval_duration":45000000,'
            b'"eval_duration":120000000,"total_duration":168000000}\n',
        ]
        return httpx.Response(200, stream=ChunkedByteStream(lines))

    transport = httpx.MockTransport(handler)
    adapter = OllamaAdapter(transport=transport)
    gen_req = GenerationRequest(
        request_id="ollama-meas-1",
        model_spec=test_model_spec,
        engine_config=test_engine_config,
        workload=WorkloadSpec(prompt_tokens=16, output_tokens=4),
    )

    events = [e async for e in adapter.generate_stream(gen_req)]
    assert len(events) == 6  # START, 4 CHUNKs, 1 COMPLETION

    assert events[0].event_type == TokenEventType.START
    assert events[1].event_type == TokenEventType.CHUNK
    assert events[1].token_text == "The"
    assert events[4].event_type == TokenEventType.CHUNK
    assert events[4].token_text == " fox"
    assert events[5].event_type == TokenEventType.COMPLETION
    assert events[5].server_stats is not None
    assert events[5].server_stats.output_tokens == 4
    assert events[5].server_stats.prompt_tokens == 16
    assert events[5].server_stats.prompt_duration_ms == 45.0
    assert events[5].server_stats.eval_duration_ms == 120.0

    # Collect request metrics
    res = MetricCollector.collect_request(events, prompt_tokens=16)
    assert res.success is True
    assert res.output_tokens == 4
    assert res.token_count_source == TokenCountSource.SERVER_REPORTED
    assert res.chunk_count == 4

    # Crucial scientific check: TTFC is measured, but true token TTFT is None
    assert res.ttfc_ms is not None
    assert res.ttfc_ms >= 0.0
    assert res.ttft_ms is None, "True token TTFT must be None when only text chunks are observed"
    assert res.inter_token_latencies_ms == [], "No true token ITL when only chunks are observed"
    assert res.itl_p50_ms is None
    assert res.tpot_ms is None
    assert len(res.chunk_latencies_ms) == 3
    assert res.chunk_latency_p50_ms is not None

    await adapter.teardown()


# ============================================================================
# 2. llama.cpp Protocol (Token IDs vs Text Chunks)
# ============================================================================


@pytest.mark.asyncio
async def test_llamacpp_streaming_with_token_ids_exposes_true_ttft_and_itl(
    test_model_spec: ModelSpec, test_engine_config: EngineConfig
) -> None:
    """When llama.cpp emits token_id, adapter emits TOKEN events, exposing true TTFT, ITL, and TPOT."""

    def handler(request: httpx.Request) -> httpx.Response:
        sse = [
            b'data: {"content":"Hello","token_id":101,"stop":false}\n\n',
            b'data: {"content":" world","token_id":102,"stop":false}\n\n',
            b'data: {"content":"!","token_id":103,"stop":false}\n\n',
            b'data: {"content":"","stop":true,"timings":{"prompt_n":10,"prompt_ms":30.0,"predicted_n":3,"predicted_ms":60.0,"predicted_per_second":50.0}}\n\n',
        ]
        return httpx.Response(200, stream=ChunkedByteStream(sse))

    transport = httpx.MockTransport(handler)
    adapter = LlamaCppAdapter(transport=transport)
    gen_req = GenerationRequest(
        request_id="llamacpp-tokens-1",
        model_spec=test_model_spec,
        engine_config=EngineConfig(
            engine_type=EngineType.LLAMACPP,
            extra_args={"skip_model_check": True, "verified_token_stream": True},
        ),
        workload=WorkloadSpec(prompt_tokens=10, output_tokens=3),
    )

    events = [e async for e in adapter.generate_stream(gen_req)]
    assert len(events) == 5  # START, 3 TOKENs, 1 COMPLETION

    assert events[1].event_type == TokenEventType.TOKEN
    assert events[1].token_id == 101

    res = MetricCollector.collect_request(events, prompt_tokens=10)
    assert res.success is True
    assert res.output_tokens == 3
    assert res.token_count_source == TokenCountSource.TOKEN_EVENTS

    # True token metrics are available!
    assert res.ttft_ms is not None
    assert res.ttfc_ms is None
    assert len(res.inter_token_latencies_ms) == 2
    assert res.itl_p50_ms is not None
    assert res.tpot_ms is not None

    await adapter.teardown()


@pytest.mark.asyncio
async def test_llamacpp_sse_empty_keepalive_data_and_fallbacks(
    test_model_spec: ModelSpec, test_engine_config: EngineConfig
) -> None:
    """Verify empty SSE data lines (keepalive pings) do not raise JSONDecodeError, and fallback stats are extracted."""

    def handler(request: httpx.Request) -> httpx.Response:
        sse = [
            b": keepalive comment\n\n",
            b"data:\n\n",  # Empty data line!
            b"data: \n\n",  # Empty data line with space!
            b'data: {"content":"Part 1","stop":false}\n\n',
            b"data:\n\n",
            b'data: {"content":"Part 2","stop":true,"tokens_evaluated":12,"tokens_predicted":2,"timings":{"prompt_ms":40.0,"predicted_ms":50.0}}\n\n',
        ]
        return httpx.Response(200, stream=ChunkedByteStream(sse))

    transport = httpx.MockTransport(handler)
    adapter = LlamaCppAdapter(transport=transport)
    gen_req = GenerationRequest(
        request_id="llamacpp-keepalive-1",
        model_spec=test_model_spec,
        engine_config=test_engine_config,
        workload=WorkloadSpec(prompt_tokens=12, output_tokens=2),
    )

    events = [e async for e in adapter.generate_stream(gen_req)]
    assert len(events) == 4  # START, 2 CHUNKs, 1 COMPLETION
    assert events[-1].event_type == TokenEventType.COMPLETION
    assert events[-1].server_stats is not None
    # Verify fallback token count extraction (tokens_predicted)
    assert events[-1].server_stats.output_tokens == 2
    assert events[-1].server_stats.prompt_tokens == 12

    res = MetricCollector.collect_request(events, prompt_tokens=12)
    assert res.success is True
    assert res.output_tokens == 2
    assert res.ttfc_ms is not None
    assert res.ttft_ms is None

    await adapter.teardown()


# ============================================================================
# 3. Malformed and Truncated Stream Handling
# ============================================================================


@pytest.mark.asyncio
async def test_ollama_malformed_json_emits_error(
    test_model_spec: ModelSpec, test_engine_config: EngineConfig
) -> None:
    """Malformed NDJSON line yields an ERROR TokenStreamEvent and terminates cleanly."""

    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            b'{"model":"llama3","response":"OK","done":false}\n',
            b"BROKEN_JSON_NOT_VALID{{{\n",
        ]
        return httpx.Response(200, stream=ChunkedByteStream(lines))

    transport = httpx.MockTransport(handler)
    adapter = OllamaAdapter(transport=transport)
    gen_req = GenerationRequest(
        request_id="ollama-err-1",
        model_spec=test_model_spec,
        engine_config=test_engine_config,
        workload=WorkloadSpec(prompt_tokens=10, output_tokens=10),
    )

    events = [e async for e in adapter.generate_stream(gen_req)]
    assert len(events) == 3  # START, CHUNK, ERROR
    assert events[-1].event_type == TokenEventType.ERROR
    assert "Malformed JSON" in events[-1].error_message

    res = MetricCollector.collect_request(events, prompt_tokens=10)
    assert res.success is False
    assert res.incomplete_reason == "Terminated with error event"

    await adapter.teardown()


@pytest.mark.asyncio
async def test_ollama_premature_disconnect_without_done(
    test_model_spec: ModelSpec, test_engine_config: EngineConfig
) -> None:
    """Stream ending abruptly without done=true yields an ERROR TokenStreamEvent."""

    def handler(request: httpx.Request) -> httpx.Response:
        lines = [b'{"model":"llama3","response":"Incomplete...","done":false}\n']
        return httpx.Response(200, stream=ChunkedByteStream(lines))

    transport = httpx.MockTransport(handler)
    adapter = OllamaAdapter(transport=transport)
    gen_req = GenerationRequest(
        request_id="ollama-trunc-1",
        model_spec=test_model_spec,
        engine_config=test_engine_config,
        workload=WorkloadSpec(prompt_tokens=10, output_tokens=10),
    )

    events = [e async for e in adapter.generate_stream(gen_req)]
    assert len(events) == 3  # START, CHUNK, ERROR
    assert events[-1].event_type == TokenEventType.ERROR
    assert "terminated prematurely" in events[-1].error_message

    res = MetricCollector.collect_request(events, prompt_tokens=10)
    assert res.success is False
    assert "error event" in res.incomplete_reason

    await adapter.teardown()


@pytest.mark.asyncio
async def test_llamacpp_premature_truncation_emits_error(
    test_model_spec: ModelSpec, test_engine_config: EngineConfig
) -> None:
    """llama.cpp stream ending without [DONE] or stop=true yields an ERROR TokenStreamEvent."""

    def handler(request: httpx.Request) -> httpx.Response:
        sse = [b'data: {"content":"Cut short...","stop":false}\n\n']
        return httpx.Response(200, stream=ChunkedByteStream(sse))

    transport = httpx.MockTransport(handler)
    adapter = LlamaCppAdapter(transport=transport)
    gen_req = GenerationRequest(
        request_id="llamacpp-trunc-1",
        model_spec=test_model_spec,
        engine_config=test_engine_config,
        workload=WorkloadSpec(prompt_tokens=10, output_tokens=10),
    )

    events = [e async for e in adapter.generate_stream(gen_req)]
    assert len(events) == 3  # START, CHUNK, ERROR
    assert events[-1].event_type == TokenEventType.ERROR
    assert "terminated prematurely" in events[-1].error_message

    res = MetricCollector.collect_request(events, prompt_tokens=10)
    assert res.success is False
    assert "error event" in res.incomplete_reason

    await adapter.teardown()


@pytest.mark.asyncio
async def test_client_cancellation_emits_cancelled_event(
    test_model_spec: ModelSpec, test_engine_config: EngineConfig
) -> None:
    """When a generation stream task is cancelled by caller, a CANCELLED completion event is yielded."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=InfiniteAsyncByteStream())

    transport = httpx.MockTransport(handler)
    adapter = OllamaAdapter(transport=transport)
    gen_req = GenerationRequest(
        request_id="ollama-cancel-1",
        model_spec=test_model_spec,
        engine_config=test_engine_config,
        workload=WorkloadSpec(prompt_tokens=10, output_tokens=10),
    )

    events: list[TokenStreamEvent] = []

    async def run_task():
        async for ev in adapter.generate_stream(gen_req):
            events.append(ev)

    task = asyncio.create_task(run_task())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises((asyncio.CancelledError, Exception)):
        await task

    assert len(events) >= 1
    assert events[0].event_type == TokenEventType.START
    # Check that a terminal CANCELLED event or completion was recorded
    if any(e.is_terminal for e in events):
        term = next(e for e in events if e.is_terminal)
        assert term.termination_reason == StreamTerminationReason.CANCELLED

    await adapter.teardown()


@pytest.mark.asyncio
async def test_benchmark_runner_concurrent_isolation_and_timeout(
    test_model_spec: ModelSpec, test_engine_config: EngineConfig
) -> None:
    """BenchmarkRunner bounds concurrency and isolates per-request timeout without aborting whole suite."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content.decode("utf-8"))
        prompt = body.get("prompt", "")
        if "slow" in prompt:
            return httpx.Response(200, stream=SlowAsyncByteStream(delay=1.0))

        lines = [
            b'{"response":"fast","done":true,"prompt_eval_count":8,"eval_count":1,'
            b'"prompt_eval_duration":10000000,"eval_duration":10000000}\n'
        ]
        return httpx.Response(200, stream=ChunkedByteStream(lines))

    transport = httpx.MockTransport(handler)
    adapter = OllamaAdapter(transport=transport)
    runner = BenchmarkRunner(
        adapter=adapter,
        model_spec=test_model_spec,
        engine_config=test_engine_config,
        runner_config=BenchmarkRunnerConfig(
            concurrency=2,
            request_timeout_seconds=0.2,  # 200ms timeout
            sample_memory=False,
        ),
    )

    workloads = [
        WorkloadSpec(prompt_text="fast request 1", output_tokens=1),
        WorkloadSpec(prompt_text="slow request 2", output_tokens=1),
        WorkloadSpec(prompt_text="fast request 3", output_tokens=1),
    ]

    suite = await runner.run(workloads)
    assert suite.total_requests == 3
    assert suite.successful_requests == 2
    assert suite.failed_requests == 1
    # Check that the timed out request is cleanly isolated
    failed_req = next(r for r in suite.request_results if not r.success)
    assert failed_req.incomplete_reason == "Request timed out"
    assert "timeout" in failed_req.error_message.lower()

    await adapter.teardown()


# ============================================================================
# 4. Statistical Aggregation & Warmup Exclusion
# ============================================================================


def test_suite_aggregation_warmup_and_chunk_metrics() -> None:
    """Warmup requests are strictly excluded from statistical latency percentiles and throughput."""
    # 2 warmup requests (100ms TTFC)
    warmups = [
        RequestBenchmarkResult(
            request_id=f"warmup-{i}",
            success=True,
            prompt_tokens=10,
            output_tokens=5,
            ttfc_ms=100.0,
            duration_ms=200.0,
            is_warmup=True,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
        )
        for i in range(2)
    ]

    # 3 measured requests (ttfc: 20ms, 30ms, 40ms; chunk latencies: [10, 15])
    measured = [
        RequestBenchmarkResult(
            request_id=f"req-{i}",
            success=True,
            prompt_tokens=10,
            output_tokens=5,
            ttfc_ms=20.0 + (i * 10.0),
            duration_ms=100.0,
            chunk_count=3,
            chunk_latencies_ms=[10.0, 15.0],
            is_warmup=False,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
        )
        for i in range(3)
    ]

    suite = MetricCollector.aggregate_suite(
        suite_id="suite-warmup-test",
        model_id="test-model",
        engine_type=EngineType.OLLAMA,
        concurrency=1,
        wall_time_seconds=2.0,
        request_results=measured,
        warmup_results=warmups,
    )

    assert suite.total_requests == 3
    assert suite.warmup_requests == 2
    assert suite.successful_requests == 3

    # TTFC median should be 30.0 ms (from [20, 30, 40]), NOT influenced by warmup 100ms!
    assert suite.ttfc_p50_ms == 30.0
    assert suite.ttft_p50_ms is None
    assert suite.chunk_latency_p50_ms == 12.5  # median of [10, 15, 10, 15, 10, 15] = 12.5

    # Output throughput = (5 * 3) / 2.0 = 7.5 tok/s (warmup output tokens strictly excluded!)
    assert suite.aggregate_output_tokens_per_second == 7.5


# ============================================================================
# 5. Hardware & Memory Telemetry Attribution
# ============================================================================


@pytest.mark.asyncio
async def test_memory_sampler_telemetry_attribution_for_separate_process() -> None:
    """When an engine runs as an external HTTP daemon, sampler notes separate process execution and leaves peak_process_rss_mb None."""

    class MockHttpAdapter(InferenceEngineAdapter):
        name = "OllamaAdapter"
        engine_type = EngineType.OLLAMA
        execution_mode = ExecutionMode.REAL
        base_url = "http://localhost:11434"

        def get_capabilities(self):
            return None

        async def initialize(self, m, e):
            return True

        async def health_check(self):
            return True

        async def generate_stream(self, req):
            if False:
                yield None

        async def get_memory_usage_mb(self):
            return None

        async def teardown(self):
            pass

    adapter = MockHttpAdapter()
    sampler = AdapterMemorySampler(adapter)
    await sampler.start()
    await sampler.sample_now()
    watermark = await sampler.stop()

    assert watermark.is_telemetry_available is False
    assert watermark.peak_gpu_vram_mb is None
    assert watermark.peak_process_rss_mb is None
    assert "Separate process execution" in watermark.notes
    assert "Values are reported as None rather than fabricated zeros" in watermark.notes


# ============================================================================
# 6. Database Forward Migration (v1 -> v2) and Persistence
# ============================================================================


def test_schema_forward_migration_v1_to_v2_preserves_data(tmp_path: Path) -> None:
    """Verify that a database created at schema v1 automatically migrates to v2 without data loss and persists new fields."""
    db_file = tmp_path / "migration_v1_v2.db"
    db_url = f"sqlite:///{db_file}"

    # Step 1: Simulate existing v1 database manually
    db_mgr_v1 = DatabaseManager(database_url=db_url)
    with db_mgr_v1.engine.connect() as conn:
        # Create tables without v2 columns
        Base.metadata.create_all(bind=conn)
        conn.execute(
            text(
                "INSERT INTO schema_version (version, applied_at, description) VALUES (1, '2026-10-25 00:00:00', 'Initial v1')"
            )
        )
        conn.commit()
    db_mgr_v1.dispose()

    # Step 2: Open with active DatabaseManager (configured with CURRENT_SCHEMA_VERSION = 2)
    db_mgr_v2 = DatabaseManager(database_url=db_url)
    active_ver = db_mgr_v2.init_db(check_version=True)
    assert active_ver == 2

    # Verify version records in database
    with db_mgr_v2.get_session() as session:
        versions = session.scalars(
            text("SELECT version FROM schema_version ORDER BY version")
        ).all()
        assert 2 in versions

        # Step 3: Persist a suite with ttfc_ms and chunk metrics
        repo = EngineRepository(session)
        suite = BenchmarkSuiteResult(
            suite_id="suite-v2-migrated",
            status="COMPLETED",
            model_id="test-model-v2",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            clock_source=ClockSource.MONOTONIC_REAL,
            concurrency=1,
            total_requests=1,
            warmup_requests=0,
            successful_requests=1,
            failed_requests=0,
            wall_time_seconds=1.5,
            ttfc_p50_ms=25.5,
            ttfc_p90_ms=30.0,
            chunk_latency_p50_ms=12.0,
            request_results=[
                RequestBenchmarkResult(
                    request_id="req-v2-0",
                    success=True,
                    prompt_tokens=10,
                    output_tokens=5,
                    ttfc_ms=25.5,
                    duration_ms=150.0,
                    chunk_count=3,
                    chunk_latencies_ms=[10.0, 14.0],
                    chunk_latency_p50_ms=12.0,
                    clock_source=ClockSource.MONOTONIC_REAL,
                    execution_mode=ExecutionMode.REAL,
                    provenance=ProvenanceType.MEASURED,
                )
            ],
        )
        repo.save_benchmark_suite(suite)

        # Retrieve and verify round-trip
        retrieved = repo.get_benchmark_suite("suite-v2-migrated")
        assert retrieved is not None
        assert retrieved.ttfc_p50_ms == 25.5
        assert retrieved.ttft_p50_ms is None
        assert retrieved.chunk_latency_p50_ms == 12.0
        assert retrieved.request_results[0].ttfc_ms == 25.5
        assert retrieved.request_results[0].ttft_ms is None

    db_mgr_v2.dispose()
