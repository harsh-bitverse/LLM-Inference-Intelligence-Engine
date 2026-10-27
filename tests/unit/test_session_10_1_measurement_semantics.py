"""Unit tests verifying measurement semantics, count provenance, and downstream contracts (Session 10.1).

Covers:
1. Multi-token text chunks.
2. Token arrays without individual timing guarantees.
3. Verified one-token-per-event streams.
4. Missing server token counts.
5. Server-reported token counts with chunk-only streaming.
6. Heuristic prompt word counts.
7. Prediction dataset target eligibility.
8. Optimization objectives with unavailable TTFT or ITL.
9. Diagnostic evidence classifications.
10. Persistence and API round-trips across schema versions.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from llm_ii_engine.adapters.llamacpp import LlamaCppAdapter
from llm_ii_engine.adapters.ollama import OllamaAdapter
from llm_ii_engine.analysis.diagnostics import DiagnosticEngine
from llm_ii_engine.benchmark.collector import MetricCollector
from llm_ii_engine.benchmark.models import (
    BenchmarkSuiteResult,
    RequestBenchmarkResult,
)
from llm_ii_engine.benchmark.runner import BenchmarkRunner, BenchmarkRunnerConfig
from llm_ii_engine.core.enums import (
    ClockSource,
    DiagnosticCategory,
    DiagnosticSeverity,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    QuantizationType,
    StreamTerminationReason,
    TokenCountSource,
    TokenEventType,
)
from llm_ii_engine.core.schemas import (
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    ServerReportedStats,
    TokenStreamEvent,
    WorkloadSpec,
)
from llm_ii_engine.optimizer.engine import ConfigurationOptimizer
from llm_ii_engine.optimizer.models import (
    CandidateConfig,
    EvidenceStatus,
    FeasibilityStatus,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    SearchSpace,
)
from llm_ii_engine.prediction.dataset import DatasetBuilder
from llm_ii_engine.prediction.models import PredictionRecord, PredictionTarget
from llm_ii_engine.storage.database import DatabaseManager
from llm_ii_engine.storage.models import (
    BenchmarkSuiteORM,
    RequestBenchmarkORM,
)
from llm_ii_engine.storage.repository import EngineRepository


class ChunkedByteStream(httpx.AsyncByteStream):
    """Deterministic async byte stream yielding discrete byte chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.fixture
def default_model_spec() -> ModelSpec:
    return ModelSpec(
        model_id="meta-llama/Llama-3-8B",
        parameter_count_billion=8.0,
        context_window=4096,
        quantization=QuantizationType.NONE,
    )


# ============================================================================
# 1. Multi-Token Text Chunks & 2. Token Arrays Without Individual Timing Guarantees
# ============================================================================


@pytest.mark.asyncio
async def test_llamacpp_multi_token_array_emits_chunks_not_tokens(
    default_model_spec: ModelSpec,
) -> None:
    """When a llama.cpp SSE event yields a multi-token array (e.g. speculative decode batch),

    it must be emitted as a CHUNK event with TTFC, NOT as a single TOKEN event with false ITL.
    """
    sse_lines = [
        b'data: {"content":"Batch1 ","tokens":[101, 102, 103],"stop":false}\n\n',
        b'data: {"content":"Batch2 ","tokens":[104, 105],"stop":false}\n\n',
        b'data: {"content":"","stop":true,"timings":{"prompt_n":10,"predicted_n":5}}\n\n',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedByteStream(sse_lines))

    transport = httpx.MockTransport(handler)
    adapter = LlamaCppAdapter(transport=transport)
    req = GenerationRequest(
        request_id="multi-tok-1",
        model_spec=default_model_spec,
        engine_config=EngineConfig(
            engine_type=EngineType.LLAMACPP,
            extra_args={"skip_model_check": True, "verified_token_stream": False},
        ),
        workload=WorkloadSpec(prompt_tokens=10, output_tokens=5),
    )

    events = [e async for e in adapter.generate_stream(req)]
    assert len(events) == 4  # START, CHUNK, CHUNK, COMPLETION
    assert events[1].event_type == TokenEventType.CHUNK
    assert events[2].event_type == TokenEventType.CHUNK

    res = MetricCollector.collect_request(events, prompt_tokens=10)
    assert res.success is True
    assert res.ttfc_ms is not None
    assert res.ttft_ms is None  # Never fabricated from multi-token chunks
    assert res.inter_token_latencies_ms == []  # No fake 1-token ITL deltas
    assert res.itl_p50_ms is None
    assert res.chunk_count == 2
    assert len(res.chunk_latencies_ms) == 1
    # Server reported count preserved
    assert res.output_tokens == 5
    assert res.token_count_source == TokenCountSource.SERVER_REPORTED

    await adapter.teardown()


@pytest.mark.asyncio
async def test_llamacpp_token_array_without_verified_timing_guarantee_emits_chunk(
    default_model_spec: ModelSpec,
) -> None:
    """Even if an SSE event contains a single token ID or array, standard HTTP streaming

    lacks an individual timing contract and must be treated as CHUNK.
    """
    sse_lines = [
        b'data: {"content":"Word","token_id":999,"stop":false}\n\n',
        b'data: {"content":"","stop":true,"timings":{"prompt_n":5,"predicted_n":1}}\n\n',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedByteStream(sse_lines))

    transport = httpx.MockTransport(handler)
    adapter = LlamaCppAdapter(transport=transport)
    # Default config without verified_token_stream=True
    req = GenerationRequest(
        request_id="unverified-tok-1",
        model_spec=default_model_spec,
        engine_config=EngineConfig(
            engine_type=EngineType.LLAMACPP, extra_args={"skip_model_check": True}
        ),
        workload=WorkloadSpec(prompt_tokens=5, output_tokens=1),
    )

    events = [e async for e in adapter.generate_stream(req)]
    assert events[1].event_type == TokenEventType.CHUNK
    assert events[1].token_index is None

    res = MetricCollector.collect_request(events, prompt_tokens=5)
    assert res.ttfc_ms is not None
    assert res.ttft_ms is None

    await adapter.teardown()


# ============================================================================
# 3. Verified One-Token-Per-Event Streams
# ============================================================================


@pytest.mark.asyncio
async def test_verified_one_token_per_event_stream(default_model_spec: ModelSpec) -> None:
    """When verified_token_stream=True and discrete token IDs are supplied one per event,

    the adapter emits verified TOKEN events, exposing true TTFT and true ITL series.
    """
    sse_lines = [
        b'data: {"content":"Tok1","token_id":1001,"stop":false}\n\n',
        b'data: {"content":"Tok2","token_id":1002,"stop":false}\n\n',
        b'data: {"content":"Tok3","token_id":1003,"stop":true,"timings":{"prompt_n":8,"predicted_n":3}}\n\n',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedByteStream(sse_lines))

    transport = httpx.MockTransport(handler)
    adapter = LlamaCppAdapter(transport=transport)
    req = GenerationRequest(
        request_id="verified-tok-1",
        model_spec=default_model_spec,
        engine_config=EngineConfig(
            engine_type=EngineType.LLAMACPP,
            extra_args={"skip_model_check": True, "verified_token_stream": True},
        ),
        workload=WorkloadSpec(prompt_tokens=8, output_tokens=3),
    )

    events = [e async for e in adapter.generate_stream(req)]
    assert len(events) == 5  # START, 3 TOKENs, COMPLETION
    assert events[1].event_type == TokenEventType.TOKEN
    assert events[1].is_first_token is True
    assert events[2].event_type == TokenEventType.TOKEN
    assert events[3].event_type == TokenEventType.TOKEN

    res = MetricCollector.collect_request(events, prompt_tokens=8)
    assert res.ttft_ms is not None  # True token TTFT
    assert len(res.inter_token_latencies_ms) == 2  # True token ITLs
    assert res.itl_p50_ms is not None
    assert res.tpot_ms is not None
    assert res.token_count_source == TokenCountSource.TOKEN_EVENTS

    await adapter.teardown()


# ============================================================================
# 4. Missing Server Token Counts & 5. Server-Reported with Chunk-Only Streaming
# ============================================================================


def test_missing_server_token_counts_with_chunk_streaming() -> None:
    """When chunks arrive but server provides no token counts in completion event,

    output_tokens is 0 with UNKNOWN source, and throughput is None (never silently 0).
    """
    t0 = 1_000_000_000
    events = [
        TokenStreamEvent(
            request_id="r1",
            sequence_index=0,
            event_type=TokenEventType.START,
            timestamp_ns=t0,
            delta_ns=0,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
        ),
        TokenStreamEvent(
            request_id="r1",
            sequence_index=1,
            event_type=TokenEventType.CHUNK,
            token_text="Hello world",
            timestamp_ns=t0 + 50_000_000,
            delta_ns=50_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
        ),
        TokenStreamEvent(
            request_id="r1",
            sequence_index=2,
            event_type=TokenEventType.CHUNK,
            token_text=" from IIE",
            timestamp_ns=t0 + 100_000_000,
            delta_ns=50_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
        ),
        TokenStreamEvent(
            request_id="r1",
            sequence_index=3,
            event_type=TokenEventType.COMPLETION,
            timestamp_ns=t0 + 150_000_000,
            delta_ns=50_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
            is_terminal=True,
            termination_reason=StreamTerminationReason.FINISHED,
            server_stats=None,  # Server provided NO stats
        ),
    ]

    res = MetricCollector.collect_request(events, prompt_tokens=10)
    assert res.output_tokens == 0
    assert res.token_count_source == TokenCountSource.UNKNOWN
    assert res.chunk_count == 2
    assert res.output_tokens_per_second is None  # Unknown output count cannot produce throughput


def test_server_reported_token_counts_with_chunk_only_streaming() -> None:
    """When chunk streaming is accompanied by authoritative server completion stats,

    server counts produce throughput, but TTFT and ITL remain strictly None.
    """
    t0 = 1_000_000_000
    server_stats = ServerReportedStats(
        prompt_tokens=16,
        output_tokens=32,
        prompt_duration_ms=40.0,
        eval_duration_ms=160.0,
        eval_tokens_per_second=200.0,
    )
    events = [
        TokenStreamEvent(
            request_id="r2",
            sequence_index=0,
            event_type=TokenEventType.START,
            timestamp_ns=t0,
            delta_ns=0,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
        ),
        TokenStreamEvent(
            request_id="r2",
            sequence_index=1,
            event_type=TokenEventType.CHUNK,
            token_text="chunk 1",
            timestamp_ns=t0 + 50_000_000,
            delta_ns=50_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
        ),
        TokenStreamEvent(
            request_id="r2",
            sequence_index=2,
            event_type=TokenEventType.CHUNK,
            token_text="chunk 2",
            timestamp_ns=t0 + 200_000_000,
            delta_ns=150_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
        ),
        TokenStreamEvent(
            request_id="r2",
            sequence_index=3,
            event_type=TokenEventType.COMPLETION,
            timestamp_ns=t0 + 210_000_000,
            delta_ns=10_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
            is_terminal=True,
            termination_reason=StreamTerminationReason.FINISHED,
            server_stats=server_stats,
        ),
    ]

    res = MetricCollector.collect_request(events, prompt_tokens=16)
    assert res.output_tokens == 32
    assert res.token_count_source == TokenCountSource.SERVER_REPORTED
    assert res.prompt_token_count_source == TokenCountSource.SERVER_REPORTED
    assert res.ttfc_ms == 50.0  # Time to first chunk
    assert res.ttft_ms is None  # Token TTFT unmeasured
    assert res.inter_token_latencies_ms == []  # No fake ITLs
    assert res.output_tokens_per_second is not None  # Throughput from server tokens
    assert res.output_tokens_per_second > 0


# ============================================================================
# 6. Heuristic Prompt Word Counts
# ============================================================================


@pytest.mark.asyncio
async def test_heuristic_prompt_word_counts_and_unknown_provenance(
    default_model_spec: ModelSpec,
) -> None:
    """BenchmarkRunner assigns HEURISTIC_WORD_COUNT when only prompt_text is supplied

    and UNKNOWN (with prompt_tokens=None) when neither count nor text exists.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        lines = [b'{"response":"hi","done":true}\n']
        return httpx.Response(200, stream=ChunkedByteStream(lines))

    transport = httpx.MockTransport(handler)
    adapter = OllamaAdapter(transport=transport)
    runner = BenchmarkRunner(
        adapter=adapter,
        model_spec=default_model_spec,
        engine_config=EngineConfig(
            engine_type=EngineType.OLLAMA, extra_args={"skip_model_check": True}
        ),
        runner_config=BenchmarkRunnerConfig(concurrency=1, sample_memory=False),
    )

    # 1. Prompt text without explicit prompt_tokens -> HEURISTIC_WORD_COUNT
    wl_text = WorkloadSpec(prompt_text="The quick brown fox jumps", output_tokens=1)
    suite1 = await runner.run([wl_text])
    res1 = suite1.request_results[0]
    assert res1.prompt_tokens == 5
    assert res1.prompt_token_count_source == TokenCountSource.HEURISTIC_WORD_COUNT

    # 2. Empty prompt text without prompt_tokens -> UNKNOWN with prompt_tokens=None
    wl_empty = WorkloadSpec(prompt_text="", output_tokens=1)
    suite2 = await runner.run([wl_empty])
    res2 = suite2.request_results[0]
    assert res2.prompt_tokens is None
    assert res2.prompt_token_count_source == TokenCountSource.UNKNOWN

    await adapter.teardown()


# ============================================================================
# 7. Prediction Dataset Target Eligibility
# ============================================================================


def test_prediction_dataset_target_eligibility_enforces_provenance() -> None:
    """PredictionDatasetGenerator strictly rejects heuristic prompt tokens for TTFT

    and non-token events for TPOT, while allowing server-reported throughput.
    """
    suite = BenchmarkSuiteResult(
        suite_id="s1",
        status="COMPLETED",
        model_id="llama3",
        engine_type=EngineType.OLLAMA,
        concurrency=1,
        total_requests=1,
        warmup_requests=0,
        successful_requests=1,
        failed_requests=0,
        wall_time_seconds=1.0,
        execution_mode=ExecutionMode.REAL,
        clock_source=ClockSource.MONOTONIC_REAL,
    )

    # Case A: Heuristic prompt count with chunk streaming
    req_heuristic = RequestBenchmarkResult(
        request_id="req-h",
        success=True,
        prompt_tokens=20,
        prompt_token_count_source=TokenCountSource.HEURISTIC_WORD_COUNT,
        output_tokens=30,
        token_count_source=TokenCountSource.SERVER_REPORTED,
        ttfc_ms=45.0,
        ttft_ms=None,
        duration_ms=500.0,
        output_tokens_per_second=60.0,
        chunk_count=3,
        chunk_latencies_ms=[100.0, 100.0],
        chunk_latency_p50_ms=100.0,
        clock_source=ClockSource.MONOTONIC_REAL,
        execution_mode=ExecutionMode.REAL,
        provenance=ProvenanceType.MEASURED,
    )

    elig, reasons = DatasetBuilder.evaluate_record_eligibility(req_heuristic, suite)
    assert elig[PredictionTarget.TTFT_MS] is False
    assert "heuristic_word_count" in reasons[PredictionTarget.TTFT_MS][0]
    assert elig[PredictionTarget.TPOT_MS] is False  # Cannot train token TPOT on chunk streams
    assert elig[PredictionTarget.CHUNK_LATENCY_MS] is True  # Eligible for chunk models
    assert elig[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S] is True  # Server tokens allow throughput

    # Case B: Verified token events with requested prompt count
    req_verified = RequestBenchmarkResult(
        request_id="req-v",
        success=True,
        prompt_tokens=20,
        prompt_token_count_source=TokenCountSource.REQUESTED,
        output_tokens=30,
        token_count_source=TokenCountSource.TOKEN_EVENTS,
        ttft_ms=50.0,
        tpot_ms=15.0,
        duration_ms=500.0,
        output_tokens_per_second=60.0,
        clock_source=ClockSource.MONOTONIC_REAL,
        execution_mode=ExecutionMode.REAL,
        provenance=ProvenanceType.MEASURED,
    )

    elig_v, _ = DatasetBuilder.evaluate_record_eligibility(req_verified, suite)
    assert elig_v[PredictionTarget.TTFT_MS] is True
    assert elig_v[PredictionTarget.TPOT_MS] is True
    assert elig_v[PredictionTarget.OUTPUT_THROUGHPUT_TOK_S] is True


# ============================================================================
# 8. Optimization Objectives With Unavailable TTFT or ITL
# ============================================================================


def test_optimizer_never_substitutes_ttfc_for_ttft(default_model_spec: ModelSpec) -> None:
    """ConfigurationOptimizer leaves TTFT unpopulated when only TTFC is observed,

    resulting in UNKNOWN_FEASIBILITY for max_ttft_ms SLA rather than false feasibility.
    """
    # Create empirical observation that only has TTFC (chunk streaming)
    obs_chunk_only = PredictionRecord(
        record_id="rec-1",
        suite_id="s-1",
        request_id="obs-1",
        model_id=default_model_spec.model_id,
        engine_type=EngineType.OLLAMA,
        execution_mode=ExecutionMode.REAL,
        provenance=ProvenanceType.MEASURED,
        prompt_tokens=128,
        actual_output_tokens=128,
        output_tokens=128,
        ttfc_ms=35.0,  # Only TTFC observed
        ttft_ms=None,  # True TTFT unmeasured
        chunk_latency_ms=25.0,
        tpot_ms=None,  # TPOT unmeasured
        output_tokens_per_second=128.0,
        token_count_source=TokenCountSource.SERVER_REPORTED,
        prompt_token_count_source=TokenCountSource.REQUESTED,
    )

    opt = ConfigurationOptimizer(empirical_observations=[obs_chunk_only])
    cand_cfg = CandidateConfig(
        model_spec=default_model_spec,
        quantization=QuantizationType.FP16,
        engine_type=EngineType.OLLAMA,
        concurrency=1,
    )
    # SLA requires max_ttft_ms = 40.0
    req = OptimizationRequest(
        workload=WorkloadSpec(prompt_tokens=128, output_tokens=128),
        search_space=SearchSpace(
            models=[default_model_spec],
            quantizations=[QuantizationType.FP16],
            concurrencies=[1],
            engines=[EngineType.OLLAMA],
        ),
        constraints=OptimizationConstraints(max_ttft_ms=40.0),
        objectives=[OptimizationObjective.MINIMIZE_TTFT],
    )

    cand = opt._evaluate_candidate(
        config=cand_cfg,
        request=req,
        gpu_spec=None,
    )

    # Must NOT substitute TTFC=35ms into TTFT objective!
    assert (
        OptimizationObjective.MINIMIZE_TTFT not in cand.objective_values
        or cand.evidence_status.get(OptimizationObjective.MINIMIZE_TTFT)
        == EvidenceStatus.INSUFFICIENT_EVIDENCE
    )
    # SLA feasibility must be UNKNOWN, never falsely FEASIBLE
    assert cand.feasibility_status == FeasibilityStatus.UNKNOWN_FEASIBILITY
    assert any("max_ttft_ms" in r for r in cand.rejection_reasons)


# ============================================================================
# 9. Diagnostic Evidence Classifications
# ============================================================================


def test_diagnostic_engine_classifies_heuristic_prompts_and_chunks(
    default_model_spec: ModelSpec,
) -> None:
    """DiagnosticEngine produces TELEMETRY_GAP findings identifying heuristic prompt counts

    and chunk streaming mode, with explicit confidence and evidence rationale.
    """
    req = RequestBenchmarkResult(
        request_id="diag-1",
        success=True,
        prompt_tokens=10,
        prompt_token_count_source=TokenCountSource.HEURISTIC_WORD_COUNT,
        output_tokens=10,
        token_count_source=TokenCountSource.SERVER_REPORTED,
        ttfc_ms=50.0,
        ttft_ms=None,
        duration_ms=200.0,
        chunk_count=2,
        chunk_latencies_ms=[100.0],
        chunk_latency_p50_ms=100.0,
        clock_source=ClockSource.MONOTONIC_REAL,
        execution_mode=ExecutionMode.REAL,
        provenance=ProvenanceType.MEASURED,
    )

    suite = BenchmarkSuiteResult(
        suite_id="s-diag",
        status="COMPLETED",
        model_id=default_model_spec.model_id,
        engine_type=EngineType.OLLAMA,
        concurrency=1,
        total_requests=1,
        warmup_requests=0,
        successful_requests=1,
        failed_requests=0,
        wall_time_seconds=0.2,
        execution_mode=ExecutionMode.REAL,
        clock_source=ClockSource.MONOTONIC_REAL,
        request_results=[req],
    )

    report = DiagnosticEngine.analyze(
        benchmark_suite=suite,
        model_spec=default_model_spec,
    )

    findings = report.findings
    finding_titles = [f.title for f in findings]

    assert "Heuristic Prompt Word-Count Detected" in finding_titles
    assert "Text-Chunk Streaming Mode Detected" in finding_titles

    h_finding = next(f for f in findings if f.title == "Heuristic Prompt Word-Count Detected")
    assert h_finding.category == DiagnosticCategory.TELEMETRY_GAP
    assert h_finding.severity == DiagnosticSeverity.INFO
    assert "whitespace word splitting" in h_finding.message


# ============================================================================
# 10. Persistence and API Round-Trips Across Schema Versions
# ============================================================================


def test_persistence_round_trip_with_null_and_unknown_fields(tmp_path: Path) -> None:
    """Historical database records with NULL prompt_tokens or legacy source strings

    must round-trip through repository cleanly without raising exceptions.
    """
    db_file = tmp_path / "test_session_10_1.db"
    db_mgr = DatabaseManager(f"sqlite:///{db_file}")
    db_mgr.init_db()

    # 1. Directly insert a legacy ORM record simulating pre-Session-10 data
    with db_mgr.get_session() as session:
        suite_orm = BenchmarkSuiteORM(
            suite_id="legacy-suite-1",
            status="COMPLETED",
            model_id="meta-llama/Llama-3-8B",
            engine_type="ollama",
            execution_mode="REAL",
            clock_source="MONOTONIC_REAL",
            provenance="MEASURED",
            concurrency=1,
            total_requests=1,
            warmup_requests=0,
            successful_requests=1,
            failed_requests=0,
            wall_time_seconds=1.0,
            aggregate_tokens_per_second=10.0,
            ttfc_p50_ms=None,  # Legacy: null ttfc
        )
        session.add(suite_orm)

        req_orm = RequestBenchmarkORM(
            request_id="legacy-req-1",
            suite_id="legacy-suite-1",
            success=True,
            is_warmup=False,
            prompt_tokens=None,  # Legacy: null prompt tokens
            output_tokens=10,
            duration_ms=1000.0,
            token_count_source="unrecognized_legacy_source",  # Legacy non-standard string
            prompt_token_count_source=None,  # Legacy null
            clock_source="MONOTONIC_REAL",
            execution_mode="REAL",
            provenance="MEASURED",
            chunk_count=0,
        )
        session.add(req_orm)

    # 2. Retrieve via EngineRepository and verify graceful handling
    with db_mgr.get_session() as session:
        repo = EngineRepository(session)
        retrieved = repo.get_benchmark_suite("legacy-suite-1")
        assert retrieved is not None
        assert len(retrieved.request_results) == 1

        r = retrieved.request_results[0]
        assert r.prompt_tokens is None
        assert r.prompt_token_count_source == TokenCountSource.UNKNOWN
        assert r.token_count_source == TokenCountSource.UNKNOWN
        assert r.ttfc_ms is None
        assert r.ttft_ms is None

    db_mgr.dispose()
