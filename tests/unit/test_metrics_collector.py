"""Unit tests for metric collector, percentile interpolation, and workload generation."""

import pytest

from llm_ii_engine.benchmark.collector import MetricCollector, calculate_percentile
from llm_ii_engine.benchmark.models import RequestBenchmarkResult
from llm_ii_engine.benchmark.workload import (
    SCENARIO_PREFILL_ONLY,
    SCENARIO_VARIABLE_GAUSSIAN,
    WorkloadScenarioConfig,
    generate_workload,
)
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    StreamTerminationReason,
    TokenEventType,
)
from llm_ii_engine.core.schemas import TokenStreamEvent


class TestPercentileCalculation:
    """Tests for pure Python percentile interpolation against known hand-calculated samples."""

    def test_hand_calculated_percentiles(self) -> None:
        # Sample: [10.0, 20.0, 30.0, 40.0, 50.0] (n = 5)
        # q = (p / 100) * 4
        # p50: q = 2.0 -> 30.0
        # p90: q = 3.6 -> 40.0 + 0.6 * 10.0 = 46.0
        # p95: q = 3.8 -> 40.0 + 0.8 * 10.0 = 48.0
        # p99: q = 3.96 -> 40.0 + 0.96 * 10.0 = 49.6
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]

        assert calculate_percentile(vals, 0.0) == 10.0
        assert calculate_percentile(vals, 50.0) == 30.0
        assert calculate_percentile(vals, 90.0) == 46.0
        assert calculate_percentile(vals, 95.0) == 48.0
        assert calculate_percentile(vals, 99.0) == 49.6
        assert calculate_percentile(vals, 100.0) == 50.0

    def test_single_element_and_empty_edge_cases(self) -> None:
        assert calculate_percentile([], 50.0) is None
        assert calculate_percentile([42.5], 50.0) == 42.5
        assert calculate_percentile([42.5], 99.0) == 42.5

    def test_invalid_percentile_bounds_rejected(self) -> None:
        with pytest.raises(ValueError):
            calculate_percentile([1.0, 2.0], -0.1)
        with pytest.raises(ValueError):
            calculate_percentile([1.0, 2.0], 100.1)


class TestRequestMetricCollection:
    """Tests for single request metric collection with hand-calculated timing values."""

    def test_hand_calculated_request_metrics(self) -> None:
        # Request with 4 output tokens:
        # t0 = 100_000_000 ns (START)
        # t1 = 150_000_000 ns (TOKEN 0) -> TTFT = 50ms
        # t2 = 170_000_000 ns (TOKEN 1) -> ITL1 = 20ms
        # t3 = 180_000_000 ns (TOKEN 2) -> ITL2 = 10ms
        # t4 = 210_000_000 ns (TOKEN 3) -> ITL3 = 30ms
        # t5 = 211_000_000 ns (COMPLETION) -> Duration = 111ms
        events = [
            TokenStreamEvent(
                request_id="req-1",
                sequence_index=0,
                event_type=TokenEventType.START,
                timestamp_ns=100_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            ),
            TokenStreamEvent(
                request_id="req-1",
                sequence_index=1,
                event_type=TokenEventType.TOKEN,
                token_index=0,
                token_text="The",
                timestamp_ns=150_000_000,
                delta_ns=50_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=True,
            ),
            TokenStreamEvent(
                request_id="req-1",
                sequence_index=2,
                event_type=TokenEventType.TOKEN,
                token_index=1,
                token_text=" quick",
                timestamp_ns=170_000_000,
                delta_ns=20_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            ),
            TokenStreamEvent(
                request_id="req-1",
                sequence_index=3,
                event_type=TokenEventType.TOKEN,
                token_index=2,
                token_text=" brown",
                timestamp_ns=180_000_000,
                delta_ns=10_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            ),
            TokenStreamEvent(
                request_id="req-1",
                sequence_index=4,
                event_type=TokenEventType.TOKEN,
                token_index=3,
                token_text=" fox",
                timestamp_ns=210_000_000,
                delta_ns=30_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            ),
            TokenStreamEvent(
                request_id="req-1",
                sequence_index=5,
                event_type=TokenEventType.COMPLETION,
                token_index=4,
                timestamp_ns=211_000_000,
                delta_ns=1_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_terminal=True,
                termination_reason=StreamTerminationReason.FINISHED,
            ),
        ]

        result = MetricCollector.collect_request(events, prompt_tokens=10)

        assert result.success is True
        assert result.output_tokens == 4
        assert result.prompt_tokens == 10
        assert result.ttft_ms == 50.0  # 150ms - 100ms
        assert result.inter_token_latencies_ms == [20.0, 10.0, 30.0]
        # TPOT = (210ms - 150ms) / 3 = 60ms / 3 = 20.0ms
        assert result.tpot_ms == 20.0
        assert result.duration_ms == 111.0  # 211ms - 100ms
        # Median of [10.0, 20.0, 30.0] is 20.0
        assert result.itl_p50_ms == 20.0
        assert result.itl_p90_ms == 28.0  # 20 + 0.8 * 10
        # Output throughput = 4 / (111ms / 1000) = 36.04 tok/s
        assert result.output_tokens_per_second == 36.04

    def test_zero_token_output_metrics(self) -> None:
        # Prefill-only workload where output_tokens == 0
        events = [
            TokenStreamEvent(
                request_id="req-0",
                sequence_index=0,
                event_type=TokenEventType.START,
                timestamp_ns=100_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            ),
            TokenStreamEvent(
                request_id="req-0",
                sequence_index=1,
                event_type=TokenEventType.COMPLETION,
                timestamp_ns=150_000_000,
                delta_ns=50_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_terminal=True,
                termination_reason=StreamTerminationReason.MAX_TOKENS,
            ),
        ]

        result = MetricCollector.collect_request(events, prompt_tokens=500)
        assert result.success is True
        assert result.output_tokens == 0
        # TTFT and TPOT must be explicitly None, NOT 0.0
        assert result.ttft_ms is None
        assert result.tpot_ms is None
        assert result.inter_token_latencies_ms == []
        assert result.itl_p50_ms is None
        assert result.output_tokens_per_second == 0.0
        assert result.duration_ms == 50.0

    def test_single_token_output_metrics(self) -> None:
        # Output tokens == 1: TTFT is defined, but TPOT and ITL must be None
        events = [
            TokenStreamEvent(
                request_id="req-single",
                sequence_index=0,
                event_type=TokenEventType.START,
                timestamp_ns=100_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            ),
            TokenStreamEvent(
                request_id="req-single",
                sequence_index=1,
                event_type=TokenEventType.TOKEN,
                token_index=0,
                token_text="Hello",
                timestamp_ns=145_000_000,
                delta_ns=45_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=True,
            ),
            TokenStreamEvent(
                request_id="req-single",
                sequence_index=2,
                event_type=TokenEventType.COMPLETION,
                timestamp_ns=146_000_000,
                delta_ns=1_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_terminal=True,
                termination_reason=StreamTerminationReason.FINISHED,
            ),
        ]

        result = MetricCollector.collect_request(events, prompt_tokens=50)
        assert result.success is True
        assert result.output_tokens == 1
        assert result.ttft_ms == 45.0
        # TPOT is undefined (None) because there are no decode steps after the first token
        assert result.tpot_ms is None
        assert result.inter_token_latencies_ms == []
        assert result.itl_p50_ms is None

    def test_non_monotonic_timestamps_flagged(self) -> None:
        events = [
            TokenStreamEvent(
                request_id="req-bad",
                sequence_index=0,
                event_type=TokenEventType.START,
                timestamp_ns=200_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            ),
            TokenStreamEvent(
                request_id="req-bad",
                sequence_index=1,
                event_type=TokenEventType.TOKEN,
                token_index=0,
                token_text="A",
                timestamp_ns=100_000_000,  # Retrograde clock!
                delta_ns=0,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=True,
            ),
            TokenStreamEvent(
                request_id="req-bad",
                sequence_index=2,
                event_type=TokenEventType.COMPLETION,
                timestamp_ns=300_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_terminal=True,
                termination_reason=StreamTerminationReason.FINISHED,
            ),
        ]

        result = MetricCollector.collect_request(events, prompt_tokens=10)
        assert result.success is False
        assert "Non-monotonic timestamps" in (result.incomplete_reason or "")

    def test_missing_terminal_event_flagged(self) -> None:
        events = [
            TokenStreamEvent(
                request_id="req-incomplete",
                sequence_index=0,
                event_type=TokenEventType.START,
                timestamp_ns=100_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            ),
            TokenStreamEvent(
                request_id="req-incomplete",
                sequence_index=1,
                event_type=TokenEventType.TOKEN,
                token_index=0,
                token_text="A",
                timestamp_ns=150_000_000,
                delta_ns=50_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_first_token=True,
            ),
            # Missing COMPLETION or ERROR event
        ]

        result = MetricCollector.collect_request(events, prompt_tokens=10)
        assert result.success is False
        assert "without a terminal COMPLETION or ERROR" in (result.incomplete_reason or "")

    def test_error_event_propagation(self) -> None:
        events = [
            TokenStreamEvent(
                request_id="req-err",
                sequence_index=0,
                event_type=TokenEventType.START,
                timestamp_ns=100_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
            ),
            TokenStreamEvent(
                request_id="req-err",
                sequence_index=1,
                event_type=TokenEventType.ERROR,
                timestamp_ns=130_000_000,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                is_terminal=True,
                termination_reason=StreamTerminationReason.ERROR,
                error_message="KV cache memory allocation failure",
            ),
        ]

        result = MetricCollector.collect_request(events, prompt_tokens=10)
        assert result.success is False
        assert result.error_message == "KV cache memory allocation failure"
        assert result.incomplete_reason == "Terminated with error event"


class TestSuiteMetricAggregation:
    """Tests for aggregate suite metric calculations and warm-up isolation."""

    def test_warmup_results_strictly_excluded_from_aggregates(self) -> None:
        # Create 1 warm-up request with high TTFT (200ms)
        warmup_req = RequestBenchmarkResult(
            request_id="warmup-0",
            success=True,
            prompt_tokens=100,
            output_tokens=10,
            ttft_ms=200.0,
            tpot_ms=50.0,
            duration_ms=650.0,
            clock_source=ClockSource.SYNTHETIC_VIRTUAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
            is_warmup=True,
            inter_token_latencies_ms=[50.0] * 9,
        )

        # Create 2 measured requests with lower TTFT (50ms and 60ms)
        measured_req1 = RequestBenchmarkResult(
            request_id="req-0",
            success=True,
            prompt_tokens=100,
            output_tokens=5,
            ttft_ms=50.0,
            tpot_ms=10.0,
            duration_ms=90.0,
            clock_source=ClockSource.SYNTHETIC_VIRTUAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
            is_warmup=False,
            inter_token_latencies_ms=[10.0] * 4,
        )
        measured_req2 = RequestBenchmarkResult(
            request_id="req-1",
            success=True,
            prompt_tokens=100,
            output_tokens=5,
            ttft_ms=60.0,
            tpot_ms=12.0,
            duration_ms=98.0,
            clock_source=ClockSource.SYNTHETIC_VIRTUAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
            is_warmup=False,
            inter_token_latencies_ms=[12.0] * 4,
        )

        suite = MetricCollector.aggregate_suite(
            suite_id="suite-test",
            model_id="test-model",
            engine_type=EngineType.MOCK,
            concurrency=1,
            wall_time_seconds=0.188,
            request_results=[measured_req1, measured_req2],
            warmup_results=[warmup_req],
        )

        assert suite.warmup_requests == 1
        assert suite.total_requests == 2
        assert suite.successful_requests == 2
        # Verify warm-up TTFT (200ms) was EXCLUDED:
        # Median of [50.0, 60.0] = 55.0 ms
        assert suite.ttft_p50_ms == 55.0
        # Mean TPOT of [10.0, 12.0] = 11.0 ms
        assert suite.tpot_mean_ms == 11.0


class TestWorkloadGeneration:
    """Tests for deterministic workload generation and scenario presets."""

    def test_scenario_reproducibility(self) -> None:
        wl1 = generate_workload(SCENARIO_VARIABLE_GAUSSIAN)
        wl2 = generate_workload(SCENARIO_VARIABLE_GAUSSIAN)

        assert len(wl1) == len(wl2) == 25
        for w1, w2 in zip(wl1, wl2):
            assert w1.prompt_tokens == w2.prompt_tokens
            assert w1.output_tokens == w2.output_tokens
            assert w1.seed == w2.seed

    def test_scenario_clipping_bounds(self) -> None:
        wl = generate_workload(SCENARIO_VARIABLE_GAUSSIAN)
        for w in wl:
            assert 256 <= w.prompt_tokens <= 1024
            assert 32 <= w.output_tokens <= 256

    def test_prefill_only_scenario(self) -> None:
        wl = generate_workload(SCENARIO_PREFILL_ONLY)
        assert len(wl) == 10
        for w in wl:
            assert w.prompt_tokens == 2048
            assert w.output_tokens == 0

    def test_invalid_workload_config_rejected(self) -> None:
        with pytest.raises(ValueError):
            WorkloadScenarioConfig(
                name="bad",
                num_requests=10,
                prompt_tokens_mean=500,
                prompt_tokens_min=1000,
                prompt_tokens_max=200,  # min > max
            )
