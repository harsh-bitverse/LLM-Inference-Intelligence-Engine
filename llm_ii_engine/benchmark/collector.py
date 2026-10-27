"""High-resolution metrics collection, statistical aggregation, and timing analysis.

Computes prefill TTFT, decode ITLs, TPOT, percentiles, and throughput metrics.
Strictly separates token decoding events from lifecycle markers (START/COMPLETION/ERROR)
and preserves execution-mode and clock-source boundaries.
"""

from collections.abc import Sequence

from llm_ii_engine.benchmark.models import (
    BenchmarkSuiteResult,
    MemoryWatermarkResult,
    RequestBenchmarkResult,
)
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    StreamTerminationReason,
    TokenCountSource,
    TokenEventType,
)
from llm_ii_engine.core.schemas import ServerReportedStats, TokenStreamEvent


def calculate_percentile(values: Sequence[float], percentile: float) -> float | None:
    """Calculate a percentile using standard linear interpolation (method='linear').

    Equivalent to numpy.percentile(values, p, method='linear') or Excel PERCENTILE.INC.

    Args:
        values: Sequence of numerical values.
        percentile: Desired percentile in [0.0, 100.0].

    Returns:
        Interpolated percentile value, or None if values is empty.
    """
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"Percentile must be between 0.0 and 100.0, got {percentile}")

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    # Calculate virtual index rank q in [0, n - 1]
    q = (percentile / 100.0) * (n - 1)
    k = int(q)
    f = q - k

    if k >= n - 1:
        return float(sorted_vals[-1])

    return round(float((1.0 - f) * sorted_vals[k] + f * sorted_vals[k + 1]), 4)


class MetricCollector:
    """Collects and aggregates timing measurements from token event streams."""

    @staticmethod
    def collect_request(
        events: list[TokenStreamEvent],
        prompt_tokens: int | None = None,
        prompt_token_count_source: TokenCountSource | None = None,
        is_warmup: bool = False,
        submission_time_ns: int | None = None,
        requested_output_tokens: int | None = None,
    ) -> RequestBenchmarkResult:
        """Process a stream of TokenStreamEvent instances into a RequestBenchmarkResult.

        Args:
            events: Ordered list of TokenStreamEvents emitted for a single request.
            prompt_tokens: Number of prompt tokens submitted for the request (None if unobserved).
            prompt_token_count_source: Provenance source for the prompt token count.
            is_warmup: Whether this request was executed as an unmeasured warm-up probe.
            submission_time_ns: Optional client submission timestamp if START event absent.
            requested_output_tokens: Optional requested output tokens limit from workload.

        Returns:
            Validated RequestBenchmarkResult with precise latencies and throughput.
        """
        if not events:
            return RequestBenchmarkResult(
                request_id="unknown",
                success=False,
                error_message="Empty event stream received",
                prompt_tokens=prompt_tokens,
                requested_output_tokens=requested_output_tokens,
                output_tokens=0,
                duration_ms=0.0,
                clock_source=ClockSource.SYNTHETIC_VIRTUAL,
                execution_mode=ExecutionMode.SIMULATED,
                provenance=ProvenanceType.MEASURED,
                is_warmup=is_warmup,
                incomplete_reason="Empty event stream",
            )

        request_id = events[0].request_id
        clock_source = events[0].clock_source
        execution_mode = events[0].execution_mode

        # 1. Enforce clock source and execution mode consistency across stream
        for idx, ev in enumerate(events):
            if ev.clock_source != clock_source or ev.execution_mode != execution_mode:
                return RequestBenchmarkResult(
                    request_id=request_id,
                    success=False,
                    error_message=(
                        f"Mixed clock sources or execution modes at event index {idx}: "
                        f"expected ({clock_source}, {execution_mode}), got ({ev.clock_source}, {ev.execution_mode})"
                    ),
                    prompt_tokens=prompt_tokens,
                    requested_output_tokens=requested_output_tokens,
                    output_tokens=0,
                    duration_ms=0.0,
                    clock_source=clock_source,
                    execution_mode=execution_mode,
                    provenance=ProvenanceType.MEASURED,
                    is_warmup=is_warmup,
                    incomplete_reason="Inconsistent clock sources in stream",
                )

        # 2. Separate token and chunk events from lifecycle events
        # CRITICAL: START, COMPLETION, and ERROR events are strictly excluded from output token count!
        start_events = [e for e in events if e.event_type == TokenEventType.START]
        token_events = [e for e in events if e.event_type == TokenEventType.TOKEN]
        chunk_events = [e for e in events if e.event_type == TokenEventType.CHUNK]
        terminal_events = [e for e in events if e.is_terminal]

        # Extract server stats if present on completion event
        server_stats: ServerReportedStats | None = None
        if terminal_events and terminal_events[-1].server_stats is not None:
            server_stats = terminal_events[-1].server_stats

        # Determine prompt token count and attribution source
        actual_prompt_tokens: int | None = prompt_tokens
        prompt_source: TokenCountSource = prompt_token_count_source or (
            TokenCountSource.REQUESTED if prompt_tokens is not None else TokenCountSource.UNKNOWN
        )
        if server_stats and server_stats.prompt_tokens is not None:
            actual_prompt_tokens = server_stats.prompt_tokens
            prompt_source = TokenCountSource.SERVER_REPORTED
        elif actual_prompt_tokens is None:
            prompt_source = TokenCountSource.UNKNOWN

        # Determine output token count and attribution source
        output_tokens: int
        output_source: TokenCountSource
        if token_events:
            output_tokens = len(token_events)
            output_source = TokenCountSource.TOKEN_EVENTS
        elif server_stats and server_stats.output_tokens is not None:
            output_tokens = server_stats.output_tokens
            output_source = TokenCountSource.SERVER_REPORTED
        elif not chunk_events:
            output_tokens = 0
            output_source = TokenCountSource.TOKEN_EVENTS
        else:
            output_tokens = 0
            output_source = TokenCountSource.UNKNOWN

        # 3. Check for terminal event completion and errors
        success = True
        error_msg: str | None = None
        incomplete_reason: str | None = None

        if not terminal_events:
            success = False
            incomplete_reason = (
                "Stream terminated prematurely without a terminal COMPLETION or ERROR event"
            )
        else:
            term_event = terminal_events[-1]
            if term_event.event_type == TokenEventType.ERROR:
                success = False
                error_msg = term_event.error_message or "Engine reported stream error"
                incomplete_reason = "Terminated with error event"
            elif term_event.termination_reason == StreamTerminationReason.CANCELLED:
                success = False
                incomplete_reason = "Request stream was cancelled"

        # 4. Resolve start reference timestamp
        start_time_ns: int | None = None
        if start_events:
            start_time_ns = start_events[0].timestamp_ns
        elif submission_time_ns is not None:
            start_time_ns = submission_time_ns
        elif token_events:
            # Fallback if no START emitted: assume first token timestamp is start
            start_time_ns = token_events[0].timestamp_ns
        elif chunk_events:
            start_time_ns = chunk_events[0].timestamp_ns
        else:
            start_time_ns = events[0].timestamp_ns

        # 5. Check for non-monotonic timestamps
        for i in range(1, len(events)):
            if events[i].timestamp_ns < events[i - 1].timestamp_ns:
                success = False
                incomplete_reason = (
                    f"Non-monotonic timestamps detected: event {i} ({events[i].timestamp_ns}) < "
                    f"event {i - 1} ({events[i - 1].timestamp_ns})"
                )
                break

        # 6. Calculate Time-to-First-Token (TTFT) and Time-to-First-Chunk (TTFC)
        # CRITICAL: TTFT requires true TOKEN events. If only CHUNK events are observable,
        # TTFC is reported and true token-level TTFT is marked unavailable (None).
        ttft_ms: float | None = None
        ttfc_ms: float | None = None

        if token_events and start_time_ns is not None:
            ttft_ns = token_events[0].timestamp_ns - start_time_ns
            if ttft_ns < 0:
                success = False
                if not incomplete_reason:
                    incomplete_reason = f"Negative TTFT calculated ({ttft_ns} ns)"
            else:
                ttft_ms = round(ttft_ns / 1_000_000.0, 4)

        if chunk_events and start_time_ns is not None:
            ttfc_ns = chunk_events[0].timestamp_ns - start_time_ns
            if ttfc_ns < 0:
                success = False
                if not incomplete_reason:
                    incomplete_reason = f"Negative TTFC calculated ({ttfc_ns} ns)"
            else:
                ttfc_ms = round(ttfc_ns / 1_000_000.0, 4)

        # 7. Calculate Inter-Token Latencies (ITL) - True Token-Level
        # CRITICAL: Only calculated from verified individual TOKEN events, NEVER from arbitrary CHUNK events!
        itls_ms: list[float] = []
        if len(token_events) >= 2:
            for i in range(1, len(token_events)):
                delta_ns = token_events[i].timestamp_ns - token_events[i - 1].timestamp_ns
                if delta_ns < 0:
                    success = False
                    if not incomplete_reason:
                        incomplete_reason = f"Negative ITL delta between token {i} and {i - 1}"
                itls_ms.append(round(delta_ns / 1_000_000.0, 4))

        # 8. Calculate Time-Per-Output-Token (TPOT) - True Token-Level
        # Defined as average decode time per token EXCLUDING TTFT.
        # Undefined (None) if token_events < 2.
        tpot_ms: float | None = None
        if len(token_events) >= 2 and itls_ms:
            # Total decode time = timestamp of last token - timestamp of first token
            decode_duration_ns = token_events[-1].timestamp_ns - token_events[0].timestamp_ns
            # Average per output decode step
            tpot_ms = round((decode_duration_ns / (len(token_events) - 1)) / 1_000_000.0, 4)

        # 9. Calculate Text-Chunk Inter-Arrival Latencies
        chunk_latencies_ms: list[float] = []
        if len(chunk_events) >= 2:
            for i in range(1, len(chunk_events)):
                c_delta_ns = chunk_events[i].timestamp_ns - chunk_events[i - 1].timestamp_ns
                if c_delta_ns < 0:
                    success = False
                    if not incomplete_reason:
                        incomplete_reason = f"Negative chunk delta between chunk {i} and {i - 1}"
                chunk_latencies_ms.append(round(c_delta_ns / 1_000_000.0, 4))

        chunk_p50 = calculate_percentile(chunk_latencies_ms, 50.0)
        chunk_p90 = calculate_percentile(chunk_latencies_ms, 90.0)
        chunk_p95 = calculate_percentile(chunk_latencies_ms, 95.0)
        chunk_p99 = calculate_percentile(chunk_latencies_ms, 99.0)

        # 10. Total generation duration
        end_time_ns = events[-1].timestamp_ns
        duration_ns = max(0, end_time_ns - (start_time_ns or events[0].timestamp_ns))
        duration_ms = round(duration_ns / 1_000_000.0, 4)

        # 11. Per-request generation throughput (output tokens / duration in seconds)
        output_tok_per_sec: float | None = None
        if duration_ms > 0.0 and output_tokens > 0 and output_source != TokenCountSource.UNKNOWN:
            output_tok_per_sec = round(output_tokens / (duration_ms / 1000.0), 2)
        elif output_tokens == 0 and not chunk_events:
            output_tok_per_sec = 0.0

        # 12. ITL Percentiles
        # Undefined (None) if no ITL intervals exist
        itl_p50 = calculate_percentile(itls_ms, 50.0)
        itl_p90 = calculate_percentile(itls_ms, 90.0)
        itl_p95 = calculate_percentile(itls_ms, 95.0)
        itl_p99 = calculate_percentile(itls_ms, 99.0)

        return RequestBenchmarkResult(
            request_id=request_id,
            success=success,
            error_message=error_msg,
            prompt_tokens=actual_prompt_tokens,
            requested_output_tokens=requested_output_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            ttfc_ms=ttfc_ms,
            tpot_ms=tpot_ms,
            duration_ms=duration_ms,
            output_tokens_per_second=output_tok_per_sec,
            inter_token_latencies_ms=itls_ms,
            itl_p50_ms=itl_p50,
            itl_p90_ms=itl_p90,
            itl_p95_ms=itl_p95,
            itl_p99_ms=itl_p99,
            chunk_count=len(chunk_events),
            chunk_latencies_ms=chunk_latencies_ms,
            chunk_latency_p50_ms=chunk_p50,
            chunk_latency_p90_ms=chunk_p90,
            chunk_latency_p95_ms=chunk_p95,
            chunk_latency_p99_ms=chunk_p99,
            server_stats=server_stats,
            token_count_source=output_source,
            prompt_token_count_source=prompt_source,
            clock_source=clock_source,
            execution_mode=execution_mode,
            provenance=ProvenanceType.MEASURED,
            is_warmup=is_warmup,
            incomplete_reason=incomplete_reason,
        )

    @staticmethod
    def aggregate_suite(
        suite_id: str,
        model_id: str,
        engine_type: EngineType,
        concurrency: int,
        wall_time_seconds: float,
        request_results: list[RequestBenchmarkResult],
        warmup_results: list[RequestBenchmarkResult],
        memory_watermark: MemoryWatermarkResult | None = None,
        status: str = "COMPLETED",
        warnings: list[str] | None = None,
    ) -> BenchmarkSuiteResult:
        """Aggregate per-request outcomes into a BenchmarkSuiteResult.

        Warmup results are strictly excluded from statistical latency averages and throughputs.

        Args:
            suite_id: Unique benchmark run identifier.
            model_id: Model identifier tested.
            engine_type: Engine runtime category.
            concurrency: Concurrency level.
            wall_time_seconds: Total elapsed wall clock time for the suite.
            request_results: Measured request outcomes.
            warmup_results: Unmeasured warm-up request outcomes.
            memory_watermark: Memory telemetry results if available.
            status: Overall status.
            warnings: List of operational warnings.

        Returns:
            Consolidated BenchmarkSuiteResult.
        """
        total_requests = len(request_results)
        warmup_count = len(warmup_results)
        successful_reqs = [r for r in request_results if r.success]
        successful_count = len(successful_reqs)
        failed_count = total_requests - successful_count

        execution_mode = (
            request_results[0].execution_mode if request_results else ExecutionMode.SIMULATED
        )
        clock_source = (
            request_results[0].clock_source if request_results else ClockSource.SYNTHETIC_VIRTUAL
        )

        # Aggregate throughputs
        agg_tokens_per_sec: float | None = None
        agg_output_per_sec: float | None = None
        req_throughput: float | None = None

        if wall_time_seconds > 0.0:
            total_prompt_tok = sum(
                r.prompt_tokens for r in successful_reqs if r.prompt_tokens is not None
            )
            total_output_tok = sum(r.output_tokens for r in successful_reqs)
            total_tokens = total_prompt_tok + total_output_tok

            agg_tokens_per_sec = round(total_tokens / wall_time_seconds, 2)
            agg_output_per_sec = round(total_output_tok / wall_time_seconds, 2)
            req_throughput = round(successful_count / wall_time_seconds, 2)

        # Collect successful TTFT values (excluding None when true tokens unavailable)
        valid_ttfts = [r.ttft_ms for r in successful_reqs if r.ttft_ms is not None]
        ttft_p50 = calculate_percentile(valid_ttfts, 50.0)
        ttft_p90 = calculate_percentile(valid_ttfts, 90.0)
        ttft_p95 = calculate_percentile(valid_ttfts, 95.0)
        ttft_p99 = calculate_percentile(valid_ttfts, 99.0)

        # Collect successful TTFC values (excluding None when no chunks received)
        valid_ttfcs = [r.ttfc_ms for r in successful_reqs if r.ttfc_ms is not None]
        ttfc_p50 = calculate_percentile(valid_ttfcs, 50.0)
        ttfc_p90 = calculate_percentile(valid_ttfcs, 90.0)
        ttfc_p95 = calculate_percentile(valid_ttfcs, 95.0)
        ttfc_p99 = calculate_percentile(valid_ttfcs, 99.0)

        # Flatten all chunk inter-arrival latencies
        all_chunk_latencies: list[float] = []
        for r in successful_reqs:
            all_chunk_latencies.extend(r.chunk_latencies_ms)
        chunk_latency_p50 = calculate_percentile(all_chunk_latencies, 50.0)

        # Collect successful TPOT values (excluding None from 0-token and 1-token requests)
        valid_tpots = [r.tpot_ms for r in successful_reqs if r.tpot_ms is not None]
        tpot_mean = round(sum(valid_tpots) / len(valid_tpots), 4) if valid_tpots else None

        # Flatten all ITLs from successful requests
        all_itls: list[float] = []
        for r in successful_reqs:
            all_itls.extend(r.inter_token_latencies_ms)

        itl_p50 = calculate_percentile(all_itls, 50.0)
        itl_p90 = calculate_percentile(all_itls, 90.0)
        itl_p95 = calculate_percentile(all_itls, 95.0)
        itl_p99 = calculate_percentile(all_itls, 99.0)

        return BenchmarkSuiteResult(
            suite_id=suite_id,
            status=status,
            model_id=model_id,
            engine_type=engine_type,
            concurrency=concurrency,
            total_requests=total_requests,
            warmup_requests=warmup_count,
            successful_requests=successful_count,
            failed_requests=failed_count,
            wall_time_seconds=round(wall_time_seconds, 4),
            aggregate_tokens_per_second=agg_tokens_per_sec,
            aggregate_output_tokens_per_second=agg_output_per_sec,
            request_throughput_per_second=req_throughput,
            ttft_p50_ms=ttft_p50,
            ttft_p90_ms=ttft_p90,
            ttft_p95_ms=ttft_p95,
            ttft_p99_ms=ttft_p99,
            ttfc_p50_ms=ttfc_p50,
            ttfc_p90_ms=ttfc_p90,
            ttfc_p95_ms=ttfc_p95,
            ttfc_p99_ms=ttfc_p99,
            chunk_latency_p50_ms=chunk_latency_p50,
            tpot_mean_ms=tpot_mean,
            itl_p50_ms=itl_p50,
            itl_p90_ms=itl_p90,
            itl_p95_ms=itl_p95,
            itl_p99_ms=itl_p99,
            memory_watermark=memory_watermark,
            execution_mode=execution_mode,
            clock_source=clock_source,
            provenance=ProvenanceType.MEASURED,
            request_results=request_results,
            warmup_results=warmup_results,
            warnings=warnings or [],
        )
