"""Asynchronous benchmark orchestrator managing concurrent execution, timeouts, and isolation.

Executes workloads against any InferenceEngineAdapter, enforces bounded concurrency,
isolates request failures, separates warm-up runs, and coordinates memory telemetry.
"""

import asyncio
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.adapters.base import InferenceEngineAdapter
from llm_ii_engine.benchmark.collector import MetricCollector
from llm_ii_engine.benchmark.models import (
    BenchmarkSuiteResult,
    MemoryWatermarkResult,
    RequestBenchmarkResult,
)
from llm_ii_engine.benchmark.sampler import AdapterMemorySampler, MemorySampler
from llm_ii_engine.core.enums import (
    ClockSource,
    ExecutionMode,
    ProvenanceType,
    TokenCountSource,
)
from llm_ii_engine.core.exceptions import EngineTimeoutError, StreamCancelledError
from llm_ii_engine.core.schemas import (
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    RequestMetadata,
    TokenStreamEvent,
    WorkloadSpec,
)


class BenchmarkRunnerConfig(BaseModel):
    """Configuration governing benchmark orchestration, concurrency limits, and timeouts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concurrency: int = Field(
        default=1, ge=1, description="Maximum number of concurrent worker requests"
    )
    warmup_requests: int = Field(
        default=0,
        ge=0,
        description="Number of unmeasured warm-up requests dispatched prior to timing",
    )
    request_timeout_seconds: float = Field(
        default=60.0, gt=0.0, description="Per-request execution timeout limit"
    )
    suite_id: str | None = Field(
        default=None, description="Optional custom identifier for the benchmark suite"
    )
    sample_memory: bool = Field(
        default=True, description="Whether to sample memory telemetry during execution"
    )


class BenchmarkRunner:
    """Asynchronous orchestrator for LLM inference benchmarking."""

    def __init__(
        self,
        adapter: InferenceEngineAdapter,
        model_spec: ModelSpec,
        engine_config: EngineConfig,
        runner_config: BenchmarkRunnerConfig | None = None,
        memory_sampler: MemorySampler | None = None,
        on_event: Any | None = None,
    ) -> None:
        self._adapter = adapter
        self._model_spec = model_spec
        self._engine_config = engine_config
        self._config = runner_config or BenchmarkRunnerConfig()
        self._memory_sampler = memory_sampler or (
            AdapterMemorySampler(self._adapter) if self._config.sample_memory else None
        )
        self._on_event = on_event

    async def _execute_single_request(
        self,
        request_idx: int,
        workload: WorkloadSpec,
        is_warmup: bool,
        semaphore: asyncio.Semaphore,
    ) -> RequestBenchmarkResult:
        """Execute an individual generation request with semaphore guarding and timeout."""
        req_id = f"req-{request_idx:04d}" if not is_warmup else f"warmup-{request_idx:04d}"
        metadata = RequestMetadata(
            request_id=req_id,
            model_id=self._model_spec.model_id,
            engine_type=self._adapter.engine_type,
            execution_mode=self._adapter.execution_mode,
        )
        gen_request = GenerationRequest(
            request_id=req_id,
            model_spec=self._model_spec,
            engine_config=self._engine_config,
            workload=workload,
            metadata=metadata,
        )

        async with semaphore:
            events: list[TokenStreamEvent] = []
            submission_ns = time.perf_counter_ns()
            # Explicitly separate verified token counts, heuristic word counts, and unknown counts
            initial_prompt_tokens: int | None = workload.prompt_tokens
            initial_prompt_source: TokenCountSource
            if workload.prompt_tokens is not None:
                initial_prompt_source = TokenCountSource.REQUESTED
            elif workload.prompt_text and len(workload.prompt_text.split()) > 0:
                initial_prompt_tokens = len(workload.prompt_text.split())
                initial_prompt_source = TokenCountSource.HEURISTIC_WORD_COUNT
            else:
                initial_prompt_tokens = None
                initial_prompt_source = TokenCountSource.UNKNOWN

            try:
                # Enforce per-request timeout
                async with asyncio.timeout(self._config.request_timeout_seconds):
                    async for event in self._adapter.generate_stream(gen_request):
                        events.append(event)
                        if self._on_event is not None:
                            try:
                                res = self._on_event(event)
                                if asyncio.iscoroutine(res):
                                    await res
                            except Exception:
                                pass

            except (TimeoutError, EngineTimeoutError, StreamCancelledError) as exc:
                elapsed_sec = (time.perf_counter_ns() - submission_ns) / 1e9
                if isinstance(exc, (TimeoutError, EngineTimeoutError)) or elapsed_sec >= (
                    self._config.request_timeout_seconds * 0.8
                ):
                    return RequestBenchmarkResult(
                        request_id=req_id,
                        success=False,
                        error_message=f"Request exceeded timeout of {self._config.request_timeout_seconds}s: {exc}",
                        prompt_tokens=initial_prompt_tokens,
                        prompt_token_count_source=initial_prompt_source,
                        requested_output_tokens=workload.output_tokens,
                        output_tokens=len([e for e in events if e.event_type.value == "token"]),
                        duration_ms=self._config.request_timeout_seconds * 1000.0,
                        clock_source=ClockSource.MONOTONIC_REAL,
                        execution_mode=self._adapter.execution_mode,
                        provenance=ProvenanceType.MEASURED,
                        is_warmup=is_warmup,
                        incomplete_reason="Request timed out",
                    )
                return RequestBenchmarkResult(
                    request_id=req_id,
                    success=False,
                    error_message="Request was cancelled",
                    prompt_tokens=initial_prompt_tokens,
                    prompt_token_count_source=initial_prompt_source,
                    requested_output_tokens=workload.output_tokens,
                    output_tokens=len([e for e in events if e.event_type.value == "token"]),
                    duration_ms=0.0,
                    clock_source=ClockSource.MONOTONIC_REAL,
                    execution_mode=self._adapter.execution_mode,
                    provenance=ProvenanceType.MEASURED,
                    is_warmup=is_warmup,
                    incomplete_reason="Client cancelled task",
                )

            except asyncio.CancelledError:
                return RequestBenchmarkResult(
                    request_id=req_id,
                    success=False,
                    error_message="Request was cancelled",
                    prompt_tokens=initial_prompt_tokens,
                    prompt_token_count_source=initial_prompt_source,
                    requested_output_tokens=workload.output_tokens,
                    output_tokens=len([e for e in events if e.event_type.value == "token"]),
                    duration_ms=0.0,
                    clock_source=ClockSource.MONOTONIC_REAL,
                    execution_mode=self._adapter.execution_mode,
                    provenance=ProvenanceType.MEASURED,
                    is_warmup=is_warmup,
                    incomplete_reason="Client cancelled task",
                )

            except Exception as exc:
                elapsed_sec = (time.perf_counter_ns() - submission_ns) / 1e9
                if "timeout" in str(exc).lower() and elapsed_sec >= (
                    self._config.request_timeout_seconds * 0.8
                ):
                    return RequestBenchmarkResult(
                        request_id=req_id,
                        success=False,
                        error_message=f"Request exceeded timeout of {self._config.request_timeout_seconds}s: {exc}",
                        prompt_tokens=initial_prompt_tokens,
                        prompt_token_count_source=initial_prompt_source,
                        requested_output_tokens=workload.output_tokens,
                        output_tokens=len([e for e in events if e.event_type.value == "token"]),
                        duration_ms=self._config.request_timeout_seconds * 1000.0,
                        clock_source=ClockSource.MONOTONIC_REAL,
                        execution_mode=self._adapter.execution_mode,
                        provenance=ProvenanceType.MEASURED,
                        is_warmup=is_warmup,
                        incomplete_reason="Request timed out",
                    )
                return RequestBenchmarkResult(
                    request_id=req_id,
                    success=False,
                    error_message=f"Engine runtime exception: {exc}",
                    prompt_tokens=initial_prompt_tokens,
                    prompt_token_count_source=initial_prompt_source,
                    requested_output_tokens=workload.output_tokens,
                    output_tokens=len([e for e in events if e.event_type.value == "token"]),
                    duration_ms=0.0,
                    clock_source=ClockSource.MONOTONIC_REAL,
                    execution_mode=self._adapter.execution_mode,
                    provenance=ProvenanceType.MEASURED,
                    is_warmup=is_warmup,
                    incomplete_reason=f"Exception: {exc}",
                )

            # Process collected events using MetricCollector
            return MetricCollector.collect_request(
                events=events,
                prompt_tokens=initial_prompt_tokens,
                prompt_token_count_source=initial_prompt_source,
                is_warmup=is_warmup,
                submission_time_ns=submission_ns,
                requested_output_tokens=workload.output_tokens,
            )

    async def run(self, workloads: list[WorkloadSpec]) -> BenchmarkSuiteResult:
        """Execute a complete benchmark suite over the provided workload specifications.

        Args:
            workloads: List of WorkloadSpec items defining prompt and output lengths.

        Returns:
            BenchmarkSuiteResult with individual and aggregate metrics.
        """
        suite_id = self._config.suite_id or f"bench-{uuid.uuid4().hex[:12]}"
        semaphore = asyncio.Semaphore(self._config.concurrency)
        warnings: list[str] = []

        if self._adapter.execution_mode == ExecutionMode.SIMULATED:
            warnings.append(
                "Benchmark executed using Mock/Simulated adapter. "
                "Results reflect configured virtual timings and do NOT indicate actual hardware performance."
            )

        # 1. Start memory sampling if active
        if self._memory_sampler:
            try:
                await self._memory_sampler.start()
            except Exception as exc:
                warnings.append(f"Failed to start memory sampler: {exc}")

        # 2. Warm-up phase (unmeasured)
        warmup_results: list[RequestBenchmarkResult] = []
        if self._config.warmup_requests > 0:
            warmup_workload = (
                workloads[: self._config.warmup_requests]
                if len(workloads) >= self._config.warmup_requests
                else workloads * self._config.warmup_requests
            )[: self._config.warmup_requests]

            warmup_tasks = [
                self._execute_single_request(i, wl, is_warmup=True, semaphore=semaphore)
                for i, wl in enumerate(warmup_workload)
            ]
            warmup_results = await asyncio.gather(*warmup_tasks)

        # 3. Measured phase
        real_start_time = time.perf_counter()
        measured_tasks = [
            self._execute_single_request(i, wl, is_warmup=False, semaphore=semaphore)
            for i, wl in enumerate(workloads)
        ]
        measured_results = await asyncio.gather(*measured_tasks)
        real_elapsed = time.perf_counter() - real_start_time

        # 4. Stop memory sampler
        memory_watermark: MemoryWatermarkResult | None = None
        if self._memory_sampler:
            try:
                memory_watermark = await self._memory_sampler.stop()
            except Exception as exc:
                warnings.append(f"Error stopping memory sampler: {exc}")

        # 5. Resolve wall clock time
        # In virtual clock mode, derive wall clock from simulated nanosecond timestamps
        wall_time = real_elapsed
        is_virtual = any(r.clock_source == ClockSource.SYNTHETIC_VIRTUAL for r in measured_results)
        if is_virtual and measured_results:
            # For virtual execution, the effective wall time across concurrent streams
            # is the maximum duration among concurrent requests or virtual span
            virtual_durations = [
                r.duration_ms / 1000.0 for r in measured_results if r.duration_ms > 0
            ]
            if virtual_durations:
                # Under concurrency C, total workload wall time is approximated by max request duration
                # or total serial time divided by concurrency
                wall_time = max(
                    max(virtual_durations), sum(virtual_durations) / self._config.concurrency
                )

        status = "COMPLETED"
        if not any(r.success for r in measured_results):
            status = "FAILED"
        elif any(not r.success for r in measured_results):
            status = "PARTIAL"

        return MetricCollector.aggregate_suite(
            suite_id=suite_id,
            model_id=self._model_spec.model_id,
            engine_type=self._adapter.engine_type,
            concurrency=self._config.concurrency,
            wall_time_seconds=wall_time,
            request_results=measured_results,
            warmup_results=warmup_results,
            memory_watermark=memory_watermark,
            status=status,
            warnings=warnings,
        )
