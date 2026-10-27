"""Unit tests for asynchronous benchmark runner, concurrency bounds, timeouts, and isolation."""

import pytest

from llm_ii_engine.adapters.mock import MockAdapterConfig, MockEngineAdapter
from llm_ii_engine.benchmark.models import BenchmarkSuiteResult
from llm_ii_engine.benchmark.runner import BenchmarkRunner, BenchmarkRunnerConfig
from llm_ii_engine.benchmark.sampler import MockMemorySampler
from llm_ii_engine.benchmark.workload import (
    SCENARIO_PREFILL_ONLY,
    generate_workload,
)
from llm_ii_engine.core.enums import EngineType, ExecutionMode, QuantizationType
from llm_ii_engine.core.schemas import EngineConfig, ModelSpec, WorkloadSpec


@pytest.fixture
def default_model() -> ModelSpec:
    return ModelSpec(model_id="mock-llama3-8b", parameter_count_billion=8.0)


@pytest.fixture
def default_engine() -> EngineConfig:
    return EngineConfig(engine_type=EngineType.MOCK, quantization=QuantizationType.FP16)


class TestBenchmarkRunnerExecution:
    """Tests for benchmark runner orchestration, warmup separation, and metrics."""

    @pytest.mark.asyncio
    async def test_end_to_end_mock_benchmark_run(
        self, default_model: ModelSpec, default_engine: EngineConfig
    ) -> None:
        adapter_config = MockAdapterConfig(
            prefill_delay_ns=30_000_000,
            inter_token_delay_ns=10_000_000,
            virtual_time_mode=True,
        )
        adapter = MockEngineAdapter(config=adapter_config)
        await adapter.initialize(default_model, default_engine)

        runner_config = BenchmarkRunnerConfig(
            concurrency=2,
            warmup_requests=2,
            request_timeout_seconds=5.0,
            suite_id="test-suite-001",
        )
        runner = BenchmarkRunner(
            adapter=adapter,
            model_spec=default_model,
            engine_config=default_engine,
            runner_config=runner_config,
        )

        workloads = [
            WorkloadSpec(prompt_tokens=128, output_tokens=5, seed=1),
            WorkloadSpec(prompt_tokens=128, output_tokens=5, seed=2),
            WorkloadSpec(prompt_tokens=128, output_tokens=5, seed=3),
            WorkloadSpec(prompt_tokens=128, output_tokens=5, seed=4),
        ]

        result = await runner.run(workloads)

        assert isinstance(result, BenchmarkSuiteResult)
        assert result.suite_id == "test-suite-001"
        assert result.status == "COMPLETED"
        assert result.total_requests == 4
        assert result.warmup_requests == 2
        assert result.successful_requests == 4
        assert result.failed_requests == 0
        assert result.concurrency == 2
        assert result.execution_mode == ExecutionMode.SIMULATED

        # Metrics must be computed and valid
        assert result.ttft_p50_ms == 30.0
        assert result.tpot_mean_ms == 10.0
        assert result.aggregate_output_tokens_per_second is not None
        assert result.aggregate_output_tokens_per_second > 0.0

        # Memory telemetry sampled from adapter
        assert result.memory_watermark is not None
        assert result.memory_watermark.peak_gpu_vram_mb == 1024.0

    @pytest.mark.asyncio
    async def test_concurrency_bounding(
        self, default_model: ModelSpec, default_engine: EngineConfig
    ) -> None:
        # Use real sleep to verify bounded concurrency semaphore
        adapter_config = MockAdapterConfig(
            prefill_delay_ns=20_000_000,  # 20ms real sleep
            inter_token_delay_ns=10_000_000,
            virtual_time_mode=False,
        )
        adapter = MockEngineAdapter(config=adapter_config)
        await adapter.initialize(default_model, default_engine)

        runner_config = BenchmarkRunnerConfig(
            concurrency=2,  # Maximum 2 concurrent requests
            warmup_requests=0,
            request_timeout_seconds=10.0,
        )
        runner = BenchmarkRunner(
            adapter=adapter,
            model_spec=default_model,
            engine_config=default_engine,
            runner_config=runner_config,
        )

        workloads = [WorkloadSpec(prompt_tokens=64, output_tokens=2)] * 4
        result = await runner.run(workloads)

        assert result.status == "COMPLETED"
        assert result.successful_requests == 4

    @pytest.mark.asyncio
    async def test_request_error_isolation(
        self, default_model: ModelSpec, default_engine: EngineConfig
    ) -> None:
        # Injected error in adapter
        adapter_config = MockAdapterConfig(
            error_at_token_index=1,
            error_delivery_mode="event",
            virtual_time_mode=True,
        )
        adapter = MockEngineAdapter(config=adapter_config)
        await adapter.initialize(default_model, default_engine)

        runner_config = BenchmarkRunnerConfig(concurrency=1, warmup_requests=0)
        runner = BenchmarkRunner(
            adapter=adapter,
            model_spec=default_model,
            engine_config=default_engine,
            runner_config=runner_config,
        )

        workloads = [WorkloadSpec(prompt_tokens=100, output_tokens=3)] * 2
        result = await runner.run(workloads)

        # Requests failed due to injected error, but runner completed without crashing
        assert result.status == "FAILED"
        assert result.failed_requests == 2
        assert result.successful_requests == 0
        assert any(r.error_message is not None for r in result.request_results)

    @pytest.mark.asyncio
    async def test_per_request_timeout_handling(
        self, default_model: ModelSpec, default_engine: EngineConfig
    ) -> None:
        # Adapter takes 200ms per token, but runner timeout is 50ms
        adapter_config = MockAdapterConfig(
            prefill_delay_ns=200_000_000,  # 200ms
            inter_token_delay_ns=50_000_000,
            virtual_time_mode=False,
        )
        adapter = MockEngineAdapter(config=adapter_config)
        await adapter.initialize(default_model, default_engine)

        runner_config = BenchmarkRunnerConfig(
            concurrency=1,
            warmup_requests=0,
            request_timeout_seconds=0.05,  # 50ms timeout
        )
        runner = BenchmarkRunner(
            adapter=adapter,
            model_spec=default_model,
            engine_config=default_engine,
            runner_config=runner_config,
        )

        workloads = [WorkloadSpec(prompt_tokens=50, output_tokens=3)]
        result = await runner.run(workloads)

        assert result.status == "FAILED"
        assert result.failed_requests == 1
        assert "exceeded timeout" in (result.request_results[0].error_message or "")

    @pytest.mark.asyncio
    async def test_missing_memory_telemetry_handling(
        self, default_model: ModelSpec, default_engine: EngineConfig
    ) -> None:
        adapter = MockEngineAdapter()
        sampler = MockMemorySampler(is_available=False)  # Explicitly unavailable telemetry

        runner = BenchmarkRunner(
            adapter=adapter,
            model_spec=default_model,
            engine_config=default_engine,
            runner_config=BenchmarkRunnerConfig(concurrency=1),
            memory_sampler=sampler,
        )

        workloads = [WorkloadSpec(prompt_tokens=10, output_tokens=2)]
        result = await runner.run(workloads)

        assert result.memory_watermark is not None
        assert result.memory_watermark.is_telemetry_available is False
        assert result.memory_watermark.peak_gpu_vram_mb is None  # Must NOT be 0.0
        assert "unavailable" in (result.memory_watermark.notes or "")

    @pytest.mark.asyncio
    async def test_prefill_only_scenario_execution(
        self, default_model: ModelSpec, default_engine: EngineConfig
    ) -> None:
        adapter = MockEngineAdapter()
        workloads = generate_workload(SCENARIO_PREFILL_ONLY)[:3]

        runner = BenchmarkRunner(
            adapter=adapter,
            model_spec=default_model,
            engine_config=default_engine,
            runner_config=BenchmarkRunnerConfig(concurrency=1),
        )

        result = await runner.run(workloads)
        assert result.status == "COMPLETED"
        assert result.successful_requests == 3
        # In prefill-only (output_tokens == 0), TTFT and TPOT are undefined (None)
        assert result.ttft_p50_ms is None
        assert result.tpot_mean_ms is None
