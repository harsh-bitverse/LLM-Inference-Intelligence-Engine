"""Versioned benchmark experiment manifests, pilot workloads, and experiment orchestrator.

Defines reproducible, evidence-qualified experiment specifications (Session 11) with
deterministic prompts, parameter tracking, warmup isolation, and failure policy controls.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.adapters.base import InferenceEngineAdapter
from llm_ii_engine.benchmark.collector import MetricCollector
from llm_ii_engine.benchmark.models import (
    BenchmarkSuiteResult,
    MemoryWatermarkResult,
    RequestBenchmarkResult,
)
from llm_ii_engine.benchmark.runner import BenchmarkRunner, BenchmarkRunnerConfig
from llm_ii_engine.benchmark.sampler import AdapterMemorySampler, MemorySampler
from llm_ii_engine.core.enums import (
    ClockSource,
    ExecutionMode,
    FailurePolicy,
    ProvenanceType,
    QuantizationType,
)
from llm_ii_engine.core.schemas import EngineConfig, ModelSpec, WorkloadSpec

# ============================================================================
# Standard Deterministic Pilot Workloads (Non-sensitive, CPU-manageable)
# ============================================================================

PILOT_SHORT_WORKLOAD = WorkloadSpec(
    prompt_tokens=32,
    output_tokens=48,
    prompt_text="Explain the principle of locality in computer memory hierarchies in two concise sentences.",
    seed=42,
)

PILOT_MEDIUM_WORKLOAD = WorkloadSpec(
    prompt_tokens=128,
    output_tokens=96,
    prompt_text=(
        "Describe the architectural difference between the prefill and decode phases in autoregressive "
        "transformer inference. Discuss operational intensity and memory bandwidth bottlenecks."
    ),
    seed=43,
)

PILOT_LONG_WORKLOAD = WorkloadSpec(
    prompt_tokens=384,
    output_tokens=128,
    prompt_text=(
        "Summarize the key engineering trade-offs between dense model inference and mixture-of-experts (MoE) "
        "architectures. Specifically compare total parameter memory storage, active parameter footprint per token, "
        "KV-cache memory scaling, and communication overhead under tensor parallelism across multi-accelerator nodes. "
        "Highlight why MoE models often achieve lower latency per token while requiring substantially higher aggregate VRAM."
    ),
    seed=44,
)

STANDARD_PILOT_SUITE: list[WorkloadSpec] = [
    PILOT_SHORT_WORKLOAD,
    PILOT_MEDIUM_WORKLOAD,
    PILOT_LONG_WORKLOAD,
]


# ============================================================================
# Benchmark Experiment Manifest
# ============================================================================


class BenchmarkExperimentManifest(BaseModel):
    """Immutable, versioned specification for a reproducible benchmark experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_version: str = Field(
        default="1.0.0", description="Semantic version of the manifest schema"
    )
    experiment_id: str = Field(
        default_factory=lambda: f"exp-{uuid.uuid4().hex[:12]}",
        description="Unique deterministic or random experiment run identifier",
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when manifest was generated or benchmark scheduled",
    )

    # Runtime & Engine Specification
    runtime_name: str = Field(
        description="Inference runtime identifier (e.g. 'ollama', 'llamacpp', 'mock')"
    )
    runtime_version: str | None = Field(
        default=None, description="Reported or detected version of the serving runtime"
    )
    adapter_version: str = Field(
        default="1.0.0", description="Version of the IIE adapter communicating with runtime"
    )
    engine_version: str = Field(
        default="1.0.0", description="Version of the IIE core benchmarking framework"
    )

    # Model Specification & Artifact Verification
    model_id: str = Field(description="Canonical model identifier (e.g. 'qwen2.5:0.5b')")
    model_digest: str | None = Field(
        default=None, description="Cryptographic hash, GGUF file digest, or artifact identifier"
    )
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE, description="Weight and activation quantization scheme"
    )
    context_window: int | None = Field(
        default=None, ge=1, description="Configured maximum context window in tokens"
    )
    gpu_layers: int | None = Field(
        default=None, ge=0, description="Number of model layers offloaded to GPU (llama.cpp)"
    )
    extra_engine_args: dict[str, Any] = Field(
        default_factory=dict, description="Custom flags passed to runtime or adapter"
    )

    # Target Hardware Environment
    hardware_profile_id: str | None = Field(
        default=None, description="Catalog ID or unique identifier for the host hardware profile"
    )
    device_name: str = Field(
        default="CPU", description="Primary execution device (e.g. 'CPU', 'NVIDIA RTX 4090')"
    )
    cpu_model: str | None = Field(default=None, description="Host CPU processor name and core count")
    host_ram_gb: float | None = Field(
        default=None, ge=0.0, description="Total host physical RAM in gigabytes"
    )
    gpu_vram_gb: float | None = Field(
        default=None, ge=0.0, description="Available GPU memory in gigabytes (None if CPU)"
    )

    # Workload Suite & Repetition Control
    workload_suite: list[WorkloadSpec] = Field(
        min_length=1, description="Deterministic sequence of workloads to execute"
    )
    warmup_repetitions: int = Field(
        default=1,
        ge=0,
        description="Number of unmeasured warm-up passes executed before recording measurements",
    )
    measured_repetitions: int = Field(
        default=3,
        ge=1,
        description="Number of measured iterations executed for each workload prompt",
    )
    concurrency: int = Field(
        default=1, ge=1, description="Maximum concurrent client requests permitted"
    )

    # Sampling Parameters (for Reproducibility)
    sampling_temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Sampling temperature (0.0 for greedy deterministic decoding)",
    )
    sampling_top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Nucleus sampling top_p threshold"
    )
    seed: int | None = Field(
        default=42, description="Random generator seed passed to engine where supported"
    )

    # Operational & Failure Governance
    timeout_seconds: float = Field(
        default=60.0, gt=0.0, description="Per-request execution timeout limit"
    )
    failure_policy: FailurePolicy = Field(
        default=FailurePolicy.CONTINUE_ON_ERROR,
        description="Behavior when a request encounters an error or timeout",
    )
    consecutive_failure_limit: int = Field(
        default=3,
        ge=1,
        description="Maximum allowable consecutive failures before aborting experiment",
    )

    # Integrity & Provenance Declarations
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.REAL,
        description="REAL for physical model inference, SIMULATED for synthetic runs",
    )
    clock_source: ClockSource = Field(
        default=ClockSource.MONOTONIC_REAL,
        description="Clock provider used for timing measurements",
    )
    metric_provenance: ProvenanceType = Field(
        default=ProvenanceType.MEASURED,
        description="Provenance of recorded timing: MEASURED from physical timers",
    )
    environmental_caveats: list[str] = Field(
        default_factory=list,
        description="Environmental notes (e.g. CPU-only host, background processes)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary user or system metadata annotations"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to a JSON-serializable dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkExperimentManifest:
        """Parse manifest from dictionary representation."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> BenchmarkExperimentManifest:
        """Parse manifest from JSON string."""
        return cls.model_validate_json(json_str)

    def save(self, path: str | Path) -> None:
        """Save manifest to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkExperimentManifest:
        """Load manifest from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def create_standard_pilot(
        cls,
        model_id: str,
        runtime_name: str = "ollama",
        quantization: QuantizationType = QuantizationType.NONE,
        device_name: str = "CPU",
        cpu_model: str | None = None,
        host_ram_gb: float | None = None,
        warmup_repetitions: int = 1,
        measured_repetitions: int = 3,
        execution_mode: ExecutionMode = ExecutionMode.REAL,
        runtime_version: str | None = None,
        environmental_caveats: list[str] | None = None,
    ) -> BenchmarkExperimentManifest:
        """Factory for generating a standardized CPU-friendly pilot benchmark manifest."""
        return cls(
            model_id=model_id,
            runtime_name=runtime_name,
            runtime_version=runtime_version,
            quantization=quantization,
            device_name=device_name,
            cpu_model=cpu_model,
            host_ram_gb=host_ram_gb,
            workload_suite=list(STANDARD_PILOT_SUITE),
            warmup_repetitions=warmup_repetitions,
            measured_repetitions=measured_repetitions,
            concurrency=1,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            seed=42,
            timeout_seconds=60.0,
            failure_policy=FailurePolicy.CONTINUE_ON_ERROR,
            execution_mode=execution_mode,
            clock_source=ClockSource.MONOTONIC_REAL,
            metric_provenance=ProvenanceType.MEASURED,
            environmental_caveats=environmental_caveats or ["CPU-only execution", "Shared OS memory"],
        )


# ============================================================================
# Experiment Result Container
# ============================================================================


class ExperimentResult(BaseModel):
    """Complete outcomes of an executed benchmark experiment manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: BenchmarkExperimentManifest
    suite_result: BenchmarkSuiteResult
    all_requests: list[RequestBenchmarkResult]
    warmup_requests: list[RequestBenchmarkResult]
    measured_requests: list[RequestBenchmarkResult]
    successful_measured_requests: list[RequestBenchmarkResult]
    failed_requests: list[RequestBenchmarkResult]
    total_wall_time_seconds: float
    status: str = Field(description="'COMPLETED', 'FAILED', or 'PARTIAL'")


# ============================================================================
# Experiment Orchestrator
# ============================================================================


class ExperimentRunner:
    """Executes a BenchmarkExperimentManifest with strict isolation, warmup exclusion, and timing."""

    def __init__(
        self,
        adapter: InferenceEngineAdapter,
        model_spec: ModelSpec,
        memory_sampler: MemorySampler | None = None,
    ) -> None:
        self._adapter = adapter
        self._model_spec = model_spec
        self._memory_sampler = memory_sampler or AdapterMemorySampler(adapter)

    async def run_experiment(
        self,
        manifest: BenchmarkExperimentManifest,
    ) -> ExperimentResult:
        """Execute all repetitions defined in the manifest.

        Args:
            manifest: The validated benchmark experiment specification.

        Returns:
            ExperimentResult containing all request outcomes, aggregates, and failure tracking.
        """
        wall_start = time.perf_counter()
        engine_config = EngineConfig(
            engine_type=self._adapter.engine_type,
            tensor_parallel_size=1,
            quantization=manifest.quantization,
            max_model_len=manifest.context_window,
            extra_args={
                **manifest.extra_engine_args,
                "temperature": manifest.sampling_temperature,
                "top_p": manifest.sampling_top_p,
                "seed": manifest.seed,
            },
        )

        all_requests: list[RequestBenchmarkResult] = []
        warmup_requests: list[RequestBenchmarkResult] = []
        measured_requests: list[RequestBenchmarkResult] = []
        consecutive_failures = 0
        aborted = False

        # Start memory sampling if requested
        if self._memory_sampler:
            await self._memory_sampler.start()

        req_global_idx = 0
        try:
            # 1. Warm-up Repetitions (Explicitly tagged and excluded from measured aggregates)
            for w_idx in range(manifest.warmup_repetitions):
                if aborted:
                    break
                for workload in manifest.workload_suite:
                    res = await self._execute_request(
                        request_idx=req_global_idx,
                        workload=workload,
                        engine_config=engine_config,
                        timeout_seconds=manifest.timeout_seconds,
                        is_warmup=True,
                    )
                    req_global_idx += 1
                    warmup_requests.append(res)
                    all_requests.append(res)

                    if not res.success:
                        consecutive_failures += 1
                        if manifest.failure_policy == FailurePolicy.ABORT_IMMEDIATELY or (
                            manifest.failure_policy == FailurePolicy.ABORT_ON_CONSECUTIVE
                            and consecutive_failures >= manifest.consecutive_failure_limit
                        ):
                            aborted = True
                            break
                    else:
                        consecutive_failures = 0

            # 2. Measured Repetitions
            for rep_idx in range(manifest.measured_repetitions):
                if aborted:
                    break
                for workload in manifest.workload_suite:
                    res = await self._execute_request(
                        request_idx=req_global_idx,
                        workload=workload,
                        engine_config=engine_config,
                        timeout_seconds=manifest.timeout_seconds,
                        is_warmup=False,
                    )
                    req_global_idx += 1
                    measured_requests.append(res)
                    all_requests.append(res)

                    if not res.success:
                        consecutive_failures += 1
                        if manifest.failure_policy == FailurePolicy.ABORT_IMMEDIATELY or (
                            manifest.failure_policy == FailurePolicy.ABORT_ON_CONSECUTIVE
                            and consecutive_failures >= manifest.consecutive_failure_limit
                        ):
                            aborted = True
                            break
                    else:
                        consecutive_failures = 0

        finally:
            mem_result: MemoryWatermarkResult | None = None
            if self._memory_sampler:
                mem_result = await self._memory_sampler.stop()

        wall_duration = time.perf_counter() - wall_start
        successful_measured = [r for r in measured_requests if r.success]
        failed_requests = [r for r in all_requests if not r.success]

        # Determine overall experiment status
        if aborted:
            status = "FAILED" if not successful_measured else "PARTIAL"
        elif not successful_measured and measured_requests:
            status = "FAILED"
        elif any(not r.success for r in measured_requests):
            status = "PARTIAL"
        else:
            status = "COMPLETED"

        # Build suite aggregates
        suite = MetricCollector.aggregate_suite(
            suite_id=manifest.experiment_id,
            model_id=manifest.model_id,
            engine_type=self._adapter.engine_type,
            concurrency=manifest.concurrency,
            wall_time_seconds=wall_duration,
            request_results=measured_requests,
            warmup_results=warmup_requests,
            memory_watermark=mem_result,
            status=status,
        )

        return ExperimentResult(
            manifest=manifest,
            suite_result=suite,
            all_requests=all_requests,
            warmup_requests=warmup_requests,
            measured_requests=measured_requests,
            successful_measured_requests=successful_measured,
            failed_requests=failed_requests,
            total_wall_time_seconds=wall_duration,
            status=status,
        )

    async def _execute_request(
        self,
        request_idx: int,
        workload: WorkloadSpec,
        engine_config: EngineConfig,
        timeout_seconds: float,
        is_warmup: bool,
    ) -> RequestBenchmarkResult:
        """Dispatch a single request via BenchmarkRunner with timeout isolation."""
        runner_cfg = BenchmarkRunnerConfig(
            concurrency=1,
            warmup_requests=0,
            request_timeout_seconds=timeout_seconds,
            sample_memory=False,  # Managed at suite level
        )
        runner = BenchmarkRunner(
            adapter=self._adapter,
            model_spec=self._model_spec,
            engine_config=engine_config,
            runner_config=runner_cfg,
        )
        sem = asyncio.Semaphore(1)
        return await runner._execute_single_request(
            request_idx=request_idx,
            workload=workload,
            is_warmup=is_warmup,
            semaphore=sem,
        )
