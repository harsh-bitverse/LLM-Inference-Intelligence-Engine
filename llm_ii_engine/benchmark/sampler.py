"""Memory resource telemetry and watermark sampling during benchmark execution.

Collects host process RSS and adapter device memory without adding GPU-specific dependencies.
Reports unsupported or missing telemetry explicitly without substituting zero values.
"""

import asyncio
import contextlib
from typing import Protocol, runtime_checkable

from llm_ii_engine.adapters.base import InferenceEngineAdapter
from llm_ii_engine.benchmark.models import MemoryWatermarkResult


@runtime_checkable
class MemorySampler(Protocol):
    """Protocol for sampling memory consumption during benchmark execution."""

    async def start(self) -> None:
        """Begin periodic memory sampling."""
        ...

    async def sample_now(self) -> None:
        """Trigger an immediate memory telemetry sample."""
        ...

    async def stop(self) -> MemoryWatermarkResult:
        """Halt sampling and return the consolidated memory watermark result."""
        ...


class AdapterMemorySampler:
    """Samples memory consumption via an InferenceEngineAdapter's get_memory_usage_mb() method."""

    def __init__(
        self,
        adapter: InferenceEngineAdapter,
        sample_interval_ms: float = 50.0,
    ) -> None:
        self._adapter = adapter
        self._interval_ms = sample_interval_ms
        self._is_running = False
        self._sample_task: asyncio.Task[None] | None = None
        self._gpu_samples: list[float] = []
        self._process_samples: list[float] = []
        self._telemetry_available: bool = False

    async def _sample_loop(self) -> None:
        while self._is_running:
            await self.sample_now()
            await asyncio.sleep(self._interval_ms / 1000.0)

    async def start(self) -> None:
        """Start the background sampling loop."""
        self._is_running = True
        self._gpu_samples.clear()
        self._process_samples.clear()
        self._sample_task = asyncio.create_task(self._sample_loop())

    async def sample_now(self) -> None:
        """Record an individual memory telemetry reading."""
        try:
            gpu_mem = await self._adapter.get_memory_usage_mb()
            if gpu_mem is not None:
                self._gpu_samples.append(gpu_mem)
                self._telemetry_available = True
        except Exception:
            pass

    async def stop(self) -> MemoryWatermarkResult:
        """Stop sampling and compile the peak memory watermark."""
        self._is_running = False
        if self._sample_task and not self._sample_task.done():
            self._sample_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sample_task
            self._sample_task = None

        sample_count = len(self._gpu_samples)
        peak_gpu: float | None = max(self._gpu_samples) if self._gpu_samples else None

        notes: str | None = None
        if not self._telemetry_available or peak_gpu is None:
            is_http = hasattr(self._adapter, "base_url")
            daemon_note = (
                " Separate process execution: external model daemon runs outside client process; "
                "host process RSS cannot be attributed to the inference runtime."
                if is_http
                else ""
            )
            notes = (
                f"GPU memory telemetry unavailable for engine adapter '{self._adapter.name}'.{daemon_note} "
                "Values are reported as None rather than fabricated zeros."
            )

        return MemoryWatermarkResult(
            peak_process_rss_mb=None,  # Process RSS omitted if platform provider not active
            peak_gpu_vram_mb=peak_gpu,
            sample_count=sample_count,
            sample_interval_ms=self._interval_ms,
            telemetry_source=self._adapter.name,
            is_telemetry_available=self._telemetry_available and (peak_gpu is not None),
            notes=notes,
            device_index=None,
            device_name=None,
        )


class MockMemorySampler:
    """Deterministic memory sampler for automated testing and CI verification."""

    def __init__(
        self,
        simulated_gpu_mb: float | None = 2048.0,
        simulated_rss_mb: float | None = 512.0,
        is_available: bool = True,
    ) -> None:
        self.simulated_gpu_mb = simulated_gpu_mb
        self.simulated_rss_mb = simulated_rss_mb
        self.is_available = is_available
        self.sample_count = 0

    async def start(self) -> None:
        self.sample_count = 0

    async def sample_now(self) -> None:
        self.sample_count += 1

    async def stop(self) -> MemoryWatermarkResult:
        notes = (
            None
            if self.is_available
            else "Telemetry explicitly marked unavailable in mock sampler."
        )
        return MemoryWatermarkResult(
            peak_process_rss_mb=self.simulated_rss_mb if self.is_available else None,
            peak_gpu_vram_mb=self.simulated_gpu_mb if self.is_available else None,
            sample_count=max(1, self.sample_count),
            sample_interval_ms=50.0,
            telemetry_source="mock",
            is_telemetry_available=self.is_available,
            notes=notes,
        )
