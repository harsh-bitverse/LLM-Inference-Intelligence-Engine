"""In-process asynchronous benchmark job execution manager with live SSE event fanout."""

import asyncio
import contextlib
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.adapters.base import InferenceEngineAdapter
from llm_ii_engine.adapters.llamacpp import LlamaCppAdapter
from llm_ii_engine.adapters.mock import MockEngineAdapter
from llm_ii_engine.adapters.ollama import OllamaAdapter
from llm_ii_engine.benchmark.models import BenchmarkSuiteResult
from llm_ii_engine.benchmark.runner import BenchmarkRunner, BenchmarkRunnerConfig
from llm_ii_engine.core.enums import EngineType
from llm_ii_engine.core.schemas import EngineConfig, ModelSpec, TokenStreamEvent, WorkloadSpec
from llm_ii_engine.storage.database import DatabaseManager
from llm_ii_engine.storage.repository import EngineRepository


class JobStatus(BaseModel):
    """Current status snapshot of an asynchronous benchmark execution job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: str  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    suite_id: str | None = None
    model_id: str
    engine_type: str
    error_message: str | None = None


class BenchmarkSubmissionRequest(BaseModel):
    """Specification submitted by a client to execute an asynchronous benchmark job."""

    model: ModelSpec
    engine_type: EngineType = Field(default=EngineType.MOCK)
    engine_config: EngineConfig | None = Field(default=None)
    workloads: list[WorkloadSpec]
    concurrency: int = Field(default=1, ge=1)
    warmup_requests: int = Field(default=0, ge=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    sample_memory: bool = Field(default=True)
    endpoint_url: str | None = Field(default=None)


class BenchmarkJobManager:
    """Manages active and historical benchmark execution jobs in memory."""

    def __init__(self, db_manager: DatabaseManager, max_concurrent_jobs: int = 4) -> None:
        self._db_manager = db_manager
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._subscribers: dict[str, list[asyncio.Queue[TokenStreamEvent | None]]] = {}
        self._lock = asyncio.Lock()

    async def submit_job(self, req: BenchmarkSubmissionRequest) -> str:
        """Enqueue and launch a background benchmark execution job."""
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        job_info: dict[str, Any] = {
            "job_id": job_id,
            "status": "PENDING",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "suite_id": None,
            "model_id": req.model.model_id,
            "engine_type": req.engine_type.value,
            "error_message": None,
            "result": None,
            "task": None,
        }

        async with self._lock:
            self._jobs[job_id] = job_info
            self._subscribers[job_id] = []

        task = asyncio.create_task(self._execute_job(job_id, req))
        job_info["task"] = task

        return job_id

    def get_job_status(self, job_id: str) -> JobStatus | None:
        """Retrieve status metadata for a job by ID."""
        info = self._jobs.get(job_id)
        if not info:
            return None
        return JobStatus(
            job_id=info["job_id"],
            status=info["status"],
            created_at=info["created_at"],
            started_at=info["started_at"],
            completed_at=info["completed_at"],
            suite_id=info["suite_id"],
            model_id=info["model_id"],
            engine_type=info["engine_type"],
            error_message=info["error_message"],
        )

    def get_job_result(self, job_id: str) -> BenchmarkSuiteResult | None:
        """Retrieve the completed BenchmarkSuiteResult for a job if available."""
        info = self._jobs.get(job_id)
        if not info:
            return None
        return info.get("result")

    def list_jobs(self, limit: int = 50) -> list[JobStatus]:
        """List active and historical job statuses."""
        sorted_jobs = sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)[
            :limit
        ]
        return [
            JobStatus(
                job_id=j["job_id"],
                status=j["status"],
                created_at=j["created_at"],
                started_at=j["started_at"],
                completed_at=j["completed_at"],
                suite_id=j["suite_id"],
                model_id=j["model_id"],
                engine_type=j["engine_type"],
                error_message=j["error_message"],
            )
            for j in sorted_jobs
        ]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending benchmark execution job."""
        info = self._jobs.get(job_id)
        if not info or info["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return False

        task: asyncio.Task[Any] | None = info.get("task")
        if task and not task.done():
            task.cancel()

        info["status"] = "CANCELLED"
        info["completed_at"] = datetime.now(UTC)
        info["error_message"] = "Job cancelled by client request."

        # Notify any waiting SSE subscribers
        await self._broadcast_event(job_id, None)
        return True

    async def subscribe_events(self, job_id: str) -> AsyncGenerator[str, None]:
        """Subscribe to live TokenStreamEvents for a job, formatted as Server-Sent Events."""
        if job_id not in self._jobs:
            return

        queue: asyncio.Queue[TokenStreamEvent | None] = asyncio.Queue(maxsize=1000)
        async with self._lock:
            if job_id not in self._subscribers:
                return
            # If job is already in a terminal state, immediately close without registering queue
            if self._jobs[job_id]["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
                return
            self._subscribers[job_id].append(queue)

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                payload = event.model_dump_json()
                yield f"event: token\ndata: {payload}\n\n"
        finally:
            async with self._lock:
                if job_id in self._subscribers and queue in self._subscribers[job_id]:
                    self._subscribers[job_id].remove(queue)

    async def _broadcast_event(self, job_id: str, event: TokenStreamEvent | None) -> None:
        """Push an event to all active SSE subscriber queues for a job."""
        async with self._lock:
            subscribers = list(self._subscribers.get(job_id, []))

        for q in subscribers:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)

    def _create_adapter(self, req: BenchmarkSubmissionRequest) -> InferenceEngineAdapter:
        """Instantiate the appropriate inference engine adapter based on request."""
        if req.engine_type == EngineType.MOCK:
            mock_cfg = None
            if req.engine_config and req.engine_config.extra_args:
                with contextlib.suppress(Exception):
                    from llm_ii_engine.adapters.mock import MockAdapterConfig

                    mock_cfg = MockAdapterConfig.model_validate(req.engine_config.extra_args)
            return MockEngineAdapter(config=mock_cfg)
        elif req.engine_type == EngineType.OLLAMA:
            url = req.endpoint_url or "http://localhost:11434"
            return OllamaAdapter(base_url=url)
        elif req.engine_type == EngineType.LLAMA_CPP:
            url = req.endpoint_url or "http://localhost:8080"
            return LlamaCppAdapter(base_url=url)
        else:
            return MockEngineAdapter()

    async def _execute_job(self, job_id: str, req: BenchmarkSubmissionRequest) -> None:
        """Internal worker executing the benchmark pipeline with concurrency control."""
        info = self._jobs[job_id]
        try:
            async with self._semaphore:
                # If job was cancelled while waiting in semaphore, exit cleanly
                if info["status"] == "CANCELLED":
                    return

                info["status"] = "RUNNING"
                info["started_at"] = datetime.now(UTC)

                suite_id = f"suite-{uuid.uuid4().hex[:12]}"
                info["suite_id"] = suite_id

                adapter = self._create_adapter(req)
                runner_config = BenchmarkRunnerConfig(
                    concurrency=req.concurrency,
                    warmup_requests=req.warmup_requests,
                    request_timeout_seconds=req.request_timeout_seconds,
                    suite_id=suite_id,
                    sample_memory=req.sample_memory,
                )

                # SSE event callback dispatch
                async def on_stream_event(event: TokenStreamEvent) -> None:
                    await self._broadcast_event(job_id, event)

                resolved_engine_config = req.engine_config or EngineConfig(
                    engine_type=req.engine_type
                )
                runner = BenchmarkRunner(
                    adapter=adapter,
                    model_spec=req.model,
                    engine_config=resolved_engine_config,
                    runner_config=runner_config,
                    on_event=on_stream_event,
                )

                result = await runner.run(req.workloads)

                # Persist to database BEFORE declaring job COMPLETED
                with self._db_manager.get_session() as session:
                    repo = EngineRepository(session)
                    repo.save_benchmark_suite(result)

                # Mark COMPLETED only after durable persistence succeeds
                info["result"] = result
                info["status"] = "COMPLETED"
                info["completed_at"] = datetime.now(UTC)

        except asyncio.CancelledError:
            info["status"] = "CANCELLED"
            info["completed_at"] = datetime.now(UTC)
            if not info.get("error_message"):
                info["error_message"] = "Benchmark job was cancelled."

        except Exception as exc:
            info["status"] = "FAILED"
            info["result"] = None
            info["completed_at"] = datetime.now(UTC)
            info["error_message"] = str(exc)

        finally:
            # Signal completion to SSE subscribers
            await self._broadcast_event(job_id, None)

    async def shutdown(self) -> None:
        """Gracefully cancel and drain all active jobs."""
        tasks: list[asyncio.Task[Any]] = []
        for info in self._jobs.values():
            task: asyncio.Task[Any] | None = info.get("task")
            if task and not task.done():
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
