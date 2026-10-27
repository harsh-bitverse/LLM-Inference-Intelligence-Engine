"""Targeted reliability, lifecycle, concurrency, SSE, and error boundary audit tests for the REST API and job execution subsystem."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from llm_ii_engine.adapters.base import InferenceEngineAdapter
from llm_ii_engine.api.app import create_app
from llm_ii_engine.api.jobs import (
    BenchmarkJobManager,
    BenchmarkSubmissionRequest,
)
from llm_ii_engine.benchmark.runner import BenchmarkRunner
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    TokenEventType,
)
from llm_ii_engine.core.schemas import (
    EngineCapabilities,
    GenerationRequest,
    ModelSpec,
    TokenStreamEvent,
    WorkloadSpec,
)
from llm_ii_engine.storage.database import DatabaseManager
from llm_ii_engine.storage.repository import EngineRepository


@pytest.fixture
def temp_db_url(tmp_path: Path) -> str:
    db_file = tmp_path / "test_api_audit.db"
    return f"sqlite:///{db_file}"


@pytest.fixture
def db_manager(temp_db_url: str) -> DatabaseManager:
    mgr = DatabaseManager(database_url=temp_db_url)
    mgr.init_db()
    return mgr


@pytest.fixture
def app_with_mgr(db_manager: DatabaseManager):
    return create_app(db_manager=db_manager)


@pytest.fixture
async def client(app_with_mgr) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app_with_mgr)
    async with (
        AsyncClient(transport=transport, base_url="http://testserver") as ac,
        app_with_mgr.router.lifespan_context(app_with_mgr),
    ):
        yield ac


# -----------------------------------------------------------------------------
# 1. Job Lifecycle and Concurrency Audit Tests
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_job_pending_to_cancelled_before_execution(db_manager: DatabaseManager) -> None:
    """Verify a job queued behind a saturated semaphore can be cancelled while PENDING."""
    job_mgr = BenchmarkJobManager(db_manager=db_manager, max_concurrent_jobs=1)

    model = ModelSpec(model_id="test-model", parameter_count_billion=1.0)
    workload = WorkloadSpec(prompt_tokens=16, output_tokens=8)
    req1 = BenchmarkSubmissionRequest(
        model=model,
        engine_type=EngineType.MOCK,
        workloads=[workload],
    )
    req2 = BenchmarkSubmissionRequest(
        model=model,
        engine_type=EngineType.MOCK,
        workloads=[workload],
    )

    # Launch Job 1 (takes worker slot)
    job1_id = await job_mgr.submit_job(req1)
    # Launch Job 2 (queued in PENDING)
    job2_id = await job_mgr.submit_job(req2)

    status2_initial = job_mgr.get_job_status(job2_id)
    assert status2_initial is not None
    assert status2_initial.status in ("PENDING", "RUNNING")

    # Cancel Job 2
    cancelled = await job_mgr.cancel_job(job2_id)
    assert cancelled is True

    status2_after = job_mgr.get_job_status(job2_id)
    assert status2_after is not None
    assert status2_after.status == "CANCELLED"
    assert status2_after.completed_at is not None
    assert "Job cancelled" in (status2_after.error_message or "")

    # Wait for Job 1 to finish
    for _ in range(50):
        s1 = job_mgr.get_job_status(job1_id)
        if s1 and s1.status == "COMPLETED":
            break
        await asyncio.sleep(0.05)

    # Verify Job 2 never ran or saved a suite
    assert job_mgr.get_job_result(job2_id) is None
    await job_mgr.shutdown()


@pytest.mark.anyio
async def test_job_running_to_cancelled(db_manager: DatabaseManager) -> None:
    """Verify a RUNNING job is successfully interrupted and marked CANCELLED."""
    job_mgr = BenchmarkJobManager(db_manager=db_manager, max_concurrent_jobs=2)

    model = ModelSpec(model_id="slow-model", parameter_count_billion=1.0)
    workload = WorkloadSpec(prompt_tokens=100, output_tokens=500)
    req = BenchmarkSubmissionRequest(
        model=model,
        engine_type=EngineType.MOCK,
        workloads=[workload],
    )

    started_event = asyncio.Event()

    async def slow_runner(*args: Any, **kwargs: Any) -> Any:
        started_event.set()
        await asyncio.sleep(10.0)

    with patch.object(BenchmarkRunner, "run", side_effect=slow_runner):
        job_id = await job_mgr.submit_job(req)
        await started_event.wait()

        # Cancel while actively running
        cancelled = await job_mgr.cancel_job(job_id)
        assert cancelled is True

        status_final = job_mgr.get_job_status(job_id)
        assert status_final is not None
        assert status_final.status == "CANCELLED"
        assert status_final.completed_at is not None

        # Repeated cancellation should return False (already terminal)
        repeat_cancel = await job_mgr.cancel_job(job_id)
        assert repeat_cancel is False

    await job_mgr.shutdown()


@pytest.mark.anyio
async def test_job_failure_on_adapter_error(db_manager: DatabaseManager) -> None:
    """Verify unhandled adapter errors transition job to FAILED without unhandled exceptions."""
    job_mgr = BenchmarkJobManager(db_manager=db_manager, max_concurrent_jobs=1)

    class FailingAdapter(InferenceEngineAdapter):
        def capabilities(self) -> EngineCapabilities:
            return EngineCapabilities(
                engine_type=EngineType.MOCK,
                execution_mode=ExecutionMode.SIMULATED,
            )

        async def generate_stream(self, request: GenerationRequest):
            raise RuntimeError("Synthetic simulated hardware fault")
            yield  # pragma: no cover

    model = ModelSpec(model_id="fail-model", parameter_count_billion=1.0)
    workload = WorkloadSpec(prompt_tokens=10, output_tokens=10)
    req = BenchmarkSubmissionRequest(
        model=model,
        engine_type=EngineType.MOCK,
        workloads=[workload],
    )

    with patch.object(
        BenchmarkRunner,
        "run",
        side_effect=RuntimeError("Synthetic simulated hardware fault"),
    ):
        job_id = await job_mgr.submit_job(req)
        for _ in range(50):
            st = job_mgr.get_job_status(job_id)
            if st and st.status in ("FAILED", "COMPLETED"):
                break
            await asyncio.sleep(0.05)

        st = job_mgr.get_job_status(job_id)
        assert st is not None
        assert st.status == "FAILED"
        assert "Synthetic simulated hardware fault" in (st.error_message or "")
        assert job_mgr.get_job_result(job_id) is None

    await job_mgr.shutdown()


@pytest.mark.anyio
async def test_persistence_failure_marks_job_failed(db_manager: DatabaseManager) -> None:
    """Verify database write failure during job execution marks job FAILED rather than COMPLETED."""
    job_mgr = BenchmarkJobManager(db_manager=db_manager, max_concurrent_jobs=1)

    model = ModelSpec(model_id="persist-fail-model", parameter_count_billion=1.0)
    workload = WorkloadSpec(prompt_tokens=10, output_tokens=5)
    req = BenchmarkSubmissionRequest(
        model=model,
        engine_type=EngineType.MOCK,
        workloads=[workload],
    )

    # Patch save_benchmark_suite to simulate disk/sqlite failure
    with patch.object(
        EngineRepository,
        "save_benchmark_suite",
        side_effect=RuntimeError("Simulated disk I/O write error"),
    ):
        job_id = await job_mgr.submit_job(req)
        for _ in range(50):
            st = job_mgr.get_job_status(job_id)
            if st and st.status in ("FAILED", "COMPLETED"):
                break
            await asyncio.sleep(0.05)

        st = job_mgr.get_job_status(job_id)
        assert st is not None
        assert st.status == "FAILED"
        assert "Simulated disk I/O write error" in (st.error_message or "")
        assert job_mgr.get_job_result(job_id) is None

    await job_mgr.shutdown()


@pytest.mark.anyio
async def test_shutdown_drains_running_jobs(db_manager: DatabaseManager) -> None:
    """Verify shutdown() cancels and awaits all tasks without hanging."""
    job_mgr = BenchmarkJobManager(db_manager=db_manager, max_concurrent_jobs=2)

    model = ModelSpec(model_id="drain-model", parameter_count_billion=1.0)
    workload = WorkloadSpec(prompt_tokens=100, output_tokens=500)
    req = BenchmarkSubmissionRequest(
        model=model,
        engine_type=EngineType.MOCK,
        workloads=[workload],
    )

    j1 = await job_mgr.submit_job(req)
    j2 = await job_mgr.submit_job(req)

    await asyncio.sleep(0.05)
    # Perform shutdown
    await job_mgr.shutdown()

    s1 = job_mgr.get_job_status(j1)
    s2 = job_mgr.get_job_status(j2)
    assert s1 is not None and s1.status in ("CANCELLED", "COMPLETED")
    assert s2 is not None and s2.status in ("CANCELLED", "COMPLETED")


# -----------------------------------------------------------------------------
# 2. SSE Telemetry Stream and Subscriber Reliability Tests
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_sse_late_subscriber_exits_immediately(db_manager: DatabaseManager) -> None:
    """Verify subscribing to a completed job immediately yields without leaking queues."""
    job_mgr = BenchmarkJobManager(db_manager=db_manager, max_concurrent_jobs=1)

    model = ModelSpec(model_id="fast-model", parameter_count_billion=1.0)
    workload = WorkloadSpec(prompt_tokens=10, output_tokens=2)
    req = BenchmarkSubmissionRequest(
        model=model,
        engine_type=EngineType.MOCK,
        workloads=[workload],
    )

    job_id = await job_mgr.submit_job(req)

    for _ in range(50):
        st = job_mgr.get_job_status(job_id)
        if st and st.status == "COMPLETED":
            break
        await asyncio.sleep(0.05)

    # Late subscriber
    events = []
    async for item in job_mgr.subscribe_events(job_id):
        events.append(item)

    assert len(events) == 0
    # Verify no leaked queues in subscribers
    assert len(job_mgr._subscribers.get(job_id, [])) == 0
    await job_mgr.shutdown()


@pytest.mark.anyio
async def test_sse_subscriber_cleanup_on_early_disconnect(db_manager: DatabaseManager) -> None:
    """Verify disconnecting early from SSE generator cleans up subscriber queue."""
    job_mgr = BenchmarkJobManager(db_manager=db_manager, max_concurrent_jobs=1)

    model = ModelSpec(model_id="stream-model", parameter_count_billion=1.0)
    workload = WorkloadSpec(prompt_tokens=20, output_tokens=50)
    req = BenchmarkSubmissionRequest(
        model=model,
        engine_type=EngineType.MOCK,
        workloads=[workload],
    )

    job_id = await job_mgr.submit_job(req)

    # Connect subscriber and read 1 event then break (simulate client disconnect)
    gen = job_mgr.subscribe_events(job_id)
    try:
        async for _event in gen:
            break  # Disconnect after 1st event
    finally:
        await gen.aclose()

    # Check that subscriber queue was removed
    assert len(job_mgr._subscribers.get(job_id, [])) == 0
    await job_mgr.shutdown()


@pytest.mark.anyio
async def test_sse_slow_consumer_queue_overflow_discard(db_manager: DatabaseManager) -> None:
    """Verify that slow consumer exceeding queue size does not block or crash runner."""
    job_mgr = BenchmarkJobManager(db_manager=db_manager, max_concurrent_jobs=1)

    # Register an artificial queue with maxsize=2
    job_id = "test-job-overflow"
    job_mgr._jobs[job_id] = {
        "job_id": job_id,
        "status": "RUNNING",
        "created_at": datetime.now(UTC),
    }
    small_queue: asyncio.Queue[TokenStreamEvent | None] = asyncio.Queue(maxsize=2)
    job_mgr._subscribers[job_id] = [small_queue]

    # Push 5 events - should not raise QueueFull
    dummy_event = TokenStreamEvent(
        request_id="req-test",
        sequence_index=0,
        event_type=TokenEventType.START,
        timestamp_ns=1000,
        clock_source=ClockSource.MONOTONIC_REAL,
        execution_mode=ExecutionMode.REAL,
    )
    for _ in range(5):
        await job_mgr._broadcast_event(job_id, dummy_event)

    # Queue should be saturated at 2 items without raising error
    assert small_queue.qsize() == 2


# -----------------------------------------------------------------------------
# 3. Route Ordering, Pagination, and Error Boundaries
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_route_ordering_jobs_endpoint_not_captured_by_suite_id(client: AsyncClient) -> None:
    """Verify /api/v1/benchmarks/jobs routes to job listing and is not interpreted as suite_id 'jobs'."""
    resp = await client.get("/api/v1/benchmarks/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.anyio
async def test_pagination_bounds_validation(client: AsyncClient) -> None:
    """Verify pagination parameters enforce ge=1, le=100 for limit and ge=0 for offset."""
    # limit = 0 -> 422
    resp1 = await client.get("/api/v1/benchmarks?limit=0")
    assert resp1.status_code == 422
    data1 = resp1.json()
    assert data1["error"] == "VALIDATION_ERROR"

    # limit = 101 -> 422
    resp2 = await client.get("/api/v1/benchmarks?limit=101")
    assert resp2.status_code == 422
    data2 = resp2.json()
    assert data2["error"] == "VALIDATION_ERROR"

    # offset = -1 -> 422
    resp3 = await client.get("/api/v1/benchmarks?offset=-1")
    assert resp3.status_code == 422
    data3 = resp3.json()
    assert data3["error"] == "VALIDATION_ERROR"

    # Valid pagination: limit=10, offset=0 -> 200
    resp4 = await client.get("/api/v1/benchmarks?limit=10&offset=0")
    assert resp4.status_code == 200


@pytest.mark.anyio
async def test_entity_not_found_standard_envelope(client: AsyncClient) -> None:
    """Verify all resource endpoints return 404 with standard ErrorDetail structure."""
    endpoints = [
        "/api/v1/hardware/non-existent-id",
        "/api/v1/benchmarks/non-existent-suite",
        "/api/v1/benchmarks/jobs/non-existent-job",
        "/api/v1/diagnostics/non-existent-diag",
        "/api/v1/predictions/evaluations/non-existent-eval",
        "/api/v1/optimize/non-existent-opt",
    ]
    for ep in endpoints:
        resp = await client.get(ep)
        assert resp.status_code == 404, f"Endpoint {ep} did not return 404"
        body = resp.json()
        assert body["error"] == "ENTITY_NOT_FOUND"
        assert "not found" in body["message"].lower()


@pytest.mark.anyio
async def test_unhandled_exception_returns_500_without_stack_trace_leak(app_with_mgr) -> None:
    """Verify unhandled exception returns HTTP 500 with ErrorDetail and no leaked stack traces."""
    transport = ASGITransport(app=app_with_mgr, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        # Patch list_benchmark_suites in repository to raise an unexpected KeyError
        with patch.object(
            EngineRepository,
            "list_benchmark_suites",
            side_effect=KeyError("secret_internal_key_that_must_not_leak"),
        ):
            resp = await test_client.get("/api/v1/benchmarks")
            assert resp.status_code == 500
            data = resp.json()
            assert data["error"] == "INTERNAL_SERVER_ERROR"
            assert "internal server error" in data["message"].lower()
            # Verify no stack trace or sensitive error string leaked in response body
            assert "secret_internal_key_that_must_not_leak" not in str(data)
            assert "Traceback" not in str(data)
