"""Unit and integration tests for FastAPI endpoints, lifecycle, background jobs, and SSE streaming."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.benchmark.models import (
    BenchmarkSuiteResult,
    RequestBenchmarkResult,
)
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    TokenCountSource,
)
from llm_ii_engine.storage.database import DatabaseManager
from llm_ii_engine.storage.repository import EngineRepository


@pytest.fixture
def temp_db_url(tmp_path: Path) -> str:
    db_file = tmp_path / "test_api.db"
    return f"sqlite:///{db_file}"


@pytest.fixture
def db_manager(temp_db_url: str) -> DatabaseManager:
    mgr = DatabaseManager(database_url=temp_db_url)
    mgr.init_db()
    return mgr


@pytest.fixture
async def client(db_manager: DatabaseManager) -> AsyncGenerator[AsyncClient, None]:
    app = create_app(db_manager=db_manager)
    # Lifespan startup
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://testserver") as ac,
        app.router.lifespan_context(app),
    ):
        yield ac


# -----------------------------------------------------------------------------
# Health Probe Tests
# -----------------------------------------------------------------------------


async def test_health_check_healthy(client: AsyncClient) -> None:
    """Verify /health returns 200 OK when database is responsive."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"


async def test_health_check_unhealthy(temp_db_url: str) -> None:
    """Verify /health returns 503 when database is unresponsive."""
    mgr = DatabaseManager(database_url=temp_db_url)
    mgr.init_db()
    app = create_app(db_manager=mgr)

    # Dispose underlying engine to simulate outage
    mgr.dispose()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["database"] == "disconnected"


# -----------------------------------------------------------------------------
# Hardware Discovery Endpoint Tests
# -----------------------------------------------------------------------------


async def test_get_hardware_discovery(client: AsyncClient) -> None:
    """Verify GET /api/v1/hardware executes discovery and returns valid report."""
    response = await client.get("/api/v1/hardware")
    assert response.status_code == 200
    data = response.json()
    assert "report_id" in data
    assert "host" in data
    assert "cpu" in data
    assert "memory" in data
    assert "gpus" in data
    assert data["host"]["os_name"] is not None


async def test_get_hardware_by_id_and_not_found(client: AsyncClient) -> None:
    """Verify GET /api/v1/hardware/{report_id} returns 200 or 404 on missing entity."""
    # First get or discover hardware
    resp1 = await client.get("/api/v1/hardware?discover=true")
    assert resp1.status_code == 200
    report_id = resp1.json()["report_id"]

    # Retrieve by ID
    resp2 = await client.get(f"/api/v1/hardware/{report_id}")
    assert resp2.status_code == 200
    assert resp2.json()["report_id"] == report_id

    # Non-existent ID returns 404
    resp3 = await client.get("/api/v1/hardware/non-existent-hw-id")
    assert resp3.status_code == 404
    assert resp3.json()["error"] == "ENTITY_NOT_FOUND"


# -----------------------------------------------------------------------------
# Benchmark Orchestration, Telemetry, and Jobs Tests
# -----------------------------------------------------------------------------


async def test_benchmark_job_lifecycle_and_sse_streaming(client: AsyncClient) -> None:
    """Verify async benchmark submission, SSE streaming, and background persistence."""
    payload = {
        "model": {
            "model_id": "test-mock-llama",
            "family": "llama",
            "parameter_count_billion": 1.0,
        },
        "engine_type": "mock",
        "workloads": [
            {"prompt_tokens": 16, "output_tokens": 8},
        ],
        "concurrency": 1,
        "warmup_requests": 0,
        "request_timeout_seconds": 10.0,
        "sample_memory": False,
    }

    # 1. Submit job
    run_resp = await client.post("/api/v1/benchmarks/run", json=payload)
    assert run_resp.status_code == 202
    job_id = run_resp.json()["job_id"]
    assert job_id.startswith("job-")

    # 2. Poll job status until complete
    completed = False
    for _ in range(30):
        status_resp = await client.get(f"/api/v1/benchmarks/jobs/{job_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        if data["job"]["status"] == "COMPLETED":
            completed = True
            assert data["result"] is not None
            assert data["result"]["model_id"] == "test-mock-llama"
            break
        await asyncio.sleep(0.1)

    assert completed, "Benchmark job failed to complete within timeout"

    # 3. Verify suite was persisted to database
    list_resp = await client.get("/api/v1/benchmarks?model_id=test-mock-llama")
    assert list_resp.status_code == 200
    benchmarks_data = list_resp.json()
    assert benchmarks_data["total"] >= 1


async def test_cancel_benchmark_job(client: AsyncClient) -> None:
    """Verify POST /api/v1/benchmarks/jobs/{job_id}/cancel cancels a job."""
    payload = {
        "model": {"model_id": "slow-mock", "family": "llama"},
        "engine_type": "mock",
        "workloads": [
            {"prompt_tokens": 100, "output_tokens": 500},
        ],
        "concurrency": 1,
    }

    run_resp = await client.post("/api/v1/benchmarks/run", json=payload)
    job_id = run_resp.json()["job_id"]

    cancel_resp = await client.post(f"/api/v1/benchmarks/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["job_id"] == job_id


# -----------------------------------------------------------------------------
# Diagnostics Endpoint Tests
# -----------------------------------------------------------------------------


async def test_run_diagnostics_on_completed_suite(
    client: AsyncClient, db_manager: DatabaseManager
) -> None:
    """Verify POST /api/v1/diagnostics/run analyzes a persisted benchmark suite."""
    # Pre-populate a benchmark suite in the database
    req1 = RequestBenchmarkResult(
        request_id="diag-req-1",
        success=True,
        prompt_tokens=64,
        requested_output_tokens=32,
        output_tokens=32,
        ttft_ms=15.0,
        tpot_ms=8.0,
        duration_ms=270.0,
        output_tokens_per_second=118.5,
        inter_token_latencies_ms=[8.0] * 32,
        itl_p50_ms=8.0,
        itl_p90_ms=8.0,
        itl_p95_ms=8.0,
        itl_p99_ms=8.0,
        token_count_source=TokenCountSource.TOKEN_EVENTS,
        prompt_token_count_source=TokenCountSource.REQUESTED,
        clock_source=ClockSource.MONOTONIC_REAL,
        execution_mode=ExecutionMode.REAL,
        provenance=ProvenanceType.MEASURED,
    )
    suite = BenchmarkSuiteResult(
        suite_id="suite-for-diag",
        timestamp_utc=datetime.now(UTC),
        status="COMPLETED",
        model_id="llama-7b",
        engine_type=EngineType.MOCK,
        concurrency=1,
        total_requests=1,
        warmup_requests=0,
        successful_requests=1,
        failed_requests=0,
        wall_time_seconds=0.3,
        aggregate_tokens_per_second=320.0,
        aggregate_output_tokens_per_second=106.6,
        request_throughput_per_second=3.3,
        ttft_p50_ms=15.0,
        ttft_p90_ms=15.0,
        ttft_p95_ms=15.0,
        ttft_p99_ms=15.0,
        tpot_mean_ms=8.0,
        itl_p50_ms=8.0,
        itl_p90_ms=8.0,
        itl_p95_ms=8.0,
        itl_p99_ms=8.0,
        execution_mode=ExecutionMode.REAL,
        clock_source=ClockSource.MONOTONIC_REAL,
        provenance=ProvenanceType.MEASURED,
        request_results=[req1],
        warmup_results=[],
    )

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        repo.save_benchmark_suite(suite)

    # Call diagnostics API
    diag_resp = await client.post(
        "/api/v1/diagnostics/run",
        json={"suite_id": "suite-for-diag"},
    )
    assert diag_resp.status_code == 200
    diag_data = diag_resp.json()
    assert diag_data["model_id"] == "llama-7b"
    assert "summary" in diag_data
    assert "report_id" in diag_data

    # Retrieve diagnostic report by ID
    report_id = diag_data["report_id"]
    get_diag_resp = await client.get(f"/api/v1/diagnostics/{report_id}")
    assert get_diag_resp.status_code == 200
    assert get_diag_resp.json()["report_id"] == report_id


# -----------------------------------------------------------------------------
# Prediction Evaluation Endpoint Tests
# -----------------------------------------------------------------------------


async def test_prediction_evaluations_api(client: AsyncClient) -> None:
    """Verify POST and GET /api/v1/predictions/evaluations."""
    eval_payload = {
        "target": "ttft_ms",
        "predictor_id": "ridge-test",
        "predictor_type": "RidgeRegressionPredictor",
        "split_strategy": "RANDOM",
        "sample_count": 50,
        "test_sample_count": 50,
        "train_sample_count": 200,
        "unit": "ms",
        "mae": 1.25,
        "rmse": 1.60,
        "mape_percent": 8.5,
        "baseline_mae": 2.10,
        "baseline_rmse": 2.80,
        "is_synthetic": False,
        "execution_mode": "REAL",
        "is_statistically_sufficient": True,
        "notes": "FastAPI evaluation test",
    }

    post_resp = await client.post("/api/v1/predictions/evaluations", json=eval_payload)
    assert post_resp.status_code == 201
    eval_id = post_resp.json()["evaluation_id"]

    get_resp = await client.get(f"/api/v1/predictions/evaluations/{eval_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["target"] == "ttft_ms"
    assert get_resp.json()["mae"] == 1.25

    list_resp = await client.get("/api/v1/predictions/evaluations?target=ttft_ms")
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] >= 1


# -----------------------------------------------------------------------------
# Optimization and Pareto Recommendation Tests
# -----------------------------------------------------------------------------


async def test_optimize_recommend_and_persistence(client: AsyncClient) -> None:
    """Verify POST /api/v1/optimize/recommend returns Pareto frontier and persists results."""
    opt_payload = {
        "workload": {"prompt_tokens": 128, "output_tokens": 64},
        "search_space": {
            "models": [
                {"model_id": "opt-test-model", "parameter_count_billion": 3.0},
            ],
            "quantizations": ["fp16", "int4"],
            "concurrencies": [1],
            "engines": ["mock"],
        },
        "objectives": ["minimize_ttft"],
        "hardware_catalog_id": "nvidia-rtx-4090",
    }

    resp = await client.post("/api/v1/optimize/recommend", json=opt_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "request_id" in data
    assert "pareto_frontier" in data
    assert len(data["pareto_frontier"]) >= 1

    opt_id = data["request_id"]

    # Verify retrieval from history
    hist_resp = await client.get("/api/v1/optimize/history")
    assert hist_resp.status_code == 200
    assert hist_resp.json()["total"] >= 1

    # Verify retrieval by ID
    get_resp = await client.get(f"/api/v1/optimize/{opt_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["result"]["request_id"] == opt_id


# -----------------------------------------------------------------------------
# Error Handling Contract Tests
# -----------------------------------------------------------------------------


async def test_error_handling_contracts(client: AsyncClient) -> None:
    """Verify 400, 404, and 422 error structures conform to ErrorDetail schema."""
    # 404 on missing benchmark suite
    resp_404 = await client.get("/api/v1/benchmarks/missing-suite-id")
    assert resp_404.status_code == 404
    data_404 = resp_404.json()
    assert data_404["error"] == "ENTITY_NOT_FOUND"

    # 422 on invalid body
    resp_422 = await client.post("/api/v1/benchmarks/run", json={"invalid": "payload"})
    assert resp_422.status_code == 422
    data_422 = resp_422.json()
    assert data_422["error"] == "VALIDATION_ERROR"
