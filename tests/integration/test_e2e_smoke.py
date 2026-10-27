"""End-to-end integration smoke tests verifying complete workflows across REST endpoints,
database persistence, SSE streaming, cancellation, diagnostics, and optimization.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.storage.database import DatabaseManager


@pytest.fixture
def temp_db_file(tmp_path: Path) -> Path:
    return tmp_path / "e2e_integration.db"


@pytest.fixture
def temp_db_url(temp_db_file: Path) -> str:
    return f"sqlite:///{temp_db_file}"


@pytest.fixture
def db_manager(temp_db_url: str) -> DatabaseManager:
    mgr = DatabaseManager(database_url=temp_db_url)
    mgr.init_db()
    return mgr


@pytest.fixture
async def client(db_manager: DatabaseManager) -> AsyncGenerator[AsyncClient, None]:
    app = create_app(db_manager=db_manager)
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://testserver") as ac,
        app.router.lifespan_context(app),
    ):
        yield ac


# -----------------------------------------------------------------------------
# 1. Health and Readiness Verification
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_e2e_health_endpoints(client: AsyncClient) -> None:
    """Verify both root /health and versioned /api/v1/health respond with 200 OK."""
    # Root level probe
    res_root = await client.get("/health")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["status"] == "ok"
    assert data_root["database"] == "connected"

    # Namespaced probe
    res_v1 = await client.get("/api/v1/health")
    assert res_v1.status_code == 200
    data_v1 = res_v1.json()
    assert data_v1["status"] == "ok"
    assert data_v1["database"] == "connected"


# -----------------------------------------------------------------------------
# 2. Hardware Discovery and Catalog Verification
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_e2e_hardware_discovery_and_catalog(client: AsyncClient) -> None:
    """Verify discovery profiler execution and authoritative GPU reference catalog."""
    # Run hardware discovery
    res_hw = await client.get("/api/v1/hardware?discover=true")
    assert res_hw.status_code == 200
    hw = res_hw.json()
    assert "report_id" in hw
    assert "host" in hw
    assert "cpu" in hw
    assert "memory" in hw
    assert "telemetry" in hw

    # Check GPU catalog
    res_cat = await client.get("/api/v1/hardware/catalog")
    assert res_cat.status_code == 200
    cat = res_cat.json()
    assert isinstance(cat, list)
    assert len(cat) > 0
    # Sourced from whitepapers
    for entry in cat:
        assert entry["provenance"] == "HEURISTIC"
        assert "theoretical_memory_bandwidth_gbps" in entry
        assert "compute" in entry


# -----------------------------------------------------------------------------
# 3. Benchmark Execution, SSE Streaming, Polling, and Persistence Workflow
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_e2e_benchmark_execution_and_telemetry_flow(client: AsyncClient) -> None:
    """Verify full benchmark lifecycle: submit -> SSE stream -> status poll -> DB persistence."""
    payload = {
        "model": {
            "model_id": "meta-llama/Llama-3-8B",
            "family": "llama",
            "parameter_count_billion": 8.0,
            "quantization": "none",
        },
        "engine_type": "mock",
        "engine_config": {
            "engine_type": "mock",
            "extra_args": {
                "virtual_time_mode": False,
                "prefill_delay_ns": 20_000_000,
                "inter_token_delay_ns": 10_000_000,
            },
        },
        "workloads": [
            {
                "prompt_tokens": 32,
                "output_tokens": 10,
            }
        ],
        "concurrency": 1,
        "warmup_requests": 0,
        "sample_memory": True,
    }

    # Submit job
    submit_res = await client.post("/api/v1/benchmarks/run", json=payload)
    assert submit_res.status_code == 202
    job_data = submit_res.json()
    job_id = job_data["job_id"]
    assert job_id.startswith("job-")

    # Connect to SSE stream concurrently while job executes
    stream_events: list[str] = []

    async def read_stream():
        async with client.stream("GET", f"/api/v1/benchmarks/jobs/{job_id}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    stream_events.append(line)

    stream_task = asyncio.create_task(read_stream())

    # Poll status until terminal state
    suite_id: str | None = None
    for _ in range(50):
        status_res = await client.get(f"/api/v1/benchmarks/jobs/{job_id}")
        assert status_res.status_code == 200
        status_info = status_res.json()
        current_status = status_info["job"]["status"]
        if current_status == "COMPLETED":
            suite_id = status_info["job"]["suite_id"]
            assert status_info["result"] is not None
            assert status_info["result"]["total_requests"] == 1
            break
        elif current_status in ("FAILED", "CANCELLED"):
            pytest.fail(f"Job terminated unexpectedly with {current_status}")
        await asyncio.sleep(0.05)

    await stream_task
    assert len(stream_events) > 0, "Expected at least one token stream event from mock adapter"

    # Poll status until terminal state
    suite_id: str | None = None
    for _ in range(50):
        status_res = await client.get(f"/api/v1/benchmarks/jobs/{job_id}")
        assert status_res.status_code == 200
        status_info = status_res.json()
        current_status = status_info["job"]["status"]
        if current_status == "COMPLETED":
            suite_id = status_info["job"]["suite_id"]
            assert status_info["result"] is not None
            assert status_info["result"]["total_requests"] == 1
            break
        elif current_status in ("FAILED", "CANCELLED"):
            pytest.fail(f"Job terminated unexpectedly with {current_status}")
        await asyncio.sleep(0.05)

    assert suite_id is not None, "Job did not complete in time"

    # Verify persisted in database
    suite_res = await client.get(f"/api/v1/benchmarks/{suite_id}")
    assert suite_res.status_code == 200
    persisted = suite_res.json()
    assert persisted["suite_id"] == suite_id
    assert persisted["model_id"] == "meta-llama/Llama-3-8B"
    assert persisted["total_requests"] == 1

    # Verify listed in historical benchmark suites
    list_res = await client.get("/api/v1/benchmarks?limit=10")
    assert list_res.status_code == 200
    history = list_res.json()
    assert history["total"] >= 1
    found = any(s["suite_id"] == suite_id for s in history["items"])
    assert found


# -----------------------------------------------------------------------------
# 4. Job Cancellation Verification
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_e2e_benchmark_cancellation(client: AsyncClient) -> None:
    """Verify in-flight job cancellation transitions job to CANCELLED state."""
    # Submit a job with multiple workloads
    payload = {
        "model": {"model_id": "meta-llama/Llama-3-8B", "parameter_count_billion": 8.0},
        "engine_type": "mock",
        "workloads": [{"prompt_tokens": 128, "output_tokens": 64}] * 5,
        "concurrency": 1,
    }

    sub_res = await client.post("/api/v1/benchmarks/run", json=payload)
    job_id = sub_res.json()["job_id"]

    # Cancel immediately
    cancel_res = await client.post(f"/api/v1/benchmarks/jobs/{job_id}/cancel")
    assert cancel_res.status_code == 200

    # Poll status: must be CANCELLED or already COMPLETED (never stuck in RUNNING)
    poll_res = await client.get(f"/api/v1/benchmarks/jobs/{job_id}")
    status = poll_res.json()["job"]["status"]
    assert status in ("CANCELLED", "COMPLETED")

    # Second cancel on terminal job must report already terminal
    second_cancel = await client.post(f"/api/v1/benchmarks/jobs/{job_id}/cancel")
    assert second_cancel.status_code == 200
    assert second_cancel.json()["cancelled"] is False


# -----------------------------------------------------------------------------
# 5. Bottleneck Diagnostics Workflow
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_e2e_diagnostics_workflow(client: AsyncClient) -> None:
    """Verify diagnostic evaluation on a completed benchmark suite via /api/v1/diagnostics/run."""
    # 1. Create a benchmark suite first
    bench_payload = {
        "model": {"model_id": "meta-llama/Llama-3-8B", "parameter_count_billion": 8.0},
        "engine_type": "mock",
        "workloads": [{"prompt_tokens": 64, "output_tokens": 32}],
        "concurrency": 1,
    }
    sub = await client.post("/api/v1/benchmarks/run", json=bench_payload)
    job_id = sub.json()["job_id"]

    suite_id = None
    for _ in range(50):
        st = await client.get(f"/api/v1/benchmarks/jobs/{job_id}")
        if st.json()["job"]["status"] == "COMPLETED":
            suite_id = st.json()["job"]["suite_id"]
            break
        await asyncio.sleep(0.05)

    assert suite_id is not None

    # 2. Run diagnostics
    diag_res = await client.post(
        "/api/v1/diagnostics/run",
        json={
            "suite_id": suite_id,
            "model": {
                "model_id": "meta-llama/Llama-3-8B",
                "parameter_count_billion": 8.0,
            },
        },
    )
    assert diag_res.status_code == 200
    report = diag_res.json()
    report_id = report["report_id"]

    assert report["model_id"] == "meta-llama/Llama-3-8B"
    assert "findings" in report
    assert "resource_estimate" in report
    assert report["resource_estimate"]["parameter_storage_mb"] > 0

    # 3. Retrieve report by ID
    get_res = await client.get(f"/api/v1/diagnostics/{report_id}")
    assert get_res.status_code == 200
    assert get_res.json()["report_id"] == report_id

    # 4. List diagnostics
    list_res = await client.get("/api/v1/diagnostics")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1


# -----------------------------------------------------------------------------
# 6. Empirical Prediction Evaluation Workflow
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_e2e_prediction_evaluation_workflow(client: AsyncClient) -> None:
    """Verify recording and querying empirical prediction evaluations."""
    eval_payload = {
        "target": "ttft_ms",
        "predictor_id": "ridge-test",
        "predictor_type": "RidgeRegressionPredictor",
        "split_strategy": "RANDOM",
        "sample_count": 25,
        "train_sample_count": 75,
        "test_sample_count": 25,
        "unit": "ms",
        "mae": 1.25,
        "rmse": 1.85,
        "mape_percent": 7.4,
        "baseline_mae": 2.50,
        "baseline_rmse": 3.10,
        "is_synthetic": False,
        "execution_mode": "REAL",
        "is_statistically_sufficient": True,
        "notes": "E2E verification sample evaluation",
    }

    create_res = await client.post("/api/v1/predictions/evaluations", json=eval_payload)
    assert create_res.status_code == 201
    eval_id = create_res.json()["evaluation_id"]

    # Retrieve by ID
    get_res = await client.get(f"/api/v1/predictions/evaluations/{eval_id}")
    assert get_res.status_code == 200
    saved = get_res.json()
    assert saved["target"] == "ttft_ms"
    assert saved["mae"] == 1.25
    assert saved["is_statistically_sufficient"] is True

    # List evaluations
    list_res = await client.get("/api/v1/predictions/evaluations?target=ttft_ms")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1


# -----------------------------------------------------------------------------
# 7. Pareto Optimization Workflow
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_e2e_optimization_workflow(client: AsyncClient) -> None:
    """Verify configuration optimization, Pareto frontier calculation, and full detail retrieval."""
    opt_payload = {
        "workload": {
            "prompt_tokens": 128,
            "output_tokens": 64,
        },
        "search_space": {
            "models": [
                {
                    "model_id": "meta-llama/Llama-3-8B",
                    "parameter_count_billion": 8.0,
                }
            ],
            "quantizations": ["fp16", "int8", "int4"],
            "concurrencies": [1, 2],
            "engines": ["mock"],
            "tensor_parallel_sizes": [1],
        },
        "objectives": ["minimize_ttft", "maximize_throughput"],
        "selection_policy": "balanced_knee_point",
    }

    opt_res = await client.post("/api/v1/optimize/recommend", json=opt_payload)
    assert opt_res.status_code == 200
    res_data = opt_res.json()
    opt_id = res_data["request_id"]

    assert res_data["total_candidates_evaluated"] > 0
    assert "pareto_frontier" in res_data
    assert "feasible_candidates_count" in res_data

    # Check history summary listing
    hist_res = await client.get("/api/v1/optimize/history")
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert hist["total"] >= 1
    found = any(h["optimization_id"] == opt_id for h in hist["items"])
    assert found

    # Retrieve full detail by ID
    detail_res = await client.get(f"/api/v1/optimize/{opt_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["result"]["request_id"] == opt_id
    assert len(detail["result"]["pareto_frontier"]) == len(res_data["pareto_frontier"])


# -----------------------------------------------------------------------------
# 8. Error Handling and Edge Case Verification
# -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_e2e_error_handling_contracts(client: AsyncClient) -> None:
    """Verify standard error payload format across 404, 422, and 400 cases."""
    # 404 on missing benchmark suite
    res_404 = await client.get("/api/v1/benchmarks/suite-non-existent")
    assert res_404.status_code == 404
    err_404 = res_404.json()
    assert err_404["error"] == "ENTITY_NOT_FOUND"
    assert "not found" in err_404["message"].lower()

    # 404 on missing diagnostics report
    res_diag_404 = await client.get("/api/v1/diagnostics/diag-missing-id")
    assert res_diag_404.status_code == 404

    # 422 on invalid benchmark payload (negative tokens)
    bad_payload = {
        "model": {"model_id": "test-model"},
        "workloads": [{"prompt_tokens": -5, "output_tokens": 10}],
    }
    res_422 = await client.post("/api/v1/benchmarks/run", json=bad_payload)
    assert res_422.status_code == 422
    err_422 = res_422.json()
    assert err_422["error"] == "VALIDATION_ERROR"
