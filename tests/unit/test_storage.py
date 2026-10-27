"""Unit tests for SQLite persistence, transactions, relationships, and repository mappings."""

import contextlib
import os
import tempfile
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from llm_ii_engine.analysis.models import (
    DiagnosticFinding,
    DiagnosticReport,
)
from llm_ii_engine.benchmark.models import (
    BenchmarkSuiteResult,
    MemoryWatermarkResult,
    RequestBenchmarkResult,
)
from llm_ii_engine.core.enums import (
    ClockSource,
    DiagnosticCategory,
    DiagnosticConfidence,
    DiagnosticSeverity,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    QuantizationType,
    TokenCountSource,
)
from llm_ii_engine.core.schemas import ModelSpec, Observation, WorkloadSpec
from llm_ii_engine.discovery.models import (
    CPUInfo,
    DetectedGPU,
    HardwareCapabilityReport,
    HostSystemInfo,
    MemoryInfo,
    TelemetryTools,
)
from llm_ii_engine.optimizer.models import (
    CandidateConfig,
    CandidateEvaluation,
    EvidenceStatus,
    FeasibilityStatus,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
    SearchSpace,
)
from llm_ii_engine.prediction.models import (
    EvaluationResult,
    PredictionTarget,
)
from llm_ii_engine.storage.database import (
    CURRENT_SCHEMA_VERSION,
    DatabaseManager,
)
from llm_ii_engine.storage.models import (
    HardwareReportORM,
    MetricObservationORM,
    RequestBenchmarkORM,
    SchemaVersionORM,
)
from llm_ii_engine.storage.repository import (
    EngineRepository,
    EntityAlreadyExistsError,
)


@pytest.fixture
def temp_db_path() -> str:
    """Fixture providing a temporary SQLite file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        with contextlib.suppress(PermissionError):
            os.remove(path)


@pytest.fixture
def db_manager(temp_db_path: str) -> DatabaseManager:
    """Fixture providing an initialized DatabaseManager on a temporary file."""
    manager = DatabaseManager(database_url=f"sqlite:///{temp_db_path}")
    manager.init_db()
    yield manager
    manager.dispose()


def test_database_initialization_and_schema_version(db_manager: DatabaseManager) -> None:
    """Verify fresh database initializes schema_version table to version 1 and health check passes."""
    assert db_manager.check_health() is True

    with db_manager.get_session() as session:
        stmt = select(SchemaVersionORM)
        versions = session.scalars(stmt).all()
        assert len(versions) == 1
        assert versions[0].version == CURRENT_SCHEMA_VERSION


def test_higher_schema_version_fails_loudly(db_manager: DatabaseManager) -> None:
    """Verify that an incompatible future database version raises RuntimeError."""
    with db_manager.get_session() as session:
        # Simulate a future schema version written by newer code
        session.add(
            SchemaVersionORM(
                version=99,
                applied_at=datetime.now(UTC),
                description="Future migration",
            )
        )
        session.commit()

    with pytest.raises(RuntimeError, match="is higher than supported"):
        db_manager.init_db(check_version=True)


def test_sqlite_foreign_key_enforcement(db_manager: DatabaseManager) -> None:
    """Verify SQLite foreign key pragma rejects child records with non-existent parent."""
    with pytest.raises(IntegrityError), db_manager.get_session() as session:
        # Attempt to insert a RequestBenchmarkORM with non-existent suite_id
        orphan_req = RequestBenchmarkORM(
            request_id="req-orphan",
            suite_id="non-existent-suite",
            success=True,
            is_warmup=False,
            prompt_tokens=10,
            output_tokens=20,
            duration_ms=100.0,
            token_count_source="token_events",
            prompt_token_count_source="requested",
            clock_source="monotonic_real",
            execution_mode="real",
            provenance="measured",
        )
        session.add(orphan_req)
        session.commit()


def test_transaction_rollback_on_failure(db_manager: DatabaseManager) -> None:
    """Verify that a failed operation within a session context rolls back all changes."""
    with (
        pytest.raises(ValueError, match="Intentional failure"),
        db_manager.get_session() as session,
    ):
        hw = HardwareReportORM(
            report_id="hw-rollback-test",
            created_at=datetime.now(UTC),
            is_simulated=False,
            execution_mode="REAL",
            os_name="Linux",
            os_version="6.5.0",
            os_release="6.5.0-generic",
            os_arch="x86_64",
            python_version="3.12.0",
            cpu_model="AMD EPYC",
            cpu_physical_cores=8,
            cpu_logical_cores=16,
            total_ram_bytes=32 * 1024**3,
            available_ram_bytes=24 * 1024**3,
        )
        session.add(hw)
        session.flush()
        raise ValueError("Intentional failure to test rollback")

    # Confirm record was not written
    with db_manager.get_session() as session:
        retrieved = session.get(HardwareReportORM, "hw-rollback-test")
        assert retrieved is None


def test_hardware_report_crud_and_restart_persistence(temp_db_path: str) -> None:
    """Verify HardwareCapabilityReport persists across database manager restarts."""
    mgr1 = DatabaseManager(database_url=f"sqlite:///{temp_db_path}")
    mgr1.init_db()

    report = HardwareCapabilityReport(
        report_id="hw-restart-01",
        timestamp_utc=datetime.now(UTC),
        is_simulated=False,
        execution_mode=ExecutionMode.REAL,
        host=HostSystemInfo(
            os_name="Windows",
            os_version="11",
            os_release="11 Pro",
            architecture="AMD64",
            python_version="3.12.10",
        ),
        cpu=CPUInfo(
            model_name="Intel Core i9-14900K",
            vendor="Intel",
            physical_cores=24,
            logical_cores=32,
            frequency_mhz=3200.0,
        ),
        memory=MemoryInfo(
            total_bytes=64 * 1024**3,
            available_bytes=48 * 1024**3,
        ),
        gpus=[
            DetectedGPU(
                index=0,
                name="NVIDIA GeForce RTX 4090",
                vendor="NVIDIA",
                total_memory_mb=24576.0,
                compute_capability="8.9",
                driver_version="550.54.14",
                detection_source="nvidia-smi",
            )
        ],
        telemetry=TelemetryTools(has_nvidia_smi=True),
        matched_catalog_specs=[],
        missing_or_incomplete_fields=[],
        warnings=[],
    )

    with mgr1.get_session() as session:
        repo = EngineRepository(session)
        repo.save_hardware_report(report)

    mgr1.dispose()

    # Re-open database with fresh DatabaseManager
    mgr2 = DatabaseManager(database_url=f"sqlite:///{temp_db_path}")
    with mgr2.get_session() as session:
        repo2 = EngineRepository(session)
        retrieved = repo2.get_hardware_report("hw-restart-01")
        assert retrieved is not None
        assert retrieved.report_id == "hw-restart-01"
        assert retrieved.cpu.model_name == "Intel Core i9-14900K"
        assert len(retrieved.gpus) == 1
        assert retrieved.gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert retrieved.telemetry.has_nvidia_smi is True

    mgr2.dispose()


def test_duplicate_hardware_insert_raises_error(db_manager: DatabaseManager) -> None:
    """Verify attempting to save duplicate hardware ID raises EntityAlreadyExistsError."""
    report = HardwareCapabilityReport(
        report_id="hw-dup-test",
        timestamp_utc=datetime.now(UTC),
        is_simulated=False,
        execution_mode=ExecutionMode.REAL,
        host=HostSystemInfo(
            os_name="Linux",
            os_version="6.5.0",
            os_release="Ubuntu 22.04",
            architecture="x86_64",
            python_version="3.12.0",
        ),
        cpu=CPUInfo(model_name="CPU", physical_cores=4, logical_cores=8),
        memory=MemoryInfo(total_bytes=16 * 1024**3, available_bytes=12 * 1024**3),
        gpus=[],
        telemetry=TelemetryTools(),
        matched_catalog_specs=[],
        missing_or_incomplete_fields=[],
        warnings=[],
    )

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        repo.save_hardware_report(report)

    with pytest.raises(EntityAlreadyExistsError), db_manager.get_session() as session:
        repo = EngineRepository(session)
        repo.save_hardware_report(report)


def test_benchmark_suite_persistence_with_observations(db_manager: DatabaseManager) -> None:
    """Verify full benchmark suite persists requests and discrete metric observations."""
    req1 = RequestBenchmarkResult(
        request_id="req-001",
        success=True,
        prompt_tokens=128,
        requested_output_tokens=64,
        output_tokens=64,
        ttft_ms=25.0,
        tpot_ms=10.0,
        duration_ms=665.0,
        output_tokens_per_second=100.0,
        inter_token_latencies_ms=[10.0] * 64,
        itl_p50_ms=10.0,
        itl_p90_ms=10.0,
        itl_p95_ms=10.0,
        itl_p99_ms=10.0,
        chunk_count=64,
        chunk_latencies_ms=[10.0] * 64,
        chunk_latency_p50_ms=10.0,
        token_count_source=TokenCountSource.TOKEN_EVENTS,
        prompt_token_count_source=TokenCountSource.REQUESTED,
        clock_source=ClockSource.MONOTONIC_REAL,
        execution_mode=ExecutionMode.REAL,
        provenance=ProvenanceType.MEASURED,
    )

    suite = BenchmarkSuiteResult(
        suite_id="bench-persist-01",
        timestamp_utc=datetime.now(UTC),
        status="COMPLETED",
        model_id="llama-7b",
        engine_type=EngineType.MOCK,
        concurrency=1,
        total_requests=1,
        warmup_requests=0,
        successful_requests=1,
        failed_requests=0,
        wall_time_seconds=0.7,
        aggregate_tokens_per_second=300.0,
        aggregate_output_tokens_per_second=100.0,
        request_throughput_per_second=1.5,
        ttft_p50_ms=25.0,
        ttft_p90_ms=25.0,
        ttft_p95_ms=25.0,
        ttft_p99_ms=25.0,
        tpot_mean_ms=10.0,
        itl_p50_ms=10.0,
        itl_p90_ms=10.0,
        itl_p95_ms=10.0,
        itl_p99_ms=10.0,
        memory_watermark=MemoryWatermarkResult(
            peak_process_rss_mb=512.0,
            peak_gpu_vram_mb=4096.0,
            sample_count=5,
            sample_interval_ms=100.0,
            telemetry_source="mock",
            is_telemetry_available=True,
        ),
        execution_mode=ExecutionMode.REAL,
        clock_source=ClockSource.MONOTONIC_REAL,
        provenance=ProvenanceType.MEASURED,
        request_results=[req1],
        warmup_results=[],
        warnings=["Test suite run warning"],
    )

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        repo.save_benchmark_suite(suite)

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        loaded = repo.get_benchmark_suite("bench-persist-01")
        assert loaded is not None
        assert loaded.suite_id == "bench-persist-01"
        assert loaded.aggregate_output_tokens_per_second == 100.0
        assert len(loaded.request_results) == 1
        assert loaded.request_results[0].request_id == "req-001"
        assert loaded.request_results[0].ttft_ms == 25.0

        # Verify discrete observations were extracted and written
        obs_stmt = select(MetricObservationORM).where(
            MetricObservationORM.suite_id == "bench-persist-01"
        )
        obs = session.scalars(obs_stmt).all()
        assert len(obs) > 0
        names = {o.name for o in obs}
        assert "ttft_p50_ms" in names
        assert "aggregate_output_tokens_per_second" in names
        assert all(o.provenance == "MEASURED" for o in obs)


def test_diagnostic_report_and_findings_persistence(db_manager: DatabaseManager) -> None:
    """Verify DiagnosticReport persists with relational DiagnosticFinding records."""
    finding = DiagnosticFinding(
        finding_id="find-01",
        category=DiagnosticCategory.EFFICIENCY_BOTTLENECK,
        severity=DiagnosticSeverity.CRITICAL,
        title="Memory Bandwidth Saturated",
        message="Low operational intensity in decode phase causes memory bottleneck.",
        confidence=DiagnosticConfidence.HIGH,
        confidence_rationale="Observed throughput closely tracks peak memory bandwidth.",
        supporting_observations=[
            Observation(
                value=0.45,
                unit="FLOPs/Byte",
                provenance=ProvenanceType.MEASURED,
                execution_mode=ExecutionMode.REAL,
                clock_source=ClockSource.MONOTONIC_REAL,
                description="operational_intensity",
            )
        ],
        assumptions=["Standard autoregressive generation"],
        limitations=["No hardware counter tracing"],
        suggested_experiments=["Quantize weights to INT4"],
    )

    report = DiagnosticReport(
        report_id="diag-01",
        model_id="llama-7b",
        engine_type=EngineType.MOCK,
        execution_mode=ExecutionMode.REAL,
        findings=[finding],
        summary="Evaluation shows memory bandwidth bottleneck in decode.",
        timestamp_utc=datetime.now(UTC),
    )

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        repo.save_diagnostic_report(report)

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        loaded = repo.get_diagnostic_report("diag-01")
        assert loaded is not None
        assert loaded.report_id == "diag-01"
        assert len(loaded.findings) == 1
        assert loaded.findings[0].category == DiagnosticCategory.EFFICIENCY_BOTTLENECK
        assert loaded.findings[0].confidence == DiagnosticConfidence.HIGH
        assert loaded.findings[0].suggested_experiments == ["Quantize weights to INT4"]
        assert len(loaded.findings[0].supporting_observations) == 1
        assert loaded.findings[0].supporting_observations[0].value == 0.45


def test_prediction_evaluation_persistence(db_manager: DatabaseManager) -> None:
    """Verify EvaluationResult persists and is queryable."""
    eval_res = EvaluationResult(
        target=PredictionTarget.TTFT_MS,
        predictor_id="ridge-01",
        predictor_type="RidgeRegressionPredictor",
        split_strategy="RANDOM",
        sample_count=20,
        test_sample_count=20,
        train_sample_count=80,
        unit="ms",
        mae=1.4,
        rmse=1.7,
        mape_percent=9.0,
        baseline_mae=2.5,
        baseline_rmse=3.0,
        is_synthetic=False,
        execution_mode=ExecutionMode.REAL,
        sufficiency_threshold=10,
        is_statistically_sufficient=True,
        notes="Evaluated with 80/20 train/test split",
    )

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        eval_id = repo.save_prediction_evaluation(eval_res, evaluation_id="eval-01")
        assert eval_id == "eval-01"

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        loaded = repo.get_prediction_evaluation("eval-01")
        assert loaded is not None
        assert loaded.target == PredictionTarget.TTFT_MS
        assert loaded.mae == 1.4
        assert loaded.is_statistically_sufficient is True

        items, total = repo.list_prediction_evaluations(target="ttft_ms")
        assert total == 1
        assert items[0].mae == 1.4


def test_optimization_and_pareto_candidates_persistence(db_manager: DatabaseManager) -> None:
    """Verify OptimizationRequest and OptimizationResult with Pareto candidates persist cleanly."""
    model = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)
    workload = WorkloadSpec(prompt_tokens=128, output_tokens=64)
    request = OptimizationRequest(
        workload=workload,
        search_space=SearchSpace(
            models=[model],
            quantizations=[QuantizationType.FP16, QuantizationType.INT4],
            concurrencies=[1],
            engines=[EngineType.MOCK],
        ),
        objectives=[OptimizationObjective.MINIMIZE_TTFT],
    )

    cand1 = CandidateEvaluation(
        config=CandidateConfig(
            model_spec=model,
            quantization=QuantizationType.FP16,
            concurrency=1,
            engine_type=EngineType.MOCK,
        ),
        feasibility_status=FeasibilityStatus.FEASIBLE,
        objective_values={OptimizationObjective.MINIMIZE_TTFT: 20.0},
        objective_units={OptimizationObjective.MINIMIZE_TTFT: "ms"},
        evidence_status={
            OptimizationObjective.MINIMIZE_TTFT: EvidenceStatus.THEORETICAL_ANALYTICAL
        },
        provenance={OptimizationObjective.MINIMIZE_TTFT: ProvenanceType.HEURISTIC},
        trade_off_summary="Optimal FP16 configuration",
    )

    result = OptimizationResult(
        request_id="opt-persist-01",
        summary="Optimized 1 candidate on Pareto frontier.",
        total_candidates_generated=2,
        total_candidates_evaluated=2,
        feasible_candidates_count=1,
        infeasible_candidates_count=1,
        unknown_feasibility_count=0,
        pareto_frontier=[cand1],
        recommended_candidate=cand1,
        selection_rationale="Lowest TTFT",
        rejections_by_reason={},
        all_evaluations=[cand1],
    )

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        repo.save_optimization(request, result)

    with db_manager.get_session() as session:
        repo = EngineRepository(session)
        loaded = repo.get_optimization("opt-persist-01")
        assert loaded is not None
        req_loaded, res_loaded = loaded
        assert res_loaded.request_id == "opt-persist-01"
        assert len(res_loaded.pareto_frontier) == 1
        assert (
            res_loaded.pareto_frontier[0].objective_values[OptimizationObjective.MINIMIZE_TTFT]
            == 20.0
        )
        assert res_loaded.recommended_candidate is not None
        assert res_loaded.selection_rationale == "Lowest TTFT"
