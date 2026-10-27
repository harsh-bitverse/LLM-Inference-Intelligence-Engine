"""Targeted audit tests for repository mapping fidelity, enum preservation, and round-trip data integrity."""

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
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
from llm_ii_engine.core.schemas import ModelSpec, Observation, ServerReportedStats, WorkloadSpec
from llm_ii_engine.discovery.models import (
    ComputeCapabilitySpec,
    CPUInfo,
    DetectedGPU,
    GPUSpecification,
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
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
    SearchSpace,
    SelectionPolicy,
)
from llm_ii_engine.prediction.models import (
    EvaluationResult,
    GeneralizationStatus,
    PredictionTarget,
)
from llm_ii_engine.storage.database import DatabaseManager
from llm_ii_engine.storage.models import GPUDeviceReportORM, RequestBenchmarkORM
from llm_ii_engine.storage.repository import (
    EngineRepository,
    EntityAlreadyExistsError,
)


@pytest.fixture
def repo(tmp_path: Path) -> Generator[EngineRepository, None, None]:
    """Yield an EngineRepository bound to a transactional session on a temporary SQLite database."""
    db_file = tmp_path / "repo_audit.db"
    mgr = DatabaseManager(database_url=f"sqlite:///{db_file}")
    mgr.init_db()
    with mgr.get_session() as session:
        yield EngineRepository(session)
    mgr.dispose()


def test_hardware_report_round_trip_integrity(repo: EngineRepository) -> None:
    """Verify hardware report, GPUs, telemetry, and matched catalog specifications survive round trip with exact fidelity."""
    gpu1 = DetectedGPU(
        index=0,
        name="NVIDIA RTX 4090",
        vendor="NVIDIA",
        total_memory_mb=24576.0,
        driver_version="550.54.14",
        uuid="GPU-1111-2222-3333",
        pci_bus_id="0000:01:00.0",
        compute_capability="8.9",
        detection_source="nvidia-smi",
    )
    gpu2 = DetectedGPU(
        index=1,
        name="NVIDIA RTX 3080",
        vendor="NVIDIA",
        total_memory_mb=10240.0,
        driver_version="550.54.14",
        detection_source="pynvml",
    )
    catalog_spec = GPUSpecification(
        catalog_id="nvidia-rtx-4090",
        name="NVIDIA GeForce RTX 4090",
        architecture="Ada Lovelace",
        form_factor="Desktop",
        memory_capacity_gb=24.0,
        memory_type="GDDR6X",
        theoretical_memory_bandwidth_gbps=1008.0,
        compute=ComputeCapabilitySpec(
            fp32_tflops=82.58,
            fp16_dense_tflops=165.2,
            fp16_sparse_tflops=330.3,
            bf16_dense_tflops=165.2,
            bf16_sparse_tflops=330.3,
            int8_dense_tops=330.3,
            int8_sparse_tops=660.6,
        ),
        tdp_watts=450,
        compute_capability="8.9",
        source_reference="Whitepaper",
    )

    report = HardwareCapabilityReport(
        report_id="hw-audit-full",
        timestamp_utc=datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC),
        is_simulated=False,
        execution_mode=ExecutionMode.REAL,
        host=HostSystemInfo(
            os_name="Linux",
            os_version="6.5.0-35-generic",
            os_release="Ubuntu 22.04.4 LTS",
            architecture="x86_64",
            python_version="3.12.3",
        ),
        cpu=CPUInfo(
            model_name="AMD EPYC 7763",
            vendor="AMD",
            physical_cores=32,
            logical_cores=64,
            frequency_mhz=2450.0,
        ),
        memory=MemoryInfo(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
        ),
        gpus=[gpu1, gpu2],
        telemetry=TelemetryTools(
            has_nvidia_smi=True,
            nvidia_smi_path="/usr/bin/nvidia-smi",
            has_rocm_smi=False,
            has_pynvml=True,
        ),
        matched_catalog_specs=[catalog_spec],
        missing_or_incomplete_fields=["ecc_support"],
        warnings=["Non-datacenter GPU detected."],
    )

    repo.save_hardware_report(report)

    # Retrieval
    loaded = repo.get_hardware_report("hw-audit-full")
    assert loaded is not None
    assert loaded.report_id == "hw-audit-full"
    assert loaded.execution_mode == ExecutionMode.REAL
    assert loaded.is_simulated is False
    assert loaded.host.os_name == "Linux"
    assert loaded.cpu.logical_cores == 64
    assert loaded.memory.total_bytes == 128 * 1024**3
    assert len(loaded.gpus) == 2
    assert loaded.gpus[0].name == "NVIDIA RTX 4090"
    assert loaded.gpus[0].compute_capability == "8.9"
    assert loaded.gpus[1].total_memory_mb == 10240.0
    assert loaded.telemetry.has_nvidia_smi is True
    assert loaded.matched_catalog_specs[0].catalog_id == "nvidia-rtx-4090"
    assert loaded.missing_or_incomplete_fields == ["ecc_support"]
    assert loaded.warnings == ["Non-datacenter GPU detected."]

    # Duplicate insertion must fail loudly
    with pytest.raises(EntityAlreadyExistsError):
        repo.save_hardware_report(report)


def test_benchmark_suite_round_trip_integrity(repo: EngineRepository) -> None:
    """Verify benchmark suite, warmup requests, measured requests, chunk metrics, and observations survive round trip."""
    warmup_req = RequestBenchmarkResult(
        request_id="req-warmup-1",
        success=True,
        is_warmup=True,
        prompt_tokens=32,
        requested_output_tokens=16,
        output_tokens=16,
        ttft_ms=25.0,
        tpot_ms=10.0,
        duration_ms=185.0,
        output_tokens_per_second=86.5,
        inter_token_latencies_ms=[10.0] * 16,
        token_count_source=TokenCountSource.TOKEN_EVENTS,
        prompt_token_count_source=TokenCountSource.REQUESTED,
        clock_source=ClockSource.MONOTONIC_REAL,
        execution_mode=ExecutionMode.REAL,
        provenance=ProvenanceType.MEASURED,
    )
    measured_req = RequestBenchmarkResult(
        request_id="req-measured-1",
        success=True,
        is_warmup=False,
        prompt_tokens=128,
        requested_output_tokens=64,
        output_tokens=64,
        ttft_ms=12.5,
        tpot_ms=8.2,
        duration_ms=537.3,
        output_tokens_per_second=119.1,
        inter_token_latencies_ms=[8.2] * 64,
        itl_p50_ms=8.2,
        itl_p90_ms=8.5,
        itl_p95_ms=8.7,
        itl_p99_ms=9.1,
        chunk_count=64,
        chunk_latencies_ms=[8.2] * 64,
        server_stats=ServerReportedStats(
            prompt_tokens=128,
            output_tokens=64,
            prompt_duration_ms=12.0,
            eval_duration_ms=525.0,
            total_duration_ms=537.0,
            eval_tokens_per_second=121.9,
        ),
        token_count_source=TokenCountSource.TOKEN_EVENTS,
        prompt_token_count_source=TokenCountSource.SERVER_REPORTED,
        clock_source=ClockSource.MONOTONIC_REAL,
        execution_mode=ExecutionMode.REAL,
        provenance=ProvenanceType.MEASURED,
    )

    suite = BenchmarkSuiteResult(
        suite_id="suite-audit-1",
        timestamp_utc=datetime(2026, 10, 25, 14, 0, 0, tzinfo=UTC),
        status="COMPLETED",
        model_id="meta-llama/Llama-3-8B",
        engine_type=EngineType.LLAMACPP,
        execution_mode=ExecutionMode.REAL,
        clock_source=ClockSource.MONOTONIC_REAL,
        provenance=ProvenanceType.MEASURED,
        concurrency=1,
        total_requests=2,
        warmup_requests=1,
        successful_requests=2,
        failed_requests=0,
        wall_time_seconds=1.2,
        aggregate_tokens_per_second=119.1,
        aggregate_output_tokens_per_second=119.1,
        request_throughput_per_second=0.83,
        ttft_p50_ms=12.5,
        ttft_p90_ms=12.5,
        ttft_p95_ms=12.5,
        ttft_p99_ms=12.5,
        tpot_mean_ms=8.2,
        itl_p50_ms=8.2,
        itl_p90_ms=8.5,
        itl_p95_ms=8.7,
        itl_p99_ms=9.1,
        memory_watermark=MemoryWatermarkResult(
            peak_process_rss_mb=4096.0,
            peak_gpu_vram_mb=16384.0,
            sample_count=10,
            sample_interval_ms=100.0,
            telemetry_source="mock",
            is_telemetry_available=True,
            notes="Normal execution",
        ),
        request_results=[measured_req],
        warmup_results=[warmup_req],
        warnings=["Workload distribution was uniform."],
    )

    repo.save_benchmark_suite(suite)

    loaded = repo.get_benchmark_suite("suite-audit-1")
    assert loaded is not None
    assert loaded.suite_id == "suite-audit-1"
    assert loaded.model_id == "meta-llama/Llama-3-8B"
    assert loaded.engine_type == EngineType.LLAMACPP
    assert loaded.execution_mode == ExecutionMode.REAL
    assert loaded.clock_source == ClockSource.MONOTONIC_REAL
    assert loaded.provenance == ProvenanceType.MEASURED
    assert len(loaded.warmup_results) == 1
    assert loaded.warmup_results[0].request_id == "req-warmup-1"
    assert loaded.warmup_results[0].is_warmup is True
    assert len(loaded.request_results) == 1
    assert loaded.request_results[0].request_id == "req-measured-1"
    assert loaded.request_results[0].server_stats is not None
    assert loaded.request_results[0].server_stats.eval_tokens_per_second == 121.9
    assert loaded.memory_watermark is not None
    assert loaded.memory_watermark.peak_gpu_vram_mb == 16384.0
    assert loaded.memory_watermark.is_telemetry_available is True
    assert loaded.warnings == ["Workload distribution was uniform."]


def test_diagnostic_report_round_trip_integrity(repo: EngineRepository) -> None:
    """Verify diagnostic reports, typed findings, severity, confidence, and nested observations survive round trip."""
    obs = Observation(
        value=85.0,
        unit="tok/s",
        provenance=ProvenanceType.MEASURED,
        execution_mode=ExecutionMode.REAL,
        clock_source=ClockSource.MONOTONIC_REAL,
        description="Measured decode throughput on RTX 4090",
    )
    finding = DiagnosticFinding(
        finding_id="find-diag-1",
        category=DiagnosticCategory.EFFICIENCY_BOTTLENECK,
        severity=DiagnosticSeverity.WARNING,
        title="Sub-optimal Roofline Efficiency",
        message="Observed decode throughput is 65% of memory-bandwidth bound ceiling.",
        confidence=DiagnosticConfidence.MEDIUM,
        confidence_rationale="Empirical tokens/s compared against 1008 GB/s bandwidth.",
        supporting_observations=[obs],
        assumptions=["Standard transformer memory operational intensity."],
        limitations=["Hardware counter memory bus saturation not directly recorded."],
        suggested_experiments=["Increase batch size concurrency to 4."],
    )

    report = DiagnosticReport(
        report_id="diag-audit-1",
        timestamp_utc=datetime(2026, 10, 25, 15, 0, 0, tzinfo=UTC),
        model_id="meta-llama/Llama-3-8B",
        engine_type=EngineType.LLAMACPP,
        execution_mode=ExecutionMode.REAL,
        hardware_catalog_id="nvidia-rtx-4090",
        findings=[finding],
        summary="Bottleneck evaluation shows memory-bandwidth headroom.",
    )

    repo.save_diagnostic_report(report)

    loaded = repo.get_diagnostic_report("diag-audit-1")
    assert loaded is not None
    assert loaded.report_id == "diag-audit-1"
    assert loaded.findings[0].category == DiagnosticCategory.EFFICIENCY_BOTTLENECK
    assert loaded.findings[0].severity == DiagnosticSeverity.WARNING
    assert loaded.findings[0].confidence == DiagnosticConfidence.MEDIUM
    assert len(loaded.findings[0].supporting_observations) == 1
    assert loaded.findings[0].supporting_observations[0].value == 85.0
    assert loaded.findings[0].supporting_observations[0].unit == "tok/s"
    assert loaded.findings[0].assumptions == ["Standard transformer memory operational intensity."]
    assert loaded.findings[0].suggested_experiments == ["Increase batch size concurrency to 4."]


def test_prediction_evaluation_round_trip_integrity(repo: EngineRepository) -> None:
    """Verify empirical prediction evaluation metrics, split strategy, and sufficiency indicators survive round trip."""
    eval_result = EvaluationResult(
        target=PredictionTarget.TTFT_MS,
        predictor_id="ridge-ttft-v1",
        predictor_type="RidgeRegressionPredictor",
        split_strategy="HoldoutByModel",
        sample_count=50,
        train_sample_count=40,
        test_sample_count=10,
        unit="ms",
        mae=2.45,
        rmse=3.12,
        mape_percent=4.85,
        baseline_mae=8.50,
        baseline_rmse=10.20,
        is_synthetic=False,
        execution_mode=ExecutionMode.REAL,
        is_statistically_sufficient=True,
        notes="Evaluated across LLaMA-2 and Mistral holdout partitions.",
    )

    eval_id = repo.save_prediction_evaluation(eval_result, evaluation_id="eval-audit-1")
    assert eval_id == "eval-audit-1"

    loaded = repo.get_prediction_evaluation("eval-audit-1")
    assert loaded is not None
    assert loaded.target == PredictionTarget.TTFT_MS
    assert loaded.predictor_id == "ridge-ttft-v1"
    assert loaded.split_strategy == "HoldoutByModel"
    assert loaded.mae == 2.45
    assert loaded.rmse == 3.12
    assert loaded.mape_percent == 4.85
    assert loaded.is_synthetic is False
    assert loaded.execution_mode == ExecutionMode.REAL
    assert loaded.is_statistically_sufficient is True
    assert loaded.notes == "Evaluated across LLaMA-2 and Mistral holdout partitions."


def test_optimization_and_candidates_round_trip_with_ood_and_fidelity(
    repo: EngineRepository,
) -> None:
    """Verify optimization request, candidate evaluations, OOD warnings, trade-offs, and Pareto frontier survive round trip."""
    model = ModelSpec(model_id="opt-llama-7b", parameter_count_billion=7.0, context_window=4096)
    workload = WorkloadSpec(prompt_tokens=256, output_tokens=128)
    space = SearchSpace(
        models=[model],
        quantizations=[QuantizationType.FP16, QuantizationType.INT4],
        concurrencies=[1, 2],
        engines=[EngineType.MOCK],
    )
    request = OptimizationRequest(
        workload=workload,
        search_space=space,
        objectives=[OptimizationObjective.MINIMIZE_TTFT, OptimizationObjective.MAXIMIZE_THROUGHPUT],
        constraints=OptimizationConstraints(max_ttft_ms=100.0, max_vram_mb=16000.0),
        selection_policy=SelectionPolicy.BALANCED_KNEE_POINT,
    )

    config_feasible = CandidateConfig(
        model_spec=model,
        quantization=QuantizationType.FP16,
        concurrency=1,
        engine_type=EngineType.MOCK,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        extra_args={"kv_cache_type": "paged"},
    )
    cand_feasible = CandidateEvaluation(
        config=config_feasible,
        feasibility_status=FeasibilityStatus.FEASIBLE,
        rejection_reasons=[],
        objective_values={
            OptimizationObjective.MINIMIZE_TTFT: 45.0,
            OptimizationObjective.MAXIMIZE_THROUGHPUT: 120.0,
        },
        objective_units={
            OptimizationObjective.MINIMIZE_TTFT: "ms",
            OptimizationObjective.MAXIMIZE_THROUGHPUT: "tok/s",
        },
        evidence_status={
            OptimizationObjective.MINIMIZE_TTFT: EvidenceStatus.MEASURED_EVALUATED,
            OptimizationObjective.MAXIMIZE_THROUGHPUT: EvidenceStatus.PREDICTED_EMPIRICAL,
        },
        provenance={
            OptimizationObjective.MINIMIZE_TTFT: ProvenanceType.MEASURED,
            OptimizationObjective.MAXIMIZE_THROUGHPUT: ProvenanceType.PREDICTED,
        },
        generalization_status=GeneralizationStatus(
            is_out_of_distribution=True,
            warnings=["Concurrency level 1 is at boundary of training envelope."],
            extrapolated_features=["concurrency"],
        ),
        has_mixed_evidence=True,
        comparability_warning="Combines measured TTFT with empirical predicted throughput.",
        trade_off_summary="Optimal knee point balancing 45ms TTFT with 120 tok/s throughput.",
        notes=["Estimated on simulated Ada profile."],
    )

    config_infeasible = CandidateConfig(
        model_spec=model,
        quantization=QuantizationType.INT4,
        concurrency=2,
        engine_type=EngineType.MOCK,
        tensor_parallel_size=1,
    )
    cand_infeasible = CandidateEvaluation(
        config=config_infeasible,
        feasibility_status=FeasibilityStatus.ESTIMATED_INFEASIBLE,
        rejection_reasons=["VRAM limit 16000 MiB exceeded by estimated 18200 MiB."],
        objective_values={
            OptimizationObjective.MINIMIZE_TTFT: 70.0,
            OptimizationObjective.MAXIMIZE_THROUGHPUT: 95.0,
        },
        objective_units={
            OptimizationObjective.MINIMIZE_TTFT: "ms",
            OptimizationObjective.MAXIMIZE_THROUGHPUT: "tok/s",
        },
        evidence_status={
            OptimizationObjective.MINIMIZE_TTFT: EvidenceStatus.THEORETICAL_ANALYTICAL,
            OptimizationObjective.MAXIMIZE_THROUGHPUT: EvidenceStatus.THEORETICAL_ANALYTICAL,
        },
        provenance={
            OptimizationObjective.MINIMIZE_TTFT: ProvenanceType.HEURISTIC,
            OptimizationObjective.MAXIMIZE_THROUGHPUT: ProvenanceType.HEURISTIC,
        },
    )

    result = OptimizationResult(
        request_id="opt-audit-1",
        summary="Evaluated 2 candidates; 1 feasible, 1 infeasible.",
        total_candidates_generated=2,
        total_candidates_evaluated=2,
        feasible_candidates_count=1,
        infeasible_candidates_count=1,
        unknown_feasibility_count=0,
        pareto_frontier=[cand_feasible],
        recommended_candidate=cand_feasible,
        selection_rationale="Selected BALANCED_KNEE_POINT from frontier.",
        rejections_by_reason={"VRAM limit exceeded": 1},
        all_evaluations=[cand_feasible, cand_infeasible],
        has_incomparable_evidence=True,
        evidence_caveats=["Candidate combines measured TTFT with empirical throughput."],
    )

    repo.save_optimization(request, result)

    loaded_pair = repo.get_optimization("opt-audit-1")
    assert loaded_pair is not None
    loaded_req, loaded_res = loaded_pair

    # Verify request reconstruction
    assert loaded_req.workload.prompt_tokens == 256
    assert loaded_req.workload.output_tokens == 128
    assert loaded_req.selection_policy == SelectionPolicy.BALANCED_KNEE_POINT
    assert loaded_req.constraints.max_ttft_ms == 100.0

    # Verify result and candidates reconstruction
    assert loaded_res.request_id == "opt-audit-1"
    assert loaded_res.feasible_candidates_count == 1
    assert loaded_res.infeasible_candidates_count == 1
    assert len(loaded_res.pareto_frontier) == 1
    assert len(loaded_res.all_evaluations) == 2
    assert loaded_res.has_incomparable_evidence is True
    assert loaded_res.evidence_caveats == [
        "Candidate combines measured TTFT with empirical throughput."
    ]

    # Verify candidate fidelity
    cand_f = loaded_res.pareto_frontier[0]
    assert cand_f.config.quantization == QuantizationType.FP16
    assert cand_f.config.gpu_memory_utilization == 0.9
    assert cand_f.config.extra_args == {"kv_cache_type": "paged"}
    assert cand_f.feasibility_status == FeasibilityStatus.FEASIBLE
    assert cand_f.has_mixed_evidence is True
    assert (
        cand_f.comparability_warning
        == "Combines measured TTFT with empirical predicted throughput."
    )
    assert (
        cand_f.trade_off_summary
        == "Optimal knee point balancing 45ms TTFT with 120 tok/s throughput."
    )

    # Verify OOD warnings survived
    assert cand_f.generalization_status is not None
    assert cand_f.generalization_status.is_out_of_distribution is True
    assert cand_f.generalization_status.warnings == [
        "Concurrency level 1 is at boundary of training envelope."
    ]
    assert cand_f.generalization_status.extrapolated_features == ["concurrency"]


def test_sqlite_foreign_key_rejects_orphaned_records(repo: EngineRepository) -> None:
    """Verify SQLite foreign key enforcement strictly rejects inserting child records with non-existent parent."""
    # Attempt to insert an orphaned GPUDeviceReportORM pointing to non-existent hardware report
    orphan_gpu = GPUDeviceReportORM(
        report_id="non-existent-hw-report",
        gpu_index=0,
        name="Ghost GPU",
        vendor="NVIDIA",
        detection_source="test",
    )
    repo._session.add(orphan_gpu)
    with pytest.raises(IntegrityError):
        repo._session.flush()

    repo._session.rollback()

    # Attempt to insert an orphaned RequestBenchmarkORM pointing to non-existent suite
    orphan_req = RequestBenchmarkORM(
        request_id="orphan-req-99",
        suite_id="non-existent-suite",
        success=True,
        is_warmup=False,
        prompt_tokens=10,
        output_tokens=10,
        duration_ms=50.0,
        token_count_source="token_events",
        prompt_token_count_source="requested",
        clock_source="monotonic_real",
        execution_mode="real",
        provenance="measured",
    )
    repo._session.add(orphan_req)
    with pytest.raises(IntegrityError):
        repo._session.flush()

    repo._session.rollback()
