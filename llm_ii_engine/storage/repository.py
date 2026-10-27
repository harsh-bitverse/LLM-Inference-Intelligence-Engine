"""Typed repository layer mapping SQLAlchemy ORM entities to canonical Pydantic domain models.

Provides transaction-aware persistence for hardware discovery reports, benchmark suites,
request measurements, metric observations, diagnostic reports, prediction evaluations,
and configuration optimization outcomes.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from llm_ii_engine.analysis.models import (
    DiagnosticFinding,
    DiagnosticReport,
    ModelResourceEstimate,
    RooflineAnalysisResult,
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
from llm_ii_engine.core.schemas import ModelSpec, Observation, ServerReportedStats
from llm_ii_engine.discovery.models import (
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
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
)
from llm_ii_engine.prediction.models import (
    EvaluationResult,
    GeneralizationStatus,
    PredictionTarget,
)
from llm_ii_engine.storage.models import (
    BenchmarkSuiteORM,
    DiagnosticFindingORM,
    DiagnosticReportORM,
    GPUDeviceReportORM,
    HardwareReportORM,
    MetricObservationORM,
    OptimizationCandidateORM,
    OptimizationRecordORM,
    PredictionEvaluationORM,
    RequestBenchmarkORM,
)


class StorageError(Exception):
    """Base exception for storage and persistence failures."""

    pass


class EntityNotFoundError(StorageError):
    """Raised when a requested entity does not exist in the database."""

    pass


class EntityAlreadyExistsError(StorageError):
    """Raised when an attempt is made to insert an entity with an existing primary key."""

    pass


class EngineRepository:
    """Repository translating between Pydantic domain models and relational database tables."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    # Hardware Discovery Persistence
    # -------------------------------------------------------------------------

    def save_hardware_report(self, report: HardwareCapabilityReport) -> None:
        """Persist a HardwareCapabilityReport with its associated GPU device metadata."""
        existing = self._session.get(HardwareReportORM, report.report_id)
        if existing:
            raise EntityAlreadyExistsError(
                f"Hardware report '{report.report_id}' is already persisted."
            )

        report_orm = HardwareReportORM(
            report_id=report.report_id,
            created_at=report.timestamp_utc,
            is_simulated=report.is_simulated,
            execution_mode=report.execution_mode.value,
            os_name=report.host.os_name,
            os_version=report.host.os_version,
            os_release=report.host.os_release,
            os_arch=report.host.architecture,
            python_version=report.host.python_version,
            cpu_model=report.cpu.model_name,
            cpu_vendor=report.cpu.vendor,
            cpu_physical_cores=report.cpu.physical_cores,
            cpu_logical_cores=report.cpu.logical_cores,
            cpu_frequency_mhz=report.cpu.frequency_mhz,
            total_ram_bytes=report.memory.total_bytes,
            available_ram_bytes=report.memory.available_bytes,
            telemetry_json=report.telemetry.model_dump(mode="json"),
            matched_catalog_specs_json=[
                spec.model_dump(mode="json") for spec in report.matched_catalog_specs
            ]
            if report.matched_catalog_specs
            else None,
            missing_or_incomplete_fields=report.missing_or_incomplete_fields or None,
            warnings=report.warnings or None,
        )

        for gpu in report.gpus:
            gpu_orm = GPUDeviceReportORM(
                report_id=report.report_id,
                gpu_index=gpu.index,
                name=gpu.name,
                vendor=gpu.vendor,
                total_memory_mb=gpu.total_memory_mb,
                driver_version=gpu.driver_version,
                uuid=gpu.uuid,
                pci_bus_id=gpu.pci_bus_id,
                compute_capability=gpu.compute_capability,
                detection_source=gpu.detection_source,
            )
            report_orm.gpus.append(gpu_orm)

        self._session.add(report_orm)
        self._session.flush()

    def get_hardware_report(self, report_id: str) -> HardwareCapabilityReport | None:
        """Retrieve a HardwareCapabilityReport by ID, returning None if not found."""
        orm = self._session.get(HardwareReportORM, report_id)
        if not orm:
            return None
        return self._hardware_orm_to_domain(orm)

    def get_latest_hardware_report(self) -> HardwareCapabilityReport | None:
        """Retrieve the most recently recorded HardwareCapabilityReport."""
        stmt = select(HardwareReportORM).order_by(HardwareReportORM.created_at.desc()).limit(1)
        orm = self._session.scalars(stmt).first()
        if not orm:
            return None
        return self._hardware_orm_to_domain(orm)

    def list_hardware_reports(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[HardwareCapabilityReport], int]:
        """List persisted hardware reports with pagination."""
        total_stmt = select(func.count(HardwareReportORM.report_id))
        total = self._session.scalar(total_stmt) or 0

        stmt = (
            select(HardwareReportORM)
            .order_by(HardwareReportORM.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        items = [self._hardware_orm_to_domain(orm) for orm in self._session.scalars(stmt).all()]
        return items, total

    @staticmethod
    def _hardware_orm_to_domain(orm: HardwareReportORM) -> HardwareCapabilityReport:
        gpus: list[DetectedGPU] = []
        for g in orm.gpus:
            gpus.append(
                DetectedGPU(
                    index=g.gpu_index,
                    name=g.name,
                    vendor=g.vendor,
                    total_memory_mb=g.total_memory_mb,
                    driver_version=g.driver_version,
                    uuid=g.uuid,
                    pci_bus_id=g.pci_bus_id,
                    compute_capability=g.compute_capability,
                    detection_source=g.detection_source,
                )
            )

        matched_specs = []
        if orm.matched_catalog_specs_json:
            for s in orm.matched_catalog_specs_json:
                matched_specs.append(GPUSpecification.model_validate(s))

        telemetry = (
            TelemetryTools.model_validate(orm.telemetry_json)
            if orm.telemetry_json
            else TelemetryTools()
        )

        return HardwareCapabilityReport(
            report_id=orm.report_id,
            timestamp_utc=orm.created_at,
            is_simulated=orm.is_simulated,
            execution_mode=ExecutionMode(orm.execution_mode),
            host=HostSystemInfo(
                os_name=orm.os_name,
                os_version=orm.os_version,
                os_release=orm.os_release,
                architecture=orm.os_arch,
                python_version=orm.python_version,
            ),
            cpu=CPUInfo(
                model_name=orm.cpu_model,
                vendor=orm.cpu_vendor,
                physical_cores=orm.cpu_physical_cores,
                logical_cores=orm.cpu_logical_cores,
                frequency_mhz=orm.cpu_frequency_mhz,
            ),
            memory=MemoryInfo(
                total_bytes=orm.total_ram_bytes,
                available_bytes=orm.available_ram_bytes,
            ),
            gpus=gpus,
            telemetry=telemetry,
            matched_catalog_specs=matched_specs,
            missing_or_incomplete_fields=orm.missing_or_incomplete_fields or [],
            warnings=orm.warnings or [],
        )

    # -------------------------------------------------------------------------
    # Benchmark Suite and Request Measurements
    # -------------------------------------------------------------------------

    def save_benchmark_suite(
        self, suite: BenchmarkSuiteResult, hardware_report_id: str | None = None
    ) -> None:
        """Persist a complete BenchmarkSuiteResult with requests and discrete observations."""
        existing = self._session.get(BenchmarkSuiteORM, suite.suite_id)
        if existing:
            raise EntityAlreadyExistsError(
                f"Benchmark suite '{suite.suite_id}' is already persisted."
            )

        suite_orm = BenchmarkSuiteORM(
            suite_id=suite.suite_id,
            created_at=suite.timestamp_utc,
            status=suite.status,
            model_id=suite.model_id,
            engine_type=suite.engine_type.value,
            execution_mode=suite.execution_mode.value,
            clock_source=suite.clock_source.value,
            provenance=suite.provenance.value,
            concurrency=suite.concurrency,
            total_requests=suite.total_requests,
            warmup_requests=suite.warmup_requests,
            successful_requests=suite.successful_requests,
            failed_requests=suite.failed_requests,
            wall_time_seconds=suite.wall_time_seconds,
            aggregate_tokens_per_second=suite.aggregate_tokens_per_second,
            aggregate_output_tokens_per_second=suite.aggregate_output_tokens_per_second,
            request_throughput_per_second=suite.request_throughput_per_second,
            ttft_p50_ms=suite.ttft_p50_ms,
            ttft_p90_ms=suite.ttft_p90_ms,
            ttft_p95_ms=suite.ttft_p95_ms,
            ttft_p99_ms=suite.ttft_p99_ms,
            ttfc_p50_ms=suite.ttfc_p50_ms,
            ttfc_p90_ms=suite.ttfc_p90_ms,
            ttfc_p95_ms=suite.ttfc_p95_ms,
            ttfc_p99_ms=suite.ttfc_p99_ms,
            chunk_latency_p50_ms=suite.chunk_latency_p50_ms,
            tpot_mean_ms=suite.tpot_mean_ms,
            itl_p50_ms=suite.itl_p50_ms,
            itl_p90_ms=suite.itl_p90_ms,
            itl_p95_ms=suite.itl_p95_ms,
            itl_p99_ms=suite.itl_p99_ms,
            memory_watermark_json=suite.memory_watermark.model_dump(mode="json")
            if suite.memory_watermark
            else None,
            warnings=suite.warnings or None,
            hardware_report_id=hardware_report_id,
        )

        all_requests = list(suite.request_results) + list(suite.warmup_results)
        for req in all_requests:
            req_orm = RequestBenchmarkORM(
                request_id=req.request_id,
                suite_id=suite.suite_id,
                success=req.success,
                is_warmup=req.is_warmup,
                prompt_tokens=req.prompt_tokens,
                requested_output_tokens=req.requested_output_tokens,
                output_tokens=req.output_tokens,
                ttft_ms=req.ttft_ms,
                ttfc_ms=req.ttfc_ms,
                tpot_ms=req.tpot_ms,
                duration_ms=req.duration_ms,
                output_tokens_per_second=req.output_tokens_per_second,
                inter_token_latencies_ms=req.inter_token_latencies_ms or None,
                itl_p50_ms=req.itl_p50_ms,
                itl_p90_ms=req.itl_p90_ms,
                itl_p95_ms=req.itl_p95_ms,
                itl_p99_ms=req.itl_p99_ms,
                chunk_count=req.chunk_count,
                chunk_latencies_ms=req.chunk_latencies_ms or None,
                chunk_latency_p50_ms=req.chunk_latency_p50_ms,
                chunk_latency_p90_ms=req.chunk_latency_p90_ms,
                chunk_latency_p95_ms=req.chunk_latency_p95_ms,
                chunk_latency_p99_ms=req.chunk_latency_p99_ms,
                server_stats_json=req.server_stats.model_dump(mode="json")
                if req.server_stats
                else None,
                token_count_source=req.token_count_source.value,
                prompt_token_count_source=req.prompt_token_count_source.value,
                clock_source=req.clock_source.value,
                execution_mode=req.execution_mode.value,
                provenance=req.provenance.value,
                error_message=req.error_message,
                incomplete_reason=req.incomplete_reason,
            )
            suite_orm.requests.append(req_orm)

        # Persist discrete metric observations with units and provenance
        key_metrics = [
            ("aggregate_tokens_per_second", suite.aggregate_tokens_per_second, "tok/s"),
            (
                "aggregate_output_tokens_per_second",
                suite.aggregate_output_tokens_per_second,
                "tok/s",
            ),
            ("request_throughput_per_second", suite.request_throughput_per_second, "req/s"),
            ("ttft_p50_ms", suite.ttft_p50_ms, "ms"),
            ("ttft_p90_ms", suite.ttft_p90_ms, "ms"),
            ("ttft_p95_ms", suite.ttft_p95_ms, "ms"),
            ("ttft_p99_ms", suite.ttft_p99_ms, "ms"),
            ("ttfc_p50_ms", suite.ttfc_p50_ms, "ms"),
            ("tpot_mean_ms", suite.tpot_mean_ms, "ms"),
            ("itl_p50_ms", suite.itl_p50_ms, "ms"),
            ("itl_p90_ms", suite.itl_p90_ms, "ms"),
            ("itl_p95_ms", suite.itl_p95_ms, "ms"),
            ("itl_p99_ms", suite.itl_p99_ms, "ms"),
        ]
        if suite.memory_watermark and suite.memory_watermark.peak_gpu_vram_mb is not None:
            key_metrics.append(("peak_gpu_vram_mb", suite.memory_watermark.peak_gpu_vram_mb, "MiB"))

        for name, val, unit in key_metrics:
            if val is not None:
                obs_orm = MetricObservationORM(
                    observation_id=f"obs-{uuid.uuid4().hex[:12]}",
                    suite_id=suite.suite_id,
                    created_at=suite.timestamp_utc,
                    name=name,
                    value=float(val),
                    unit=unit,
                    provenance=suite.provenance.value,
                    execution_mode=suite.execution_mode.value,
                    clock_source=suite.clock_source.value,
                )
                suite_orm.observations.append(obs_orm)

        self._session.add(suite_orm)
        self._session.flush()

    def get_benchmark_suite(self, suite_id: str) -> BenchmarkSuiteResult | None:
        """Retrieve a BenchmarkSuiteResult by suite_id, returning None if not found."""
        orm = self._session.get(BenchmarkSuiteORM, suite_id)
        if not orm:
            return None
        return self._suite_orm_to_domain(orm)

    def list_benchmark_suites(
        self,
        limit: int = 50,
        offset: int = 0,
        model_id: str | None = None,
        engine_type: str | None = None,
    ) -> tuple[list[BenchmarkSuiteResult], int]:
        """List benchmark suite runs with optional model and engine filters."""
        query = select(BenchmarkSuiteORM)
        count_query = select(func.count(BenchmarkSuiteORM.suite_id))

        if model_id:
            query = query.where(BenchmarkSuiteORM.model_id == model_id)
            count_query = count_query.where(BenchmarkSuiteORM.model_id == model_id)
        if engine_type:
            query = query.where(BenchmarkSuiteORM.engine_type == engine_type)
            count_query = count_query.where(BenchmarkSuiteORM.engine_type == engine_type)

        total = self._session.scalar(count_query) or 0
        stmt = query.order_by(BenchmarkSuiteORM.created_at.desc()).offset(offset).limit(limit)

        items = [self._suite_orm_to_domain(orm) for orm in self._session.scalars(stmt).all()]
        return items, total

    def _suite_orm_to_domain(self, orm: BenchmarkSuiteORM) -> BenchmarkSuiteResult:
        measured_results: list[RequestBenchmarkResult] = []
        warmup_results: list[RequestBenchmarkResult] = []

        for r in orm.requests:
            server_stats = (
                ServerReportedStats.model_validate(r.server_stats_json)
                if r.server_stats_json
                else None
            )
            req_domain = RequestBenchmarkResult(
                request_id=r.request_id,
                success=r.success,
                error_message=r.error_message,
                prompt_tokens=r.prompt_tokens,
                requested_output_tokens=r.requested_output_tokens,
                output_tokens=r.output_tokens,
                ttft_ms=r.ttft_ms,
                ttfc_ms=r.ttfc_ms,
                tpot_ms=r.tpot_ms,
                duration_ms=r.duration_ms,
                output_tokens_per_second=r.output_tokens_per_second,
                inter_token_latencies_ms=r.inter_token_latencies_ms or [],
                itl_p50_ms=r.itl_p50_ms,
                itl_p90_ms=r.itl_p90_ms,
                itl_p95_ms=r.itl_p95_ms,
                itl_p99_ms=r.itl_p99_ms,
                chunk_count=r.chunk_count,
                chunk_latencies_ms=r.chunk_latencies_ms or [],
                chunk_latency_p50_ms=r.chunk_latency_p50_ms,
                chunk_latency_p90_ms=r.chunk_latency_p90_ms,
                chunk_latency_p95_ms=r.chunk_latency_p95_ms,
                chunk_latency_p99_ms=r.chunk_latency_p99_ms,
                server_stats=server_stats,
                token_count_source=(
                    TokenCountSource(r.token_count_source)
                    if r.token_count_source
                    and r.token_count_source in TokenCountSource.__members__.values()
                    else TokenCountSource.UNKNOWN
                ),
                prompt_token_count_source=(
                    TokenCountSource(r.prompt_token_count_source)
                    if r.prompt_token_count_source
                    and r.prompt_token_count_source in TokenCountSource.__members__.values()
                    else TokenCountSource.UNKNOWN
                ),
                clock_source=ClockSource(r.clock_source),
                execution_mode=ExecutionMode(r.execution_mode),
                provenance=ProvenanceType(r.provenance),
                is_warmup=r.is_warmup,
                incomplete_reason=r.incomplete_reason,
            )
            if r.is_warmup:
                warmup_results.append(req_domain)
            else:
                measured_results.append(req_domain)

        mem_watermark = (
            MemoryWatermarkResult.model_validate(orm.memory_watermark_json)
            if orm.memory_watermark_json
            else None
        )

        return BenchmarkSuiteResult(
            suite_id=orm.suite_id,
            timestamp_utc=orm.created_at,
            status=orm.status,
            model_id=orm.model_id,
            engine_type=EngineType(orm.engine_type),
            concurrency=orm.concurrency,
            total_requests=orm.total_requests,
            warmup_requests=orm.warmup_requests,
            successful_requests=orm.successful_requests,
            failed_requests=orm.failed_requests,
            wall_time_seconds=orm.wall_time_seconds,
            aggregate_tokens_per_second=orm.aggregate_tokens_per_second,
            aggregate_output_tokens_per_second=orm.aggregate_output_tokens_per_second,
            request_throughput_per_second=orm.request_throughput_per_second,
            ttft_p50_ms=orm.ttft_p50_ms,
            ttft_p90_ms=orm.ttft_p90_ms,
            ttft_p95_ms=orm.ttft_p95_ms,
            ttft_p99_ms=orm.ttft_p99_ms,
            ttfc_p50_ms=orm.ttfc_p50_ms,
            ttfc_p90_ms=orm.ttfc_p90_ms,
            ttfc_p95_ms=orm.ttfc_p95_ms,
            ttfc_p99_ms=orm.ttfc_p99_ms,
            chunk_latency_p50_ms=orm.chunk_latency_p50_ms,
            tpot_mean_ms=orm.tpot_mean_ms,
            itl_p50_ms=orm.itl_p50_ms,
            itl_p90_ms=orm.itl_p90_ms,
            itl_p95_ms=orm.itl_p95_ms,
            itl_p99_ms=orm.itl_p99_ms,
            memory_watermark=mem_watermark,
            execution_mode=ExecutionMode(orm.execution_mode),
            clock_source=ClockSource(orm.clock_source),
            provenance=ProvenanceType(orm.provenance),
            request_results=measured_results,
            warmup_results=warmup_results,
            warnings=orm.warnings or [],
        )

    # -------------------------------------------------------------------------
    # Diagnostic Reports
    # -------------------------------------------------------------------------

    def save_diagnostic_report(
        self, report: DiagnosticReport, benchmark_suite_id: str | None = None
    ) -> None:
        """Persist a DiagnosticReport with its bottleneck findings."""
        existing = self._session.get(DiagnosticReportORM, report.report_id)
        if existing:
            raise EntityAlreadyExistsError(
                f"Diagnostic report '{report.report_id}' is already persisted."
            )

        report_orm = DiagnosticReportORM(
            report_id=report.report_id,
            benchmark_suite_id=benchmark_suite_id,
            created_at=report.timestamp_utc,
            model_id=report.model_id,
            engine_type=report.engine_type.value,
            execution_mode=report.execution_mode.value,
            hardware_catalog_id=report.hardware_catalog_id,
            summary=report.summary,
            roofline_analysis_json=report.roofline_analysis.model_dump(mode="json")
            if report.roofline_analysis
            else None,
            resource_estimate_json=report.resource_estimate.model_dump(mode="json")
            if report.resource_estimate
            else None,
        )

        for finding in report.findings:
            f_orm = DiagnosticFindingORM(
                finding_id=finding.finding_id,
                report_id=report.report_id,
                category=finding.category.value,
                severity=finding.severity.value,
                title=finding.title,
                message=finding.message,
                confidence=finding.confidence.value,
                confidence_rationale=finding.confidence_rationale,
                supporting_observations_json=[
                    obs.model_dump(mode="json") for obs in finding.supporting_observations
                ]
                if finding.supporting_observations
                else None,
                assumptions=finding.assumptions or None,
                limitations=finding.limitations or None,
                suggested_experiments=finding.suggested_experiments or None,
            )
            report_orm.findings.append(f_orm)

        self._session.add(report_orm)
        self._session.flush()

    def get_diagnostic_report(self, report_id: str) -> DiagnosticReport | None:
        """Retrieve a DiagnosticReport by ID, returning None if not found."""
        orm = self._session.get(DiagnosticReportORM, report_id)
        if not orm:
            return None
        return self._diagnostic_orm_to_domain(orm)

    def list_diagnostic_reports(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[DiagnosticReport], int]:
        """List diagnostic reports with pagination."""
        total_stmt = select(func.count(DiagnosticReportORM.report_id))
        total = self._session.scalar(total_stmt) or 0

        stmt = (
            select(DiagnosticReportORM)
            .order_by(DiagnosticReportORM.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        items = [self._diagnostic_orm_to_domain(orm) for orm in self._session.scalars(stmt).all()]
        return items, total

    @staticmethod
    def _diagnostic_orm_to_domain(orm: DiagnosticReportORM) -> DiagnosticReport:
        findings: list[DiagnosticFinding] = []
        for f in orm.findings:
            obs_list = []
            if f.supporting_observations_json:
                for obs_dict in f.supporting_observations_json:
                    obs_list.append(Observation.model_validate(obs_dict))

            findings.append(
                DiagnosticFinding(
                    finding_id=f.finding_id,
                    category=DiagnosticCategory(f.category),
                    severity=DiagnosticSeverity(f.severity),
                    title=f.title,
                    message=f.message,
                    confidence=DiagnosticConfidence(f.confidence),
                    confidence_rationale=f.confidence_rationale,
                    supporting_observations=obs_list,
                    assumptions=f.assumptions or [],
                    limitations=f.limitations or [],
                    suggested_experiments=f.suggested_experiments or [],
                )
            )

        roofline = (
            RooflineAnalysisResult.model_validate(orm.roofline_analysis_json)
            if orm.roofline_analysis_json
            else None
        )
        res_est = (
            ModelResourceEstimate.model_validate(orm.resource_estimate_json)
            if orm.resource_estimate_json
            else None
        )

        return DiagnosticReport(
            report_id=orm.report_id,
            timestamp_utc=orm.created_at,
            model_id=orm.model_id,
            engine_type=EngineType(orm.engine_type),
            execution_mode=ExecutionMode(orm.execution_mode),
            hardware_catalog_id=orm.hardware_catalog_id,
            findings=findings,
            resource_estimate=res_est,
            roofline_analysis=roofline,
            summary=orm.summary,
        )

    # -------------------------------------------------------------------------
    # Prediction Evaluations
    # -------------------------------------------------------------------------

    def save_prediction_evaluation(
        self, evaluation: EvaluationResult, evaluation_id: str | None = None
    ) -> str:
        """Persist an EvaluationResult record, returning the evaluation_id."""
        eval_id = evaluation_id or f"eval-{uuid.uuid4().hex[:12]}"
        existing = self._session.get(PredictionEvaluationORM, eval_id)
        if existing:
            raise EntityAlreadyExistsError(f"Prediction evaluation '{eval_id}' already exists.")

        orm = PredictionEvaluationORM(
            evaluation_id=eval_id,
            created_at=datetime.now(UTC),
            target=evaluation.target.value,
            predictor_id=evaluation.predictor_id,
            predictor_type=evaluation.predictor_type,
            split_strategy=evaluation.split_strategy,
            sample_count=evaluation.sample_count,
            train_sample_count=evaluation.train_sample_count,
            test_sample_count=evaluation.test_sample_count,
            unit=evaluation.unit,
            mae=evaluation.mae,
            rmse=evaluation.rmse,
            mape_percent=evaluation.mape_percent,
            baseline_mae=evaluation.baseline_mae,
            baseline_rmse=evaluation.baseline_rmse,
            is_synthetic=evaluation.is_synthetic,
            execution_mode=evaluation.execution_mode.value,
            is_statistically_sufficient=evaluation.is_statistically_sufficient,
            notes=evaluation.notes or None,
        )
        self._session.add(orm)
        self._session.flush()
        return eval_id

    def get_prediction_evaluation(self, evaluation_id: str) -> EvaluationResult | None:
        """Retrieve an EvaluationResult by evaluation_id, returning None if not found."""
        orm = self._session.get(PredictionEvaluationORM, evaluation_id)
        if not orm:
            return None
        return self._prediction_orm_to_domain(orm)

    def list_prediction_evaluations(
        self,
        limit: int = 50,
        offset: int = 0,
        target: str | None = None,
    ) -> tuple[list[EvaluationResult], int]:
        """List prediction evaluation benchmarks with pagination and optional target filter."""
        query = select(PredictionEvaluationORM)
        count_query = select(func.count(PredictionEvaluationORM.evaluation_id))

        if target:
            query = query.where(PredictionEvaluationORM.target == target)
            count_query = count_query.where(PredictionEvaluationORM.target == target)

        total = self._session.scalar(count_query) or 0
        stmt = query.order_by(PredictionEvaluationORM.created_at.desc()).offset(offset).limit(limit)

        items = [self._prediction_orm_to_domain(orm) for orm in self._session.scalars(stmt).all()]
        return items, total

    @staticmethod
    def _prediction_orm_to_domain(orm: PredictionEvaluationORM) -> EvaluationResult:
        return EvaluationResult(
            target=PredictionTarget(orm.target),
            predictor_id=orm.predictor_id,
            predictor_type=orm.predictor_type,
            split_strategy=orm.split_strategy,
            sample_count=orm.sample_count,
            train_sample_count=orm.train_sample_count,
            test_sample_count=orm.test_sample_count,
            unit=orm.unit,
            mae=orm.mae,
            rmse=orm.rmse,
            mape_percent=orm.mape_percent,
            baseline_mae=orm.baseline_mae,
            baseline_rmse=orm.baseline_rmse,
            is_synthetic=orm.is_synthetic,
            execution_mode=ExecutionMode(orm.execution_mode),
            is_statistically_sufficient=orm.is_statistically_sufficient,
            notes=orm.notes or "",
        )

    # -------------------------------------------------------------------------
    # Optimization Runs and Pareto Frontiers
    # -------------------------------------------------------------------------

    def save_optimization(
        self,
        request: OptimizationRequest,
        result: OptimizationResult,
    ) -> None:
        """Persist an OptimizationRequest and its evaluated Pareto frontier."""
        existing = self._session.get(OptimizationRecordORM, result.request_id)
        if existing:
            raise EntityAlreadyExistsError(
                f"Optimization '{result.request_id}' is already persisted."
            )

        model_id = (
            request.search_space.models[0].model_id if request.search_space.models else "unknown"
        )

        opt_orm = OptimizationRecordORM(
            optimization_id=result.request_id,
            created_at=datetime.now(UTC),
            summary=result.summary,
            model_id=model_id,
            prompt_tokens=request.workload.prompt_tokens or 0,
            output_tokens=request.workload.output_tokens,
            total_candidates_generated=result.total_candidates_generated,
            total_candidates_evaluated=result.total_candidates_evaluated,
            feasible_candidates_count=result.feasible_candidates_count,
            infeasible_candidates_count=result.infeasible_candidates_count,
            unknown_feasibility_count=result.unknown_feasibility_count,
            recommended_config_id=result.recommended_candidate.config.config_id
            if result.recommended_candidate
            else None,
            selection_rationale=result.selection_rationale,
            has_incomparable_evidence=result.has_incomparable_evidence,
            evidence_caveats=result.evidence_caveats,
            rejections_by_reason=result.rejections_by_reason,
            request_spec_json=request.model_dump(mode="json"),
        )

        frontier_config_ids = {c.config.config_id for c in result.pareto_frontier}

        for cand in result.all_evaluations:
            # Map objective values & evidence status
            obj_vals = {k.value: v for k, v in cand.objective_values.items()}
            obj_units = {k.value: v for k, v in cand.objective_units.items()}
            ev_statuses = {k.value: v.value for k, v in cand.evidence_status.items()}
            provenances = {k.value: v.value for k, v in cand.provenance.items()}

            c_orm = OptimizationCandidateORM(
                optimization_id=result.request_id,
                config_id=cand.config.config_id,
                model_id=cand.config.model_spec.model_id,
                quantization=cand.config.quantization.value,
                concurrency=cand.config.concurrency,
                engine_type=cand.config.engine_type.value,
                tensor_parallel_size=cand.config.tensor_parallel_size,
                gpu_memory_utilization=cand.config.gpu_memory_utilization,
                extra_args_json=cand.config.extra_args or None,
                feasibility_status=cand.feasibility_status.value,
                is_pareto_optimal=cand.config.config_id in frontier_config_ids,
                has_mixed_evidence=cand.has_mixed_evidence,
                comparability_warning=cand.comparability_warning,
                trade_off_summary=cand.trade_off_summary,
                rejection_reasons=cand.rejection_reasons,
                objective_values=obj_vals,
                objective_units=obj_units,
                evidence_status=ev_statuses,
                provenance=provenances,
                resource_estimate_json=cand.resource_estimate.model_dump(mode="json")
                if cand.resource_estimate
                else None,
                roofline_result_json=cand.roofline_result.model_dump(mode="json")
                if cand.roofline_result
                else None,
                generalization_status_json=cand.generalization_status.model_dump(mode="json")
                if cand.generalization_status
                else None,
                notes=cand.notes,
            )
            opt_orm.candidates.append(c_orm)

        self._session.add(opt_orm)
        self._session.flush()

    def get_optimization(
        self, optimization_id: str
    ) -> tuple[OptimizationRequest, OptimizationResult] | None:
        """Retrieve a persisted OptimizationRequest and OptimizationResult by ID."""
        orm = self._session.get(OptimizationRecordORM, optimization_id)
        if not orm:
            return None

        request = OptimizationRequest.model_validate(orm.request_spec_json)

        # Reconstruct candidates
        all_evals: list[CandidateEvaluation] = []
        frontier_evals: list[CandidateEvaluation] = []

        # Find model spec from request to reconstruct CandidateConfig
        model_map = {m.model_id: m for m in request.search_space.models}

        for c_orm in orm.candidates:
            model_spec = model_map.get(c_orm.model_id, ModelSpec(model_id=c_orm.model_id))
            config = CandidateConfig(
                model_spec=model_spec,
                quantization=QuantizationType(c_orm.quantization),
                concurrency=c_orm.concurrency,
                engine_type=EngineType(c_orm.engine_type),
                tensor_parallel_size=c_orm.tensor_parallel_size,
                gpu_memory_utilization=c_orm.gpu_memory_utilization,
                extra_args=c_orm.extra_args_json or {},
            )

            obj_vals = {OptimizationObjective(k): v for k, v in c_orm.objective_values.items()}
            obj_units = {OptimizationObjective(k): v for k, v in c_orm.objective_units.items()}
            ev_statuses = {
                OptimizationObjective(k): EvidenceStatus(v)
                for k, v in c_orm.evidence_status.items()
            }
            provenances = {
                OptimizationObjective(k): ProvenanceType(v) for k, v in c_orm.provenance.items()
            }

            res_est = (
                ModelResourceEstimate.model_validate(c_orm.resource_estimate_json)
                if c_orm.resource_estimate_json
                else None
            )
            roofline = (
                RooflineAnalysisResult.model_validate(c_orm.roofline_result_json)
                if c_orm.roofline_result_json
                else None
            )
            gen_status = (
                GeneralizationStatus.model_validate(c_orm.generalization_status_json)
                if c_orm.generalization_status_json
                else None
            )

            cand_eval = CandidateEvaluation(
                config=config,
                feasibility_status=FeasibilityStatus(c_orm.feasibility_status),
                rejection_reasons=c_orm.rejection_reasons or [],
                objective_values=obj_vals,
                objective_units=obj_units,
                evidence_status=ev_statuses,
                provenance=provenances,
                resource_estimate=res_est,
                roofline_result=roofline,
                generalization_status=gen_status,
                trade_off_summary=c_orm.trade_off_summary,
                has_mixed_evidence=c_orm.has_mixed_evidence,
                comparability_warning=c_orm.comparability_warning,
                notes=c_orm.notes or [],
            )
            all_evals.append(cand_eval)
            if c_orm.is_pareto_optimal:
                frontier_evals.append(cand_eval)

        recommended = None
        if orm.recommended_config_id:
            for c in frontier_evals:
                if c.config.config_id == orm.recommended_config_id:
                    recommended = c
                    break

        result = OptimizationResult(
            request_id=orm.optimization_id,
            summary=orm.summary,
            total_candidates_generated=orm.total_candidates_generated,
            total_candidates_evaluated=orm.total_candidates_evaluated,
            feasible_candidates_count=orm.feasible_candidates_count,
            infeasible_candidates_count=orm.infeasible_candidates_count,
            unknown_feasibility_count=orm.unknown_feasibility_count,
            pareto_frontier=frontier_evals,
            recommended_candidate=recommended,
            selection_rationale=orm.selection_rationale,
            rejections_by_reason=orm.rejections_by_reason or {},
            all_evaluations=all_evals,
            has_incomparable_evidence=orm.has_incomparable_evidence,
            evidence_caveats=orm.evidence_caveats or [],
        )

        return request, result

    def list_optimizations(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """List summary records of persisted optimizations."""
        total_stmt = select(func.count(OptimizationRecordORM.optimization_id))
        total = self._session.scalar(total_stmt) or 0

        stmt = (
            select(OptimizationRecordORM)
            .order_by(OptimizationRecordORM.created_at.desc())
            .offset(offset)
            .limit(limit)
        )

        items = []
        for orm in self._session.scalars(stmt).all():
            items.append(
                {
                    "optimization_id": orm.optimization_id,
                    "created_at": orm.created_at,
                    "model_id": orm.model_id,
                    "prompt_tokens": orm.prompt_tokens,
                    "output_tokens": orm.output_tokens,
                    "summary": orm.summary,
                    "recommended_config_id": orm.recommended_config_id,
                    "has_incomparable_evidence": orm.has_incomparable_evidence,
                    "feasible_candidates_count": orm.feasible_candidates_count,
                    "infeasible_candidates_count": orm.infeasible_candidates_count,
                    "unknown_feasibility_count": orm.unknown_feasibility_count,
                }
            )
        return items, total
