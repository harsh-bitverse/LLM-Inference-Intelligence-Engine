"""SQLAlchemy ORM models for the LLM Inference Intelligence Engine persistence layer.

Defines relational entities for hardware discovery reports, benchmark suites,
request-level measurements, metric observations, diagnostics, prediction evaluations,
and configuration optimization results.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all engine ORM models."""

    pass


class SchemaVersionORM(Base):
    """Tracks applied database schema versions for reproducible migrations."""

    __tablename__ = "schema_version"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    description: Mapped[str] = mapped_column(String(256), nullable=True)


class HardwareReportORM(Base):
    """Persisted hardware capability discovery report."""

    __tablename__ = "hardware_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_mode: Mapped[str] = mapped_column(String(32))

    # Host system
    os_name: Mapped[str] = mapped_column(String(64))
    os_version: Mapped[str] = mapped_column(String(128))
    os_release: Mapped[str] = mapped_column(String(128))
    os_arch: Mapped[str] = mapped_column(String(64))
    python_version: Mapped[str] = mapped_column(String(64))

    # CPU topology
    cpu_model: Mapped[str] = mapped_column(String(256))
    cpu_vendor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cpu_physical_cores: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cpu_logical_cores: Mapped[int] = mapped_column(Integer)
    cpu_frequency_mhz: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Host Memory
    total_ram_bytes: Mapped[int] = mapped_column(BigInteger)
    available_ram_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # Telemetry and matched catalog metadata
    telemetry_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    matched_catalog_specs_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    missing_or_incomplete_fields: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    warnings: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # 1-to-many relationship with GPU devices
    gpus: Mapped[list["GPUDeviceReportORM"]] = relationship(
        "GPUDeviceReportORM",
        back_populates="hardware_report",
        cascade="all, delete-orphan",
        order_by="GPUDeviceReportORM.gpu_index",
    )


class GPUDeviceReportORM(Base):
    """Persisted GPU device metadata."""

    __tablename__ = "gpu_device_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("hardware_reports.report_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    gpu_index: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(128))
    vendor: Mapped[str] = mapped_column(String(64))
    total_memory_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    driver_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    uuid: Mapped[str | None] = mapped_column(String(128), nullable=True)
    pci_bus_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    compute_capability: Mapped[str | None] = mapped_column(String(32), nullable=True)
    detection_source: Mapped[str] = mapped_column(String(64))

    hardware_report: Mapped["HardwareReportORM"] = relationship(
        "HardwareReportORM", back_populates="gpus"
    )


class BenchmarkSuiteORM(Base):
    """Persisted benchmark execution suite with aggregate metrics and operational summary."""

    __tablename__ = "benchmark_suites"

    suite_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    status: Mapped[str] = mapped_column(String(32))
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    engine_type: Mapped[str] = mapped_column(String(32), index=True)
    execution_mode: Mapped[str] = mapped_column(String(32))
    clock_source: Mapped[str] = mapped_column(String(32))
    provenance: Mapped[str] = mapped_column(String(32))
    concurrency: Mapped[int] = mapped_column(Integer)
    total_requests: Mapped[int] = mapped_column(Integer)
    warmup_requests: Mapped[int] = mapped_column(Integer)
    successful_requests: Mapped[int] = mapped_column(Integer)
    failed_requests: Mapped[int] = mapped_column(Integer)
    wall_time_seconds: Mapped[float] = mapped_column(Float)

    # Aggregates
    aggregate_tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    aggregate_output_tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    request_throughput_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_p90_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_p99_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttfc_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttfc_p90_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttfc_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttfc_p99_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunk_latency_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tpot_mean_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_p90_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_p99_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    memory_watermark_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    warnings: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Optional foreign key to hardware report
    hardware_report_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("hardware_reports.report_id", ondelete="SET NULL"),
        nullable=True,
    )

    requests: Mapped[list["RequestBenchmarkORM"]] = relationship(
        "RequestBenchmarkORM",
        back_populates="suite",
        cascade="all, delete-orphan",
        order_by="RequestBenchmarkORM.request_id",
    )
    observations: Mapped[list["MetricObservationORM"]] = relationship(
        "MetricObservationORM",
        back_populates="suite",
        cascade="all, delete-orphan",
    )
    diagnostic_reports: Mapped[list["DiagnosticReportORM"]] = relationship(
        "DiagnosticReportORM",
        back_populates="suite",
    )


class RequestBenchmarkORM(Base):
    """Persisted request-level benchmark observation."""

    __tablename__ = "request_benchmarks"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    suite_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("benchmark_suites.suite_id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
        nullable=False,
    )
    success: Mapped[bool] = mapped_column(Boolean)
    is_warmup: Mapped[bool] = mapped_column(Boolean, default=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    requested_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int] = mapped_column(Integer)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttfc_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tpot_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float)
    output_tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    inter_token_latencies_ms: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    itl_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_p90_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_p99_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_latencies_ms: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    chunk_latency_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunk_latency_p90_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunk_latency_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunk_latency_p99_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    server_stats_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    token_count_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    prompt_token_count_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    clock_source: Mapped[str] = mapped_column(String(32))
    execution_mode: Mapped[str] = mapped_column(String(32))
    provenance: Mapped[str] = mapped_column(String(32))
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    incomplete_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    suite: Mapped["BenchmarkSuiteORM"] = relationship(
        "BenchmarkSuiteORM", back_populates="requests"
    )


class MetricObservationORM(Base):
    """Persisted discrete metric observation with provenance and unit."""

    __tablename__ = "metric_observations"

    observation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    suite_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("benchmark_suites.suite_id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    name: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32))
    provenance: Mapped[str] = mapped_column(String(32))  # MEASURED, PREDICTED, HEURISTIC
    execution_mode: Mapped[str] = mapped_column(String(32))  # REAL, SIMULATED
    clock_source: Mapped[str] = mapped_column(String(32))

    suite: Mapped["BenchmarkSuiteORM | None"] = relationship(
        "BenchmarkSuiteORM", back_populates="observations"
    )


class DiagnosticReportORM(Base):
    """Persisted performance bottleneck diagnostic report."""

    __tablename__ = "diagnostic_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    benchmark_suite_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("benchmark_suites.suite_id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    engine_type: Mapped[str] = mapped_column(String(32))
    execution_mode: Mapped[str] = mapped_column(String(32))
    hardware_catalog_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    roofline_analysis_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    resource_estimate_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    suite: Mapped["BenchmarkSuiteORM | None"] = relationship(
        "BenchmarkSuiteORM", back_populates="diagnostic_reports"
    )
    findings: Mapped[list["DiagnosticFindingORM"]] = relationship(
        "DiagnosticFindingORM",
        back_populates="report",
        cascade="all, delete-orphan",
    )


class DiagnosticFindingORM(Base):
    """Individual evidence-backed bottleneck finding."""

    __tablename__ = "diagnostic_findings"

    finding_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    report_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("diagnostic_reports.report_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    category: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(256))
    message: Mapped[str] = mapped_column(Text)
    confidence: Mapped[str] = mapped_column(String(32))
    confidence_rationale: Mapped[str] = mapped_column(Text)
    supporting_observations_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    assumptions: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    limitations: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    suggested_experiments: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    report: Mapped["DiagnosticReportORM"] = relationship(
        "DiagnosticReportORM", back_populates="findings"
    )


class PredictionEvaluationORM(Base):
    """Persisted empirical prediction model validation evaluation."""

    __tablename__ = "prediction_evaluations"

    evaluation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    target: Mapped[str] = mapped_column(String(64), index=True)
    predictor_id: Mapped[str] = mapped_column(String(128), index=True)
    predictor_type: Mapped[str] = mapped_column(String(128))
    split_strategy: Mapped[str] = mapped_column(String(64))
    sample_count: Mapped[int] = mapped_column(Integer)
    train_sample_count: Mapped[int] = mapped_column(Integer)
    test_sample_count: Mapped[int] = mapped_column(Integer)
    unit: Mapped[str] = mapped_column(String(32))
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    mape_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_mode: Mapped[str] = mapped_column(String(32))
    is_statistically_sufficient: Mapped[bool] = mapped_column(Boolean)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class OptimizationRecordORM(Base):
    """Persisted optimization run and Pareto-frontier recommendation."""

    __tablename__ = "optimizations"

    optimization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    summary: Mapped[str] = mapped_column(Text)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int] = mapped_column(Integer)
    total_candidates_generated: Mapped[int] = mapped_column(Integer)
    total_candidates_evaluated: Mapped[int] = mapped_column(Integer)
    feasible_candidates_count: Mapped[int] = mapped_column(Integer)
    infeasible_candidates_count: Mapped[int] = mapped_column(Integer)
    unknown_feasibility_count: Mapped[int] = mapped_column(Integer)
    recommended_config_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    selection_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    has_incomparable_evidence: Mapped[bool] = mapped_column(Boolean, default=False)
    evidence_caveats: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    rejections_by_reason: Mapped[dict[str, int] | None] = mapped_column(JSON, nullable=True)
    request_spec_json: Mapped[dict[str, Any]] = mapped_column(JSON)

    candidates: Mapped[list["OptimizationCandidateORM"]] = relationship(
        "OptimizationCandidateORM",
        back_populates="optimization",
        cascade="all, delete-orphan",
    )


class OptimizationCandidateORM(Base):
    """Evaluated inference configuration candidate for an optimization request."""

    __tablename__ = "optimization_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    optimization_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("optimizations.optimization_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    config_id: Mapped[str] = mapped_column(String(128), index=True)
    model_id: Mapped[str] = mapped_column(String(128))
    quantization: Mapped[str] = mapped_column(String(32))
    concurrency: Mapped[int] = mapped_column(Integer)
    engine_type: Mapped[str] = mapped_column(String(32))
    tensor_parallel_size: Mapped[int] = mapped_column(Integer)
    gpu_memory_utilization: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra_args_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    feasibility_status: Mapped[str] = mapped_column(String(32))
    is_pareto_optimal: Mapped[bool] = mapped_column(Boolean, default=False)
    has_mixed_evidence: Mapped[bool] = mapped_column(Boolean, default=False)
    comparability_warning: Mapped[str | None] = mapped_column(Text, nullable=True)
    trade_off_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    rejection_reasons: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    objective_values: Mapped[dict[str, float]] = mapped_column(JSON)
    objective_units: Mapped[dict[str, str]] = mapped_column(JSON)
    evidence_status: Mapped[dict[str, str]] = mapped_column(JSON)
    provenance: Mapped[dict[str, str]] = mapped_column(JSON)
    resource_estimate_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    roofline_result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    generalization_status_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    optimization: Mapped["OptimizationRecordORM"] = relationship(
        "OptimizationRecordORM", back_populates="candidates"
    )
