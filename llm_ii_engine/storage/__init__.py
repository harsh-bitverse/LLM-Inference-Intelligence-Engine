"""Persistent storage package providing SQLite persistence and repositories."""

from llm_ii_engine.storage.database import (
    CURRENT_SCHEMA_VERSION,
    DatabaseManager,
    create_engine_and_session,
    get_default_database_url,
)
from llm_ii_engine.storage.models import (
    Base,
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
    SchemaVersionORM,
)
from llm_ii_engine.storage.repository import (
    EngineRepository,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    StorageError,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "Base",
    "BenchmarkSuiteORM",
    "DatabaseManager",
    "DiagnosticFindingORM",
    "DiagnosticReportORM",
    "EngineRepository",
    "EntityAlreadyExistsError",
    "EntityNotFoundError",
    "GPUDeviceReportORM",
    "HardwareReportORM",
    "MetricObservationORM",
    "OptimizationCandidateORM",
    "OptimizationRecordORM",
    "PredictionEvaluationORM",
    "RequestBenchmarkORM",
    "SchemaVersionORM",
    "StorageError",
    "create_engine_and_session",
    "get_default_database_url",
]
