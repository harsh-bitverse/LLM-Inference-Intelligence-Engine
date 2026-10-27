# Session 08 Design Document — Persistent Storage and REST API Foundation

## 1. System Overview

Session 08 introduces the persistence subsystem and REST API foundation for the **LLM Inference Intelligence Engine (IIE)**. The goal is to expose existing measurement, diagnostics, prediction evaluation, and constrained optimization capabilities through typed, documented HTTP interfaces backed by an ACID-compliant relational SQLite store.

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│   /health, /api/v1/hardware, /api/v1/benchmarks,            │
│   /api/v1/diagnostics, /api/v1/predictions, /api/v1/optimize │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    ┌──────────────────────┐        ┌──────────────────────┐
    │   Domain Services    │        │  Engine Repository   │
    │  - HardwareProfiler  │        │   (Unit of Work &    │
    │  - BenchmarkRunner   │        │   Domain Mappings)   │
    │  - DiagnosticEngine  │        └──────────┬───────────┘
    │  - ConfigOptimizer   │                   │
    └──────────────────────┘                   ▼
                                    ┌──────────────────────┐
                                    │    SQLAlchemy 2.x    │
                                    │      (SQLite)        │
                                    └──────────────────────┘
```

---

## 2. Storage Architecture & Relational Entities

### 2.1 Database Technology
- **Engine**: SQLite via SQLAlchemy 2.x (`sqlite:///...`).
- **Connection Configuration**:
  - `PRAGMA foreign_keys = ON;` (Strict referential integrity enforcement).
  - `PRAGMA journal_mode = WAL;` (Concurrent readers, non-blocking writes).
  - `PRAGMA synchronous = NORMAL;` (High durability with minimal I/O stalls).
  - `PRAGMA busy_timeout = 5000;` (Graceful lock wait under concurrency).
- **Isolation**: Supports memory isolation (`sqlite:///:memory:`) and file isolation (`sqlite:///path/to/test.db`) for test runs without touching development databases.

### 2.2 Relational Entity Schema

1. **`schema_version`**:
   - `version` (INT, PK): Current migration schema version.
   - `applied_at` (TIMESTAMP): UTC timestamp of schema creation.

2. **`hardware_reports`**:
   - `report_id` (VARCHAR(64), PK): Unique hardware discovery identifier.
   - `created_at` (TIMESTAMP WITH TIMEZONE): Discovery timestamp.
   - `os_name`, `os_version`, `os_arch`, `python_version` (VARCHAR): Operating system details.
   - `discovery_source` (VARCHAR(32)): `DETECTION`, `SIMULATION`, `FALLBACK`.
   - `is_simulated` (BOOLEAN): Distinguishes real from synthetic profiles.
   - `cpu_model` (VARCHAR(256)), `cpu_physical_cores` (INT), `cpu_logical_cores` (INT), `cpu_frequency_mhz` (FLOAT).
   - `total_ram_bytes` (BIGINT), `available_ram_bytes` (BIGINT).
   - `environment_notes` (JSON): Sanitized host notes (no secrets/passwords).

3. **`gpu_device_reports`** (1-to-many child of `hardware_reports`):
   - `id` (INTEGER, PK, AUTOINCREMENT).
   - `report_id` (VARCHAR(64), FK -> `hardware_reports.report_id`, ON DELETE CASCADE).
   - `gpu_index` (INT): Physical device index.
   - `name` (VARCHAR(128)), `vendor` (VARCHAR(64)).
   - `memory_total_bytes` (BIGINT).
   - `compute_capability`, `cuda_driver_version`, `device_id` (VARCHAR, Nullable).
   - `is_simulated` (BOOLEAN).
   - `catalog_id` (VARCHAR(64), Nullable): Catalog hardware reference.
   - `catalog_spec_json` (JSON, Nullable).
   - `confidence_notes` (JSON, Nullable).

4. **`benchmark_suites`**:
   - `suite_id` (VARCHAR(64), PK): Benchmark suite ID (e.g. `bench-xxx`).
   - `created_at` (TIMESTAMP WITH TIMEZONE).
   - `model_id` (VARCHAR(128), INDEX): Tested model identifier.
   - `engine_type` (VARCHAR(32), INDEX): `mock`, `ollama`, `llamacpp`, `vllm`.
   - `execution_mode` (VARCHAR(32)): `REAL`, `SIMULATED`.
   - `clock_source` (VARCHAR(32)): `MONOTONIC_REAL`, `SYNTHETIC_VIRTUAL`.
   - `concurrency` (INT): Max concurrent client requests.
   - `total_requests_executed`, `successful_requests`, `failed_requests` (INT).
   - `start_time`, `end_time` (TIMESTAMP WITH TIMEZONE), `duration_seconds` (FLOAT).
   - `workload_prompt_tokens`, `workload_output_tokens` (INT).
   - `workload_spec_json` (JSON): Full workload specification.
   - `metrics_json` (JSON): Aggregate metrics (TTFT, TPOT, Throughput, VRAM watermarks).
   - `notes` (JSON, Nullable): Run warnings or timing notes.
   - `hardware_report_id` (VARCHAR(64), FK -> `hardware_reports.report_id`, Nullable).

5. **`request_benchmarks`** (1-to-many child of `benchmark_suites`):
   - `request_id` (VARCHAR(64), PK): Request identifier.
   - `suite_id` (VARCHAR(64), FK -> `benchmark_suites.suite_id`, ON DELETE CASCADE).
   - `status` (VARCHAR(32)): `SUCCESS`, `ERROR`, `TIMEOUT`, `CANCELLED`.
   - `submission_timestamp`, `start_timestamp`, `first_token_timestamp`, `completion_timestamp` (TIMESTAMP).
   - `prompt_tokens`, `completion_tokens`, `total_tokens` (INT).
   - `ttft_ms`, `tpot_ms`, `total_latency_ms` (FLOAT, Nullable).
   - `error_message` (TEXT, Nullable).
   - `inter_token_latencies_ms` (JSON, Nullable): Per-token delta distributions.

6. **`metric_observations`** (Individual metrics with provenance):
   - `observation_id` (VARCHAR(64), PK).
   - `suite_id` (VARCHAR(64), FK -> `benchmark_suites.suite_id`, ON DELETE CASCADE, Nullable).
   - `created_at` (TIMESTAMP WITH TIMEZONE).
   - `name` (VARCHAR(64), INDEX): Metric name (e.g., `ttft_p50_ms`).
   - `value` (FLOAT), `unit` (VARCHAR(32)).
   - `provenance` (VARCHAR(32)): `MEASURED`, `PREDICTED`, `HEURISTIC`.
   - `execution_mode` (VARCHAR(32)): `REAL`, `SIMULATED`.
   - `clock_source` (VARCHAR(32)).

7. **`diagnostic_reports`**:
   - `report_id` (VARCHAR(64), PK): Diagnostic report identifier.
   - `benchmark_suite_id` (VARCHAR(64), FK -> `benchmark_suites.suite_id`, Nullable, INDEX).
   - `created_at` (TIMESTAMP WITH TIMEZONE).
   - `model_id` (VARCHAR(128), INDEX).
   - `primary_bottleneck` (VARCHAR(64)): `MEMORY_BANDWIDTH`, `COMPUTE`, `LATENCY_OVERHEAD`.
   - `severity` (VARCHAR(32)): `HEALTHY`, `MINOR_BOTTLENECK`, `SEVERE_BOTTLENECK`.
   - `operational_intensity_prefill`, `operational_intensity_decode` (FLOAT, Nullable).
   - `roofline_result_json`, `resource_estimate_json`, `notes` (JSON, Nullable).

8. **`diagnostic_findings`** (1-to-many child of `diagnostic_reports`):
   - `finding_id` (VARCHAR(64), PK).
   - `report_id` (VARCHAR(64), FK -> `diagnostic_reports.report_id`, ON DELETE CASCADE).
   - `regime` (VARCHAR(64)), `severity` (VARCHAR(32)).
   - `headline` (VARCHAR(256)), `detail` (TEXT).
   - `metrics_observed_json`, `counter_evidence`, `remediation_suggestions` (JSON).

9. **`prediction_evaluations`**:
   - `evaluation_id` (VARCHAR(64), PK).
   - `created_at` (TIMESTAMP WITH TIMEZONE).
   - `target` (VARCHAR(64), INDEX): Predicted target.
   - `model_type` (VARCHAR(128)): Regressor or baseline class name.
   - `feature_names` (JSON), `split_strategy` (VARCHAR(64)).
   - `train_record_count`, `val_record_count` (INT).
   - `train_metrics_json`, `val_metrics_json` (JSON): MAE, RMSE, MAPE.
   - `ood_warning_count` (INT).

10. **`optimizations`**:
    - `optimization_id` (VARCHAR(64), PK): Optimization request identifier.
    - `created_at` (TIMESTAMP WITH TIMEZONE).
    - `summary` (TEXT): Executive recommendation summary.
    - `model_id` (VARCHAR(128), INDEX), `prompt_tokens` (INT), `output_tokens` (INT).
    - `total_candidates_generated`, `total_candidates_evaluated`, `feasible_candidates_count`, `infeasible_candidates_count`, `unknown_feasibility_count` (INT).
    - `recommended_config_id` (VARCHAR(128), Nullable).
    - `selection_rationale` (TEXT, Nullable).
    - `has_incomparable_evidence` (BOOLEAN).
    - `evidence_caveats`, `rejections_by_reason`, `request_spec_json` (JSON).

11. **`optimization_candidates`** (1-to-many child of `optimizations`):
    - `id` (INTEGER, PK, AUTOINCREMENT).
    - `optimization_id` (VARCHAR(64), FK -> `optimizations.optimization_id`, ON DELETE CASCADE).
    - `config_id` (VARCHAR(128), INDEX).
    - `model_id` (VARCHAR(128)), `quantization` (VARCHAR(32)), `concurrency` (INT), `engine_type` (VARCHAR(32)), `tensor_parallel_size` (INT).
    - `feasibility_status` (VARCHAR(32)): `FEASIBLE`, `PROVEN_INFEASIBLE`, `ESTIMATED_INFEASIBLE`, `UNKNOWN_FEASIBILITY`.
    - `is_pareto_optimal` (BOOLEAN), `has_mixed_evidence` (BOOLEAN).
    - `comparability_warning`, `trade_off_summary` (TEXT, Nullable).
    - `rejection_reasons`, `objective_values`, `objective_units`, `evidence_status`, `provenance`, `resource_estimate_json`, `roofline_result_json`, `notes` (JSON).

---

## 3. REST API Interface Specification

| Method | Path | Description | Status Codes |
|---|---|---|---|
| `GET` | `/health`, `/api/v1/health` | Liveness and database readiness probe | 200, 503 |
| `GET` | `/api/v1/hardware` | Retrieve stored hardware report or run discovery | 200, 404 |
| `GET` | `/api/v1/hardware/catalog` | List theoretical GPU specifications catalog | 200 |
| `GET` | `/api/v1/benchmarks` | List benchmark suite runs (paginated & filtered) | 200, 422 |
| `GET` | `/api/v1/benchmarks/{id}` | Retrieve benchmark suite and request metrics | 200, 404 |
| `POST` | `/api/v1/benchmarks/run` | Execute bounded benchmark run or queue job | 200, 202, 422 |
| `GET` | `/api/v1/benchmarks/jobs` | List active and historical benchmark jobs | 200 |
| `GET` | `/api/v1/benchmarks/jobs/{job_id}` | Check status of async benchmark job | 200, 404 |
| `POST` | `/api/v1/benchmarks/jobs/{job_id}/cancel` | Request cancellation of running job | 200, 404 |
| `GET` | `/api/v1/benchmarks/jobs/{job_id}/stream` | Server-Sent Events (SSE) live token telemetry | 200, 404 |
| `GET` | `/api/v1/diagnostics` | List historical diagnostic reports | 200 |
| `GET` | `/api/v1/diagnostics/{id}` | Retrieve diagnostic report and findings | 200, 404 |
| `POST` | `/api/v1/diagnostics/run` | Run diagnostic analysis on suite or model | 200, 400, 404, 422 |
| `GET` | `/api/v1/predictions/evaluations` | List prediction model evaluations | 200 |
| `GET` | `/api/v1/predictions/evaluations/{id}` | Retrieve specific prediction evaluation | 200, 404 |
| `POST` | `/api/v1/predictions/evaluations` | Record a model evaluation result | 201, 422 |
| `POST` | `/api/v1/optimize/recommend` | Run Pareto configuration optimization | 200, 400, 422 |
| `GET` | `/api/v1/optimize/history` | List historical optimization runs | 200 |
| `GET` | `/api/v1/optimize/{id}` | Retrieve persisted optimization recommendation | 200, 404 |

---

## 4. Epistemic Integrity and Error Semantics

1. **Non-Failure Domain Outcomes**:
   - `UNKNOWN_FEASIBILITY`: Returned in 200 OK payloads when hardware specifications or target metrics are insufficient to prove feasibility.
   - `INSUFFICIENT_EVIDENCE`: Returned in 200 OK payloads when empirical models detect out-of-distribution feature spaces.
   - Empty Pareto Frontier: Returned in 200 OK when hard constraints filter out all candidates.

2. **Standardized Error Responses**:
   ```json
   {
     "error": "RESOURCE_NOT_FOUND",
     "message": "Benchmark suite 'bench-invalid' was not found",
     "details": null
   }
   ```
   - 400 Bad Request: Invalid parameters, negative token counts, or combinatorial explosion.
   - 404 Not Found: Requested entity does not exist.
   - 409 Conflict: Duplicate write of an immutable resource.
   - 422 Unprocessable Entity: Pydantic validation failure.
   - 500 Internal Server Error: Sanitized error response without stack trace or secrets.

3. **Telemetry Streaming (SSE) Guarantees**:
   - Streams discrete `TokenStreamEvent`s directly from adapter generators.
   - Distinguishes real monotonic clock from synthetic clock.
   - Supports graceful client disconnection without orphaned generator loops.
