# LLM Inference Intelligence Engine — Implementation Plan & Session Roadmap

## 1. Overview & Phased Roadmap Strategy

The implementation plan is designed for disciplined, incremental execution by a solo developer or pair programming team. Each session is strictly scoped with clear dependencies, explicit deliverables, and testable acceptance criteria.

The project follows a **CPU-First & Measurement-Integrity** methodology:
* The core package, domain models, metric collectors, and diagnostic engines will be fully operational and testable using the `MockEngineAdapter` on standard CPU workstations.
* Real GPU acceleration and real engine adapters (e.g., Ollama, vLLM, llama.cpp) are integrated modularly as pluggable extensions.
* Data provenance (`MEASURED`, `PREDICTED`, `HEURISTIC`) is baked into domain models from day one.

---

## 2. Reconciled Workspace Layout

The current repository is freshly initialized (`git status` clean, no commits). The implementation roadmap builds out the following unified workspace structure:

| Path | Status at Session 00 | Target Role |
|---|---|---|
| `.gitignore` | Created in Session 00 | Prevents weights, SQLite databases, venvs, and node artifacts from entering Git |
| `docs/requirements.md` | Created in Session 00 | Product specification, personas, workflows, and acceptance criteria |
| `docs/architecture.md` | Created in Session 00 | Architecture diagrams, domain schemas, API specs, and ERD |
| `docs/implementation-plan.md` | Created in Session 00 | Phased session roadmap, risk matrix, and verification plan |
| `pyproject.toml` | Target (Session 01) | Unified Python project manifest (FastAPI, Pydantic, SQLAlchemy, NumPy, Pytest) |
| `llm_ii_engine/` | Target (Session 01+) | Core domain package: discovery, benchmark, adapters, diagnostics, prediction, optimizer, storage |
| `backend/` | Target (Session 08) | FastAPI REST & SSE endpoints, lifespan, and dependency injection |
| `frontend/` | Target (Session 09) | React + TypeScript + Vite single-page dashboard |
| `tests/` | Target (Session 01+) | Unit, integration, and end-to-end automated tests |

---

## 3. Session-by-Session Implementation Roadmap

### Session 00: Project Specification and Repository Audit *(Current Session)*
* **Objective**: Audit workspace, establish architecture, data schemas, and engineering constraints; create foundational planning documentation.
* **Deliverables**:
  * Comprehensive `.gitignore` protecting against credentials, datasets, and large weights.
  * `docs/requirements.md` (Product specification & acceptance criteria).
  * `docs/architecture.md` (System design, Mermaid diagrams, schemas, interfaces).
  * `docs/implementation-plan.md` (Session roadmap, risk matrix, reconciliation).
* **Acceptance Criteria**:
  * Git status is clean and reflects created documentation.
  * Architectural documentation strictly defines data provenance and CPU-first portability.
  * No heavy dependencies, models, or runtime code prematurely introduced.

---

### Session 01: Project Scaffolding, Core Infrastructure & Mock Engine Adapter
* **Objective**: Initialize Python package structure, define core domain models (Pydantic v2), and implement the deterministic `MockEngineAdapter`.
* **Deliverables**:
  * `pyproject.toml` with dependencies (`pydantic>=2.7`, `pytest`, `numpy`, `scipy`).
  * `llm_ii_engine/core/enums.py` (`ProvenanceType`, `EngineType`, `MetricType`, `Severity`).
  * `llm_ii_engine/core/schemas.py` (`HardwareProfile`, `ModelSpec`, `EngineConfig`, `WorkloadProfile`, `RequestMeasurement`, `BenchmarkAggregateMetrics`, `BenchmarkRun`).
  * `llm_ii_engine/adapters/base.py` (`InferenceEngineAdapter` async protocol).
  * `llm_ii_engine/adapters/mock.py` (`MockEngineAdapter` with configurable TTFT, ITL distribution, and memory simulation).
  * Initial unit test suite: `tests/unit/test_mock_adapter.py` and `tests/unit/test_schemas.py`.
* **Verification & Acceptance Criteria**:
  * `pytest tests/unit/` passes 100% on CPU in $< 3$ seconds.
  * `MockEngineAdapter` streams synthetic tokens with realistic simulated monotonic timestamps and respects prompt/generation length configurations.

---

### Session 02: Hardware Capability Discovery & Roofline Profiling
* **Objective**: Implement host hardware introspection (CPU, RAM, GPU) and theoretical compute/bandwidth ceiling catalog.
* **Deliverables**:
  * `llm_ii_engine/discovery/profiler.py` (introspects CPU model, core counts, RAM, and NVIDIA GPU topology via `pynvml` / WMI / `sysfs` with safe fallbacks).
  * `llm_ii_engine/discovery/hardware_db.py` (hardware specifications lookup table for memory bandwidth in GB/s and peak FP16/INT8 TFLOPs).
  * Graceful fallback logic for CPU-only environments (such as developer laptops or CI nodes without GPU).
  * `tests/unit/test_discovery.py` testing both physical inspection and simulated hardware profiles.
* **Verification & Acceptance Criteria**:
  * Hardware discovery executes on Windows CPU host without errors, returning structured `HardwareProfile` JSON.
  * Hardware database resolves nominal bandwidth and TFLOPs for common GPUs (A100, H100, RTX 4090, RTX 3090, T4) and fallback CPU memory channels.

---

### Session 03: Benchmark Execution Engine & Metric Instrumentation
* **Objective**: Build the asynchronous benchmark runner, concurrency worker pool, high-resolution timing collector, and memory sampler.
* **Deliverables**:
  * `llm_ii_engine/benchmark/collector.py` (captures high-precision monotonic nanosecond timestamps for prefill TTFT and per-token decode ITL; calculates p50, p90, p95, p99 percentiles, TPOT, and throughput).
  * `llm_ii_engine/benchmark/workload.py` (deterministic workload generator supporting fixed and Gaussian/Poisson distributed prompt and output lengths).
  * `llm_ii_engine/benchmark/runner.py` (async orchestrator managing concurrency levels, request dispatch, timeouts, and error handling).
  * `llm_ii_engine/benchmark/memory.py` (background memory sampler recording baseline and peak watermarks).
  * `tests/unit/test_metrics_collector.py` and `tests/integration/test_benchmark_runner.py`.
* **Verification & Acceptance Criteria**:
  * Executing a 50-request benchmark at concurrency 4 against `MockEngineAdapter` completes successfully.
  * Metric collector computes valid TTFT, ITL percentiles, TPOT, and throughput with `provenance: MEASURED`.
  * Request failures or timeouts are recorded without crashing the benchmark harness.

---

### Session 04: Real Engine Adapters (Ollama & llama.cpp)
* **Objective**: Implement real HTTP and streaming adapters for popular local inference engines.
* **Deliverables**:
  * `llm_ii_engine/adapters/ollama.py` (asynchronous HTTP streaming client interfacing with Ollama `/api/generate` and `/api/chat`).
  * `llm_ii_engine/adapters/llamacpp.py` (client interfacing with `llama-server` or `llama-cpp-python` OpenAI-compatible endpoints).
  * Dynamic adapter factory: `get_adapter(engine_type: str) -> InferenceEngineAdapter`.
  * Integration tests using mocked HTTP server (`respx` or `httpx` mock transport).
* **Verification & Acceptance Criteria**:
  * Unit tests with mocked HTTP streaming simulate real token chunks and verify timestamp extraction.
  * Health check verifies whether local engine daemon is alive and reports clear errors when offline.

---

### Session 05: Evidence-Based Bottleneck Diagnostics & Roofline Analysis
* **Objective**: Implement the diagnostic engine mapping benchmark observations onto the Roofline Model and flagging root-cause bottlenecks.
* **Deliverables**:
  * `llm_ii_engine/diagnostics/roofline.py` (computes Arithmetic Intensity for prefill and decode phases; calculates machine balance point; maps to compute vs. memory bandwidth limits).
  * `llm_ii_engine/diagnostics/kv_cache.py` (estimates KV cache memory footprint: $2 \times b \times l \times h \times d_{head} \times \text{bytes}$; evaluates headroom).
  * `llm_ii_engine/diagnostics/rules.py` (diagnostic rule engine classifying memory saturation, under-subscription, queue contention, and context thrashing).
  * `tests/unit/test_roofline_diagnostics.py`.
* **Verification & Acceptance Criteria**:
  * Accurately categorizes batch=1 decode as `MEMORY_BOUND` with $> 90\%$ utilization of memory bandwidth.
  * Accurately categorizes large-prompt batched prefill as `COMPUTE_BOUND`.
  * Emits structured diagnostic findings with severity, empirical evidence, and actionable recommendations.

---

### Session 06: Performance Prediction with Held-Out Evaluation
* **Objective**: Implement the hybrid analytical/empirical performance predictor and train/test generalization evaluator.
* **Deliverables**:
  * `llm_ii_engine/prediction/analytical.py` (first-principles transformer scaling calculations for parameter footprint, KV cache bytes, and compute operations).
  * `llm_ii_engine/prediction/empirical.py` (parametric regression / gradient booster predicting TTFT and throughput from model size, quantization, prompt length, output length, and concurrency).
  * `llm_ii_engine/prediction/evaluation.py` (train/test splitter, MAE, MAPE, $R^2$ evaluation pipeline).
  * `tests/unit/test_prediction_models.py`.
* **Verification & Acceptance Criteria**:
  * Prediction outputs are strictly marked with `provenance: PREDICTED`.
  * Evaluator outputs quantitative validation metrics (MAE, MAPE, RMSE) on held-out benchmark splits.
  * Model predicts latency for unseen sequence lengths and concurrency tiers without crashing.

---

### Session 07: Constrained Configuration Recommender & Pareto Optimizer
* **Objective**: Implement the multi-objective Pareto optimization engine to recommend optimal deployment configurations under user SLA constraints.
* **Deliverables**:
  * `llm_ii_engine/optimizer/pareto.py` (efficient non-dominated sorting algorithm across TTFT, TPOT, Throughput, and VRAM).
  * `llm_ii_engine/optimizer/recommender.py` (applies user SLA filters: max TTFT, max TPOT, max memory, min throughput; ranks Pareto-optimal candidate configurations).
  * `tests/unit/test_pareto_optimizer.py`.
* **Verification & Acceptance Criteria**:
  * Correctly eliminates dominated configurations and returns only non-dominated frontier recipes.
  * Given tight constraints (e.g., VRAM $\le 8\,\text{GB}$), successfully filters out invalid models/quantizations and returns valid recipes with human-readable rationale.

---

### Session 08: SQLite Persistence Layer & FastAPI REST Backend
* **Objective**: Implement the database persistence schema and expose all engine capabilities via a FastAPI REST service.
* **Deliverables**:
  * `llm_ii_engine/storage/database.py` and `models.py` (SQLAlchemy / SQLModel SQLite repository).
  * `backend/main.py` and routers (`hardware.py`, `benchmarks.py`, `diagnostics.py`, `predictions.py`, `recommendations.py`).
  * Streaming SSE endpoint for real-time benchmark progress monitoring.
  * `tests/integration/test_sqlite_persistence.py` and `tests/integration/test_api_endpoints.py`.
* **Verification & Acceptance Criteria**:
  * API starts cleanly and passes all endpoint integration tests (`TestClient`).
  * Benchmark runs persist to SQLite and are retrievable across server restarts with full metric fidelity.

---

### Session 09: React + TypeScript Interactive Analytics Dashboard
* **Objective**: Build the single-page web dashboard visualizing benchmarks, latency distributions, roofline charts, and Pareto recommendations.
* **Deliverables**:
  * `frontend/` React + Vite + TypeScript application.
  * Views:
    * *Hardware & Environment Overview*: System specs, theoretical ceilings.
    * *Benchmark Launcher*: Concurrency, model, and workload configuration.
    * *Live Telemetry & Comparison Matrix*: Interactive CDFs, TTFT/ITL histograms, throughput vs. concurrency plots.
    * *Roofline Explorer*: Interactive visualizer showing prefill and decode operational intensity points.
    * *Configuration Sizer & Recommender*: Interactive sliders for latency and VRAM constraints with recommended recipes.
* **Verification & Acceptance Criteria**:
  * `npm run build` succeeds with zero TypeScript or bundling errors.
  * UI connects to FastAPI backend and renders benchmark history and charts.

---

### Session 10: End-to-End Integration, Verification & Documentation
* **Objective**: Perform end-to-end testing across all subsystems, assemble CLI tooling, and finalize deployment documentation.
* **Deliverables**:
  * CLI entrypoint (`iie` command: `iie discover`, `iie bench`, `iie diagnose`, `iie predict`, `iie recommend`, `iie serve`).
  * End-to-end integration test verifying discovery $\to$ benchmark $\to$ diagnostics $\to$ prediction $\to$ recommendation pipeline.
  * Comprehensive `README.md` with quickstart guides and architecture overview.
* **Verification & Acceptance Criteria**:
  * Full test suite (`pytest`) passes with $\ge 85\%$ coverage on CPU environment.
  * Complete workflow executes smoothly via both CLI and Web Dashboard.

---

## 4. Risk Analysis & Mitigation Matrix

| Risk | Impact | Likelihood | Mitigation Strategy |
|---|---|---|---|
| **GPU / Driver Incompatibility** (e.g. Windows CUDA issues, differing driver versions) | High | High | Strict CPU-first architecture; `MockEngineAdapter` allows 100% CI/CD and developer testing without GPU drivers. Real GPU calls use graceful exception handling and telemetry fallbacks. |
| **Engine Runtime Portability** (e.g. vLLM has limited Windows native support) | Medium | High | Decouple engine runners behind HTTP streaming adapters (e.g. Ollama or Dockerized vLLM). Benchmarking interacts over network/HTTP APIs rather than requiring direct C++ bindings on host. |
| **Timing Inaccuracy & System Noise** (Jitter, OS scheduling delays) | Medium | Medium | Use monotonic nanosecond clock (`time.perf_counter_ns()`); run warmup requests before timing; record full ITL distribution rather than only aggregate averages; report p50, p95, p99. |
| **Memory Leak in Repeated Benchmarks** (VRAM not freed after crashed runs) | High | Medium | Execute benchmarks in isolated subprocesses where appropriate; enforce timeout limits; implement explicit garbage collection and cache flushing hooks in engine teardown. |
| **Predictive Model Overfitting** | Medium | Medium | Validate predictive models against strict held-out test splits (unseen sequence lengths or concurrency tiers); report MAE/MAPE prominently; combine analytical scaling with empirical regressors. |
| **Accidental Repository Bloat** (Committing weights, datasets, or SQLite binaries) | High | Medium | Comprehensive `.gitignore` established in Session 00; strict pre-commit checks; store model weights and local runs outside Git-tracked paths. |

---

## 5. Verification Commands for Subsequent Sessions

* **Python Environment Setup**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install -e ".[dev]"
  ```
* **Run Unit Tests**:
  ```powershell
  pytest tests/unit/ -v
  ```
* **Run Full Integration Test Suite**:
  ```powershell
  pytest tests/ -v --cov=llm_ii_engine
  ```
* **Start API Server**:
  ```powershell
  uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
  ```
* **Build Frontend**:
  ```powershell
  cd frontend
  npm install
  npm run build
  ```
