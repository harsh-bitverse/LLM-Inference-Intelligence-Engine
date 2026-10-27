# LLM Inference Intelligence Engine

The **LLM Inference Intelligence Engine (IIE)** is an open, modular, measurement-driven platform for LLM inference benchmarking, hardware capability discovery, evidence-based bottleneck diagnostics, performance prediction, and constrained configuration optimization.

---

## Project Documentation

* [Product Requirements & Specification](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/docs/requirements.md)
* [System Architecture & Contracts](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/docs/architecture.md)
* [Implementation Plan & Session Roadmap](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/docs/implementation-plan.md)
* [Session 08 Persistent Storage & REST API Design](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/docs/session-08-design.md)

---

## Core Capabilities Overview

1. **Hardware Discovery & Capacity Baselines**: Introspects host CPU, memory channels, and GPU topologies; evaluates theoretical memory bandwidth and compute ceilings.
2. **Empirical Benchmarking**: Instruments prefill (Time-to-First-Token) and decode (Inter-Token Latency, TPOT) phases under configurable concurrency and workload distributions.
3. **Evidence-Based Diagnostics**: Roofline model analysis to isolate memory-bandwidth saturation, compute limits, and KV cache allocation headroom.
4. **Performance Prediction**: Hybrid analytical and empirical models with held-out generalization validation (MAE, MAPE).
5. **Constrained Configuration Recommendation**: Pareto-optimal sizing across latency SLAs, throughput targets, and memory budgets.
6. **Data Provenance**: Every metric is strictly labeled as `MEASURED`, `PREDICTED`, or `HEURISTIC`.
7. **CPU-First Development**: Full platform support and test execution on standard CPU workstations via high-fidelity mock engine simulation.
8. **Persistent Relational Storage & REST API**: SQLite database with WAL and foreign-key integrity, coupled with a FastAPI backend offering background benchmark execution and live Server-Sent Events (SSE) telemetry.

---

## Package Architecture & Layout

The project follows a modular Python package structure under `llm_ii_engine/`:

```text
llm_ii_engine/
├── __init__.py           # Package version
├── core/                 # Validated domain models, enums, exceptions
│   ├── __init__.py
│   ├── enums.py          # ProvenanceType, ExecutionMode, ClockSource, EngineType, QuantizationType
│   ├── exceptions.py     # Domain exception hierarchy (IIEError, EngineExecutionError)
│   └── schemas.py        # Pydantic v2 models (ModelSpec, WorkloadSpec, TokenStreamEvent)
├── analysis/             # Analytical modeling, Roofline analysis, evidence-based diagnostics
│   ├── __init__.py
│   ├── models.py         # ModelResourceEstimate, RooflineAnalysisResult, DiagnosticReport
│   ├── resource_estimator.py # Parameter storage, MHA/GQA/MQA KV-cache, FLOPs, operational intensity
│   ├── roofline.py       # Ridge point derivation, attainable ceilings, token throughput bounds
│   └── diagnostics.py    # Conservative diagnostic engine with typed findings & confidence
├── benchmark/            # Execution runner, metrics collector, workload sampler
│   ├── __init__.py
│   ├── collector.py      # MetricCollector with hand-derived percentile interpolation
│   ├── models.py         # BenchmarkSuiteResult, RequestBenchmarkResult, MemoryWatermarkResult
│   ├── runner.py         # BenchmarkRunner with bounded asyncio concurrency, warmup, and event callbacks
│   ├── sampler.py        # TokenStreamSampler for watermark telemetry
│   └── workload.py       # Deterministic reproducible synthetic workload generator
├── discovery/            # Host hardware discovery, GPU catalog, simulated profiles
│   ├── __init__.py
│   ├── catalog.py        # Authoritative GPU hardware specification catalog (H100, A100, RTX 4090, T4)
│   ├── models.py         # HardwareCapabilityReport, DetectedGPU, GPUSpecification
│   ├── profiler.py       # Cross-platform HardwareProfiler with graceful telemetry fallback
│   └── simulated.py      # Deterministic simulated hardware profiles for CI and tests
├── prediction/           # Empirical prediction models, baseline predictors, evaluation
│   ├── __init__.py
│   ├── models.py         # PredictionRecord, PredictionDataset, PredictionOutput, EvaluationResult
│   ├── dataset.py        # DatasetBuilder, target eligibility rules, warmup/chunk exclusion
│   ├── baselines.py      # ConstantMedianPredictor, RidgeRegressionPredictor, AnalyticalBaselinePredictor
│   ├── evaluation.py     # DataSplitter (Random, GroupedBySuite, HoldoutByModel), PredictionEvaluator (MAE/RMSE/MAPE)
│   └── generalization.py # OODDetector, numerical support envelopes, unseen categorical checks
├── optimizer/            # Constrained Pareto optimization and configuration recommender
│   ├── __init__.py
│   ├── models.py         # OptimizationRequest, CandidateConfig, CandidateEvaluation, OptimizationResult
│   ├── feasibility.py    # Hard constraint feasibility checker (context length, quantization, VRAM, SLAs)
│   ├── pareto.py         # Non-dominated sorting and knee-point selection engine
│   ├── search.py         # Bounded search space generator
│   └── engine.py         # ConfigurationOptimizer orchestrating search, evaluation, and ranking
├── storage/              # Relational persistence layer (SQLAlchemy 2.x + SQLite)
│   ├── __init__.py
│   ├── models.py         # Relational declarative ORM tables (Hardware, Benchmarks, Observations, Diagnostics, etc.)
│   ├── database.py       # Connection lifecycle, WAL mode, foreign keys, schema versioning
│   └── repository.py     # Typed repository mapping ORM entities to canonical Pydantic domain models
├── api/                  # FastAPI REST and SSE application foundation
│   ├── __init__.py
│   ├── app.py            # FastAPI factory `create_app` with lifespan and error handlers
│   ├── jobs.py           # BenchmarkJobManager with in-memory execution and SSE event fanout
│   ├── dependencies.py   # Database and repository injection dependencies
│   ├── errors.py         # Uniform error models and HTTP exception handlers
│   └── routes/           # REST sub-routers (/health, /hardware, /benchmarks, /diagnostics, /predictions, /optimize)
└── adapters/             # Asynchronous inference engine adapters
    ├── __init__.py
    ├── base.py           # InferenceEngineAdapter protocol & BaseInferenceAdapter ABC
    ├── http_base.py      # BaseHttpEngineAdapter with connection & timeout management
    ├── mock.py           # Deterministic MockEngineAdapter with virtual clock timing
    ├── ollama.py         # OllamaAdapter for /api/generate NDJSON streaming
    └── llamacpp.py       # LlamaCppAdapter for /completion SSE streaming
```

---

## Domain Contracts & Event Semantics

### 1. Token Stream Lifecycle Contract
Inference engines stream discrete `TokenStreamEvent` instances following a strict lifecycle:

1. **`START` Event (`sequence_index=0`)**: Emitted when the engine accepts the request and initiates the prefill phase. Contains base timestamp and `is_terminal=False`.
2. **`TOKEN` Events (`sequence_index=1..N`)**: Emitted sequentially as tokens are decoded.
   * `token_index=0`: First output token, marking Time-to-First-Token (`is_first_token=True`).
   * `token_index>0`: Subsequent decode tokens (`is_first_token=False`).
   * Each token event records `delta_ns` (elapsed nanoseconds since previous event) and monotonic `timestamp_ns`.
3. **`COMPLETION` Event (`is_terminal=True`)**: Terminal event indicating successful conclusion with `termination_reason` (`FINISHED` or `MAX_TOKENS`).
4. **`ERROR` Event (`is_terminal=True`)**: Terminal event emitted if an unrecoverable failure halts the stream (`termination_reason=ERROR`, with non-empty `error_message`).

#### Edge Case Handling:
* **Empty Output (`output_tokens=0`)**: Emits `START` immediately followed by `COMPLETION` with `termination_reason=MAX_TOKENS`.
* **Single Token Output (`output_tokens=1`)**: Emits `START`, 1 `TOKEN` event (`is_first_token=True`), and `COMPLETION`.
* **Stream Cancellation**: Triggers `StreamCancelledError` ensuring clean teardown without orphan tasks.

### 2. Real vs. Simulated Execution & Data Provenance
Data integrity is enforced at the schema validation level:
* **Execution Mode**: `ExecutionMode.REAL` (physical hardware & model runtime) vs `ExecutionMode.SIMULATED` (mock adapter or synthetic generator).
* **Clock Source**: `ClockSource.MONOTONIC_REAL` (physical hardware timer) vs `ClockSource.SYNTHETIC_VIRTUAL` (deterministic virtual nanosecond advancement with zero sleep).
* **Provenance**: `ProvenanceType.MEASURED`, `PREDICTED`, or `HEURISTIC`. Synthetic simulation events can never claim real hardware measurements.

---

## Inference Engine Adapter Protocol

All engine runtimes implement the typed asynchronous `InferenceEngineAdapter` protocol defined in `llm_ii_engine/adapters/base.py`:

```python
class InferenceEngineAdapter(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def engine_type(self) -> EngineType: ...
    @property
    def execution_mode(self) -> ExecutionMode: ...
    def get_capabilities(self) -> EngineCapabilities: ...
    async def initialize(self, model_spec: ModelSpec, engine_config: EngineConfig) -> bool: ...
    async def health_check(self) -> bool: ...
    def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenStreamEvent]: ...
    async def get_memory_usage_mb(self) -> float | None: ...
    async def teardown(self) -> None: ...
```

---

## Hardware Discovery & Specification Catalog

The `llm_ii_engine.discovery` subsystem introspects host hardware, inspects telemetry utilities, and resolves theoretical ceilings against an authoritative catalog:

```text
llm_ii_engine/discovery/
├── __init__.py           # Unified exports for profiler, catalog, and simulated profiles
├── models.py             # HostSystemInfo, CPUInfo, MemoryInfo, DetectedGPU, GPUSpecification
├── profiler.py           # Cross-platform introspection (Windows, Linux, Darwin)
├── catalog.py            # Authoritative NVIDIA specifications (H100, A100, RTX 4090, 3090, T4)
└── simulated.py          # Deterministic simulated profiles for CPU/CI environments
```

### 1. Cross-Platform Discovery Mechanisms
* **Windows**: CPU brand string via `winreg`, physical cores via PowerShell CIM query, total and available physical RAM via `ctypes.windll.kernel32.GlobalMemoryStatusEx`.
* **Linux**: CPU brand and core counts via `/proc/cpuinfo`, RAM capacity via `/proc/meminfo`.
* **Darwin (macOS)**: CPU brand, physical cores, and RAM via `sysctl`.
* **GPU & Telemetry**: Safe path resolution for `nvidia-smi` and `rocm-smi`; query GPU name, VRAM, driver version, and compute capability with zero-cost fallback when utilities are absent.

### 2. Hardware Specification Catalog
Covers NVIDIA architectures with documented units, precision ratings, source citations, and caveats:
* **H100 SXM5 (80GB)**: 3,350 GB/s HBM3, 989 TFLOPs dense FP16 / 1,979 TFLOPs sparse FP16, 700W.
* **H100 PCIe (80GB)**: 2,000 GB/s HBM2e, 756 TFLOPs dense FP16 / 1,513 TFLOPs sparse FP16, 350W.
* **H100 NVL (94GB)**: 3,900 GB/s HBM3, 835 TFLOPs dense FP16 / 1,671 TFLOPs sparse FP16, 400W.
* **A100 SXM4 (80GB)**: 2,039 GB/s HBM2e, 312 TFLOPs dense FP16 / 624 TFLOPs sparse FP16, 400W.
* **A100 PCIe (80GB / 40GB)**: 1,935 GB/s / 1,555 GB/s HBM2, 312 TFLOPs dense FP16, 250W.
* **GeForce RTX 4090 (24GB)**: 1,008 GB/s GDDR6X, 165.2 TFLOPs dense FP16 / 330.3 TFLOPs sparse FP16, 450W.
* **GeForce RTX 3090 (24GB)**: 936.2 GB/s GDDR6X, 71.0 TFLOPs dense FP16 / 142.0 TFLOPs sparse FP16, 350W.
* **Tesla T4 (16GB)**: 320 GB/s GDDR6, 65.0 TFLOPs dense FP16, 70W.

All catalog entries are strictly labeled with `ProvenanceType.HEURISTIC`.

### 3. Roofline Boundary & Requirements for Future Analysis
The discovery subsystem exposes static theoretical ceilings via `report.get_theoretical_roofline_ceiling(precision)`.
* **Important Distinction**: Theoretical ceilings reflect manufacturer limits, NOT runtime measurements.
* **Future Roofline Analyzer Requirements**:
  1. Workload token counts: Prompt tokens ($L_{\text{prompt}}$) and generated decode tokens ($L_{\text{decode}}$).
  2. Latency timing measurements: Physical prefill TTFT and per-token decode ITL.
  3. Model parameters & precision: Weight bytes ($P \times b_{\text{model}}$) and layer dimensions.
  4. Dynamic KV cache memory footprint: $2 \times b \times l \times h \times d_{\text{head}} \times b_{\text{kv}}$.
  5. Empirical Operational Intensity ($AI = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$) compared against the machine ridge point ($I_{\text{ridge}} = \frac{P_{\text{peak}}}{BW_{\text{peak}}}$).

---

## Benchmark Execution & Metric Collection Subsystem

The `llm_ii_engine.benchmark` package orchestrates asynchronous benchmark runs, collects empirical latency metrics, profiles memory consumption, and compiles statistical summaries:

```text
llm_ii_engine/benchmark/
├── __init__.py           # Exports BenchmarkRunner, MetricCollector, WorkloadGenerator, etc.
├── models.py             # RequestBenchmarkResult, MemoryWatermarkResult, BenchmarkSuiteResult
├── collector.py          # Pure-Python linear-interpolation percentiles & MetricCollector
├── workload.py           # Reproducible seed-based workload scenario generator
├── sampler.py            # MemorySampler protocol, AdapterMemorySampler, MockMemorySampler
└── runner.py             # Asynchronous runner with bounded concurrency and request isolation
```

### 1. Mathematical Metric Definitions

* **Time-to-First-Token (TTFT)**:
  $$\text{TTFT} = t_{\text{token}_0} - t_{\text{submit}}$$
  Represents prefill latency. For prefill-only requests ($\text{output\_tokens} = 0$), $\text{TTFT} = \text{None}$.
* **Inter-Token Latency (ITL)**:
  $$\text{ITL}_i = t_{\text{token}_i} - t_{\text{token}_{i-1}} \quad \text{for } i \in [1, \text{output\_tokens}-1]$$
  Discrete time deltas between successive decoded tokens. Empty for requests with $\le 1$ token.
* **Time-Per-Output-Token (TPOT)**:
  $$\text{TPOT} = \frac{t_{\text{token}_{N-1}} - t_{\text{token}_0}}{N - 1} \quad (N = \text{output\_tokens} \ge 2)$$
  Strictly measures decode-phase cadence, completely isolated from prefill/TTFT overhead. When $N < 2$, $\text{TPOT} = \text{None}$.
* **Percentiles ($p_{50}, p_{90}, p_{95}, p_{99}$)**:
  Computed via pure-Python linear rank interpolation:
  $$\text{rank} = 1 + \frac{p}{100} \times (n - 1)$$
* **Throughput Metrics**:
  * **Per-Request Output Throughput**: $\frac{\text{output\_tokens}}{\text{request\_duration\_seconds}}$
  * **Aggregate Output Throughput**: $\frac{\sum_{r \in \text{measured}} \text{output\_tokens}_r}{\text{suite\_wall\_duration\_seconds}}$
  * **Aggregate Total Throughput**: $\frac{\sum_{r \in \text{measured}} (\text{prompt\_tokens}_r + \text{output\_tokens}_r)}{\text{suite\_wall\_duration\_seconds}}$

### 2. Edge Case and Failure Integrity Rules
* **No Zero Substitution**: Undefined metrics (e.g. TTFT for 0 tokens, TPOT for 1 token, GPU memory without telemetry) are stored as explicit `None`, never converted to deceptive zeros.
* **Warmup Exclusion**: Warmup requests run through the complete engine lifecycle and are validated in `warmup_results`, but are strictly excluded from all suite aggregate statistics and percentiles.
* **Request Isolation**: A timeout or unrecoverable error on a single request is captured in `RequestBenchmarkResult.error_message` with `is_valid=False`; it never aborts or corrupts the remaining benchmark suite.
* **Clock Monotonicity Check**: Events are verified for strictly non-decreasing timestamps. Mixed clock sources or timestamp reversals immediately invalidate the measurement with detailed diagnostic notes.

---

## Real Inference Engine Adapters (Ollama & llama.cpp)

Production-conscious HTTP streaming adapters interface directly with live inference daemons using asynchronous HTTP client infrastructure (`httpx`).

### 1. Architectural Principles & Clock Integrity
* **Physical Monotonic Timing**: All client-side observations use the local monotonic clock (`time.perf_counter_ns()`, `ClockSource.MONOTONIC_REAL`).
* **Chunk vs Token Discrimination**: Real HTTP streaming servers emit variable-length text chunks over the wire, which rarely correspond 1:1 to model tokens. The adapters emit discrete `CHUNK` events for text deltas and strictly avoid converting arbitrary text fragments into fabricated `TOKEN` events.
* **Metric Provenance & Attribution**:
  * **Client-Observed**: Time-to-First-Token/Chunk (TTFT), text-chunk inter-arrival latencies, total generation duration.
  * **Server-Reported**: Prompt evaluation count, generated token count (`eval_count` / `predicted_n`), server decode duration (`eval_duration` / `predicted_ms`), and server-calculated throughput.
  * **True Token-Level**: Inter-Token Latency (ITL) percentiles and Time-Per-Output-Token (TPOT) are computed *if and only if* the underlying engine exposes verified discrete token IDs. For text-chunk streams, ITL and TPOT are preserved as `None` rather than fabricated.
* **No Implicit Model Pulls**: Adapters strictly inspect server model inventories (`/api/tags`) during initialization and raise descriptive `ModelNotFoundError` exceptions if a model is missing, preventing unexpected multi-gigabyte background downloads.

### 2. Supported Adapters
1. **Ollama Adapter (`OllamaAdapter`)**:
   * Endpoint: `POST /api/generate` (`stream: true`).
   * Incremental NDJSON (newline-delimited JSON) streaming parser handling packet-boundary line splits.
   * Health and version validation via `GET /api/version`.
   * Server evaluation statistics extracted on `done: true` (`prompt_eval_count`, `eval_count`, `prompt_eval_duration`, `eval_duration`).
2. **llama.cpp Server Adapter (`LlamaCppAdapter`)**:
   * Endpoint: `POST /completion` (`stream: true`).
   * Incremental Server-Sent Events (SSE) streaming parser (`data: {...}`, `[DONE]`, comments).
   * Health check via `GET /health` (handles `"status": "ok"` vs `"status": "loading model"`).
   * Server timings extracted from terminal message (`timings.prompt_n`, `timings.predicted_n`, `timings.predicted_ms`).
   * Dynamic token-id detection: emits true `TOKEN` events when token IDs are exposed in the stream.

### 3. Optional Manual Integration Procedure with Live Servers
For users with locally installed engines:

#### Ollama:
```powershell
# 1. Start Ollama and verify model
ollama run llama3 "Hello"

# 2. Run benchmark script or integration test
python -c "
import asyncio
from llm_ii_engine.adapters.ollama import OllamaAdapter
from llm_ii_engine.core.schemas import ModelSpec, EngineConfig, GenerationRequest, WorkloadSpec

async def run():
    adapter = OllamaAdapter(base_url='http://localhost:11434')
    model = ModelSpec(model_id='llama3:latest')
    config = EngineConfig(engine_type='ollama')
    await adapter.initialize(model, config)
    req = GenerationRequest(request_id='manual-1', model_spec=model, engine_config=config, workload=WorkloadSpec(prompt_text='What is inference optimization?', output_tokens=32))
    async for event in adapter.generate_stream(req):
        if event.event_type.value == 'chunk':
            print(event.token_text, end='', flush=True)
        elif event.is_terminal:
            print('\nStats:', event.server_stats)
    await adapter.teardown()

asyncio.run(run())
"
```

#### llama.cpp server:
```powershell
# 1. Launch llama.cpp server
./llama-server -m models/llama-2-7b.Q4_K_M.gguf --port 8080

# 2. Test connection with LlamaCppAdapter
python -c "
import asyncio
from llm_ii_engine.adapters.llamacpp import LlamaCppAdapter
from llm_ii_engine.core.schemas import ModelSpec, EngineConfig, GenerationRequest, WorkloadSpec

async def run():
    adapter = LlamaCppAdapter(base_url='http://localhost:8080')
    model = ModelSpec(model_id='llama-2-7b')
    config = EngineConfig(engine_type='llamacpp')
    await adapter.initialize(model, config)
    req = GenerationRequest(request_id='manual-1', model_spec=model, engine_config=config, workload=WorkloadSpec(prompt_text='Hello llama', output_tokens=16))
    async for event in adapter.generate_stream(req):
        if event.event_type.value == 'chunk':
            print(event.token_text, end='', flush=True)
    await adapter.teardown()

asyncio.run(run())
"
```

---

## Analytical Resource Modeling, Rooflines & Diagnostics

The `llm_ii_engine.analysis` subsystem provides analytical resource estimation, theoretical Roofline modeling, and evidence-based bottleneck diagnostics.

### 1. Mathematical Resource Formulation

1. **Parameter Weight Storage**:
   $$\text{Bytes} = P \times 10^9 \times b_{param}$$
   * FP16 / BF16: $2.0 \text{ bytes/param}$ (exact).
   * INT8: $1.0 \text{ byte/param}$ (exact).
   * INT4: $0.5 \text{ bytes/param}$ (raw weights).
   * GGUF Q8_0: $1.0625 \text{ bytes/param}$ (32-element INT8 blocks with FP16 scale = $(32 + 2)/32$).
   * GGUF Q4_K_M: $0.5625 \text{ bytes/param}$ (mixed 4/6-bit blocks = $\sim 4.5 \text{ bits/param}$).
   * AWQ / GPTQ: $0.53125 \text{ bytes/param}$ (4-bit weights with group size 128 = $\sim 4.25 \text{ bits/param}$).

2. **Attention KV-Cache Footprint**:
   * Per-token byte formula across attention architectures:
     $$\text{KV bytes per token} = 2 \times l \times h_{kv} \times d_{head} \times b_{kv}$$
     Where:
     * $l$: number of transformer layers (`num_layers`).
     * $h_{kv}$: number of key/value heads (`num_kv_heads`).
     * $d_{head}$: head dimension (`head_dimension` or `hidden_dimension // num_attention_heads`).
     * $b_{kv}$: bytes per KV tensor element (default FP16 = 2 bytes).
     * Factor $2$: accounts for separate Key and Value tensor allocations.
   * Architecture variants supported:
     * **MHA (Multi-Head Attention)**: $h_{kv} = h_q$.
     * **GQA (Grouped-Query Attention)**: $1 < h_{kv} < h_q$ (e.g. LLaMA-3 8B: $h_{kv}=8$, $h_q=32 \rightarrow 4\times$ cache compression).
     * **MQA (Multi-Query Attention)**: $h_{kv} = 1$ ($h_q\times$ cache compression).
   * Total KV cache memory:
     $$\text{Total KV Bytes} = \text{KV bytes per token} \times b \times s$$
     Where $b$ is concurrency / batch size and $s$ is evaluated sequence length ($s = S_{prompt} + S_{output}$).

3. **Computational Work (FLOPs)**:
   * Prefill Phase: $2 P S_{prompt} + 4 l d_{hidden} (S_{prompt})^2$
   * Decode Phase: $2 P S_{decode} + \sum_{i=1}^{S_{decode}} [4 l d_{hidden} (S_{prompt} + i)]$

4. **Operational Intensity (Arithmetic Intensity)**:
   $$AI = \frac{\text{Total FLOPs}}{\text{Bytes Transferred from Memory}}$$
   * Prefill is matrix-matrix compute intensive ($AI \gg 50 \text{ FLOPs/Byte}$).
   * Decode at batch size 1 is memory-bandwidth bound ($AI \approx 1 \text{ to } 2 \text{ FLOPs/Byte}$).

### 2. Theoretical Roofline Modeling

* **Machine Balance Point (Ridge Point)**:
  $$I_{ridge} = \frac{P_{peak} \times 10^{12} \text{ FLOP/s}}{BW_{peak} \times 10^9 \text{ Byte/s}} = \frac{P_{peak} \times 1000}{BW_{peak}} \text{ FLOPs/Byte}$$
* **Attainable Compute Throughput**:
  $$P_{attainable} = \min\left(P_{peak}, \frac{AI \times BW_{peak}}{1000}\right) \text{ TFLOPs}$$
* **Theoretical Upper Bound on Token Generation Rate**:
  $$\text{Tokens/s Bound} = \frac{P_{attainable} \times 1000}{2 \times P} \text{ tokens/second}$$

### 3. Evidence-Based Diagnostic Engine

The `DiagnosticEngine` evaluates benchmark suite results against analytical models and hardware specifications using conservative, scientifically honest rules:
* **Memory Feasibility & Headroom (`MEMORY_CAPACITY`)**: Identifies models that mathematically exceed VRAM/RAM (`CRITICAL`) or risk OOM due to allocator margins (`WARNING`).
* **Context Window Violations (`CONFIGURATION_VIABILITY`)**: Flags requests exceeding the model's architectural context window (`CRITICAL`).
* **Elevated TTFT Latency Overhead (`LATENCY_OVERHEAD`)**: Compares measured TTFT against theoretical prefill compute duration ($> 3\times$ trigger). Explicitly notes that server queueing vs prompt tokenization cannot be isolated without internal profiler traces.
* **Streaming Inter-Arrival Jitter (`LATENCY_OVERHEAD`)**: Flags high $p99 / p50$ ratios in token streams ($> 3\times$ with $> 20\text{ ms}$ delta).
* **Decode Throughput Efficiency (`EFFICIENCY_BOTTLENECK`)**: Compares observed decode throughput against the theoretical roofline ceiling. Assigned `LOW` confidence with explicit documentation that hardware counters (DRAM bus traffic, SM active cycles) are required to conclusively prove memory-bandwidth saturation.
* **Telemetry & Chunk Streaming Gaps (`TELEMETRY_GAP`)**: Explicitly flags unmatched GPUs, absent memory watermarks, and chunk-mode streaming where true token-level TPOT is unavailable.

---

## Empirical Performance Prediction & Generalization Evaluation

The `llm_ii_engine.prediction` subsystem implements typed prediction datasets, non-parametric and regression baselines, data partitioning, and Out-of-Distribution (OOD) envelope detection.

### 1. Defensible Dataset & Target Eligibility Rules

Inference benchmark evidence is converted into typed `PredictionRecord` instances with explicit eligibility criteria per target metric:
* **Warmup Exclusion**: Any request executed during warmup (`is_warmup=True`) is **strictly excluded** from all training targets.
* **Failure Exclusion**: Failed or prematurely cancelled requests are excluded with explicit error reasoning.
* **Token vs. Chunk Isolation**:
  * **Time-to-First-Token (`TTFT_MS`)**: Requires $\ge 1$ output token and verified timing.
  * **Token-Level TPOT (`TPOT_MS`)**: Requires $\ge 2$ output tokens and verified token event boundaries (`TokenCountSource.TOKEN_EVENTS`). Text chunks are **never** treated as individual tokens; chunk streams without per-token boundaries are marked strictly ineligible.
  * **Chunk Inter-Arrival Latency (`CHUNK_LATENCY_MS`)**: Requires $\ge 2$ text chunks.
  * **Output Throughput (`OUTPUT_THROUGHPUT_TOK_S`)**: Requires $\ge 1$ decoded token.
  * **Peak Memory (`PEAK_VRAM_MB`)**: Requires verified telemetry provider availability.
* **Feature Preservation**: Missing optional architectural or hardware fields remain `None`. The system never fabricates or silently fills unobserved features.
* **Data Leakage Prevention**: Identifiers (`record_id`, `suite_id`, `request_id`, `model_id`) and post-outcome metrics (`duration_ms`, chunk counts, server stats) are strictly excluded from input feature vectors.

### 2. Predictor Baselines

* **Constant Median Baseline (`ConstantMedianPredictor`)**: Non-parametric baseline predicting the training-set median. Provides the reference error floor for all empirical evaluations.
* **Ridge Regression (`RidgeRegressionPredictor`)**: Standardized multiple linear regression with L2 regularization ($A w = X^T y$ where $A = X^T X + \alpha I$). Solved deterministically using Gaussian elimination with partial pivoting in pure Python (no heavyweight ML dependencies required). Missing input features are imputed with the training mean.
* **Analytical Roofline Baseline (`AnalyticalBaselinePredictor`)**: Calculates theoretical ceilings derived from Session 05 analytical resource models and GPU datasheets. Output provenance is strictly `ProvenanceType.HEURISTIC` / `PREDICTED` and validation status is `THEORETICAL_ANALYTICAL`. It is never presented as an empirical model.

### 3. Generalization & Out-of-Distribution (OOD) Analysis

The `OODDetector` tracks the empirical support envelope established during training:
* **Numerical Bounds**: Verifies that candidate inputs (e.g. prompt length, parameter count, concurrency) fall within $[min, max]$ training bounds.
* **Categorical Support**: Identifies unseen model IDs, quantizations, or hardware variants.
* **Missing Critical Inputs**: Alerts when required inference dimensions are missing.
* **Scientific Honesty**: OOD warnings indicate geometric support violations and are explicitly not represented as calibrated Bayesian error probabilities.

### 4. Data Partitioning & Evaluation Metrics

* **Random Split**: Reproducible pseudo-random split.
* **Grouped Split by Suite (`suite_id`)**: Keeps all requests from the same benchmark suite within either train or test, preventing intra-suite correlation leakage.
* **Holdout Split by Model (`model_id`)**: Reserves entire model architectures for the test set to evaluate out-of-distribution transfer.
* **Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE, handled safely with $\epsilon = 10^{-4}$ for zero/near-zero targets). Evaluates baseline comparisons on the identical test partition and flags sample sizes $N < 10$ as exploratory.

---

## Development Setup & Testing

### 1. Environment Setup (using uv)
```powershell
# Create virtual environment
uv venv

# Install package in editable mode with development dependencies
uv pip install -e ".[dev]"
```

### 2. Running Unit Tests
```powershell
# Run the complete unit test suite
.venv\Scripts\pytest.exe

# Run with verbose output and coverage
.venv\Scripts\pytest.exe -v
```

### 3. Code Quality & Formatting
```powershell
# Check code style with ruff
.venv\Scripts\ruff.exe check .

# Format code
.venv\Scripts\ruff.exe format .
```

---

---

## Persistent Relational Storage & REST API

The `llm_ii_engine.storage` and `llm_ii_engine.api` packages provide SQLite persistence and a FastAPI foundation for orchestrating benchmarks, running diagnostics, and querying optimization recommendations over HTTP.

### 1. Database Configuration & Reliability Guarantees
* **Engine**: SQLite managed by SQLAlchemy 2.x ORM with strict declarative typings.
* **Foreign Key Enforcement**: `PRAGMA foreign_keys = ON;` is enforced on all connections.
* **Concurrency**: `PRAGMA journal_mode = WAL;` (Write-Ahead Logging) enables non-blocking read queries while asynchronous benchmarks persist results.
* **Lock Management**: `PRAGMA busy_timeout = 5000;` prevents database lock failures.
* **Schema Evolution**: Tracked via `schema_version` table with startup compatibility checks.

### 2. Starting the API Service
Run the FastAPI development server using `uvicorn`:
```powershell
uv run uvicorn llm_ii_engine.api.app:create_app --factory --host 127.0.0.1 --port 8000
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Key REST API Endpoints & cURL Examples

#### Health Probe
```bash
curl -X GET http://127.0.0.1:8000/health
```

#### Hardware Discovery
```bash
curl -X GET http://127.0.0.1:8000/api/v1/hardware
```

#### Launch Asynchronous Benchmark Run
```bash
curl -X POST http://127.0.0.1:8000/api/v1/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{
    "model": {"model_id": "llama-3-8b", "parameter_count_billion": 8.0},
    "engine_type": "mock",
    "workloads": [{"prompt_tokens": 128, "output_tokens": 64}],
    "concurrency": 2
  }'
```
Returns `{"job_id": "job-...", "status": "PENDING"}` with HTTP 202.

#### Subscribe to Live Token Telemetry (Server-Sent Events)
```bash
curl -N -H "Accept: text/event-stream" http://127.0.0.1:8000/api/v1/benchmarks/jobs/{job_id}/stream
```

#### Execute Bottleneck & Roofline Diagnostics
```bash
curl -X POST http://127.0.0.1:8000/api/v1/diagnostics/run \
  -H "Content-Type: application/json" \
  -d '{"suite_id": "suite-..."}'
```

#### Query Constrained Pareto Recommendations
```bash
curl -X POST http://127.0.0.1:8000/api/v1/optimize/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "workload": {"prompt_tokens": 256, "output_tokens": 128},
    "search_space": {
      "models": [{"model_id": "llama-3-8b", "parameter_count_billion": 8.0}],
      "quantizations": ["fp16", "int4"],
      "concurrencies": [1, 2],
      "engines": ["mock"]
    },
    "objectives": ["minimize_ttft", "maximize_throughput"],
    "hardware_catalog_id": "nvidia-rtx-4090"
  }'
```

### 4. Interactive Web Dashboard (React + TypeScript)

A single-page dashboard application built with React 18, TypeScript, Tailwind CSS, and Vite is provided in the `frontend/` directory.

#### Starting the Frontend Development Server
Ensure the backend API server is running on `http://127.0.0.1:8000`, then launch Vite dev server:
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` to access the interactive dashboard. The Vite dev server proxies `/api` and `/health` requests directly to `http://127.0.0.1:8000`.

#### Running Frontend Typecheck & Tests
```powershell
cd frontend
npm run typecheck    # Strict TypeScript checks without emit
npm test             # Vitest unit & component test suite (16 tests)
npm run build        # Production bundle build (Vite)
```

#### Dashboard Modules
* **Overview**: Real-time service health, recent benchmark runs, active diagnostic summaries, and navigation quick-links.
* **Hardware**: CPU details, system RAM, detected GPU topology, GPU catalog specifications, and capability baselines.
* **Benchmarks**: Live parameter configuration, asynchronous background job submission, real-time SSE token stream monitoring, and historical suite inspection.
* **Diagnostics**: Root-cause bottleneck identification, Roofline operational intensity visualization, and memory breakdown ceilings.
* **Predictions**: Held-out empirical validation metrics (MAE, RMSE, MAPE), sample sufficiency indicators, and synthetic dataset flags.
* **Optimization**: Multi-objective search space configuration, feasibility checking, knee-point selection, and non-dominated Pareto frontier inspection.

---

## Project Status

* **Session 00 (Complete)**: Project Specification, Architecture, and Repository Audit.
* **Session 01 (Complete)**: Core Infrastructure, Domain Contracts, and Deterministic Mock Adapter (48 passing unit tests).
* **Session 02 (Complete)**: Hardware Capability Discovery, Specification Catalog, and Simulated Profiles (73 passing unit tests).
* **Session 03 (Complete)**: Benchmark Execution, Concurrency Runner, Memory Watermarking, and Metric Collection (93 passing unit tests).
* **Session 04 (Complete)**: Real Inference Engine Adapters for Ollama & llama.cpp HTTP/REST Streaming (122 passing unit tests).
* **Session 05 (Complete)**: Analytical Performance Modeling, Theoretical Roofline Analysis, and Evidence-Based Diagnostics (156 passing unit tests).
* **Session 06 (Complete)**: Empirical Performance Prediction, Baseline Models, and Generalization Evaluation (189 passing unit tests).
* **Session 06.1 (Complete)**: Prediction Pipeline Correctness Audit & Feature Availability Isolation.
* **Session 07 (Complete)**: Constrained Configuration Optimization and Multi-Objective Pareto Search (232 passing unit tests).
* **Session 07.1 (Complete)**: Optimizer Correctness & Evidence Attribution Audit.
* **Session 08 (Complete)**: Persistent Storage and REST API Foundation (252 passing unit tests).
* **Session 08.1 (Complete)**: Persistence, API, and SSE Reliability Audit (277 passing unit tests).
* **Session 09 (Complete)**: Interactive Dashboard Foundation in React 18, TypeScript, Tailwind CSS, and Vite with typed REST client, live SSE token streaming, and 6 core functional modules (16 passing frontend tests, 277 passing backend tests).




