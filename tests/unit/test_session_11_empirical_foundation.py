"""Unit tests for reproducible real-model benchmarking and empirical dataset foundation (Session 11).

Covers:
1. Manifest creation, validation, and JSON serialization.
2. Standard deterministic pilot workload suite definitions.
3. Experiment execution with warmup separation and monotonic timing.
4. Failure policy enforcement and failed request isolation.
5. Dataset export to JSONL and CSV with schema metadata.
6. Execution mode filtering (REAL vs SIMULATED).
7. Provenance preservation and missing-data indicators.
8. Separate preservation of failed runs for reliability analysis.
9. Repeatability and measurement quality analysis (CV, warmup penalty, client-server divergence).
10. Downstream compatibility with DatasetBuilder and grouped train/test splitting.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from llm_ii_engine.adapters.base import BaseInferenceAdapter
from llm_ii_engine.benchmark.dataset_export import (
    EmpiricalDatasetExporter,
    EmpiricalObservationRecord,
)
from llm_ii_engine.benchmark.experiment import (
    PILOT_LONG_WORKLOAD,
    PILOT_MEDIUM_WORKLOAD,
    PILOT_SHORT_WORKLOAD,
    STANDARD_PILOT_SUITE,
    BenchmarkExperimentManifest,
    ExperimentRunner,
)
from llm_ii_engine.benchmark.repeatability import RepeatabilityAnalyzer
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    FailurePolicy,
    ProvenanceType,
    QuantizationType,
    StreamTerminationReason,
    TokenCountSource,
    TokenEventType,
)
from llm_ii_engine.core.schemas import (
    EngineCapabilities,
    EngineConfig,
    GenerationRequest,
    ModelSpec,
    ServerReportedStats,
    TokenStreamEvent,
)
from llm_ii_engine.prediction.dataset import DatasetBuilder
from llm_ii_engine.prediction.evaluation import DataSplitter


class MockBenchmarkingAdapter(BaseInferenceAdapter):
    """Deterministic adapter for verifying experiment orchestration, timing, and error policies."""

    def __init__(
        self,
        engine_type: EngineType = EngineType.OLLAMA,
        execution_mode: ExecutionMode = ExecutionMode.REAL,
        fail_request_ids: set[str] | None = None,
        simulate_server_stats: bool = True,
    ) -> None:
        super().__init__(name="mock-bench", engine_type=engine_type, execution_mode=execution_mode)
        self.fail_request_ids = fail_request_ids or set()
        self.simulate_server_stats = simulate_server_stats
        self.dispatched_requests: list[str] = []

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(supports_streaming=True)

    async def initialize(self, model_spec: ModelSpec, engine_config: EngineConfig) -> bool:
        return True

    async def health_check(self) -> bool:
        return True

    async def get_memory_usage_mb(self) -> float | None:
        return 512.0

    async def teardown(self) -> None:
        pass

    async def generate_stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[TokenStreamEvent]:
        self.dispatched_requests.append(request.request_id)

        if any(f_id in request.request_id for f_id in self.fail_request_ids):
            raise RuntimeError(f"Simulated network drop on request {request.request_id}")

        t0 = 1_000_000_000
        # Yield START event
        yield TokenStreamEvent(
            request_id=request.request_id,
            sequence_index=0,
            event_type=TokenEventType.START,
            timestamp_ns=t0,
            delta_ns=0,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=self.execution_mode,
        )

        # Yield 2 text chunks (simulating Ollama / llama.cpp text chunk stream)
        yield TokenStreamEvent(
            request_id=request.request_id,
            sequence_index=1,
            event_type=TokenEventType.CHUNK,
            token_text="First text chunk",
            timestamp_ns=t0 + 40_000_000,
            delta_ns=40_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=self.execution_mode,
        )

        yield TokenStreamEvent(
            request_id=request.request_id,
            sequence_index=2,
            event_type=TokenEventType.CHUNK,
            token_text=" second text chunk.",
            timestamp_ns=t0 + 80_000_000,
            delta_ns=40_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=self.execution_mode,
        )

        server_stats = None
        if self.simulate_server_stats:
            server_stats = ServerReportedStats(
                prompt_tokens=request.workload.prompt_tokens or 32,
                output_tokens=request.workload.output_tokens or 48,
                prompt_duration_ms=30.0,
                eval_duration_ms=45.0,
                total_duration_ms=78.0,
            )

        yield TokenStreamEvent(
            request_id=request.request_id,
            sequence_index=3,
            event_type=TokenEventType.COMPLETION,
            timestamp_ns=t0 + 85_000_000,
            delta_ns=5_000_000,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=self.execution_mode,
            is_terminal=True,
            termination_reason=StreamTerminationReason.FINISHED,
            server_stats=server_stats,
        )


# ============================================================================
# 1. Manifest Validation and Serialization
# ============================================================================


def test_manifest_creation_and_json_round_trip(tmp_path: Path) -> None:
    """Experiment manifest serializes to formatted JSON and deserializes identically."""
    manifest = BenchmarkExperimentManifest.create_standard_pilot(
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        quantization=QuantizationType.GGUF_Q4_K_M,
        device_name="CPU",
        cpu_model="Intel Core i7-14700",
        host_ram_gb=32.0,
        warmup_repetitions=1,
        measured_repetitions=2,
        execution_mode=ExecutionMode.REAL,
    )

    assert manifest.model_id == "qwen2.5:0.5b"
    assert manifest.runtime_name == "ollama"
    assert manifest.device_name == "CPU"
    assert len(manifest.workload_suite) == 3
    assert manifest.warmup_repetitions == 1
    assert manifest.measured_repetitions == 2

    # Save to disk and reload
    save_file = tmp_path / "test_manifest.json"
    manifest.save(save_file)
    assert save_file.exists()

    loaded = BenchmarkExperimentManifest.load(save_file)
    assert loaded.experiment_id == manifest.experiment_id
    assert loaded.model_id == manifest.model_id
    assert loaded.quantization == QuantizationType.GGUF_Q4_K_M
    assert loaded.execution_mode == ExecutionMode.REAL


# ============================================================================
# 2. Standard Pilot Workload Suite
# ============================================================================


def test_standard_pilot_workloads_properties() -> None:
    """Pilot workloads have deterministic prompts and bounded token requirements."""
    assert len(STANDARD_PILOT_SUITE) == 3

    short_w = PILOT_SHORT_WORKLOAD
    med_w = PILOT_MEDIUM_WORKLOAD
    long_w = PILOT_LONG_WORKLOAD

    assert short_w.prompt_tokens == 32
    assert short_w.output_tokens == 48
    assert "locality" in short_w.prompt_text

    assert med_w.prompt_tokens == 128
    assert med_w.output_tokens == 96
    assert "prefill and decode" in med_w.prompt_text

    assert long_w.prompt_tokens == 384
    assert long_w.output_tokens == 128
    assert "mixture-of-experts" in long_w.prompt_text

    # Verify seeds are unique and non-zero
    seeds = [w.seed for w in STANDARD_PILOT_SUITE]
    assert len(set(seeds)) == 3
    assert all(s is not None and s > 0 for s in seeds)


# ============================================================================
# 3. Experiment Execution with Warmup Separation and Timing
# ============================================================================


@pytest.mark.asyncio
async def test_experiment_runner_warmup_separation() -> None:
    """Warmup requests are executed first and separated from measured aggregates."""
    adapter = MockBenchmarkingAdapter(engine_type=EngineType.OLLAMA)
    model_spec = ModelSpec(
        model_id="qwen2.5:0.5b",
        parameter_count_billion=0.5,
        quantization=QuantizationType.GGUF_Q4_K_M,
    )
    manifest = BenchmarkExperimentManifest.create_standard_pilot(
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        warmup_repetitions=1,
        measured_repetitions=2,
    )

    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)
    result = await runner.run_experiment(manifest)

    # 1 warmup repetition * 3 workloads = 3 warmup requests
    # 2 measured repetitions * 3 workloads = 6 measured requests
    assert len(result.warmup_requests) == 3
    assert len(result.measured_requests) == 6
    assert len(result.all_requests) == 9
    assert result.status == "COMPLETED"

    # Verify warmup requests are tagged
    for w_req in result.warmup_requests:
        assert w_req.is_warmup is True
    for m_req in result.measured_requests:
        assert m_req.is_warmup is False

    # Suite aggregates must reflect measured requests only
    assert result.suite_result.warmup_requests == 3
    assert result.suite_result.total_requests == 6
    assert result.suite_result.successful_requests == 6


# ============================================================================
# 4. Failure Policy Enforcement
# ============================================================================


@pytest.mark.asyncio
async def test_experiment_runner_failure_policy_continue_on_error() -> None:
    """Under CONTINUE_ON_ERROR, failed requests are captured and remaining requests continue."""
    adapter = MockBenchmarkingAdapter(
        engine_type=EngineType.OLLAMA,
        fail_request_ids={"req-0001"},  # Force 1 failure
    )
    model_spec = ModelSpec(model_id="qwen2.5:0.5b", parameter_count_billion=0.5)
    manifest = BenchmarkExperimentManifest(
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        workload_suite=[PILOT_SHORT_WORKLOAD, PILOT_MEDIUM_WORKLOAD],
        warmup_repetitions=0,
        measured_repetitions=2,
        failure_policy=FailurePolicy.CONTINUE_ON_ERROR,
    )

    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)
    result = await runner.run_experiment(manifest)

    assert result.status == "PARTIAL"
    assert len(result.all_requests) == 4
    assert len(result.failed_requests) == 1
    assert len(result.successful_measured_requests) == 3

    failed = result.failed_requests[0]
    assert failed.success is False
    assert "Simulated network drop" in (failed.error_message or "")


@pytest.mark.asyncio
async def test_experiment_runner_failure_policy_abort_immediately() -> None:
    """Under ABORT_IMMEDIATELY, the experiment halts on first error."""
    adapter = MockBenchmarkingAdapter(
        engine_type=EngineType.OLLAMA,
        fail_request_ids={"req-0000"},  # Fails immediately on first measured request
    )
    model_spec = ModelSpec(model_id="qwen2.5:0.5b", parameter_count_billion=0.5)
    manifest = BenchmarkExperimentManifest(
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        workload_suite=[PILOT_SHORT_WORKLOAD, PILOT_MEDIUM_WORKLOAD],
        warmup_repetitions=0,
        measured_repetitions=2,
        failure_policy=FailurePolicy.ABORT_IMMEDIATELY,
    )

    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)
    result = await runner.run_experiment(manifest)

    assert result.status == "FAILED"
    # Should have stopped after request 0, so only 1 request dispatched
    assert len(result.all_requests) == 1
    assert len(result.failed_requests) == 1
    assert len(result.successful_measured_requests) == 0


# ============================================================================
# 5. Dataset Export (JSONL and CSV)
# ============================================================================


@pytest.mark.asyncio
async def test_dataset_export_jsonl_and_csv(tmp_path: Path) -> None:
    """Empirical observations export accurately to both JSONL and CSV."""
    adapter = MockBenchmarkingAdapter(engine_type=EngineType.OLLAMA)
    model_spec = ModelSpec(model_id="qwen2.5:0.5b", parameter_count_billion=0.5)
    manifest = BenchmarkExperimentManifest.create_standard_pilot(
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        warmup_repetitions=1,
        measured_repetitions=2,
    )

    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)
    result = await runner.run_experiment(manifest)

    records = EmpiricalDatasetExporter.from_experiment_result(result)
    assert len(records) == 9  # 3 warmup + 6 measured

    jsonl_path = tmp_path / "dataset.jsonl"
    csv_path = tmp_path / "dataset.csv"

    # Export measured only (exclude warmup by default)
    count_jsonl = EmpiricalDatasetExporter.export_to_jsonl(records, jsonl_path, exclude_warmup=True)
    count_csv = EmpiricalDatasetExporter.export_to_csv(records, csv_path, exclude_warmup=True)

    assert count_jsonl == 6
    assert count_csv == 6
    assert jsonl_path.exists()
    assert csv_path.exists()

    # Verify JSONL lines can be parsed as EmpiricalObservationRecord
    with open(jsonl_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 6
    parsed_rec = EmpiricalObservationRecord.model_validate_json(lines[0])
    assert parsed_rec.model_id == "qwen2.5:0.5b"
    assert parsed_rec.runtime_name == "ollama"
    assert parsed_rec.ttfc_ms is not None  # Text chunk TTFC
    assert parsed_rec.ttft_ms is None  # Discrete TTFT unmeasured on text chunks!

    # Verify CSV header contains required columns
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert "record_id" in header
    assert "duration_ms" in header
    assert "ttfc_ms" in header
    assert "token_count_source" in header


# ============================================================================
# 6. Real-Measurement Filtering and Execution Mode
# ============================================================================


def test_real_only_filtering_excludes_simulated_runs(tmp_path: Path) -> None:
    """filter_real_only strictly excludes SIMULATED runs from primary empirical export."""
    real_rec = EmpiricalObservationRecord(
        record_id="r1",
        experiment_id="exp-1",
        suite_id="s1",
        request_id="req1",
        repetition_index=1,
        is_warmup=False,
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        adapter_version="1.0.0",
        device_name="CPU",
        prompt_token_count_source=TokenCountSource.REQUESTED.value,
        actual_output_tokens=48,
        token_count_source=TokenCountSource.SERVER_REPORTED.value,
        duration_ms=400.0,
        chunk_count=2,
        execution_mode=ExecutionMode.REAL.value,
        clock_source=ClockSource.MONOTONIC_REAL.value,
        provenance=ProvenanceType.MEASURED.value,
        success=True,
    )

    sim_rec = EmpiricalObservationRecord(
        record_id="r2",
        experiment_id="exp-1",
        suite_id="s1",
        request_id="req2",
        repetition_index=1,
        is_warmup=False,
        model_id="qwen2.5:0.5b",
        runtime_name="mock",
        adapter_version="1.0.0",
        device_name="MockCPU",
        prompt_token_count_source=TokenCountSource.REQUESTED.value,
        actual_output_tokens=48,
        token_count_source=TokenCountSource.TOKEN_EVENTS.value,
        duration_ms=400.0,
        chunk_count=0,
        execution_mode=ExecutionMode.SIMULATED.value,  # Simulated!
        clock_source=ClockSource.SYNTHETIC_VIRTUAL.value,
        provenance=ProvenanceType.HEURISTIC.value,
        success=True,
    )

    out_path = tmp_path / "filtered.jsonl"

    # Default filter_real_only=True
    written = EmpiricalDatasetExporter.export_to_jsonl(
        [real_rec, sim_rec], out_path, filter_real_only=True
    )
    assert written == 1

    # Permissive filter_real_only=False
    written_all = EmpiricalDatasetExporter.export_to_jsonl(
        [real_rec, sim_rec], out_path, filter_real_only=False
    )
    assert written_all == 2


# ============================================================================
# 7. Provenance Preservation and Missing-Data Indicators
# ============================================================================


def test_missing_data_reasons_audit_disclosure() -> None:
    """Missing metrics are accompanied by explicit justifications in missing_data_reasons."""
    adapter = MockBenchmarkingAdapter(
        engine_type=EngineType.OLLAMA,
        simulate_server_stats=False,  # No server stats
    )
    model_spec = ModelSpec(model_id="qwen2.5:0.5b", parameter_count_billion=0.5)
    manifest = BenchmarkExperimentManifest(
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        workload_suite=[PILOT_SHORT_WORKLOAD],
        warmup_repetitions=0,
        measured_repetitions=1,
    )

    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)
    # Run synchronous test helper via asyncio
    import asyncio

    result = asyncio.run(runner.run_experiment(manifest))
    records = EmpiricalDatasetExporter.from_experiment_result(result)
    assert len(records) == 1

    rec = records[0]
    assert rec.output_tokens_per_second is None
    assert any("output_tokens_per_second is null" in reason for reason in rec.missing_data_reasons)
    assert any("ttft_ms is null" in reason for reason in rec.missing_data_reasons)


# ============================================================================
# 8. Separate Preservation of Failed Runs
# ============================================================================


def test_separate_preservation_of_failed_runs(tmp_path: Path) -> None:
    """Failed requests are excluded from main dataset and written to a separate audit file."""
    good_rec = EmpiricalObservationRecord(
        record_id="good-1",
        experiment_id="exp-1",
        suite_id="s1",
        request_id="req-good",
        repetition_index=1,
        is_warmup=False,
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        adapter_version="1.0.0",
        device_name="CPU",
        prompt_token_count_source=TokenCountSource.REQUESTED.value,
        actual_output_tokens=48,
        token_count_source=TokenCountSource.SERVER_REPORTED.value,
        duration_ms=400.0,
        chunk_count=2,
        execution_mode=ExecutionMode.REAL.value,
        clock_source=ClockSource.MONOTONIC_REAL.value,
        provenance=ProvenanceType.MEASURED.value,
        success=True,
    )

    bad_rec = EmpiricalObservationRecord(
        record_id="bad-1",
        experiment_id="exp-1",
        suite_id="s1",
        request_id="req-bad",
        repetition_index=1,
        is_warmup=False,
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        adapter_version="1.0.0",
        device_name="CPU",
        prompt_token_count_source=TokenCountSource.REQUESTED.value,
        actual_output_tokens=0,
        token_count_source=TokenCountSource.UNKNOWN.value,
        duration_ms=120.0,
        chunk_count=0,
        execution_mode=ExecutionMode.REAL.value,
        clock_source=ClockSource.MONOTONIC_REAL.value,
        provenance=ProvenanceType.MEASURED.value,
        success=False,
        error_message="Connection reset by peer",
    )

    clean_file = tmp_path / "clean_dataset.jsonl"
    fail_file = tmp_path / "failures.jsonl"

    clean_count = EmpiricalDatasetExporter.export_to_jsonl(
        [good_rec, bad_rec], clean_file, include_failed=False
    )
    fail_count = EmpiricalDatasetExporter.export_failed_runs_to_jsonl([good_rec, bad_rec], fail_file)

    assert clean_count == 1
    assert fail_count == 1

    # Verify fail file contains error message
    with open(fail_file, encoding="utf-8") as f:
        content = f.read()
    assert "Connection reset by peer" in content


# ============================================================================
# 9. Repeatability and Variability Analysis
# ============================================================================


@pytest.mark.asyncio
async def test_repeatability_analyzer_variability_and_warmup() -> None:
    """RepeatabilityAnalyzer computes CV%, warmup cold-start penalties, and divergence."""
    adapter = MockBenchmarkingAdapter(engine_type=EngineType.OLLAMA)
    model_spec = ModelSpec(model_id="qwen2.5:0.5b", parameter_count_billion=0.5)
    manifest = BenchmarkExperimentManifest.create_standard_pilot(
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        warmup_repetitions=1,
        measured_repetitions=3,
    )

    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)
    result = await runner.run_experiment(manifest)

    report = RepeatabilityAnalyzer.analyze_experiment(result)

    assert report.model_id == "qwen2.5:0.5b"
    assert report.total_measured_requests == 9  # 3 workloads * 3 measured reps
    assert report.total_warmup_requests == 3
    assert len(report.workload_stats) == 3

    # Check workload stats
    ws0 = report.workload_stats[0]
    assert ws0.measured_repetitions_count == 3
    assert ws0.duration_stats is not None
    assert ws0.duration_stats.cv_percent >= 0.0

    # Under mock timing (constant 85ms duration), CV is 0.0%
    assert ws0.duration_stats.cv_percent == 0.0

    # Warmup penalty should be 0.0% with deterministic mock
    assert ws0.warmup_overhead_percent == 0.0

    # Divergence: duration 85ms - server_stats (30ms + 45ms = 75ms) = 10ms
    assert ws0.client_server_divergence_ms == 10.0

    # Flag for pilot statistical validation should be False (since sample size < 30)
    assert report.is_statistically_validated is False
    assert any("pilot dataset" in d.lower() for d in report.disclaimers)


# ============================================================================
# 10. Downstream Compatibility with DatasetBuilder and Splitter
# ============================================================================


@pytest.mark.asyncio
async def test_downstream_dataset_builder_and_grouped_split() -> None:
    """Exported observations convert to PredictionDataset and group split prevents repetition leakage."""
    adapter = MockBenchmarkingAdapter(engine_type=EngineType.OLLAMA)
    model_spec = ModelSpec(
        model_id="qwen2.5:0.5b",
        parameter_count_billion=0.5,
        quantization=QuantizationType.GGUF_Q4_K_M,
    )
    manifest = BenchmarkExperimentManifest.create_standard_pilot(
        model_id="qwen2.5:0.5b",
        runtime_name="ollama",
        warmup_repetitions=1,
        measured_repetitions=2,
    )

    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)
    result = await runner.run_experiment(manifest)
    records = EmpiricalDatasetExporter.from_experiment_result(result)

    # Ingest into PredictionDataset
    pred_dataset = DatasetBuilder.from_empirical_observations(
        observations=records,
        model_spec=model_spec,
    )

    assert len(pred_dataset.records) == 9

    # Verify target eligibility:
    # 1. Warmup records (first 3) must be ineligible for all targets
    warmup_preds = [r for r in pred_dataset.records if r.is_warmup]
    assert len(warmup_preds) == 3
    for wp in warmup_preds:
        assert wp.is_eligible_for_ttft is False
        assert wp.is_eligible_for_tpot is False
        assert wp.is_eligible_for_throughput is False

    # 2. Measured records (last 6):
    measured_preds = [r for r in pred_dataset.records if not r.is_warmup]
    assert len(measured_preds) == 6
    for mp in measured_preds:
        # Text chunk stream -> TTFT is None, therefore ineligible for TTFT
        assert mp.is_eligible_for_ttft is False
        # Text chunk stream with SERVER_REPORTED tokens -> ineligible for TPOT
        assert mp.is_eligible_for_tpot is False
        # Text chunk stream with 2 chunks -> eligible for chunk latency!
        assert mp.is_eligible_for_chunk_latency is True
        # Output throughput is present and positive -> eligible for throughput!
        assert mp.is_eligible_for_throughput is True

    # 3. Test grouped train/test split:
    # Multiple experiments or suite groups stay together
    # Let's create records from 2 distinct suites/experiments
    records_exp2 = [
        rec.model_copy(update={"experiment_id": "exp-2", "suite_id": "suite-2"})
        for rec in pred_dataset.records
    ]
    all_combined = list(pred_dataset.records) + records_exp2

    train_set, test_set = DataSplitter.grouped_split(
        records=all_combined,
        group_key="experiment_id",
        test_ratio=0.5,
        seed=42,
    )

    train_exp_ids = {r.experiment_id for r in train_set}
    test_exp_ids = {r.experiment_id for r in test_set}

    # Grouped split must completely isolate groups between train and test (no overlap / leakage!)
    assert train_exp_ids.isdisjoint(test_exp_ids)
