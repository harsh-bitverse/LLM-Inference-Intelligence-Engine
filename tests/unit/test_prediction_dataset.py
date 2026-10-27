"""Unit tests for prediction dataset construction and target eligibility rules.

Tests strict separation between token and chunk metrics, warmup exclusion,
failure exclusion, and feature preservation without synthetic fabrication.
"""

from llm_ii_engine.benchmark.models import (
    BenchmarkSuiteResult,
    MemoryWatermarkResult,
    RequestBenchmarkResult,
)
from llm_ii_engine.core.enums import (
    ClockSource,
    EngineType,
    ExecutionMode,
    ProvenanceType,
    QuantizationType,
    TokenCountSource,
)
from llm_ii_engine.core.schemas import ModelSpec
from llm_ii_engine.discovery.simulated import get_simulated_profile
from llm_ii_engine.prediction.dataset import DatasetBuilder
from llm_ii_engine.prediction.models import PredictionDataset, PredictionTarget


def _create_mock_suite(
    model_id: str = "llama-7b",
    requests: list[RequestBenchmarkResult] | None = None,
    warmup_requests: list[RequestBenchmarkResult] | None = None,
    telemetry_available: bool = True,
) -> BenchmarkSuiteResult:
    """Helper to build a valid BenchmarkSuiteResult."""
    reqs = requests if requests is not None else []
    warmups = warmup_requests if warmup_requests is not None else []
    return BenchmarkSuiteResult(
        suite_id="suite-test-01",
        status="COMPLETED",
        model_id=model_id,
        engine_type=EngineType.MOCK,
        concurrency=1,
        total_requests=len(reqs),
        warmup_requests=len(warmups),
        successful_requests=len(reqs),
        failed_requests=0,
        wall_time_seconds=1.0,
        aggregate_tokens_per_second=50.0,
        aggregate_output_tokens_per_second=50.0,
        request_throughput_per_second=1.0,
        ttft_p50_ms=20.0,
        itl_p50_ms=15.0,
        memory_watermark=MemoryWatermarkResult(
            peak_process_rss_mb=500.0,
            peak_gpu_vram_mb=4000.0 if telemetry_available else None,
            sample_count=10,
            sample_interval_ms=100.0,
            telemetry_source="nvidia-smi" if telemetry_available else "unavailable",
            is_telemetry_available=telemetry_available,
        ),
        execution_mode=ExecutionMode.SIMULATED,
        clock_source=ClockSource.MONOTONIC_REAL,
        provenance=ProvenanceType.MEASURED,
        request_results=reqs,
        warmup_results=warmups,
    )


class TestTargetEligibilityRules:
    """Test suite for explicit target eligibility filtering."""

    def test_warmup_observation_strictly_ineligible_for_all_targets(self) -> None:
        """Warmup records must never be eligible as training targets."""
        req = RequestBenchmarkResult(
            request_id="req-warmup-1",
            success=True,
            prompt_tokens=128,
            output_tokens=64,
            ttft_ms=25.0,
            tpot_ms=15.0,
            duration_ms=1000.0,
            output_tokens_per_second=64.0,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
            is_warmup=True,
        )
        suite = _create_mock_suite(warmup_requests=[req])
        spec = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)

        record = DatasetBuilder.create_record(req, suite, spec)
        assert record.is_warmup is True
        assert record.is_eligible_for_ttft is False
        assert record.is_eligible_for_tpot is False
        assert record.is_eligible_for_chunk_latency is False
        assert record.is_eligible_for_throughput is False
        assert record.is_eligible_for_vram is False
        assert "warmup" in record.ineligibility_reasons[PredictionTarget.TTFT_MS][0].lower()

    def test_failed_or_incomplete_request_ineligible_for_all_targets(self) -> None:
        """Failed or prematurely cancelled requests must be excluded from all targets."""
        req = RequestBenchmarkResult(
            request_id="req-fail-1",
            success=False,
            error_message="CUDA out of memory",
            prompt_tokens=128,
            output_tokens=0,
            duration_ms=50.0,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
        )
        suite = _create_mock_suite(requests=[req])
        spec = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)

        record = DatasetBuilder.create_record(req, suite, spec)
        assert record.is_eligible_for_ttft is False
        assert record.is_eligible_for_tpot is False
        assert record.is_eligible_for_throughput is False
        assert any(
            "unsuccessful" in r for r in record.ineligibility_reasons[PredictionTarget.TTFT_MS]
        )

    def test_zero_output_tokens_ineligible_for_ttft_and_tpot(self) -> None:
        """Prefill-only requests producing 0 output tokens cannot provide TTFT or TPOT."""
        req = RequestBenchmarkResult(
            request_id="req-prefill-only",
            success=True,
            prompt_tokens=256,
            output_tokens=0,
            ttft_ms=None,
            tpot_ms=None,
            duration_ms=100.0,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
        )
        suite = _create_mock_suite(requests=[req])
        spec = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)

        record = DatasetBuilder.create_record(req, suite, spec)
        assert record.is_eligible_for_ttft is False
        assert record.is_eligible_for_tpot is False

    def test_single_token_output_eligible_for_ttft_but_ineligible_for_tpot(self) -> None:
        """A single output token provides TTFT but has no inter-token interval (TPOT requires >= 2 tokens)."""
        req = RequestBenchmarkResult(
            request_id="req-single-token",
            success=True,
            prompt_tokens=128,
            output_tokens=1,
            ttft_ms=30.0,
            tpot_ms=None,
            duration_ms=30.0,
            output_tokens_per_second=33.3,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
            token_count_source=TokenCountSource.TOKEN_EVENTS,
        )
        suite = _create_mock_suite(requests=[req])
        spec = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)

        record = DatasetBuilder.create_record(req, suite, spec)
        assert record.is_eligible_for_ttft is True
        assert record.is_eligible_for_tpot is False
        assert ">= 2 output tokens" in record.ineligibility_reasons[PredictionTarget.TPOT_MS][0]

    def test_chunk_streaming_mode_strictly_ineligible_for_token_tpot(self) -> None:
        """Text chunks must NEVER be treated as tokens. Chunk streams without token IDs are ineligible for TPOT."""
        req = RequestBenchmarkResult(
            request_id="req-chunk-stream",
            success=True,
            prompt_tokens=64,
            output_tokens=32,
            ttft_ms=40.0,
            tpot_ms=None,  # No token-level TPOT
            duration_ms=600.0,
            output_tokens_per_second=53.3,
            chunk_count=8,
            chunk_latencies_ms=[20.0] * 7,
            chunk_latency_p50_ms=20.0,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
            token_count_source=TokenCountSource.SERVER_REPORTED,
        )
        suite = _create_mock_suite(requests=[req])
        spec = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)

        record = DatasetBuilder.create_record(req, suite, spec)
        assert record.is_eligible_for_tpot is False
        assert (
            "chunks must not be treated as tokens"
            in record.ineligibility_reasons[PredictionTarget.TPOT_MS][0].lower()
        )
        # But eligible for chunk latency!
        assert record.is_eligible_for_chunk_latency is True

    def test_valid_token_stream_eligible_for_tpot(self) -> None:
        """Valid discrete token events yield eligible token-level TPOT targets."""
        req = RequestBenchmarkResult(
            request_id="req-token-stream",
            success=True,
            prompt_tokens=64,
            output_tokens=32,
            ttft_ms=25.0,
            tpot_ms=12.5,
            duration_ms=425.0,
            output_tokens_per_second=75.3,
            inter_token_latencies_ms=[12.5] * 31,
            itl_p50_ms=12.5,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
            token_count_source=TokenCountSource.TOKEN_EVENTS,
        )
        suite = _create_mock_suite(requests=[req])
        spec = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)

        record = DatasetBuilder.create_record(req, suite, spec)
        assert record.is_eligible_for_ttft is True
        assert record.is_eligible_for_tpot is True
        assert record.is_eligible_for_throughput is True

    def test_unmetered_vram_telemetry_ineligible_for_memory_target(self) -> None:
        """If memory telemetry was unavailable, record is marked ineligible for PEAK_VRAM_MB."""
        req = RequestBenchmarkResult(
            request_id="req-unmetered",
            success=True,
            prompt_tokens=64,
            output_tokens=16,
            ttft_ms=20.0,
            tpot_ms=10.0,
            duration_ms=180.0,
            output_tokens_per_second=88.9,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
        )
        suite = _create_mock_suite(requests=[req], telemetry_available=False)
        spec = ModelSpec(model_id="llama-7b", parameter_count_billion=7.0)

        record = DatasetBuilder.create_record(req, suite, spec)
        assert record.is_eligible_for_vram is False
        assert (
            "telemetry was unavailable"
            in record.ineligibility_reasons[PredictionTarget.PEAK_VRAM_MB][0].lower()
        )


class TestDatasetBuilderAndFeaturePreservation:
    """Test suite for PredictionDataset compilation and metadata extraction."""

    def test_missing_features_preserved_as_none_without_fabrication(self) -> None:
        """Optional architectural and hardware fields must remain None when missing."""
        req = RequestBenchmarkResult(
            request_id="req-minimal",
            success=True,
            prompt_tokens=100,
            output_tokens=50,
            ttft_ms=15.0,
            tpot_ms=10.0,
            duration_ms=515.0,
            output_tokens_per_second=97.0,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
        )
        suite = _create_mock_suite(requests=[req])
        # Minimal spec with missing optional dimensions
        spec = ModelSpec(model_id="sparse-spec-model")

        record = DatasetBuilder.create_record(req, suite, spec, hardware_report=None)
        assert record.parameter_count_billion is None
        assert record.num_layers is None
        assert record.num_attention_heads is None
        assert record.num_kv_heads is None
        assert record.hidden_dimension is None
        assert record.head_dimension is None
        assert record.hardware_catalog_id is None
        assert record.gpu_memory_capacity_gb is None
        assert record.gpu_memory_bandwidth_gbps is None
        assert record.gpu_peak_compute_tflops is None
        assert record.host_ram_total_gb is None

    def test_from_benchmark_suite_builds_prediction_dataset(self) -> None:
        """Valid benchmark suite builds typed PredictionDataset with query methods."""
        req1 = RequestBenchmarkResult(
            request_id="req-1",
            success=True,
            prompt_tokens=64,
            output_tokens=32,
            ttft_ms=20.0,
            tpot_ms=10.0,
            duration_ms=340.0,
            output_tokens_per_second=94.1,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
        )
        warmup1 = RequestBenchmarkResult(
            request_id="req-w1",
            success=True,
            prompt_tokens=64,
            output_tokens=32,
            ttft_ms=30.0,
            tpot_ms=12.0,
            duration_ms=414.0,
            output_tokens_per_second=77.3,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.SIMULATED,
            provenance=ProvenanceType.MEASURED,
            is_warmup=True,
        )
        suite = _create_mock_suite(requests=[req1], warmup_requests=[warmup1])
        spec = ModelSpec(
            model_id="meta-llama/Llama-3-8B",
            parameter_count_billion=8.0,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dimension=128,
            quantization=QuantizationType.FP16,
        )
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-h100-sxm5-80gb")

        ds = DatasetBuilder.from_benchmark_suite(suite, spec, hw_report, dataset_id="ds-test-01")
        assert isinstance(ds, PredictionDataset)
        assert len(ds) == 2  # 1 measured + 1 warmup

        # Query eligible records for TTFT: warmup must be excluded
        ttft_eligible = ds.get_eligible_records(PredictionTarget.TTFT_MS)
        assert len(ttft_eligible) == 1
        assert ttft_eligible[0].request_id == "req-1"
        assert ttft_eligible[0].parameter_count_billion == 8.0
        assert ttft_eligible[0].hardware_catalog_id == "nvidia-h100-sxm5-80gb"
        assert ttft_eligible[0].gpu_memory_bandwidth_gbps == 3350.0
