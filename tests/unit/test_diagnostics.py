"""Unit tests for the evidence-based diagnostic engine.

Validates memory capacity bounds, context window violations, latency anomalies,
throughput efficiency vs roofline ceilings, and telemetry gap detections.
"""

from llm_ii_engine.analysis.diagnostics import DiagnosticEngine
from llm_ii_engine.analysis.models import DiagnosticReport
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
from llm_ii_engine.core.schemas import ModelSpec
from llm_ii_engine.discovery.simulated import get_simulated_profile


def _build_suite_result(
    model_id: str = "test-model-7b",
    concurrency: int = 1,
    ttft_p50_ms: float = 25.0,
    itl_p50_ms: float = 20.0,
    itl_p99_ms: float = 22.0,
    aggregate_out_tok_s: float = 50.0,
    requests: list[RequestBenchmarkResult] | None = None,
    execution_mode: ExecutionMode = ExecutionMode.REAL,
    telemetry_available: bool = True,
) -> BenchmarkSuiteResult:
    """Helper to construct a mock BenchmarkSuiteResult."""
    if requests is None:
        requests = [
            RequestBenchmarkResult(
                request_id="req-1",
                success=True,
                prompt_tokens=128,
                output_tokens=64,
                ttft_ms=ttft_p50_ms,
                tpot_ms=itl_p50_ms,
                duration_ms=1300.0,
                output_tokens_per_second=aggregate_out_tok_s,
                inter_token_latencies_ms=[itl_p50_ms] * 63,
                itl_p50_ms=itl_p50_ms,
                itl_p99_ms=itl_p99_ms,
                clock_source=ClockSource.MONOTONIC_REAL,
                execution_mode=execution_mode,
                provenance=ProvenanceType.MEASURED,
                token_count_source=TokenCountSource.TOKEN_EVENTS,
            )
        ]

    return BenchmarkSuiteResult(
        suite_id="suite-diag-001",
        status="COMPLETED",
        model_id=model_id,
        engine_type=EngineType.MOCK,
        concurrency=concurrency,
        total_requests=len(requests),
        warmup_requests=0,
        successful_requests=len(requests),
        failed_requests=0,
        wall_time_seconds=2.0,
        aggregate_tokens_per_second=aggregate_out_tok_s * 1.5,
        aggregate_output_tokens_per_second=aggregate_out_tok_s,
        request_throughput_per_second=float(len(requests)) / 2.0,
        ttft_p50_ms=ttft_p50_ms,
        itl_p50_ms=itl_p50_ms,
        itl_p99_ms=itl_p99_ms,
        memory_watermark=MemoryWatermarkResult(
            peak_process_rss_mb=500.0,
            peak_gpu_vram_mb=4000.0 if telemetry_available else None,
            sample_count=10,
            sample_interval_ms=100.0,
            telemetry_source="nvidia-smi" if telemetry_available else "unavailable",
            is_telemetry_available=telemetry_available,
        ),
        execution_mode=execution_mode,
        clock_source=ClockSource.MONOTONIC_REAL,
        provenance=ProvenanceType.MEASURED,
        request_results=requests,
    )


class TestDiagnosticEngineRules:
    """Test suite for individual diagnostic rules and report compilation."""

    def test_infeasible_memory_capacity_triggers_critical_finding(self) -> None:
        """70B unquantized FP16 model (~133.5 GB) on 24GB RTX 4090 must trigger CRITICAL."""
        model_spec = ModelSpec(
            model_id="meta-llama/Llama-2-70b-hf",
            parameter_count_billion=70.0,
            num_layers=80,
            num_attention_heads=64,
            num_kv_heads=8,
            head_dimension=128,
            context_window=4096,
            quantization=QuantizationType.FP16,
        )
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-rtx-4090")
        suite = _build_suite_result(model_id=model_spec.model_id)

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=hw_report,
        )

        assert isinstance(report, DiagnosticReport)
        crit_findings = [f for f in report.findings if f.severity == DiagnosticSeverity.CRITICAL]
        assert len(crit_findings) >= 1

        mem_finding = next(
            f for f in crit_findings if f.category == DiagnosticCategory.MEMORY_CAPACITY
        )
        assert "Infeasible Memory Configuration" in mem_finding.title
        assert mem_finding.confidence == DiagnosticConfidence.HIGH
        assert len(mem_finding.suggested_experiments) > 0
        assert "CRITICAL" in report.summary

    def test_high_memory_utilization_triggers_headroom_warning(self) -> None:
        """Model whose runtime VRAM occupies > 85% of GPU memory triggers WARNING."""
        # 10B model at FP16 = ~19.07 GB payload + 15% margin + 512MB = ~22.4 GB runtime on 24GB RTX 4090 (> 85%)
        model_spec = ModelSpec(
            model_id="custom-10b",
            parameter_count_billion=10.0,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dimension=128,
            quantization=QuantizationType.FP16,
        )
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-rtx-4090")
        suite = _build_suite_result(model_id=model_spec.model_id)

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=hw_report,
        )

        warn_findings = [f for f in report.findings if f.severity == DiagnosticSeverity.WARNING]
        mem_warn = next(
            f for f in warn_findings if f.category == DiagnosticCategory.MEMORY_CAPACITY
        )
        assert "High Memory Footprint" in mem_warn.title
        assert mem_warn.confidence == DiagnosticConfidence.MEDIUM

    def test_context_window_violation_triggers_critical_finding(self) -> None:
        """Workload sequence length exceeding context window must trigger CRITICAL."""
        model_spec = ModelSpec(
            model_id="restricted-context-model",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            head_dimension=128,
            context_window=100,  # Small context window
        )
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-h100-sxm5-80gb")
        # Suite has prompt 128 + output 64 = 192 tokens > 100
        suite = _build_suite_result(model_id=model_spec.model_id)

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=hw_report,
        )

        crit_findings = [f for f in report.findings if f.severity == DiagnosticSeverity.CRITICAL]
        ctx_finding = next(
            f for f in crit_findings if f.category == DiagnosticCategory.CONFIGURATION_VIABILITY
        )
        assert "Context Window Exceeded" in ctx_finding.title
        assert ctx_finding.confidence == DiagnosticConfidence.HIGH

    def test_elevated_ttft_triggers_latency_warning(self) -> None:
        """Measured TTFT much higher than theoretical compute time triggers WARNING."""
        model_spec = ModelSpec(
            model_id="fast-7b",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dimension=128,
            context_window=4096,
        )
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-h100-sxm5-80gb")
        # On H100 SXM5, 128 prompt tokens for 7B requires ~1.8 TFLOPs -> compute duration < 2 ms.
        # Setting measured TTFT to 250 ms triggers the > 3x compute threshold.
        suite = _build_suite_result(
            model_id=model_spec.model_id,
            ttft_p50_ms=250.0,
        )

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=hw_report,
        )

        ttft_warn = next(
            f
            for f in report.findings
            if f.category == DiagnosticCategory.LATENCY_OVERHEAD and "TTFT" in f.title
        )
        assert ttft_warn.severity == DiagnosticSeverity.WARNING
        assert ttft_warn.confidence == DiagnosticConfidence.MEDIUM
        assert "server queue delay" in ttft_warn.limitations[0]

    def test_high_streaming_jitter_triggers_warning(self) -> None:
        """ITL p99 / p50 > 3.0 with significant delta triggers streaming jitter WARNING."""
        model_spec = ModelSpec(
            model_id="jitter-model",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            head_dimension=128,
        )
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-h100-sxm5-80gb")
        # p50 = 10 ms, p99 = 80 ms (8x ratio, 70 ms delta)
        suite = _build_suite_result(
            model_id=model_spec.model_id,
            itl_p50_ms=10.0,
            itl_p99_ms=80.0,
        )

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=hw_report,
        )

        jitter_finding = next(
            f
            for f in report.findings
            if f.category == DiagnosticCategory.LATENCY_OVERHEAD and "Jitter" in f.title
        )
        assert jitter_finding.severity == DiagnosticSeverity.WARNING
        assert jitter_finding.confidence == DiagnosticConfidence.HIGH

    def test_sub_roofline_decode_throughput_diagnosed_with_low_confidence(self) -> None:
        """Throughput < 40% roofline ceiling is flagged with LOW confidence and DRAM limitation notes."""
        model_spec = ModelSpec(
            model_id="low-throughput-7b",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dimension=128,
        )
        # H100 SXM5 roofline ceiling for 7B batch 1 decode is ~239.29 tok/s.
        # Setting measured aggregate decode throughput to 20.0 tok/s (< 10% of ceiling).
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-h100-sxm5-80gb")
        suite = _build_suite_result(
            model_id=model_spec.model_id,
            aggregate_out_tok_s=20.0,
        )

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=hw_report,
        )

        eff_finding = next(
            f for f in report.findings if f.category == DiagnosticCategory.EFFICIENCY_BOTTLENECK
        )
        assert eff_finding.confidence == DiagnosticConfidence.LOW
        assert "hardware performance counters" in eff_finding.confidence_rationale
        assert any("DRAM" in lim for lim in eff_finding.limitations)

    def test_telemetry_gap_reported_when_unmetered(self) -> None:
        """Report notes telemetry gaps when GPU is unmatched or memory watermark is unavailable."""
        model_spec = ModelSpec(
            model_id="gap-model",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            head_dimension=128,
        )
        # No hardware report, memory telemetry unavailable
        suite = _build_suite_result(
            model_id=model_spec.model_id,
            telemetry_available=False,
        )

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=None,
        )

        telemetry_findings = [
            f for f in report.findings if f.category == DiagnosticCategory.TELEMETRY_GAP
        ]
        assert len(telemetry_findings) >= 2  # Unmatched GPU + Missing memory telemetry

    def test_chunk_streaming_mode_triggers_info_finding(self) -> None:
        """Requests with chunk counts or server-reported tokens are flagged as chunk mode."""
        model_spec = ModelSpec(
            model_id="chunk-model",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            head_dimension=128,
        )
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-h100-sxm5-80gb")
        req = RequestBenchmarkResult(
            request_id="chunk-req-1",
            success=True,
            prompt_tokens=64,
            output_tokens=32,
            ttft_ms=30.0,
            tpot_ms=None,  # No per-token TPOT
            duration_ms=500.0,
            output_tokens_per_second=64.0,
            chunk_count=8,
            chunk_latencies_ms=[15.0] * 7,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            token_count_source=TokenCountSource.SERVER_REPORTED,
        )
        suite = _build_suite_result(model_id=model_spec.model_id, requests=[req])

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=hw_report,
        )

        chunk_finding = next(f for f in report.findings if "Text-Chunk Streaming" in f.title)
        assert chunk_finding.severity == DiagnosticSeverity.INFO
        assert chunk_finding.confidence == DiagnosticConfidence.HIGH

    def test_clean_operating_envelope_produces_info_finding(self) -> None:
        """Healthy workload within limits produces INFO 'Normal Operating Envelope' finding."""
        model_spec = ModelSpec(
            model_id="healthy-7b",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dimension=128,
            context_window=4096,
        )
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-h100-sxm5-80gb")
        # Moderate TTFT, low jitter, high throughput
        suite = _build_suite_result(
            model_id=model_spec.model_id,
            ttft_p50_ms=5.0,  # very low TTFT
            itl_p50_ms=5.0,
            itl_p99_ms=5.5,  # minimal jitter
            aggregate_out_tok_s=150.0,  # > 40% of roofline ceiling
        )

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=hw_report,
        )

        clean_finding = next(f for f in report.findings if "Normal Operating Envelope" in f.title)
        assert clean_finding.severity == DiagnosticSeverity.INFO
        assert clean_finding.confidence == DiagnosticConfidence.HIGH
        assert not any(
            f.severity in (DiagnosticSeverity.CRITICAL, DiagnosticSeverity.WARNING)
            for f in report.findings
        )

    def test_execution_mode_and_provenance_preservation(self) -> None:
        """Diagnostic report strictly preserves SIMULATED vs REAL execution mode."""
        model_spec = ModelSpec(
            model_id="sim-model",
            parameter_count_billion=7.0,
            num_layers=32,
            num_attention_heads=32,
            head_dimension=128,
        )
        hw_report = get_simulated_profile(gpu_catalog_id="nvidia-h100-sxm5-80gb")
        suite = _build_suite_result(
            model_id=model_spec.model_id,
            execution_mode=ExecutionMode.SIMULATED,
        )

        report = DiagnosticEngine.analyze(
            benchmark_suite=suite,
            model_spec=model_spec,
            hardware_report=hw_report,
        )

        assert report.execution_mode == ExecutionMode.SIMULATED
        for finding in report.findings:
            for obs in finding.supporting_observations:
                assert obs.execution_mode == ExecutionMode.SIMULATED
