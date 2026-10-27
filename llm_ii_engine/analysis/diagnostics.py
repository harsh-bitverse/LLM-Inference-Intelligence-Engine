"""Evidence-based diagnostic engine for LLM inference benchmarks.

Evaluates benchmark results against analytical resource models and theoretical rooflines
to detect memory bottlenecks, latency anomalies, configuration violations, and telemetry gaps.
All findings strictly enforce scientific honesty and explicit confidence rationales.
"""

import contextlib
import statistics
import uuid
from datetime import UTC, datetime

from llm_ii_engine.analysis.models import (
    DiagnosticFinding,
    DiagnosticReport,
    ModelResourceEstimate,
    RooflineAnalysisResult,
)
from llm_ii_engine.analysis.resource_estimator import ModelResourceEstimator
from llm_ii_engine.analysis.roofline import RooflineAnalyzer
from llm_ii_engine.benchmark.models import BenchmarkSuiteResult
from llm_ii_engine.core.enums import (
    DiagnosticCategory,
    DiagnosticConfidence,
    DiagnosticSeverity,
    ProvenanceType,
    QuantizationType,
    TokenCountSource,
)
from llm_ii_engine.core.schemas import ModelSpec, Observation
from llm_ii_engine.discovery.models import HardwareCapabilityReport


class DiagnosticEngine:
    """Analyzes benchmark execution evidence and compiles structured diagnostic reports."""

    @classmethod
    def analyze(
        cls,
        benchmark_suite: BenchmarkSuiteResult,
        model_spec: ModelSpec,
        hardware_report: HardwareCapabilityReport | None = None,
        resource_estimate: ModelResourceEstimate | None = None,
        roofline_analysis: RooflineAnalysisResult | None = None,
    ) -> DiagnosticReport:
        """Execute diagnostic rules against benchmark results, model specs, and hardware.

        Args:
            benchmark_suite: Completed benchmark suite results.
            model_spec: Model architecture specifications.
            hardware_report: Optional hardware discovery report.
            resource_estimate: Optional precomputed model resource estimates.
            roofline_analysis: Optional precomputed roofline analysis.

        Returns:
            Validated DiagnosticReport with structured findings, estimates, and summary.
        """
        findings: list[DiagnosticFinding] = []
        exec_mode = benchmark_suite.execution_mode

        # 1. Resolve or compute ModelResourceEstimate
        if resource_estimate is None:
            # Derive prompt and output lengths from benchmark suite if available
            prompt_len = 128
            output_len = 128
            if benchmark_suite.request_results:
                prompts = [
                    r.prompt_tokens
                    for r in benchmark_suite.request_results
                    if r.prompt_tokens is not None and r.prompt_tokens > 0
                ]
                outputs = [
                    r.output_tokens for r in benchmark_suite.request_results if r.output_tokens > 0
                ]
                if prompts:
                    prompt_len = max(1, int(statistics.median(prompts)))
                if outputs:
                    output_len = max(1, int(statistics.median(outputs)))

            weight_prec = model_spec.quantization or QuantizationType.NONE
            resource_estimate = ModelResourceEstimator.estimate_model_resources(
                model_spec=model_spec,
                prompt_tokens=prompt_len,
                output_tokens=output_len,
                batch_size=benchmark_suite.concurrency,
                weight_precision=weight_prec,
            )

        # 2. Resolve or compute RooflineAnalysisResult
        primary_gpu_spec = None
        if hardware_report is not None and hardware_report.matched_catalog_specs:
            primary_gpu_spec = hardware_report.matched_catalog_specs[0]

        if roofline_analysis is None and primary_gpu_spec is not None:
            # Use decode operational intensity if computed, else default fallback
            oi = resource_estimate.operational_intensity_decode or 1.0
            oi_prov = (
                resource_estimate.provenance
                if resource_estimate.operational_intensity_decode is not None
                else ProvenanceType.HEURISTIC
            )
            # Match precision
            prec = "fp16_dense"
            if model_spec.quantization in (QuantizationType.INT8, QuantizationType.GGUF_Q8_0):
                prec = "int8_dense"
            elif model_spec.quantization == QuantizationType.BF16:
                prec = "bf16_dense"

            with contextlib.suppress(ValueError):
                roofline_analysis = RooflineAnalyzer.evaluate_roofline(
                    gpu_spec=primary_gpu_spec,
                    operational_intensity=oi,
                    operational_intensity_provenance=oi_prov,
                    precision=prec,
                    parameter_count_billion=model_spec.parameter_count_billion,
                )

        # --- Rule 1: Memory Feasibility & Headroom ---
        available_memory_mb: float | None = None
        device_label = "Memory"
        if primary_gpu_spec is not None:
            available_memory_mb = primary_gpu_spec.memory_capacity_gb * 1024.0
            device_label = f"GPU VRAM ({primary_gpu_spec.name})"
        elif hardware_report is not None:
            available_memory_mb = hardware_report.memory.total_mb
            device_label = "Host RAM"

        if available_memory_mb is not None:
            payload_mb = resource_estimate.total_payload_memory_mb
            runtime_vram_mb = resource_estimate.estimated_runtime_vram_mb or payload_mb

            if payload_mb > available_memory_mb:
                findings.append(
                    DiagnosticFinding(
                        finding_id=f"MEM-CAP-CRIT-{uuid.uuid4().hex[:6]}",
                        category=DiagnosticCategory.MEMORY_CAPACITY,
                        severity=DiagnosticSeverity.CRITICAL,
                        title="Infeasible Memory Configuration: Model Exceeds Available Capacity",
                        message=(
                            f"Theoretical model payload memory ({payload_mb:.1f} MiB) exceeds available {device_label} "
                            f"capacity ({available_memory_mb:.1f} MiB). The model cannot be loaded into active device "
                            "memory without aggressive layer offloading or lower-bit quantization."
                        ),
                        confidence=DiagnosticConfidence.HIGH,
                        confidence_rationale=(
                            "Derived from deterministic transformer weight count and KV-cache byte formulas "
                            "compared directly against verified device capacity limits."
                        ),
                        supporting_observations=[
                            Observation[float](
                                value=payload_mb,
                                provenance=ProvenanceType.PREDICTED,
                                execution_mode=exec_mode,
                                unit="MiB",
                                description="Theoretical model payload memory (weights + KV cache)",
                            ),
                            Observation[float](
                                value=available_memory_mb,
                                provenance=ProvenanceType.HEURISTIC,
                                execution_mode=exec_mode,
                                unit="MiB",
                                description=f"Documented capacity of {device_label}",
                            ),
                        ],
                        assumptions=[
                            "Model weights and KV cache reside entirely in active device memory without external paging."
                        ],
                        limitations=[
                            "Does not account for partial offload layers or unified memory paging if supported by the engine."
                        ],
                        suggested_experiments=[
                            "Apply lower-bit quantization (e.g. INT4, AWQ, GGUF_Q4_K_M)",
                            "Reduce max context sequence length or concurrency batch size",
                            "Enable engine-specific GPU layer offloading",
                        ],
                    )
                )
            elif runtime_vram_mb > (0.85 * available_memory_mb):
                findings.append(
                    DiagnosticFinding(
                        finding_id=f"MEM-CAP-WARN-{uuid.uuid4().hex[:6]}",
                        category=DiagnosticCategory.MEMORY_CAPACITY,
                        severity=DiagnosticSeverity.WARNING,
                        title="High Memory Footprint: Elevated OOM Risk",
                        message=(
                            f"Estimated runtime memory footprint ({runtime_vram_mb:.1f} MiB) occupies over 85% of "
                            f"available {device_label} capacity ({available_memory_mb:.1f} MiB). Transient KV-cache "
                            "growth or allocator fragmentation may trigger out-of-memory errors during sustained workloads."
                        ),
                        confidence=DiagnosticConfidence.MEDIUM,
                        confidence_rationale=(
                            "Estimated runtime memory incorporates a 15% allocator margin; actual fragmentation "
                            "depends on engine paging implementation."
                        ),
                        supporting_observations=[
                            Observation[float](
                                value=runtime_vram_mb,
                                provenance=ProvenanceType.PREDICTED,
                                execution_mode=exec_mode,
                                unit="MiB",
                                description="Estimated runtime VRAM footprint",
                            ),
                            Observation[float](
                                value=available_memory_mb,
                                provenance=ProvenanceType.HEURISTIC,
                                execution_mode=exec_mode,
                                unit="MiB",
                                description=f"Documented capacity of {device_label}",
                            ),
                        ],
                        assumptions=[
                            "15% allocator/activation margin and 512 MiB baseline driver/runtime overhead."
                        ],
                        limitations=[
                            "Actual allocation depends on engine paging algorithms (e.g. PagedAttention vs contiguous buffer)."
                        ],
                        suggested_experiments=[
                            "Monitor peak memory watermark under full sequence length",
                            "Reduce batch concurrency",
                        ],
                    )
                )

        # --- Rule 2: Context Window Viability ---
        if model_spec.context_window is not None:
            evaluated_seq_len = resource_estimate.sequence_length
            if evaluated_seq_len > model_spec.context_window:
                findings.append(
                    DiagnosticFinding(
                        finding_id=f"CFG-CTX-CRIT-{uuid.uuid4().hex[:6]}",
                        category=DiagnosticCategory.CONFIGURATION_VIABILITY,
                        severity=DiagnosticSeverity.CRITICAL,
                        title="Context Window Exceeded",
                        message=(
                            f"Evaluated sequence length ({evaluated_seq_len} tokens) exceeds the architecture's "
                            f"maximum context window ({model_spec.context_window} tokens). The engine will reject "
                            "the request or truncate tokens, causing context loss or positional embedding degradation."
                        ),
                        confidence=DiagnosticConfidence.HIGH,
                        confidence_rationale="Direct violation of model architectural context boundary.",
                        supporting_observations=[
                            Observation[int](
                                value=evaluated_seq_len,
                                provenance=ProvenanceType.PREDICTED,
                                execution_mode=exec_mode,
                                unit="tokens",
                                description="Evaluated total sequence length",
                            ),
                            Observation[int](
                                value=model_spec.context_window,
                                provenance=ProvenanceType.HEURISTIC,
                                execution_mode=exec_mode,
                                unit="tokens",
                                description="Model architectural context window limit",
                            ),
                        ],
                        assumptions=[
                            "Model uses fixed positional embeddings without dynamic RoPE scaling or context extension."
                        ],
                        limitations=[
                            "Engine may support dynamic RoPE scaling / YaRN context extension if configured."
                        ],
                        suggested_experiments=[
                            "Truncate prompt length to fit within the context window",
                            "Enable RoPE scaling in engine configuration if supported",
                        ],
                    )
                )

        # --- Rule 3: Elevated TTFT Latency Overhead ---
        if benchmark_suite.ttft_p50_ms is not None and resource_estimate.prefill_flops is not None:
            # Derive prefill attainable compute TFLOPs using prefill operational intensity
            prefill_attainable_tflops: float | None = None
            if primary_gpu_spec is not None:
                prefill_oi = resource_estimate.operational_intensity_prefill or 50.0
                peak_c = primary_gpu_spec.compute.fp16_dense_tflops or 100.0
                bw = primary_gpu_spec.theoretical_memory_bandwidth_gbps
                prefill_attainable_tflops = min(peak_c, (prefill_oi * bw) / 1000.0)
            elif roofline_analysis is not None:
                prefill_attainable_tflops = roofline_analysis.peak_compute_tflops

            if prefill_attainable_tflops is not None and prefill_attainable_tflops > 0:
                t_compute_ms = (
                    resource_estimate.prefill_flops / (prefill_attainable_tflops * 1e12)
                ) * 1000.0
                if (
                    benchmark_suite.ttft_p50_ms > (3.0 * t_compute_ms)
                    and benchmark_suite.ttft_p50_ms > 50.0
                ):
                    findings.append(
                        DiagnosticFinding(
                            finding_id=f"LAT-TTFT-WARN-{uuid.uuid4().hex[:6]}",
                            category=DiagnosticCategory.LATENCY_OVERHEAD,
                            severity=DiagnosticSeverity.WARNING,
                            title="Elevated Time-To-First-Token (TTFT) Relative to Theoretical Compute",
                            message=(
                                f"Measured median TTFT ({benchmark_suite.ttft_p50_ms:.1f} ms) exceeds theoretical prefill "
                                f"compute duration ({t_compute_ms:.1f} ms) by more than 3x. TTFT reflects prefill matrix "
                                "computations plus queueing delay, prompt tokenization, KV-cache allocation, and HTTP network "
                                "roundtrips. Without server-side execution traces, queue wait time vs tokenization cannot be isolated."
                            ),
                            confidence=DiagnosticConfidence.MEDIUM,
                            confidence_rationale=(
                                "TTFT is directly measured at the client boundary, but component-level breakdown "
                                "(tokenization vs queue vs kernel launch) requires server-internal telemetry."
                            ),
                            supporting_observations=[
                                Observation[float](
                                    value=benchmark_suite.ttft_p50_ms,
                                    provenance=benchmark_suite.provenance,
                                    execution_mode=exec_mode,
                                    clock_source=benchmark_suite.clock_source,
                                    unit="ms",
                                    description="Measured median TTFT (p50)",
                                ),
                                Observation[float](
                                    value=round(t_compute_ms, 2),
                                    provenance=ProvenanceType.HEURISTIC,
                                    execution_mode=exec_mode,
                                    unit="ms",
                                    description="Theoretical prefill compute duration under roofline ceiling",
                                ),
                            ],
                            assumptions=[
                                "Theoretical prefill FLOPs assume dense GEMM throughput without kernel launch delays."
                            ],
                            limitations=[
                                "Client-side measurements cannot separate prompt tokenization from server queue delay."
                            ],
                            suggested_experiments=[
                                "Run TTFT sweep across varying prompt lengths (e.g. 16, 64, 256, 1024) to isolate fixed launch/network latency from marginal compute time per prompt token.",
                            ],
                        )
                    )

        # --- Rule 4: Streaming Inter-Arrival Jitter ---
        if (
            benchmark_suite.itl_p50_ms is not None
            and benchmark_suite.itl_p99_ms is not None
            and benchmark_suite.itl_p50_ms > 0
        ):
            ratio = benchmark_suite.itl_p99_ms / benchmark_suite.itl_p50_ms
            delta = benchmark_suite.itl_p99_ms - benchmark_suite.itl_p50_ms
            if ratio > 3.0 and delta > 20.0:
                findings.append(
                    DiagnosticFinding(
                        finding_id=f"LAT-JITTER-WARN-{uuid.uuid4().hex[:6]}",
                        category=DiagnosticCategory.LATENCY_OVERHEAD,
                        severity=DiagnosticSeverity.WARNING,
                        title="High Inter-Token Latency (ITL) Jitter in Stream",
                        message=(
                            f"Streaming inter-token latency exhibits high jitter (p50: {benchmark_suite.itl_p50_ms:.1f} ms, "
                            f"p99: {benchmark_suite.itl_p99_ms:.1f} ms, ratio: {ratio:.1f}x). Jitter can result from "
                            "engine batch preemption, GC pauses, memory reallocation, or network packet buffering."
                        ),
                        confidence=DiagnosticConfidence.HIGH,
                        confidence_rationale="Direct statistical measurement from monotonic timestamp deltas across stream events.",
                        supporting_observations=[
                            Observation[float](
                                value=benchmark_suite.itl_p50_ms,
                                provenance=benchmark_suite.provenance,
                                execution_mode=exec_mode,
                                clock_source=benchmark_suite.clock_source,
                                unit="ms",
                                description="Measured median ITL (p50)",
                            ),
                            Observation[float](
                                value=benchmark_suite.itl_p99_ms,
                                provenance=benchmark_suite.provenance,
                                execution_mode=exec_mode,
                                clock_source=benchmark_suite.clock_source,
                                unit="ms",
                                description="Measured 99th percentile ITL (p99)",
                            ),
                        ],
                        assumptions=[
                            "Client timer has sufficient resolution to detect millisecond inter-token intervals."
                        ],
                        limitations=[
                            "Does not reveal whether jitter is caused by OS scheduling, network transport, or GPU memory contention."
                        ],
                        suggested_experiments=[
                            "Test on localhost to eliminate network variance",
                            "Check engine thread affinity and GPU power governance",
                        ],
                    )
                )

        # --- Rule 5: Decode Throughput vs Roofline Ceiling ---
        if (
            roofline_analysis is not None
            and roofline_analysis.theoretical_max_tokens_per_second is not None
            and benchmark_suite.aggregate_output_tokens_per_second is not None
            and benchmark_suite.aggregate_output_tokens_per_second > 0
        ):
            measured_tok_s = benchmark_suite.aggregate_output_tokens_per_second
            ceiling_tok_s = roofline_analysis.theoretical_max_tokens_per_second
            ratio = measured_tok_s / ceiling_tok_s
            if ratio < 0.40:
                severity = DiagnosticSeverity.WARNING if ratio < 0.25 else DiagnosticSeverity.INFO
                findings.append(
                    DiagnosticFinding(
                        finding_id=f"EFF-THROUGHPUT-{uuid.uuid4().hex[:6]}",
                        category=DiagnosticCategory.EFFICIENCY_BOTTLENECK,
                        severity=severity,
                        title="Observed Decode Throughput Substantially Below Theoretical Hardware Ceiling",
                        message=(
                            f"Observed decode throughput ({measured_tok_s:.1f} tok/s) is {ratio * 100:.1f}% of the theoretical "
                            f"roofline ceiling ({ceiling_tok_s:.1f} tok/s). Potential explanations include: memory bandwidth latency "
                            f"(incomplete warp occupancy at concurrency {benchmark_suite.concurrency}), Python HTTP client overhead, "
                            "engine orchestration overhead, or CPU-GPU synchronization bubbles. CRITICALLY: without hardware "
                            "performance counters (such as DRAM bus utilization or GPU SM active cycles), memory bandwidth "
                            "saturation vs compute stall cannot be conclusively proven."
                        ),
                        confidence=DiagnosticConfidence.LOW,
                        confidence_rationale=(
                            "Assigned LOW confidence because hardware performance counters (DRAM bus traffic, GPU SM active cycles) "
                            "are absent from client-side measurements."
                        ),
                        supporting_observations=[
                            Observation[float](
                                value=measured_tok_s,
                                provenance=benchmark_suite.provenance,
                                execution_mode=exec_mode,
                                unit="tok/s",
                                description="Measured aggregate decode throughput",
                            ),
                            Observation[float](
                                value=ceiling_tok_s,
                                provenance=ProvenanceType.HEURISTIC,
                                execution_mode=exec_mode,
                                unit="tok/s",
                                description="Theoretical roofline token generation ceiling",
                            ),
                        ],
                        assumptions=[
                            "Theoretical roofline represents an ideal 100% memory bus or compute utilization ceiling."
                        ],
                        limitations=[
                            "Hardware counter telemetry (NVLink, DRAM bus, SM occupancy) is unavailable; memory-bandwidth vs compute saturation cannot be conclusively confirmed."
                        ],
                        suggested_experiments=[
                            "Profile with NVIDIA Nsight Compute to capture DRAM throughput and warp stall reasons",
                            "Increase batch concurrency to amortize weight-loading across multiple sequences",
                        ],
                    )
                )

        # --- Rule 6: Telemetry and Observability Gaps ---
        if hardware_report is None or not hardware_report.matched_catalog_specs:
            findings.append(
                DiagnosticFinding(
                    finding_id=f"OBS-HW-INFO-{uuid.uuid4().hex[:6]}",
                    category=DiagnosticCategory.TELEMETRY_GAP,
                    severity=DiagnosticSeverity.INFO,
                    title="Hardware Telemetry Gap: Unmatched GPU Specification",
                    message=(
                        "Benchmark was executed without an authoritative GPU catalog match. "
                        "Roofline ceilings and memory capacity bounds rely on analytical approximations."
                    ),
                    confidence=DiagnosticConfidence.HIGH,
                    confidence_rationale="Direct verification of hardware discovery report completeness.",
                    supporting_observations=[],
                    assumptions=[],
                    limitations=["Live VRAM usage and hardware counters could not be inspected."],
                    suggested_experiments=[
                        "Run on an environment with nvidia-smi / pynvml available and supported GPU",
                    ],
                )
            )

        if (
            benchmark_suite.memory_watermark is None
            or not benchmark_suite.memory_watermark.is_telemetry_available
        ):
            findings.append(
                DiagnosticFinding(
                    finding_id=f"OBS-MEM-INFO-{uuid.uuid4().hex[:6]}",
                    category=DiagnosticCategory.TELEMETRY_GAP,
                    severity=DiagnosticSeverity.INFO,
                    title="Live Memory Telemetry Unavailable",
                    message=(
                        "Runtime memory watermark telemetry was unavailable during the benchmark run. "
                        "Actual peak VRAM and host RSS consumption were not captured."
                    ),
                    confidence=DiagnosticConfidence.HIGH,
                    confidence_rationale="Direct inspection of benchmark memory watermark metadata.",
                    supporting_observations=[],
                    assumptions=[],
                    limitations=["Process RSS and GPU VRAM memory watermarks are unmeasured."],
                    suggested_experiments=[
                        "Ensure pynvml or nvidia-smi is installed and accessible to benchmark process",
                    ],
                )
            )

        # --- Rule 7: Text-Chunk Streaming Mode ---
        has_chunk_streaming = any(
            r.chunk_count > 0 or r.token_count_source != TokenCountSource.TOKEN_EVENTS
            for r in benchmark_suite.request_results
        )
        if has_chunk_streaming:
            findings.append(
                DiagnosticFinding(
                    finding_id=f"STREAM-CHUNK-INFO-{uuid.uuid4().hex[:6]}",
                    category=DiagnosticCategory.TELEMETRY_GAP,
                    severity=DiagnosticSeverity.INFO,
                    title="Text-Chunk Streaming Mode Detected",
                    message=(
                        "Adapter streamed multi-character text chunks rather than discrete token events. "
                        "Calculated latencies reflect network chunk arrival times rather than native token-by-token "
                        "generation TPOT. Token-level TPOT cannot be inferred from chunk-level streams."
                    ),
                    confidence=DiagnosticConfidence.HIGH,
                    confidence_rationale="Verified from event stream structure and engine capabilities.",
                    supporting_observations=[],
                    assumptions=[],
                    limitations=["Engine HTTP API does not expose per-token boundary timestamps."],
                    suggested_experiments=[
                        "Use an engine adapter that yields discrete token events with token IDs for true token-level TPOT",
                    ],
                )
            )

        # --- Rule 8: Heuristic Prompt Word Count ---
        has_heuristic_prompt = any(
            r.prompt_token_count_source == TokenCountSource.HEURISTIC_WORD_COUNT
            for r in benchmark_suite.request_results
        )
        if has_heuristic_prompt:
            findings.append(
                DiagnosticFinding(
                    finding_id=f"PROMPT-HEURISTIC-INFO-{uuid.uuid4().hex[:6]}",
                    category=DiagnosticCategory.TELEMETRY_GAP,
                    severity=DiagnosticSeverity.INFO,
                    title="Heuristic Prompt Word-Count Detected",
                    message=(
                        "Prompt token counts were derived from whitespace word splitting rather than an "
                        "authoritative tokenizer or server-reported evaluation metric. Theoretical prefill "
                        "FLOP and throughput metrics relying on prompt sequence length should be treated as approximations."
                    ),
                    confidence=DiagnosticConfidence.HIGH,
                    confidence_rationale="Direct provenance verification from prompt_token_count_source.",
                    supporting_observations=[],
                    assumptions=["Whitespace word count approximates prompt tokens within ~15-30%."],
                    limitations=["Actual tokenizer-derived prompt length was unobserved."],
                    suggested_experiments=[
                        "Configure explicit prompt_tokens in workload specification or use a server endpoint that reports prompt_eval_count",
                    ],
                )
            )

        # --- Clean / Normal Operating Envelope Fallback ---
        has_critical_or_warning = any(
            f.severity in (DiagnosticSeverity.CRITICAL, DiagnosticSeverity.WARNING)
            for f in findings
        )
        if not has_critical_or_warning:
            findings.append(
                DiagnosticFinding(
                    finding_id=f"CFG-OK-INFO-{uuid.uuid4().hex[:6]}",
                    category=DiagnosticCategory.CONFIGURATION_VIABILITY,
                    severity=DiagnosticSeverity.INFO,
                    title="Configuration Operates Within Normal Operating Envelope",
                    message=(
                        "No severe memory capacity pressure, context window violations, or excessive latency "
                        "anomalies were detected under the evaluated benchmark workload."
                    ),
                    confidence=DiagnosticConfidence.HIGH,
                    confidence_rationale="All diagnostic thresholds were satisfied without critical or warning triggers.",
                    supporting_observations=[],
                    assumptions=[],
                    limitations=[],
                    suggested_experiments=[],
                )
            )

        # 3. Generate Executive Summary
        crit_count = sum(1 for f in findings if f.severity == DiagnosticSeverity.CRITICAL)
        warn_count = sum(1 for f in findings if f.severity == DiagnosticSeverity.WARNING)
        info_count = sum(1 for f in findings if f.severity == DiagnosticSeverity.INFO)

        roofline_note = ""
        if roofline_analysis is not None:
            roofline_note = (
                f" Theoretical roofline classifies workload as {roofline_analysis.limiting_factor} "
                f"(ridge point: {roofline_analysis.ridge_point_flops_per_byte:.1f} FLOPs/B, "
                f"operational intensity: {roofline_analysis.operational_intensity:.1f} FLOPs/B)."
            )

        summary = (
            f"Diagnostic evaluation completed with {crit_count} CRITICAL, {warn_count} WARNING, and {info_count} INFO findings."
            f"{roofline_note}"
        )
        if crit_count > 0:
            crit_titles = [f.title for f in findings if f.severity == DiagnosticSeverity.CRITICAL]
            summary += f" Immediate attention required: {', '.join(crit_titles)}."

        return DiagnosticReport(
            report_id=f"diag-{uuid.uuid4().hex[:12]}",
            timestamp_utc=datetime.now(UTC),
            model_id=benchmark_suite.model_id,
            engine_type=benchmark_suite.engine_type,
            execution_mode=benchmark_suite.execution_mode,
            hardware_catalog_id=primary_gpu_spec.catalog_id if primary_gpu_spec else None,
            findings=findings,
            resource_estimate=resource_estimate,
            roofline_analysis=roofline_analysis,
            summary=summary,
        )
