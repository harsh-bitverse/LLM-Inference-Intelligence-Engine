"""Empirical dataset export foundation for benchmark results (Session 11).

Transforms raw benchmark suite requests and experiment manifests into structured,
versioned JSONL and CSV datasets with explicit provenance, missing-data indicators,
real-measurement filtering, and separate failure audit trails.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_ii_engine.benchmark.experiment import BenchmarkExperimentManifest, ExperimentResult
from llm_ii_engine.benchmark.models import BenchmarkSuiteResult, RequestBenchmarkResult
from llm_ii_engine.core.enums import ExecutionMode, TokenCountSource


class EmpiricalObservationRecord(BaseModel):
    """Normalized, flat observation record designed for machine-readable export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(description="Unique observation identifier")
    experiment_id: str = Field(description="Experiment manifest identifier")
    suite_id: str = Field(description="Benchmark suite execution ID")
    request_id: str = Field(description="Individual request execution ID")
    repetition_index: int = Field(ge=0, description="Repetition index within the experiment")
    is_warmup: bool = Field(description="True if request occurred during warmup phase")

    # Runtime & Model Identity
    model_id: str = Field(description="Canonical model identifier")
    model_digest: str | None = Field(default=None, description="Artifact hash or digest")
    runtime_name: str = Field(description="Serving runtime ('ollama', 'llamacpp', 'mock')")
    runtime_version: str | None = Field(default=None, description="Version of the serving engine")
    adapter_version: str = Field(description="Version of the engine adapter")

    # Hardware Environment
    hardware_id: str | None = Field(default=None, description="Hardware profile ID")
    device_name: str = Field(description="Execution device ('CPU', 'GPU-0', etc.)")
    cpu_model: str | None = Field(default=None, description="Host CPU model")
    host_ram_gb: float | None = Field(default=None, ge=0.0, description="Host RAM capacity in GB")
    gpu_vram_gb: float | None = Field(default=None, ge=0.0, description="GPU memory capacity in GB")

    # Workload & Token Counts
    prompt_tokens: int | None = Field(
        default=None, ge=0, description="Input prompt token count. None if unobserved/unknown."
    )
    prompt_token_count_source: str = Field(
        description="Source of prompt tokens (tokenizer, server_reported, requested, heuristic, unknown)"
    )
    requested_output_tokens: int | None = Field(
        default=None, ge=0, description="Upper generation limit requested"
    )
    actual_output_tokens: int = Field(ge=0, description="Actual output tokens generated")
    token_count_source: str = Field(
        description="Source of output tokens (token_events, server_reported, requested, unknown)"
    )

    # Physical Timing & Metric Measurements
    duration_ms: float = Field(ge=0.0, description="End-to-end request duration in ms")
    ttfc_ms: float | None = Field(
        default=None, ge=0.0, description="Time to first text chunk in ms"
    )
    ttft_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time to first discrete token event in ms. None for chunk streams.",
    )
    tpot_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time per output token in ms. None for chunk streams or <2 tokens.",
    )
    chunk_count: int = Field(ge=0, description="Number of text chunks received")
    chunk_latency_p50_ms: float | None = Field(
        default=None, ge=0.0, description="Median chunk inter-arrival latency in ms"
    )
    output_tokens_per_second: float | None = Field(
        default=None,
        ge=0.0,
        description="Output throughput in tok/s. None if token count is unknown.",
    )

    # Distinct Semantic Metrics (Session 16)
    client_e2e_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Client monotonic end-to-end duration in ms"
    )
    client_e2e_output_throughput_tok_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Client end-to-end output throughput (output tokens / client duration in s)",
    )
    server_total_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server reported total generation time in ms"
    )
    server_load_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server reported model loading time in ms"
    )
    server_prefill_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server reported prompt prefill duration in ms"
    )
    server_decode_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server reported decode evaluation duration in ms"
    )
    server_decode_throughput_tok_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Server decode throughput (output tokens / server decode duration in s) in tok/s",
    )

    # Server-Reported Telemetry (backward compatibility aliases)
    server_prompt_eval_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server prefill duration in ms"
    )
    server_eval_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Server decode duration in ms"
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_distinct_metrics(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Synchronize client e2e duration
            if data.get("client_e2e_duration_ms") is None and data.get("duration_ms") is not None:
                data["client_e2e_duration_ms"] = data.get("duration_ms")
            elif data.get("duration_ms") is None and data.get("client_e2e_duration_ms") is not None:
                data["duration_ms"] = data.get("client_e2e_duration_ms")

            # Synchronize client e2e throughput
            if data.get("client_e2e_output_throughput_tok_s") is None and data.get("output_tokens_per_second") is not None:
                data["client_e2e_output_throughput_tok_s"] = data.get("output_tokens_per_second")
            elif data.get("output_tokens_per_second") is None and data.get("client_e2e_output_throughput_tok_s") is not None:
                data["output_tokens_per_second"] = data.get("client_e2e_output_throughput_tok_s")

            # Synchronize server prefill duration
            if data.get("server_prefill_duration_ms") is None and data.get("server_prompt_eval_duration_ms") is not None:
                data["server_prefill_duration_ms"] = data.get("server_prompt_eval_duration_ms")
            elif data.get("server_prompt_eval_duration_ms") is None and data.get("server_prefill_duration_ms") is not None:
                data["server_prompt_eval_duration_ms"] = data.get("server_prefill_duration_ms")

            # Synchronize server decode duration
            if data.get("server_decode_duration_ms") is None and data.get("server_eval_duration_ms") is not None:
                data["server_decode_duration_ms"] = data.get("server_eval_duration_ms")
            elif data.get("server_eval_duration_ms") is None and data.get("server_decode_duration_ms") is not None:
                data["server_eval_duration_ms"] = data.get("server_decode_duration_ms")

            # Compute server decode throughput if tokens and decode duration present but throughput is None
            if data.get("server_decode_throughput_tok_s") is None:
                dec_ms = data.get("server_decode_duration_ms") or data.get("server_eval_duration_ms")
                act_tok = data.get("actual_output_tokens")
                if dec_ms and dec_ms > 0 and act_tok and act_tok > 0:
                    data["server_decode_throughput_tok_s"] = round(act_tok / (dec_ms / 1000.0), 2)
        return data

    # Memory Telemetry
    peak_vram_mb: float | None = Field(
        default=None, ge=0.0, description="Peak GPU VRAM allocated in MiB"
    )
    peak_process_rss_mb: float | None = Field(
        default=None, ge=0.0, description="Peak host process RSS in MiB"
    )

    # Cache State & Model Residency Telemetry (Session 15)
    model_residency_state: str | None = Field(
        default=None, description="Model memory residency state: cold, warm, or unknown"
    )
    cache_state: str | None = Field(
        default=None, description="Prompt KV cache state classification: cold_prefix, warm_prefix, partial_prefix, or unknown"
    )
    predecessor_request_id: str | None = Field(
        default=None, description="Request ID of immediately preceding request on the inference server"
    )
    server_prompt_eval_cached_count: int | None = Field(
        default=None, ge=0, description="Cached prompt tokens reused as reported by server"
    )
    prefix_relationship: str | None = Field(
        default=None, description="Prefix relationship to immediately preceding request: identical, shared_prefix, unrelated, or none"
    )

    # Execution Integrity Declarations
    execution_mode: str = Field(description="REAL vs SIMULATED")
    clock_source: str = Field(description="MONOTONIC_REAL vs SYNTHETIC_VIRTUAL")
    provenance: str = Field(description="MEASURED vs HEURISTIC")
    success: bool = Field(description="True if request completed without unhandled errors")
    error_message: str | None = Field(default=None, description="Failure details if success is False")

    # Missing Data & Environmental Disclosures
    missing_data_reasons: list[str] = Field(
        default_factory=list, description="Explicit disclosures explaining why metrics are None"
    )
    environmental_caveats: list[str] = Field(
        default_factory=list, description="Host environmental constraints or anomalies"
    )


class EmpiricalDatasetExporter:
    """Manages serialization, filtering, and schema documentation for empirical datasets."""

    @classmethod
    def record_from_benchmark_request(
        cls,
        req: RequestBenchmarkResult,
        suite: BenchmarkSuiteResult,
        manifest: BenchmarkExperimentManifest | None = None,
        repetition_index: int = 0,
    ) -> EmpiricalObservationRecord:
        """Convert a single RequestBenchmarkResult into an EmpiricalObservationRecord."""
        missing: list[str] = []

        if req.prompt_tokens is None:
            missing.append("prompt_tokens is null: no tokenizer or server-reported prompt count available")
        elif req.prompt_token_count_source == TokenCountSource.HEURISTIC_WORD_COUNT:
            missing.append("prompt_tokens derived from whitespace word splitting; not actual tokenizer count")

        if req.ttft_ms is None:
            if req.ttfc_ms is not None:
                missing.append("ttft_ms is null: stream emitted text chunks without 1:1 token timing guarantees; ttfc_ms recorded instead")
            else:
                missing.append("ttft_ms is null: no first-token or first-chunk timing recorded")

        if req.tpot_ms is None:
            if req.token_count_source != TokenCountSource.TOKEN_EVENTS:
                missing.append(f"tpot_ms is null: token_count_source is '{req.token_count_source.value}'; per-token ITL requires discrete TOKEN events")
            elif req.output_tokens < 2:
                missing.append(f"tpot_ms is null: requires >= 2 output tokens (got {req.output_tokens})")

        if req.output_tokens_per_second is None:
            missing.append("output_tokens_per_second is null: output token count is unknown or generation duration is 0")

        server_prompt_ms = None
        server_eval_ms = None
        server_total_ms = None
        server_load_ms = None
        server_decode_thru = None
        server_prompt_cached = None
        if req.server_stats:
            server_prompt_ms = req.server_stats.prompt_duration_ms
            server_eval_ms = req.server_stats.eval_duration_ms
            server_total_ms = req.server_stats.total_duration_ms
            server_load_ms = req.server_stats.load_duration_ms
            server_decode_thru = req.server_stats.eval_tokens_per_second
            server_prompt_cached = req.server_stats.prompt_eval_cached_count

        peak_vram = None
        peak_rss = None
        if suite.memory_watermark and suite.memory_watermark.is_telemetry_available:
            peak_vram = suite.memory_watermark.peak_gpu_vram_mb
            peak_rss = suite.memory_watermark.peak_process_rss_mb
        else:
            missing.append("memory telemetry is null: host hardware memory sampler unmetered or unavailable")

        env_caveats = list(manifest.environmental_caveats) if manifest else []

        return EmpiricalObservationRecord(
            record_id=f"obs-{req.request_id}",
            experiment_id=manifest.experiment_id if manifest else suite.suite_id,
            suite_id=suite.suite_id,
            request_id=req.request_id,
            repetition_index=repetition_index,
            is_warmup=req.is_warmup,
            model_id=manifest.model_id if manifest else suite.model_id,
            model_digest=manifest.model_digest if manifest else None,
            runtime_name=manifest.runtime_name if manifest else suite.engine_type.value,
            runtime_version=manifest.runtime_version if manifest else None,
            adapter_version=manifest.adapter_version if manifest else "1.0.0",
            hardware_id=manifest.hardware_profile_id if manifest else None,
            device_name=manifest.device_name if manifest else "CPU",
            cpu_model=manifest.cpu_model if manifest else None,
            host_ram_gb=manifest.host_ram_gb if manifest else None,
            gpu_vram_gb=manifest.gpu_vram_gb if manifest else None,
            prompt_tokens=req.prompt_tokens,
            prompt_token_count_source=req.prompt_token_count_source.value,
            requested_output_tokens=req.requested_output_tokens,
            actual_output_tokens=req.output_tokens,
            token_count_source=req.token_count_source.value,
            duration_ms=req.duration_ms,
            ttfc_ms=req.ttfc_ms,
            ttft_ms=req.ttft_ms,
            tpot_ms=req.tpot_ms,
            chunk_count=req.chunk_count,
            chunk_latency_p50_ms=req.chunk_latency_p50_ms,
            output_tokens_per_second=req.output_tokens_per_second,
            client_e2e_duration_ms=req.duration_ms,
            client_e2e_output_throughput_tok_s=req.output_tokens_per_second,
            server_total_duration_ms=server_total_ms,
            server_load_duration_ms=server_load_ms,
            server_prefill_duration_ms=server_prompt_ms,
            server_decode_duration_ms=server_eval_ms,
            server_decode_throughput_tok_s=server_decode_thru,
            server_prompt_eval_duration_ms=server_prompt_ms,
            server_eval_duration_ms=server_eval_ms,
            peak_vram_mb=peak_vram,
            peak_process_rss_mb=peak_rss,
            model_residency_state=(
                req.model_residency_state.value
                if getattr(req, "model_residency_state", None)
                else None
            ),
            cache_state=(
                req.cache_state.value
                if getattr(req, "cache_state", None)
                else None
            ),
            predecessor_request_id=getattr(req, "predecessor_request_id", None),
            server_prompt_eval_cached_count=server_prompt_cached,
            prefix_relationship=(
                req.prefix_relationship.value
                if getattr(req, "prefix_relationship", None)
                else None
            ),
            execution_mode=req.execution_mode.value,
            clock_source=req.clock_source.value,
            provenance=req.provenance.value,
            success=req.success,
            error_message=req.error_message,
            missing_data_reasons=missing,
            environmental_caveats=env_caveats,
        )

    @classmethod
    def from_experiment_result(
        cls,
        result: ExperimentResult,
    ) -> list[EmpiricalObservationRecord]:
        """Convert all requests from an ExperimentResult into EmpiricalObservationRecord list."""
        records: list[EmpiricalObservationRecord] = []
        manifest = result.manifest
        suite = result.suite_result

        # Map warmup requests (repetition 0)
        for req in result.warmup_requests:
            rec = cls.record_from_benchmark_request(
                req=req,
                suite=suite,
                manifest=manifest,
                repetition_index=0,
            )
            records.append(rec)

        # Map measured requests (repetition index inferred from request count or id)
        workload_count = max(1, len(manifest.workload_suite))
        for idx, req in enumerate(result.measured_requests):
            rep_idx = (idx // workload_count) + 1  # 1-based index for measured repetitions
            rec = cls.record_from_benchmark_request(
                req=req,
                suite=suite,
                manifest=manifest,
                repetition_index=rep_idx,
            )
            records.append(rec)

        return records

    @classmethod
    def export_to_jsonl(
        cls,
        records: list[EmpiricalObservationRecord],
        output_path: str | Path,
        filter_real_only: bool = True,
        include_failed: bool = False,
        exclude_warmup: bool = True,
    ) -> int:
        """Export observation records to JSON Lines format.

        Returns number of records written.
        """
        filtered = cls._filter_records(
            records=records,
            filter_real_only=filter_real_only,
            include_failed=include_failed,
            exclude_warmup=exclude_warmup,
        )

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "w", encoding="utf-8") as f:
            for rec in filtered:
                line = rec.model_dump_json()
                f.write(f"{line}\n")

        return len(filtered)

    @classmethod
    def export_to_csv(
        cls,
        records: list[EmpiricalObservationRecord],
        output_path: str | Path,
        filter_real_only: bool = True,
        include_failed: bool = False,
        exclude_warmup: bool = True,
    ) -> int:
        """Export observation records to CSV format.

        Returns number of records written.
        """
        filtered = cls._filter_records(
            records=records,
            filter_real_only=filter_real_only,
            include_failed=include_failed,
            exclude_warmup=exclude_warmup,
        )

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if not filtered:
            with open(target, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(EmpiricalObservationRecord.model_fields.keys()))
            return 0

        # Extract field names
        fieldnames = list(filtered[0].model_dump().keys())

        with open(target, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in filtered:
                row = rec.model_dump(mode="json")
                # Flatten lists for clean CSV formatting
                row["missing_data_reasons"] = "; ".join(row.get("missing_data_reasons", []))
                row["environmental_caveats"] = "; ".join(row.get("environmental_caveats", []))
                writer.writerow(row)

        return len(filtered)

    @classmethod
    def export_failed_runs_to_jsonl(
        cls,
        records: list[EmpiricalObservationRecord],
        output_path: str | Path,
    ) -> int:
        """Export failed or errored requests to a dedicated failure audit JSONL file."""
        failed = [r for r in records if not r.success]
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "w", encoding="utf-8") as f:
            for rec in failed:
                line = rec.model_dump_json()
                f.write(f"{line}\n")

        return len(failed)

    @classmethod
    def export_failed_runs_to_csv(
        cls,
        records: list[EmpiricalObservationRecord],
        output_path: str | Path,
    ) -> int:
        """Export failed or errored requests to a dedicated failure audit CSV file."""
        failed = [r for r in records if not r.success]
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if not failed:
            with open(target, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(EmpiricalObservationRecord.model_fields.keys()))
            return 0

        fieldnames = list(failed[0].model_dump().keys())
        with open(target, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in failed:
                row = rec.model_dump(mode="json")
                row["missing_data_reasons"] = "; ".join(row.get("missing_data_reasons", []))
                row["environmental_caveats"] = "; ".join(row.get("environmental_caveats", []))
                writer.writerow(row)

        return len(failed)

    @classmethod
    def _filter_records(
        cls,
        records: list[EmpiricalObservationRecord],
        filter_real_only: bool,
        include_failed: bool,
        exclude_warmup: bool,
    ) -> list[EmpiricalObservationRecord]:
        """Apply provenance and execution filters."""
        result = []
        for r in records:
            if filter_real_only and r.execution_mode != ExecutionMode.REAL.value:
                continue
            if not include_failed and not r.success:
                continue
            if exclude_warmup and r.is_warmup:
                continue
            result.append(r)
        return result

    @classmethod
    def generate_schema_documentation(cls) -> str:
        """Produce GitHub-flavored markdown describing the dataset schema."""
        return """# Empirical Benchmark Observation Dataset Schema (v1.0.0)

This schema defines the columns exported by the LLM Inference Intelligence Engine (IIE)
empirical dataset foundation.

## Field Dictionary

| Field | Type | Description | Nullable / Example |
| :--- | :--- | :--- | :--- |
| `record_id` | String | Unique observation identifier | No (`obs-meas-rep1-req0`) |
| `experiment_id` | String | BenchmarkExperimentManifest ID | No (`exp-3fa85f64d`) |
| `suite_id` | String | Execution suite ID | No (`suite-001`) |
| `request_id` | String | Individual request ID | No (`meas-rep1-req0`) |
| `repetition_index` | Integer | Iteration sequence index (0=warmup) | No (`1`) |
| `is_warmup` | Boolean | Warmup exclusion flag | No (`false`) |
| `model_id` | String | Model identifier | No (`qwen2.5:0.5b`) |
| `model_digest` | String | Artifact hash/digest | Yes (`sha256:7b5...`) |
| `runtime_name` | String | Serving runtime ('ollama', 'llamacpp') | No (`ollama`) |
| `runtime_version` | String | Runtime software version | Yes (`0.5.4`) |
| `adapter_version` | String | IIE adapter contract version | No (`1.0.0`) |
| `hardware_id` | String | Hardware catalog specification ID | Yes (`intel-uhd-770`) |
| `device_name` | String | Active computing device | No (`CPU`) |
| `cpu_model` | String | Processor model name | Yes (`Intel Core i7-14700`) |
| `host_ram_gb` | Float | Host total RAM in GB | Yes (`31.69`) |
| `gpu_vram_gb` | Float | Dedicated GPU VRAM in GB | Yes (null for CPU) |
| `prompt_tokens` | Integer | Actual prompt token count | Yes (null if unmeasured) |
| `prompt_token_count_source` | String | Provenance (`tokenizer`, `server_reported`, `heuristic_word_count`, `unknown`) | No |
| `requested_output_tokens` | Integer | Requested output limit | Yes (`48`) |
| `actual_output_tokens` | Integer | Actual decoded token count | No (`48`) |
| `token_count_source` | String | Provenance (`token_events`, `server_reported`, `unknown`) | No |
| `duration_ms` | Float | Client-observed wall time in ms | No (`450.2`) |
| `ttfc_ms` | Float | Time to first chunk in ms | Yes (`120.4`) |
| `ttft_ms` | Float | Verified discrete time to first token | Yes (null for chunk streams) |
| `tpot_ms` | Float | Verified discrete time per token | Yes (null for chunk streams) |
| `chunk_count` | Integer | Stream chunk count received | No (`12`) |
| `chunk_latency_p50_ms` | Float | Median chunk inter-arrival time in ms | Yes (`32.1`) |
| `output_tokens_per_second` | Float | Output throughput in tok/s | Yes (`24.8`) |
| `server_prompt_eval_duration_ms` | Float | Server-reported prefill time in ms | Yes (`85.2`) |
| `server_eval_duration_ms` | Float | Server-reported decode time in ms | Yes (`320.1`) |
| `peak_vram_mb` | Float | Measured peak GPU VRAM in MiB | Yes (null if unmetered) |
| `peak_process_rss_mb` | Float | Host resident memory peak in MiB | Yes (`245.8`) |
| `execution_mode` | String | `REAL` or `SIMULATED` | No (`REAL`) |
| `clock_source` | String | `MONOTONIC_REAL` or `SYNTHETIC_VIRTUAL` | No (`MONOTONIC_REAL`) |
| `provenance` | String | `MEASURED` or `HEURISTIC` | No (`MEASURED`) |
| `success` | Boolean | True if completed without error | No (`true`) |
| `error_message` | String | Failure error text if failed | Yes |
| `missing_data_reasons` | List[String] | Audit justifications for null fields | No (`[]`) |
| `environmental_caveats` | List[String] | Host execution caveats | No (`["CPU-only"]`) |
"""
