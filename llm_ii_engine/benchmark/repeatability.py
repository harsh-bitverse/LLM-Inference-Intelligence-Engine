"""Repeatability, variability, and measurement quality analysis (Session 11).

Evaluates run-to-run variability (CV), warmup cold-start penalties, client-observed
versus server-reported timing divergence, and output-token count stability across
repeated workload iterations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.benchmark.dataset_export import EmpiricalObservationRecord
from llm_ii_engine.benchmark.experiment import ExperimentResult


class MetricVariabilityStats(BaseModel):
    """Statistical summary of a repeated metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str = Field(description="Name of the metric evaluated")
    unit: str = Field(description="Physical unit of the metric")
    count: int = Field(ge=0, description="Number of valid samples")
    mean: float = Field(description="Sample mean")
    std_dev: float = Field(ge=0.0, description="Sample standard deviation")
    min_value: float = Field(description="Minimum observed value")
    max_value: float = Field(description="Maximum observed value")
    cv_percent: float = Field(
        ge=0.0,
        description="Coefficient of variation (std_dev / mean * 100%). Measures relative dispersion.",
    )


class WorkloadRepeatabilityStats(BaseModel):
    """Repeatability and variability metrics for a specific workload across repetitions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workload_index: int = Field(ge=0, description="Index of workload prompt within suite")
    prompt_tokens: int | None = Field(default=None, description="Prompt token count")
    requested_output_tokens: int | None = Field(default=None, description="Requested output limit")
    measured_repetitions_count: int = Field(ge=0, description="Number of measured iterations")

    # Metric Statistics
    duration_stats: MetricVariabilityStats | None = None
    ttfc_stats: MetricVariabilityStats | None = None
    throughput_stats: MetricVariabilityStats | None = None
    output_tokens_stats: MetricVariabilityStats | None = None

    # Warmup and Timing Discrepancies
    warmup_duration_ms: float | None = Field(
        default=None, description="Duration of initial unmeasured warmup request"
    )
    warmup_overhead_percent: float | None = Field(
        default=None,
        description="Cold-start penalty: ((warmup_duration - measured_mean) / measured_mean) * 100%",
    )
    client_server_divergence_ms: float | None = Field(
        default=None,
        description="Average client overhead: duration_ms - (server_prefill_ms + server_decode_ms)",
    )


class RepeatabilityReport(BaseModel):
    """Comprehensive measurement quality and repeatability audit report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Experiment identifier")
    model_id: str = Field(description="Model evaluated")
    runtime_name: str = Field(description="Runtime engine tested")
    total_measured_requests: int = Field(ge=0)
    total_warmup_requests: int = Field(ge=0)
    workload_stats: list[WorkloadRepeatabilityStats] = Field(default_factory=list)

    # Summary Aggregates
    average_duration_cv_percent: float | None = Field(
        default=None, description="Mean CV% of request duration across workloads"
    )
    average_throughput_cv_percent: float | None = Field(
        default=None, description="Mean CV% of output throughput across workloads"
    )
    average_warmup_penalty_percent: float | None = Field(
        default=None, description="Mean cold-start warmup overhead percentage"
    )
    average_client_server_divergence_ms: float | None = Field(
        default=None, description="Mean network, framing, and client loop overhead in ms"
    )
    memory_telemetry_available: bool = Field(
        default=False, description="Whether memory telemetry was collected"
    )
    is_statistically_validated: bool = Field(
        default=False,
        description="False for descriptive pilot experiments (<30 samples). True only for large controlled cohorts.",
    )
    disclaimers: list[str] = Field(
        default_factory=list, description="Explicit caveats and scope boundaries"
    )


class RepeatabilityAnalyzer:
    """Computes variability, warmup effects, and timing divergences from benchmark observations."""

    @classmethod
    def _compute_stats(cls, values: Sequence[float], name: str, unit: str) -> MetricVariabilityStats | None:
        """Compute sample mean, standard deviation, and CV%."""
        if not values:
            return None
        n = len(values)
        mean_val = sum(values) / n
        if n > 1:
            variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
            std_dev = math.sqrt(max(0.0, variance))
        else:
            std_dev = 0.0

        cv = (std_dev / mean_val * 100.0) if mean_val > 0 else 0.0
        return MetricVariabilityStats(
            metric_name=name,
            unit=unit,
            count=n,
            mean=round(mean_val, 2),
            std_dev=round(std_dev, 2),
            min_value=round(min(values), 2),
            max_value=round(max(values), 2),
            cv_percent=round(cv, 2),
        )

    @classmethod
    def analyze_experiment(cls, result: ExperimentResult) -> RepeatabilityReport:
        """Compute repeatability report from an ExperimentResult."""
        from llm_ii_engine.benchmark.dataset_export import EmpiricalDatasetExporter

        records = EmpiricalDatasetExporter.from_experiment_result(result)
        return cls.analyze_observations(
            records=records,
            experiment_id=result.manifest.experiment_id,
            model_id=result.manifest.model_id,
            runtime_name=result.manifest.runtime_name,
            workload_count=len(result.manifest.workload_suite),
            memory_telemetry_available=(
                result.suite_result.memory_watermark.is_telemetry_available
                if result.suite_result.memory_watermark
                else False
            ),
        )

    @classmethod
    def analyze_observations(
        cls,
        records: list[EmpiricalObservationRecord],
        experiment_id: str | None = None,
        model_id: str | None = None,
        runtime_name: str | None = None,
        workload_count: int | None = None,
        memory_telemetry_available: bool = False,
    ) -> RepeatabilityReport:
        """Analyze a collection of observation records partitioned by workload."""
        if not records:
            return RepeatabilityReport(
                experiment_id=experiment_id or "unknown",
                model_id=model_id or "unknown",
                runtime_name=runtime_name or "unknown",
                total_measured_requests=0,
                total_warmup_requests=0,
                disclaimers=["No observations provided for repeatability analysis."],
            )

        exp_id = experiment_id or records[0].experiment_id
        mod_id = model_id or records[0].model_id
        rt_name = runtime_name or records[0].runtime_name

        warmup_records = [r for r in records if r.is_warmup]
        measured_records = [r for r in records if not r.is_warmup and r.success]

        # Determine effective workload count if not supplied
        if workload_count is None:
            # Group by unique prompt_tokens and requested_output_tokens
            unique_workloads = {(r.prompt_tokens, r.requested_output_tokens) for r in measured_records}
            w_count = max(1, len(unique_workloads))
        else:
            w_count = max(1, workload_count)

        # Partition measured records by workload slot
        workload_stats_list: list[WorkloadRepeatabilityStats] = []
        for w_idx in range(w_count):
            # Select records corresponding to this workload index
            w_measured = [r for idx, r in enumerate(measured_records) if (idx % w_count) == w_idx]
            w_warmup = [r for idx, r in enumerate(warmup_records) if (idx % w_count) == w_idx]

            if not w_measured:
                continue

            prompt_toks = w_measured[0].prompt_tokens
            req_out_toks = w_measured[0].requested_output_tokens

            # Compute metric variability
            durations = [r.duration_ms for r in w_measured]
            ttfcs = [r.ttfc_ms for r in w_measured if r.ttfc_ms is not None]
            throughputs = [
                r.output_tokens_per_second
                for r in w_measured
                if r.output_tokens_per_second is not None
            ]
            out_toks = [float(r.actual_output_tokens) for r in w_measured]

            dur_stats = cls._compute_stats(durations, "duration_ms", "ms")
            ttfc_stats = cls._compute_stats(ttfcs, "ttfc_ms", "ms")
            thru_stats = cls._compute_stats(throughputs, "output_tokens_per_second", "tok/s")
            tok_stats = cls._compute_stats(out_toks, "actual_output_tokens", "tokens")

            # Warmup comparison
            warmup_dur = w_warmup[0].duration_ms if (w_warmup and w_warmup[0].success) else None
            warmup_penalty = None
            if warmup_dur is not None and dur_stats and dur_stats.mean > 0:
                warmup_penalty = round(
                    ((warmup_dur - dur_stats.mean) / dur_stats.mean) * 100.0, 2
                )

            # Client vs Server timing divergence
            divergences: list[float] = []
            for r in w_measured:
                if (
                    r.server_prompt_eval_duration_ms is not None
                    and r.server_eval_duration_ms is not None
                ):
                    server_total = r.server_prompt_eval_duration_ms + r.server_eval_duration_ms
                    divergences.append(max(0.0, r.duration_ms - server_total))
            avg_divergence = (sum(divergences) / len(divergences)) if divergences else None

            workload_stats_list.append(
                WorkloadRepeatabilityStats(
                    workload_index=w_idx,
                    prompt_tokens=prompt_toks,
                    requested_output_tokens=req_out_toks,
                    measured_repetitions_count=len(w_measured),
                    duration_stats=dur_stats,
                    ttfc_stats=ttfc_stats,
                    throughput_stats=thru_stats,
                    output_tokens_stats=tok_stats,
                    warmup_duration_ms=warmup_dur,
                    warmup_overhead_percent=warmup_penalty,
                    client_server_divergence_ms=(
                        round(avg_divergence, 2) if avg_divergence is not None else None
                    ),
                )
            )

        # Summary aggregates
        all_dur_cvs = [
            ws.duration_stats.cv_percent
            for ws in workload_stats_list
            if ws.duration_stats and ws.duration_stats.count > 1
        ]
        all_thru_cvs = [
            ws.throughput_stats.cv_percent
            for ws in workload_stats_list
            if ws.throughput_stats and ws.throughput_stats.count > 1
        ]
        all_warmup_pens = [
            ws.warmup_overhead_percent
            for ws in workload_stats_list
            if ws.warmup_overhead_percent is not None
        ]
        all_divergences = [
            ws.client_server_divergence_ms
            for ws in workload_stats_list
            if ws.client_server_divergence_ms is not None
        ]

        avg_dur_cv = (sum(all_dur_cvs) / len(all_dur_cvs)) if all_dur_cvs else None
        avg_thru_cv = (sum(all_thru_cvs) / len(all_thru_cvs)) if all_thru_cvs else None
        avg_warmup_pen = (sum(all_warmup_pens) / len(all_warmup_pens)) if all_warmup_pens else None
        avg_divergence_total = (
            (sum(all_divergences) / len(all_divergences)) if all_divergences else None
        )

        disclaimers = [
            "Descriptive pilot dataset with limited sample repetitions per prompt.",
            "Results reflect local workstation execution and cannot be generalized to multi-socket servers or accelerators.",
            "Operating system scheduling jitter and background process activity may contribute to dispersion.",
        ]

        is_validated = len(measured_records) >= 30 and (avg_dur_cv is not None and avg_dur_cv < 10.0)

        return RepeatabilityReport(
            experiment_id=exp_id,
            model_id=mod_id,
            runtime_name=rt_name,
            total_measured_requests=len(measured_records),
            total_warmup_requests=len(warmup_records),
            workload_stats=workload_stats_list,
            average_duration_cv_percent=round(avg_dur_cv, 2) if avg_dur_cv is not None else None,
            average_throughput_cv_percent=(
                round(avg_thru_cv, 2) if avg_thru_cv is not None else None
            ),
            average_warmup_penalty_percent=(
                round(avg_warmup_pen, 2) if avg_warmup_pen is not None else None
            ),
            average_client_server_divergence_ms=(
                round(avg_divergence_total, 2) if avg_divergence_total is not None else None
            ),
            memory_telemetry_available=memory_telemetry_available,
            is_statistically_validated=is_validated,
            disclaimers=disclaimers,
        )
