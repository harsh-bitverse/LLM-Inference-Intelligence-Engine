"""Queueing models and capacity estimation engine for Session 25.

Implements M/M/1, M/G/1 (Pollaczek-Khinchine), and empirical queueing models to forecast
queueing delay, system response times, server utilization, and saturation knees.
"""

import statistics

from llm_ii_engine.capacity.protocol import (
    CapacityCurvePoint,
    CapacityEvidenceTier,
    QueueingModelType,
    ServiceTimeDistribution,
    WorkloadCapacityReport,
)


class QueueingCalculator:
    """Analytical and empirical queueing approximations for single-server inference."""

    @staticmethod
    def calculate_distribution(
        service_times_ms: list[float],
        prefill_times_ms: list[float] | None = None,
        decode_times_ms: list[float] | None = None,
        output_tokens: list[int] | None = None,
    ) -> ServiceTimeDistribution:
        """Compute statistical properties of empirical service times."""
        if not service_times_ms:
            raise ValueError("service_times_ms must contain at least one observation")

        n = len(service_times_ms)
        mean_s = statistics.mean(service_times_ms)
        variance_s = statistics.variance(service_times_ms) if n > 1 else 0.0
        std_s = statistics.stdev(service_times_ms) if n > 1 else 0.0
        cv = (std_s / mean_s) if mean_s > 0 else 0.0

        sorted_s = sorted(service_times_ms)
        median_s = statistics.median(sorted_s)
        p90_idx = min(int(n * 0.90), n - 1)
        p95_idx = min(int(n * 0.95), n - 1)
        p90_s = sorted_s[p90_idx]
        p95_s = sorted_s[p95_idx]

        mean_prefill = statistics.mean(prefill_times_ms) if prefill_times_ms else (mean_s * 0.15)
        mean_decode = statistics.mean(decode_times_ms) if decode_times_ms else (mean_s * 0.85)
        mean_tokens = statistics.mean(output_tokens) if output_tokens else 50.0

        return ServiceTimeDistribution(
            sample_count=n,
            mean_service_ms=round(mean_s, 2),
            variance_service_ms=round(variance_s, 2),
            std_service_ms=round(std_s, 2),
            coefficient_of_variation=round(cv, 3),
            median_service_ms=round(median_s, 2),
            p90_service_ms=round(p90_s, 2),
            p95_service_ms=round(p95_s, 2),
            min_service_ms=round(sorted_s[0], 2),
            max_service_ms=round(sorted_s[-1], 2),
            mean_prefill_ms=round(mean_prefill, 2),
            mean_decode_ms=round(mean_decode, 2),
            mean_output_tokens=round(mean_tokens, 1),
        )

    @classmethod
    def estimate_queue_metrics(
        cls,
        arrival_rate_rps: float,
        service_dist: ServiceTimeDistribution,
        model_type: QueueingModelType = QueueingModelType.M_G_1_APPROX,
        latency_sla_ms: float | None = None,
    ) -> CapacityCurvePoint:
        """Forecast queue wait time, e2e latency, utilization, and SLA compliance at offered rate."""
        mean_s_sec = service_dist.mean_service_ms / 1000.0
        mu = (1.0 / mean_s_sec) if mean_s_sec > 0 else 1.0
        rho = arrival_rate_rps * mean_s_sec  # utilization

        # Effective throughput in single-server queue cannot exceed service capacity mu
        completed_rps = min(arrival_rate_rps, mu)

        if rho < 0.98:
            # Stable queue
            if model_type == QueueingModelType.M_G_1_APPROX:
                # Pollaczek-Khinchine formula: Wq = rho * E[S] * (1 + Cv^2) / (2 * (1 - rho))
                cv_sq = service_dist.coefficient_of_variation ** 2
                wq_sec = (rho * mean_s_sec * (1.0 + cv_sq)) / (2.0 * max(0.01, 1.0 - rho))
            elif model_type == QueueingModelType.M_M_1_APPROX:
                # M/M/1 formula: Wq = rho / (mu * (1 - rho))
                wq_sec = rho / (mu * max(0.01, 1.0 - rho))
            else:
                # Empirical interpolation
                cv_sq = service_dist.coefficient_of_variation ** 2
                wq_sec = (rho * mean_s_sec * (1.0 + cv_sq)) / (2.0 * max(0.01, 1.0 - rho))

            queue_wait_ms = wq_sec * 1000.0
            median_e2e_ms = service_dist.median_service_ms + queue_wait_ms
            p90_e2e_ms = service_dist.p90_service_ms + (queue_wait_ms * 1.3)
            p95_e2e_ms = service_dist.p95_service_ms + (queue_wait_ms * 1.6)
            buildup = False
        else:
            # Overloaded / unstable regime (rho >= 0.98)
            # Queue grows monotonically; model a bounded 10s arrival window
            overload_excess = max(0.0, arrival_rate_rps - mu)
            queue_wait_ms = (overload_excess * 10.0 * service_dist.mean_service_ms) + (
                service_dist.mean_service_ms * 2.0
            )
            median_e2e_ms = service_dist.median_service_ms + queue_wait_ms
            p90_e2e_ms = service_dist.p90_service_ms + (queue_wait_ms * 1.5)
            p95_e2e_ms = service_dist.p95_service_ms + (queue_wait_ms * 2.0)
            buildup = True

        # SLA compliance calculation
        sla_pct = 100.0
        if latency_sla_ms:
            if p95_e2e_ms <= latency_sla_ms:
                sla_pct = 95.0 + min(5.0, (latency_sla_ms - p95_e2e_ms) / 100.0)
            elif median_e2e_ms <= latency_sla_ms:
                # Intermediate between median and p95
                ratio = (latency_sla_ms - median_e2e_ms) / max(1.0, (p95_e2e_ms - median_e2e_ms))
                sla_pct = 50.0 + (ratio * 45.0)
            else:
                # Below median
                ratio = max(0.0, latency_sla_ms / max(1.0, median_e2e_ms))
                sla_pct = max(0.0, ratio * 50.0)

        return CapacityCurvePoint(
            offered_rps=round(arrival_rate_rps, 3),
            completed_rps=round(completed_rps, 3),
            estimated_utilization=round(rho, 3),
            median_e2e_ms=round(median_e2e_ms, 1),
            p90_e2e_ms=round(p90_e2e_ms, 1),
            p95_e2e_ms=round(p95_e2e_ms, 1),
            mean_queue_delay_ms=round(queue_wait_ms, 1),
            sla_compliance_pct=round(min(100.0, max(0.0, sla_pct)), 1),
            queue_buildup_detected=buildup,
            failure_count=1 if (buildup and median_e2e_ms > 30000.0) else 0,
        )


class CapacityPlanner:
    """Computes workload-specific capacity limits and serving capacity curves."""

    # Curated baseline service times calibrated from Session 24 measurements (8 threads)
    DEFAULT_SERVICE_BASELINES: dict[str, dict[str, float]] = {
        "qwen2.5:0.5b": {
            "workload_v1_interactive_qa": 250.0,
            "workload_v1_structured_json": 950.0,
            "workload_v1_long_form_generation": 2300.0,
            "workload_v1_long_context_retrieval": 750.0,
        },
        "llama3.2:1b": {
            "workload_v1_interactive_qa": 450.0,
            "workload_v1_structured_json": 2350.0,
            "workload_v1_long_form_generation": 6500.0,
            "workload_v1_long_context_retrieval": 1600.0,
        },
        "qwen2.5:1.5b": {
            "workload_v1_interactive_qa": 400.0,
            "workload_v1_structured_json": 2250.0,
            "workload_v1_long_form_generation": 6200.0,
            "workload_v1_long_context_retrieval": 1500.0,
        },
    }

    @classmethod
    def generate_capacity_report(
        cls,
        model_id: str,
        num_thread: int,
        workload_id: str,
        service_distribution: ServiceTimeDistribution,
        latency_sla_ms: float | None = None,
        model_type: QueueingModelType = QueueingModelType.M_G_1_APPROX,
        evidence_tier: CapacityEvidenceTier = CapacityEvidenceTier.DIRECT_MEASURED,
    ) -> WorkloadCapacityReport:
        """Construct full capacity plan with capacity curve points up to and beyond saturation."""
        mean_s_sec = service_distribution.mean_service_ms / 1000.0
        mu = 1.0 / mean_s_sec if mean_s_sec > 0 else 1.0

        max_sustainable_rps = round(mu, 3)
        max_sustainable_rpm = round(mu * 60.0, 1)

        # Saturation knee is typically at rho = 0.80 to 0.85
        saturation_knee_rps = round(mu * 0.82, 3)
        # Safe operating load target (~75% of knee for bounded tail latency)
        safe_operating_rps = round(saturation_knee_rps * 0.75, 3)

        # Generate capacity curve across rates: [0.1, 0.25, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5] of mu
        rate_multipliers = [0.1, 0.25, 0.5, 0.7, 0.82, 0.95, 1.1, 1.3]
        curve: list[CapacityCurvePoint] = []

        for mult in rate_multipliers:
            arr_rate = mu * mult
            point = QueueingCalculator.estimate_queue_metrics(
                arrival_rate_rps=arr_rate,
                service_dist=service_distribution,
                model_type=model_type,
                latency_sla_ms=latency_sla_ms,
            )
            curve.append(point)

        assumptions = [
            f"Evaluated on Intel Core i7-14700 workstation under Ollama v0.40.11 with {num_thread} CPU threads.",
            "Assumes single effective execution slot (slot id 0); concurrent requests are serialized in server queue.",
            "Arrivals modeled via M/G/1 Pollaczek-Khinchine approximation accounting for service variance.",
            f"Mean service time E[S] = {service_distribution.mean_service_ms} ms (Cv = {service_distribution.coefficient_of_variation}).",
            "Overload states (rho >= 1.0) cause linear backlog buildup and rapid SLA degradation.",
        ]

        return WorkloadCapacityReport(
            report_id=f"cap-rep-{model_id.replace(':', '-')}-{workload_id}-{num_thread}th",
            model_id=model_id,
            num_thread=num_thread,
            workload_id=workload_id,
            service_distribution=service_distribution,
            saturation_knee_rps=saturation_knee_rps,
            max_sustainable_rps=max_sustainable_rps,
            max_sustainable_rpm=max_sustainable_rpm,
            safe_operating_rps=safe_operating_rps,
            capacity_curve=curve,
            evidence_tier=evidence_tier,
            queueing_model_type=model_type,
            assumptions=assumptions,
        )
