"""Mixed-workload queueing models, statistical moment calculus, and discrete-event simulation.

Extends the single-workload capacity planner to support heterogeneous traffic mixtures,
computing mixed service moments (E[S_mix], E[S_mix^2], Var[S_mix], Cv_mix), M/G/1 Pollaczek-Khinchine
mean queue delays, and deterministic discrete-event simulation for accurate tail-latency forecasting.
"""

import math
import random
import statistics
from typing import Any

from llm_ii_engine.capacity.mixed_workload_protocol import (
    MixedServiceMoments,
    MixedWorkloadItem,
    MixedWorkloadProfile,
    WorkloadClass,
    WorkloadProfileType,
)
from llm_ii_engine.capacity.protocol import ServiceTimeDistribution

# =========================================================================
# Predefined Versioned Workload Profiles (Phase 3)
# =========================================================================

PROFILE_A_INTERACTIVE_DOMINANT = MixedWorkloadProfile(
    profile_id="profile-v1-interactive-dominant",
    name="Profile A: Interactive Dominant",
    profile_type=WorkloadProfileType.PROFILE_A_INTERACTIVE_DOMINANT,
    description="80% short interactive requests, 15% structured JSON extraction, 5% long-form generation.",
    items=[
        MixedWorkloadItem(
            workload_class=WorkloadClass.INTERACTIVE_QA,
            weight=0.80,
            prompt_tokens=20,
            expected_output_tokens=16,
            latency_sla_ms=500.0,
        ),
        MixedWorkloadItem(
            workload_class=WorkloadClass.STRUCTURED_JSON,
            weight=0.15,
            prompt_tokens=70,
            expected_output_tokens=96,
            latency_sla_ms=3000.0,
        ),
        MixedWorkloadItem(
            workload_class=WorkloadClass.LONG_FORM_GENERATION,
            weight=0.05,
            prompt_tokens=30,
            expected_output_tokens=160,
            latency_sla_ms=8000.0,
        ),
    ],
)

PROFILE_B_BALANCED = MixedWorkloadProfile(
    profile_id="profile-v1-balanced",
    name="Profile B: Balanced Mixture",
    profile_type=WorkloadProfileType.PROFILE_B_BALANCED,
    description="50% short interactive, 30% structured JSON extraction, 20% long-form generation.",
    items=[
        MixedWorkloadItem(
            workload_class=WorkloadClass.INTERACTIVE_QA,
            weight=0.50,
            prompt_tokens=20,
            expected_output_tokens=16,
            latency_sla_ms=500.0,
        ),
        MixedWorkloadItem(
            workload_class=WorkloadClass.STRUCTURED_JSON,
            weight=0.30,
            prompt_tokens=70,
            expected_output_tokens=96,
            latency_sla_ms=3000.0,
        ),
        MixedWorkloadItem(
            workload_class=WorkloadClass.LONG_FORM_GENERATION,
            weight=0.20,
            prompt_tokens=30,
            expected_output_tokens=160,
            latency_sla_ms=8000.0,
        ),
    ],
)

PROFILE_C_GENERATION_HEAVY = MixedWorkloadProfile(
    profile_id="profile-v1-generation-heavy",
    name="Profile C: Generation Heavy",
    profile_type=WorkloadProfileType.PROFILE_C_GENERATION_HEAVY,
    description="20% short interactive, 30% structured extraction, 50% sustained long-form generation.",
    items=[
        MixedWorkloadItem(
            workload_class=WorkloadClass.INTERACTIVE_QA,
            weight=0.20,
            prompt_tokens=20,
            expected_output_tokens=16,
            latency_sla_ms=500.0,
        ),
        MixedWorkloadItem(
            workload_class=WorkloadClass.STRUCTURED_JSON,
            weight=0.30,
            prompt_tokens=70,
            expected_output_tokens=96,
            latency_sla_ms=3000.0,
        ),
        MixedWorkloadItem(
            workload_class=WorkloadClass.LONG_FORM_GENERATION,
            weight=0.50,
            prompt_tokens=30,
            expected_output_tokens=160,
            latency_sla_ms=8000.0,
        ),
    ],
)

PROFILE_D_BURSTY = MixedWorkloadProfile(
    profile_id="profile-v1-bursty",
    name="Profile D: Bursty Traffic Mixture",
    profile_type=WorkloadProfileType.PROFILE_D_BURSTY,
    description="Alternating idle periods with bursts of short interactive requests arriving during long generations.",
    items=[
        MixedWorkloadItem(
            workload_class=WorkloadClass.INTERACTIVE_QA,
            weight=0.75,
            prompt_tokens=20,
            expected_output_tokens=16,
            latency_sla_ms=500.0,
        ),
        MixedWorkloadItem(
            workload_class=WorkloadClass.LONG_FORM_GENERATION,
            weight=0.25,
            prompt_tokens=30,
            expected_output_tokens=160,
            latency_sla_ms=8000.0,
        ),
    ],
    burst_idle_interval_s=2.0,
    burst_batch_size=4,
)

STANDARD_PROFILES: dict[WorkloadProfileType, MixedWorkloadProfile] = {
    WorkloadProfileType.PROFILE_A_INTERACTIVE_DOMINANT: PROFILE_A_INTERACTIVE_DOMINANT,
    WorkloadProfileType.PROFILE_B_BALANCED: PROFILE_B_BALANCED,
    WorkloadProfileType.PROFILE_C_GENERATION_HEAVY: PROFILE_C_GENERATION_HEAVY,
    WorkloadProfileType.PROFILE_D_BURSTY: PROFILE_D_BURSTY,
}


class MixedQueueingCalculator:
    """Calculates multi-class service moments and theoretical M/G/1 queue delays."""

    @staticmethod
    def calculate_mixed_moments(
        profile: MixedWorkloadProfile,
        per_class_distributions: dict[WorkloadClass, ServiceTimeDistribution],
    ) -> MixedServiceMoments:
        """Calculate first moment E[S_mix], second moment E[S_mix^2], and mixed variance.

        Equations:
            E[S_mix]   = sum(p_i * E[S_i])
            E[S_i^2]   = Var[S_i] + (E[S_i])^2
            E[S_mix^2] = sum(p_i * E[S_i^2])
            Var[S_mix] = E[S_mix^2] - (E[S_mix])^2
            Cv_mix     = sqrt(Var[S_mix]) / E[S_mix]
        """
        e_s_mix = 0.0
        e_s2_mix = 0.0
        per_class_info: dict[str, dict[str, float]] = {}

        total_weight = sum(item.weight for item in profile.items)
        if total_weight <= 0.0:
            raise ValueError("Profile item weights must sum to > 0")

        for item in profile.items:
            norm_weight = item.weight / total_weight
            dist = per_class_distributions.get(item.workload_class)
            if not dist:
                # Default fallback calibrated from Session 25
                mean_s = 500.0
                var_s = 2500.0
            else:
                mean_s = dist.mean_service_ms
                var_s = dist.variance_service_ms

            e_s2_i = var_s + (mean_s ** 2)
            e_s_mix += norm_weight * mean_s
            e_s2_mix += norm_weight * e_s2_i

            per_class_info[item.workload_class.value] = {
                "weight": round(norm_weight, 3),
                "mean_service_ms": round(mean_s, 2),
                "variance_service_ms2": round(var_s, 2),
                "second_moment_ms2": round(e_s2_i, 2),
            }

        variance_mix = max(0.0, e_s2_mix - (e_s_mix ** 2))
        std_mix = math.sqrt(variance_mix)
        cv_mix = (std_mix / e_s_mix) if e_s_mix > 0.0 else 0.0

        return MixedServiceMoments(
            mean_service_ms=round(e_s_mix, 2),
            second_moment_ms2=round(e_s2_mix, 2),
            variance_service_ms2=round(variance_mix, 2),
            std_service_ms=round(std_mix, 2),
            coefficient_of_variation=round(cv_mix, 3),
            per_class_moments=per_class_info,
        )

    @classmethod
    def estimate_theoretical_mean_queue_delay(
        cls,
        arrival_rate_rps: float,
        moments: MixedServiceMoments,
    ) -> tuple[float, float, bool]:
        """Estimate theoretical M/G/1 Pollaczek-Khinchine mean waiting time in queue.

        Formula:
            rho = lambda * E[S_mix] (in seconds)
            Wq  = lambda * E[S_mix^2] / (2 * (1 - rho))
        Returns:
            (wq_ms, rho, is_stable)
        """
        mean_s_sec = moments.mean_service_ms / 1000.0
        second_moment_sec2 = moments.second_moment_ms2 / 1e6
        rho = arrival_rate_rps * mean_s_sec

        if rho < 0.95:
            # Stable queue
            wq_sec = (arrival_rate_rps * second_moment_sec2) / (2.0 * max(0.01, 1.0 - rho))
            return round(wq_sec * 1000.0, 2), round(rho, 3), True
        else:
            # Overload regime: queue diverges; approximate bounded window wait
            excess_rate = max(0.0, arrival_rate_rps - (1.0 / max(0.001, mean_s_sec)))
            wq_ms = (moments.mean_service_ms * 3.0) + (excess_rate * 10.0 * moments.mean_service_ms)
            return round(wq_ms, 2), round(rho, 3), False


class MixedWorkloadSimulator:
    """Deterministic discrete-event simulator for mixed-workload FCFS single-slot queues."""

    @classmethod
    def simulate(
        cls,
        profile: MixedWorkloadProfile,
        arrival_rate_rps: float,
        per_class_distributions: dict[WorkloadClass, ServiceTimeDistribution],
        request_count: int = 150,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Simulate single-slot FCFS queueing to calculate per-class tail latencies.

        Uses deterministic random seed and finite workload horizon.
        """
        rng = random.Random(seed)
        items = profile.items
        weights = [it.weight for it in items]

        # Generate inter-arrival times and workload assignments
        current_time_s = 0.0
        events: list[dict[str, Any]] = []

        for i in range(request_count):
            if profile.profile_type == WorkloadProfileType.PROFILE_D_BURSTY:
                # Bursty pattern: 4 requests in rapid succession, followed by idle pause
                if (i % 4) == 0 and i > 0:
                    current_time_s += profile.burst_idle_interval_s or 1.5
                else:
                    current_time_s += rng.expovariate(max(0.1, arrival_rate_rps * 3.0))
            else:
                inter_arr = rng.expovariate(max(0.01, arrival_rate_rps))
                current_time_s += inter_arr

            chosen_item = rng.choices(items, weights=weights, k=1)[0]
            dist = per_class_distributions.get(chosen_item.workload_class)
            mean_s = dist.mean_service_ms if dist else 500.0
            std_s = dist.std_service_ms if dist else 50.0

            # Sample service time from truncated normal distribution matching empirical moments
            sampled_s = max(10.0, rng.gauss(mean_s, max(1.0, std_s)))

            events.append({
                "index": i,
                "arrival_s": current_time_s,
                "workload_class": chosen_item.workload_class,
                "service_time_ms": sampled_s,
                "latency_sla_ms": chosen_item.latency_sla_ms,
            })

        # Run single-server FCFS queue simulation
        server_free_at_s = 0.0
        results: list[dict[str, Any]] = []

        for ev in events:
            arr_s = ev["arrival_s"]
            svc_s = ev["service_time_ms"] / 1000.0

            start_s = max(arr_s, server_free_at_s)
            finish_s = start_s + svc_s
            server_free_at_s = finish_s

            queue_wait_ms = (start_s - arr_s) * 1000.0
            total_turnaround_ms = (finish_s - arr_s) * 1000.0
            sla_met = total_turnaround_ms <= ev["latency_sla_ms"]

            results.append({
                "index": ev["index"],
                "workload_class": ev["workload_class"],
                "queue_wait_ms": queue_wait_ms,
                "total_turnaround_ms": total_turnaround_ms,
                "sla_met": sla_met,
            })

        # Aggregate statistics
        all_e2es = [r["total_turnaround_ms"] for r in results]
        all_waits = [r["queue_wait_ms"] for r in results]

        sorted_e2es = sorted(all_e2es)
        n = len(sorted_e2es)
        overall_p50 = statistics.median(sorted_e2es)
        overall_p90 = sorted_e2es[min(int(n * 0.90), n - 1)]
        overall_p95 = sorted_e2es[min(int(n * 0.95), n - 1)]
        overall_sla_met_pct = round((sum(1 for r in results if r["sla_met"]) / n) * 100.0, 1)

        per_class_summary: dict[str, dict[str, float]] = {}
        for item in items:
            cls_name = item.workload_class.value
            cls_results = [r for r in results if r["workload_class"] == item.workload_class]
            if cls_results:
                cls_e2es = sorted([r["total_turnaround_ms"] for r in cls_results])
                cn = len(cls_e2es)
                cp50 = statistics.median(cls_e2es)
                cp95 = cls_e2es[min(int(cn * 0.95), cn - 1)]
                c_sla_pct = round((sum(1 for r in cls_results if r["sla_met"]) / cn) * 100.0, 1)
                c_mean_wait = round(statistics.mean([r["queue_wait_ms"] for r in cls_results]), 1)
                per_class_summary[cls_name] = {
                    "count": cn,
                    "p50_e2e_ms": round(cp50, 1),
                    "p95_e2e_ms": round(cp95, 1),
                    "mean_wait_ms": c_mean_wait,
                    "sla_attainment_pct": c_sla_pct,
                }

        return {
            "request_count": n,
            "overall_p50_e2e_ms": round(overall_p50, 1),
            "overall_p90_e2e_ms": round(overall_p90, 1),
            "overall_p95_e2e_ms": round(overall_p95, 1),
            "mean_queue_delay_ms": round(statistics.mean(all_waits), 1),
            "overall_sla_attainment_pct": overall_sla_met_pct,
            "per_class_summary": per_class_summary,
        }
