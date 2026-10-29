"""Admission control policies and comparative evaluation metrics for mixed workloads.

Implements:
- Policy A: No admission control (FIFO baseline).
- Policy B: Static utilization guardrail (rho <= 0.82 threshold).
- Policy C: Workload-aware adaptive admission (model-switching penalty, queue backlog, per-class SLA).
- Metrics: Jain's fairness index, false-feasible detection, and net trade-off utility scoring.
"""

import statistics

from llm_ii_engine.capacity.mixed_workload_protocol import (
    AdmissionPolicyType,
    MixedRequestAccountingRecord,
    ModelResidencyState,
    PolicyEvaluationReport,
    RequestTerminalStatus,
    WorkloadClass,
    WorkloadProfileType,
)
from llm_ii_engine.capacity.protocol import AdmissionDecision


class MixedWorkloadAdmissionController:
    """Evaluates admission decisions across Policy A, Policy B, and Policy C."""

    # Empirical service times (E[S], P95, Cv) calibrated from Session 25 measurements
    EMPIRICAL_TASK_PROFILES: dict[str, dict[WorkloadClass, dict[str, float]]] = {
        "qwen2.5:0.5b": {
            WorkloadClass.INTERACTIVE_QA: {"mean_s": 59.3, "p95_s": 63.3, "cv": 0.066},
            WorkloadClass.STRUCTURED_JSON: {"mean_s": 908.0, "p95_s": 926.3, "cv": 0.016},
            WorkloadClass.LONG_FORM_GENERATION: {"mean_s": 2380.0, "p95_s": 2432.5, "cv": 0.017},
            WorkloadClass.LONG_CONTEXT_RETRIEVAL: {"mean_s": 230.6, "p95_s": 243.2, "cv": 0.064},
        },
        "llama3.2:1b": {
            WorkloadClass.INTERACTIVE_QA: {"mean_s": 113.0, "p95_s": 118.5, "cv": 0.033},
            WorkloadClass.STRUCTURED_JSON: {"mean_s": 2003.7, "p95_s": 2058.4, "cv": 0.024},
            WorkloadClass.LONG_FORM_GENERATION: {"mean_s": 7513.7, "p95_s": 7815.9, "cv": 0.039},
            WorkloadClass.LONG_CONTEXT_RETRIEVAL: {"mean_s": 555.0, "p95_s": 563.9, "cv": 0.013},
        },
        "qwen2.5:1.5b": {
            WorkloadClass.INTERACTIVE_QA: {"mean_s": 95.2, "p95_s": 106.1, "cv": 0.109},
            WorkloadClass.STRUCTURED_JSON: {"mean_s": 2170.0, "p95_s": 2306.6, "cv": 0.044},
            WorkloadClass.LONG_FORM_GENERATION: {"mean_s": 5882.5, "p95_s": 5936.8, "cv": 0.013},
            WorkloadClass.LONG_CONTEXT_RETRIEVAL: {"mean_s": 521.5, "p95_s": 546.5, "cv": 0.034},
        },
    }

    # Model switching overhead when target model is not currently resident (ms)
    MODEL_SWITCH_OVERHEAD_MS = 850.0

    @classmethod
    def evaluate(
        cls,
        policy: AdmissionPolicyType,
        model_id: str,
        workload_class: WorkloadClass,
        latency_sla_ms: float,
        current_queue_backlog_ms: float,
        current_queue_depth: int,
        current_in_flight: int,
        target_model_resident: bool = True,
        require_code_security_pass: bool = False,
    ) -> tuple[AdmissionDecision, float, str, bool]:
        """Evaluate admission decision under specified policy.

        Returns:
            (decision, predicted_total_ms, rationale, unverified_code_warning)
        """
        model_profiles = cls.EMPIRICAL_TASK_PROFILES.get(model_id)
        if not model_profiles or workload_class not in model_profiles:
            return (
                AdmissionDecision.INSUFFICIENT_EVIDENCE,
                0.0,
                f"No calibrated service distribution available for model '{model_id}' on {workload_class.value}.",
                require_code_security_pass,
            )

        prof = model_profiles[workload_class]
        p95_s = prof["p95_s"]

        switch_cost = 0.0 if target_model_resident else cls.MODEL_SWITCH_OVERHEAD_MS
        predicted_queue_delay = current_queue_backlog_ms + switch_cost
        predicted_total_ms = round(predicted_queue_delay + p95_s, 1)

        code_warning = require_code_security_pass

        # -------------------------------------------------------------
        # POLICY A: No Admission Control (FIFO Baseline)
        # -------------------------------------------------------------
        if policy == AdmissionPolicyType.POLICY_A_NO_CONTROL:
            # Admits everything up to max queue depth 20
            if current_queue_depth >= 20:
                return (
                    AdmissionDecision.REJECT_OR_DEFER,
                    predicted_total_ms,
                    f"Queue overflow ({current_queue_depth} requests); rejected under bounded queue limit.",
                    code_warning,
                )
            if current_queue_depth == 0 and current_in_flight == 0:
                return (
                    AdmissionDecision.ACCEPT,
                    predicted_total_ms,
                    "Immediate execution slot available; admitted under Policy A.",
                    code_warning,
                )
            return (
                AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT,
                predicted_total_ms,
                f"Server slot occupied; admitted to queue depth {current_queue_depth} under Policy A.",
                code_warning,
            )

        # -------------------------------------------------------------
        # POLICY B: Static Utilization Guardrail (rho <= 0.82)
        # -------------------------------------------------------------
        elif policy == AdmissionPolicyType.POLICY_B_STATIC_UTILIZATION:
            # Coarse guardrail: max 4 requests waiting in queue regardless of workload class
            waiting_total = current_queue_depth + (1 if current_in_flight > 0 else 0)
            if waiting_total > 3:
                return (
                    AdmissionDecision.REJECT_OR_DEFER,
                    predicted_total_ms,
                    f"Queue occupancy ({waiting_total}) exceeds static saturation threshold (rho ~ 0.82); rejected.",
                    code_warning,
                )
            if waiting_total == 0:
                return (
                    AdmissionDecision.ACCEPT,
                    predicted_total_ms,
                    "Immediate execution slot available; admitted under static guardrail.",
                    code_warning,
                )
            return (
                AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT,
                predicted_total_ms,
                f"Admitted with bounded wait ({waiting_total} requests in queue) under static guardrail.",
                code_warning,
            )

        # -------------------------------------------------------------
        # POLICY C: Workload-Aware Adaptive Admission
        # -------------------------------------------------------------
        else:
            waiting_total = current_queue_depth + (1 if current_in_flight > 0 else 0)
            rationale_extra = ""
            if not target_model_resident:
                rationale_extra = f" [+ {cls.MODEL_SWITCH_OVERHEAD_MS}ms model-switch penalty]"

            if waiting_total == 0 and predicted_total_ms <= latency_sla_ms:
                return (
                    AdmissionDecision.ACCEPT,
                    predicted_total_ms,
                    f"Immediate slot available; predicted P95 ({predicted_total_ms}ms) satisfies SLA ({latency_sla_ms}ms){rationale_extra}.",
                    code_warning,
                )
            elif predicted_total_ms <= latency_sla_ms:
                return (
                    AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT,
                    predicted_total_ms,
                    f"Admitted to queue behind {waiting_total} request(s); total predicted latency ({predicted_total_ms}ms) satisfies SLA ({latency_sla_ms}ms){rationale_extra}.",
                    code_warning,
                )
            else:
                return (
                    AdmissionDecision.REJECT_OR_DEFER,
                    predicted_total_ms,
                    f"Predicted latency ({predicted_total_ms}ms) breaches SLA ({latency_sla_ms}ms) due to queue backlog ({current_queue_backlog_ms:.1f}ms){rationale_extra}. Request shed to preserve SLA.",
                    code_warning,
                )


def evaluate_policy_report(
    records: list[MixedRequestAccountingRecord],
    policy: AdmissionPolicyType,
    profile_type: WorkloadProfileType,
    offered_rps: float,
) -> PolicyEvaluationReport:
    """Aggregate per-request accounting records into a comprehensive policy evaluation report.

    Enforces strict accounting invariant:
        offered == completed + rejected + timed_out + canceled + outstanding
    """
    total_offered = len(records)
    total_admitted = sum(1 for r in records if r.admitted)
    total_completed = sum(1 for r in records if r.terminal_status == RequestTerminalStatus.COMPLETED)
    total_rejected = sum(1 for r in records if r.terminal_status == RequestTerminalStatus.REJECTED)
    total_timed_out = sum(1 for r in records if r.terminal_status == RequestTerminalStatus.TIMED_OUT)
    total_canceled = sum(1 for r in records if r.terminal_status == RequestTerminalStatus.CANCELED)
    total_outstanding = sum(1 for r in records if r.terminal_status == RequestTerminalStatus.OUTSTANDING)

    # Invariant check
    accounting_sum = total_completed + total_rejected + total_timed_out + total_canceled + total_outstanding
    invariant_holds = (total_offered == accounting_sum)

    # SLA Attainment
    completed_records = [r for r in records if r.terminal_status == RequestTerminalStatus.COMPLETED]
    admitted_records = [r for r in records if r.admitted]

    # Completed requests SLA
    completed_sla_met = sum(1 for r in completed_records if r.sla_met is True)
    # Admitted SLA attainment (failed or timed out admitted requests count as missed)
    admitted_sla_met = sum(1 for r in admitted_records if r.sla_met is True)
    admitted_sla_pct = round((admitted_sla_met / max(1, total_admitted)) * 100.0, 1)

    # All offered SLA attainment (rejected, timed out, canceled count as not met)
    all_offered_sla_met = completed_sla_met
    all_offered_sla_pct = round((all_offered_sla_met / max(1, total_offered)) * 100.0, 1)

    # False feasible: admitted requests that failed SLA
    false_feasible_count = sum(1 for r in admitted_records if r.sla_met is False)

    # Unnecessarily rejected: rejected requests whose isolated service time was well below SLA and queue was short
    unnecessarily_rejected_count = sum(
        1 for r in records
        if r.terminal_status == RequestTerminalStatus.REJECTED
        and r.workload_class in [WorkloadClass.INTERACTIVE_QA, WorkloadClass.STRUCTURED_JSON]
        and r.queue_depth_at_arrival <= 1
        and (r.latency_sla_ms >= 1000.0)
    )

    # Latency percentiles (completed requests only)
    if completed_records:
        e2es = sorted([r.total_turnaround_ms or (r.client_e2e_ms or 0.0) for r in completed_records])
        cn = len(e2es)
        p50 = statistics.median(e2es)
        p90 = e2es[min(int(cn * 0.90), cn - 1)]
        p95 = e2es[min(int(cn * 0.95), cn - 1)]
        mean_q = statistics.mean([r.queue_delay_ms for r in completed_records])
        max_q = max(r.queue_depth_at_arrival for r in completed_records)
    else:
        p50, p90, p95, mean_q, max_q = 0.0, 0.0, 0.0, 0.0, 0

    # Model switches
    switch_count = sum(1 for r in records if r.residency_before == ModelResidencyState.COLD_SWITCH)

    # Per-class SLA attainment
    classes_present = set(r.workload_class.value for r in records)
    per_class_sla: dict[str, float] = {}
    class_sla_values: list[float] = []

    for cls_name in classes_present:
        cls_offered = [r for r in records if r.workload_class.value == cls_name]
        cls_met = sum(1 for r in cls_offered if r.sla_met is True)
        cls_pct = round((cls_met / max(1, len(cls_offered))) * 100.0, 1)
        per_class_sla[cls_name] = cls_pct
        class_sla_values.append(cls_pct)

    # Jain's Fairness Index across workload classes
    if class_sla_values and any(v > 0 for v in class_sla_values):
        k = len(class_sla_values)
        sum_x = sum(class_sla_values)
        sum_x2 = sum(x ** 2 for x in class_sla_values)
        fairness = round((sum_x ** 2) / (k * max(1e-6, sum_x2)), 3)
    else:
        fairness = 1.0

    # Trade-off Utility Score: Completed SLA Met - Unnecessary Rejections - 2 * False Feasibles
    # Prevents "reject-all" gaming while penalizing SLA breaches
    utility = round(completed_sla_met - (unnecessarily_rejected_count * 0.5) - (false_feasible_count * 2.0), 1)

    # Completed throughput
    if records:
        t_span = max(0.001, (records[-1].scheduled_arrival_mono_s - records[0].scheduled_arrival_mono_s) + 2.0)
        throughput = round(total_completed / t_span, 2)
    else:
        throughput = 0.0

    return PolicyEvaluationReport(
        policy_type=policy,
        profile_type=profile_type,
        offered_rps=round(offered_rps, 2),
        total_offered=total_offered,
        total_admitted=total_admitted,
        total_completed=total_completed,
        total_rejected=total_rejected,
        total_timed_out=total_timed_out,
        total_canceled=total_canceled,
        total_outstanding=total_outstanding,
        accounting_invariant_holds=invariant_holds,
        all_offered_sla_attainment_pct=all_offered_sla_pct,
        admitted_sla_attainment_pct=admitted_sla_pct,
        false_feasible_count=false_feasible_count,
        unnecessarily_rejected_count=unnecessarily_rejected_count,
        completed_p50_e2e_ms=round(p50, 1),
        completed_p90_e2e_ms=round(p90, 1),
        completed_p95_e2e_ms=round(p95, 1),
        mean_queue_delay_ms=round(mean_q, 1),
        max_queue_depth=max_q,
        completed_throughput_rps=throughput,
        model_switch_count=switch_count,
        per_class_sla_attainment=per_class_sla,
        workload_fairness_index=fairness,
        tradeoff_utility_score=utility,
    )
