"""SLA-aware inference request admission controller for Session 25.

Evaluates incoming requests against current queue depth, in-flight state, service time distributions,
and latency SLAs to make conservative admission decisions (ACCEPT, QUEUE, REJECT, INSUFFICIENT_EVIDENCE).
"""

from llm_ii_engine.capacity.protocol import (
    AdmissionDecision,
    AdmissionRequest,
    AdmissionResult,
    CapacityEvidenceTier,
)


class SLAAdmissionController:
    """Deterministic, SLA-aware request admission controller."""

    # Empirical service rates (ms per prompt token, ms per decode token, Cv)
    # Empirical service rates (ms per prompt token, ms per decode token, Cv) calibrated on Intel i7-14700
    EMPIRICAL_PROFILES: dict[str, dict[int, dict[str, float]]] = {
        "qwen2.5:0.5b": {
            8: {"prefill_ms_per_tok": 1.50, "decode_ms_per_tok": 3.5, "cv": 0.06},
            4: {"prefill_ms_per_tok": 2.20, "decode_ms_per_tok": 4.5, "cv": 0.08},
            2: {"prefill_ms_per_tok": 3.80, "decode_ms_per_tok": 6.0, "cv": 0.10},
        },
        "llama3.2:1b": {
            8: {"prefill_ms_per_tok": 2.00, "decode_ms_per_tok": 6.8, "cv": 0.04},
            4: {"prefill_ms_per_tok": 3.20, "decode_ms_per_tok": 8.5, "cv": 0.06},
            2: {"prefill_ms_per_tok": 5.50, "decode_ms_per_tok": 12.0, "cv": 0.09},
        },
        "qwen2.5:1.5b": {
            8: {"prefill_ms_per_tok": 2.20, "decode_ms_per_tok": 5.8, "cv": 0.05},
            4: {"prefill_ms_per_tok": 3.50, "decode_ms_per_tok": 7.5, "cv": 0.07},
            2: {"prefill_ms_per_tok": 6.00, "decode_ms_per_tok": 11.0, "cv": 0.09},
        },
    }

    def evaluate_admission(self, request: AdmissionRequest) -> AdmissionResult:
        """Evaluate admission decision for an incoming inference request."""
        model_id = request.model_id
        threads = request.num_thread

        # Check empirical profile availability
        model_profiles = self.EMPIRICAL_PROFILES.get(model_id)
        if not model_profiles or threads not in model_profiles:
            # Fallback or unknown configuration
            return AdmissionResult(
                decision=AdmissionDecision.INSUFFICIENT_EVIDENCE,
                model_id=model_id,
                num_thread=threads,
                predicted_queue_delay_ms=0.0,
                predicted_service_time_ms=0.0,
                predicted_total_latency_ms=0.0,
                sla_compliance_probability=0.0,
                current_server_utilization=0.0,
                capacity_headroom_pct=0.0,
                rationale=f"No calibrated service distribution available for model '{model_id}' with {threads} threads.",
                evidence_tier=CapacityEvidenceTier.UNKNOWN_CAPACITY,
                unverified_code_warning=request.require_code_execution_pass,
            )

        prof = model_profiles[threads]
        prefill_ms = request.prompt_tokens * prof["prefill_ms_per_tok"]
        decode_ms = request.expected_output_tokens * prof["decode_ms_per_tok"]
        mean_service_ms = round(prefill_ms + decode_ms, 1)
        cv = prof["cv"]
        p95_service_ms = round(mean_service_ms * (1.0 + (cv * 1.645)), 1)

        # In single-slot Ollama execution (slot id 0):
        # Existing in-flight requests plus queue depth determine pending delay
        waiting_requests = request.current_queue_depth + (1 if request.current_in_flight > 0 else 0)
        predicted_queue_delay_ms = round(waiting_requests * mean_service_ms, 1)

        predicted_total_latency_ms = round(predicted_queue_delay_ms + mean_service_ms, 1)
        p95_total_latency_ms = round(predicted_queue_delay_ms + p95_service_ms, 1)

        # Current utilization and headroom estimate
        # Max queue headroom set to 5 requests before guaranteed saturation
        utilization = min(1.0, round(waiting_requests / 4.0, 2))
        headroom_pct = round(max(0.0, (1.0 - utilization) * 100.0), 1)

        # Code security warning check (Session 23 guardrail)
        code_warning = request.require_code_execution_pass

        # SLA Decision Evaluation
        sla = request.latency_sla_ms
        if sla is None:
            # No SLA constraint: decide based on queue backlog
            if waiting_requests == 0:
                decision = AdmissionDecision.ACCEPT
                prob = 1.0
                rationale = "Immediate server slot available; request admitted for execution."
            elif waiting_requests <= 3:
                decision = AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT
                prob = 0.85
                rationale = f"Slot busy; admitted to queue behind {waiting_requests} request(s) with bounded wait."
            else:
                decision = AdmissionDecision.REJECT_OR_DEFER
                prob = 0.20
                rationale = f"Queue backlog ({waiting_requests} waiting) exceeds safety threshold; request deferred to prevent overload."
        else:
            # SLA constrained admission
            if waiting_requests == 0 and p95_total_latency_ms <= sla:
                decision = AdmissionDecision.ACCEPT
                prob = 0.95
                rationale = (
                    f"Immediate execution slot available; predicted P95 response time ({p95_total_latency_ms} ms) "
                    f"satisfies target SLA ({sla} ms)."
                )
            elif waiting_requests > 0 and p95_total_latency_ms <= sla:
                decision = AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT
                prob = 0.90
                rationale = (
                    f"Slot busy; admitted to queue behind {waiting_requests} request(s). Predicted P95 response time "
                    f"({p95_total_latency_ms} ms) satisfies SLA ({sla} ms) with bounded queue delay ({predicted_queue_delay_ms} ms)."
                )
            elif predicted_total_latency_ms <= sla < p95_total_latency_ms:
                decision = AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT
                prob = 0.70
                rationale = (
                    f"Mean response time ({predicted_total_latency_ms} ms) satisfies SLA ({sla} ms), "
                    f"but tail risk exists ({p95_total_latency_ms} ms P95). Admitted with queue monitoring."
                )
            else:
                decision = AdmissionDecision.REJECT_OR_DEFER
                prob = max(0.05, round(sla / max(1.0, predicted_total_latency_ms), 2))
                rationale = (
                    f"Predicted response time ({predicted_total_latency_ms} ms) violates latency SLA ({sla} ms) "
                    f"due to accumulated queue delay ({predicted_queue_delay_ms} ms). Request rejected/deferred."
                )

        if code_warning:
            rationale += " [Warning: Model-generated code correctness is unverified under fail-closed security mode]."

        return AdmissionResult(
            decision=decision,
            model_id=model_id,
            num_thread=threads,
            predicted_queue_delay_ms=predicted_queue_delay_ms,
            predicted_service_time_ms=mean_service_ms,
            predicted_total_latency_ms=predicted_total_latency_ms,
            sla_compliance_probability=prob,
            current_server_utilization=utilization,
            capacity_headroom_pct=headroom_pct,
            rationale=rationale,
            evidence_tier=CapacityEvidenceTier.DIRECT_MEASURED,
            unverified_code_warning=code_warning,
        )
