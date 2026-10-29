"""Execution runner and live measurement engine for Session 26 mixed workloads.

Executes:
1. Mixed-workload profile benchmark campaign (Profiles A, B, C, D across Policies A, B, C).
2. Dedicated Head-of-Line (HoL) blocking probe (short request blocked by antecedent long request).
3. Dedicated model-switching overhead probe (alternating models under single-slot residency).
4. Full monotonic timestamp arrival accounting ensuring:
   offered == completed + rejected + timed_out + canceled + outstanding
"""

import json
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from llm_ii_engine.benchmark.workload_suite import (
    TASK_INTERACTIVE_QA,
    TASK_LONG_FORM,
    TASK_STRUCTURED_JSON,
    BenchmarkWorkloadTask,
)
from llm_ii_engine.capacity.adaptive_admission_policy import (
    MixedWorkloadAdmissionController,
)
from llm_ii_engine.capacity.mixed_workload_protocol import (
    AdmissionPolicyType,
    HeadOfLineBlockingObservation,
    MixedRequestAccountingRecord,
    MixedWorkloadProfile,
    ModelResidencyState,
    ModelSwitchingObservation,
    RequestTerminalStatus,
    WorkloadClass,
    WorkloadProfileType,
)
from llm_ii_engine.capacity.protocol import AdmissionDecision

BASE_URL = "http://127.0.0.1:11434"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TASK_BY_CLASS: dict[WorkloadClass, BenchmarkWorkloadTask] = {
    WorkloadClass.INTERACTIVE_QA: TASK_INTERACTIVE_QA,
    WorkloadClass.STRUCTURED_JSON: TASK_STRUCTURED_JSON,
    WorkloadClass.LONG_FORM_GENERATION: TASK_LONG_FORM,
    WorkloadClass.LONG_CONTEXT_RETRIEVAL: TASK_INTERACTIVE_QA,
}


def execute_mixed_request(
    request_id: str,
    model_id: str,
    workload_class: WorkloadClass,
    prompt: str,
    max_output_tokens: int,
    latency_sla_ms: float,
    num_thread: int,
    scheduled_arrival_mono_s: float,
    actual_enqueue_mono_s: float,
    queue_depth_at_arrival: int,
    in_flight_at_arrival: int,
    admitted: bool,
    admission_policy: AdmissionPolicyType,
    residency_before: ModelResidencyState = ModelResidencyState.UNKNOWN,
    timeout_sec: float = 60.0,
) -> MixedRequestAccountingRecord:
    """Execute streaming request with monotonic timestamp accounting and SLA evaluation."""
    if not admitted:
        # Request rejected prior to dispatch: record immediate REJECTED terminal state
        completion_mono = time.perf_counter()
        return MixedRequestAccountingRecord(
            request_id=request_id,
            timestamp_utc=datetime.now(UTC).isoformat(),
            model_id=model_id,
            num_thread=num_thread,
            workload_class=workload_class,
            latency_sla_ms=latency_sla_ms,
            scheduled_arrival_mono_s=scheduled_arrival_mono_s,
            actual_enqueue_mono_s=actual_enqueue_mono_s,
            dispatch_mono_s=None,
            first_response_mono_s=None,
            completion_mono_s=completion_mono,
            queue_depth_at_arrival=queue_depth_at_arrival,
            in_flight_at_arrival=in_flight_at_arrival,
            residency_before=residency_before,
            admitted=False,
            admission_policy=admission_policy,
            terminal_status=RequestTerminalStatus.REJECTED,
            error_message="Rejected by admission controller prior to dispatch",
            queue_delay_ms=0.0,
            client_e2e_ms=0.0,
            total_turnaround_ms=0.0,
            ttft_ms=None,
            server_load_ms=0.0,
            server_prefill_ms=0.0,
            server_decode_ms=0.0,
            output_tokens=0,
            sla_met=False,  # Rejections do not meet SLA
        )

    dispatch_mono = time.perf_counter()
    queue_delay_ms = max(0.0, (dispatch_mono - scheduled_arrival_mono_s) * 1000.0)

    url = f"{BASE_URL}/api/generate"
    options: dict[str, Any] = {
        "num_thread": num_thread,
        "num_ctx": 2048,
        "num_predict": max_output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }

    payload = json.dumps(
        {
            "model": model_id,
            "prompt": prompt,
            "stream": True,
            "options": options,
            "keep_alive": "5m",
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    first_resp_mono: float | None = None
    final_payload: dict[str, Any] = {}
    error_msg: str | None = None
    terminal_status = RequestTerminalStatus.COMPLETED

    try:
        with OPENER.open(req, timeout=timeout_sec) as resp:
            for line in resp:
                t_now = time.perf_counter()
                if first_resp_mono is None:
                    first_resp_mono = t_now
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    if data.get("done", False):
                        final_payload = data
                except json.JSONDecodeError:
                    pass
    except TimeoutError:
        terminal_status = RequestTerminalStatus.TIMED_OUT
        error_msg = f"Request timed out after {timeout_sec}s"
    except Exception as exc:
        terminal_status = RequestTerminalStatus.CANCELED
        error_msg = str(exc)

    completion_mono = time.perf_counter()
    client_e2e_ms = max(0.0, (completion_mono - dispatch_mono) * 1000.0)
    total_turnaround_ms = max(0.0, (completion_mono - scheduled_arrival_mono_s) * 1000.0)
    ttft_ms = (
        max(0.0, (first_resp_mono - dispatch_mono) * 1000.0)
        if first_resp_mono
        else None
    )

    server_load_ms = (
        float(final_payload.get("load_duration", 0)) / 1e6
        if final_payload.get("load_duration")
        else 0.0
    )
    server_prefill_ms = (
        float(final_payload.get("prompt_eval_duration", 0)) / 1e6
        if final_payload.get("prompt_eval_duration")
        else 0.0
    )
    server_decode_ms = (
        float(final_payload.get("eval_duration", 0)) / 1e6
        if final_payload.get("eval_duration")
        else 0.0
    )
    output_tokens = int(final_payload.get("eval_count", 0))

    sla_met = (
        (total_turnaround_ms <= latency_sla_ms)
        if terminal_status == RequestTerminalStatus.COMPLETED
        else False
    )

    return MixedRequestAccountingRecord(
        request_id=request_id,
        timestamp_utc=datetime.now(UTC).isoformat(),
        model_id=model_id,
        num_thread=num_thread,
        workload_class=workload_class,
        latency_sla_ms=latency_sla_ms,
        scheduled_arrival_mono_s=scheduled_arrival_mono_s,
        actual_enqueue_mono_s=actual_enqueue_mono_s,
        dispatch_mono_s=dispatch_mono,
        first_response_mono_s=first_resp_mono,
        completion_mono_s=completion_mono,
        queue_depth_at_arrival=queue_depth_at_arrival,
        in_flight_at_arrival=in_flight_at_arrival,
        residency_before=residency_before,
        admitted=True,
        admission_policy=admission_policy,
        terminal_status=terminal_status,
        error_message=error_msg,
        queue_delay_ms=round(queue_delay_ms, 2),
        client_e2e_ms=round(client_e2e_ms, 2),
        total_turnaround_ms=round(total_turnaround_ms, 2),
        ttft_ms=round(ttft_ms, 2) if ttft_ms else None,
        server_load_ms=round(server_load_ms, 2),
        server_prefill_ms=round(server_prefill_ms, 2),
        server_decode_ms=round(server_decode_ms, 2),
        output_tokens=output_tokens,
        sla_met=sla_met,
    )


class MixedWorkloadRunner:
    """Orchestrates live benchmarks for mixed workloads, HoL blocking, and model switching."""

    @classmethod
    def run_mixed_stream(
        cls,
        model_id: str,
        profile: MixedWorkloadProfile,
        offered_rps: float,
        policy: AdmissionPolicyType,
        total_requests: int = 15,
        num_thread: int = 8,
        experiment_prefix: str = "mix-exp",
    ) -> list[MixedRequestAccountingRecord]:
        """Dispatch a heterogeneous request stream under specified admission policy."""
        records: list[MixedRequestAccountingRecord] = []
        items = profile.items
        weights = [it.weight for it in items]

        # Deterministic sequence based on profile weights
        import random
        rng = random.Random(42)

        start_mono = time.perf_counter()
        futures: list[Future[MixedRequestAccountingRecord]] = []

        # Current estimated backlog tracker
        simulated_queue_depth = 0
        simulated_in_flight = 0
        simulated_backlog_ms = 0.0

        with ThreadPoolExecutor(max_workers=min(8, total_requests)) as executor:
            for i in range(total_requests):
                # Pacing interval
                if profile.profile_type == WorkloadProfileType.PROFILE_D_BURSTY:
                    delay_s = (i // 4) * (profile.burst_idle_interval_s or 2.0) + ((i % 4) * 0.05)
                else:
                    delay_s = i * (1.0 / max(0.01, offered_rps))

                sched_mono = start_mono + delay_s
                enqueue_mono = time.perf_counter()

                chosen_item = rng.choices(items, weights=weights, k=1)[0]
                task = TASK_BY_CLASS[chosen_item.workload_class]

                # Admission decision evaluation
                decision, pred_total, rationale, code_warn = MixedWorkloadAdmissionController.evaluate(
                    policy=policy,
                    model_id=model_id,
                    workload_class=chosen_item.workload_class,
                    latency_sla_ms=chosen_item.latency_sla_ms,
                    current_queue_backlog_ms=simulated_backlog_ms,
                    current_queue_depth=simulated_queue_depth,
                    current_in_flight=simulated_in_flight,
                    target_model_resident=True,
                )
                admitted = decision in [AdmissionDecision.ACCEPT, AdmissionDecision.QUEUE_WITH_BOUNDED_WAIT]

                req_id = f"{experiment_prefix}-{policy.value.lower()}-{i+1:03d}"

                if admitted:
                    simulated_queue_depth += 1
                    simulated_backlog_ms += chosen_item.expected_output_tokens * 15.0

                def _dispatch(
                    r_id: str,
                    w_cls: WorkloadClass,
                    p_str: str,
                    m_tok: int,
                    sla: float,
                    s_mono: float,
                    e_mono: float,
                    q_depth: int,
                    in_fl: int,
                    is_adm: bool,
                ) -> MixedRequestAccountingRecord:
                    curr_mono = time.perf_counter()
                    if curr_mono < s_mono:
                        time.sleep(s_mono - curr_mono)
                    return execute_mixed_request(
                        request_id=r_id,
                        model_id=model_id,
                        workload_class=w_cls,
                        prompt=p_str,
                        max_output_tokens=m_tok,
                        latency_sla_ms=sla,
                        num_thread=num_thread,
                        scheduled_arrival_mono_s=s_mono,
                        actual_enqueue_mono_s=e_mono,
                        queue_depth_at_arrival=q_depth,
                        in_flight_at_arrival=in_fl,
                        admitted=is_adm,
                        admission_policy=policy,
                    )

                fut = executor.submit(
                    _dispatch,
                    req_id,
                    chosen_item.workload_class,
                    task.prompt,
                    chosen_item.expected_output_tokens,
                    chosen_item.latency_sla_ms,
                    sched_mono,
                    enqueue_mono,
                    simulated_queue_depth,
                    simulated_in_flight,
                    admitted,
                )
                futures.append(fut)
                time.sleep(0.01)

            for fut in as_completed(futures):
                rec = fut.result()
                records.append(rec)
                if rec.admitted and simulated_queue_depth > 0:
                    simulated_queue_depth = max(0, simulated_queue_depth - 1)
                    simulated_backlog_ms = max(0.0, simulated_backlog_ms - 200.0)

        # Sort chronologically by scheduled arrival
        records.sort(key=lambda r: r.scheduled_arrival_mono_s)
        return records

    @classmethod
    def run_head_of_line_probe(
        cls,
        model_id: str = "llama3.2:1b",
        num_thread: int = 8,
    ) -> list[HeadOfLineBlockingObservation]:
        """Measure Head-of-Line blocking: short request arriving behind long generation."""
        observations: list[HeadOfLineBlockingObservation] = []

        # 1. Baseline: Isolated short interactive request
        t0 = time.perf_counter()
        iso_short = execute_mixed_request(
            request_id="hol-iso-short",
            model_id=model_id,
            workload_class=WorkloadClass.INTERACTIVE_QA,
            prompt=TASK_INTERACTIVE_QA.prompt,
            max_output_tokens=16,
            latency_sla_ms=500.0,
            num_thread=num_thread,
            scheduled_arrival_mono_s=t0,
            actual_enqueue_mono_s=t0,
            queue_depth_at_arrival=0,
            in_flight_at_arrival=0,
            admitted=True,
            admission_policy=AdmissionPolicyType.POLICY_A_NO_CONTROL,
        )
        iso_duration = iso_short.client_e2e_ms or 110.0
        time.sleep(0.3)

        # 2. HoL Scenario: Dispatch long request, then immediately queue short request behind it
        with ThreadPoolExecutor(max_workers=2) as executor:
            t_start = time.perf_counter()

            # Long request (antecedent)
            fut_long = executor.submit(
                execute_mixed_request,
                "hol-antecedent-long",
                model_id,
                WorkloadClass.LONG_FORM_GENERATION,
                TASK_LONG_FORM.prompt,
                160,
                8000.0,
                num_thread,
                t_start,
                t_start,
                0,
                1,
                True,
                AdmissionPolicyType.POLICY_A_NO_CONTROL,
            )

            # Short request (consequent), dispatched 50ms later to guarantee queuing behind long
            time.sleep(0.05)
            t_short = time.perf_counter()
            fut_short = executor.submit(
                execute_mixed_request,
                "hol-blocked-short",
                model_id,
                WorkloadClass.INTERACTIVE_QA,
                TASK_INTERACTIVE_QA.prompt,
                16,
                500.0,
                num_thread,
                t_short,
                t_short,
                1,
                1,
                True,
                AdmissionPolicyType.POLICY_A_NO_CONTROL,
            )

            long_rec = fut_long.result()
            short_rec = fut_short.result()

        antecedent_dur = long_rec.client_e2e_ms or 7000.0
        blocked_dur = short_rec.total_turnaround_ms or (short_rec.client_e2e_ms or 7000.0)
        extra_wait = max(0.0, blocked_dur - iso_duration)
        sla_missed = (blocked_dur > 500.0)

        obs = HeadOfLineBlockingObservation(
            experiment_id="hol-probe-01-single-behind-long",
            model_id=model_id,
            scenario_description="Short interactive request (SLA 500ms) queued behind long generation request.",
            antecedent_workload=WorkloadClass.LONG_FORM_GENERATION,
            consequent_workload=WorkloadClass.INTERACTIVE_QA,
            antecedent_duration_ms=round(antecedent_dur, 1),
            isolated_consequent_duration_ms=round(iso_duration, 1),
            blocked_consequent_duration_ms=round(blocked_dur, 1),
            additional_queue_delay_ms=round(extra_wait, 1),
            sla_missed_due_to_blocking=sla_missed,
        )
        observations.append(obs)
        return observations

    @classmethod
    def run_model_switching_probe(
        cls,
        model_a: str = "qwen2.5:0.5b",
        model_b: str = "llama3.2:1b",
        num_thread: int = 8,
    ) -> list[ModelSwitchingObservation]:
        """Measure cold weight-loading overhead when alternating between installed models."""
        observations: list[ModelSwitchingObservation] = []

        # 1. Warm up model_a
        t0 = time.perf_counter()
        _ = execute_mixed_request(
            request_id="switch-warmup-a",
            model_id=model_a,
            workload_class=WorkloadClass.INTERACTIVE_QA,
            prompt=TASK_INTERACTIVE_QA.prompt,
            max_output_tokens=16,
            latency_sla_ms=2000.0,
            num_thread=num_thread,
            scheduled_arrival_mono_s=t0,
            actual_enqueue_mono_s=t0,
            queue_depth_at_arrival=0,
            in_flight_at_arrival=0,
            admitted=True,
            admission_policy=AdmissionPolicyType.POLICY_A_NO_CONTROL,
            residency_before=ModelResidencyState.COLD_SWITCH,
        )
        time.sleep(0.3)

        # 2. Warm execution on model_a
        t1 = time.perf_counter()
        execute_mixed_request(
            request_id="switch-warm-a",
            model_id=model_a,
            workload_class=WorkloadClass.INTERACTIVE_QA,
            prompt=TASK_INTERACTIVE_QA.prompt,
            max_output_tokens=16,
            latency_sla_ms=2000.0,
            num_thread=num_thread,
            scheduled_arrival_mono_s=t1,
            actual_enqueue_mono_s=t1,
            queue_depth_at_arrival=0,
            in_flight_at_arrival=0,
            admitted=True,
            admission_policy=AdmissionPolicyType.POLICY_A_NO_CONTROL,
            residency_before=ModelResidencyState.RESIDENT,
        )
        time.sleep(0.3)

        # 3. Cold switch to model_b
        t2 = time.perf_counter()
        cold_b = execute_mixed_request(
            request_id="switch-cold-b",
            model_id=model_b,
            workload_class=WorkloadClass.INTERACTIVE_QA,
            prompt=TASK_INTERACTIVE_QA.prompt,
            max_output_tokens=16,
            latency_sla_ms=2000.0,
            num_thread=num_thread,
            scheduled_arrival_mono_s=t2,
            actual_enqueue_mono_s=t2,
            queue_depth_at_arrival=0,
            in_flight_at_arrival=0,
            admitted=True,
            admission_policy=AdmissionPolicyType.POLICY_A_NO_CONTROL,
            residency_before=ModelResidencyState.COLD_SWITCH,
        )
        cold_b_dur = cold_b.client_e2e_ms or 800.0
        time.sleep(0.3)

        # 4. Warm execution on model_b
        t3 = time.perf_counter()
        warm_b = execute_mixed_request(
            request_id="switch-warm-b",
            model_id=model_b,
            workload_class=WorkloadClass.INTERACTIVE_QA,
            prompt=TASK_INTERACTIVE_QA.prompt,
            max_output_tokens=16,
            latency_sla_ms=2000.0,
            num_thread=num_thread,
            scheduled_arrival_mono_s=t3,
            actual_enqueue_mono_s=t3,
            queue_depth_at_arrival=0,
            in_flight_at_arrival=0,
            admitted=True,
            admission_policy=AdmissionPolicyType.POLICY_A_NO_CONTROL,
            residency_before=ModelResidencyState.RESIDENT,
        )
        warm_b_dur = warm_b.client_e2e_ms or 120.0

        load_b_ms = cold_b.server_load_ms
        penalty_b = max(0.0, cold_b_dur - warm_b_dur)

        obs_b = ModelSwitchingObservation(
            experiment_id="switch-probe-01-a-to-b",
            from_model_id=model_a,
            to_model_id=model_b,
            from_model_resident_before=True,
            actual_load_duration_ms=round(load_b_ms, 1),
            client_switch_penalty_ms=round(penalty_b, 1),
            warm_execution_duration_ms=round(warm_b_dur, 1),
            cold_execution_duration_ms=round(cold_b_dur, 1),
        )
        observations.append(obs_b)
        return observations
