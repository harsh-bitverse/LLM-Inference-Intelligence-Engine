"""Workload generator and load scheduler for Session 25 capacity and queueing benchmarks.

Executes:
1. Stage A: Baseline isolated service time distributions (E[S], Var[S], Cv) across models and tasks.
2. Stage B: Open-loop arrival rate pacing (constant and Poisson) to observe queue knee and divergence.
3. Stage C: Closed-loop concurrency scaling to measure server serialization and queue buildup.
"""

import json
import random
import statistics
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from llm_ii_engine.benchmark.workload_suite import BenchmarkWorkloadTask
from llm_ii_engine.capacity.protocol import (
    ArrivalProcess,
    CapacityCurvePoint,
    CapacityObservation,
    ServiceTimeDistribution,
)
from llm_ii_engine.capacity.queueing_model import QueueingCalculator

BASE_URL = "http://127.0.0.1:11434"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def execute_capacity_request(
    model_id: str,
    prompt: str,
    max_output_tokens: int,
    num_thread: int,
    workload_id: str,
    arrival_process: ArrivalProcess,
    scheduled_arrival_ms: float,
    concurrency_level: int = 1,
    in_flight_count: int = 1,
    observation_id: str = "obs-s25-0001",
    num_ctx: int = 2048,
    timeout_sec: float = 60.0,
) -> CapacityObservation:
    """Execute a single generation request against Ollama and record fine-grained timing."""
    actual_dispatch_s = time.perf_counter()
    actual_dispatch_ms = actual_dispatch_s * 1000.0
    dispatch_delay_ms = max(0.0, actual_dispatch_ms - scheduled_arrival_ms)

    url = f"{BASE_URL}/api/generate"
    options: dict[str, Any] = {
        "num_thread": num_thread,
        "num_ctx": num_ctx,
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

    first_chunk_s: float | None = None
    final_payload: dict[str, Any] = {}
    error_msg: str | None = None
    status = "SUCCESS"

    try:
        with OPENER.open(req, timeout=timeout_sec) as resp:
            for line in resp:
                t_now = time.perf_counter()
                if first_chunk_s is None:
                    first_chunk_s = t_now
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
        status = "TIMEOUT"
        error_msg = f"Request timed out after {timeout_sec}s"
    except Exception as exc:
        status = "ERROR"
        error_msg = str(exc)

    t_end_s = time.perf_counter()
    client_e2e_ms = max(0.0, (t_end_s - actual_dispatch_s) * 1000.0)

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

    # In Ollama single-slot serving, if server was occupied, client_e2e exceeds internal server processing
    server_active_ms = server_load_ms + server_prefill_ms + server_decode_ms
    estimated_queue_wait_ms = max(0.0, client_e2e_ms - server_active_ms)

    return CapacityObservation(
        observation_id=observation_id,
        timestamp_utc=datetime.now(UTC).isoformat(),
        model_id=model_id,
        num_thread=num_thread,
        workload_id=workload_id,
        arrival_process=arrival_process,
        scheduled_arrival_ms=round(scheduled_arrival_ms, 2),
        actual_dispatch_ms=round(actual_dispatch_ms, 2),
        dispatch_delay_ms=round(dispatch_delay_ms, 2),
        client_e2e_ms=round(client_e2e_ms, 2),
        server_prefill_ms=round(server_prefill_ms, 2),
        server_decode_ms=round(server_decode_ms, 2),
        estimated_queue_wait_ms=round(estimated_queue_wait_ms, 2),
        output_tokens=output_tokens,
        concurrency_level=concurrency_level,
        in_flight_count=in_flight_count,
        status=status,
        error_message=error_msg,
    )


class WorkloadLoadGenerator:
    """Orchestrates Stage A, B, and C capacity experiments with bounded execution budgets."""

    @staticmethod
    def measure_service_distribution(
        model_id: str,
        task: BenchmarkWorkloadTask,
        num_thread: int = 8,
        sample_count: int = 5,
        observation_prefix: str = "stage_a",
    ) -> tuple[ServiceTimeDistribution, list[CapacityObservation]]:
        """Run isolated sequential requests to compute accurate service time distribution."""
        observations: list[CapacityObservation] = []
        e2e_list: list[float] = []
        prefill_list: list[float] = []
        decode_list: list[float] = []
        tokens_list: list[int] = []

        # Warm up request once to stabilize model weight paging
        warmup_sched = time.perf_counter() * 1000.0
        _ = execute_capacity_request(
            model_id=model_id,
            prompt=task.prompt,
            max_output_tokens=task.max_output_tokens,
            num_thread=num_thread,
            workload_id=task.task_id,
            arrival_process=ArrivalProcess.CLOSED_LOOP,
            scheduled_arrival_ms=warmup_sched,
            concurrency_level=1,
            in_flight_count=1,
            observation_id=f"{observation_prefix}-warmup",
        )
        time.sleep(0.3)

        for i in range(sample_count):
            now_ms = time.perf_counter() * 1000.0
            obs = execute_capacity_request(
                model_id=model_id,
                prompt=task.prompt,
                max_output_tokens=task.max_output_tokens,
                num_thread=num_thread,
                workload_id=task.task_id,
                arrival_process=ArrivalProcess.CLOSED_LOOP,
                scheduled_arrival_ms=now_ms,
                concurrency_level=1,
                in_flight_count=1,
                observation_id=f"{observation_prefix}-{i+1:03d}",
            )
            observations.append(obs)
            if obs.status == "SUCCESS":
                e2e_list.append(obs.client_e2e_ms)
                prefill_list.append(obs.server_prefill_ms)
                decode_list.append(obs.server_decode_ms)
                tokens_list.append(obs.output_tokens)
            time.sleep(0.2)

        dist = QueueingCalculator.calculate_distribution(
            service_times_ms=e2e_list,
            prefill_times_ms=prefill_list,
            decode_times_ms=decode_list,
            output_tokens=tokens_list,
        )
        return dist, observations

    @staticmethod
    def run_scheduled_arrivals(
        model_id: str,
        task: BenchmarkWorkloadTask,
        target_arrival_rps: float,
        request_count: int = 10,
        num_thread: int = 8,
        is_poisson: bool = False,
        latency_sla_ms: float | None = None,
        observation_prefix: str = "stage_b",
    ) -> tuple[CapacityCurvePoint, list[CapacityObservation]]:
        """Dispatch requests according to an open-loop scheduled arrival process."""
        observations: list[CapacityObservation] = []
        process_type = ArrivalProcess.POISSON if is_poisson else ArrivalProcess.OPEN_LOOP_SCHEDULED

        # Generate scheduled arrival times
        inter_arrival_times: list[float] = []
        for _ in range(request_count):
            if is_poisson:
                # Exponentially distributed inter-arrival time
                ia = random.expovariate(target_arrival_rps)
            else:
                ia = 1.0 / max(0.01, target_arrival_rps)
            inter_arrival_times.append(ia)

        # Baseline start time
        t0 = time.perf_counter()
        scheduled_delays: list[float] = []
        accum = 0.0
        for ia in inter_arrival_times:
            accum += ia
            scheduled_delays.append(accum)

        max_workers = min(16, request_count)
        active_in_flight = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: list[Future[CapacityObservation]] = []
            start_monotonic = time.perf_counter()
            start_ms = start_monotonic * 1000.0

            for idx, delay_sec in enumerate(scheduled_delays):
                sched_ms = start_ms + (delay_sec * 1000.0)
                obs_id = f"{observation_prefix}-r{target_arrival_rps:.2f}-{idx+1:03d}"

                def _dispatch(
                    s_delay: float, s_sched_ms: float, o_id: str, in_fl: int
                ) -> CapacityObservation:
                    # Sleep until scheduled arrival time
                    target_time = start_monotonic + s_delay
                    curr_time = time.perf_counter()
                    if curr_time < target_time:
                        time.sleep(target_time - curr_time)

                    return execute_capacity_request(
                        model_id=model_id,
                        prompt=task.prompt,
                        max_output_tokens=task.max_output_tokens,
                        num_thread=num_thread,
                        workload_id=task.task_id,
                        arrival_process=process_type,
                        scheduled_arrival_ms=s_sched_ms,
                        concurrency_level=max_workers,
                        in_flight_count=in_fl,
                        observation_id=o_id,
                    )

                active_in_flight += 1
                fut = executor.submit(_dispatch, delay_sec, sched_ms, obs_id, active_in_flight)
                futures.append(fut)

            for fut in as_completed(futures):
                obs = fut.result()
                observations.append(obs)

        total_elapsed_sec = max(0.001, time.perf_counter() - t0)
        successful_obs = [o for o in observations if o.status == "SUCCESS"]
        completed_rps = len(successful_obs) / total_elapsed_sec

        e2e_latencies = [o.client_e2e_ms for o in successful_obs]
        queue_delays = [o.estimated_queue_wait_ms for o in successful_obs]

        if not e2e_latencies:
            median_e2e = 99999.0
            p90_e2e = 99999.0
            p95_e2e = 99999.0
            mean_q = 99999.0
            sla_pct = 0.0
        else:
            sorted_e2e = sorted(e2e_latencies)
            n = len(sorted_e2e)
            median_e2e = statistics.median(sorted_e2e)
            p90_e2e = sorted_e2e[min(int(n * 0.90), n - 1)]
            p95_e2e = sorted_e2e[min(int(n * 0.95), n - 1)]
            mean_q = statistics.mean(queue_delays) if queue_delays else 0.0

            if latency_sla_ms:
                met = sum(1 for lat in e2e_latencies if lat <= latency_sla_ms)
                sla_pct = round((met / len(e2e_latencies)) * 100.0, 1)
            else:
                sla_pct = 100.0

        # Detect queue buildup: if later requests have significantly higher queue delay
        buildup = False
        if len(queue_delays) >= 6:
            half = len(queue_delays) // 2
            first_half_mean = statistics.mean(queue_delays[:half])
            second_half_mean = statistics.mean(queue_delays[half:])
            if second_half_mean > (first_half_mean * 1.5) and second_half_mean > 500.0:
                buildup = True

        failures = sum(1 for o in observations if o.status != "SUCCESS")

        point = CapacityCurvePoint(
            offered_rps=round(target_arrival_rps, 3),
            completed_rps=round(completed_rps, 3),
            estimated_utilization=round(min(2.0, completed_rps * (median_e2e / 1000.0)), 3),
            median_e2e_ms=round(median_e2e, 1),
            p90_e2e_ms=round(p90_e2e, 1),
            p95_e2e_ms=round(p95_e2e, 1),
            mean_queue_delay_ms=round(mean_q, 1),
            sla_compliance_pct=sla_pct,
            queue_buildup_detected=buildup,
            failure_count=failures,
        )
        return point, observations

    @staticmethod
    def run_concurrency_sweep(
        model_id: str,
        task: BenchmarkWorkloadTask,
        concurrency_levels: list[int] = [1, 2, 4, 8],
        requests_per_level: int = 6,
        num_thread: int = 8,
        latency_sla_ms: float | None = None,
        observation_prefix: str = "stage_c",
    ) -> tuple[dict[int, CapacityCurvePoint], list[CapacityObservation]]:
        """Evaluate closed-loop concurrency scaling to characterize serialization behavior."""
        all_observations: list[CapacityObservation] = []
        curve_points: dict[int, CapacityCurvePoint] = {}

        for concurrency in concurrency_levels:
            concurrency_obs: list[CapacityObservation] = []
            t_start = time.perf_counter()

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures: list[Future[CapacityObservation]] = []
                for i in range(requests_per_level):
                    sched_ms = time.perf_counter() * 1000.0
                    obs_id = f"{observation_prefix}-c{concurrency}-{i+1:03d}"
                    fut = executor.submit(
                        execute_capacity_request,
                        model_id=model_id,
                        prompt=task.prompt,
                        max_output_tokens=task.max_output_tokens,
                        num_thread=num_thread,
                        workload_id=task.task_id,
                        arrival_process=ArrivalProcess.CLOSED_LOOP,
                        scheduled_arrival_ms=sched_ms,
                        concurrency_level=concurrency,
                        in_flight_count=concurrency,
                        observation_id=obs_id,
                    )
                    futures.append(fut)

                for fut in as_completed(futures):
                    obs = fut.result()
                    concurrency_obs.append(obs)
                    all_observations.append(obs)

            elapsed_sec = max(0.001, time.perf_counter() - t_start)
            successful = [o for o in concurrency_obs if o.status == "SUCCESS"]
            completed_rps = len(successful) / elapsed_sec
            e2es = [o.client_e2e_ms for o in successful]
            q_waits = [o.estimated_queue_wait_ms for o in successful]

            if e2es:
                sorted_e = sorted(e2es)
                n = len(sorted_e)
                med_e = statistics.median(sorted_e)
                p90_e = sorted_e[min(int(n * 0.90), n - 1)]
                p95_e = sorted_e[min(int(n * 0.95), n - 1)]
                mean_q = statistics.mean(q_waits) if q_waits else 0.0
                sla_pct = (
                    round((sum(1 for e in e2es if e <= latency_sla_ms) / n) * 100.0, 1)
                    if latency_sla_ms
                    else 100.0
                )
            else:
                med_e = 99999.0
                p90_e = 99999.0
                p95_e = 99999.0
                mean_q = 99999.0
                sla_pct = 0.0

            point = CapacityCurvePoint(
                offered_rps=round(concurrency / max(0.001, med_e / 1000.0), 3),
                completed_rps=round(completed_rps, 3),
                estimated_utilization=1.0 if concurrency > 1 else round(min(1.0, completed_rps * (med_e / 1000.0)), 3),
                median_e2e_ms=round(med_e, 1),
                p90_e2e_ms=round(p90_e, 1),
                p95_e2e_ms=round(p95_e, 1),
                mean_queue_delay_ms=round(mean_q, 1),
                sla_compliance_pct=sla_pct,
                queue_buildup_detected=concurrency >= 4,
                failure_count=sum(1 for o in concurrency_obs if o.status != "SUCCESS"),
            )
            curve_points[concurrency] = point
            time.sleep(0.3)

        return curve_points, all_observations
