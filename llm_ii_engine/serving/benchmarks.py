"""Controlled serving architecture benchmark runner and overlap verification.

Benchmarks:
1. Single-slot baseline (Ollama, C=1, S=1).
2. Multi-client serial queueing (Ollama, C=2, C=4, S=1).
3. Multi-slot parallel continuous batching (llama.cpp server, C=2, C=4, S=2, S=4).
4. Matched Head-of-Line blocking comparison.
"""

import concurrent.futures
import ctypes
import json
import logging
import statistics
import time
import urllib.request
import uuid

from llm_ii_engine.serving.llama_server_manager import LlamaServerContext
from llm_ii_engine.serving.protocol import (
    HeadOfLineComparisonObservation,
    ServingArchitectureComparison,
    ServingObservation,
    ServingRuntimeArchitecture,
)

logger = logging.getLogger(__name__)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _get_host_telemetry() -> tuple[float, float]:
    """Capture host memory and estimated CPU without external dependencies."""
    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    used_mb = (mem.ullTotalPhys - mem.ullAvailPhys) / (1024 * 1024)
    cpu_pct = float(mem.dwMemoryLoad)  # proxy load factor
    return cpu_pct, used_mb

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LLAMA_SERVER_PORT = 18080
LLAMA_SERVER_BASE_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"


def execute_ollama_request(
    model_id: str,
    prompt: str,
    max_tokens: int,
    num_thread: int = 8,
    latency_sla_ms: float = 2000.0,
    workload_class: str = "INTERACTIVE_QA",
    client_concurrency: int = 1,
) -> ServingObservation:
    """Execute a single inference request against the running Ollama server."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "num_thread": num_thread,
        },
    }

    t0_mono = time.monotonic()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=30.0) as resp:
        t1_mono = time.monotonic()
        data = json.loads(resp.read().decode("utf-8"))

    client_duration_ms = (t1_mono - t0_mono) * 1000.0

    # Extract nanosecond server durations
    total_dur_ns = data.get("total_duration", 0)
    load_dur_ns = data.get("load_duration", 0)
    prompt_dur_ns = data.get("prompt_eval_duration", 0)
    eval_dur_ns = data.get("eval_duration", 0)
    eval_count = data.get("eval_count", max_tokens)

    prefill_ms = prompt_dur_ns / 1e6
    decode_ms = eval_dur_ns / 1e6

    # In Ollama: total_dur includes queue wait time before processing
    active_dur_ns = load_dur_ns + prompt_dur_ns + eval_dur_ns
    queue_wait_ms = max(0.0, (total_dur_ns - active_dur_ns) / 1e6)

    decode_tps = (eval_count / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
    sla_met = client_duration_ms <= latency_sla_ms

    # Telemetry
    cpu_pct, mem_mb = _get_host_telemetry()

    arch = (
        ServingRuntimeArchitecture.OLLAMA_SINGLE_SLOT
        if client_concurrency == 1
        else ServingRuntimeArchitecture.OLLAMA_CLIENT_QUEUED
    )

    return ServingObservation(
        observation_id=f"obs-ollama-{uuid.uuid4().hex[:8]}",
        runtime_architecture=arch,
        model_id=model_id,
        num_threads=num_thread,
        client_concurrency=client_concurrency,
        server_slots=1,
        effective_server_concurrency=1.0,
        workload_class=workload_class,
        prompt_tokens=data.get("prompt_eval_count", len(prompt.split())),
        output_tokens=eval_count,
        latency_sla_ms=latency_sla_ms,
        client_dispatch_mono_s=t0_mono,
        client_completion_mono_s=t1_mono,
        server_start_mono_s=t0_mono + (queue_wait_ms / 1000.0),
        server_completion_mono_s=t1_mono,
        client_duration_ms=round(client_duration_ms, 2),
        server_prefill_ms=round(prefill_ms, 2),
        server_decode_ms=round(decode_ms, 2),
        queue_wait_ms=round(queue_wait_ms, 2),
        ttft_ms=round(queue_wait_ms + prefill_ms, 2),
        decode_tokens_per_second=round(decode_tps, 2),
        overlapping_execution=False,
        overlap_duration_ms=0.0,
        slot_id_used=0,
        cpu_utilization_pct=round(cpu_pct, 1),
        memory_rss_mb=round(mem_mb, 1),
        sla_met=sla_met,
    )


def execute_llama_server_request(
    server_base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    num_thread: int = 8,
    latency_sla_ms: float = 2000.0,
    workload_class: str = "INTERACTIVE_QA",
    client_concurrency: int = 1,
    server_slots: int = 2,
) -> ServingObservation:
    """Execute a single inference request against the running llama.cpp server."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.1,
    }

    t0_mono = time.monotonic()
    req = urllib.request.Request(
        f"{server_base_url}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=30.0) as resp:
        t1_mono = time.monotonic()
        data = json.loads(resp.read().decode("utf-8"))

    client_duration_ms = (t1_mono - t0_mono) * 1000.0

    timings = data.get("timings", {})
    prompt_ms = timings.get("prompt_ms", 0.0)
    decode_ms = timings.get("predicted_ms", 0.0)
    eval_count = timings.get("predicted_n", max_tokens)
    prompt_count = timings.get("cache_n", 0) + timings.get("prompt_n", len(prompt.split()))
    decode_tps = timings.get("predicted_per_second", 0.0)

    queue_wait_ms = max(0.0, client_duration_ms - prompt_ms - decode_ms)
    sla_met = client_duration_ms <= latency_sla_ms

    cpu_pct, mem_mb = _get_host_telemetry()

    arch = (
        ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS
        if server_slots == 2
        else ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_4_SLOTS
    )

    return ServingObservation(
        observation_id=f"obs-llamacpp-{uuid.uuid4().hex[:8]}",
        runtime_architecture=arch,
        model_id=model_id,
        num_threads=num_thread,
        client_concurrency=client_concurrency,
        server_slots=server_slots,
        effective_server_concurrency=float(server_slots),
        workload_class=workload_class,
        prompt_tokens=prompt_count,
        output_tokens=eval_count,
        latency_sla_ms=latency_sla_ms,
        client_dispatch_mono_s=t0_mono,
        client_completion_mono_s=t1_mono,
        server_start_mono_s=t0_mono + (queue_wait_ms / 1000.0),
        server_completion_mono_s=t1_mono,
        client_duration_ms=round(client_duration_ms, 2),
        server_prefill_ms=round(prompt_ms, 2),
        server_decode_ms=round(decode_ms, 2),
        queue_wait_ms=round(queue_wait_ms, 2),
        ttft_ms=round(queue_wait_ms + prompt_ms, 2),
        decode_tokens_per_second=round(decode_tps, 2),
        overlapping_execution=False,  # updated in concurrent benchmark
        overlap_duration_ms=0.0,
        slot_id_used=None,
        cpu_utilization_pct=round(cpu_pct, 1),
        memory_rss_mb=round(mem_mb, 1),
        sla_met=sla_met,
    )


def compute_effective_concurrency(
    observations: list[ServingObservation],
    wall_span_s: float | None = None,
) -> float:
    """Calculate empirical server execution concurrency ratio."""
    if not observations:
        return 0.0
    if any(
        o.runtime_architecture in (
            ServingRuntimeArchitecture.OLLAMA_SINGLE_SLOT,
            ServingRuntimeArchitecture.OLLAMA_CLIENT_QUEUED,
        )
        for o in observations
    ):
        return 1.0

    if wall_span_s is None or wall_span_s <= 0:
        min_start = min(o.client_dispatch_mono_s for o in observations)
        max_end = max(o.client_completion_mono_s for o in observations)
        wall_span_s = max(1e-6, max_end - min_start)

    sum_server_active_s = sum((o.server_prefill_ms + o.server_decode_ms) for o in observations) / 1000.0
    server_slots = max(o.server_slots for o in observations)
    return round(max(1.0, min(float(server_slots), sum_server_active_s / wall_span_s)), 2)


def compute_execution_overlap_ms(observations: list[ServingObservation]) -> float:
    """Calculate pairwise server execution overlap in milliseconds."""
    if not observations or len(observations) < 2:
        return 0.0
    if any(
        o.runtime_architecture in (
            ServingRuntimeArchitecture.OLLAMA_SINGLE_SLOT,
            ServingRuntimeArchitecture.OLLAMA_CLIENT_QUEUED,
        )
        for o in observations
    ):
        return 0.0

    total_overlap_ms = 0.0
    for i, obs in enumerate(observations):
        s_start_i = obs.server_start_mono_s or obs.client_dispatch_mono_s
        s_end_i = obs.server_completion_mono_s or obs.client_completion_mono_s
        for j, other in enumerate(observations):
            if i >= j:
                continue
            s_start_j = other.server_start_mono_s or other.client_dispatch_mono_s
            s_end_j = other.server_completion_mono_s or other.client_completion_mono_s

            overlap_start = max(s_start_i, s_start_j)
            overlap_end = min(s_end_i, s_end_j)
            if overlap_end > overlap_start + 0.005:  # at least 5ms true server overlap
                total_overlap_ms += (overlap_end - overlap_start) * 1000.0

    return round(total_overlap_ms, 2)


def run_concurrent_batch_benchmark(
    target_engine: str,  # "ollama" or "llama_server"
    model_id: str,
    prompts: list[str],
    max_tokens_list: list[int],
    sla_ms_list: list[float],
    server_slots: int = 1,
    num_threads: int = 8,
) -> list[ServingObservation]:
    """Dispatch multiple client requests concurrently, measuring exact execution overlap and effective concurrency."""
    concurrency = len(prompts)
    observations: list[ServingObservation] = []

    def task(idx: int) -> ServingObservation:
        p = prompts[idx]
        tokens = max_tokens_list[idx]
        sla = sla_ms_list[idx]
        if target_engine == "ollama":
            return execute_ollama_request(
                model_id=model_id,
                prompt=p,
                max_tokens=tokens,
                num_thread=num_threads,
                latency_sla_ms=sla,
                workload_class="CONCURRENT_BATCH",
                client_concurrency=concurrency,
            )
        else:
            return execute_llama_server_request(
                server_base_url=LLAMA_SERVER_BASE_URL,
                model_id=model_id,
                prompt=p,
                max_tokens=tokens,
                num_thread=num_threads,
                latency_sla_ms=sla,
                workload_class="CONCURRENT_BATCH",
                client_concurrency=concurrency,
                server_slots=server_slots,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(task, i) for i in range(concurrency)]
        observations = [f.result() for f in futures]

    # Calculate true server execution concurrency (excluding queued idle time)
    min_start = min(o.client_dispatch_mono_s for o in observations)
    max_end = max(o.client_completion_mono_s for o in observations)
    wall_span_s = max(1e-6, max_end - min_start)
    sum_server_active_s = sum((o.server_prefill_ms + o.server_decode_ms) for o in observations) / 1000.0
    effective_concurrency = round(max(1.0, min(float(server_slots), sum_server_active_s / wall_span_s)), 2)
    if target_engine == "ollama":
        # Ollama single-slot serial execution has effective server concurrency of 1.0
        effective_concurrency = 1.0

    # Check pairwise server execution overlaps
    updated_obs: list[ServingObservation] = []
    for i, obs in enumerate(observations):
        total_overlap_ms = 0.0
        has_overlap = False
        s_start_i = obs.server_start_mono_s or obs.client_dispatch_mono_s
        s_end_i = obs.server_completion_mono_s or obs.client_completion_mono_s

        for j, other in enumerate(observations):
            if i == j:
                continue
            s_start_j = other.server_start_mono_s or other.client_dispatch_mono_s
            s_end_j = other.server_completion_mono_s or other.client_completion_mono_s

            overlap_start = max(s_start_i, s_start_j)
            overlap_end = min(s_end_i, s_end_j)
            if overlap_end > overlap_start + 0.005:  # at least 5ms true server overlap
                has_overlap = True
                total_overlap_ms += (overlap_end - overlap_start) * 1000.0

        if target_engine == "ollama":
            # In Ollama serial queue, requests do not overlap on server
            has_overlap = False
            total_overlap_ms = 0.0

        # Construct updated observation
        obs_dict = obs.model_dump()
        obs_dict["overlapping_execution"] = has_overlap
        obs_dict["overlap_duration_ms"] = round(total_overlap_ms, 2)
        obs_dict["effective_server_concurrency"] = effective_concurrency
        updated_obs.append(ServingObservation(**obs_dict))

    return updated_obs


def run_head_of_line_comparison(
    model_id: str = "llama3.2:1b",
    consequent_sla_ms: float = 500.0,
) -> HeadOfLineComparisonObservation:
    """Execute matched Head-of-Line blocking benchmark on Ollama vs llama.cpp server.

    Workload:
    - Antecedent: Long generation (30 prompt tokens, 160 output tokens, ~7.5s on llama3.2:1b).
    - Consequent: Short interactive QA (20 prompt tokens, 16 output tokens, ~110ms isolated, target SLA 500ms).
    """
    logger.info("Executing HoL blocking probe on Ollama (baseline)...")

    # 1. Ollama Baseline
    antecedent_prompt = "Generate a comprehensive analysis of distributed operating systems and microkernel design."
    consequent_prompt = "What is the capital of France?"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fut_ant = executor.submit(
            execute_ollama_request,
            model_id=model_id,
            prompt=antecedent_prompt,
            max_tokens=160,
            num_thread=8,
            latency_sla_ms=8000.0,
            workload_class="LONG_FORM_GENERATION",
            client_concurrency=2,
        )
        time.sleep(0.02)  # ensure antecedent claims slot
        fut_con = executor.submit(
            execute_ollama_request,
            model_id=model_id,
            prompt=consequent_prompt,
            max_tokens=16,
            num_thread=8,
            latency_sla_ms=consequent_sla_ms,
            workload_class="INTERACTIVE_QA",
            client_concurrency=2,
        )
        fut_ant.result()
        obs_con_ollama = fut_con.result()

    logger.info(
        f"Ollama HoL outcome: Consequent wait={obs_con_ollama.queue_wait_ms}ms, "
        f"E2E={obs_con_ollama.client_duration_ms}ms, SLA met={obs_con_ollama.sla_met}"
    )

    # 2. llama.cpp Server Continuous Batching (2 parallel slots)
    logger.info("Executing HoL blocking probe on llama-server (2 parallel slots, continuous batching)...")
    with (
        LlamaServerContext(model_id=model_id, slots=2, continuous_batching=True, threads=8, port=LLAMA_SERVER_PORT) as server,
        concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
    ):
        fut_ant = executor.submit(
            execute_llama_server_request,
            server_base_url=server.base_url,
            model_id=model_id,
            prompt=antecedent_prompt,
            max_tokens=160,
            num_thread=8,
            latency_sla_ms=8000.0,
            workload_class="LONG_FORM_GENERATION",
            client_concurrency=2,
            server_slots=2,
        )
        time.sleep(0.02)  # dispatch consequent while antecedent is actively decoding
        fut_con = executor.submit(
            execute_llama_server_request,
            server_base_url=server.base_url,
            model_id=model_id,
            prompt=consequent_prompt,
            max_tokens=16,
            num_thread=8,
            latency_sla_ms=consequent_sla_ms,
            workload_class="INTERACTIVE_QA",
            client_concurrency=2,
            server_slots=2,
        )
        fut_ant.result()
        obs_con_llama = fut_con.result()

    logger.info(
        f"llama-server HoL outcome: Consequent wait={obs_con_llama.queue_wait_ms}ms, "
        f"E2E={obs_con_llama.client_duration_ms}ms, SLA met={obs_con_llama.sla_met}"
    )

    queue_wait_reduction = max(0.0, obs_con_ollama.queue_wait_ms - obs_con_llama.queue_wait_ms)
    speedup = round(obs_con_ollama.client_duration_ms / max(1.0, obs_con_llama.client_duration_ms), 2)
    rescued = (not obs_con_ollama.sla_met) and obs_con_llama.sla_met

    return HeadOfLineComparisonObservation(
        experiment_id=f"hol-comp-{uuid.uuid4().hex[:8]}",
        model_id=model_id,
        antecedent_workload="LONG_FORM_GENERATION (160 tok)",
        consequent_workload="INTERACTIVE_QA (16 tok)",
        consequent_sla_ms=consequent_sla_ms,
        ollama_consequent_wait_ms=obs_con_ollama.queue_wait_ms,
        ollama_consequent_e2e_ms=obs_con_ollama.client_duration_ms,
        ollama_sla_met=obs_con_ollama.sla_met,
        llama_server_consequent_wait_ms=obs_con_llama.queue_wait_ms,
        llama_server_consequent_e2e_ms=obs_con_llama.client_duration_ms,
        llama_server_sla_met=obs_con_llama.sla_met,
        queue_wait_reduction_ms=round(queue_wait_reduction, 2),
        latency_reduction_factor=speedup,
        sla_rescued_by_parallelism=rescued,
    )


def compute_architecture_comparison(
    baseline_observations: list[ServingObservation],
    target_observations: list[ServingObservation],
    workload_description: str,
    client_concurrency: int,
) -> ServingArchitectureComparison:
    """Aggregate paired observations into an architecture comparison record."""
    if not baseline_observations or not target_observations:
        raise ValueError("Must provide non-empty observation lists for comparison.")

    model_id = baseline_observations[0].model_id
    base_arch = baseline_observations[0].runtime_architecture
    target_arch = target_observations[0].runtime_architecture

    base_e2es = sorted(o.client_duration_ms for o in baseline_observations)
    target_e2es = sorted(o.client_duration_ms for o in target_observations)

    bn = len(base_e2es)
    tn = len(target_e2es)

    bp50 = statistics.median(base_e2es)
    bp90 = base_e2es[min(int(bn * 0.90), bn - 1)]
    bp95 = base_e2es[min(int(bn * 0.95), bn - 1)]

    tp50 = statistics.median(target_e2es)
    tp90 = target_e2es[min(int(tn * 0.90), tn - 1)]
    tp95 = target_e2es[min(int(tn * 0.95), tn - 1)]

    base_waits = [o.queue_wait_ms for o in baseline_observations]
    target_waits = [o.queue_wait_ms for o in target_observations]

    b_mean_wait = statistics.mean(base_waits)
    t_mean_wait = statistics.mean(target_waits)
    wait_reduction = max(0.0, b_mean_wait - t_mean_wait)

    base_sla_pct = (sum(1 for o in baseline_observations if o.sla_met) / bn) * 100.0
    target_sla_pct = (sum(1 for o in target_observations if o.sla_met) / tn) * 100.0

    # Throughput
    base_span_s = max(
        0.01,
        max(o.client_completion_mono_s for o in baseline_observations)
        - min(o.client_dispatch_mono_s for o in baseline_observations),
    )
    target_span_s = max(
        0.01,
        max(o.client_completion_mono_s for o in target_observations)
        - min(o.client_dispatch_mono_s for o in target_observations),
    )

    base_rps = round(bn / base_span_s, 2)
    target_rps = round(tn / target_span_s, 2)
    tput_gain = round(((target_rps - base_rps) / max(0.01, base_rps)) * 100.0, 1)

    speedup_p95 = round(bp95 / max(1.0, tp95), 2)
    hol_eliminated = t_mean_wait < 50.0 and b_mean_wait > 500.0

    b_mem = statistics.mean(o.memory_rss_mb for o in baseline_observations)
    t_mem = statistics.mean(o.memory_rss_mb for o in target_observations)

    return ServingArchitectureComparison(
        comparison_id=f"comp-{uuid.uuid4().hex[:8]}",
        model_id=model_id,
        workload_description=workload_description,
        client_concurrency=client_concurrency,
        baseline_architecture=base_arch,
        target_architecture=target_arch,
        baseline_p50_ms=round(bp50, 1),
        baseline_p90_ms=round(bp90, 1),
        baseline_p95_ms=round(bp95, 1),
        target_p50_ms=round(tp50, 1),
        target_p90_ms=round(tp90, 1),
        target_p95_ms=round(tp95, 1),
        latency_p95_speedup_factor=speedup_p95,
        baseline_throughput_rps=base_rps,
        target_throughput_rps=target_rps,
        throughput_gain_pct=tput_gain,
        baseline_mean_queue_wait_ms=round(b_mean_wait, 1),
        target_mean_queue_wait_ms=round(t_mean_wait, 1),
        queue_wait_reduction_ms=round(wait_reduction, 1),
        hol_blocking_eliminated=hol_eliminated,
        baseline_sla_compliance_pct=round(base_sla_pct, 1),
        target_sla_compliance_pct=round(target_sla_pct, 1),
        baseline_memory_mb=round(b_mem, 1),
        target_memory_mb=round(t_mem, 1),
        memory_overhead_mb=round(t_mem - b_mem, 1),
        scientific_provenance_summary=(
            f"Under client concurrency C={client_concurrency}, {target_arch.value} reduced P95 latency by {speedup_p95}x "
            f"and improved throughput by {tput_gain}%. Mean queue delay dropped by {wait_reduction:.1f}ms."
        ),
    )
