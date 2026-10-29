"""Execution orchestrator for Session 24 hardware-aware runtime tuning benchmarks.

Executes:
- Stage A: CPU thread-count tuning (2, 4, 6, 8, 12, 14, 20, 28 threads across 3 models x 4 workloads).
- Stage B: Concurrency scaling (1, 2, 4 concurrent requests across key thread counts).
- Stage C: Context window controls (2048, 4096, 8192 context sizes).
- Incremental checkpointing, bounded 25-minute budget enforcement, and robust error handling.
"""

import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_ii_engine.benchmark.runtime_tuning_protocol import (
    RuntimeConfigurationSpec,
    RuntimeExperimentManifest,
    RuntimeObservation,
    WorkstationHardwareProfile,
)
from llm_ii_engine.benchmark.workload_suite import (
    TASK_INTERACTIVE_QA,
    TASK_LONG_CONTEXT,
    TASK_LONG_FORM,
    TASK_STRUCTURED_JSON,
    BenchmarkWorkloadTask,
)

BASE_URL = "http://127.0.0.1:11434"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

MODEL_METADATA: dict[str, dict[str, str]] = {
    "qwen2.5:0.5b": {
        "digest": "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67",
        "quantization": "Q4_K_M",
    },
    "llama3.2:1b": {
        "digest": "baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878",
        "quantization": "Q8_0",
    },
    "qwen2.5:1.5b": {
        "digest": "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b",
        "quantization": "Q4_K_M",
    },
}

STAGE_A_TASKS: list[BenchmarkWorkloadTask] = [
    TASK_INTERACTIVE_QA,
    TASK_STRUCTURED_JSON,
    TASK_LONG_FORM,
    TASK_LONG_CONTEXT,
]

STAGE_A_THREADS: list[int] = [2, 4, 6, 8, 12, 14, 20, 28]


def execute_stream_request(
    model_id: str,
    task: BenchmarkWorkloadTask,
    config: RuntimeConfigurationSpec,
    sequence_index: int,
    repetition_index: int,
    stage: str,
    timeout_sec: float = 45.0,
) -> RuntimeObservation:
    """Execute a single streaming generation request against Ollama and measure timings."""
    obs_id = f"obs-s24-{stage.lower()}-{sequence_index:04d}"
    meta = MODEL_METADATA.get(
        model_id, {"digest": "unknown", "quantization": "unknown"}
    )

    url = f"{BASE_URL}/api/generate"
    options: dict[str, Any] = {
        "num_thread": config.num_thread,
        "num_ctx": config.num_ctx,
        "num_predict": task.max_output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }
    if config.num_batch:
        options["num_batch"] = config.num_batch

    payload = json.dumps(
        {
            "model": model_id,
            "prompt": task.prompt,
            "stream": True,
            "options": options,
            "keep_alive": config.keep_alive,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    t_start = time.perf_counter()
    first_chunk_time: float | None = None
    final_payload: dict[str, Any] = {}
    error_msg: str | None = None
    status = "SUCCESS"

    try:
        with OPENER.open(req, timeout=timeout_sec) as resp:
            for line in resp:
                t_now = time.perf_counter()
                if first_chunk_time is None:
                    first_chunk_time = t_now
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

    t_end = time.perf_counter()
    client_e2e_ms = max(0.0, (t_end - t_start) * 1000.0)
    ttft_ms = (
        max(0.0, (first_chunk_time - t_start) * 1000.0)
        if first_chunk_time
        else None
    )

    prompt_tokens = int(final_payload.get("prompt_eval_count", 0))
    generated_tokens = int(final_payload.get("eval_count", 0))
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

    decode_tps = (
        (generated_tokens / (server_decode_ms / 1000.0))
        if server_decode_ms > 0
        else 0.0
    )
    client_tps = (
        (generated_tokens / (client_e2e_ms / 1000.0))
        if client_e2e_ms > 0
        else 0.0
    )

    return RuntimeObservation(
        observation_id=obs_id,
        timestamp_utc=datetime.now(UTC).isoformat(),
        model_id=model_id,
        model_digest=meta["digest"],
        quantization=meta["quantization"],
        requested_config=config,
        task_id=task.task_id,
        task_name=task.name,
        stage=stage,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        client_e2e_duration_ms=round(client_e2e_ms, 2),
        time_to_first_chunk_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
        server_load_duration_ms=round(server_load_ms, 2),
        server_prefill_duration_ms=round(server_prefill_ms, 2),
        server_decode_duration_ms=round(server_decode_ms, 2),
        decode_tokens_per_second=round(decode_tps, 2),
        client_tokens_per_second=round(client_tps, 2),
        sequence_index=sequence_index,
        repetition_index=repetition_index,
        status=status,
        error_message=error_msg,
    )


def run_session_24_benchmarks(
    output_dir: Path,
    max_budget_seconds: float = 1400.0,
) -> RuntimeExperimentManifest:
    """Execute complete Stage A, Stage B, and Stage C benchmarks within the time budget."""
    manifest_id = f"exp-manifest-s24-{int(time.time())}"
    start_time = time.perf_counter()
    observations: list[RuntimeObservation] = []
    seq_idx = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_obs_path = output_dir / "session_24_raw_observations.json"
    manifest_path = output_dir / "session_24_manifest.json"

    def checkpoint():
        with open(raw_obs_path, "w", encoding="utf-8") as f:
            json.dump([o.model_dump() for o in observations], f, indent=2)

    print("===================================================================")
    print("SESSION 24 — HARDWARE-AWARE RUNTIME OPTIMIZATION BENCHMARK")
    print(f"Max active budget: {max_budget_seconds:.1f}s (~{max_budget_seconds/60:.1f} min)")
    print("Workstation: Intel Core i7-14700 (8P + 12E, 28 threads)")
    print("===================================================================")

    # Check for existing checkpointed observations
    if raw_obs_path.exists():
        try:
            raw_data = json.loads(raw_obs_path.read_text(encoding="utf-8"))
            observations = [RuntimeObservation(**item) for item in raw_data]
            seq_idx = len(observations)
            print(f"[RESUME] Loaded {len(observations)} existing observations from checkpoint.")
        except Exception as e:
            print(f"[RESUME ERROR] Could not load checkpoint: {e}")
            observations = []

    # -------------------------------------------------------------
    # STAGE A: Thread-count tuning across 3 models x 8 thread counts
    # -------------------------------------------------------------
    stage_a_completed = len([o for o in observations if o.stage == "STAGE_A_THREADS"])
    if stage_a_completed >= 192:
        print(f"\n--- [STAGE A] CPU Thread-Count Tuning ALREADY COMPLETE ({stage_a_completed} observations) ---")
    else:
        print("\n--- [STAGE A] CPU Thread-Count Tuning ---")
        models = ["qwen2.5:0.5b", "llama3.2:1b", "qwen2.5:1.5b"]

        for model_id in models:
            # Check budget
            if time.perf_counter() - start_time > max_budget_seconds:
                print("[BUDGET REACHED] Stopping Stage A dispatches.")
                break

            print(f"\n[Model: {model_id}] Warming up...")
            # Warmup request
            warmup_cfg = RuntimeConfigurationSpec(num_thread=8)
            execute_stream_request(
                model_id=model_id,
                task=TASK_INTERACTIVE_QA,
                config=warmup_cfg,
                sequence_index=seq_idx,
                repetition_index=1,
                stage="WARMUP",
            )
            seq_idx += 1

            for threads in STAGE_A_THREADS:
                for task in STAGE_A_TASKS:
                    if time.perf_counter() - start_time > max_budget_seconds:
                        print("[BUDGET REACHED] Stopping early.")
                        break

                    for rep in range(1, 3):  # 2 measured repetitions
                        cfg = RuntimeConfigurationSpec(num_thread=threads)
                        obs = execute_stream_request(
                            model_id=model_id,
                            task=task,
                            config=cfg,
                            sequence_index=seq_idx,
                            repetition_index=rep,
                            stage="STAGE_A_THREADS",
                        )
                        observations.append(obs)
                        seq_idx += 1
                        checkpoint()
                        print(
                            f"  [{model_id} | {threads:2d} threads | {task.task_id[:20]} | rep {rep}] "
                            f"E2E: {obs.client_e2e_duration_ms:6.1f}ms | "
                            f"Prefill: {obs.server_prefill_duration_ms:5.1f}ms | "
                            f"Decode: {obs.server_decode_duration_ms:6.1f}ms ({obs.decode_tokens_per_second:5.1f} t/s)"
                        )

    # -------------------------------------------------------------
    # STAGE B: Concurrency Scaling
    # -------------------------------------------------------------
    print("\n--- [STAGE B] Concurrency Scaling ---")
    concurrency_levels = [1, 2, 4]
    concurrency_threads = [4, 8, 14]
    concurrency_models = ["qwen2.5:0.5b", "llama3.2:1b"]
    task_b = TASK_STRUCTURED_JSON

    for model_id in concurrency_models:
        for threads in concurrency_threads:
            for c_level in concurrency_levels:
                if time.perf_counter() - start_time > max_budget_seconds:
                    print("[BUDGET REACHED] Stopping Stage B.")
                    break

                for rep in range(1, 3):
                    cfg = RuntimeConfigurationSpec(
                        num_thread=threads, concurrency=c_level
                    )
                    t_batch_start = time.perf_counter()

                    batch_observations: list[RuntimeObservation] = []
                    with ThreadPoolExecutor(max_workers=c_level) as executor:
                        futures = [
                            executor.submit(
                                execute_stream_request,
                                model_id,
                                task_b,
                                cfg,
                                seq_idx + i,
                                rep,
                                "STAGE_B_CONCURRENCY",
                            )
                            for i in range(c_level)
                        ]
                        for f in futures:
                            batch_observations.append(f.result())

                    t_batch_end = time.perf_counter()
                    batch_wall_sec = max(0.001, t_batch_end - t_batch_start)
                    total_tokens = sum(o.generated_tokens for o in batch_observations)
                    aggregate_tps = round(total_tokens / batch_wall_sec, 2)

                    # Update aggregate throughput & queue estimates
                    for idx, o in enumerate(batch_observations):
                        o.aggregate_tokens_per_second = aggregate_tps
                        # In serialized single-slot server, request i queues behind prior requests
                        if idx > 0 and len(batch_observations) > 1:
                            prior_dur = sum(
                                b.server_prefill_duration_ms
                                + b.server_decode_duration_ms
                                for b in batch_observations[:idx]
                            )
                            o.queuing_delay_estimate_ms = round(prior_dur, 2)
                        observations.append(o)
                        seq_idx += 1

                    checkpoint()
                    e2es = [o.client_e2e_duration_ms for o in batch_observations]
                    print(
                        f"  [Conc {c_level} | {model_id} | {threads:2d} th | rep {rep}] "
                        f"Wall: {batch_wall_sec*1000:6.1f}ms | Aggregate: {aggregate_tps:5.1f} tok/s | "
                        f"E2E Min/Max: {min(e2es):.1f}ms / {max(e2es):.1f}ms"
                    )

    # -------------------------------------------------------------
    # STAGE C: Context Window Controls
    # -------------------------------------------------------------
    print("\n--- [STAGE C] Context Window Controls ---")
    context_sizes = [2048, 4096, 8192]
    task_c = TASK_LONG_CONTEXT

    for model_id in ["qwen2.5:0.5b", "llama3.2:1b"]:
        for ctx_size in context_sizes:
            if time.perf_counter() - start_time > max_budget_seconds:
                print("[BUDGET REACHED] Stopping Stage C.")
                break

            for rep in range(1, 3):
                cfg = RuntimeConfigurationSpec(num_thread=8, num_ctx=ctx_size)
                obs = execute_stream_request(
                    model_id=model_id,
                    task=task_c,
                    config=cfg,
                    sequence_index=seq_idx,
                    repetition_index=rep,
                    stage="STAGE_C_CONTEXT",
                )
                observations.append(obs)
                seq_idx += 1
                checkpoint()
                print(
                    f"  [Context {ctx_size} | {model_id} | rep {rep}] "
                    f"E2E: {obs.client_e2e_duration_ms:6.1f}ms | "
                    f"Prefill: {obs.server_prefill_duration_ms:5.1f}ms | "
                    f"Decode: {obs.server_decode_duration_ms:6.1f}ms ({obs.decode_tokens_per_second:5.1f} t/s)"
                )

    total_time_sec = round(time.perf_counter() - start_time, 2)
    print("\n===================================================================")
    print(f"BENCHMARK COMPLETED in {total_time_sec:.1f}s ({total_time_sec/60:.2f} min)")
    print(f"Total observations recorded: {len(observations)}")
    print("===================================================================")

    # Compute statistical summaries
    summary_by_thread = compute_thread_summary(observations)
    summary_by_concurrency = compute_concurrency_summary(observations)
    summary_by_context = compute_context_summary(observations)

    manifest = RuntimeExperimentManifest(
        manifest_id=manifest_id,
        session_id="session_24",
        created_at_utc=datetime.now(UTC).isoformat(),
        hardware_profile=WorkstationHardwareProfile(),
        total_active_duration_seconds=total_time_sec,
        observations=observations,
        summary_by_thread_count=summary_by_thread,
        summary_by_concurrency=summary_by_concurrency,
        summary_by_context=summary_by_context,
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    print(f"Manifest written to: {manifest_path}")
    return manifest


def compute_thread_summary(
    observations: list[RuntimeObservation],
) -> dict[str, Any]:
    """Compute aggregated metrics by model and thread count for Stage A."""
    stage_a = [o for o in observations if o.stage == "STAGE_A_THREADS"]
    summary: dict[str, Any] = {}

    models = sorted(list({o.model_id for o in stage_a}))
    for m in models:
        summary[m] = {}
        m_obs = [o for o in stage_a if o.model_id == m]
        threads = sorted(list({o.requested_config.num_thread for o in m_obs}))

        for th in threads:
            th_obs = [o for o in m_obs if o.requested_config.num_thread == th]
            e2e_vals = [o.client_e2e_duration_ms for o in th_obs]
            decode_tps_vals = [
                o.decode_tokens_per_second
                for o in th_obs
                if o.decode_tokens_per_second > 0
            ]
            prefill_vals = [
                o.server_prefill_duration_ms
                for o in th_obs
                if o.server_prefill_duration_ms > 0
            ]

            summary[m][str(th)] = {
                "sample_count": len(th_obs),
                "median_e2e_ms": round(statistics.median(e2e_vals), 2)
                if e2e_vals
                else 0.0,
                "mean_e2e_ms": round(statistics.mean(e2e_vals), 2)
                if e2e_vals
                else 0.0,
                "median_decode_tps": round(
                    statistics.median(decode_tps_vals), 2
                )
                if decode_tps_vals
                else 0.0,
                "mean_decode_tps": round(statistics.mean(decode_tps_vals), 2)
                if decode_tps_vals
                else 0.0,
                "median_prefill_ms": round(statistics.median(prefill_vals), 2)
                if prefill_vals
                else 0.0,
            }
    return summary


def compute_concurrency_summary(
    observations: list[RuntimeObservation],
) -> dict[str, Any]:
    """Compute aggregated metrics by concurrency level for Stage B."""
    stage_b = [o for o in observations if o.stage == "STAGE_B_CONCURRENCY"]
    summary: dict[str, Any] = {}

    models = sorted(list({o.model_id for o in stage_b}))
    for m in models:
        summary[m] = {}
        m_obs = [o for o in stage_b if o.model_id == m]
        c_levels = sorted(list({o.requested_config.concurrency for o in m_obs}))

        for c in c_levels:
            c_obs = [o for o in m_obs if o.requested_config.concurrency == c]
            e2es = [o.client_e2e_duration_ms for o in c_obs]
            agg_tps = [
                o.aggregate_tokens_per_second
                for o in c_obs
                if o.aggregate_tokens_per_second
            ]
            queue_delays = [
                o.queuing_delay_estimate_ms
                for o in c_obs
                if o.queuing_delay_estimate_ms > 0
            ]

            summary[m][str(c)] = {
                "sample_count": len(c_obs),
                "median_e2e_ms": round(statistics.median(e2es), 2)
                if e2es
                else 0.0,
                "p95_e2e_ms": round(
                    sorted(e2es)[int(len(e2es) * 0.95)]
                    if len(e2es) > 1
                    else e2es[0],
                    2,
                )
                if e2es
                else 0.0,
                "mean_aggregate_tps": round(statistics.mean(agg_tps), 2)
                if agg_tps
                else 0.0,
                "median_queuing_delay_ms": round(
                    statistics.median(queue_delays), 2
                )
                if queue_delays
                else 0.0,
            }
    return summary


def compute_context_summary(
    observations: list[RuntimeObservation],
) -> dict[str, Any]:
    """Compute aggregated metrics by context size for Stage C."""
    stage_c = [o for o in observations if o.stage == "STAGE_C_CONTEXT"]
    summary: dict[str, Any] = {}

    models = sorted(list({o.model_id for o in stage_c}))
    for m in models:
        summary[m] = {}
        m_obs = [o for o in stage_c if o.model_id == m]
        ctx_sizes = sorted(list({o.requested_config.num_ctx for o in m_obs}))

        for ctx in ctx_sizes:
            ctx_obs = [o for o in m_obs if o.requested_config.num_ctx == ctx]
            prefill_vals = [o.server_prefill_duration_ms for o in ctx_obs]
            e2e_vals = [o.client_e2e_duration_ms for o in ctx_obs]

            summary[m][str(ctx)] = {
                "sample_count": len(ctx_obs),
                "median_prefill_ms": round(
                    statistics.median(prefill_vals), 2
                )
                if prefill_vals
                else 0.0,
                "median_e2e_ms": round(statistics.median(e2e_vals), 2)
                if e2e_vals
                else 0.0,
            }
    return summary


if __name__ == "__main__":
    out_dir = Path("storage/experiments")
    run_session_24_benchmarks(out_dir)
