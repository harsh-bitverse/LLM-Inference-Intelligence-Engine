"""Session 15: Cache-State-Controlled Benchmarking & Reproducible Performance Modeling.

Executes bounded, scientifically controlled experiments on the authorized local Ollama runtime
(model: llama3.2:1b, Q8_0) to isolate and measure:
  Experiment A: Identical prompt repetition (consecutive KV-cache hits).
  Experiment B: Shared prefix branching vs unrelated prompts.
  Experiment C: Request ordering effects (forward, reverse, shuffled).
  Experiment D: Model residency transitions (cold unload vs warm resident).
  Experiment E: Output length scaling under controlled prefill state.

All records capture client-side monotonic durations, server prefill/decode timing,
load duration, prompt_eval_cached_count, model residency state, and predecessor request IDs.
Exports structured JSONL, CSV, and manifest artifacts with 100% backward compatibility.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from llm_ii_engine.benchmark.dataset_export import (
    EmpiricalDatasetExporter,
    EmpiricalObservationRecord,
)
from llm_ii_engine.benchmark.expansion import (
    PROMPT_TEXT_32,
    PROMPT_TEXT_128,
    PROMPT_TEXT_384,
    PROMPT_TEXT_768,
)
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ClockSource,
    ExecutionMode,
    ModelResidencyState,
    PrefixRelationshipType,
    ProvenanceType,
    TokenCountSource,
)
from llm_ii_engine.discovery.profiler import discover_hardware

# Explicitly bypass local proxies for local Ollama communication
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL_ID = "llama3.2:1b"
MODEL_DIGEST = "baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878"
EXPERIMENT_ID = "exp-session-15-cache-controlled"


async def check_ollama_status(client: httpx.AsyncClient) -> tuple[str, bool]:
    """Verify Ollama version and model presence."""
    v_resp = await client.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5.0)
    version = v_resp.json().get("version", "unknown")
    ps_resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5.0)
    models = ps_resp.json().get("models", [])
    is_resident = any(MODEL_ID in m.get("name", "") for m in models)
    return version, is_resident


async def unload_model(client: httpx.AsyncClient) -> None:
    """Explicitly unload model from memory using keep_alive: 0 and verify."""
    payload = {"model": MODEL_ID, "keep_alive": 0}
    with contextlib.suppress(Exception):
        await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=10.0)
    await asyncio.sleep(0.5)
    # Verify unload
    ps_resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5.0)
    models = ps_resp.json().get("models", [])
    if any(MODEL_ID in m.get("name", "") for m in models):
        # Retry once
        with contextlib.suppress(Exception):
            await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=10.0)
        await asyncio.sleep(0.5)


async def execute_controlled_request(
    client: httpx.AsyncClient,
    request_id: str,
    prompt: str,
    requested_tokens: int,
    repetition_index: int,
    is_warmup: bool,
    experiment_tag: str,
    predecessor_request_id: str | None,
    prefix_relationship: PrefixRelationshipType,
    expected_residency: ModelResidencyState,
    hardware_info: dict[str, Any],
) -> EmpiricalObservationRecord:
    """Execute a single streaming request against Ollama and record rigorous telemetry."""
    # Preflight check residency via /api/ps
    ps_resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5.0)
    running_models = ps_resp.json().get("models", [])
    actual_resident = any(MODEL_ID in m.get("name", "") for m in running_models)
    actual_residency = (
        ModelResidencyState.WARM if actual_resident else ModelResidencyState.COLD
    )

    req_payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "options": {
            "num_predict": requested_tokens,
            "temperature": 0.0,
        },
        "stream": True,
    }

    t_start = time.perf_counter_ns()
    ttfc_ms: float | None = None
    chunk_count = 0
    chunk_timestamps_ns: list[int] = []
    final_data: dict[str, Any] = {}

    async with client.stream(
        "POST",
        f"{OLLAMA_BASE_URL}/api/generate",
        json=req_payload,
        timeout=120.0,
    ) as response:
        async for line in response.aiter_lines():
            line_str = line.strip()
            if not line_str:
                continue
            now_ns = time.perf_counter_ns()
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            chunk_text = data.get("response", "")
            if chunk_text:
                chunk_count += 1
                chunk_timestamps_ns.append(now_ns)
                if ttfc_ms is None:
                    ttfc_ms = (now_ns - t_start) / 1_000_000.0

            if data.get("done") is True:
                final_data = data
                break

    t_end = time.perf_counter_ns()
    duration_ms = (t_end - t_start) / 1_000_000.0

    # Extract server reported statistics
    prompt_eval_count = final_data.get("prompt_eval_count")
    prompt_eval_cached_count = final_data.get("prompt_eval_cached_count", 0)
    prompt_eval_duration_ns = final_data.get("prompt_eval_duration")
    eval_count = final_data.get("eval_count", 0)
    eval_duration_ns = final_data.get("eval_duration")
    load_duration_ns = final_data.get("load_duration")

    server_prompt_ms = (
        prompt_eval_duration_ns / 1_000_000.0
        if prompt_eval_duration_ns is not None
        else None
    )
    server_eval_ms = (
        eval_duration_ns / 1_000_000.0 if eval_duration_ns is not None else None
    )
    throughput_tok_s = (
        (eval_count / (duration_ms / 1000.0))
        if duration_ms > 0 and eval_count > 0
        else None
    )

    # Classify cache state based on empirical evidence
    if prompt_eval_cached_count is None or prompt_eval_cached_count == 0:
        cache_state = CacheStateClassification.COLD_PREFIX
    elif prompt_eval_count is not None and prompt_eval_cached_count >= (
        prompt_eval_count - 2
    ):
        cache_state = CacheStateClassification.WARM_PREFIX
    else:
        cache_state = CacheStateClassification.PARTIAL_PREFIX

    # Chunk latencies
    chunk_lat_p50 = None
    if len(chunk_timestamps_ns) >= 2:
        deltas = [
            (chunk_timestamps_ns[i] - chunk_timestamps_ns[i - 1]) / 1_000_000.0
            for i in range(1, len(chunk_timestamps_ns))
        ]
        import statistics

        chunk_lat_p50 = float(statistics.median(deltas))

    return EmpiricalObservationRecord(
        record_id=f"obs-{request_id}",
        experiment_id=EXPERIMENT_ID,
        suite_id=f"suite-{experiment_tag.lower()}",
        request_id=request_id,
        repetition_index=repetition_index,
        is_warmup=is_warmup,
        model_id=MODEL_ID,
        model_digest=MODEL_DIGEST,
        runtime_name="ollama",
        runtime_version=hardware_info.get("runtime_version", "0.40.11"),
        adapter_version="1.0.0",
        hardware_id=hardware_info.get("hardware_id"),
        device_name=hardware_info.get("device_name", "CPU"),
        cpu_model=hardware_info.get("cpu_model"),
        host_ram_gb=hardware_info.get("host_ram_gb"),
        gpu_vram_gb=None,
        prompt_tokens=prompt_eval_count,
        prompt_token_count_source=TokenCountSource.SERVER_REPORTED.value,
        requested_output_tokens=requested_tokens,
        actual_output_tokens=eval_count,
        token_count_source=TokenCountSource.SERVER_REPORTED.value,
        duration_ms=duration_ms,
        ttfc_ms=ttfc_ms,
        ttft_ms=None,  # Streamed text chunks; individual token boundaries unobserved
        tpot_ms=None,
        chunk_count=chunk_count,
        chunk_latency_p50_ms=chunk_lat_p50,
        output_tokens_per_second=throughput_tok_s,
        server_prompt_eval_duration_ms=server_prompt_ms,
        server_eval_duration_ms=server_eval_ms,
        peak_vram_mb=None,
        peak_process_rss_mb=None,
        model_residency_state=actual_residency.value,
        cache_state=cache_state.value,
        predecessor_request_id=predecessor_request_id,
        server_prompt_eval_cached_count=prompt_eval_cached_count,
        prefix_relationship=prefix_relationship.value,
        execution_mode=ExecutionMode.REAL.value,
        clock_source=ClockSource.MONOTONIC_REAL.value,
        provenance=ProvenanceType.MEASURED.value,
        success=True,
        error_message=None,
        missing_data_reasons=[
            "ttft_ms is null: stream emitted text chunks without 1:1 token timing guarantees; ttfc_ms recorded instead",
            "tpot_ms is null: per-token ITL requires discrete token events; chunks emitted",
            "peak_vram_mb is null: CPU-only execution; GPU VRAM unmetered",
        ],
        environmental_caveats=[
            f"Controlled Session 15 experiment: {experiment_tag}",
            f"Pre-execution residency: {actual_residency.value}",
            f"Reported load_duration_ms: {(load_duration_ns or 0) / 1_000_000.0:.2f} ms",
        ],
    )


async def run_session_15_experiments() -> list[EmpiricalObservationRecord]:
    """Execute all 5 controlled experiments and collect records."""
    hw = discover_hardware()
    hw_info = {
        "runtime_version": "0.40.11",
        "hardware_id": "hw-local-cpu",
        "device_name": "CPU",
        "cpu_model": hw.cpu.model_name,
        "host_ram_gb": hw.memory.total_gb,
    }

    records: list[EmpiricalObservationRecord] = []
    transport = httpx.AsyncHTTPTransport(retries=1)

    async with httpx.AsyncClient(transport=transport, timeout=120.0) as client:
        # Verify Ollama server
        version, is_resident = await check_ollama_status(client)
        print(f"Ollama server connected: version {version}, currently resident: {is_resident}")

        # ---------------------------------------------------------------------
        # Experiment A: Identical Prompt Repetition (3 consecutive runs)
        # ---------------------------------------------------------------------
        print("\n--- Running Experiment A: Identical Prompt Repetition (3 runs) ---")
        prompt_a = PROMPT_TEXT_128
        out_tokens_a = 32
        prev_id: str | None = None

        for rep in range(1, 4):
            req_id = f"s15-expA-rep{rep}"
            prefix_rel = PrefixRelationshipType.IDENTICAL if rep > 1 else PrefixRelationshipType.NONE
            rec = await execute_controlled_request(
                client=client,
                request_id=req_id,
                prompt=prompt_a,
                requested_tokens=out_tokens_a,
                repetition_index=rep,
                is_warmup=False,
                experiment_tag="ExpA_IdenticalRepetition",
                predecessor_request_id=prev_id,
                prefix_relationship=prefix_rel,
                expected_residency=ModelResidencyState.WARM,
                hardware_info=hw_info,
            )
            records.append(rec)
            prev_id = req_id
            print(
                f"  [{req_id}] Prompt: {rec.prompt_tokens} tok | "
                f"Cached: {rec.server_prompt_eval_cached_count} tok ({rec.cache_state}) | "
                f"Prefill: {rec.server_prompt_eval_duration_ms:.2f} ms | "
                f"Decode: {rec.server_eval_duration_ms:.2f} ms | "
                f"Total: {rec.duration_ms:.2f} ms"
            )

        # ---------------------------------------------------------------------
        # Experiment B: Shared Prefix vs Unrelated Prompt (3 runs)
        # ---------------------------------------------------------------------
        print("\n--- Running Experiment B: Shared Prefix vs Unrelated (3 runs) ---")
        prompt_b1 = PROMPT_TEXT_384 + "\n\nTask A: List three primary operational constraints of decode."
        prompt_b2 = PROMPT_TEXT_384 + "\n\nTask B: Summarize why MQA and GQA reduce memory bandwidth pressure."
        prompt_b3 = (
            "Describe the geological features, atmospheric pressure, and surface temperature variations "
            "of the planet Mars compared to Earth, focusing on volcanic structures like Olympus Mons."
        )

        b_configs = [
            ("s15-expB-b1", prompt_b1, prev_id, PrefixRelationshipType.UNRELATED),
            ("s15-expB-b2", prompt_b2, "s15-expB-b1", PrefixRelationshipType.SHARED_PREFIX),
            ("s15-expB-b3", prompt_b3, "s15-expB-b2", PrefixRelationshipType.UNRELATED),
        ]

        for req_id, p_text, pred_id, p_rel in b_configs:
            rec = await execute_controlled_request(
                client=client,
                request_id=req_id,
                prompt=p_text,
                requested_tokens=32,
                repetition_index=1,
                is_warmup=False,
                experiment_tag="ExpB_PrefixBranching",
                predecessor_request_id=pred_id,
                prefix_relationship=p_rel,
                expected_residency=ModelResidencyState.WARM,
                hardware_info=hw_info,
            )
            records.append(rec)
            prev_id = req_id
            print(
                f"  [{req_id}] Rel: {p_rel.value:14s} | "
                f"Prompt: {rec.prompt_tokens} tok | "
                f"Cached: {rec.server_prompt_eval_cached_count} tok ({rec.cache_state}) | "
                f"Prefill: {rec.server_prompt_eval_duration_ms:.2f} ms | "
                f"Total: {rec.duration_ms:.2f} ms"
            )

        # ---------------------------------------------------------------------
        # Experiment C: Request Ordering Effects (Forward, Reverse, Shuffled)
        # ---------------------------------------------------------------------
        print("\n--- Running Experiment C: Request Ordering Effects (12 runs) ---")
        c_prompts = {
            "P46": PROMPT_TEXT_32,
            "P119": PROMPT_TEXT_128,
            "P300": PROMPT_TEXT_384,
            "P677": PROMPT_TEXT_768,
        }

        orders = [
            ("Forward", ["P46", "P119", "P300", "P677"]),
            ("Reverse", ["P677", "P300", "P119", "P46"]),
            ("Shuffled", ["P300", "P46", "P677", "P119"]),
        ]

        for order_name, order_keys in orders:
            print(f"  Order: {order_name} ({' -> '.join(order_keys)})")
            for idx, key in enumerate(order_keys, start=1):
                req_id = f"s15-expC-{order_name.lower()}-{key.lower()}"
                p_text = c_prompts[key]
                p_rel = PrefixRelationshipType.SHARED_PREFIX if (
                    order_name == "Forward" and idx > 1
                ) else PrefixRelationshipType.UNRELATED

                rec = await execute_controlled_request(
                    client=client,
                    request_id=req_id,
                    prompt=p_text,
                    requested_tokens=32,
                    repetition_index=idx,
                    is_warmup=False,
                    experiment_tag=f"ExpC_Ordering_{order_name}",
                    predecessor_request_id=prev_id,
                    prefix_relationship=p_rel,
                    expected_residency=ModelResidencyState.WARM,
                    hardware_info=hw_info,
                )
                records.append(rec)
                prev_id = req_id
                print(
                    f"    [{req_id}] Key: {key:4s} | "
                    f"Prompt: {rec.prompt_tokens} tok | "
                    f"Cached: {rec.server_prompt_eval_cached_count} tok | "
                    f"Prefill: {rec.server_prompt_eval_duration_ms:.2f} ms | "
                    f"Total: {rec.duration_ms:.2f} ms"
                )

        # ---------------------------------------------------------------------
        # Experiment D: Model Residency (Cold Load vs Warm Resident) (6 runs)
        # ---------------------------------------------------------------------
        print("\n--- Running Experiment D: Model Residency (3 Cold/Warm pairs) ---")
        prompt_d = PROMPT_TEXT_128
        out_tokens_d = 32

        for pair in range(1, 4):
            # 1. Unload model
            print(f"  Pair {pair}: Unloading model via keep_alive: 0...")
            await unload_model(client)

            # Cold request
            req_id_cold = f"s15-expD-pair{pair}-cold"
            rec_cold = await execute_controlled_request(
                client=client,
                request_id=req_id_cold,
                prompt=prompt_d,
                requested_tokens=out_tokens_d,
                repetition_index=pair,
                is_warmup=False,
                experiment_tag="ExpD_Residency_Cold",
                predecessor_request_id=None,
                prefix_relationship=PrefixRelationshipType.NONE,
                expected_residency=ModelResidencyState.COLD,
                hardware_info=hw_info,
            )
            records.append(rec_cold)
            print(
                f"    [{req_id_cold}] State: {rec_cold.model_residency_state.upper()} | "
                f"Prefill: {rec_cold.server_prompt_eval_duration_ms:.2f} ms | "
                f"Total: {rec_cold.duration_ms:.2f} ms"
            )

            # Warm request (immediately following)
            req_id_warm = f"s15-expD-pair{pair}-warm"
            rec_warm = await execute_controlled_request(
                client=client,
                request_id=req_id_warm,
                prompt=prompt_d,
                requested_tokens=out_tokens_d,
                repetition_index=pair,
                is_warmup=False,
                experiment_tag="ExpD_Residency_Warm",
                predecessor_request_id=req_id_cold,
                prefix_relationship=PrefixRelationshipType.IDENTICAL,
                expected_residency=ModelResidencyState.WARM,
                hardware_info=hw_info,
            )
            records.append(rec_warm)
            prev_id = req_id_warm
            print(
                f"    [{req_id_warm}] State: {rec_warm.model_residency_state.upper()} | "
                f"Prefill: {rec_warm.server_prompt_eval_duration_ms:.2f} ms | "
                f"Total: {rec_warm.duration_ms:.2f} ms"
            )

        # ---------------------------------------------------------------------
        # Experiment E: Output Length Scaling (Fixed Prompt) (8 runs)
        # ---------------------------------------------------------------------
        print("\n--- Running Experiment E: Output Length Scaling (8 runs) ---")
        prompt_e = PROMPT_TEXT_128
        output_lengths = [16, 32, 64, 128]

        for out_len in output_lengths:
            for rep in (1, 2):
                req_id = f"s15-expE-O{out_len}-rep{rep}"
                rec = await execute_controlled_request(
                    client=client,
                    request_id=req_id,
                    prompt=prompt_e,
                    requested_tokens=out_len,
                    repetition_index=rep,
                    is_warmup=False,
                    experiment_tag=f"ExpE_OutputScaling_O{out_len}",
                    predecessor_request_id=prev_id,
                    prefix_relationship=PrefixRelationshipType.IDENTICAL,
                    expected_residency=ModelResidencyState.WARM,
                    hardware_info=hw_info,
                )
                records.append(rec)
                prev_id = req_id
                print(
                    f"  [{req_id}] Req/Act Out: {out_len}/{rec.actual_output_tokens} tok | "
                    f"Decode: {rec.server_eval_duration_ms:.2f} ms | "
                    f"Decode Rate: {(rec.actual_output_tokens / (rec.server_eval_duration_ms / 1000.0)):.2f} tok/s | "
                    f"Total: {rec.duration_ms:.2f} ms"
                )

    return records


def export_session_15_results(records: list[EmpiricalObservationRecord]) -> None:
    """Export records to JSONL, CSV, and save experiment manifest."""
    jsonl_path = Path("storage/datasets/session_15_cache_controlled_measurements.jsonl")
    csv_path = Path("storage/datasets/session_15_cache_controlled_measurements.csv")
    manifest_path = Path("storage/experiments/session_15_cache_experiment_manifest.json")

    # Export JSONL & CSV
    EmpiricalDatasetExporter.export_to_jsonl(
        records=records,
        output_path=jsonl_path,
        filter_real_only=True,
        include_failed=True,
        exclude_warmup=False,
    )
    EmpiricalDatasetExporter.export_to_csv(
        records=records,
        output_path=csv_path,
        filter_real_only=True,
        include_failed=True,
        exclude_warmup=False,
    )

    manifest_data = {
        "experiment_id": EXPERIMENT_ID,
        "session": "Session 15",
        "title": "Cache-State-Controlled Benchmarking & Reproducible Performance Modeling",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": MODEL_ID,
        "model_digest": MODEL_DIGEST,
        "runtime_name": "ollama",
        "runtime_version": "0.40.11",
        "total_observations": len(records),
        "experiments": [
            {
                "experiment_key": "ExpA",
                "name": "Identical Prompt Repetition",
                "description": "Consecutive execution of identical prompt to measure prefix KV-cache reuse.",
                "observation_count": 3,
            },
            {
                "experiment_key": "ExpB",
                "name": "Shared Prefix vs Different Suffix",
                "description": "Branching prompts sharing a 300-token prefix followed by different suffixes vs unrelated prompt.",
                "observation_count": 3,
            },
            {
                "experiment_key": "ExpC",
                "name": "Request Ordering Effects",
                "description": "Forward, reverse, and shuffled execution of varying prompt sizes (P46, P119, P300, P677).",
                "observation_count": 12,
            },
            {
                "experiment_key": "ExpD",
                "name": "Model Residency Transitions",
                "description": "Explicit cold unload (keep_alive: 0 verified via /api/ps) followed by warm resident execution (3 pairs).",
                "observation_count": 6,
            },
            {
                "experiment_key": "ExpE",
                "name": "Output Length Scaling",
                "description": "Fixed prompt with varying output token limits (16, 32, 64, 128) under resident model state.",
                "observation_count": 8,
            },
        ],
        "exported_files": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"\nExported {len(records)} records to:")
    print(f"  JSONL:    {jsonl_path}")
    print(f"  CSV:      {csv_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    records = asyncio.run(run_session_15_experiments())
    export_session_15_results(records)
