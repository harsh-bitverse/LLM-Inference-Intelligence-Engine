"""Session 22 Quality Benchmark Runner.

Executes the expanded objective quality suite (Development and Selection splits),
captures granular per-category quality metrics and variability, exports datasets,
and prepares calibration profiles for multi-model recommendations.
"""

import contextlib
import csv
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llm_ii_engine.benchmark.expanded_quality_suite import (
    DEV_TASKS,
    SELECTION_TASKS,
    QualityBenchmarkTask,
    QualityTaskCategory,
    compute_suite_hashes,
    evaluate_quality_task,
)
from llm_ii_engine.discovery.model_catalog import (
    CANDIDATE_MODELS,
    fetch_installed_ollama_models,
)


def run_benchmark_pass(
    client: httpx.Client,
    model_id: str,
    installed_dict: dict[str, Any],
    tasks: list[QualityBenchmarkTask],
    repetition_index: int,
    exp_id: str,
    obs_counter: int,
) -> tuple[list[dict[str, Any]], int]:
    """Execute a sequence of tasks against a single model and return observation records."""
    model_meta = installed_dict[model_id]
    catalog_spec = next((m for m in CANDIDATE_MODELS if m.model_id == model_id), None)
    family = catalog_spec.model_family if catalog_spec else "unknown"
    params = catalog_spec.parameter_count_billion if catalog_spec else 1.0
    quant = model_meta.get("details", {}).get("quantization_level", "unknown")
    digest = model_meta.get("digest", "")

    records: list[dict[str, Any]] = []

    # Warmup request before task block
    with (
        httpx.Client(base_url="http://127.0.0.1:11434", timeout=30.0, trust_env=False) as warmup_client,
        contextlib.suppress(Exception),
    ):
        warmup_client.post(
            "/api/generate",
            json={
                "model": model_id,
                "prompt": "Warmup request to ensure resident model weights in memory.",
                "options": {"num_predict": 4, "temperature": 0.0},
                "stream": False,
            },
        )

    for task in tasks:
        payload = {
            "model": model_id,
            "prompt": task.prompt,
            "options": {
                "num_predict": task.max_output_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "seed": 42,
            },
            "stream": False,
        }

        t0 = time.perf_counter()
        try:
            resp = client.post("/api/generate", json=payload)
            t1 = time.perf_counter()
            client_e2e_ms = round((t1 - t0) * 1000.0, 3)

            if resp.status_code == 200:
                raw_data = resp.json()
                text = raw_data.get("response", "")
                prompt_tokens = raw_data.get("prompt_eval_count", 0)
                output_tokens = raw_data.get("eval_count", 0)
                server_total_ms = round(raw_data.get("total_duration", 0) / 1e6, 3)
                server_load_ms = round(raw_data.get("load_duration", 0) / 1e6, 3)
                server_prefill_ms = round(raw_data.get("prompt_eval_duration", 0) / 1e6, 3)
                server_decode_ms = round(raw_data.get("eval_duration", 0) / 1e6, 3)

                raw_decode_tps = (
                    round(output_tokens / (server_decode_ms / 1000.0), 2)
                    if server_decode_ms > 0
                    else 0.0
                )
                char_count = len(text)
                sec = client_e2e_ms / 1000.0
                chars_per_sec = round(char_count / sec, 2) if sec > 0 else 0.0

                eval_result = evaluate_quality_task(task, text)
            else:
                text = ""
                prompt_tokens = 0
                output_tokens = 0
                server_total_ms = 0.0
                server_load_ms = 0.0
                server_prefill_ms = 0.0
                server_decode_ms = 0.0
                raw_decode_tps = 0.0
                char_count = 0
                chars_per_sec = 0.0
                eval_result = evaluate_quality_task(task, "")
        except Exception as e:
            t1 = time.perf_counter()
            client_e2e_ms = round((t1 - t0) * 1000.0, 3)
            text = ""
            prompt_tokens = 0
            output_tokens = 0
            server_total_ms = 0.0
            server_load_ms = 0.0
            server_prefill_ms = 0.0
            server_decode_ms = 0.0
            raw_decode_tps = 0.0
            char_count = 0
            chars_per_sec = 0.0
            eval_result = evaluate_quality_task(task, "")
            eval_result = eval_result.model_copy(update={"evaluation_notes": f"Network/Inference error: {e}"})

        record = {
            "observation_id": f"obs-{exp_id}-{obs_counter:04d}",
            "experiment_id": exp_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "model_id": model_id,
            "model_family": family,
            "parameter_count_billion": params,
            "quantization": quant,
            "model_digest": digest[:16],
            "task_id": task.task_id,
            "task_name": task.name,
            "category": task.category.value,
            "split": task.split.value,
            "difficulty": task.difficulty,
            "prompt_hash": task.prompt_hash,
            "repetition_index": repetition_index,
            "client_e2e_duration_ms": client_e2e_ms,
            "server_prefill_duration_ms": server_prefill_ms,
            "server_decode_duration_ms": server_decode_ms,
            "server_load_duration_ms": server_load_ms,
            "server_total_duration_ms": server_total_ms,
            "raw_prompt_tokens": prompt_tokens,
            "raw_output_tokens": output_tokens,
            "raw_decode_tokens_per_second": raw_decode_tps,
            "characters_generated": char_count,
            "characters_per_second": chars_per_sec,
            "format_compliant": eval_result.format_compliant,
            "task_success": eval_result.task_success,
            "correctness_score": eval_result.correctness_score,
            "evaluation_method": eval_result.evaluation_method,
            "evaluation_notes": eval_result.evaluation_notes,
            "response_snippet": text[:150],
        }
        records.append(record)
        obs_counter += 1

    return records, obs_counter


def compute_category_summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate quality and performance metrics grouped by model and category."""
    models = ["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"]
    categories = [c.value for c in QualityTaskCategory]

    summary: dict[str, Any] = {}

    for m_id in models:
        m_obs = [o for o in observations if o["model_id"] == m_id]
        if not m_obs:
            continue

        m_durations = [o["client_e2e_duration_ms"] for o in m_obs]
        m_decode_tps = [o["raw_decode_tokens_per_second"] for o in m_obs if o["raw_decode_tokens_per_second"] > 0]
        m_chars_s = [o["characters_per_second"] for o in m_obs]
        m_scores = [o["correctness_score"] for o in m_obs if o["correctness_score"] is not None]
        m_compliances = [o["format_compliant"] for o in m_obs if o["format_compliant"] is not None]

        cat_breakdown: dict[str, Any] = {}
        for cat in categories:
            c_obs = [o for o in m_obs if o["category"] == cat]
            if not c_obs:
                continue

            c_dur = [o["client_e2e_duration_ms"] for o in c_obs]
            c_scores = [o["correctness_score"] for o in c_obs if o["correctness_score"] is not None]
            c_success = [o["task_success"] for o in c_obs if o["task_success"] is not None]
            c_format = [o["format_compliant"] for o in c_obs if o["format_compliant"] is not None]
            c_decode = [o["raw_decode_tokens_per_second"] for o in c_obs if o["raw_decode_tokens_per_second"] > 0]
            c_chars = [o["characters_per_second"] for o in c_obs]

            cat_breakdown[cat] = {
                "task_count": len(c_obs),
                "successful_count": sum(1 for s in c_success if s),
                "accuracy_pct": round(100.0 * sum(1 for s in c_success if s) / len(c_success), 1) if c_success else 0.0,
                "mean_correctness_score": round(statistics.mean(c_scores), 3) if c_scores else 0.0,
                "format_compliance_pct": round(100.0 * sum(1 for f in c_format if f) / len(c_format), 1) if c_format else 0.0,
                "median_duration_ms": round(statistics.median(c_dur), 2) if c_dur else 0.0,
                "mean_decode_tps": round(statistics.mean(c_decode), 2) if c_decode else 0.0,
                "mean_characters_per_s": round(statistics.mean(c_chars), 2) if c_chars else 0.0,
            }

        summary[m_id] = {
            "total_observations": len(m_obs),
            "median_duration_ms": round(statistics.median(m_durations), 2) if m_durations else 0.0,
            "mean_decode_tps": round(statistics.mean(m_decode_tps), 2) if m_decode_tps else 0.0,
            "mean_characters_per_s": round(statistics.mean(m_chars_s), 2) if m_chars_s else 0.0,
            "mean_correctness_score": round(statistics.mean(m_scores), 3) if m_scores else 0.0,
            "format_compliance_rate_pct": round(100.0 * sum(1 for f in m_compliances if f) / len(m_compliances), 1) if m_compliances else 0.0,
            "categories": cat_breakdown,
        }

    return summary


def run_session_22_quality_benchmark() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute complete Phase 5 benchmark across DEV and SELECTION tasks with variability repeats."""
    print("==================================================================")
    print("  SESSION 22: EXPANDED QUALITY BENCHMARKING (DEV & SELECTION)     ")
    print("==================================================================")

    installed_dict = fetch_installed_ollama_models("http://127.0.0.1:11434")
    target_models = ["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"]

    exp_id = f"exp-s22-quality-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    observations: list[dict[str, Any]] = []
    obs_counter = 1

    # Main pass: DEV tasks (6) + SELECTION tasks (18) = 24 tasks
    main_tasks = DEV_TASKS + SELECTION_TASKS
    print(f"Executing Main Pass: {len(main_tasks)} tasks across 3 models (72 requests)...")

    # Order rotation across models:
    # Model 1 -> Model 2 -> Model 3
    with httpx.Client(base_url="http://127.0.0.1:11434", timeout=60.0, trust_env=False) as client:
        # Repetition 1: All 24 main tasks across all 3 models
        for m_idx, model_id in enumerate(target_models):
            print(f" -> Benchmarking {model_id} on {len(main_tasks)} tasks (Rep 1)...")
            recs, obs_counter = run_benchmark_pass(
                client=client,
                model_id=model_id,
                installed_dict=installed_dict,
                tasks=main_tasks,
                repetition_index=1,
                exp_id=exp_id,
                obs_counter=obs_counter,
            )
            observations.extend(recs)

        # Repetition 2 & 3 on 6 representative tasks (1 per category) to measure variability
        rep_tasks = [
            t for t in SELECTION_TASKS
            if t.task_id in [
                "factual_select_01",
                "num_select_01",
                "json_select_01",
                "code_select_01",
                "retrieval_select_01",
                "longform_select_01",
            ]
        ]
        print(f"\nExecuting Variability Passes: {len(rep_tasks)} tasks x 2 reps across 3 models (36 requests)...")
        for rep in [2, 3]:
            # Rotate model order
            offset = (rep - 1) % len(target_models)
            rotated_models = target_models[offset:] + target_models[:offset]
            for model_id in rotated_models:
                print(f" -> Benchmarking {model_id} on variability tasks (Rep {rep})...")
                recs, obs_counter = run_benchmark_pass(
                    client=client,
                    model_id=model_id,
                    installed_dict=installed_dict,
                    tasks=rep_tasks,
                    repetition_index=rep,
                    exp_id=exp_id,
                    obs_counter=obs_counter,
                )
                observations.extend(recs)

    print(f"\nTotal benchmark observations recorded: {len(observations)}")

    # Compute summaries
    summary = compute_category_summary(observations)

    # Export manifest, JSONL, and CSV
    manifest_path = Path("storage/experiments/session_22_quality_benchmark_manifest.json")
    jsonl_path = Path("storage/datasets/session_22_quality_measurements.jsonl")
    csv_path = Path("storage/datasets/session_22_quality_measurements.csv")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "experiment_id": exp_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_version": "0.40.11",
        "workstation_cpu": "Intel Core i7-14700",
        "models_evaluated": target_models,
        "task_set_hashes": compute_suite_hashes(),
        "total_observations": len(observations),
        "summary_statistics": summary,
        "observations": observations,
    }

    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Exported manifest to {manifest_path}")

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for r in observations:
            jf.write(json.dumps(r) + "\n")
    print(f"Exported JSONL to {jsonl_path}")

    if observations:
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(observations[0].keys()))
            writer.writeheader()
            writer.writerows(observations)
        print(f"Exported CSV to {csv_path}")

    return observations, summary


if __name__ == "__main__":
    run_session_22_quality_benchmark()
