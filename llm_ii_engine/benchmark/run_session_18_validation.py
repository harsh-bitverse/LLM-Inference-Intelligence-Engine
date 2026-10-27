"""Session 18 Prospective Validation Runner.

Executes prospective performance predictions and persists them with a pre-execution UTC
timestamp BEFORE live inference. Then executes bounded empirical benchmarks against the local
authorized Ollama llama3.2:1b (Q8_0) instance on the Intel Core i7-14700 CPU, compares observed
metrics against the pre-execution forecasts, and exports reproducible validation artifacts.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llm_ii_engine.benchmark.expansion import (
    PROMPT_TEXT_32,
    PROMPT_TEXT_128,
    PROMPT_TEXT_768,
)
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ModelResidencyState,
)
from llm_ii_engine.optimizer.models import OptimizationObjective
from llm_ii_engine.recommendation.engine import WorkloadRecommendationEngine
from llm_ii_engine.recommendation.models import WorkloadRecommendationRequest


def run_session_18_validation_pipeline() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent.parent
    engine = WorkloadRecommendationEngine(dataset_path=root)

    print("=== Step 1: Generating and Persisting Pre-Execution Prospective Predictions ===")
    pre_timestamp = datetime.now(UTC).isoformat()

    # Define test workloads
    test_workloads = [
        {
            "workload_id": "VAL_WORKLOAD_SHORT",
            "name": "Short Output Workload (O=16)",
            "prompt_text": PROMPT_TEXT_32,
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=46,
                expected_output_tokens=16,
                concurrency=1,
                expected_model_residency=ModelResidencyState.WARM,
                expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
                latency_sla_ms=2500.0,
                objective=OptimizationObjective.MINIMIZE_DURATION,
            ),
        },
        {
            "workload_id": "VAL_WORKLOAD_LONG",
            "name": "Long Output Workload (O=128)",
            "prompt_text": PROMPT_TEXT_128,
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=119,
                expected_output_tokens=128,
                concurrency=1,
                expected_model_residency=ModelResidencyState.WARM,
                expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
                latency_sla_ms=8000.0,
                objective=OptimizationObjective.MAXIMIZE_DECODE_THROUGHPUT,
            ),
        },
        {
            "workload_id": "VAL_WORKLOAD_CACHE_COLD",
            "name": "Long Prompt Cold Prefix (P=677, O=32)",
            "prompt_text": PROMPT_TEXT_768,
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=677,
                expected_output_tokens=32,
                concurrency=1,
                expected_model_residency=ModelResidencyState.WARM,
                expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
                latency_sla_ms=5000.0,
                objective=OptimizationObjective.MINIMIZE_DURATION,
            ),
        },
        {
            "workload_id": "VAL_WORKLOAD_CACHE_WARM",
            "name": "Long Prompt Warm Prefix Replay (P=677, O=32)",
            "prompt_text": PROMPT_TEXT_768,
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=677,
                expected_output_tokens=32,
                concurrency=1,
                expected_model_residency=ModelResidencyState.WARM,
                expected_prefix_cache=CacheStateClassification.WARM_PREFIX,
                latency_sla_ms=2500.0,
                objective=OptimizationObjective.MINIMIZE_DURATION,
            ),
        },
    ]

    prospective_predictions: dict[str, Any] = {}
    recommendations_data: list[dict[str, Any]] = []

    for item in test_workloads:
        w_id = item["workload_id"]
        req = item["request"]
        rec = engine.recommend(req)

        prospective_predictions[w_id] = {
            "predicted_client_e2e_duration_ms": rec.predicted_metrics.client_e2e_duration_ms,
            "predicted_server_decode_throughput_tok_s": rec.predicted_metrics.server_decode_throughput_tok_s,
            "exploratory_prediction_range": rec.exploratory_prediction_range,
            "confidence_label_status": rec.confidence_label_status,
            "range_coverage_percent": rec.range_coverage_percent,
            "evidence_source": rec.evidence_source.value,
            "status": rec.status.value,
            "sla_assessment": rec.sla_assessments.get("latency_sla", {}),
        }

        recommendations_data.append({
            "workload_id": w_id,
            "name": item["name"],
            "request": req.model_dump(),
            "recommendation": rec.model_dump(),
        })

    payload: dict[str, Any] = {
        "timestamp_utc_pre_execution": pre_timestamp,
        "model_id": "llama3.2:1b",
        "runtime": "ollama",
        "hardware": "Intel(R) Core(TM) i7-14700",
        "prospective_predictions": prospective_predictions,
        "workloads": recommendations_data,
        "timestamp_utc_post_execution": None,
        "live_validation_results": [],
    }

    out_json = root / "storage" / "experiments" / "session_18_controlled_validation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Persisted frozen prospective forecasts to {out_json}")

    # Step 2: Execute Live Bounded Benchmarks
    print("\n=== Step 2: Executing Controlled Live Inference Benchmarks ===")
    client = httpx.Client(base_url="http://127.0.0.1:11434", trust_env=False, timeout=120.0)

    # Warm up Ollama server & model weights into memory
    print("Prewarming model weights in memory...")
    client.post("/api/generate", json={"model": "llama3.2:1b", "prompt": "warmup test", "options": {"num_predict": 1}, "stream": False})

    live_results: list[dict[str, Any]] = []

    for item in test_workloads:
        w_id = item["workload_id"]
        w_name = item["name"]
        prompt = item["prompt_text"]
        out_tokens = item["request"].expected_output_tokens
        forecast = prospective_predictions[w_id]

        print(f"\nExecuting {w_name} ({w_id})...")
        repetitions = []

        # 3 measured repetitions
        for rep in range(1, 4):
            t0 = time.perf_counter()
            resp = client.post(
                "/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "options": {
                        "num_predict": out_tokens,
                        "temperature": 0.0,
                    },
                    "stream": False,
                },
            )
            t1 = time.perf_counter()
            client_dur_ms = round((t1 - t0) * 1000.0, 2)

            data = resp.json()
            eval_count = data.get("eval_count", 0)
            eval_dur_ns = data.get("eval_duration", 1)
            prompt_dur_ns = data.get("prompt_eval_duration", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            load_dur_ns = data.get("load_duration", 0)

            decode_dur_ms = round(eval_dur_ns / 1e6, 2)
            prefill_dur_ms = round(prompt_dur_ns / 1e6, 2)
            load_dur_ms = round(load_dur_ns / 1e6, 2)
            decode_tps = round(eval_count / (eval_dur_ns / 1e9), 2)
            e2e_tps = round(eval_count / (client_dur_ms / 1000.0), 2)

            rep_data = {
                "repetition": rep,
                "client_e2e_duration_ms": client_dur_ms,
                "server_decode_duration_ms": decode_dur_ms,
                "server_prefill_duration_ms": prefill_dur_ms,
                "server_load_duration_ms": load_dur_ms,
                "server_decode_throughput_tok_s": decode_tps,
                "client_e2e_output_throughput_tok_s": e2e_tps,
                "actual_output_tokens": eval_count,
                "actual_prompt_tokens": prompt_eval_count,
            }
            repetitions.append(rep_data)
            print(f"  Rep {rep}: Client E2E = {client_dur_ms} ms | Prefill = {prefill_dur_ms} ms | Decode = {decode_dur_ms} ms ({decode_tps} tok/s) | Tokens = {eval_count}")

        # Compute median metrics
        med_e2e = round(statistics.median([r["client_e2e_duration_ms"] for r in repetitions]), 2)
        med_decode_tps = round(statistics.median([r["server_decode_throughput_tok_s"] for r in repetitions]), 2)
        med_decode_dur = round(statistics.median([r["server_decode_duration_ms"] for r in repetitions]), 2)
        med_prefill_dur = round(statistics.median([r["server_prefill_duration_ms"] for r in repetitions]), 2)
        med_out_tokens = int(statistics.median([r["actual_output_tokens"] for r in repetitions]))

        pred_dur = forecast["predicted_client_e2e_duration_ms"]
        pred_tps = forecast["predicted_server_decode_throughput_tok_s"]
        range_lower, range_upper = forecast["exploratory_prediction_range"]

        dur_abs_err = round(abs(med_e2e - pred_dur), 2)
        dur_rel_err = round((dur_abs_err / pred_dur) * 100.0, 2)
        within_range = range_lower <= med_e2e <= range_upper

        # SLA Check
        sla_target = item["request"].latency_sla_ms
        sla_satisfied = med_e2e <= sla_target if sla_target else None

        result_entry = {
            "workload_id": w_id,
            "name": w_name,
            "repetitions": repetitions,
            "median_observed": {
                "client_e2e_duration_ms": med_e2e,
                "server_decode_duration_ms": med_decode_dur,
                "server_prefill_duration_ms": med_prefill_dur,
                "server_decode_throughput_tok_s": med_decode_tps,
                "actual_output_tokens": med_out_tokens,
            },
            "prospective_forecast": {
                "predicted_client_e2e_duration_ms": pred_dur,
                "predicted_server_decode_throughput_tok_s": pred_tps,
                "exploratory_prediction_range": [range_lower, range_upper],
                "confidence_label_status": forecast["confidence_label_status"],
            },
            "comparison": {
                "duration_absolute_error_ms": dur_abs_err,
                "duration_relative_error_pct": dur_rel_err,
                "within_exploratory_range": within_range,
                "sla_target_ms": sla_target,
                "sla_satisfied_observed": sla_satisfied,
                "predicted_sla_status": forecast["sla_assessment"].get("status"),
            },
        }
        live_results.append(result_entry)

    post_timestamp = datetime.now(UTC).isoformat()
    payload["timestamp_utc_post_execution"] = post_timestamp
    payload["live_validation_results"] = live_results

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved updated prospective validation results to {out_json}")

    return payload


if __name__ == "__main__":
    run_session_18_validation_pipeline()
