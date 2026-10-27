"""Session 17 Prospective Validation & Scenario Evaluation Pipeline.

Executes prospective predictions, persists them before inference, runs genuine
live inference against Ollama on Intel Core i7-14700 CPU, compares observed metrics,
and saves timestamped forensic artifacts.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ModelResidencyState,
)
from llm_ii_engine.optimizer.models import OptimizationObjective
from llm_ii_engine.recommendation.engine import WorkloadRecommendationEngine
from llm_ii_engine.recommendation.models import WorkloadRecommendationRequest


def run_session_17_pipeline() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    engine = WorkloadRecommendationEngine(dataset_path=root)

    # 1. Define the Six Standard Scenarios
    scenarios = [
        {
            "scenario_id": "SCENARIO_A",
            "name": "Scenario A — Short Interactive Request",
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=32,
                expected_output_tokens=16,
                concurrency=1,
                expected_model_residency=ModelResidencyState.WARM,
                expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
                latency_sla_ms=2500.0,
                objective=OptimizationObjective.MINIMIZE_DURATION,
            ),
            "benchmark_prompt": "Explain the concept of latency in computer systems in one short sentence.",
        },
        {
            "scenario_id": "SCENARIO_B",
            "name": "Scenario B — Long-Context Request",
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=677,
                expected_output_tokens=32,
                concurrency=1,
                expected_model_residency=ModelResidencyState.WARM,
                expected_prefix_cache=CacheStateClassification.UNKNOWN,
                latency_sla_ms=5000.0,
                objective=OptimizationObjective.MINIMIZE_DURATION,
            ),
            "benchmark_prompt": None,
        },
        {
            "scenario_id": "SCENARIO_C",
            "name": "Scenario C — Long Generation Request",
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=128,
                expected_output_tokens=128,
                concurrency=1,
                expected_model_residency=ModelResidencyState.WARM,
                expected_prefix_cache=CacheStateClassification.COLD_PREFIX,
                latency_sla_ms=8000.0,
                objective=OptimizationObjective.MAXIMIZE_DECODE_THROUGHPUT,
            ),
            "benchmark_prompt": "Provide a comprehensive step-by-step tutorial on designing an event-driven microservices architecture using Apache Kafka and Python.",
        },
        {
            "scenario_id": "SCENARIO_D",
            "name": "Scenario D — Cold-Start-Sensitive Request",
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=119,
                expected_output_tokens=32,
                concurrency=1,
                expected_model_residency=ModelResidencyState.COLD,
                require_cold_start=True,
                latency_sla_ms=4000.0,
                objective=OptimizationObjective.MINIMIZE_DURATION,
            ),
            "benchmark_prompt": None,
        },
        {
            "scenario_id": "SCENARIO_E",
            "name": "Scenario E — Strict TTFT SLA Request",
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=46,
                expected_output_tokens=32,
                concurrency=1,
                ttft_sla_ms=100.0,
                objective=OptimizationObjective.MINIMIZE_DURATION,
            ),
            "benchmark_prompt": None,
        },
        {
            "scenario_id": "SCENARIO_F",
            "name": "Scenario F — Out-Of-Distribution Request",
            "request": WorkloadRecommendationRequest(
                model_id="llama3.2:1b",
                prompt_tokens=4000,
                expected_output_tokens=500,
                concurrency=1,
                latency_sla_ms=30000.0,
                objective=OptimizationObjective.MINIMIZE_DURATION,
            ),
            "benchmark_prompt": None,
        },
    ]

    scenario_results = []
    prospective_predictions = {}

    print("--- Step 1: Generating Prospective Recommendations ---")
    for item in scenarios:
        req = item["request"]
        res = engine.recommend(req)
        scenario_results.append({
            "scenario_id": item["scenario_id"],
            "name": item["name"],
            "request": req.model_dump(),
            "recommendation": res.model_dump(),
        })
        prospective_predictions[item["scenario_id"]] = {
            "predicted_client_e2e_duration_ms": res.predicted_metrics.client_e2e_duration_ms,
            "predicted_server_decode_throughput_tok_s": res.predicted_metrics.server_decode_throughput_tok_s,
            "prediction_interval": res.prediction_uncertainty_interval,
            "evidence_source": res.evidence_source.value,
            "status": res.status.value,
        }

    # Save Prospective Predictions BEFORE running benchmarks
    timestamp_pre = datetime.now(UTC).isoformat()
    prospective_payload = {
        "timestamp_utc_pre_execution": timestamp_pre,
        "model_id": "llama3.2:1b",
        "runtime": "ollama",
        "hardware": "Intel(R) Core(TM) i7-14700",
        "prospective_predictions": prospective_predictions,
        "scenarios": scenario_results,
    }

    out_json = root / "storage" / "experiments" / "session_17_prospective_validation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(prospective_payload, f, indent=2)
    print(f"Persisted prospective predictions to {out_json}")

    # 2. Execute Live Benchmark Runs for Supported Scenarios
    print("\n--- Step 2: Executing Live Validation Benchmarks ---")
    benchmark_runs = []

    client = httpx.Client(base_url="http://127.0.0.1:11434", trust_env=False, timeout=60.0)

    # Prewarm model into memory
    print("Prewarming model weights...")
    client.post("/api/generate", json={"model": "llama3.2:1b", "prompt": "warmup", "options": {"num_predict": 1}, "stream": False})

    for item in scenarios:
        if not item.get("benchmark_prompt"):
            continue

        sc_id = item["scenario_id"]
        sc_name = item["name"]
        prompt = item["benchmark_prompt"]
        out_tokens = item["request"].expected_output_tokens
        pred = prospective_predictions[sc_id]

        print(f"Executing {sc_name} (Requested Output: {out_tokens} tokens)...")
        repetitions = []

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
            load_dur_ns = data.get("load_duration", 0)

            decode_dur_ms = round(eval_dur_ns / 1e6, 2)
            prefill_dur_ms = round(prompt_dur_ns / 1e6, 2)
            load_dur_ms = round(load_dur_ns / 1e6, 2)
            decode_tps = round(eval_count / (eval_dur_ns / 1e9), 2)
            e2e_tps = round(eval_count / (client_dur_ms / 1000.0), 2)

            repetitions.append({
                "repetition": rep,
                "client_e2e_duration_ms": client_dur_ms,
                "server_decode_duration_ms": decode_dur_ms,
                "server_prefill_duration_ms": prefill_dur_ms,
                "server_load_duration_ms": load_dur_ms,
                "server_decode_throughput_tok_s": decode_tps,
                "client_e2e_output_throughput_tok_s": e2e_tps,
                "actual_output_tokens": eval_count,
            })
            print(f"  Rep {rep}: Client E2E = {client_dur_ms} ms, Decode = {decode_tps} tok/s")

        # Compute median across 3 repetitions
        import statistics
        med_client_dur = round(statistics.median([r["client_e2e_duration_ms"] for r in repetitions]), 2)
        med_decode_tps = round(statistics.median([r["server_decode_throughput_tok_s"] for r in repetitions]), 2)
        med_decode_dur = round(statistics.median([r["server_decode_duration_ms"] for r in repetitions]), 2)
        med_prefill_dur = round(statistics.median([r["server_prefill_duration_ms"] for r in repetitions]), 2)

        pred_dur = pred["predicted_client_e2e_duration_ms"]
        pred_tps = pred["predicted_server_decode_throughput_tok_s"]

        dur_abs_err = round(abs(med_client_dur - pred_dur), 2)
        dur_rel_err = round((dur_abs_err / pred_dur) * 100.0, 2)
        tps_abs_err = round(abs(med_decode_tps - pred_tps), 2)
        tps_rel_err = round((tps_abs_err / pred_tps) * 100.0, 2)

        benchmark_runs.append({
            "scenario_id": sc_id,
            "name": sc_name,
            "repetitions": repetitions,
            "median_observed": {
                "client_e2e_duration_ms": med_client_dur,
                "server_decode_duration_ms": med_decode_dur,
                "server_prefill_duration_ms": med_prefill_dur,
                "server_decode_throughput_tok_s": med_decode_tps,
            },
            "prospective_forecast": {
                "predicted_client_e2e_duration_ms": pred_dur,
                "predicted_server_decode_throughput_tok_s": pred_tps,
            },
            "comparison": {
                "duration_absolute_error_ms": dur_abs_err,
                "duration_relative_error_pct": dur_rel_err,
                "throughput_absolute_error_tok_s": tps_abs_err,
                "throughput_relative_error_pct": tps_rel_err,
            },
        })

    # 3. Update the persisted JSON artifact with observed validation results
    prospective_payload["timestamp_utc_post_execution"] = datetime.now(UTC).isoformat()
    prospective_payload["live_validation_benchmarks"] = benchmark_runs

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(prospective_payload, f, indent=2)

    print(f"\nUpdated {out_json} with live validation results.")
    print("Execution complete.")


if __name__ == "__main__":
    run_session_17_pipeline()
