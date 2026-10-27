"""Dataset v2 builder and predictor evaluation pipeline (Session 16).

1. Reads historical datasets and SQLite server_stats_json.
2. Populates all 10 distinct measurement contract metrics with zero ambiguity.
3. Exports derived storage/datasets/expansion_llama3_2_1b_v2.jsonl and .csv.
4. Executes configuration-grouped cross-validation across:
   - ConstantMedianBaseline
   - RidgeRegressionPredictor
   - WorkloadAwareLatencyPredictor (prospective)
   - WorkloadAwareLatencyPredictor (retrospective)
5. Evaluates targets:
   - client_e2e_duration_ms
   - server_decode_duration_ms
   - client_e2e_output_throughput_tok_s
   - server_decode_throughput_tok_s
6. Exports storage/experiments/session_16_predictor_comparison.json.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

from llm_ii_engine.benchmark.dataset_export import (
    EmpiricalDatasetExporter,
    EmpiricalObservationRecord,
)
from llm_ii_engine.core.enums import ModelResidencyState
from llm_ii_engine.core.schemas import ModelSpec
from llm_ii_engine.prediction.baselines import (
    ConstantMedianPredictor,
    RidgeRegressionPredictor,
)
from llm_ii_engine.prediction.dataset import DatasetBuilder
from llm_ii_engine.prediction.evaluation import DataSplitter, PredictionEvaluator
from llm_ii_engine.prediction.models import (
    PredictionRecord,
    PredictionTarget,
)
from llm_ii_engine.prediction.workload_aware import WorkloadAwareLatencyPredictor


def load_sqlite_server_stats(db_path: Path) -> dict[str, dict]:
    """Load raw server_stats from SQLite database by request_id."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT request_id, server_stats_json FROM request_benchmarks")
    stats_map = {}
    for req_id, stats_json in cur.fetchall():
        if stats_json:
            with contextlib.suppress(Exception):
                stats_map[req_id] = json.loads(stats_json)
    conn.close()
    return stats_map


def build_v2_records() -> tuple[list[EmpiricalObservationRecord], list[EmpiricalObservationRecord]]:
    """Build enriched v2 observation records for Expansion and Combined datasets."""
    root = Path(__file__).resolve().parent.parent.parent
    db_path = root / "storage" / "llm_ii_engine.db"
    sqlite_stats = load_sqlite_server_stats(db_path)

    # 1. Load expansion_llama3_2_1b.jsonl
    expansion_path = root / "storage" / "datasets" / "expansion_llama3_2_1b.jsonl"
    expansion_records: list[EmpiricalObservationRecord] = []
    if expansion_path.exists():
        with open(expansion_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                req_id = data.get("request_id")
                # Enrich with SQLite server_stats if available
                if req_id in sqlite_stats:
                    sq = sqlite_stats[req_id]
                    tot_ms = sq.get("total_duration_ms")
                    if tot_ms:
                        data["server_total_duration_ms"] = round(tot_ms, 2)
                    load_ms = sq.get("load_duration_ms")
                    if not load_ms and "load_duration" in sq.get("raw_stats", {}):
                        load_ms = sq["raw_stats"]["load_duration"] / 1e6
                    if load_ms:
                        data["server_load_duration_ms"] = round(load_ms, 2)
                    pref_ms = sq.get("prompt_duration_ms")
                    if pref_ms:
                        data["server_prefill_duration_ms"] = round(pref_ms, 2)
                        data["server_prompt_eval_duration_ms"] = round(pref_ms, 2)
                    dec_ms = sq.get("eval_duration_ms")
                    if dec_ms:
                        data["server_decode_duration_ms"] = round(dec_ms, 2)
                        data["server_eval_duration_ms"] = round(dec_ms, 2)
                    dec_thru = sq.get("eval_tokens_per_second")
                    if dec_thru:
                        data["server_decode_throughput_tok_s"] = round(dec_thru, 2)

                # Ensure client e2e fields are set
                data["client_e2e_duration_ms"] = data.get("duration_ms")
                data["client_e2e_output_throughput_tok_s"] = data.get("output_tokens_per_second")

                rec = EmpiricalObservationRecord.model_validate(data)
                expansion_records.append(rec)

    # 2. Load session_15_cache_controlled_measurements.jsonl
    s15_path = root / "storage" / "datasets" / "session_15_cache_controlled_measurements.jsonl"
    s15_records: list[EmpiricalObservationRecord] = []
    if s15_path.exists():
        with open(s15_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # Parse load_duration_ms from caveats if present
                for cav in data.get("environmental_caveats", []):
                    if "Reported load_duration_ms:" in cav:
                        val_str = cav.split("Reported load_duration_ms:")[1].replace("ms", "").strip()
                        with contextlib.suppress(ValueError):
                            data["server_load_duration_ms"] = float(val_str)

                # Derive server total duration if prefill, decode, and load are known
                load_ms = data.get("server_load_duration_ms") or 0.0
                pref_ms = data.get("server_prompt_eval_duration_ms") or 0.0
                dec_ms = data.get("server_eval_duration_ms") or 0.0
                if pref_ms > 0 and dec_ms > 0:
                    data["server_total_duration_ms"] = round(load_ms + pref_ms + dec_ms, 2)

                data["client_e2e_duration_ms"] = data.get("duration_ms")
                data["client_e2e_output_throughput_tok_s"] = data.get("output_tokens_per_second")
                data["server_prefill_duration_ms"] = pref_ms
                data["server_decode_duration_ms"] = dec_ms

                rec = EmpiricalObservationRecord.model_validate(data)
                s15_records.append(rec)

    # 3. Load pilot_llama3_2_1b.jsonl
    pilot_path = root / "storage" / "datasets" / "pilot_llama3_2_1b.jsonl"
    pilot_records: list[EmpiricalObservationRecord] = []
    if pilot_path.exists():
        with open(pilot_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                req_id = data.get("request_id")
                if req_id in sqlite_stats:
                    sq = sqlite_stats[req_id]
                    if sq.get("total_duration_ms"):
                        data["server_total_duration_ms"] = round(sq["total_duration_ms"], 2)
                    load_ms = sq.get("load_duration_ms")
                    if not load_ms and "load_duration" in sq.get("raw_stats", {}):
                        load_ms = sq["raw_stats"]["load_duration"] / 1e6
                    if load_ms:
                        data["server_load_duration_ms"] = round(load_ms, 2)
                    if sq.get("eval_tokens_per_second"):
                        data["server_decode_throughput_tok_s"] = round(sq["eval_tokens_per_second"], 2)

                data["client_e2e_duration_ms"] = data.get("duration_ms")
                data["client_e2e_output_throughput_tok_s"] = data.get("output_tokens_per_second")
                data["server_prefill_duration_ms"] = data.get("server_prompt_eval_duration_ms")
                data["server_decode_duration_ms"] = data.get("server_eval_duration_ms")

                rec = EmpiricalObservationRecord.model_validate(data)
                pilot_records.append(rec)

    combined_records = pilot_records + expansion_records + s15_records
    return expansion_records, combined_records


def run_configuration_grouped_evaluation(
    records: list[PredictionRecord],
    targets: list[PredictionTarget],
) -> dict:
    """Run configuration-grouped Leave-One-Group-Out cross-validation across predictors."""
    results = {}

    for target in targets:
        target_results = {
            "constant_median": {"actuals": [], "predictions": [], "maes": [], "rmses": []},
            "ridge_regression": {"actuals": [], "predictions": [], "maes": [], "rmses": []},
            "workload_aware_prospective": {"actuals": [], "predictions": [], "maes": [], "rmses": []},
            "workload_aware_retrospective": {"actuals": [], "predictions": [], "maes": [], "rmses": []},
        }

        # Filter eligible records
        eligible = [r for r in records if r.is_eligible_for_target(target)]
        if len(eligible) < 4:
            continue

        # Group by configuration PxO
        try:
            folds = DataSplitter.leave_one_group_out_split(eligible, group_key="workload_config")
        except ValueError:
            continue

        for train_set, test_set, group_id in folds:
            if not train_set or not test_set:
                continue

            # 1. Constant Median Baseline
            cm = ConstantMedianPredictor(target)
            try:
                cm.fit(train_set)
                eval_cm = PredictionEvaluator.evaluate(cm, test_set, split_strategy=f"logo_{group_id}")
                target_results["constant_median"]["maes"].append(eval_cm.mae)
                target_results["constant_median"]["rmses"].append(eval_cm.rmse)
            except Exception as e:
                print(f"CM error: {e}")

            # 2. Ridge Regression Predictor
            ridge = RidgeRegressionPredictor(target)
            try:
                ridge.fit(train_set)
                eval_ridge = PredictionEvaluator.evaluate(
                    ridge, test_set, split_strategy=f"logo_{group_id}", baseline_predictor=cm
                )
                target_results["ridge_regression"]["maes"].append(eval_ridge.mae)
                target_results["ridge_regression"]["rmses"].append(eval_ridge.rmse)
            except Exception as e:
                print(f"Ridge error: {e}")

            # 3. WorkloadAwareLatencyPredictor (Prospective)
            wa_pro = WorkloadAwareLatencyPredictor(target)
            try:
                wa_pro.fit(train_set)
                # Evaluated prospectively
                actuals = [wa_pro._extract_target_value(r) for r in test_set if wa_pro._extract_target_value(r) is not None]
                preds = [wa_pro.predict(r).predicted_value for r in test_set if wa_pro._extract_target_value(r) is not None]
                if actuals and preds:
                    mae, rmse, _ = PredictionEvaluator.calculate_metrics(actuals, preds)
                    target_results["workload_aware_prospective"]["maes"].append(mae)
                    target_results["workload_aware_prospective"]["rmses"].append(rmse)
            except Exception:
                pass

            # 4. WorkloadAwareLatencyPredictor (Retrospective)
            try:
                actuals = [wa_pro._extract_target_value(r) for r in test_set if wa_pro._extract_target_value(r) is not None]
                preds_retro = [wa_pro.predict_retrospective(r).predicted_value for r in test_set if wa_pro._extract_target_value(r) is not None]
                if actuals and preds_retro:
                    mae, rmse, _ = PredictionEvaluator.calculate_metrics(actuals, preds_retro)
                    target_results["workload_aware_retrospective"]["maes"].append(mae)
                    target_results["workload_aware_retrospective"]["rmses"].append(rmse)
            except Exception:
                pass

        # Compute summary metrics across folds
        summary = {}
        for model_name, data in target_results.items():
            maes = data["maes"]
            rmses = data["rmses"]
            if maes:
                summary[model_name] = {
                    "mean_mae": round(sum(maes) / len(maes), 2),
                    "mean_rmse": round(sum(rmses) / len(rmses), 2),
                    "fold_count": len(maes),
                }
            else:
                summary[model_name] = {"mean_mae": None, "mean_rmse": None, "fold_count": 0}

        results[target.value] = summary

    return results


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent

    # Step 1: Build enriched records
    expansion_v2_recs, combined_recs = build_v2_records()

    # Step 2: Export expansion_llama3_2_1b_v2 datasets
    v2_jsonl_path = root / "storage" / "datasets" / "expansion_llama3_2_1b_v2.jsonl"
    v2_csv_path = root / "storage" / "datasets" / "expansion_llama3_2_1b_v2.csv"

    EmpiricalDatasetExporter.export_to_jsonl(
        records=expansion_v2_recs,
        output_path=v2_jsonl_path,
        filter_real_only=True,
        exclude_warmup=True,
    )
    EmpiricalDatasetExporter.export_to_csv(
        records=expansion_v2_recs,
        output_path=v2_csv_path,
        filter_real_only=True,
        exclude_warmup=True,
    )
    print(f"Exported {len(expansion_v2_recs)} v2 records to {v2_jsonl_path} and {v2_csv_path}")

    # Step 3: Convert observation records to PredictionRecord instances
    model_spec = ModelSpec(
        model_id="llama3.2:1b",
        parameter_count_billion=1.23,
        num_layers=16,
        num_attention_heads=32,
        num_kv_heads=8,
        hidden_dimension=2048,
        head_dimension=64,
        vocab_size=128256,
        context_window=131072,
    )

    pred_dataset = DatasetBuilder.from_empirical_observations(
        observations=combined_recs,
        model_spec=model_spec,
        dataset_id="ds-session-16-combined",
    )
    pred_records = pred_dataset.records

    # Step 4: Run evaluation across full population and subsets
    targets = [
        PredictionTarget.CLIENT_E2E_DURATION_MS,
        PredictionTarget.SERVER_DECODE_DURATION_MS,
        PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S,
        PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S,
    ]

    # Full population
    full_eval = run_configuration_grouped_evaluation(pred_records, targets)

    # Warm-only population
    warm_records = [r for r in pred_records if r.model_residency_state == ModelResidencyState.WARM]
    warm_eval = run_configuration_grouped_evaluation(warm_records, targets)

    # Cold-only population
    cold_records = [r for r in pred_records if r.model_residency_state == ModelResidencyState.COLD]

    # Output compilation
    comparison_artifact = {
        "evaluation_name": "Session 16 Configuration-Grouped Predictor Benchmark",
        "total_records": len(pred_records),
        "warm_records_count": len(warm_records),
        "cold_records_count": len(cold_records),
        "split_strategy": "leave_one_configuration_out_grouped",
        "full_population_evaluation": full_eval,
        "warm_population_evaluation": warm_eval,
    }

    out_file = root / "storage" / "experiments" / "session_16_predictor_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_artifact, f, indent=2)

    print(f"Saved predictor comparison artifact to {out_file}")
    print(json.dumps(comparison_artifact, indent=2))


if __name__ == "__main__":
    main()
