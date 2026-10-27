"""Execute Session 12B Authorized Live LLM Benchmark using Ollama and llama3.2:1b.

Performs:
1. Database initialization and SQLite WAL verification.
2. Hardware capability discovery.
3. Benchmark manifest construction and serialization.
4. Live inference execution across pilot workloads (1 warmup + 3 measured).
5. SQLite persistence of benchmark suite and request outcomes.
6. Empirical dataset export (JSONL and CSV) with real-measurement qualification.
7. Repeatability analysis (CV%, warmup overhead, client-server divergence).
8. Downstream prediction dataset ingestion and eligibility validation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm_ii_engine.adapters.ollama import OllamaAdapter
from llm_ii_engine.benchmark.dataset_export import EmpiricalDatasetExporter
from llm_ii_engine.benchmark.experiment import (
    PILOT_LONG_WORKLOAD,
    PILOT_MEDIUM_WORKLOAD,
    PILOT_SHORT_WORKLOAD,
    BenchmarkExperimentManifest,
    ExperimentResult,
    ExperimentRunner,
)
from llm_ii_engine.benchmark.repeatability import RepeatabilityAnalyzer
from llm_ii_engine.core.enums import (
    ClockSource,
    ExecutionMode,
    FailurePolicy,
    ProvenanceType,
    QuantizationType,
)
from llm_ii_engine.core.schemas import ModelSpec
from llm_ii_engine.discovery.profiler import discover_hardware
from llm_ii_engine.prediction.dataset import DatasetBuilder
from llm_ii_engine.prediction.models import PredictionTarget
from llm_ii_engine.storage.database import DatabaseManager
from llm_ii_engine.storage.repository import EngineRepository


async def main() -> None:
    print("=" * 70)
    print("SESSION 12B: LIVE INFERENCE BENCHMARK EXECUTION")
    print("=" * 70)

    # Ensure storage subdirectories exist
    Path("storage/experiments").mkdir(parents=True, exist_ok=True)
    Path("storage/datasets").mkdir(parents=True, exist_ok=True)

    # 1. Database Initialization
    db_path = "storage/llm_ii_engine.db"
    db_mgr = DatabaseManager(f"sqlite:///{db_path}")
    version = db_mgr.init_db()
    print(f"[1/8] Initialized SQLite Database: {db_path} (Schema version: {version})")

    # 2. Hardware Discovery
    print("[2/8] Performing hardware capability discovery...")
    hw_report = discover_hardware()
    with db_mgr.get_session() as session:
        repo = EngineRepository(session)
        repo.save_hardware_report(hw_report)
    print(f"      Report ID: {hw_report.report_id}")
    print(f"      Host OS: {hw_report.host.os_name} {hw_report.host.os_release} ({hw_report.host.architecture})")
    print(f"      CPU: {hw_report.cpu.model_name} ({hw_report.cpu.logical_cores} logical cores)")
    ram_gb = round(hw_report.memory.total_bytes / (1024**3), 2)
    avail_ram_gb = round(hw_report.memory.available_bytes / (1024**3), 2) if hw_report.memory.available_bytes else None
    print(f"      RAM: {ram_gb} GB total ({avail_ram_gb} GB available)")
    print(f"      Discrete GPUs: {len(hw_report.gpus)} (CPU-only execution)")

    # 3. Model Spec & Experiment Manifest Construction
    model_spec = ModelSpec(
        model_id="llama3.2:1b",
        family="llama",
        parameter_count_billion=1.235,
        context_window=131072,
        num_layers=16,
        num_attention_heads=32,
        num_kv_heads=8,
        hidden_dimension=2048,
        quantization=QuantizationType.GGUF_Q8_0,
    )

    manifest = BenchmarkExperimentManifest(
        experiment_id="exp-pilot-llama3-2-1b-s12b",
        runtime_name="ollama",
        runtime_version="0.40.11",
        adapter_version="1.0.0",
        engine_version="1.0.0",
        model_id="llama3.2:1b",
        model_digest="baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878",
        quantization=QuantizationType.GGUF_Q8_0,
        context_window=131072,
        hardware_profile_id=hw_report.report_id,
        device_name="CPU",
        cpu_model=f"{hw_report.cpu.model_name} (14 threads active)",
        host_ram_gb=ram_gb,
        workload_suite=[PILOT_SHORT_WORKLOAD, PILOT_MEDIUM_WORKLOAD, PILOT_LONG_WORKLOAD],
        warmup_repetitions=1,
        measured_repetitions=3,
        concurrency=1,
        sampling_temperature=0.0,
        sampling_top_p=1.0,
        seed=42,
        timeout_seconds=120.0,
        failure_policy=FailurePolicy.CONTINUE_ON_ERROR,
        consecutive_failure_limit=3,
        execution_mode=ExecutionMode.REAL,
        clock_source=ClockSource.MONOTONIC_REAL,
        metric_provenance=ProvenanceType.MEASURED,
        environmental_caveats=[
            "CPU-only execution via ggml-cpu-alderlake.dll",
            "Intel UHD 770 integrated GPU disabled by Ollama (OLLAMA_IGPU_ENABLE not set)",
            "Local loopback communication bypassing corporate proxy via NO_PROXY=localhost,127.0.0.1",
        ],
        metadata={
            "session": "12B",
            "purpose": "First live local-model empirical baseline",
            "pilot_workload_count": 3,
            "warmup_requests_count": 3,
            "measured_requests_count": 9,
            "total_requests_count": 12,
        },
    )

    manifest_path = "storage/experiments/pilot_llama3_2_1b_manifest.json"
    manifest.save(manifest_path)
    print(f"[3/8] Saved Experiment Manifest: {manifest_path}")

    # 4. Live Benchmark Execution
    print("[4/8] Initializing OllamaAdapter and executing live inference suite...")
    adapter = OllamaAdapter(base_url="http://127.0.0.1:11434", timeout_seconds=120.0)
    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)

    print("      Executing 1 warmup repetition (3 workloads) + 3 measured repetitions (9 workloads)...")
    exp_result: ExperimentResult = await runner.run_experiment(manifest)
    await adapter.teardown()

    print(f"      Execution finished in {exp_result.total_wall_time_seconds:.2f}s")
    print(f"      Overall status: {exp_result.status}")
    print(f"      Total requests dispatched: {len(exp_result.all_requests)}")
    print(f"      Warmup requests: {len(exp_result.warmup_requests)} (excluded from measured aggregates)")
    print(f"      Measured requests: {len(exp_result.measured_requests)} (successful: {len(exp_result.successful_measured_requests)})")

    # 5. SQLite Persistence
    print("[5/8] Persisting BenchmarkSuiteResult and request observations to SQLite...")
    with db_mgr.get_session() as session:
        repo = EngineRepository(session)
        try:
            repo.save_benchmark_suite(exp_result.suite_result, hardware_report_id=hw_report.report_id)
            print(f"      Suite '{exp_result.suite_result.suite_id}' successfully committed.")
        except Exception as exc:
            print(f"      Suite '{exp_result.suite_result.suite_id}' already persisted or error: {exc}")

    # 6. Empirical Dataset Export
    print("[6/8] Exporting empirical datasets (JSONL, CSV, failure audit)...")
    records = EmpiricalDatasetExporter.from_experiment_result(exp_result)
    
    jsonl_path = "storage/datasets/pilot_llama3_2_1b.jsonl"
    csv_path = "storage/datasets/pilot_llama3_2_1b.csv"
    fail_jsonl_path = "storage/datasets/pilot_llama3_2_1b_failures.jsonl"
    fail_csv_path = "storage/datasets/pilot_llama3_2_1b_failures.csv"

    jsonl_count = EmpiricalDatasetExporter.export_to_jsonl(records, jsonl_path, filter_real_only=True, exclude_warmup=True)
    csv_count = EmpiricalDatasetExporter.export_to_csv(records, csv_path, filter_real_only=True, exclude_warmup=True)
    fail_count = EmpiricalDatasetExporter.export_failed_runs_to_jsonl(records, fail_jsonl_path)
    EmpiricalDatasetExporter.export_failed_runs_to_csv(records, fail_csv_path)

    print(f"      Exported {jsonl_count} clean measured records to {jsonl_path}")
    print(f"      Exported {csv_count} clean measured records to {csv_path}")
    print(f"      Recorded {fail_count} failed requests in {fail_jsonl_path}")

    # 7. Repeatability & Quality Analysis
    print("[7/8] Computing repeatability, variability (CV%), and client-server divergence...")
    rep_report = RepeatabilityAnalyzer.analyze_experiment(exp_result)
    rep_path = "storage/experiments/pilot_llama3_2_1b_repeatability.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep_report.model_dump_json(indent=2))
    print(f"      Saved Repeatability Report: {rep_path}")
    print(f"      Average Duration CV%: {rep_report.average_duration_cv_percent:.2f}%" if rep_report.average_duration_cv_percent is not None else "      Average Duration CV%: N/A")
    print(f"      Average Throughput CV%: {rep_report.average_throughput_cv_percent:.2f}%" if rep_report.average_throughput_cv_percent is not None else "      Average Throughput CV%: N/A")
    print(f"      Average Warmup Penalty: {rep_report.average_warmup_penalty_percent:.2f}%" if rep_report.average_warmup_penalty_percent is not None else "      Average Warmup Penalty: N/A")
    print(f"      Average Client-Server Divergence: {rep_report.average_client_server_divergence_ms:.2f} ms" if rep_report.average_client_server_divergence_ms is not None else "      Average Client-Server Divergence: N/A")

    for w_stat in rep_report.workload_stats:
        print(f"      - Workload {w_stat.workload_index}: "
              f"Prompt {w_stat.prompt_tokens} tok | "
              f"Mean dur {w_stat.duration_stats.mean:.1f}ms (CV {w_stat.duration_stats.cv_percent:.2f}%) | "
              f"Mean thru {w_stat.throughput_stats.mean:.2f} tok/s (CV {w_stat.throughput_stats.cv_percent:.2f}%) | "
              f"Warmup overhead {w_stat.warmup_overhead_percent:.1f}%")

    # 8. Downstream Dataset Ingestion & Eligibility Verification
    print("[8/8] Verifying downstream prediction dataset builder and target eligibility...")
    pred_dataset = DatasetBuilder.from_empirical_observations(
        records,
        model_spec=model_spec,
        hardware_report=hw_report,
    )
    print(f"      Constructed PredictionDataset with {len(pred_dataset.records)} records.")

    eligible_ttft = sum(1 for r in pred_dataset.records if r.is_eligible_for_ttft)
    eligible_tpot = sum(1 for r in pred_dataset.records if r.is_eligible_for_tpot)
    eligible_chunk = sum(1 for r in pred_dataset.records if r.is_eligible_for_chunk_latency)
    eligible_thru = sum(1 for r in pred_dataset.records if r.is_eligible_for_throughput)

    print("      Eligibility summary across all 12 records (warmup + measured):")
    print(f"      * ttft_ms: {eligible_ttft} eligible records")
    print(f"      * tpot_ms: {eligible_tpot} eligible records")
    print(f"      * chunk_latency_ms: {eligible_chunk} eligible records")
    print(f"      * output_throughput_tok_s: {eligible_thru} eligible records")

    # Detailed eligibility verification for measured records
    measured_precords = [r for r in pred_dataset.records if not r.is_warmup]
    print(f"      Measured records count: {len(measured_precords)}")
    for i, prec in enumerate(measured_precords):
        print(f"        Req {i+1} ({prec.request_id}):")
        print(f"          - TTFT eligible: {prec.is_eligible_for_ttft} (Reasons: {prec.ineligibility_reasons.get(PredictionTarget.TTFT_MS, ['N/A'])})")
        print(f"          - TPOT eligible: {prec.is_eligible_for_tpot} (Reasons: {prec.ineligibility_reasons.get(PredictionTarget.TPOT_MS, ['N/A'])})")
        print(f"          - Chunk Latency eligible: {prec.is_eligible_for_chunk_latency}")
        print(f"          - Throughput eligible: {prec.is_eligible_for_throughput}")

    print("=" * 70)
    print("SESSION 12B BENCHMARK EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
