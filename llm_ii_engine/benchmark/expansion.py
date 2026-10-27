"""Controlled benchmark expansion workloads and orchestrator for Session 13.

Defines systematically varied workload configurations across prompt lengths (~32, ~128, ~384, ~768 tokens)
and output lengths (32, 64, 128 tokens) with deterministic non-sensitive prompts, bounded execution,
and scientific measurement qualification.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from llm_ii_engine.adapters.ollama import OllamaAdapter
from llm_ii_engine.benchmark.dataset_export import (
    EmpiricalDatasetExporter,
    EmpiricalObservationRecord,
)
from llm_ii_engine.benchmark.experiment import (
    BenchmarkExperimentManifest,
    ExperimentResult,
    ExperimentRunner,
)
from llm_ii_engine.benchmark.repeatability import RepeatabilityAnalyzer, RepeatabilityReport
from llm_ii_engine.core.enums import (
    ClockSource,
    ExecutionMode,
    FailurePolicy,
    ProvenanceType,
    QuantizationType,
)
from llm_ii_engine.core.schemas import ModelSpec, WorkloadSpec
from llm_ii_engine.discovery.profiler import discover_hardware
from llm_ii_engine.storage.database import DatabaseManager
from llm_ii_engine.storage.repository import EngineRepository

# ============================================================================
# Deterministic Prompt Specifications
# ============================================================================

PROMPT_TEXT_32 = (
    "Explain the principle of cache locality in computer memory hierarchies, distinguishing between temporal and spatial locality."
)

PROMPT_TEXT_128 = (
    "Explain the principle of cache locality in computer memory hierarchies, distinguishing between temporal and spatial locality. "
    "Discuss why modern processor architectures utilize multi-level hierarchical caches (L1, L2, and L3) to bridge the persistent performance gap "
    "between high-frequency arithmetic logic units and high-latency main system memory. Describe how cache lines are typically structured and transferred "
    "in 64-byte blocks, and analyze the computational consequences of sequential array traversal versus random pointer chasing in linked structures."
)

PROMPT_TEXT_384 = (
    "Analyze the operational intensity of transformer neural network inference across both the prefill and autoregressive decode phases. "
    "In the prefill phase, the entire prompt sequence is processed simultaneously, allowing matrix-matrix multiplications (GEMM) that achieve "
    "high arithmetic intensity and saturate compute units (ALUs/tensor cores). In contrast, during the autoregressive decode phase, generation proceeds "
    "one token at a time, requiring matrix-vector operations (GEMV) where all model parameters must be read from memory for each single generated token. "
    "Discuss why the decode phase becomes severely memory-bandwidth bound rather than compute bound. Explain the role of the key-value (KV) cache "
    "in storing previous token representations to avoid redundant quadratic computation, and how the KV-cache size scales with batch size, sequence length, "
    "number of attention layers, and hidden dimensions. Finally, contrast standard multi-head attention (MHA) with grouped-query attention (GQA) and "
    "multi-query attention (MQA) in reducing memory bandwidth demands during decoding. Explain how KV-cache paging mechanisms like vLLM PageAttention "
    "mitigate external and internal memory fragmentation during multi-tenant serving, and why memory capacity often forms a hard upper ceiling "
    "on achievable batch concurrency before arithmetic execution units reach thermal or utilization limits. "
    "Examine how prompt caching enables recurrent system prompts to bypass the quadratic prefill computation entirely by reusing prefix KV states."
)

PROMPT_TEXT_768 = PROMPT_TEXT_384 + " " + (
    "In addition to single-node memory bandwidth bottlenecks, evaluate how distributed inference strategies alter throughput and latency scaling across clusters. "
    "Tensor parallelism partitions weight matrices across multiple accelerators within a high-bandwidth NVLink domain, requiring all-reduce communication "
    "at every transformer layer. Pipeline parallelism instead partitions layers sequentially across devices, introducing pipeline bubbles and communication "
    "at stage boundaries. Sequence parallelism further shards activation tensors to reduce per-device memory consumption for very long context lengths. "
    "Compare these parallelization paradigms against mixture-of-experts (MoE) architectures, where only a sparse subset of expert feed-forward networks "
    "is activated per token. Detail how all-to-all communication overhead across multi-node interconnects can degrade throughput if communication bandwidth "
    "is constrained, and why expert placement strategies are vital for maintaining high hardware utilization. "
    "Furthermore, consider the operational trade-offs of post-training quantization schemes. Weight-only quantization, such as INT4 or INT8 GPTQ and AWQ, "
    "reduces the total memory bandwidth required to load model parameters on each decoding iteration without increasing compute requirements. "
    "However, when batch sizes grow sufficiently large that matrix operations transition from memory-bound GEMV to compute-bound GEMM, "
    "weight-only dequantization overhead can introduce computational bottlenecks unless mixed-precision tensor cores natively support low-precision arithmetic. "
    "Analyze how activation quantization (e.g., FP8 or INT8 W8A8) addresses this limitation by accelerating both compute and memory transfer, "
    "and discuss the numerical stability challenges associated with dynamic activation outliers across deep attention layers. "
    "Finally, assess speculative decoding techniques where a smaller draft model generates candidate tokens that are subsequently verified in parallel "
    "by the target model during a single forward pass. Discuss why acceptance rates vary based on prompt domain, draft model capacity, and temperature, "
    "and under what conditions speculative execution succeeds in breaking the traditional memory-bandwidth wall of autoregressive decoding."
)

# 7 Controlled Expansion Workloads
WORKLOAD_P32_O32 = WorkloadSpec(
    prompt_tokens=32,
    output_tokens=32,
    prompt_text=PROMPT_TEXT_32,
    seed=101,
)

WORKLOAD_P32_O64 = WorkloadSpec(
    prompt_tokens=32,
    output_tokens=64,
    prompt_text=PROMPT_TEXT_32,
    seed=102,
)

WORKLOAD_P128_O64 = WorkloadSpec(
    prompt_tokens=128,
    output_tokens=64,
    prompt_text=PROMPT_TEXT_128,
    seed=103,
)

WORKLOAD_P128_O128 = WorkloadSpec(
    prompt_tokens=128,
    output_tokens=128,
    prompt_text=PROMPT_TEXT_128,
    seed=104,
)

WORKLOAD_P384_O64 = WorkloadSpec(
    prompt_tokens=384,
    output_tokens=64,
    prompt_text=PROMPT_TEXT_384,
    seed=105,
)

WORKLOAD_P768_O64 = WorkloadSpec(
    prompt_tokens=768,
    output_tokens=64,
    prompt_text=PROMPT_TEXT_768,
    seed=106,
)

WORKLOAD_P768_O128 = WorkloadSpec(
    prompt_tokens=768,
    output_tokens=128,
    prompt_text=PROMPT_TEXT_768,
    seed=107,
)

EXPANSION_SUITE: list[WorkloadSpec] = [
    WORKLOAD_P32_O32,
    WORKLOAD_P32_O64,
    WORKLOAD_P128_O64,
    WORKLOAD_P128_O128,
    WORKLOAD_P384_O64,
    WORKLOAD_P768_O64,
    WORKLOAD_P768_O128,
]


def create_expansion_manifest(
    hardware_report_id: str | None = None,
    cpu_model: str = "Intel(R) Core(TM) i7-14700 (14 threads active)",
    host_ram_gb: float = 31.69,
) -> BenchmarkExperimentManifest:
    """Create the official Session 13 controlled benchmark expansion manifest."""
    return BenchmarkExperimentManifest(
        experiment_id="exp-expansion-llama3-2-1b-s13",
        runtime_name="ollama",
        runtime_version="0.40.11",
        adapter_version="1.0.0",
        engine_version="1.0.0",
        model_id="llama3.2:1b",
        model_digest="baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878",
        quantization=QuantizationType.GGUF_Q8_0,
        context_window=131072,
        hardware_profile_id=hardware_report_id,
        device_name="CPU",
        cpu_model=cpu_model,
        host_ram_gb=host_ram_gb,
        workload_suite=list(EXPANSION_SUITE),
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
            "session": "13",
            "purpose": "Controlled benchmark expansion & empirical predictor validation",
            "workload_configurations_count": 7,
            "warmup_requests_count": 7,
            "measured_requests_count": 21,
            "total_requests_count": 28,
            "prompt_token_targets": [32, 128, 384, 768],
            "output_token_targets": [32, 64, 128],
        },
    )


async def execute_expansion_benchmark() -> tuple[ExperimentResult, RepeatabilityReport, list[EmpiricalObservationRecord]]:
    """Execute live Session 13 benchmark, persist outcomes, and export empirical records."""
    Path("storage/experiments").mkdir(parents=True, exist_ok=True)
    Path("storage/datasets").mkdir(parents=True, exist_ok=True)

    db_mgr = DatabaseManager("sqlite:///storage/llm_ii_engine.db")
    db_mgr.init_db()

    hw_report = discover_hardware()
    with db_mgr.get_session() as session:
        repo = EngineRepository(session)
        with contextlib.suppress(Exception):
            repo.save_hardware_report(hw_report)

    ram_gb = round(hw_report.memory.total_bytes / (1024**3), 2)
    manifest = create_expansion_manifest(
        hardware_report_id=hw_report.report_id,
        cpu_model=f"{hw_report.cpu.model_name} (14 threads active)",
        host_ram_gb=ram_gb,
    )
    manifest_path = "storage/experiments/expansion_llama3_2_1b_manifest.json"
    manifest.save(manifest_path)

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

    adapter = OllamaAdapter(base_url="http://127.0.0.1:11434", timeout_seconds=120.0)
    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)

    exp_result = await runner.run_experiment(manifest)
    await adapter.teardown()

    # Persist suite to DB
    with db_mgr.get_session() as session:
        repo = EngineRepository(session)
        try:
            repo.save_benchmark_suite(exp_result.suite_result, hardware_report_id=hw_report.report_id)
        except Exception as exc:
            print(f"Suite persistence note: {exc}")

    # Export records
    records = EmpiricalDatasetExporter.from_experiment_result(exp_result)
    EmpiricalDatasetExporter.export_to_jsonl(records, "storage/datasets/expansion_llama3_2_1b.jsonl", filter_real_only=True, exclude_warmup=True)
    EmpiricalDatasetExporter.export_to_csv(records, "storage/datasets/expansion_llama3_2_1b.csv", filter_real_only=True, exclude_warmup=True)
    EmpiricalDatasetExporter.export_failed_runs_to_jsonl(records, "storage/datasets/expansion_llama3_2_1b_failures.jsonl")
    EmpiricalDatasetExporter.export_failed_runs_to_csv(records, "storage/datasets/expansion_llama3_2_1b_failures.csv")

    # Repeatability analysis
    rep_report = RepeatabilityAnalyzer.analyze_experiment(exp_result)
    with open("storage/experiments/expansion_llama3_2_1b_repeatability.json", "w", encoding="utf-8") as f:
        f.write(rep_report.model_dump_json(indent=2))

    return exp_result, rep_report, records
