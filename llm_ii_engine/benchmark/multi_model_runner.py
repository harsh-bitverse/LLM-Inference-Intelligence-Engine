"""Reusable multi-model benchmark runner for local LLM inference engines.

Executes the versioned workload suite across candidate models, records tokenizer-aware
normalized metrics, evaluates automated quality contracts, and exports comparison datasets.
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
from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.benchmark.multi_model_protocol import (
    BenchmarkExecutionCondition,
    BenchmarkSamplingConfig,
    ModelBenchmarkMetadata,
    MultiModelBenchmarkObservation,
    TaskNormalizedMetrics,
)
from llm_ii_engine.benchmark.workload_suite import (
    WORKLOAD_SUITE,
    BenchmarkWorkloadTask,
    evaluate_task_output,
)
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ModelResidencyState,
    ProvenanceType,
)
from llm_ii_engine.discovery.model_catalog import (
    CANDIDATE_MODELS,
    CandidateModelSpec,
    fetch_installed_ollama_models,
)


class MultiModelExperimentManifest(BaseModel):
    """Manifest describing a multi-model benchmark experiment run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Unique experiment identifier")
    created_at_utc: str = Field(description="ISO-8601 UTC creation timestamp")
    workstation_cpu: str = Field(default="Intel Core i7-14700")
    runtime_version: str = Field(description="Ollama server version")
    target_models: list[str] = Field(description="List of model IDs targeted")
    installed_models: list[str] = Field(description="List of models actually installed locally")
    uninstalled_models: list[str] = Field(description="List of candidate models not installed")
    tasks_executed: list[str] = Field(description="List of task IDs executed")
    observations: list[MultiModelBenchmarkObservation] = Field(
        default_factory=list, description="All measured benchmark observations"
    )
    summary_statistics: dict[str, Any] = Field(
        default_factory=dict, description="Aggregated summary statistics per model"
    )


class MultiModelRunner:
    """Orchestrates multi-model benchmarking against local Ollama runtime."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 60.0,
        storage_dir: Path | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.storage_dir = storage_dir or Path("storage")

    def run_experiment(
        self,
        target_model_ids: list[str],
        workload_tasks: list[BenchmarkWorkloadTask] | None = None,
        warmup: bool = True,
        repetitions: int = 1,
        max_budget_seconds: float = 1200.0,
        experiment_id_prefix: str = "exp-multi-model",
    ) -> MultiModelExperimentManifest:
        """Execute workload suite across candidate models, logging observations and quality results."""
        tasks = workload_tasks or WORKLOAD_SUITE
        installed_dict = fetch_installed_ollama_models(self.base_url)
        now_utc = datetime.now(UTC).isoformat()
        exp_id = f"{experiment_id_prefix}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

        installed_ids = [m for m in target_model_ids if m in installed_dict]
        uninstalled_ids = [m for m in target_model_ids if m not in installed_dict]

        observations: list[MultiModelBenchmarkObservation] = []
        obs_counter = 1
        start_time = time.perf_counter()

        with httpx.Client(
            base_url=self.base_url, timeout=self.timeout_seconds, trust_env=False
        ) as client:
            for rep in range(1, repetitions + 1):
                if (time.perf_counter() - start_time) > max_budget_seconds:
                    break

                # Rotate model order per repetition to minimize systematic thermal/order bias
                offset = (rep - 1) % len(installed_ids) if installed_ids else 0
                rotated_ids = installed_ids[offset:] + installed_ids[:offset]

                for model_id in rotated_ids:
                    if (time.perf_counter() - start_time) > max_budget_seconds:
                        break

                    installed_meta = installed_dict[model_id]
                    catalog_spec = next(
                        (m for m in CANDIDATE_MODELS if m.model_id == model_id),
                        CandidateModelSpec(
                            model_id=model_id,
                            display_name=model_id,
                            model_family=installed_meta.get("details", {}).get("family", "unknown"),
                            parameter_count_billion=1.0,
                            default_quantization=installed_meta.get("details", {}).get(
                                "quantization_level", "unknown"
                            ),
                            expected_disk_size_mb=round(installed_meta.get("size", 0) / (1024**2), 1),
                            expected_ram_mb=1500.0,
                            context_window=installed_meta.get("details", {}).get(
                                "context_length", 4096
                            ),
                            tokenizer_name="unknown",
                            vocabulary_size=128000,
                            license="Unspecified",
                            intended_evaluation_role="Dynamic model",
                            download_command=f"ollama pull {model_id}",
                        ),
                    )

                    meta = ModelBenchmarkMetadata(
                        model_id=model_id,
                        model_digest=installed_meta.get("digest"),
                        model_family=catalog_spec.model_family,
                        parameter_count_billion=catalog_spec.parameter_count_billion,
                        quantization=installed_meta.get("details", {}).get(
                            "quantization_level", catalog_spec.default_quantization
                        ),
                        model_size_bytes=installed_meta.get("size", int(catalog_spec.expected_disk_size_mb * 1024 * 1024)),
                        context_window=catalog_spec.context_window,
                        tokenizer_family=catalog_spec.tokenizer_name,
                        vocabulary_size=catalog_spec.vocabulary_size,
                        runtime_name="ollama",
                        runtime_version="0.40.11",
                        hardware_catalog_id="cpu-intel-i7-14700",
                        execution_backend="cpu_ggml_avx2",
                        license=catalog_spec.license,
                    )

                    sampling = BenchmarkSamplingConfig(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        seed=42,
                    )

                    # Warmup if requested before executing model task block
                    if warmup:
                        with contextlib.suppress(Exception):
                            client.post(
                                "/api/generate",
                                json={
                                    "model": model_id,
                                    "prompt": "Warmup request to ensure model weights reside in memory.",
                                    "options": {"num_predict": 4, "temperature": 0.0},
                                    "stream": False,
                                },
                            )

                    # Execute tasks sequentially
                    for task_idx, task in enumerate(tasks, start=1):
                        if (time.perf_counter() - start_time) > max_budget_seconds:
                            break

                        exec_condition = BenchmarkExecutionCondition(
                            concurrency=1,
                            model_residency=ModelResidencyState.WARM,
                            prompt_cache_state=CacheStateClassification.COLD_PREFIX,
                            warmup_policy="1_warmup_excluded",
                            request_index=obs_counter,
                            repetition_index=rep,
                        )

                        payload = {
                            "model": model_id,
                            "prompt": task.prompt,
                            "options": {
                                "num_predict": task.max_output_tokens,
                                "temperature": sampling.temperature,
                                "top_p": sampling.top_p,
                                "top_k": sampling.top_k,
                                "seed": sampling.seed,
                            },
                            "stream": False,
                        }

                        t0 = time.perf_counter()
                        resp = client.post("/api/generate", json=payload)
                        t1 = time.perf_counter()

                        client_e2e_ms = round((t1 - t0) * 1000.0, 3)

                        if resp.status_code == 200:
                            raw_data = resp.json()
                            response_text = raw_data.get("response", "")
                            prompt_tokens = raw_data.get("prompt_eval_count", 0)
                            output_tokens = raw_data.get("eval_count", 0)
                            server_total_ms = round(raw_data.get("total_duration", 0) / 1e6, 3)
                            server_load_ms = round(raw_data.get("load_duration", 0) / 1e6, 3)
                            server_prefill_ms = round(
                                raw_data.get("prompt_eval_duration", 0) / 1e6, 3
                            )
                            server_decode_ms = round(raw_data.get("eval_duration", 0) / 1e6, 3)

                            raw_decode_tps = (
                                round(output_tokens / (server_decode_ms / 1000.0), 2)
                                if server_decode_ms > 0
                                else None
                            )
                            raw_client_tps = (
                                round(output_tokens / (client_e2e_ms / 1000.0), 2)
                                if client_e2e_ms > 0
                                else None
                            )

                            char_count = len(response_text)
                            word_count = len(response_text.split())
                            client_sec = client_e2e_ms / 1000.0
                            chars_per_sec = round(char_count / client_sec, 2) if client_sec > 0 else 0.0
                            words_per_sec = round(word_count / client_sec, 2) if client_sec > 0 else 0.0

                            quality_result = evaluate_task_output(task, response_text)

                            metrics = TaskNormalizedMetrics(
                                client_e2e_duration_ms=client_e2e_ms,
                                server_total_duration_ms=server_total_ms,
                                server_load_duration_ms=server_load_ms,
                                server_prefill_duration_ms=server_prefill_ms,
                                server_decode_duration_ms=server_decode_ms,
                                raw_prompt_tokens=prompt_tokens,
                                raw_output_tokens=output_tokens,
                                raw_decode_tokens_per_second=raw_decode_tps,
                                raw_client_tokens_per_second=raw_client_tps,
                                task_elapsed_ms=client_e2e_ms,
                                characters_generated=char_count,
                                words_generated=word_count,
                                characters_per_second=chars_per_sec,
                                words_per_second=words_per_sec,
                                cross_tokenizer_comparable=False,
                            )
                        else:
                            response_text = ""
                            metrics = TaskNormalizedMetrics(
                                client_e2e_duration_ms=client_e2e_ms,
                                raw_prompt_tokens=0,
                                raw_output_tokens=0,
                                task_elapsed_ms=client_e2e_ms,
                                characters_generated=0,
                                words_generated=0,
                                characters_per_second=0.0,
                                words_per_second=0.0,
                                cross_tokenizer_comparable=False,
                            )
                            quality_result = evaluate_task_output(task, "")

                        obs = MultiModelBenchmarkObservation(
                            observation_id=f"obs-{exp_id}-{obs_counter:03d}",
                            timestamp_utc=datetime.now(UTC).isoformat(),
                            experiment_id=exp_id,
                            model_metadata=meta,
                            sampling_config=sampling,
                            execution_condition=exec_condition,
                            performance_metrics=metrics,
                            quality_result=quality_result,
                            process_rss_mb=None,
                            provenance=ProvenanceType.MEASURED,
                        )
                        observations.append(obs)
                        obs_counter += 1

        summary_stats = self._compute_summary(observations)

        manifest = MultiModelExperimentManifest(
            experiment_id=exp_id,
            created_at_utc=now_utc,
            runtime_version="0.40.11",
            target_models=target_model_ids,
            installed_models=installed_ids,
            uninstalled_models=uninstalled_ids,
            tasks_executed=[t.task_id for t in tasks],
            observations=observations,
            summary_statistics=summary_stats,
        )

        return manifest

    def _compute_summary(
        self, observations: list[MultiModelBenchmarkObservation]
    ) -> dict[str, Any]:
        """Compute aggregated performance and quality summaries per model and per task."""
        summary: dict[str, Any] = {}
        by_model: dict[str, list[MultiModelBenchmarkObservation]] = {}
        for obs in observations:
            by_model.setdefault(obs.model_metadata.model_id, []).append(obs)

        for model_id, obs_list in by_model.items():
            durations = [o.performance_metrics.client_e2e_duration_ms for o in obs_list]
            decode_rates = [
                o.performance_metrics.raw_decode_tokens_per_second
                for o in obs_list
                if o.performance_metrics.raw_decode_tokens_per_second is not None
            ]
            char_rates = [o.performance_metrics.characters_per_second for o in obs_list]
            correctness_scores = [
                o.quality_result.correctness_score
                for o in obs_list
                if o.quality_result.correctness_score is not None
            ]
            format_compliances = [
                o.quality_result.format_compliant
                for o in obs_list
                if o.quality_result.format_compliant is not None
            ]

            completed_reps = len({o.execution_condition.repetition_index for o in obs_list})

            # Per-task breakdown
            by_task: dict[str, list[MultiModelBenchmarkObservation]] = {}
            for o in obs_list:
                by_task.setdefault(o.quality_result.task_id, []).append(o)

            task_summaries: dict[str, Any] = {}
            for task_id, t_obs in by_task.items():
                t_durations = [o.performance_metrics.client_e2e_duration_ms for o in t_obs]
                t_decode_rates = [
                    o.performance_metrics.raw_decode_tokens_per_second
                    for o in t_obs
                    if o.performance_metrics.raw_decode_tokens_per_second is not None
                ]
                t_chars = [o.performance_metrics.characters_per_second for o in t_obs]
                t_out_toks = [o.performance_metrics.raw_output_tokens for o in t_obs]
                t_correctness = [
                    o.quality_result.correctness_score
                    for o in t_obs
                    if o.quality_result.correctness_score is not None
                ]
                t_formats = [
                    o.quality_result.format_compliant
                    for o in t_obs
                    if o.quality_result.format_compliant is not None
                ]
                t_succ = [
                    o.quality_result.task_success
                    for o in t_obs
                    if o.quality_result.task_success is not None
                ]

                task_summaries[task_id] = {
                    "successful_repetitions": len(t_obs),
                    "median_duration_ms": round(statistics.median(t_durations), 3) if t_durations else 0.0,
                    "mean_duration_ms": round(statistics.mean(t_durations), 3) if t_durations else 0.0,
                    "stddev_duration_ms": round(statistics.stdev(t_durations), 3) if len(t_durations) > 1 else 0.0,
                    "median_decode_tps": round(statistics.median(t_decode_rates), 2) if t_decode_rates else None,
                    "mean_characters_per_s": round(statistics.mean(t_chars), 2) if t_chars else 0.0,
                    "mean_output_tokens": round(statistics.mean(t_out_toks), 1) if t_out_toks else 0.0,
                    "quality_success_count": sum(1 for s in t_succ if s),
                    "format_compliance_rate_pct": round(
                        100.0 * sum(1 for c in t_formats if c) / len(t_formats), 1
                    ) if t_formats else None,
                    "mean_correctness_score": round(
                        statistics.mean(t_correctness), 3
                    ) if t_correctness else None,
                }

            summary[model_id] = {
                "task_count": len(obs_list),
                "repetitions": completed_reps,
                "median_duration_ms": round(statistics.median(durations), 3) if durations else 0.0,
                "mean_duration_ms": round(statistics.mean(durations), 3) if durations else 0.0,
                "mean_raw_decode_tok_s": round(sum(decode_rates) / len(decode_rates), 2)
                if decode_rates
                else None,
                "mean_characters_per_s": round(sum(char_rates) / len(char_rates), 2)
                if char_rates
                else 0.0,
                "mean_correctness_score": round(
                    sum(correctness_scores) / len(correctness_scores), 3
                )
                if correctness_scores
                else None,
                "format_compliance_rate_pct": round(
                    100.0 * sum(1 for c in format_compliances if c) / len(format_compliances), 1
                )
                if format_compliances
                else None,
                "task_summaries": task_summaries,
            }
        return summary

    def export_manifest(
        self, manifest: MultiModelExperimentManifest, output_path: Path
    ) -> None:
        """Serialize experiment manifest to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")

    def export_dataset_jsonl_csv(
        self, manifest: MultiModelExperimentManifest, jsonl_path: Path, csv_path: Path
    ) -> None:
        """Export tabular observations to JSONL and CSV for cross-model dataset analysis."""
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        with jsonl_path.open("w", encoding="utf-8") as jf:
            for obs in manifest.observations:
                row = {
                    "observation_id": obs.observation_id,
                    "experiment_id": obs.experiment_id,
                    "timestamp_utc": obs.timestamp_utc,
                    "request_index": obs.execution_condition.request_index,
                    "repetition_index": obs.execution_condition.repetition_index,
                    "model_id": obs.model_metadata.model_id,
                    "model_family": obs.model_metadata.model_family,
                    "parameter_count_billion": obs.model_metadata.parameter_count_billion,
                    "quantization": obs.model_metadata.quantization,
                    "task_id": obs.quality_result.task_id,
                    "task_type": obs.quality_result.task_type.value,
                    "client_e2e_duration_ms": obs.performance_metrics.client_e2e_duration_ms,
                    "server_prefill_duration_ms": obs.performance_metrics.server_prefill_duration_ms,
                    "server_decode_duration_ms": obs.performance_metrics.server_decode_duration_ms,
                    "raw_prompt_tokens": obs.performance_metrics.raw_prompt_tokens,
                    "raw_output_tokens": obs.performance_metrics.raw_output_tokens,
                    "raw_decode_tokens_per_second": obs.performance_metrics.raw_decode_tokens_per_second,
                    "characters_per_second": obs.performance_metrics.characters_per_second,
                    "words_per_second": obs.performance_metrics.words_per_second,
                    "format_compliant": obs.quality_result.format_compliant,
                    "task_success": obs.quality_result.task_success,
                    "correctness_score": obs.quality_result.correctness_score,
                    "evaluation_method": obs.quality_result.evaluation_method,
                    "provenance": obs.provenance.value,
                }
                jf.write(json.dumps(row) + "\n")
                rows.append(row)

        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
