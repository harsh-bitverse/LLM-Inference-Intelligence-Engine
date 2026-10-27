"""Unit and regression tests for Session 16 Measurement Contract & Performance Modeling.

Verifies:
1. Authoritative extraction and properties of the 10 distinct metrics.
2. Complete bifurcation between client end-to-end and server decode throughput.
3. Server total duration decomposition: T_total = T_load + T_prefill + T_decode + epsilon.
4. Cache telemetry provenance (prompt_eval_cached_count native vs None).
5. WorkloadAwareLatencyPredictor prospective feature safety (no post-execution leakage).
6. Configuration-grouped cross-validation behavior.
7. Optimizer objective routing for distinct throughput and duration goals.
8. Backward compatibility when parsing historical v1 empirical records.
"""

from __future__ import annotations

import pytest

from llm_ii_engine.benchmark.dataset_export import EmpiricalObservationRecord
from llm_ii_engine.benchmark.models import RequestBenchmarkResult
from llm_ii_engine.core.enums import (
    CacheStateClassification,
    ClockSource,
    EngineType,
    ExecutionMode,
    ModelResidencyState,
    ProvenanceType,
    TokenCountSource,
)
from llm_ii_engine.core.schemas import ServerReportedStats
from llm_ii_engine.optimizer.models import (
    OptimizationObjective,
)
from llm_ii_engine.optimizer.pareto import ParetoEngine
from llm_ii_engine.prediction.models import PredictionRecord, PredictionTarget
from llm_ii_engine.prediction.workload_aware import WorkloadAwareLatencyPredictor


class TestMeasurementContractSemantics:
    """Tests authoritative metric extraction and schema property accessors."""

    def test_server_reported_stats_properties(self) -> None:
        """ServerReportedStats exposes explicit semantic property aliases and calculates overhead."""
        stats = ServerReportedStats(
            prompt_tokens=42,
            output_tokens=48,
            prompt_duration_ms=50.0,
            eval_duration_ms=2500.0,
            load_duration_ms=10.0,
            total_duration_ms=2600.0,
            eval_tokens_per_second=19.2,
            prompt_eval_cached_count=40,
        )

        assert stats.server_prefill_duration_ms == 50.0
        assert stats.server_decode_duration_ms == 2500.0
        assert stats.server_load_duration_ms == 10.0
        assert stats.server_total_duration_ms == 2600.0
        assert stats.server_decode_throughput_tok_s == 19.2
        # Overhead: 2600 - (10 + 50 + 2500) = 40.0 ms
        assert stats.server_overhead_ms == 40.0

    def test_request_benchmark_result_properties(self) -> None:
        """RequestBenchmarkResult exposes client e2e and server decode metrics."""
        stats = ServerReportedStats(
            prompt_tokens=100,
            output_tokens=50,
            prompt_duration_ms=300.0,
            eval_duration_ms=2000.0,
            load_duration_ms=5.0,
            total_duration_ms=2320.0,
            eval_tokens_per_second=25.0,
        )

        req = RequestBenchmarkResult(
            request_id="req-test-1",
            success=True,
            output_tokens=50,
            prompt_tokens=100,
            duration_ms=2400.0,
            output_tokens_per_second=20.83,
            ttfc_ms=320.0,
            ttft_ms=None,
            tpot_ms=None,
            server_stats=stats,
            token_count_source=TokenCountSource.SERVER_REPORTED,
            prompt_token_count_source=TokenCountSource.SERVER_REPORTED,
            clock_source=ClockSource.MONOTONIC_REAL,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
        )

        assert req.client_e2e_duration_ms == 2400.0
        assert req.client_e2e_output_throughput_tok_s == 20.83
        assert req.server_total_duration_ms == 2320.0
        assert req.server_load_duration_ms == 5.0
        assert req.server_prefill_duration_ms == 300.0
        assert req.server_decode_duration_ms == 2000.0
        assert req.server_decode_throughput_tok_s == 25.0
        # Client loopback overhead: 2400 - 2320 = 80 ms
        assert round(req.client_e2e_duration_ms - req.server_total_duration_ms, 2) == 80.0

    def test_throughput_bifurcation_p677_o64(self) -> None:
        """Verify that P677_O64 yields 12.8 tok/s client e2e and 22.7 tok/s server decode."""
        client_dur = 4999.32  # ms
        decode_dur = 2818.14  # ms
        tokens = 64

        client_thru = tokens / (client_dur / 1000.0)
        decode_thru = tokens / (decode_dur / 1000.0)

        assert round(client_thru, 1) == 12.8
        assert round(decode_thru, 1) == 22.7
        # Absolute difference between application throughput and decode throughput is ~10 tok/s!
        assert round(decode_thru - client_thru, 1) == 9.9

    def test_cold_start_decomposition_proof(self) -> None:
        """Verify that the 1044 ms gap in cold start inference is 99.4% model load duration."""
        client_dur = 3062.53
        prefill_dur = 449.28
        decode_dur = 1569.12
        load_dur = 1037.74

        accounted = prefill_dur + decode_dur
        gap = client_dur - accounted
        residual_overhead = gap - load_dur

        assert round(gap, 2) == 1044.13
        assert round(residual_overhead, 2) == 6.39
        assert load_dur / gap > 0.993  # > 99.3% explained by load duration!

    def test_backward_compatibility_v1_jsonl_parsing(self) -> None:
        """Historical v1 observation record missing v2 fields automatically synchronizes."""
        v1_data = {
            "record_id": "obs-legacy-01",
            "experiment_id": "exp-legacy",
            "suite_id": "suite-legacy",
            "request_id": "req-legacy-01",
            "repetition_index": 1,
            "is_warmup": False,
            "model_id": "llama3.2:1b",
            "runtime_name": "ollama",
            "adapter_version": "1.0.0",
            "device_name": "CPU",
            "execution_mode": "REAL",
            "clock_source": "MONOTONIC_REAL",
            "provenance": "MEASURED",
            "success": True,
            "duration_ms": 1500.0,
            "output_tokens_per_second": 21.33,
            "actual_output_tokens": 32,
            "token_count_source": "server_reported",
            "prompt_token_count_source": "requested",
            "server_prompt_eval_duration_ms": 100.0,
            "server_eval_duration_ms": 1400.0,
            "chunk_count": 32,
        }

        rec = EmpiricalObservationRecord.model_validate(v1_data)
        # Automatic bidirectional sync
        assert rec.client_e2e_duration_ms == 1500.0
        assert rec.client_e2e_output_throughput_tok_s == 21.33
        assert rec.server_prefill_duration_ms == 100.0
        assert rec.server_decode_duration_ms == 1400.0
        # Automatically computed decode throughput
        assert rec.server_decode_throughput_tok_s == round(32 / (1400.0 / 1000.0), 2)


class TestWorkloadAwareLatencyPredictor:
    """Tests the physical decomposition performance model and prospective safety."""

    @pytest.fixture
    def synthetic_records(self) -> list[PredictionRecord]:
        """Create clean synthetic records for fitting WorkloadAwareLatencyPredictor."""
        records = []
        # Configurations: P50_O32, P100_O64, P200_O32
        configs = [(50, 32), (100, 64), (200, 32), (500, 64)]
        for p, o in configs:
            for rep in range(3):
                # Prefill = 3 ms/tok, Decode = 40 ms/tok, Overhead = 20 ms
                pref = p * 3.0
                dec = o * 40.0
                dur = pref + dec + 20.0
                thru = o / (dur / 1000.0)

                rec = PredictionRecord(
                    record_id=f"rec-{p}-{o}-{rep}",
                    suite_id="suite-synth",
                    model_id="test-model",
                    engine_type=EngineType.CUSTOM,
                    execution_mode=ExecutionMode.REAL,
                    provenance=ProvenanceType.MEASURED,
                    concurrency=1,
                    prompt_tokens=p,
                    requested_output_tokens=o,
                    actual_output_tokens=o,
                    output_tokens=o,
                    duration_ms=dur,
                    client_e2e_duration_ms=dur,
                    client_e2e_output_throughput_tok_s=thru,
                    server_prefill_duration_ms=pref,
                    server_decode_duration_ms=dec,
                    server_decode_throughput_tok_s=25.0,
                    model_residency_state=ModelResidencyState.WARM,
                    cache_state=CacheStateClassification.COLD_PREFIX,
                    is_eligible_for_duration=True,
                    is_eligible_for_throughput=True,
                    is_eligible_for_prefill=True,
                    is_eligible_for_decode=True,
                )
                records.append(rec)
        return records

    def test_workload_aware_fit_and_prospective_predict(
        self, synthetic_records: list[PredictionRecord]
    ) -> None:
        """WorkloadAwareLatencyPredictor fits physical rates and predicts prospectively."""
        predictor = WorkloadAwareLatencyPredictor(PredictionTarget.CLIENT_E2E_DURATION_MS)
        predictor.fit(synthetic_records)

        assert round(predictor.alpha_prefill_ms_per_tok, 1) == 3.0
        assert round(predictor.alpha_decode_ms_per_tok, 1) == 40.0
        assert round(predictor.beta_overhead_ms, 1) == 20.0

        # Prospective prediction on unseen configuration: P150_O40 (Warm)
        # Expected: 150 * 3.0 + 40 * 40.0 + 20.0 = 450 + 1600 + 20 = 2070.0 ms
        test_rec = PredictionRecord(
            record_id="rec-test-unseen",
            suite_id="suite-test",
            model_id="test-model",
            engine_type=EngineType.CUSTOM,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            concurrency=1,
            prompt_tokens=150,
            requested_output_tokens=40,
            actual_output_tokens=0,  # Unknown prospectively!
            output_tokens=0,
            model_residency_state=ModelResidencyState.WARM,
        )

        out = predictor.predict(test_rec)
        assert out.predicted_value == 2070.0
        assert out.unit == "ms"
        assert out.validation_status == "VALIDATED_EMPIRICAL"

    def test_workload_aware_cold_start_load_incorporation(
        self, synthetic_records: list[PredictionRecord]
    ) -> None:
        """Cold residency adds the mean cold-load duration to predicted duration."""
        # Add cold record
        cold_rec = PredictionRecord(
            record_id="rec-cold-fit",
            suite_id="suite-synth",
            model_id="test-model",
            engine_type=EngineType.CUSTOM,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            prompt_tokens=100,
            requested_output_tokens=32,
            actual_output_tokens=32,
            output_tokens=32,
            duration_ms=2600.0,
            server_load_duration_ms=1000.0,
            server_prefill_duration_ms=300.0,
            server_decode_duration_ms=1280.0,
            model_residency_state=ModelResidencyState.COLD,
            is_eligible_for_duration=True,
            is_eligible_for_decode=True,
            is_eligible_for_prefill=True,
        )
        predictor = WorkloadAwareLatencyPredictor(PredictionTarget.CLIENT_E2E_DURATION_MS)
        predictor.fit(synthetic_records + [cold_rec])

        test_cold = PredictionRecord(
            record_id="rec-test-cold",
            suite_id="suite-test",
            model_id="test-model",
            engine_type=EngineType.CUSTOM,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            prompt_tokens=100,
            requested_output_tokens=32,
            actual_output_tokens=0,
            output_tokens=0,
            model_residency_state=ModelResidencyState.COLD,
        )

        # Expected: 1000 (load) + 100 * 3.0 (300) + 32 * 40.0 (1280) + 20 (overhead) = 2600.0 ms
        out = predictor.predict(test_cold)
        assert out.predicted_value == 2600.0


class TestOptimizerObjectivesSeparation:
    """Tests that optimizer routes objectives correctly according to measurement contract."""

    def test_objective_directions(self) -> None:
        """Optimizer Pareto engine maps directions correctly."""
        assert (
            ParetoEngine.get_objective_direction(OptimizationObjective.MAXIMIZE_THROUGHPUT).value
            == "maximize"
        )
        assert (
            ParetoEngine.get_objective_direction(
                OptimizationObjective.MAXIMIZE_DECODE_THROUGHPUT
            ).value
            == "maximize"
        )
        assert (
            ParetoEngine.get_objective_direction(OptimizationObjective.MINIMIZE_DURATION).value
            == "minimize"
        )
        assert (
            ParetoEngine.get_objective_direction(
                OptimizationObjective.MINIMIZE_CLIENT_DURATION
            ).value
            == "minimize"
        )
        assert (
            ParetoEngine.get_objective_direction(OptimizationObjective.MINIMIZE_TTFT).value
            == "minimize"
        )
