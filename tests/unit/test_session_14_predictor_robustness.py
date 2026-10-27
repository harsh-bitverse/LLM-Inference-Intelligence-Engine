"""Unit tests for Session 14: Predictor Robustness, LOCO Cross-Validation & Evidence Calibration."""

import pytest

from llm_ii_engine.core.enums import (
    EngineType,
    ExecutionMode,
    ProvenanceType,
    QuantizationType,
    TokenCountSource,
)
from llm_ii_engine.core.schemas import ModelSpec, WorkloadSpec
from llm_ii_engine.optimizer.engine import ConfigurationOptimizer
from llm_ii_engine.optimizer.models import (
    CandidateConfig,
    FeasibilityStatus,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    SearchSpace,
)
from llm_ii_engine.prediction.baselines import (
    ConstantMedianPredictor,
    RidgeRegressionPredictor,
)
from llm_ii_engine.prediction.evaluation import DataSplitter, PredictionEvaluator
from llm_ii_engine.prediction.generalization import OODDetector
from llm_ii_engine.prediction.models import (
    PredictionRecord,
    PredictionTarget,
)


@pytest.fixture
def sample_records() -> list[PredictionRecord]:
    """Create synthetic records across 3 distinct configurations (2 repetitions each)."""
    configs = [
        (46, 32, 22.0, 1500.0, 150.0, 45.0, 1400.0),
        (46, 32, 22.4, 1480.0, 155.0, 47.0, 1390.0),
        (119, 64, 22.8, 2800.0, 180.0, 75.0, 2700.0),
        (119, 64, 22.6, 2820.0, 175.0, 78.0, 2710.0),
        (677, 64, 13.0, 5000.0, 2100.0, 2050.0, 2800.0),
        (677, 64, 12.6, 5100.0, 2150.0, 2080.0, 2820.0),
    ]
    records: list[PredictionRecord] = []
    for idx, (p, o, thru, dur, ttfc, s_pref, s_dec) in enumerate(configs):
        records.append(
            PredictionRecord(
                record_id=f"rec-{idx}",
                suite_id=f"suite-P{p}_O{o}",
                model_id="llama3.2:1b",
                model_digest="sha256:baf6a787",
                engine_type=EngineType.OLLAMA,
                execution_mode=ExecutionMode.REAL,
                provenance=ProvenanceType.MEASURED,
                is_warmup=False,
                concurrency=1,
                prompt_tokens=p,
                prompt_token_count_source=TokenCountSource.REQUESTED,
                requested_output_tokens=o,
                actual_output_tokens=o,
                output_tokens=o,
                output_tokens_per_second=thru,
                duration_ms=dur,
                ttfc_ms=ttfc,
                server_prompt_eval_duration_ms=s_pref,
                server_eval_duration_ms=s_dec,
                is_eligible_for_throughput=True,
                is_eligible_for_duration=True,
                is_eligible_for_ttfc=True,
                is_eligible_for_prefill=True,
                is_eligible_for_decode=True,
            )
        )
    return records


class TestLeaveOneConfigurationOutSplitting:
    """Tests for DataSplitter.leave_one_group_out_split."""

    def test_loco_split_creates_n_folds_without_leakage(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        folds = DataSplitter.leave_one_group_out_split(sample_records, group_key="configuration")
        assert len(folds) == 3

        for train_set, test_set, group_id in folds:
            # Check test set contains only records for the designated configuration
            test_cfgs = {f"P{r.prompt_tokens}_O{r.requested_output_tokens}" for r in test_set}
            assert test_cfgs == {group_id}

            # Check train set contains NO records from the designated configuration
            train_cfgs = {f"P{r.prompt_tokens}_O{r.requested_output_tokens}" for r in train_set}
            assert group_id not in train_cfgs

            # Check total records partition
            assert len(train_set) + len(test_set) == len(sample_records)

    def test_loco_split_raises_on_fewer_than_two_groups(self) -> None:
        single_group = [
            PredictionRecord(
                record_id="r1",
                suite_id="s1",
                model_id="m1",
                engine_type=EngineType.MOCK,
                execution_mode=ExecutionMode.SIMULATED,
                provenance=ProvenanceType.HEURISTIC,
                prompt_tokens=46,
                requested_output_tokens=32,
                actual_output_tokens=32,
            )
        ]
        with pytest.raises(ValueError, match="fewer than 2 distinct groups"):
            DataSplitter.leave_one_group_out_split(single_group, group_key="configuration")


class TestLOCOEvaluationAndMetrics:
    """Tests for PredictionEvaluator.evaluate_leave_one_group_out."""

    def test_evaluate_loco_throughput_constant_median(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        res = PredictionEvaluator.evaluate_leave_one_group_out(
            predictor_factory=lambda train: ConstantMedianPredictor(
                PredictionTarget.OUTPUT_THROUGHPUT_TOK_S
            ).fit(train),
            records=sample_records,
            group_key="configuration",
        )
        assert res["num_groups"] == 3
        assert res["total_records"] == 6
        assert len(res["per_group_results"]) == 3

        agg = res["aggregate_metrics"]
        assert agg["mae"] > 0.0
        assert agg["rmse"] >= agg["mae"]
        assert agg["mape_percent"] is not None
        assert "group_mae_std" in agg

    def test_evaluate_loco_duration_ridge_regression(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        res = PredictionEvaluator.evaluate_leave_one_group_out(
            predictor_factory=lambda train: RidgeRegressionPredictor(
                PredictionTarget.DURATION_MS,
                feature_names=["prompt_tokens", "requested_output_tokens"],
                alpha=1.0,
            ).fit(train),
            records=sample_records,
            group_key="configuration",
        )
        assert res["num_groups"] == 3
        assert res["total_records"] == 6
        agg = res["aggregate_metrics"]
        assert agg["mae"] > 0.0
        assert agg["mape_percent"] is not None


class TestProspectiveFeatureSafetyAndIntervals:
    """Tests for prospective latency features and evidence-aware uncertainty."""

    def test_prospective_features_do_not_leak_actual_output_tokens(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        predictor = RidgeRegressionPredictor(
            PredictionTarget.DURATION_MS,
            feature_names=["prompt_tokens", "requested_output_tokens"],
        ).fit(sample_records)

        # Record with requested_output_tokens=32 but unobserved actual_output_tokens=0 (pre-execution query)
        pre_exec_query = PredictionRecord(
            record_id="query-pre-exec",
            suite_id="eval-suite",
            model_id="llama3.2:1b",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            prompt_tokens=46,
            requested_output_tokens=32,
            actual_output_tokens=0,
            output_tokens=0,
        )

        out = predictor.predict(pre_exec_query)
        assert out.predicted_value > 0.0
        assert out.target == PredictionTarget.DURATION_MS

    def test_insufficient_sample_size_emits_exploratory_interval(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        # With N=6 (< 30), uncertainty interval must be flagged as exploratory
        predictor = ConstantMedianPredictor(PredictionTarget.OUTPUT_THROUGHPUT_TOK_S).fit(
            sample_records
        )
        out = predictor.predict(sample_records[0])
        assert out.interval_status == "EXPLORATORY_INSUFFICIENT_SAMPLE"
        assert any("Sample size is inadequate" in limitation for limitation in out.limitations)


class TestTargetEligibilityExpansion:
    """Tests for expanded target eligibility across latency and breakdown metrics."""

    def test_target_eligibility_helper_flags(self) -> None:
        rec = PredictionRecord(
            record_id="rec-test",
            suite_id="s1",
            model_id="llama3",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            is_eligible_for_throughput=True,
            is_eligible_for_duration=True,
            is_eligible_for_ttfc=True,
            is_eligible_for_prefill=True,
            is_eligible_for_decode=True,
            is_eligible_for_ttft=False,
            is_eligible_for_tpot=False,
            actual_output_tokens=10,
        )
        assert rec.is_eligible_for_target(PredictionTarget.OUTPUT_THROUGHPUT_TOK_S) is True
        assert rec.is_eligible_for_target(PredictionTarget.DURATION_MS) is True
        assert rec.is_eligible_for_target(PredictionTarget.TTFC_MS) is True
        assert rec.is_eligible_for_target(PredictionTarget.SERVER_PROMPT_EVAL_MS) is True
        assert rec.is_eligible_for_target(PredictionTarget.SERVER_EVAL_MS) is True
        assert rec.is_eligible_for_target(PredictionTarget.TTFT_MS) is False
        assert rec.is_eligible_for_target(PredictionTarget.TPOT_MS) is False


class TestOptimizerEvidenceMatchingHardening:
    """Tests for hardened _match_empirical_observation logic."""

    def test_optimizer_excludes_warmup_runs(self) -> None:
        warmup_rec = PredictionRecord(
            record_id="warmup-1",
            suite_id="s1",
            model_id="llama3.2:1b",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            is_warmup=True,
            prompt_tokens=46,
            requested_output_tokens=32,
            actual_output_tokens=32,
            output_tokens_per_second=11.0,  # Cold warmup with penalty
        )
        steady_rec = PredictionRecord(
            record_id="steady-1",
            suite_id="s1",
            model_id="llama3.2:1b",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            is_warmup=False,
            prompt_tokens=46,
            requested_output_tokens=32,
            actual_output_tokens=32,
            output_tokens_per_second=22.0,  # Steady-state
        )

        optimizer = ConfigurationOptimizer(empirical_observations=[warmup_rec, steady_rec])
        cand = CandidateConfig(
            model_spec=ModelSpec(model_id="llama3.2:1b", parameter_count_billion=1.235),
            quantization=QuantizationType.NONE,
            concurrency=1,
            engine_type=EngineType.OLLAMA,
        )
        req = OptimizationRequest(
            workload=WorkloadSpec(prompt_tokens=46, output_tokens=32),
            search_space=SearchSpace(models=[cand.model_spec]),
            objectives=[OptimizationObjective.MAXIMIZE_THROUGHPUT],
        )

        matched = optimizer._match_empirical_observation(cand, req, None)
        assert matched is not None
        assert matched.record_id == "steady-1"
        assert matched.output_tokens_per_second == 22.0

    def test_optimizer_aggregates_repetitions_with_median(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        optimizer = ConfigurationOptimizer(empirical_observations=sample_records)
        cand = CandidateConfig(
            model_spec=ModelSpec(model_id="llama3.2:1b", parameter_count_billion=1.235),
            quantization=QuantizationType.NONE,
            concurrency=1,
            engine_type=EngineType.OLLAMA,
        )
        req = OptimizationRequest(
            workload=WorkloadSpec(prompt_tokens=46, output_tokens=32),
            search_space=SearchSpace(models=[cand.model_spec]),
            objectives=[OptimizationObjective.MAXIMIZE_THROUGHPUT],
        )

        matched = optimizer._match_empirical_observation(cand, req, None)
        assert matched is not None
        assert matched.record_id.startswith("agg-median-2-reps")
        # 22.0 and 22.4 -> median = 22.2
        assert matched.output_tokens_per_second == pytest.approx(22.2, abs=1e-4)

    def test_optimizer_prompt_mismatch_does_not_assert_measured(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        optimizer = ConfigurationOptimizer(empirical_observations=sample_records)
        cand = CandidateConfig(
            model_spec=ModelSpec(model_id="llama3.2:1b", parameter_count_billion=1.235),
            quantization=QuantizationType.NONE,
            concurrency=1,
            engine_type=EngineType.OLLAMA,
        )
        # Workload prompt=250 has no match
        req = OptimizationRequest(
            workload=WorkloadSpec(prompt_tokens=250, output_tokens=32),
            search_space=SearchSpace(models=[cand.model_spec]),
            objectives=[OptimizationObjective.MAXIMIZE_THROUGHPUT],
        )

        matched = optimizer._match_empirical_observation(cand, req, None)
        assert matched is None

    def test_optimizer_unmeasured_ttft_sla_evaluates_unknown_feasibility(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        optimizer = ConfigurationOptimizer(empirical_observations=sample_records)
        model_spec = ModelSpec(model_id="llama3.2:1b", parameter_count_billion=1.235)
        req = OptimizationRequest(
            workload=WorkloadSpec(prompt_tokens=46, output_tokens=32),
            search_space=SearchSpace(
                models=[model_spec],
                quantizations=[QuantizationType.NONE],
                concurrencies=[1],
                engines=[EngineType.OLLAMA],
            ),
            objectives=[OptimizationObjective.MAXIMIZE_THROUGHPUT, OptimizationObjective.MINIMIZE_TTFT],
            constraints=OptimizationConstraints(max_ttft_ms=50.0),
        )

        res = optimizer.optimize(req)
        assert len(res.all_evaluations) == 1
        eval_cand = res.all_evaluations[0]
        assert eval_cand.feasibility_status == FeasibilityStatus.UNKNOWN_FEASIBILITY
        assert any("TTFT was not evaluated" in reason for reason in eval_cand.rejection_reasons)


class TestGeneralizationOODDetection:
    """Tests for out-of-distribution assessment."""

    def test_ood_detection_on_extrapolated_prompt_tokens(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        assessor = OODDetector(
            numerical_features=["prompt_tokens", "requested_output_tokens"],
            categorical_features=["model_id", "model_digest"],
        ).fit(sample_records)

        # In-distribution record
        in_dist = PredictionRecord(
            record_id="in-dist",
            suite_id="s1",
            model_id="llama3.2:1b",
            model_digest="sha256:baf6a787",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            prompt_tokens=119,
            requested_output_tokens=64,
            actual_output_tokens=64,
        )
        status_in = assessor.assess(in_dist)
        assert status_in.is_out_of_distribution is False

        # Out-of-distribution record (prompt_tokens=1500 > max 677)
        ood_rec = PredictionRecord(
            record_id="ood-rec",
            suite_id="s1",
            model_id="llama3.2:1b",
            model_digest="sha256:baf6a787",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            prompt_tokens=1500,
            requested_output_tokens=64,
            actual_output_tokens=64,
        )
        status_ood = assessor.assess(ood_rec)
        assert status_ood.is_out_of_distribution is True
        assert "prompt_tokens" in status_ood.extrapolated_features
        assert any("prompt_tokens" in w for w in status_ood.warnings)

    def test_ood_detection_on_unseen_model_digest(
        self, sample_records: list[PredictionRecord]
    ) -> None:
        assessor = OODDetector(
            numerical_features=["prompt_tokens"],
            categorical_features=["model_id", "model_digest"],
        ).fit(sample_records)

        unseen_digest_rec = PredictionRecord(
            record_id="unseen-digest",
            suite_id="s1",
            model_id="llama3.2:1b",
            model_digest="sha256:new_unseen_weights_hash",
            engine_type=EngineType.OLLAMA,
            execution_mode=ExecutionMode.REAL,
            provenance=ProvenanceType.MEASURED,
            prompt_tokens=46,
            requested_output_tokens=32,
            actual_output_tokens=32,
        )
        status = assessor.assess(unseen_digest_rec)
        assert status.is_out_of_distribution is True
        assert "model_digest" in status.unseen_categories
