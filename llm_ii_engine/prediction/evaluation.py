"""Data partitioning, split strategies, and honest evaluation metrics for prediction models.

Implements random, grouped (by suite), and holdout (by model/hardware) splits along with
MAE, RMSE, and MAPE metrics to assess in-sample and out-of-distribution generalization.
"""

import math
import random
from typing import Any

from llm_ii_engine.core.enums import ExecutionMode
from llm_ii_engine.prediction.baselines import BasePredictor
from llm_ii_engine.prediction.models import EvaluationResult, PredictionRecord


class DataSplitter:
    """Provides reproducible data partitioning strategies to evaluate generalization without leakage."""

    @classmethod
    def random_split(
        cls,
        records: list[PredictionRecord],
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> tuple[list[PredictionRecord], list[PredictionRecord]]:
        """Perform a standard deterministic random train/test split.

        Args:
            records: List of PredictionRecord instances.
            test_ratio: Fraction of records assigned to test set.
            seed: PRNG seed for deterministic reproducibility.

        Returns:
            Tuple of (train_records, test_records).
        """
        if not records:
            return [], []
        if not 0.0 < test_ratio < 1.0:
            raise ValueError(f"test_ratio must be strictly between 0 and 1, got {test_ratio}")

        rng = random.Random(seed)
        shuffled = records[:]
        rng.shuffle(shuffled)

        n_test = max(1, int(len(shuffled) * test_ratio))
        test_set = shuffled[:n_test]
        train_set = shuffled[n_test:]
        return train_set, test_set

    @classmethod
    def grouped_split(
        cls,
        records: list[PredictionRecord],
        group_key: str | Any = "suite_id",
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> tuple[list[PredictionRecord], list[PredictionRecord]]:
        """Perform a grouped partition ensuring all records from a group stay together.

        Keeps repeated runs or requests from the same benchmark suite or configuration entirely within
        either the training set or test set, preventing intra-group correlation leakage.

        Args:
            records: List of PredictionRecord instances.
            group_key: Attribute name, configuration alias, or callable used for grouping (default: 'suite_id').
            test_ratio: Approximate fraction of groups assigned to test set.
            seed: PRNG seed for reproducibility.

        Returns:
            Tuple of (train_records, test_records).
        """
        if not records:
            return [], []

        # Group records by attribute or callable extractor
        groups: dict[Any, list[PredictionRecord]] = {}
        for r in records:
            if callable(group_key):
                k = group_key(r)
            elif group_key in ("configuration", "workload_config", "config_key"):
                k = f"{r.prompt_tokens}x{r.requested_output_tokens or r.actual_output_tokens}"
            elif hasattr(r, str(group_key)):
                k = getattr(r, str(group_key))
            else:
                k = "unknown"
            groups.setdefault(k, []).append(r)

        unique_keys = sorted(list(groups.keys()), key=str)
        if len(unique_keys) < 2:
            raise ValueError(
                f"Cannot perform grouped split: dataset contains fewer than 2 distinct groups for '{group_key}' (found {len(unique_keys)}). "
                "A grouped split requires at least 2 distinct groups to prevent intra-group data leakage."
            )

        if not 0.0 < test_ratio < 1.0:
            raise ValueError(f"test_ratio must be strictly between 0 and 1, got {test_ratio}")

        rng = random.Random(seed)
        rng.shuffle(unique_keys)

        n_test_groups = min(len(unique_keys) - 1, max(1, int(len(unique_keys) * test_ratio)))
        test_keys = set(unique_keys[:n_test_groups])

        train_set: list[PredictionRecord] = []
        test_set: list[PredictionRecord] = []

        for k, group_recs in groups.items():
            if k in test_keys:
                test_set.extend(group_recs)
            else:
                train_set.extend(group_recs)

        return train_set, test_set

    @classmethod
    def holdout_split(
        cls,
        records: list[PredictionRecord],
        holdout_key: str = "model_id",
        holdout_values: list[Any] | None = None,
    ) -> tuple[list[PredictionRecord], list[PredictionRecord]]:
        """Partition by holding out specified models or hardware variants for out-of-distribution evaluation.

        Args:
            records: List of PredictionRecord instances.
            holdout_key: Attribute to filter on (e.g. 'model_id', 'hardware_catalog_id').
            holdout_values: Values designated strictly for test set.

        Returns:
            Tuple of (train_records, test_records).
        """
        if not records:
            return [], []

        target_vals = set(holdout_values or [])
        if not target_vals:
            raise ValueError("holdout_split requires non-empty holdout_values")

        train_set: list[PredictionRecord] = []
        test_set: list[PredictionRecord] = []

        for r in records:
            val = getattr(r, holdout_key, None)
            if val in target_vals:
                test_set.append(r)
            else:
                train_set.append(r)

        if not train_set:
            raise ValueError(
                f"Holdout split resulted in an empty training set: all records matched holdout values {target_vals} for key '{holdout_key}'."
            )
        if not test_set:
            raise ValueError(
                f"Holdout split resulted in an empty test set: no records matched holdout values {target_vals} for key '{holdout_key}'."
            )

        return train_set, test_set

    @classmethod
    def leave_one_group_out_split(
        cls,
        records: list[PredictionRecord],
        group_key: str | Any = "configuration",
    ) -> list[tuple[list[PredictionRecord], list[PredictionRecord], str]]:
        """Generate Leave-One-Group-Out partitions for cross-validation without leakage.

        Args:
            records: List of PredictionRecord instances.
            group_key: Attribute name, configuration alias, or callable used for grouping.

        Returns:
            List of (train_records, test_records, group_id) tuples.
        """
        if not records:
            return []

        # Group records by attribute or callable extractor
        groups: dict[str, list[PredictionRecord]] = {}
        for r in records:
            if callable(group_key):
                k = str(group_key(r))
            elif group_key in ("configuration", "workload_config", "config_key"):
                k = f"P{r.prompt_tokens}_O{r.requested_output_tokens or r.actual_output_tokens}"
            elif hasattr(r, str(group_key)):
                k = str(getattr(r, str(group_key)))
            else:
                k = "unknown"
            groups.setdefault(k, []).append(r)

        unique_keys = sorted(list(groups.keys()), key=str)
        if len(unique_keys) < 2:
            raise ValueError(
                f"Cannot perform leave-one-group-out split: dataset contains fewer than 2 distinct groups for '{group_key}' (found {len(unique_keys)})."
            )

        folds: list[tuple[list[PredictionRecord], list[PredictionRecord], str]] = []
        for holdout_k in unique_keys:
            test_set = groups[holdout_k]
            train_set = [r for k, recs in groups.items() if k != holdout_k for r in recs]
            folds.append((train_set, test_set, holdout_k))

        return folds


class PredictionEvaluator:
    """Evaluates prediction models against held-out benchmark partitions."""

    @classmethod
    def calculate_metrics(
        cls,
        actuals: list[float],
        predictions: list[float],
    ) -> tuple[float, float, float | None]:
        """Calculate MAE, RMSE, and MAPE with zero-target handling.

        Args:
            actuals: Ground truth target values.
            predictions: Predicted values.

        Returns:
            Tuple of (mae, rmse, mape_percent).
        """
        n = len(actuals)
        if n == 0 or len(predictions) != n:
            return 0.0, 0.0, None

        abs_errors = [abs(y - y_hat) for y, y_hat in zip(actuals, predictions)]
        sq_errors = [(y - y_hat) ** 2 for y, y_hat in zip(actuals, predictions)]

        mae = sum(abs_errors) / n
        rmse = math.sqrt(sum(sq_errors) / n)

        # MAPE: return None when any target is near zero (abs(y) < 1e-3) to avoid misleading percentage claims
        eps = 1e-3
        has_near_zero = any(abs(y) < eps for y in actuals)
        mape: float | None = None
        if not has_near_zero:
            rel_errors = [abs(y - y_hat) / abs(y) for y, y_hat in zip(actuals, predictions)]
            mape = (sum(rel_errors) / n) * 100.0

        return round(mae, 2), round(rmse, 2), (round(mape, 2) if mape is not None else None)

    @classmethod
    def evaluate(
        cls,
        predictor: BasePredictor,
        test_records: list[PredictionRecord],
        split_strategy: str = "HELD_OUT",
        baseline_predictor: BasePredictor | None = None,
    ) -> EvaluationResult:
        """Evaluate a fitted predictor on designated test records.

        Args:
            predictor: The predictor being evaluated.
            test_records: Held-out validation records.
            split_strategy: Label describing data partition method.
            baseline_predictor: Optional baseline (e.g. ConstantMedian) evaluated on identical split.

        Returns:
            Validated EvaluationResult with comparison metrics.
        """
        eligible: list[PredictionRecord] = []
        actuals: list[float] = []

        for r in test_records:
            val = predictor._extract_target_value(r)
            if val is not None and not math.isnan(val):
                eligible.append(r)
                actuals.append(val)

        sample_count = len(eligible)
        if sample_count == 0:
            return EvaluationResult(
                target=predictor.target,
                predictor_id=predictor.predictor_id,
                predictor_type=predictor.predictor_type,
                split_strategy=split_strategy,
                sample_count=0,
                test_sample_count=0,
                train_sample_count=predictor.training_sample_count,
                unit=predictor.unit,
                mae=0.0,
                rmse=0.0,
                mape_percent=None,
                baseline_mae=None,
                baseline_rmse=None,
                is_synthetic=(predictor.training_execution_mode == ExecutionMode.SIMULATED),
                execution_mode=predictor.training_execution_mode,
                sufficiency_threshold=10,
                is_statistically_sufficient=False,
                notes=f"No eligible test records available for target '{predictor.target.value}'.",
            )

        predictions: list[float] = []
        for r in eligible:
            out = predictor.predict(r)
            predictions.append(out.predicted_value)

        mae, rmse, mape = cls.calculate_metrics(actuals, predictions)

        # Baseline comparison on identical partition
        base_mae: float | None = None
        base_rmse: float | None = None
        if baseline_predictor is not None and baseline_predictor.is_fitted:
            base_preds = [baseline_predictor.predict(r).predicted_value for r in eligible]
            base_mae, base_rmse, _ = cls.calculate_metrics(actuals, base_preds)

        is_sufficient = sample_count >= 10
        notes = (
            f"Evaluated {sample_count} held-out records."
            if is_sufficient
            else f"Evaluation is exploratory: sample count (N={sample_count}) is inadequate for strong generalization claims."
        )

        is_synthetic = any(r.execution_mode == ExecutionMode.SIMULATED for r in eligible)
        mode = (
            ExecutionMode.REAL
            if all(r.execution_mode == ExecutionMode.REAL for r in eligible)
            else ExecutionMode.SIMULATED
        )

        return EvaluationResult(
            target=predictor.target,
            predictor_id=predictor.predictor_id,
            predictor_type=predictor.predictor_type,
            split_strategy=split_strategy,
            sample_count=sample_count,
            test_sample_count=sample_count,
            train_sample_count=predictor.training_sample_count,
            unit=predictor.unit,
            mae=mae,
            rmse=rmse,
            mape_percent=mape,
            baseline_mae=base_mae,
            baseline_rmse=base_rmse,
            is_synthetic=is_synthetic,
            execution_mode=mode,
            sufficiency_threshold=10,
            is_statistically_sufficient=is_sufficient,
            notes=notes,
        )

    @classmethod
    def evaluate_leave_one_group_out(
        cls,
        predictor_factory: Any,
        records: list[PredictionRecord],
        group_key: str | Any = "configuration",
        baseline_factory: Any | None = None,
    ) -> dict[str, Any]:
        """Perform Leave-One-Group-Out Cross-Validation across all independent groups.

        Trains a new predictor instance on N-1 groups and evaluates on the held-out group,
        aggregating per-group metrics and overall cross-validation performance.

        Args:
            predictor_factory: Callable taking training records and returning a fitted BasePredictor.
            records: List of PredictionRecord instances.
            group_key: Grouping key ('configuration', 'suite_id', etc.) or callable.
            baseline_factory: Optional callable returning a fitted baseline predictor.

        Returns:
            Dictionary containing per_group_results, aggregate_metrics, and residual statistics.
        """
        folds = DataSplitter.leave_one_group_out_split(records, group_key=group_key)
        if not folds:
            return {"num_groups": 0, "total_records": 0, "per_group_results": [], "aggregate_metrics": None}

        per_group_results: list[dict[str, Any]] = []
        all_actuals: list[float] = []
        all_predictions: list[float] = []
        all_baseline_predictions: list[float] = []

        for train_set, test_set, group_id in folds:
            predictor: BasePredictor = predictor_factory(train_set)
            baseline: BasePredictor | None = baseline_factory(train_set) if baseline_factory else None

            # Filter eligible test records
            eligible = [
                r
                for r in test_set
                if predictor._extract_target_value(r) is not None
                and not math.isnan(predictor._extract_target_value(r))
            ]
            if not eligible:
                continue

            actuals = [predictor._extract_target_value(r) for r in eligible]
            preds = [predictor.predict(r).predicted_value for r in eligible]
            all_actuals.extend(actuals)
            all_predictions.extend(preds)

            base_preds: list[float] = []
            if baseline is not None:
                base_preds = [baseline.predict(r).predicted_value for r in eligible]
                all_baseline_predictions.extend(base_preds)

            mae, rmse, mape = cls.calculate_metrics(actuals, preds)
            base_mae, base_rmse, base_mape = (
                cls.calculate_metrics(actuals, base_preds) if base_preds else (None, None, None)
            )

            residuals = [round(y - y_hat, 2) for y, y_hat in zip(actuals, preds)]
            abs_errors = [round(abs(y - y_hat), 2) for y, y_hat in zip(actuals, preds)]

            per_group_results.append({
                "group_id": group_id,
                "n_train": len(train_set),
                "n_test": len(eligible),
                "actual_mean": round(sum(actuals) / len(actuals), 2),
                "pred_mean": round(sum(preds) / len(preds), 2),
                "mae": mae,
                "rmse": rmse,
                "mape_percent": mape,
                "baseline_mae": base_mae,
                "residuals": residuals,
                "max_error": max(abs_errors) if abs_errors else 0.0,
                "mean_bias": round(sum(residuals) / len(residuals), 2) if residuals else 0.0,
            })

        agg_mae, agg_rmse, agg_mape = cls.calculate_metrics(all_actuals, all_predictions)
        base_agg_mae, base_agg_rmse, base_agg_mape = (
            cls.calculate_metrics(all_actuals, all_baseline_predictions)
            if all_baseline_predictions
            else (None, None, None)
        )

        all_residuals = [y - y_hat for y, y_hat in zip(all_actuals, all_predictions)]
        agg_bias = sum(all_residuals) / len(all_residuals) if all_residuals else 0.0
        max_err = max(abs(r) for r in all_residuals) if all_residuals else 0.0

        # Variation across held-out groups (std dev of group MAEs)
        group_maes = [g["mae"] for g in per_group_results]
        group_mae_std = (
            math.sqrt(
                sum((m - (sum(group_maes) / len(group_maes))) ** 2 for m in group_maes)
                / len(group_maes)
            )
            if group_maes
            else 0.0
        )

        return {
            "group_key": str(group_key),
            "num_groups": len(folds),
            "total_records": len(all_actuals),
            "per_group_results": per_group_results,
            "aggregate_metrics": {
                "mae": agg_mae,
                "rmse": agg_rmse,
                "mape_percent": agg_mape,
                "mean_bias": round(agg_bias, 2),
                "max_error": round(max_err, 2),
                "group_mae_std": round(group_mae_std, 2),
            },
            "baseline_aggregate_metrics": {
                "mae": base_agg_mae,
                "rmse": base_agg_rmse,
                "mape_percent": base_agg_mape,
            }
            if all_baseline_predictions
            else None,
        }
