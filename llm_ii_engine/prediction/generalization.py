"""Out-of-Distribution (OOD) detection and generalization support envelope analysis.

Assesses whether candidate inference configurations fall within the convex hull or support
range of training data, emitting explicit warnings for extrapolated or unseen dimensions.
"""

from typing import Any

from llm_ii_engine.prediction.models import GeneralizationStatus, PredictionRecord


class OODDetector:
    """Evaluates input records against feature envelopes established during predictor training.

    Notice: OOD warnings report feature support boundaries; they do not represent
    calibrated error probabilities or Bayesian posterior uncertainties.
    """

    def __init__(
        self,
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        critical_features: list[str] | None = None,
    ) -> None:
        self.numerical_features = numerical_features or [
            "parameter_count_billion",
            "concurrency",
            "prompt_tokens",
            "output_tokens",
            "total_sequence_length",
            "bytes_per_parameter",
            "gpu_memory_bandwidth_gbps",
        ]
        self.categorical_features = categorical_features or [
            "model_id",
            "engine_type",
            "quantization",
            "hardware_catalog_id",
            "model_digest",
            "is_warmup",
        ]
        self.critical_features = critical_features or [
            "prompt_tokens",
            "output_tokens",
            "concurrency",
        ]

        self.numerical_ranges: dict[str, tuple[float, float]] = {}
        self.categorical_vocab: dict[str, set[Any]] = {}
        self.is_fitted = False

    def fit(self, records: list[PredictionRecord]) -> "OODDetector":
        """Compute support bounds and observed categories across training records."""
        if not records:
            self.is_fitted = True
            return self

        # 1. Numerical bounding boxes
        for feat in self.numerical_features:
            values: list[float] = []
            for r in records:
                val = getattr(r, feat, None)
                if val is not None and isinstance(val, (int, float)):
                    values.append(float(val))
            if values:
                self.numerical_ranges[feat] = (min(values), max(values))

        # 2. Categorical support sets
        for feat in self.categorical_features:
            cats: set[Any] = set()
            for r in records:
                val = getattr(r, feat, None)
                if val is not None:
                    cats.add(val)
            self.categorical_vocab[feat] = cats

        self.is_fitted = True
        return self

    def assess(self, record: PredictionRecord) -> GeneralizationStatus:
        """Inspect a candidate record against training support ranges."""
        if not self.is_fitted:
            return GeneralizationStatus(
                is_out_of_distribution=True,
                warnings=["OODDetector has not been fitted on training data."],
            )

        warnings: list[str] = []
        extrapolated: list[str] = []
        unseen: list[str] = []
        missing: list[str] = []

        # 1. Critical feature presence check
        for feat in self.critical_features:
            val = getattr(record, feat, None)
            if val is None:
                missing.append(feat)
                warnings.append(f"Critical feature '{feat}' is missing (None).")

        # 2. Numerical range check
        for feat, (min_v, max_v) in self.numerical_ranges.items():
            val = getattr(record, feat, None)
            if val is not None and isinstance(val, (int, float)):
                f_val = float(val)
                if f_val < min_v or f_val > max_v:
                    extrapolated.append(feat)
                    warnings.append(
                        f"Numerical feature '{feat}' value {f_val} falls outside training range [{min_v}, {max_v}]."
                    )

        # 3. Categorical support check
        for feat, known_cats in self.categorical_vocab.items():
            val = getattr(record, feat, None)
            if val is not None and known_cats and val not in known_cats:
                unseen.append(feat)
                warnings.append(
                    f"Categorical feature '{feat}' has unseen value '{val}' (training observed: {sorted(str(c) for c in known_cats)})."
                )

        is_ood = bool(extrapolated or unseen)

        return GeneralizationStatus(
            is_out_of_distribution=is_ood,
            warnings=warnings,
            extrapolated_features=extrapolated,
            unseen_categories=unseen,
            missing_features=missing,
        )
