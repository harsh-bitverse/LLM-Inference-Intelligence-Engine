"""Baseline performance predictors for LLM inference benchmarking.

Implements ConstantMedianPredictor, RidgeRegressionPredictor, and AnalyticalBaselinePredictor.
All predictors adhere to strict typing, no data leakage, and clear provenance labeling.
"""

import math
import statistics
import uuid
from abc import ABC, abstractmethod

from llm_ii_engine.analysis.resource_estimator import ModelResourceEstimator
from llm_ii_engine.analysis.roofline import RooflineAnalyzer
from llm_ii_engine.core.enums import ExecutionMode, ProvenanceType, QuantizationType
from llm_ii_engine.core.exceptions import IIEError
from llm_ii_engine.core.schemas import ModelSpec
from llm_ii_engine.discovery.catalog import match_gpu_name
from llm_ii_engine.discovery.models import GPUSpecification
from llm_ii_engine.prediction.generalization import OODDetector
from llm_ii_engine.prediction.models import (
    PredictionOutput,
    PredictionRecord,
    PredictionTarget,
)


class InsufficientDataError(IIEError):
    """Raised when training data is inadequate to fit a predictor."""


def _solve_linear_system(a_matrix: list[list[float]], b_vector: list[float]) -> list[float]:
    """Solve A * x = b via Gaussian elimination with partial pivoting.

    Args:
        a_matrix: Square matrix (n x n).
        b_vector: Target vector (n).

    Returns:
        Solution vector x (n).

    Raises:
        ValueError: If matrix is non-square or singular.
    """
    n = len(a_matrix)
    if any(len(row) != n for row in a_matrix):
        raise ValueError("Matrix A must be square")
    if len(b_vector) != n:
        raise ValueError(f"Vector b length ({len(b_vector)}) must match matrix dimension ({n})")

    # Check for non-finite values (NaN / Inf)
    for row in a_matrix:
        if any(math.isnan(v) or math.isinf(v) for v in row):
            raise ValueError("Matrix A contains non-finite (NaN or Inf) values")
    if any(math.isnan(v) or math.isinf(v) for v in b_vector):
        raise ValueError("Vector b contains non-finite (NaN or Inf) values")

    # Make working copies
    a = [row[:] for row in a_matrix]
    b = b_vector[:]

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_row = k
        max_val = abs(a[k][k])
        for i in range(k + 1, n):
            if abs(a[i][k]) > max_val:
                max_val = abs(a[i][k])
                max_row = i

        if max_val < 1e-12:
            raise ValueError("Singular matrix encountered during linear system solve")

        # Swap rows
        if max_row != k:
            a[k], a[max_row] = a[max_row], a[k]
            b[k], b[max_row] = b[max_row], b[k]

        # Eliminate entries below pivot
        for i in range(k + 1, n):
            factor = a[i][k] / a[k][k]
            a[i][k] = 0.0
            for j in range(k + 1, n):
                a[i][j] -= factor * a[k][j]
            b[i] -= factor * b[k]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(a[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_ax) / a[i][i]

    return x


class BasePredictor(ABC):
    """Abstract base class for inference performance predictors."""

    def __init__(self, target: PredictionTarget) -> None:
        self.target = target
        self.is_fitted = False
        self.training_sample_count = 0
        self.training_execution_mode = ExecutionMode.SIMULATED
        self.ood_detector = OODDetector()

    @property
    @abstractmethod
    def predictor_id(self) -> str:
        """Unique identifier for this predictor instance."""

    @property
    @abstractmethod
    def predictor_type(self) -> str:
        """Classification type of predictor."""

    @property
    def unit(self) -> str:
        """Physical unit of target metric."""
        units = {
            PredictionTarget.TTFT_MS: "ms",
            PredictionTarget.TPOT_MS: "ms",
            PredictionTarget.CHUNK_LATENCY_MS: "ms",
            PredictionTarget.OUTPUT_THROUGHPUT_TOK_S: "tok/s",
            PredictionTarget.PEAK_VRAM_MB: "MiB",
            PredictionTarget.DURATION_MS: "ms",
            PredictionTarget.TTFC_MS: "ms",
            PredictionTarget.SERVER_PROMPT_EVAL_MS: "ms",
            PredictionTarget.SERVER_EVAL_MS: "ms",
            PredictionTarget.CLIENT_E2E_DURATION_MS: "ms",
            PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S: "tok/s",
            PredictionTarget.SERVER_TOTAL_DURATION_MS: "ms",
            PredictionTarget.SERVER_LOAD_DURATION_MS: "ms",
            PredictionTarget.SERVER_PREFILL_DURATION_MS: "ms",
            PredictionTarget.SERVER_DECODE_DURATION_MS: "ms",
            PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S: "tok/s",
        }
        return units[self.target]

    @abstractmethod
    def fit(self, records: list[PredictionRecord]) -> "BasePredictor":
        """Fit model parameters on the provided training partition."""

    @abstractmethod
    def predict(self, record: PredictionRecord) -> PredictionOutput:
        """Generate a point prediction for a candidate input record."""

    def _extract_target_value(self, record: PredictionRecord) -> float | None:
        """Extract the numeric ground-truth target value from a record."""
        if self.target == PredictionTarget.TTFT_MS:
            return record.ttft_ms if record.is_eligible_for_ttft else None
        elif self.target == PredictionTarget.TPOT_MS:
            return record.tpot_ms if record.is_eligible_for_tpot else None
        elif self.target == PredictionTarget.CHUNK_LATENCY_MS:
            return record.chunk_latency_ms if record.is_eligible_for_chunk_latency else None
        elif self.target in (
            PredictionTarget.OUTPUT_THROUGHPUT_TOK_S,
            PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S,
        ):
            return (
                record.client_e2e_output_throughput_tok_s
                if record.client_e2e_output_throughput_tok_s is not None
                else record.output_tokens_per_second
            ) if record.is_eligible_for_throughput else None
        elif self.target == PredictionTarget.PEAK_VRAM_MB:
            return record.peak_vram_mb if record.is_eligible_for_vram else None
        elif self.target in (
            PredictionTarget.DURATION_MS,
            PredictionTarget.CLIENT_E2E_DURATION_MS,
        ):
            return (
                record.client_e2e_duration_ms
                if record.client_e2e_duration_ms is not None
                else record.duration_ms
            ) if record.is_eligible_for_duration else None
        elif self.target == PredictionTarget.TTFC_MS:
            return record.ttfc_ms if record.is_eligible_for_ttfc else None
        elif self.target in (
            PredictionTarget.SERVER_PROMPT_EVAL_MS,
            PredictionTarget.SERVER_PREFILL_DURATION_MS,
        ):
            return (
                record.server_prefill_duration_ms
                if record.server_prefill_duration_ms is not None
                else record.server_prompt_eval_duration_ms
            ) if record.is_eligible_for_prefill else None
        elif self.target in (
            PredictionTarget.SERVER_EVAL_MS,
            PredictionTarget.SERVER_DECODE_DURATION_MS,
        ):
            return (
                record.server_decode_duration_ms
                if record.server_decode_duration_ms is not None
                else record.server_eval_duration_ms
            ) if record.is_eligible_for_decode else None
        elif self.target == PredictionTarget.SERVER_TOTAL_DURATION_MS:
            return record.server_total_duration_ms if record.is_eligible_for_target(self.target) else None
        elif self.target == PredictionTarget.SERVER_LOAD_DURATION_MS:
            return record.server_load_duration_ms if record.is_eligible_for_target(self.target) else None
        elif self.target == PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S:
            return record.server_decode_throughput_tok_s if record.is_eligible_for_target(self.target) else None
        return None


class ConstantMedianPredictor(BasePredictor):
    """Non-parametric baseline that predicts the constant median of training observations."""

    def __init__(self, target: PredictionTarget) -> None:
        super().__init__(target)
        self.median_value: float = 0.0

    @property
    def predictor_id(self) -> str:
        return f"constant-median-{self.target.value}"

    @property
    def predictor_type(self) -> str:
        return "ConstantMedianBaseline"

    def fit(self, records: list[PredictionRecord]) -> "ConstantMedianPredictor":
        """Calculate the median of eligible training targets."""
        targets: list[float] = []
        for r in records:
            val = self._extract_target_value(r)
            if val is not None and not math.isnan(val):
                targets.append(val)

        if not targets:
            raise InsufficientDataError(
                f"Cannot fit ConstantMedianPredictor: 0 eligible observations for target '{self.target.value}'"
            )

        self.median_value = float(statistics.median(targets))
        self.training_sample_count = len(targets)
        # Inherit execution mode (REAL if any record is REAL, else SIMULATED)
        self.training_execution_mode = (
            ExecutionMode.REAL
            if any(r.execution_mode == ExecutionMode.REAL for r in records)
            else ExecutionMode.SIMULATED
        )
        self.ood_detector.fit(records)
        self.is_fitted = True
        return self

    def predict(self, record: PredictionRecord) -> PredictionOutput:
        """Return the training set median with generalization analysis."""
        if not self.is_fitted:
            raise IIEError("Predictor must be fitted before calling predict()")

        status = self.ood_detector.assess(record)
        val_status = (
            "VALIDATED_EMPIRICAL"
            if self.training_execution_mode == ExecutionMode.REAL
            and self.training_sample_count >= 10
            else "EXPLORATORY_INSUFFICIENT_DATA"
        )
        interval_status = (
            "CALIBRATED"
            if self.training_sample_count >= 30
            else "EXPLORATORY_INSUFFICIENT_SAMPLE"
        )
        limits = [
            "Constant median baseline does not model feature sensitivity or prompt/workload scaling.",
        ]
        if self.training_sample_count < 30:
            limits.append("Sample size is inadequate for calibrated prediction intervals.")

        return PredictionOutput(
            prediction_id=f"pred-{uuid.uuid4().hex[:8]}",
            target=self.target,
            predicted_value=round(self.median_value, 2),
            unit=self.unit,
            predictor_id=self.predictor_id,
            predictor_type=self.predictor_type,
            provenance=ProvenanceType.PREDICTED,
            execution_mode=self.training_execution_mode,
            generalization_status=status,
            training_sample_count=self.training_sample_count,
            validation_status=val_status,
            prediction_interval=None,
            interval_status=interval_status,
            limitations=limits,
            notes=f"Predicts constant training-set median ({self.median_value:.2f} {self.unit}).",
        )


class RidgeRegressionPredictor(BasePredictor):
    """Linear regression with L2 Ridge regularization and standardized features."""

    DEFAULT_FEATURES_BY_TARGET: dict[PredictionTarget, list[str]] = {
        PredictionTarget.TTFT_MS: [
            "parameter_count_billion",
            "prompt_tokens",
            "concurrency",
            "bytes_per_parameter",
        ],
        PredictionTarget.TPOT_MS: [
            "parameter_count_billion",
            "bytes_per_parameter",
            "concurrency",
            "requested_output_tokens",
        ],
        PredictionTarget.CHUNK_LATENCY_MS: [
            "concurrency",
            "requested_output_tokens",
        ],
        PredictionTarget.OUTPUT_THROUGHPUT_TOK_S: [
            "parameter_count_billion",
            "concurrency",
            "prompt_tokens",
            "requested_output_tokens",
        ],
        PredictionTarget.PEAK_VRAM_MB: [
            "parameter_count_billion",
            "bytes_per_parameter",
            "planned_sequence_length",
            "concurrency",
        ],
        PredictionTarget.DURATION_MS: [
            "prompt_tokens",
            "requested_output_tokens",
            "concurrency",
        ],
        PredictionTarget.TTFC_MS: [
            "prompt_tokens",
            "concurrency",
        ],
        PredictionTarget.SERVER_PROMPT_EVAL_MS: [
            "prompt_tokens",
            "concurrency",
        ],
        PredictionTarget.SERVER_EVAL_MS: [
            "requested_output_tokens",
            "concurrency",
        ],
        PredictionTarget.CLIENT_E2E_DURATION_MS: [
            "prompt_tokens",
            "requested_output_tokens",
            "concurrency",
        ],
        PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S: [
            "parameter_count_billion",
            "concurrency",
            "prompt_tokens",
            "requested_output_tokens",
        ],
        PredictionTarget.SERVER_TOTAL_DURATION_MS: [
            "prompt_tokens",
            "requested_output_tokens",
            "concurrency",
        ],
        PredictionTarget.SERVER_LOAD_DURATION_MS: [
            "parameter_count_billion",
            "bytes_per_parameter",
        ],
        PredictionTarget.SERVER_PREFILL_DURATION_MS: [
            "prompt_tokens",
            "concurrency",
        ],
        PredictionTarget.SERVER_DECODE_DURATION_MS: [
            "requested_output_tokens",
            "concurrency",
        ],
        PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S: [
            "parameter_count_billion",
            "concurrency",
            "requested_output_tokens",
        ],
    }

    POST_EXECUTION_FIELDS: frozenset[str] = frozenset({
        "server_prompt_eval_cached_count",
        "actual_output_tokens",
        "output_tokens",
        "chunk_count",
        "server_prompt_eval_duration_ms",
        "server_eval_duration_ms",
        "duration_ms",
        "ttfc_ms",
        "ttft_ms",
        "tpot_ms",
        "chunk_latency_ms",
        "output_tokens_per_second",
        "peak_vram_mb",
    })

    def __init__(
        self,
        target: PredictionTarget,
        feature_names: list[str] | None = None,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(target)
        features = feature_names or self.DEFAULT_FEATURES_BY_TARGET[target]
        forbidden = set(features).intersection(self.POST_EXECUTION_FIELDS)
        if forbidden:
            raise ValueError(
                f"Cannot use post-execution empirical telemetry fields {sorted(forbidden)} as prospective prediction features"
            )
        self.feature_names = features
        self.alpha = max(0.01, alpha)  # L2 regularization strength
        self.means: list[float] = []
        self.stds: list[float] = []
        self.weights: list[float] = []  # Length len(features) + 1 (last is intercept)

    @property
    def predictor_id(self) -> str:
        return f"ridge-regression-{self.target.value}"

    @property
    def predictor_type(self) -> str:
        return "RidgeRegression"

    def fit(self, records: list[PredictionRecord]) -> "RidgeRegressionPredictor":
        """Fit Ridge weights on training partition using Gaussian elimination."""
        raw_rows: list[list[float | None]] = []
        y_vals: list[float] = []
        valid_records: list[PredictionRecord] = []

        for r in records:
            y = self._extract_target_value(r)
            if y is None or math.isnan(y) or math.isinf(y):
                continue

            row: list[float | None] = []
            valid = True
            for f in self.feature_names:
                v = getattr(r, f, None)
                if v is None:
                    row.append(None)
                elif isinstance(v, (int, float)):
                    if math.isnan(v) or math.isinf(v):
                        valid = False
                        break
                    row.append(float(v))
                else:
                    valid = False
                    break

            if valid:
                raw_rows.append(row)
                y_vals.append(y)
                valid_records.append(r)

        if len(y_vals) < 3:
            raise InsufficientDataError(
                f"Cannot fit RidgeRegressionPredictor: requires >= 3 valid records, got {len(y_vals)}"
            )

        n_samples = len(y_vals)
        n_features = len(self.feature_names)

        # 1. Feature normalization (computed strictly on non-missing training entries)
        self.means = [0.0] * n_features
        self.stds = [0.0] * n_features

        for j in range(n_features):
            observed = [raw_rows[i][j] for i in range(n_samples) if raw_rows[i][j] is not None]
            if observed:
                mean_j = sum(observed) / len(observed)
                var_j = sum((val - mean_j) ** 2 for val in observed) / len(observed)
                std_j = math.sqrt(var_j)
                self.means[j] = mean_j
                self.stds[j] = std_j if std_j > 1e-9 else 1.0
            else:
                self.means[j] = 0.0
                self.stds[j] = 1.0

        # Standardize features (imputing missing training values with 0.0, the training mean) and add bias term (column 0 = 1.0)
        x_norm: list[list[float]] = []
        for i in range(n_samples):
            norm_row = [1.0]  # Intercept
            for j in range(n_features):
                val = raw_rows[i][j]
                if val is not None:
                    norm_row.append((val - self.means[j]) / self.stds[j])
                else:
                    norm_row.append(0.0)
            x_norm.append(norm_row)

        dim = n_features + 1

        # 2. Compute Normal Equations: A = (X^T * X + alpha * I), b = X^T * y
        a_matrix = [[0.0] * dim for _ in range(dim)]
        b_vec = [0.0] * dim

        for i in range(n_samples):
            row = x_norm[i]
            y = y_vals[i]
            for r_idx in range(dim):
                b_vec[r_idx] += row[r_idx] * y
                for c_idx in range(dim):
                    a_matrix[r_idx][c_idx] += row[r_idx] * row[c_idx]

        # Apply Ridge L2 regularization to feature weights (skip intercept at index 0)
        for j in range(1, dim):
            a_matrix[j][j] += self.alpha

        # 3. Solve for weights w
        try:
            self.weights = _solve_linear_system(a_matrix, b_vec)
        except ValueError as err:
            raise InsufficientDataError(f"Matrix inversion failed during Ridge fit: {err}") from err

        self.training_sample_count = n_samples
        self.training_execution_mode = (
            ExecutionMode.REAL
            if any(r.execution_mode == ExecutionMode.REAL for r in valid_records)
            else ExecutionMode.SIMULATED
        )
        self.ood_detector.fit(valid_records)
        self.is_fitted = True
        return self

    def predict(self, record: PredictionRecord) -> PredictionOutput:
        """Evaluate standardized regression equation for input record."""
        if not self.is_fitted:
            raise IIEError("Predictor must be fitted before calling predict()")

        status = self.ood_detector.assess(record)

        # Standardize features using training partition statistics
        # Missing features are imputed with the training mean (normalized value = 0.0)
        y_hat = self.weights[0]  # Intercept
        for j, feat in enumerate(self.feature_names):
            val = getattr(record, feat, None)
            if val is not None and isinstance(val, (int, float)):
                x_std = (float(val) - self.means[j]) / self.stds[j]
            else:
                x_std = 0.0  # Training mean imputation
            y_hat += self.weights[j + 1] * x_std

        # Physical metrics cannot be negative
        predicted_val = max(0.0, y_hat)

        val_status = (
            "VALIDATED_EMPIRICAL"
            if self.training_execution_mode == ExecutionMode.REAL
            and self.training_sample_count >= 10
            else "EXPLORATORY_INSUFFICIENT_DATA"
        )
        interval_status = (
            "CALIBRATED"
            if self.training_sample_count >= 30
            else "EXPLORATORY_INSUFFICIENT_SAMPLE"
        )
        limits = [
            "Linear ridge regression assumes monotonic linear relationships with L2 penalty.",
            "Missing input features are imputed with the training-partition mean.",
        ]
        if self.training_sample_count < 30:
            limits.append("Sample size is inadequate for calibrated prediction intervals.")

        return PredictionOutput(
            prediction_id=f"pred-{uuid.uuid4().hex[:8]}",
            target=self.target,
            predicted_value=round(predicted_val, 2),
            unit=self.unit,
            predictor_id=self.predictor_id,
            predictor_type=self.predictor_type,
            provenance=ProvenanceType.PREDICTED,
            execution_mode=self.training_execution_mode,
            generalization_status=status,
            training_sample_count=self.training_sample_count,
            validation_status=val_status,
            prediction_interval=None,
            interval_status=interval_status,
            limitations=limits,
            notes=f"Evaluated with features: {self.feature_names} (alpha={self.alpha}).",
        )


class AnalyticalBaselinePredictor(BasePredictor):
    """Theoretical baseline utilizing Session 05 architectural equations and Roofline ceilings.

    Notice: Output provenance is strictly HEURISTIC/PREDICTED (analytical).
    This baseline represents ideal hardware limits, not empirical model behavior.
    """

    def __init__(
        self,
        target: PredictionTarget,
        gpu_spec: GPUSpecification | None = None,
    ) -> None:
        super().__init__(target)
        self.gpu_spec = gpu_spec

    @property
    def predictor_id(self) -> str:
        return f"analytical-roofline-{self.target.value}"

    @property
    def predictor_type(self) -> str:
        return "AnalyticalRooflineBaseline"

    def fit(self, records: list[PredictionRecord]) -> "AnalyticalBaselinePredictor":
        """Fitting is a no-op for analytical equations, but establishes OOD envelopes."""
        self.training_sample_count = len(records)
        self.ood_detector.fit(records)
        self.is_fitted = True
        return self

    def predict(self, record: PredictionRecord) -> PredictionOutput:
        """Calculate theoretical expectation from Session 05 analytical equations."""
        status = self.ood_detector.assess(record)

        if record.parameter_count_billion is None:
            raise ValueError(
                "AnalyticalBaselinePredictor requires 'parameter_count_billion' to evaluate theoretical resource and roofline metrics."
            )

        if self.target == PredictionTarget.CHUNK_LATENCY_MS:
            raise ValueError(
                "AnalyticalBaselinePredictor does not support chunk latency: text chunks are transport/framing artifacts, not architectural hardware ceilings."
            )

        # Resolve GPU spec
        gpu_spec = self.gpu_spec
        if gpu_spec is None and record.hardware_catalog_id:
            gpu_spec = match_gpu_name(record.hardware_catalog_id)

        if gpu_spec is None and self.target in (
            PredictionTarget.TTFT_MS,
            PredictionTarget.TPOT_MS,
            PredictionTarget.OUTPUT_THROUGHPUT_TOK_S,
        ):
            raise ValueError(
                f"AnalyticalBaselinePredictor requires a resolved GPUSpecification or matched 'hardware_catalog_id' to evaluate theoretical ceilings for '{self.target.value}'."
            )

        p_count = record.parameter_count_billion
        prompt_len = record.prompt_tokens or 128
        out_len = (
            record.requested_output_tokens
            if record.requested_output_tokens is not None
            else (record.actual_output_tokens or 1)
        )
        batch = record.concurrency or 1

        # Synthesize ModelSpec
        model_spec = ModelSpec(
            model_id=record.model_id,
            parameter_count_billion=p_count,
            num_layers=record.num_layers,
            num_attention_heads=record.num_attention_heads,
            num_kv_heads=record.num_kv_heads,
            hidden_dimension=record.hidden_dimension,
            head_dimension=record.head_dimension,
            quantization=record.quantization,
        )

        val: float = 0.0
        limitations: list[str] = [
            "Analytical baseline calculates theoretical hardware ceiling, not empirical execution timing.",
            "Assumes 100% compute or bandwidth efficiency without runtime framework overhead.",
        ]

        # 1. Estimate resources
        res_est = ModelResourceEstimator.estimate_model_resources(
            model_spec=model_spec,
            prompt_tokens=prompt_len,
            output_tokens=out_len,
            batch_size=batch,
            weight_precision=record.quantization or QuantizationType.NONE,
        )

        roofline = None
        if gpu_spec is not None:
            oi = res_est.operational_intensity_decode or 1.0
            prec = "fp16_dense"
            if record.quantization in (QuantizationType.INT8, QuantizationType.GGUF_Q8_0):
                prec = "int8_dense"
            roofline = RooflineAnalyzer.evaluate_roofline(
                gpu_spec=gpu_spec,
                operational_intensity=oi,
                precision=prec,
                parameter_count_billion=p_count,
            )

        if self.target == PredictionTarget.TTFT_MS:
            # Prefill compute duration: FLOPs / Attainable Compute FLOP/s
            if roofline is None or not res_est.prefill_flops or roofline.peak_compute_tflops <= 0:
                raise ValueError(
                    f"Unable to calculate theoretical prefill FLOPs or Roofline peak compute for '{record.model_id}'."
                )
            val = (res_est.prefill_flops / (roofline.peak_compute_tflops * 1e12)) * 1000.0
        elif self.target == PredictionTarget.TPOT_MS:
            # Time per output token: 1000 / (tokens/second bound)
            if roofline is None or not roofline.theoretical_max_tokens_per_second:
                raise ValueError(
                    f"Unable to calculate theoretical decode token rate for '{record.model_id}'."
                )
            val = 1000.0 / roofline.theoretical_max_tokens_per_second
        elif self.target == PredictionTarget.OUTPUT_THROUGHPUT_TOK_S:
            if roofline is None or not roofline.theoretical_max_tokens_per_second:
                raise ValueError(
                    f"Unable to calculate theoretical max throughput for '{record.model_id}'."
                )
            val = roofline.theoretical_max_tokens_per_second
        elif self.target == PredictionTarget.PEAK_VRAM_MB:
            val = res_est.estimated_runtime_vram_mb or res_est.total_payload_memory_mb

        return PredictionOutput(
            prediction_id=f"pred-{uuid.uuid4().hex[:8]}",
            target=self.target,
            predicted_value=round(val, 2),
            unit=self.unit,
            predictor_id=self.predictor_id,
            predictor_type=self.predictor_type,
            provenance=ProvenanceType.HEURISTIC,
            execution_mode=ExecutionMode.SIMULATED,
            generalization_status=status,
            training_sample_count=self.training_sample_count,
            validation_status="THEORETICAL_ANALYTICAL",
            limitations=limitations,
            notes="Derived from Session 05 analytical resource models and theoretical roofline equations.",
        )
