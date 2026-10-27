"""Lightweight, interpretable, workload-aware latency predictor (Session 16).

Implements WorkloadAwareLatencyPredictor decomposing end-to-end latency into:
    T_e2e = T_load + T_prefill + T_decode + T_overhead
where:
    T_load = cold-start model load time (if cold residency, 0 if warm)
    T_prefill = alpha_prefill * prompt_tokens
    T_decode = alpha_decode * output_tokens
    T_overhead = constant client loopback/HTTP framing overhead

Strictly prohibits post-execution data leakage (never uses actual_output_tokens
or server_prompt_eval_cached_count during prospective prediction).
Provides both prospective prediction and retrospective post-hoc sanity analysis.
"""

from __future__ import annotations

import statistics
import uuid

from llm_ii_engine.core.enums import (
    ExecutionMode,
    ModelResidencyState,
    ProvenanceType,
)
from llm_ii_engine.prediction.baselines import BasePredictor, InsufficientDataError
from llm_ii_engine.prediction.generalization import OODDetector
from llm_ii_engine.prediction.models import (
    PredictionOutput,
    PredictionRecord,
    PredictionTarget,
)


class WorkloadAwareLatencyPredictor(BasePredictor):
    """Interpretable physical performance model separating load, prefill, decode, and overhead.

    Decomposition:
        T_e2e = T_load + (alpha_prefill * prompt_tokens) + (alpha_decode * output_tokens) + beta_overhead

    Parameters:
        alpha_prefill_ms_per_tok: Marginal prompt prefill latency in ms/token.
        alpha_decode_ms_per_tok: Marginal decode latency in ms/token.
        mean_cold_load_duration_ms: Empirical model loading time into memory in ms.
        beta_overhead_ms: Residual HTTP loopback and serialization overhead in ms.
        output_token_ratio: Empirical ratio of actual_output_tokens / requested_output_tokens.
    """

    POST_EXECUTION_FIELDS: frozenset[str] = frozenset({
        "server_prompt_eval_cached_count",
        "actual_output_tokens",
        "output_tokens",
        "chunk_count",
        "ttft_ms",
        "ttfc_ms",
        "tpot_ms",
        "chunk_latency_ms",
        "output_tokens_per_second",
        "client_e2e_output_throughput_tok_s",
        "server_decode_throughput_tok_s",
        "duration_ms",
        "client_e2e_duration_ms",
        "server_total_duration_ms",
        "server_load_duration_ms",
        "server_prefill_duration_ms",
        "server_decode_duration_ms",
        "server_prompt_eval_duration_ms",
        "server_eval_duration_ms",
        "peak_vram_mb",
    })

    def __init__(
        self,
        target: PredictionTarget = PredictionTarget.CLIENT_E2E_DURATION_MS,
    ) -> None:
        super().__init__(target)
        self.alpha_prefill_ms_per_tok: float = 3.0  # default prior ~330 tok/s prefill
        self.alpha_decode_ms_per_tok: float = 44.0  # default prior ~22.7 tok/s decode
        self.mean_cold_load_duration_ms: float = 1000.0  # default cold load ~1s
        self.beta_overhead_ms: float = 20.0  # default client loopback overhead
        self.output_token_ratio: float = 1.0
        self.residual_std_ms: float = 50.0

        # Sub-population statistics
        self.cold_count: int = 0
        self.warm_count: int = 0
        self.unknown_count: int = 0

    @property
    def predictor_id(self) -> str:
        return f"workload-aware-{self.target.value}"

    @property
    def predictor_type(self) -> str:
        return "WorkloadAwareLatencyPredictor"

    def fit(self, records: list[PredictionRecord]) -> WorkloadAwareLatencyPredictor:
        """Fit decomposition parameters from empirical training observations."""
        eligible = [r for r in records if r.is_eligible_for_target(self.target)]
        if len(eligible) < 2:
            raise InsufficientDataError(
                f"WorkloadAwareLatencyPredictor requires >= 2 eligible training records, got {len(eligible)}"
            )

        self.training_sample_count = len(eligible)
        self.training_execution_mode = (
            ExecutionMode.REAL
            if all(r.execution_mode == ExecutionMode.REAL for r in eligible)
            else ExecutionMode.SIMULATED
        )

        # 1. Fit cold load duration from cold-residency observations
        cold_loads: list[float] = []
        for r in eligible:
            if r.model_residency_state == ModelResidencyState.COLD:
                self.cold_count += 1
                if r.server_load_duration_ms is not None and r.server_load_duration_ms > 0:
                    cold_loads.append(r.server_load_duration_ms)
                elif r.server_total_duration_ms and r.server_prefill_duration_ms and r.server_decode_duration_ms:
                    # Unaccounted server load
                    est_load = r.server_total_duration_ms - (r.server_prefill_duration_ms + r.server_decode_duration_ms)
                    if est_load > 100.0:
                        cold_loads.append(est_load)
            elif r.model_residency_state == ModelResidencyState.WARM:
                self.warm_count += 1
            else:
                self.unknown_count += 1

        if cold_loads:
            self.mean_cold_load_duration_ms = statistics.mean(cold_loads)
        else:
            self.mean_cold_load_duration_ms = 1000.0

        # 2. Fit alpha_prefill (ms/token)
        prefill_rates: list[float] = []
        for r in eligible:
            pref_ms = r.server_prefill_duration_ms or r.server_prompt_eval_duration_ms
            p_tok = r.prompt_tokens
            if pref_ms and pref_ms > 0 and p_tok and p_tok > 0:
                prefill_rates.append(pref_ms / p_tok)

        if prefill_rates:
            self.alpha_prefill_ms_per_tok = statistics.median(prefill_rates)

        # 3. Fit alpha_decode (ms/token)
        decode_rates: list[float] = []
        ratios: list[float] = []
        for r in eligible:
            dec_ms = r.server_decode_duration_ms or r.server_eval_duration_ms
            out_tok = r.actual_output_tokens or r.output_tokens
            if dec_ms and dec_ms > 0 and out_tok and out_tok > 0:
                decode_rates.append(dec_ms / out_tok)
            req_tok = r.requested_output_tokens
            if req_tok and req_tok > 0 and out_tok and out_tok > 0:
                ratios.append(min(1.0, out_tok / req_tok))

        if decode_rates:
            self.alpha_decode_ms_per_tok = statistics.median(decode_rates)

        if ratios:
            self.output_token_ratio = statistics.mean(ratios)

        # 4. Fit beta_overhead (ms)
        overheads: list[float] = []
        residuals: list[float] = []
        for r in eligible:
            actual_duration = self._extract_target_value(r)
            if actual_duration is None:
                continue

            # Model components with known training features
            p_tok = r.prompt_tokens or 0
            act_out = r.actual_output_tokens or r.output_tokens or 0
            is_cold = (r.model_residency_state == ModelResidencyState.COLD)
            comp_load = self.mean_cold_load_duration_ms if is_cold else 0.0
            comp_pref = self.alpha_prefill_ms_per_tok * p_tok
            comp_dec = self.alpha_decode_ms_per_tok * act_out
            accounted = comp_load + comp_pref + comp_dec

            residual = actual_duration - accounted
            if residual > -200.0:  # exclude severe anomalies
                overheads.append(max(0.0, residual))
                residuals.append(residual)

        if overheads:
            self.beta_overhead_ms = max(5.0, statistics.median(overheads))
        else:
            self.beta_overhead_ms = 15.0

        if len(residuals) >= 2:
            self.residual_std_ms = max(10.0, statistics.stdev(residuals))
        else:
            self.residual_std_ms = 50.0

        # Fit OOD detector on prospective features only
        self.ood_detector = OODDetector(
            numerical_features=["prompt_tokens", "requested_output_tokens", "concurrency"],
            critical_features=["prompt_tokens", "concurrency"],
        )
        self.ood_detector.fit(eligible)

        self.is_fitted = True
        return self

    def predict(self, record: PredictionRecord) -> PredictionOutput:
        """Prospective prediction using expected output tokens (no post-execution leakage)."""
        return self._predict_internal(record, mode="expected")

    def predict_at_max_output(self, record: PredictionRecord) -> PredictionOutput:
        """Prospective prediction assuming generation runs to requested_output_tokens limit."""
        return self._predict_internal(record, mode="max_output")

    def predict_expected_output(
        self,
        record: PredictionRecord,
        expected_tokens: int | None = None,
        ratio: float | None = None,
    ) -> PredictionOutput:
        """Prospective prediction using specified expected output tokens or custom ratio."""
        return self._predict_internal(record, mode="custom", custom_tokens=expected_tokens, custom_ratio=ratio)

    def predict_retrospective(self, record: PredictionRecord) -> PredictionOutput:
        """Post-hoc retrospective prediction using actual_output_tokens to evaluate physical model error."""
        return self._predict_internal(record, mode="retrospective")

    def _predict_internal(
        self,
        record: PredictionRecord,
        mode: str = "expected",
        custom_tokens: int | None = None,
        custom_ratio: float | None = None,
    ) -> PredictionOutput:
        """Internal parameterized prediction core."""
        if not self.is_fitted:
            raise InsufficientDataError("Cannot predict before fitting the WorkloadAwareLatencyPredictor.")

        gen_status = self.ood_detector.assess(record)

        # Prospective prompt tokens
        prompt_tokens = record.prompt_tokens or 0

        # Determine output tokens based on prediction mode
        if mode == "retrospective":
            output_tokens = record.actual_output_tokens or record.output_tokens or 0
        elif mode == "max_output":
            output_tokens = record.requested_output_tokens or 64
        elif mode == "custom":
            if custom_tokens is not None:
                output_tokens = custom_tokens
            elif custom_ratio is not None and record.requested_output_tokens:
                output_tokens = max(1, int(record.requested_output_tokens * custom_ratio))
            else:
                output_tokens = record.requested_output_tokens or 64
        else:  # mode == "expected"
            req = record.requested_output_tokens or 64
            output_tokens = max(1, int(req * self.output_token_ratio))

        # Model residency load component
        if record.model_residency_state == ModelResidencyState.COLD:
            t_load = self.mean_cold_load_duration_ms
        elif record.model_residency_state == ModelResidencyState.WARM:
            t_load = 0.0
        else:
            # UNKNOWN: default to warm baseline and disclose assumption
            t_load = 0.0

        t_prefill = self.alpha_prefill_ms_per_tok * prompt_tokens
        t_decode = self.alpha_decode_ms_per_tok * output_tokens
        t_overhead = self.beta_overhead_ms

        t_e2e = t_load + t_prefill + t_decode + t_overhead

        # Map to target metric if specific component is requested
        if self.target in (
            PredictionTarget.CLIENT_E2E_DURATION_MS,
            PredictionTarget.DURATION_MS,
            PredictionTarget.SERVER_TOTAL_DURATION_MS,
        ):
            predicted_value = t_e2e if self.target != PredictionTarget.SERVER_TOTAL_DURATION_MS else (t_e2e - t_overhead)
        elif self.target in (
            PredictionTarget.SERVER_PREFILL_DURATION_MS,
            PredictionTarget.SERVER_PROMPT_EVAL_MS,
        ):
            predicted_value = t_prefill
        elif self.target in (
            PredictionTarget.SERVER_DECODE_DURATION_MS,
            PredictionTarget.SERVER_EVAL_MS,
        ):
            predicted_value = t_decode
        elif self.target == PredictionTarget.SERVER_LOAD_DURATION_MS:
            predicted_value = t_load
        elif self.target in (
            PredictionTarget.CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S,
            PredictionTarget.OUTPUT_THROUGHPUT_TOK_S,
        ):
            predicted_value = (output_tokens / (t_e2e / 1000.0)) if t_e2e > 0 else 0.0
        elif self.target == PredictionTarget.SERVER_DECODE_THROUGHPUT_TOK_S:
            predicted_value = (1000.0 / self.alpha_decode_ms_per_tok) if self.alpha_decode_ms_per_tok > 0 else 0.0
        else:
            predicted_value = t_e2e

        predicted_value = max(0.0, round(predicted_value, 2))

        # Calibrated prediction interval
        half_width = 1.96 * self.residual_std_ms
        lower_bound = max(0.0, round(predicted_value - half_width, 2))
        upper_bound = round(predicted_value + half_width, 2)

        pred_interval = (lower_bound, upper_bound)
        interval_status = (
            "EXPLORATORY_EXTRAPOLATED"
            if gen_status.is_out_of_distribution
            else "CALIBRATED"
        )

        val_status = (
            "EXPLORATORY_INSUFFICIENT_DATA"
            if gen_status.is_out_of_distribution
            else "VALIDATED_EMPIRICAL"
        )

        notes_parts = [
            f"Workload-Aware decomposition (mode={mode}):",
            f"load={t_load:.1f}ms,",
            f"prefill={t_prefill:.1f}ms ({self.alpha_prefill_ms_per_tok:.2f}ms/tok * {prompt_tokens}tok),",
            f"decode={t_decode:.1f}ms ({self.alpha_decode_ms_per_tok:.2f}ms/tok * {output_tokens}tok),",
            f"overhead={t_overhead:.1f}ms.",
        ]

        limits = [
            "Decomposition assumes linear prefill scaling and constant autoregressive decode rate.",
            "Cold load time applies only when model_residency_state is explicitly COLD.",
        ]
        if mode != "retrospective":
            limits.append("Prospective prediction: actual output token count is uncertain prior to execution.")

        return PredictionOutput(
            prediction_id=f"wapred-{uuid.uuid4().hex[:8]}",
            target=self.target,
            predicted_value=predicted_value,
            unit=self.unit,
            predictor_id=self.predictor_id,
            predictor_type=self.predictor_type,
            provenance=ProvenanceType.PREDICTED,
            execution_mode=self.training_execution_mode,
            generalization_status=gen_status,
            training_sample_count=self.training_sample_count,
            validation_status=val_status,
            prediction_interval=pred_interval,
            interval_status=interval_status,
            limitations=limits,
            notes=" ".join(notes_parts),
        )
