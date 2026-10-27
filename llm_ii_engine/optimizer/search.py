"""Bounded, reproducible search space candidate generation.

Generates candidate inference deployment configurations from discrete search spaces,
guards against combinatorial explosion, eliminates duplicates, and enforces deterministic ordering.
"""

from llm_ii_engine.core.enums import EngineType
from llm_ii_engine.optimizer.feasibility import FeasibilityChecker
from llm_ii_engine.optimizer.models import (
    CandidateConfig,
    OptimizationConstraints,
    SearchSpace,
)


class SearchSpaceGenerator:
    """Enumerates bounded, validated candidate configurations from a discrete SearchSpace."""

    @classmethod
    def generate(
        cls,
        search_space: SearchSpace,
        constraints: OptimizationConstraints | None = None,
    ) -> tuple[list[CandidateConfig], dict[str, int]]:
        """Generate a deterministic, deduplicated list of candidate configurations.

        Args:
            search_space: Bounded search space definition.
            constraints: Optional hard constraints to filter candidates early.

        Returns:
            Tuple of (list of CandidateConfig, dict of early rejection counts by reason).

        Raises:
            ValueError: If search space dimensions are empty or exceed max_candidates budget.
        """
        if not search_space.models:
            raise ValueError("SearchSpace must contain at least one ModelSpec.")
        if not search_space.quantizations:
            raise ValueError("SearchSpace must contain at least one QuantizationType.")
        if not search_space.concurrencies:
            raise ValueError("SearchSpace must contain at least one concurrency level.")
        if not search_space.engines:
            raise ValueError("SearchSpace must contain at least one EngineType.")
        if not search_space.tensor_parallel_sizes:
            raise ValueError("SearchSpace must contain at least one tensor parallel size.")

        # Calculate nominal combinatorial size before generation
        nominal_size = (
            len(search_space.models)
            * len(search_space.quantizations)
            * len(search_space.concurrencies)
            * len(search_space.engines)
            * len(search_space.tensor_parallel_sizes)
        )

        if nominal_size > search_space.max_candidates:
            raise ValueError(
                f"Search space nominal size ({nominal_size} candidates) exceeds the maximum safety "
                f"budget of {search_space.max_candidates}. Narrow the search dimensions to avoid combinatorial explosion."
            )

        candidates: list[CandidateConfig] = []
        seen_config_ids: set[str] = set()
        rejections: dict[str, int] = {}

        # Deterministic nested traversal
        # Sort models by model_id, quantizations by enum value, concurrencies ascending
        sorted_models = sorted(search_space.models, key=lambda m: m.model_id)
        sorted_quants = sorted(search_space.quantizations, key=lambda q: q.value)
        sorted_concs = sorted(search_space.concurrencies)
        sorted_engines = sorted(search_space.engines, key=lambda e: e.value)
        sorted_tps = sorted(search_space.tensor_parallel_sizes)

        for model in sorted_models:
            for quant in sorted_quants:
                for engine in sorted_engines:
                    # Early check: Engine-quantization compatibility
                    supported_q = FeasibilityChecker.SUPPORTED_QUANTIZATIONS_BY_ENGINE.get(
                        engine, set()
                    )
                    if quant not in supported_q:
                        reason = (
                            f"Engine '{engine.value}' does not support quantization '{quant.value}'"
                        )
                        rejections[reason] = rejections.get(reason, 0) + 1
                        continue

                    # Early check: Whitelist constraints
                    if constraints is not None:
                        if (
                            constraints.allowed_quantizations is not None
                            and quant not in constraints.allowed_quantizations
                        ):
                            reason = f"Quantization '{quant.value}' not in allowed list"
                            rejections[reason] = rejections.get(reason, 0) + 1
                            continue

                        if (
                            constraints.allowed_engines is not None
                            and engine not in constraints.allowed_engines
                        ):
                            reason = f"Engine '{engine.value}' not in allowed list"
                            rejections[reason] = rejections.get(reason, 0) + 1
                            continue

                    for conc in sorted_concs:
                        if (
                            constraints is not None
                            and constraints.max_concurrency is not None
                            and conc > constraints.max_concurrency
                        ):
                            reason = f"Concurrency {conc} exceeds max_concurrency ({constraints.max_concurrency})"
                            rejections[reason] = rejections.get(reason, 0) + 1
                            continue

                        for tp in sorted_tps:
                            if (
                                constraints is not None
                                and constraints.available_gpus is not None
                                and tp > constraints.available_gpus
                            ):
                                reason = f"Tensor parallel size {tp} exceeds available_gpus ({constraints.available_gpus})"
                                rejections[reason] = rejections.get(reason, 0) + 1
                                continue

                            if tp > 1 and engine in (EngineType.OLLAMA, EngineType.LLAMACPP):
                                reason = f"Engine '{engine.value}' does not support tensor parallelism (TP={tp})"
                                rejections[reason] = rejections.get(reason, 0) + 1
                                continue

                            config = CandidateConfig(
                                model_spec=model,
                                quantization=quant,
                                concurrency=conc,
                                engine_type=engine,
                                tensor_parallel_size=tp,
                            )

                            if config.config_id in seen_config_ids:
                                reason = "Duplicate configuration"
                                rejections[reason] = rejections.get(reason, 0) + 1
                                continue

                            seen_config_ids.add(config.config_id)
                            candidates.append(config)

                            if len(candidates) > search_space.max_candidates:
                                raise ValueError(
                                    f"Generated candidate count exceeded safety limit of {search_space.max_candidates}."
                                )

        return candidates, rejections
