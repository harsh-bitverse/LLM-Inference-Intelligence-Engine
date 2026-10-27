"""Multi-objective Pareto dominance engine and frontier extraction.

Implements rigorous Pareto dominance for mixed minimization and maximization objectives,
deterministic tie-breaking, trade-off explanation, and documented single-candidate selection policies.
"""

import math

from llm_ii_engine.optimizer.models import (
    CandidateEvaluation,
    FeasibilityStatus,
    ObjectiveDirection,
    OptimizationObjective,
    SelectionPolicy,
)


class ParetoEngine:
    """Computes non-dominated Pareto frontiers across multiple optimization objectives."""

    OBJECTIVE_DIRECTIONS: dict[OptimizationObjective, ObjectiveDirection] = {
        OptimizationObjective.MINIMIZE_TTFT: ObjectiveDirection.MINIMIZE,
        OptimizationObjective.MINIMIZE_TPOT: ObjectiveDirection.MINIMIZE,
        OptimizationObjective.MAXIMIZE_THROUGHPUT: ObjectiveDirection.MAXIMIZE,
        OptimizationObjective.MINIMIZE_VRAM: ObjectiveDirection.MINIMIZE,
        OptimizationObjective.MINIMIZE_COST: ObjectiveDirection.MINIMIZE,
        OptimizationObjective.MAXIMIZE_DECODE_THROUGHPUT: ObjectiveDirection.MAXIMIZE,
        OptimizationObjective.MINIMIZE_DURATION: ObjectiveDirection.MINIMIZE,
        OptimizationObjective.MINIMIZE_CLIENT_DURATION: ObjectiveDirection.MINIMIZE,
    }

    @classmethod
    def get_objective_direction(cls, objective: OptimizationObjective) -> ObjectiveDirection:
        """Return the optimization direction for a designated objective."""
        return cls.OBJECTIVE_DIRECTIONS.get(objective, ObjectiveDirection.MINIMIZE)

    @classmethod
    def _normalize_vector(
        cls,
        candidate: CandidateEvaluation,
        objectives: list[OptimizationObjective],
    ) -> list[float] | None:
        """Transform candidate objective values into a normalized space where all objectives are minimized.

        Returns None if any requested objective is missing or non-finite.
        """
        norm_vec: list[float] = []
        for obj in objectives:
            val = candidate.objective_values.get(obj)
            if val is None or math.isnan(val) or math.isinf(val):
                return None

            direction = cls.get_objective_direction(obj)
            if direction == ObjectiveDirection.MINIMIZE:
                norm_vec.append(val)
            else:
                # Maximization -> negate value so lower is better
                norm_vec.append(-val)

        return norm_vec

    @classmethod
    def dominates(
        cls,
        candidate_a: CandidateEvaluation,
        candidate_b: CandidateEvaluation,
        objectives: list[OptimizationObjective],
    ) -> bool:
        """Determine if Candidate A Pareto-dominates Candidate B (A ≻ B).

        Candidate A dominates Candidate B iff:
        1. A is at least as good as B on all objectives (norm_a[i] <= norm_b[i]).
        2. A is strictly better than B on at least one objective (norm_a[j] < norm_b[j]).

        Feasibility Rule:
        Infeasible candidates cannot dominate feasible candidates.
        """
        # Feasibility check
        if candidate_a.feasibility_status != FeasibilityStatus.FEASIBLE:
            return False
        if candidate_b.feasibility_status != FeasibilityStatus.FEASIBLE:
            # Feasible candidate A dominates infeasible candidate B
            return True

        vec_a = cls._normalize_vector(candidate_a, objectives)
        vec_b = cls._normalize_vector(candidate_b, objectives)

        if vec_a is None or vec_b is None:
            return False

        at_least_as_good = True
        strictly_better = False

        for a_val, b_val in zip(vec_a, vec_b):
            # Check with small numerical tolerance to avoid floating point precision artifacts
            if a_val > b_val + 1e-9:
                at_least_as_good = False
                break
            if a_val < b_val - 1e-9:
                strictly_better = True

        return at_least_as_good and strictly_better

    @classmethod
    def extract_pareto_frontier(
        cls,
        candidates: list[CandidateEvaluation],
        objectives: list[OptimizationObjective],
    ) -> list[CandidateEvaluation]:
        """Extract the non-dominated Pareto frontier from a list of candidate evaluations.

        Excludes infeasible candidates and candidates missing required objectives.
        Handles equal objective vectors deterministically.
        """
        if not candidates or not objectives:
            return []

        # 1. Filter to candidates that are feasible and have all required objectives
        eligible: list[CandidateEvaluation] = []
        for c in candidates:
            if c.feasibility_status != FeasibilityStatus.FEASIBLE:
                continue
            vec = cls._normalize_vector(c, objectives)
            if vec is not None:
                eligible.append(c)

        if not eligible:
            return []

        # 2. Non-dominated sort: candidate c is in frontier iff no other candidate dominates it
        frontier: list[CandidateEvaluation] = []

        for i, cand_a in enumerate(eligible):
            is_dominated = False
            for j, cand_b in enumerate(eligible):
                if i == j:
                    continue
                if cls.dominates(cand_b, cand_a, objectives):
                    is_dominated = True
                    break

            if not is_dominated:
                frontier.append(cand_a)

        # 3. Deduplicate configurations with identical recipe IDs
        unique_frontier: list[CandidateEvaluation] = []
        seen_config_ids: set[str] = set()

        for c in frontier:
            if c.config.config_id not in seen_config_ids:
                seen_config_ids.add(c.config.config_id)
                unique_frontier.append(c)

        # 4. Generate explainable trade-off summaries for each frontier candidate
        annotated_frontier: list[CandidateEvaluation] = []
        for c in unique_frontier:
            summary = cls._generate_trade_off_summary(c, unique_frontier, objectives)
            # Reconstruct model with updated trade_off_summary
            updated = c.model_copy(update={"trade_off_summary": summary})
            annotated_frontier.append(updated)

        return annotated_frontier

    @classmethod
    def _generate_trade_off_summary(
        cls,
        candidate: CandidateEvaluation,
        frontier: list[CandidateEvaluation],
        objectives: list[OptimizationObjective],
    ) -> str:
        """Explain why this configuration is non-dominated and what trade-offs it embodies."""
        if len(frontier) <= 1:
            return "Sole feasible configuration satisfying all hard constraints."

        advantages: list[str] = []
        for obj in objectives:
            val = candidate.objective_values.get(obj)
            if val is None:
                continue

            all_vals = [c.objective_values[obj] for c in frontier if obj in c.objective_values]
            if not all_vals:
                continue

            direction = cls.get_objective_direction(obj)
            best_val = min(all_vals) if direction == ObjectiveDirection.MINIMIZE else max(all_vals)
            unit = candidate.objective_units.get(obj, "")

            if abs(val - best_val) < 1e-6:
                if direction == ObjectiveDirection.MINIMIZE:
                    advantages.append(
                        f"lowest {obj.value.replace('minimize_', '')} ({val:.2f} {unit})"
                    )
                else:
                    advantages.append(
                        f"highest {obj.value.replace('maximize_', '')} ({val:.2f} {unit})"
                    )

        if advantages:
            base_summary = f"Pareto optimal: achieves {', '.join(advantages)} across the frontier."
        else:
            base_summary = "Pareto optimal: non-dominated multi-objective trade-off balance."

        if candidate.comparability_warning:
            return f"{base_summary} [{candidate.comparability_warning}]"
        return base_summary

    @classmethod
    def select_candidate(
        cls,
        frontier: list[CandidateEvaluation],
        objectives: list[OptimizationObjective],
        policy: SelectionPolicy,
        primary_objective: OptimizationObjective | None = None,
    ) -> tuple[CandidateEvaluation | None, str | None]:
        """Select a single candidate from the Pareto frontier according to an explicit policy.

        Args:
            frontier: Non-dominated Pareto frontier candidates.
            objectives: Objectives evaluated.
            policy: Selection policy to apply.
            primary_objective: Targeted objective if policy is PRIORITIZE_OBJECTIVE.

        Returns:
            Tuple of (Selected CandidateEvaluation or None, Rationale string).
        """
        if not frontier:
            return None, "Pareto frontier is empty; no candidate to select."

        def _with_caveat(cand: CandidateEvaluation, rat: str) -> tuple[CandidateEvaluation, str]:
            if cand.comparability_warning:
                return cand, f"{rat} [{cand.comparability_warning}]"
            return cand, rat

        if policy == SelectionPolicy.NONE:
            return None, "No single-candidate selection policy requested; full frontier returned."

        if policy == SelectionPolicy.PRIORITIZE_OBJECTIVE:
            target_obj = primary_objective or objectives[0]
            direction = cls.get_objective_direction(target_obj)

            candidates_with_target = [c for c in frontier if target_obj in c.objective_values]
            if not candidates_with_target:
                return _with_caveat(
                    frontier[0],
                    f"Target objective '{target_obj.value}' missing; defaulted to first frontier candidate.",
                )

            if direction == ObjectiveDirection.MINIMIZE:
                best_cand = min(
                    candidates_with_target, key=lambda c: c.objective_values[target_obj]
                )
            else:
                best_cand = max(
                    candidates_with_target, key=lambda c: c.objective_values[target_obj]
                )

            val = best_cand.objective_values[target_obj]
            unit = best_cand.objective_units.get(target_obj, "")
            rationale = (
                f"Selected by policy PRIORITIZE_OBJECTIVE for {target_obj.value}: "
                f"achieves optimal value of {val:.2f} {unit}."
            )
            return _with_caveat(best_cand, rationale)

        if policy == SelectionPolicy.LOWEST_COST:
            cost_obj = OptimizationObjective.MINIMIZE_COST
            candidates_with_cost = [c for c in frontier if cost_obj in c.objective_values]
            if not candidates_with_cost:
                return _with_caveat(
                    frontier[0],
                    "Cost objective was unavailable; defaulted to first frontier candidate.",
                )

            best_cand = min(candidates_with_cost, key=lambda c: c.objective_values[cost_obj])
            val = best_cand.objective_values[cost_obj]
            return _with_caveat(
                best_cand,
                f"Selected by policy LOWEST_COST: lowest cost of ${val:.4f}/1M tokens.",
            )

        if policy == SelectionPolicy.BALANCED_KNEE_POINT:
            # Utopia point distance minimization:
            # 1. Normalize each objective across the frontier to [0, 1] where 0 is ideal and 1 is worst.
            # 2. Compute Euclidean distance d = sqrt(sum(s_i^2)).
            # 3. Select candidate with minimum distance.
            best_cand = None
            min_dist = float("inf")

            # Compute min and max for each objective across frontier
            obj_bounds: dict[OptimizationObjective, tuple[float, float]] = {}
            for obj in objectives:
                vals = [c.objective_values[obj] for c in frontier if obj in c.objective_values]
                if vals:
                    obj_bounds[obj] = (min(vals), max(vals))

            for c in frontier:
                dist_sq = 0.0
                valid_count = 0
                for obj in objectives:
                    val = c.objective_values.get(obj)
                    if val is None or obj not in obj_bounds:
                        continue

                    min_val, max_val = obj_bounds[obj]
                    direction = cls.get_objective_direction(obj)

                    if max_val - min_val > 1e-9:
                        if direction == ObjectiveDirection.MINIMIZE:
                            # 0 is best (min_val), 1 is worst (max_val)
                            norm_s = (val - min_val) / (max_val - min_val)
                        else:
                            # 0 is best (max_val), 1 is worst (min_val)
                            norm_s = (max_val - val) / (max_val - min_val)
                    else:
                        norm_s = 0.0

                    dist_sq += norm_s**2
                    valid_count += 1

                dist = math.sqrt(dist_sq) if valid_count > 0 else 0.0
                if dist < min_dist:
                    min_dist = dist
                    best_cand = c

            rationale = (
                f"Selected by policy BALANCED_KNEE_POINT: closest candidate to the ideal utopia "
                f"point (normalized distance = {min_dist:.3f})."
            )
            return _with_caveat(best_cand or frontier[0], rationale)

        return _with_caveat(
            frontier[0], f"Defaulted to first frontier candidate under policy '{policy.value}'."
        )
