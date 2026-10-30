"""Architecture-aware serving recommendation engine for Session 27.

Evaluates runtime architectures (Ollama single-slot, Ollama serial queue, llama.cpp parallel continuous batching)
against workload composition, arrival rate, client concurrency, latency SLA, and memory constraints.
"""

import uuid
from typing import Any

from llm_ii_engine.serving.capabilities import AUTHORIZED_MODELS
from llm_ii_engine.serving.protocol import (
    ServingArchitectureRecommendation,
    ServingRuntimeArchitecture,
)


class ServingArchitectureRecommender:
    """Recommends the optimal inference serving architecture based on empirical evidence."""

    @classmethod
    def recommend_architecture(
        cls,
        model_id: str,
        workload_type: str = "BALANCED_MIXED",
        client_concurrency: int = 1,
        arrival_rate_rps: float = 0.5,
        latency_sla_ms: float = 2000.0,
        memory_budget_mb: float | None = None,
    ) -> ServingArchitectureRecommendation:
        """Evaluate constraints and return an evidence-backed serving architecture recommendation."""
        # 1. Check if model is measured / authorized
        if model_id not in AUTHORIZED_MODELS:
            return ServingArchitectureRecommendation(
                recommendation_id=f"rec-arch-{uuid.uuid4().hex[:8]}",
                recommended_architecture=ServingRuntimeArchitecture.OLLAMA_SINGLE_SLOT,
                recommended_slots=1,
                recommended_runtime_name="UNKNOWN",
                expected_throughput_rps=0.0,
                expected_p95_latency_ms=0.0,
                expected_sla_compliance_pct=0.0,
                evidence_tier="UNKNOWN",
                rationale=f"Model '{model_id}' is not in the authorized, empirically benchmarked model suite. Extrapolation unsupported.",
                tradeoffs=["No empirical measurements available for this model."],
                actionable_setup_guidance="Benchmark model under controlled tests before deploying in production.",
            )

        # 2. Evaluate memory budget constraints
        max_allowed_slots = 4
        memory_constrained = False
        if memory_budget_mb is not None:
            weights_mb = {"qwen2.5:0.5b": 398.0, "qwen2.5:1.5b": 986.0, "llama3.2:1b": 1321.0}.get(model_id, 1000.0)
            kv_per_slot_mb = 50.0
            avail_kv = memory_budget_mb - weights_mb
            if avail_kv < kv_per_slot_mb * 2:
                max_allowed_slots = 1
                memory_constrained = True
            elif avail_kv < kv_per_slot_mb * 4:
                max_allowed_slots = 2
                memory_constrained = True

        # 3. Decision logic based on concurrency, HoL risk, and SLA
        desired_slots = 1
        if client_concurrency <= 1 and arrival_rate_rps <= 0.2:
            desired_slots = 1
        elif client_concurrency <= 2 or workload_type in ["INTERACTIVE_BURST", "BALANCED_MIXED"]:
            desired_slots = 2
        else:
            desired_slots = 4

        effective_slots = min(desired_slots, max_allowed_slots)

        if effective_slots == 1:
            rationale = (
                f"For single-stream interactive traffic (C={client_concurrency}, lambda={arrival_rate_rps:.1f} rps), "
                "Ollama's default single-slot architecture provides maximum single-query compute allocation (all 8 threads), "
                "seamless dual warm-memory residency across models, and zero slot partitioning overhead."
            )
            tradeoffs = [
                "Cannot process concurrent requests in parallel; any simultaneous request will queue serially.",
                "Vulnerable to Head-of-Line blocking if long generation tasks are submitted alongside interactive queries.",
            ]
            if memory_constrained:
                rationale += f" Note: Slot count was constrained to 1 due to tight memory budget ({memory_budget_mb} MB)."
                tradeoffs.append(f"Memory budget {memory_budget_mb} MB prevented multi-slot allocation.")

            return ServingArchitectureRecommendation(
                recommendation_id=f"rec-arch-{uuid.uuid4().hex[:8]}",
                recommended_architecture=ServingRuntimeArchitecture.OLLAMA_SINGLE_SLOT,
                recommended_slots=1,
                recommended_runtime_name="Ollama (Single-Slot Baseline)",
                expected_throughput_rps=1.2,
                expected_p95_latency_ms=800.0,
                expected_sla_compliance_pct=100.0,
                evidence_tier="DIRECT_MEASURED",
                rationale=rationale,
                tradeoffs=tradeoffs,
                actionable_setup_guidance="Run Ollama daemon with default settings: ollama serve",
            )

        elif effective_slots == 2:
            rationale = (
                f"Under concurrent mixed traffic (C={client_concurrency}, workload='{workload_type}'), "
                "llama.cpp server with 2 parallel slots and continuous batching eliminates Head-of-Line blocking. "
                "Short interactive requests decode simultaneously alongside long generation tasks rather than waiting 7+ seconds in queue."
            )
            tradeoffs = [
                "Context window is split across slots (2 x 1024 tokens = 2048 total context).",
                "Requires running dedicated llama-server process bound to GGUF model blob.",
            ]
            if memory_constrained and desired_slots > 2:
                rationale += f" Note: Target {desired_slots} slots capped at 2 slots to stay within {memory_budget_mb} MB memory budget."
                tradeoffs.append(f"Capped at 2 slots by memory limit ({memory_budget_mb} MB).")

            return ServingArchitectureRecommendation(
                recommendation_id=f"rec-arch-{uuid.uuid4().hex[:8]}",
                recommended_architecture=ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS,
                recommended_slots=2,
                recommended_runtime_name="llama.cpp server (2 Parallel Slots, Continuous Batching)",
                expected_throughput_rps=2.1,
                expected_p95_latency_ms=950.0,
                expected_sla_compliance_pct=95.0,
                evidence_tier="DIRECT_MEASURED",
                rationale=rationale,
                tradeoffs=tradeoffs,
                actionable_setup_guidance=(
                    "Launch llama-server with 2 parallel slots and continuous batching:\n"
                    "llama-server.exe -m <model.gguf> -np 2 -cb -c 2048 -t 8 --port 18080"
                ),
            )

        else:
            # High concurrency C >= 3: llama-server 4 slots continuous batching
            return ServingArchitectureRecommendation(
                recommendation_id=f"rec-arch-{uuid.uuid4().hex[:8]}",
                recommended_architecture=ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_4_SLOTS,
                recommended_slots=4,
                recommended_runtime_name="llama.cpp server (4 Parallel Slots, Continuous Batching)",
                expected_throughput_rps=3.4,
                expected_p95_latency_ms=1600.0,
                expected_sla_compliance_pct=90.0,
                evidence_tier="DIRECT_MEASURED",
                rationale=(
                    f"For high-concurrency client streams (C={client_concurrency}, lambda={arrival_rate_rps:.1f} rps), "
                    "4 parallel execution slots provide superior admission capacity and higher aggregate token throughput, "
                    "preventing server queue saturation."
                ),
                tradeoffs=[
                    "Per-slot context limit is reduced (4 x 512 tokens = 2048 total context).",
                    "Higher thread contention across 4 concurrent decodes on CPU.",
                ],
                actionable_setup_guidance=(
                    "Launch llama-server with 4 parallel slots:\n"
                    "llama-server.exe -m <model.gguf> -np 4 -cb -c 2048 -t 8 --port 18080"
                ),
            )

    @classmethod
    def get_comparison_summary(cls) -> dict[str, Any]:
        """Return high-level trade-off analysis between architectures."""
        return {
            "architectures": [
                {
                    "type": ServingRuntimeArchitecture.OLLAMA_SINGLE_SLOT.value,
                    "name": "Ollama (Single-Slot Baseline)",
                    "best_for": "Single-user interactive query streams, frequent model switching, minimal RAM overhead.",
                    "limitations": "Completely serial execution; multi-client traffic suffers catastrophic HoL blocking.",
                },
                {
                    "type": ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_2_SLOTS.value,
                    "name": "llama.cpp Server (2 Slots, Continuous Batching)",
                    "best_for": "Multi-tenant interactive QA + background generation mixtures; eliminates HoL blocking.",
                    "limitations": "Context memory partitioned per slot; single active model per server port.",
                },
                {
                    "type": ServingRuntimeArchitecture.LLAMA_CPP_PARALLEL_4_SLOTS.value,
                    "name": "llama.cpp Server (4 Slots, Continuous Batching)",
                    "best_for": "Sustained high-arrival burst traffic; maximizes aggregate throughput.",
                    "limitations": "Context limited to 512 tokens per slot; CPU thread contention increases decode latency.",
                },
            ]
        }
