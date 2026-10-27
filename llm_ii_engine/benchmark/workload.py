"""Workload generation, distribution modeling, and controlled scenario presets.

Generates reproducible sequences of WorkloadSpec instances without downloading
tokenizers or loading physical model weights.
"""

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_ii_engine.core.schemas import WorkloadSpec


class WorkloadScenarioConfig(BaseModel):
    """Specification defining prompt and output length distributions across a benchmark suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="Descriptive identifier for the workload scenario")
    num_requests: int = Field(gt=0, description="Total number of requests to generate")

    # Prompt distribution
    prompt_distribution: Literal["fixed", "uniform", "gaussian"] = Field(
        default="fixed", description="Statistical distribution for prompt token lengths"
    )
    prompt_tokens_mean: int = Field(ge=0, description="Mean (or fixed) prompt token length")
    prompt_tokens_std: float = Field(
        default=0.0, ge=0.0, description="Standard deviation for Gaussian distribution"
    )
    prompt_tokens_min: int = Field(default=1, ge=0, description="Minimum clipped prompt tokens")
    prompt_tokens_max: int = Field(default=32768, gt=0, description="Maximum clipped prompt tokens")

    # Output distribution
    output_distribution: Literal["fixed", "uniform", "gaussian"] = Field(
        default="fixed", description="Statistical distribution for generated output token lengths"
    )
    output_tokens_mean: int = Field(
        ge=0, description="Mean (or fixed) output token length; 0 allows prefill-only"
    )
    output_tokens_std: float = Field(
        default=0.0, ge=0.0, description="Standard deviation for Gaussian distribution"
    )
    output_tokens_min: int = Field(default=0, ge=0, description="Minimum clipped output tokens")
    output_tokens_max: int = Field(default=8192, ge=0, description="Maximum clipped output tokens")

    seed: int = Field(default=42, description="Random generator seed ensuring 100% reproducibility")
    default_prompt_text: str | None = Field(
        default=None, description="Optional raw text prompt to attach to generated workloads"
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "WorkloadScenarioConfig":
        if self.prompt_tokens_min > self.prompt_tokens_max:
            raise ValueError(
                f"prompt_tokens_min ({self.prompt_tokens_min}) must be <= prompt_tokens_max ({self.prompt_tokens_max})"
            )
        if self.output_tokens_min > self.output_tokens_max:
            raise ValueError(
                f"output_tokens_min ({self.output_tokens_min}) must be <= output_tokens_max ({self.output_tokens_max})"
            )
        return self


def _sample_token_count(
    dist_type: str,
    mean: int,
    std: float,
    min_val: int,
    max_val: int,
    rng: random.Random,
) -> int:
    """Sample an integer token length according to the specified distribution."""
    if dist_type == "fixed" or std == 0.0:
        raw_val = mean
    elif dist_type == "uniform":
        raw_val = rng.randint(min_val, max_val)
    elif dist_type == "gaussian":
        # Gaussian distribution rounded to nearest integer and clipped
        sampled = rng.gauss(mean, std)
        raw_val = round(sampled)
    else:
        raw_val = mean

    # Strict clipping to [min_val, max_val]
    return max(min_val, min(max_val, int(raw_val)))


def generate_workload(config: WorkloadScenarioConfig) -> list[WorkloadSpec]:
    """Generate a reproducible sequence of WorkloadSpec instances.

    Args:
        config: Validated WorkloadScenarioConfig.

    Returns:
        List of WorkloadSpec instances with deterministic seeds.
    """
    rng = random.Random(config.seed)
    workloads: list[WorkloadSpec] = []

    for i in range(config.num_requests):
        prompt_len = _sample_token_count(
            config.prompt_distribution,
            config.prompt_tokens_mean,
            config.prompt_tokens_std,
            config.prompt_tokens_min,
            config.prompt_tokens_max,
            rng,
        )
        output_len = _sample_token_count(
            config.output_distribution,
            config.output_tokens_mean,
            config.output_tokens_std,
            config.output_tokens_min,
            config.output_tokens_max,
            rng,
        )
        # Assign deterministic per-request seed
        req_seed = rng.randint(1, 1_000_000_000)

        workloads.append(
            WorkloadSpec(
                prompt_tokens=prompt_len,
                output_tokens=output_len,
                prompt_text=config.default_prompt_text,
                seed=req_seed,
            )
        )

    return workloads


def resolve_prompt_text(workload: WorkloadSpec) -> str:
    """Resolve or construct the prompt text string for real inference requests.

    If workload.prompt_text is explicitly provided, it is returned directly.
    Otherwise, if prompt_tokens is specified, a deterministic synthetic prompt text
    is constructed to provide the requested payload length for engines requiring string input.

    Note: This does not fabricate exact token lengths by assuming characters or words
    equal model tokens. The actual evaluated prompt token count must be obtained from
    server-reported statistics or tokenizer metadata where supported.
    """
    if workload.prompt_text is not None:
        return workload.prompt_text

    tokens = workload.prompt_tokens or 0
    if tokens == 0:
        return ""

    # Generate synthetic text payload of approximate length without claiming exact word-to-token equality
    return f"Benchmark synthetic payload (~{tokens} requested tokens): " + " ".join(
        ["token"] * max(1, tokens)
    )


# --- Pre-configured Standard Scenarios for Concurrency Benchmarking ---

SCENARIO_SHORT_CHAT = WorkloadScenarioConfig(
    name="short_conversational_chat",
    num_requests=20,
    prompt_distribution="fixed",
    prompt_tokens_mean=128,
    output_distribution="fixed",
    output_tokens_mean=64,
    seed=42,
)

SCENARIO_MEDIUM_RAG = WorkloadScenarioConfig(
    name="medium_rag_qa",
    num_requests=20,
    prompt_distribution="fixed",
    prompt_tokens_mean=1024,
    output_distribution="fixed",
    output_tokens_mean=256,
    seed=42,
)

SCENARIO_LONG_CONTEXT = WorkloadScenarioConfig(
    name="long_document_summary",
    num_requests=10,
    prompt_distribution="fixed",
    prompt_tokens_mean=4096,
    output_distribution="fixed",
    output_tokens_mean=512,
    seed=42,
)

SCENARIO_PREFILL_ONLY = WorkloadScenarioConfig(
    name="prefill_only_latency",
    num_requests=10,
    prompt_distribution="fixed",
    prompt_tokens_mean=2048,
    output_distribution="fixed",
    output_tokens_mean=0,
    output_tokens_min=0,
    seed=42,
)

SCENARIO_VARIABLE_GAUSSIAN = WorkloadScenarioConfig(
    name="stochastic_gaussian_workload",
    num_requests=25,
    prompt_distribution="gaussian",
    prompt_tokens_mean=512,
    prompt_tokens_std=64.0,
    prompt_tokens_min=256,
    prompt_tokens_max=1024,
    output_distribution="gaussian",
    output_tokens_mean=128,
    output_tokens_std=32.0,
    output_tokens_min=32,
    output_tokens_max=256,
    seed=1337,
)
