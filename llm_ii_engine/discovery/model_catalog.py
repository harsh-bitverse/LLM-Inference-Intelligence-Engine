"""Curated candidate model catalog for small local LLMs on CPU workstations.

Maintains authoritative metadata, architectural properties, licensing, and expected
resource footprint for small local models, distinguishing installed from uninstalled candidates.
"""

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.benchmark.multi_model_protocol import ModelInstallationStatus


class CandidateModelSpec(BaseModel):
    """Specification of a candidate model for multi-model evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Official model tag, e.g. 'llama3.2:1b'")
    display_name: str = Field(description="Human-readable title")
    model_family: str = Field(description="Model family, e.g. 'llama', 'qwen2', 'phi'")
    parameter_count_billion: float = Field(gt=0, description="Total parameters in billions")
    default_quantization: str = Field(description="Default Ollama quantization level")
    expected_disk_size_mb: float = Field(gt=0, description="Estimated download size on disk in MiB")
    expected_ram_mb: float = Field(gt=0, description="Estimated runtime memory working set in MiB")
    context_window: int = Field(gt=0, description="Maximum architectural context window")
    tokenizer_name: str = Field(description="Tokenizer identifier")
    vocabulary_size: int = Field(gt=0, description="Vocabulary size")
    license: str = Field(description="Software / model weights license")
    intended_evaluation_role: str = Field(description="Scientific role in multi-model benchmark")
    download_command: str = Field(description="Exact Ollama pull command")
    installation_status: ModelInstallationStatus = Field(
        default=ModelInstallationStatus.UNINSTALLED_CANDIDATE,
        description="Current installation status",
    )
    is_installed: bool = Field(
        default=False, description="True if verified locally in Ollama runtime"
    )
    digest: str | None = Field(default=None, description="SHA256 digest if installed")


# --- Curated Candidate Catalog for CPU Workstations ---

CANDIDATE_MODELS: list[CandidateModelSpec] = [
    CandidateModelSpec(
        model_id="llama3.2:1b",
        display_name="Llama 3.2 1B (Instruct)",
        model_family="llama",
        parameter_count_billion=1.24,
        default_quantization="Q8_0",
        expected_disk_size_mb=1260.0,
        expected_ram_mb=1550.0,
        context_window=131072,
        tokenizer_name="llama3_tiktoken_128k",
        vocabulary_size=128256,
        license="Meta Llama 3.2 Community License",
        intended_evaluation_role="Baseline verified model: compact, low-latency conversational and extraction model.",
        download_command="ollama pull llama3.2:1b",
        installation_status=ModelInstallationStatus.INSTALLED,
        is_installed=True,
        digest="baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878",
    ),
    CandidateModelSpec(
        model_id="qwen2.5:0.5b",
        display_name="Qwen 2.5 0.5B (Instruct)",
        model_family="qwen2",
        parameter_count_billion=0.49,
        default_quantization="Q4_K_M",
        expected_disk_size_mb=398.0,
        expected_ram_mb=720.0,
        context_window=32768,
        tokenizer_name="qwen2_bpe_152k",
        vocabulary_size=152064,
        license="Apache 2.0",
        intended_evaluation_role="Sub-billion parameter tier: ultra-low latency, CPU-friendly baseline for constrained environments.",
        download_command="ollama pull qwen2.5:0.5b",
        installation_status=ModelInstallationStatus.UNINSTALLED_CANDIDATE,
        is_installed=False,
    ),
    CandidateModelSpec(
        model_id="qwen2.5:1.5b",
        display_name="Qwen 2.5 1.5B (Instruct)",
        model_family="qwen2",
        parameter_count_billion=1.54,
        default_quantization="Q4_K_M",
        expected_disk_size_mb=986.0,
        expected_ram_mb=1750.0,
        context_window=32768,
        tokenizer_name="qwen2_bpe_152k",
        vocabulary_size=152064,
        license="Apache 2.0",
        intended_evaluation_role="Direct peer comparison to Llama 3.2 1B: similar parameter scale (~1.5B) with larger vocabulary and high code/reasoning benchmark scores.",
        download_command="ollama pull qwen2.5:1.5b",
        installation_status=ModelInstallationStatus.UNINSTALLED_CANDIDATE,
        is_installed=False,
    ),
    CandidateModelSpec(
        model_id="qwen2.5:3b",
        display_name="Qwen 2.5 3B (Instruct)",
        model_family="qwen2",
        parameter_count_billion=3.09,
        default_quantization="Q4_K_M",
        expected_disk_size_mb=1900.0,
        expected_ram_mb=3200.0,
        context_window=32768,
        tokenizer_name="qwen2_bpe_152k",
        vocabulary_size=152064,
        license="Qwen Research License",
        intended_evaluation_role="Parameter scale scaling study: 2x parameter expansion to evaluate throughput degradation vs reasoning gain on CPU.",
        download_command="ollama pull qwen2.5:3b",
        installation_status=ModelInstallationStatus.UNINSTALLED_CANDIDATE,
        is_installed=False,
    ),
    CandidateModelSpec(
        model_id="llama3.2:3b",
        display_name="Llama 3.2 3B (Instruct)",
        model_family="llama",
        parameter_count_billion=3.21,
        default_quantization="Q4_K_M",
        expected_disk_size_mb=2000.0,
        expected_ram_mb=3400.0,
        context_window=131072,
        tokenizer_name="llama3_tiktoken_128k",
        vocabulary_size=128256,
        license="Meta Llama 3.2 Community License",
        intended_evaluation_role="Within-family scaling comparison: evaluating Llama 3.2 1B vs 3B with identical tokenizer and context architecture.",
        download_command="ollama pull llama3.2:3b",
        installation_status=ModelInstallationStatus.UNINSTALLED_CANDIDATE,
        is_installed=False,
    ),
    CandidateModelSpec(
        model_id="phi3.5:3.8b",
        display_name="Phi-3.5 Mini 3.8B (Instruct)",
        model_family="phi",
        parameter_count_billion=3.82,
        default_quantization="Q4_K_M",
        expected_disk_size_mb=2200.0,
        expected_ram_mb=4100.0,
        context_window=131072,
        tokenizer_name="phi3_tiktoken_32k",
        vocabulary_size=32064,
        license="MIT License",
        intended_evaluation_role="Reasoning and synthetic training reference: highly optimized small reasoning model with 128k context.",
        download_command="ollama pull phi3.5:3.8b",
        installation_status=ModelInstallationStatus.UNINSTALLED_CANDIDATE,
        is_installed=False,
    ),
]


def fetch_installed_ollama_models(
    base_url: str = "http://127.0.0.1:11434", timeout_s: float = 3.0
) -> dict[str, dict[str, Any]]:
    """Query local Ollama server tags API to discover physically installed models."""
    try:
        with httpx.Client(trust_env=False, timeout=timeout_s) as client:
            resp = client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models_info = {}
                for m in data.get("models", []):
                    name = m.get("name", "")
                    models_info[name] = m
                return models_info
    except Exception:
        pass
    return {}


def get_model_catalog(
    installed_tags: dict[str, dict[str, Any]] | None = None,
) -> list[CandidateModelSpec]:
    """Retrieve catalog models with updated local installation status."""
    if installed_tags is None:
        installed_tags = fetch_installed_ollama_models()

    catalog: list[CandidateModelSpec] = []
    for model in CANDIDATE_MODELS:
        installed_meta = installed_tags.get(model.model_id)
        if installed_meta is not None:
            digest = installed_meta.get("digest")
            catalog.append(
                CandidateModelSpec(
                    model_id=model.model_id,
                    display_name=model.display_name,
                    model_family=model.model_family,
                    parameter_count_billion=model.parameter_count_billion,
                    default_quantization=model.default_quantization,
                    expected_disk_size_mb=model.expected_disk_size_mb,
                    expected_ram_mb=model.expected_ram_mb,
                    context_window=model.context_window,
                    tokenizer_name=model.tokenizer_name,
                    vocabulary_size=model.vocabulary_size,
                    license=model.license,
                    intended_evaluation_role=model.intended_evaluation_role,
                    download_command=model.download_command,
                    installation_status=ModelInstallationStatus.INSTALLED,
                    is_installed=True,
                    digest=digest,
                )
            )
        else:
            catalog.append(
                CandidateModelSpec(
                    model_id=model.model_id,
                    display_name=model.display_name,
                    model_family=model.model_family,
                    parameter_count_billion=model.parameter_count_billion,
                    default_quantization=model.default_quantization,
                    expected_disk_size_mb=model.expected_disk_size_mb,
                    expected_ram_mb=model.expected_ram_mb,
                    context_window=model.context_window,
                    tokenizer_name=model.tokenizer_name,
                    vocabulary_size=model.vocabulary_size,
                    license=model.license,
                    intended_evaluation_role=model.intended_evaluation_role,
                    download_command=model.download_command,
                    installation_status=ModelInstallationStatus.UNINSTALLED_CANDIDATE,
                    is_installed=False,
                    digest=None,
                )
            )
    return catalog


def get_recommended_authorization_shortlist() -> list[CandidateModelSpec]:
    """Return the curated shortlist of candidate models recommended for user download authorization."""
    # Top two candidates for user authorization:
    # 1. qwen2.5:1.5b (Direct peer to Llama 3.2 1B, Apache 2.0, ~986MB disk)
    # 2. qwen2.5:0.5b (Ultra-lightweight reference, Apache 2.0, ~398MB disk)
    return [
        m for m in CANDIDATE_MODELS if m.model_id in ("qwen2.5:1.5b", "qwen2.5:0.5b")
    ]
