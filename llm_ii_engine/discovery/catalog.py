"""Curated hardware specification catalog for NVIDIA GPUs.

Contains authoritative manufacturer specifications from NVIDIA whitepapers and datasheets.
Specifications represent theoretical ceilings (ProvenanceType.HEURISTIC) and must never
be presented as empirical runtime measurements.
"""

import re
from collections.abc import Mapping

from llm_ii_engine.core.enums import ProvenanceType
from llm_ii_engine.discovery.models import ComputeCapabilitySpec, GPUSpecification

# --- Authoritative Catalog Entries ---

H100_SXM5_80GB = GPUSpecification(
    catalog_id="nvidia-h100-sxm5-80gb",
    name="NVIDIA H100 SXM5 80GB",
    architecture="Hopper",
    form_factor="SXM5",
    memory_capacity_gb=80.0,
    memory_type="HBM3",
    memory_bus_width_bits=5120,
    theoretical_memory_bandwidth_gbps=3350.0,
    compute=ComputeCapabilitySpec(
        fp32_tflops=67.0,
        fp16_dense_tflops=989.0,
        fp16_sparse_tflops=1979.0,
        bf16_dense_tflops=989.0,
        bf16_sparse_tflops=1979.0,
        fp8_dense_tflops=1979.0,
        fp8_sparse_tflops=3958.0,
        int8_dense_tops=1979.0,
        int8_sparse_tops=3958.0,
    ),
    tdp_watts=700,
    compute_capability="9.0",
    source_reference="NVIDIA H100 Tensor Core GPU Architecture Whitepaper (v1.03)",
    caveats=[
        "Peak FP16/BF16/FP8/INT8 ratings with sparsity require 2:4 structured sparsity patterns.",
        "Dense Tensor Core throughput is exactly 50% of peak sparse ratings.",
        "Sustained clocks require adequate data center liquid or high-flow forced-air cooling.",
    ],
    provenance=ProvenanceType.HEURISTIC,
)

H100_PCIE_80GB = GPUSpecification(
    catalog_id="nvidia-h100-pcie-80gb",
    name="NVIDIA H100 PCIe 80GB",
    architecture="Hopper",
    form_factor="PCIe",
    memory_capacity_gb=80.0,
    memory_type="HBM2e",
    memory_bus_width_bits=5120,
    theoretical_memory_bandwidth_gbps=2000.0,
    compute=ComputeCapabilitySpec(
        fp32_tflops=51.0,
        fp16_dense_tflops=756.0,
        fp16_sparse_tflops=1513.0,
        bf16_dense_tflops=756.0,
        bf16_sparse_tflops=1513.0,
        fp8_dense_tflops=1513.0,
        fp8_sparse_tflops=3026.0,
        int8_dense_tops=1513.0,
        int8_sparse_tops=3026.0,
    ),
    tdp_watts=350,
    compute_capability="9.0",
    source_reference="NVIDIA H100 Tensor Core GPU PCIe Datasheet",
    caveats=[
        "PCIe version uses HBM2e memory at 2.0 TB/s (compared to 3.35 TB/s on SXM5).",
        "Compute ratings are scaled down relative to SXM5 due to 350W TDP limit.",
        "Structural sparsity requires 2:4 sparse weight representation.",
    ],
    provenance=ProvenanceType.HEURISTIC,
)

H100_NVL_94GB = GPUSpecification(
    catalog_id="nvidia-h100-nvl-94gb",
    name="NVIDIA H100 NVL 94GB",
    architecture="Hopper",
    form_factor="PCIe NVLink Pair",
    memory_capacity_gb=94.0,
    memory_type="HBM3",
    memory_bus_width_bits=6144,
    theoretical_memory_bandwidth_gbps=3900.0,
    compute=ComputeCapabilitySpec(
        fp32_tflops=60.0,
        fp16_dense_tflops=835.0,
        fp16_sparse_tflops=1671.0,
        bf16_dense_tflops=835.0,
        bf16_sparse_tflops=1671.0,
        fp8_dense_tflops=1671.0,
        fp8_sparse_tflops=3341.0,
        int8_dense_tops=1671.0,
        int8_sparse_tops=3341.0,
    ),
    tdp_watts=400,
    compute_capability="9.0",
    source_reference="NVIDIA H100 NVL Datasheet",
    caveats=[
        "Specifications listed per single GPU card within an NVLink dual-card bridge pair.",
        "Full pair delivers 188 GB combined memory and 7.8 TB/s aggregate memory bandwidth.",
    ],
    provenance=ProvenanceType.HEURISTIC,
)

A100_SXM4_80GB = GPUSpecification(
    catalog_id="nvidia-a100-sxm4-80gb",
    name="NVIDIA A100 SXM4 80GB",
    architecture="Ampere",
    form_factor="SXM4",
    memory_capacity_gb=80.0,
    memory_type="HBM2e",
    memory_bus_width_bits=5120,
    theoretical_memory_bandwidth_gbps=2039.0,
    compute=ComputeCapabilitySpec(
        fp32_tflops=19.5,
        fp16_dense_tflops=312.0,
        fp16_sparse_tflops=624.0,
        bf16_dense_tflops=312.0,
        bf16_sparse_tflops=624.0,
        fp8_dense_tflops=None,  # FP8 not supported on Ampere architecture
        fp8_sparse_tflops=None,
        int8_dense_tops=624.0,
        int8_sparse_tops=1248.0,
    ),
    tdp_watts=400,
    compute_capability="8.0",
    source_reference="NVIDIA A100 Tensor Core GPU Architecture Whitepaper",
    caveats=[
        "FP8 precision is not supported on Ampere architecture.",
        "Dense Tensor Core compute is 312 TFLOPs; 624 TFLOPs requires 2:4 structured sparsity.",
    ],
    provenance=ProvenanceType.HEURISTIC,
)

A100_PCIE_80GB = GPUSpecification(
    catalog_id="nvidia-a100-pcie-80gb",
    name="NVIDIA A100 PCIe 80GB",
    architecture="Ampere",
    form_factor="PCIe",
    memory_capacity_gb=80.0,
    memory_type="HBM2e",
    memory_bus_width_bits=5120,
    theoretical_memory_bandwidth_gbps=1935.0,
    compute=ComputeCapabilitySpec(
        fp32_tflops=19.5,
        fp16_dense_tflops=312.0,
        fp16_sparse_tflops=624.0,
        bf16_dense_tflops=312.0,
        bf16_sparse_tflops=624.0,
        fp8_dense_tflops=None,
        fp8_sparse_tflops=None,
        int8_dense_tops=624.0,
        int8_sparse_tops=1248.0,
    ),
    tdp_watts=250,
    compute_capability="8.0",
    source_reference="NVIDIA A100 Tensor Core GPU PCIe Datasheet",
    caveats=[
        "PCIe version memory bandwidth is 1935 GB/s compared to 2039 GB/s on SXM4.",
        "FP8 is unsupported on Ampere.",
    ],
    provenance=ProvenanceType.HEURISTIC,
)

A100_PCIE_40GB = GPUSpecification(
    catalog_id="nvidia-a100-pcie-40gb",
    name="NVIDIA A100 PCIe 40GB",
    architecture="Ampere",
    form_factor="PCIe",
    memory_capacity_gb=40.0,
    memory_type="HBM2",
    memory_bus_width_bits=5120,
    theoretical_memory_bandwidth_gbps=1555.0,
    compute=ComputeCapabilitySpec(
        fp32_tflops=19.5,
        fp16_dense_tflops=312.0,
        fp16_sparse_tflops=624.0,
        bf16_dense_tflops=312.0,
        bf16_sparse_tflops=624.0,
        fp8_dense_tflops=None,
        fp8_sparse_tflops=None,
        int8_dense_tops=624.0,
        int8_sparse_tops=1248.0,
    ),
    tdp_watts=250,
    compute_capability="8.0",
    source_reference="NVIDIA A100 40GB Datasheet",
    caveats=[
        "40 GB model features HBM2 at 1555 GB/s.",
        "FP8 is unsupported on Ampere.",
    ],
    provenance=ProvenanceType.HEURISTIC,
)

RTX_4090 = GPUSpecification(
    catalog_id="nvidia-rtx-4090",
    name="NVIDIA GeForce RTX 4090",
    architecture="Ada Lovelace",
    form_factor="Desktop",
    memory_capacity_gb=24.0,
    memory_type="GDDR6X",
    memory_bus_width_bits=384,
    theoretical_memory_bandwidth_gbps=1008.0,
    compute=ComputeCapabilitySpec(
        fp32_tflops=82.58,
        fp16_dense_tflops=165.2,
        fp16_sparse_tflops=330.3,
        bf16_dense_tflops=165.2,
        bf16_sparse_tflops=330.3,
        fp8_dense_tflops=330.3,
        fp8_sparse_tflops=660.6,
        int8_dense_tops=330.3,
        int8_sparse_tops=660.6,
    ),
    tdp_watts=450,
    compute_capability="8.9",
    source_reference="NVIDIA Ada Lovelace Architecture Whitepaper",
    caveats=[
        "Consumer gaming GPU lacking ECC memory by default.",
        "FP8 Tensor Core supported via Ada Lovelace 4th-gen Tensor Cores.",
        "Sustained boost clocks depend heavily on case airflow and thermal dissipation.",
    ],
    provenance=ProvenanceType.HEURISTIC,
)

RTX_3090 = GPUSpecification(
    catalog_id="nvidia-rtx-3090",
    name="NVIDIA GeForce RTX 3090",
    architecture="Ampere",
    form_factor="Desktop",
    memory_capacity_gb=24.0,
    memory_type="GDDR6X",
    memory_bus_width_bits=384,
    theoretical_memory_bandwidth_gbps=936.2,
    compute=ComputeCapabilitySpec(
        fp32_tflops=35.58,
        fp16_dense_tflops=71.0,
        fp16_sparse_tflops=142.0,
        bf16_dense_tflops=71.0,
        bf16_sparse_tflops=142.0,
        fp8_dense_tflops=None,  # FP8 unsupported on Ampere
        fp8_sparse_tflops=None,
        int8_dense_tops=142.0,
        int8_sparse_tops=284.0,
    ),
    tdp_watts=350,
    compute_capability="8.6",
    source_reference="NVIDIA Ampere GA102 Architecture Whitepaper",
    caveats=[
        "Consumer gaming GPU with double-sided GDDR6X VRAM; VRAM junction temperatures can run high.",
        "FP8 is unsupported on Ampere architecture.",
    ],
    provenance=ProvenanceType.HEURISTIC,
)

TESLA_T4 = GPUSpecification(
    catalog_id="nvidia-t4",
    name="NVIDIA Tesla T4",
    architecture="Turing",
    form_factor="PCIe Low Profile",
    memory_capacity_gb=16.0,
    memory_type="GDDR6",
    memory_bus_width_bits=256,
    theoretical_memory_bandwidth_gbps=320.0,
    compute=ComputeCapabilitySpec(
        fp32_tflops=8.1,
        fp16_dense_tflops=65.0,
        fp16_sparse_tflops=None,  # Turing architecture does not support 2:4 structural sparsity
        bf16_dense_tflops=None,  # BF16 unsupported on Turing
        bf16_sparse_tflops=None,
        fp8_dense_tflops=None,  # FP8 unsupported on Turing
        fp8_sparse_tflops=None,
        int8_dense_tops=130.0,
        int8_sparse_tops=None,
    ),
    tdp_watts=70,
    compute_capability="7.5",
    source_reference="NVIDIA T4 Tensor Core GPU Datasheet",
    caveats=[
        "Turing architecture does not support 2:4 structured sparsity or BF16/FP8 Tensor Cores.",
        "Low-profile 70W card widely deployed in edge and inference cloud clusters.",
    ],
    provenance=ProvenanceType.HEURISTIC,
)

# Registry mapping canonical catalog IDs to specifications
_GPU_CATALOG: Mapping[str, GPUSpecification] = {
    H100_SXM5_80GB.catalog_id: H100_SXM5_80GB,
    H100_PCIE_80GB.catalog_id: H100_PCIE_80GB,
    H100_NVL_94GB.catalog_id: H100_NVL_94GB,
    A100_SXM4_80GB.catalog_id: A100_SXM4_80GB,
    A100_PCIE_80GB.catalog_id: A100_PCIE_80GB,
    A100_PCIE_40GB.catalog_id: A100_PCIE_40GB,
    RTX_4090.catalog_id: RTX_4090,
    RTX_3090.catalog_id: RTX_3090,
    TESLA_T4.catalog_id: TESLA_T4,
}


def get_gpu_spec(catalog_id: str) -> GPUSpecification | None:
    """Retrieve a GPU specification by exact canonical catalog ID.

    Args:
        catalog_id: Canonical identifier (e.g. 'nvidia-h100-sxm5-80gb').

    Returns:
        GPUSpecification if found, else None.
    """
    return _GPU_CATALOG.get(catalog_id.strip().lower())


def list_gpu_specs() -> list[GPUSpecification]:
    """Return all catalogued GPU specifications."""
    return list(_GPU_CATALOG.values())


def match_gpu_name(device_name: str) -> GPUSpecification | None:
    """Attempt to match a runtime-detected GPU name string to a catalog specification.

    Normalizes vendor prefixes, suffixes, and memory indicators to resolve the best
    matching catalog specification.

    Args:
        device_name: String from nvidia-smi (e.g. 'NVIDIA A100-SXM4-80GB', 'Tesla T4').

    Returns:
        Matched GPUSpecification, or None if the device is unrecognized.
    """
    if not device_name or not device_name.strip():
        return None

    norm = device_name.strip().lower()
    # Normalize common separators
    norm_clean = re.sub(r"[\s\-_]+", " ", norm)

    # 1. H100 checks
    if "h100" in norm_clean:
        if "nvl" in norm_clean:
            return H100_NVL_94GB
        if "pcie" in norm_clean:
            return H100_PCIE_80GB
        if "sxm" in norm_clean or "sxm5" in norm_clean:
            return H100_SXM5_80GB
        # Default H100 to SXM5 if unspecified
        return H100_SXM5_80GB

    # 2. A100 checks
    if "a100" in norm_clean:
        if "40gb" in norm_clean:
            return A100_PCIE_40GB
        if "pcie" in norm_clean:
            return A100_PCIE_80GB
        if "sxm" in norm_clean or "sxm4" in norm_clean:
            return A100_SXM4_80GB
        # Default A100 to SXM4 80GB if unspecified
        return A100_SXM4_80GB

    # 3. RTX 4090 checks
    if "4090" in norm_clean:
        return RTX_4090

    # 4. RTX 3090 checks
    if "3090" in norm_clean:
        return RTX_3090

    # 5. T4 checks
    if "t4" in norm_clean:
        return TESLA_T4

    return None
