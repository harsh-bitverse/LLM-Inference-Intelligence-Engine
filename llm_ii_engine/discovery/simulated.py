"""Deterministic simulated hardware profiles for automated testing and CI.

All simulated profiles are explicitly marked with is_simulated=True and
ExecutionMode.SIMULATED. They must never be confused with physical hardware measurements.
"""

from datetime import UTC, datetime

from llm_ii_engine.core.enums import ExecutionMode
from llm_ii_engine.discovery.catalog import (
    get_gpu_spec,
)
from llm_ii_engine.discovery.models import (
    CPUInfo,
    DetectedGPU,
    HardwareCapabilityReport,
    HostSystemInfo,
    MemoryInfo,
    TelemetryTools,
)


def get_simulated_profile(
    gpu_catalog_id: str | None = None,
    num_gpus: int = 1,
    cpu_model: str = "AMD EPYC 7763 64-Core Processor",
    ram_gb: float = 512.0,
) -> HardwareCapabilityReport:
    """Generate a deterministic simulated hardware capability report.

    Args:
        gpu_catalog_id: Canonical catalog ID (e.g. 'nvidia-h100-sxm5-80gb') or None for CPU-only.
        num_gpus: Number of simulated GPUs of the given catalog variant.
        cpu_model: Simulated CPU model name.
        ram_gb: Simulated total host RAM in GiB.

    Returns:
        HardwareCapabilityReport explicitly tagged as SIMULATED.
    """
    host = HostSystemInfo(
        os_name="Linux",
        os_version="6.5.0-44-generic",
        os_release="Ubuntu 22.04.4 LTS",
        architecture="x86_64",
        python_version="3.12.0",
    )
    cpu = CPUInfo(
        model_name=cpu_model,
        vendor="AMD",
        physical_cores=64,
        logical_cores=128,
        frequency_mhz=2450.0,
    )
    ram_bytes = int(ram_gb * 1024 * 1024 * 1024)
    memory = MemoryInfo(
        total_bytes=ram_bytes,
        available_bytes=int(ram_bytes * 0.8),
    )

    gpus: list[DetectedGPU] = []
    matched_specs = []

    if gpu_catalog_id:
        spec = get_gpu_spec(gpu_catalog_id)
        if spec:
            matched_specs.append(spec)
            for i in range(num_gpus):
                gpus.append(
                    DetectedGPU(
                        index=i,
                        name=spec.name,
                        vendor="NVIDIA",
                        total_memory_mb=spec.memory_capacity_gb * 1024.0,
                        driver_version="550.54.14",
                        uuid=f"GPU-SIM-{i:04d}-{spec.architecture.upper()}",
                        pci_bus_id=f"0000:{i + 1:02x}:00.0",
                        compute_capability=spec.compute_capability,
                        detection_source="simulated",
                    )
                )

    telemetry = TelemetryTools(
        has_nvidia_smi=bool(gpus),
        nvidia_smi_path="/usr/bin/nvidia-smi" if gpus else None,
        has_rocm_smi=False,
        rocm_smi_path=None,
        has_pynvml=bool(gpus),
    )

    return HardwareCapabilityReport(
        report_id=f"sim-profile-{gpu_catalog_id or 'cpu-only'}",
        timestamp_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
        is_simulated=True,
        execution_mode=ExecutionMode.SIMULATED,
        host=host,
        cpu=cpu,
        memory=memory,
        gpus=gpus,
        telemetry=telemetry,
        matched_catalog_specs=matched_specs,
        missing_or_incomplete_fields=[],
        warnings=["This hardware capability report is SIMULATED for testing purposes only."],
    )


# Standard Preset Simulated Profiles for deterministic CI/CD
SIMULATED_H100_SXM5_PROFILE = get_simulated_profile("nvidia-h100-sxm5-80gb", num_gpus=1)
SIMULATED_A100_SXM4_80GB_PROFILE = get_simulated_profile("nvidia-a100-sxm4-80gb", num_gpus=1)
SIMULATED_RTX4090_PROFILE = get_simulated_profile("nvidia-rtx-4090", num_gpus=1)
SIMULATED_T4_PROFILE = get_simulated_profile("nvidia-t4", num_gpus=1)
SIMULATED_CPU_ONLY_PROFILE = get_simulated_profile(None, num_gpus=0)
