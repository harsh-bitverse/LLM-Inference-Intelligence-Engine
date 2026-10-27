"""Cross-platform hardware discovery profiler.

Discovers operating system, CPU, RAM, and GPU hardware capabilities
using standard-library facilities and optional system telemetry tools.
Fails gracefully on CPU-only machines, missing tools, or non-Windows/non-Linux environments.
"""

import ctypes
import os
import platform
import re
import shutil
import subprocess
import uuid

from llm_ii_engine.core.enums import ExecutionMode
from llm_ii_engine.discovery.catalog import match_gpu_name
from llm_ii_engine.discovery.models import (
    CPUInfo,
    DetectedGPU,
    GPUSpecification,
    HardwareCapabilityReport,
    HostSystemInfo,
    MemoryInfo,
    TelemetryTools,
)


def discover_host_info() -> HostSystemInfo:
    """Discover operating system, platform architecture, and interpreter version."""
    return HostSystemInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        os_release=platform.release(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
    )


def discover_cpu_info() -> tuple[CPUInfo, list[str]]:
    """Discover CPU model, vendor, and core counts.

    Returns:
        Tuple of (CPUInfo, list_of_missing_or_unresolved_fields).
    """
    missing: list[str] = []
    os_name = platform.system()
    logical_cores = os.cpu_count() or 1
    model_name: str | None = None
    vendor: str | None = None
    physical_cores: int | None = None
    frequency_mhz: float | None = None

    # 1. Windows CPU introspection
    if os_name == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            model_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            winreg.CloseKey(key)
        except Exception:
            model_name = platform.processor() or None

        # Check vendor
        if model_name:
            if "Intel" in model_name:
                vendor = "Intel"
            elif "AMD" in model_name:
                vendor = "AMD"

        # Attempt physical cores lookup via PowerShell (fast timeout)
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip().isdigit():
                physical_cores = int(res.stdout.strip())
        except Exception:
            pass

    # 2. Linux CPU introspection
    elif os_name == "Linux":
        try:
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", encoding="utf-8") as f:
                    content = f.read()

                # Extract model name
                model_match = re.search(r"model name\s*:\s*(.+)", content)
                if model_match:
                    model_name = model_match.group(1).strip()

                # Extract vendor
                vendor_match = re.search(r"vendor_id\s*:\s*(.+)", content)
                if vendor_match:
                    raw_vendor = vendor_match.group(1).strip()
                    if "Intel" in raw_vendor or "GenuineIntel" in raw_vendor:
                        vendor = "Intel"
                    elif "AMD" in raw_vendor or "AuthenticAMD" in raw_vendor:
                        vendor = "AMD"
                    else:
                        vendor = raw_vendor

                # Extract physical cores from distinct core id / physical id
                core_ids = set(re.findall(r"core id\s*:\s*(\d+)", content))
                if core_ids:
                    physical_cores = len(core_ids)
        except Exception:
            pass

    # 3. Darwin / macOS CPU introspection
    elif os_name == "Darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                model_name = res.stdout.strip()
                if "Apple" in model_name:
                    vendor = "Apple"
                elif "Intel" in model_name:
                    vendor = "Intel"

            phys_res = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if phys_res.returncode == 0 and phys_res.stdout.strip().isdigit():
                physical_cores = int(phys_res.stdout.strip())
        except Exception:
            pass

    # Fallback model name
    if not model_name:
        model_name = platform.processor() or "Unknown CPU Architecture"
        missing.append("cpu.model_name")

    if physical_cores is None:
        missing.append("cpu.physical_cores")

    cpu_info = CPUInfo(
        model_name=model_name,
        vendor=vendor,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        frequency_mhz=frequency_mhz,
    )
    return cpu_info, missing


class _MEMORYSTATUSEX(ctypes.Structure):
    """Win32 MEMORYSTATUSEX struct for GlobalMemoryStatusEx."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def discover_memory_info() -> tuple[MemoryInfo, list[str]]:
    """Discover total and available host system physical memory (RAM).

    Returns:
        Tuple of (MemoryInfo, list_of_missing_fields).
    """
    missing: list[str] = []
    os_name = platform.system()
    total_bytes: int | None = None
    avail_bytes: int | None = None

    # 1. Windows memory discovery via ctypes kernel32.GlobalMemoryStatusEx
    if os_name == "Windows":
        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                total_bytes = int(stat.ullTotalPhys)
                avail_bytes = int(stat.ullAvailPhys)
        except Exception:
            pass

    # 2. Linux memory discovery via /proc/meminfo
    elif os_name == "Linux":
        try:
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            parts = line.split()
                            total_bytes = int(parts[1]) * 1024
                        elif line.startswith("MemAvailable:"):
                            parts = line.split()
                            avail_bytes = int(parts[1]) * 1024
        except Exception:
            pass

    # 3. Darwin / macOS memory discovery via sysctl
    elif os_name == "Darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip().isdigit():
                total_bytes = int(res.stdout.strip())
        except Exception:
            pass

    # 4. Fallback if OS-specific method failed
    if total_bytes is None or total_bytes <= 0:
        # Fallback to minimal conservative placeholder if system refuses inspection
        total_bytes = 8 * 1024 * 1024 * 1024  # 8 GiB baseline
        missing.append("memory.total_bytes")

    if avail_bytes is None:
        missing.append("memory.available_bytes")

    return MemoryInfo(total_bytes=total_bytes, available_bytes=avail_bytes), missing


def discover_telemetry_tools() -> TelemetryTools:
    """Identify availability and filesystem paths of GPU management utilities."""
    nvidia_smi = shutil.which("nvidia-smi")
    rocm_smi = shutil.which("rocm-smi")

    has_pynvml = False
    try:
        import importlib.util

        has_pynvml = importlib.util.find_spec("pynvml") is not None
    except Exception:
        has_pynvml = False

    return TelemetryTools(
        has_nvidia_smi=nvidia_smi is not None,
        nvidia_smi_path=nvidia_smi,
        has_rocm_smi=rocm_smi is not None,
        rocm_smi_path=rocm_smi,
        has_pynvml=has_pynvml,
    )


def discover_gpus(
    telemetry: TelemetryTools,
) -> tuple[list[DetectedGPU], list[str], list[str]]:
    """Discover physical NVIDIA GPUs using nvidia-smi if available.

    Returns:
        Tuple of (detected_gpus, missing_fields, warnings).
    """
    gpus: list[DetectedGPU] = []
    missing: list[str] = []
    warnings: list[str] = []

    if not telemetry.has_nvidia_smi or not telemetry.nvidia_smi_path:
        missing.append("gpu.telemetry_tool")
        warnings.append(
            "nvidia-smi utility was not found on PATH. GPU hardware discovery unavailable; "
            "engine will operate in CPU or simulated mode."
        )
        return gpus, missing, warnings

    query_fields = [
        "index",
        "name",
        "memory.total",
        "driver_version",
        "uuid",
        "pci.bus_id",
        "compute_cap",
    ]
    cmd = [
        telemetry.nvidia_smi_path,
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if proc.returncode != 0:
            warnings.append(
                f"nvidia-smi returned exit code {proc.returncode}: {proc.stderr.strip() or 'Unknown error'}"
            )
            return gpus, missing, warnings

        lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                idx = int(parts[0]) if parts[0].isdigit() else len(gpus)
                name = parts[1]
                vram_mb = (
                    float(parts[2])
                    if len(parts) > 2 and parts[2].replace(".", "", 1).isdigit()
                    else None
                )
                driver = parts[3] if len(parts) > 3 and parts[3] != "[N/A]" else None
                dev_uuid = parts[4] if len(parts) > 4 and parts[4] != "[N/A]" else None
                pci_bus = parts[5] if len(parts) > 5 and parts[5] != "[N/A]" else None
                compute_cap = parts[6] if len(parts) > 6 and parts[6] != "[N/A]" else None

                gpu = DetectedGPU(
                    index=idx,
                    name=name,
                    vendor="NVIDIA",
                    total_memory_mb=vram_mb,
                    driver_version=driver,
                    uuid=dev_uuid,
                    pci_bus_id=pci_bus,
                    compute_capability=compute_cap,
                    detection_source="nvidia-smi",
                )
                gpus.append(gpu)

    except subprocess.TimeoutExpired:
        warnings.append("nvidia-smi query timed out after 5 seconds.")
    except Exception as exc:
        warnings.append(f"Unexpected error querying nvidia-smi: {exc}")

    if not gpus:
        warnings.append("No NVIDIA GPUs were detected by nvidia-smi.")

    return gpus, missing, warnings


def discover_hardware() -> HardwareCapabilityReport:
    """Execute complete host system hardware discovery.

    Discovers host operating system, CPU specifications, RAM capacity,
    telemetry utilities, and physical GPUs. Matches discovered GPUs to
    canonical catalog specifications where recognized.

    Returns:
        HardwareCapabilityReport reflecting physical host state.
    """
    host = discover_host_info()
    cpu, cpu_missing = discover_cpu_info()
    memory, mem_missing = discover_memory_info()
    telemetry = discover_telemetry_tools()
    gpus, gpu_missing, gpu_warnings = discover_gpus(telemetry)

    # Resolve catalog specifications for detected GPUs
    matched_specs: list[GPUSpecification] = []
    for gpu in gpus:
        matched = match_gpu_name(gpu.name)
        if matched and matched not in matched_specs:
            matched_specs.append(matched)

    all_missing = cpu_missing + mem_missing + gpu_missing
    all_warnings = list(gpu_warnings)

    report_id = f"hw-report-{uuid.uuid4().hex[:12]}"

    return HardwareCapabilityReport(
        report_id=report_id,
        is_simulated=False,
        execution_mode=ExecutionMode.REAL,
        host=host,
        cpu=cpu,
        memory=memory,
        gpus=gpus,
        telemetry=telemetry,
        matched_catalog_specs=matched_specs,
        missing_or_incomplete_fields=all_missing,
        warnings=all_warnings,
    )
