"""Unit tests for hardware discovery, specification catalog, and capability reporting."""

import subprocess
from unittest.mock import MagicMock, mock_open, patch

from llm_ii_engine.core.enums import ExecutionMode, ProvenanceType
from llm_ii_engine.discovery.catalog import (
    A100_PCIE_40GB,
    A100_PCIE_80GB,
    A100_SXM4_80GB,
    H100_NVL_94GB,
    H100_PCIE_80GB,
    H100_SXM5_80GB,
    RTX_3090,
    RTX_4090,
    TESLA_T4,
    get_gpu_spec,
    list_gpu_specs,
    match_gpu_name,
)
from llm_ii_engine.discovery.models import (
    CPUInfo,
    HardwareCapabilityReport,
    HostSystemInfo,
    MemoryInfo,
    TelemetryTools,
)
from llm_ii_engine.discovery.profiler import (
    discover_cpu_info,
    discover_gpus,
    discover_hardware,
    discover_host_info,
    discover_memory_info,
    discover_telemetry_tools,
)
from llm_ii_engine.discovery.simulated import (
    SIMULATED_A100_SXM4_80GB_PROFILE,
    SIMULATED_CPU_ONLY_PROFILE,
    SIMULATED_H100_SXM5_PROFILE,
    get_simulated_profile,
)


class TestHostAndMemoryDiscovery:
    """Tests for host system info and memory introspection."""

    def test_discover_host_info(self) -> None:
        host = discover_host_info()
        assert isinstance(host, HostSystemInfo)
        assert host.os_name in ("Windows", "Linux", "Darwin")
        assert len(host.architecture) > 0
        assert len(host.python_version) > 0

    def test_discover_memory_info_real_host(self) -> None:
        mem, missing = discover_memory_info()
        assert isinstance(mem, MemoryInfo)
        assert mem.total_bytes > 0
        assert mem.total_mb > 0
        assert mem.total_gb > 0
        if mem.available_bytes is not None:
            assert mem.available_mb is not None
            assert mem.available_gb is not None
            assert mem.available_bytes <= mem.total_bytes

    def test_memory_info_mock_linux_proc_meminfo(self) -> None:
        fake_meminfo = (
            "MemTotal:       65874288 kB\n"
            "MemFree:        30124800 kB\n"
            "MemAvailable:   48123456 kB\n"
            "Buffers:          524288 kB\n"
        )
        with (
            patch("platform.system", return_value="Linux"),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=fake_meminfo)),
        ):
            mem, missing = discover_memory_info()
            assert mem.total_bytes == 65874288 * 1024
            assert mem.available_bytes == 48123456 * 1024
            assert len(missing) == 0

    def test_memory_info_mock_darwin_sysctl(self) -> None:
        fake_proc = MagicMock(returncode=0, stdout="34359738368\n")
        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", return_value=fake_proc),
        ):
            mem, missing = discover_memory_info()
            assert mem.total_bytes == 34359738368
            assert mem.total_gb == 32.0


class TestCPUDiscovery:
    """Tests for CPU model and core count introspection across platforms."""

    def test_discover_cpu_info_real_host(self) -> None:
        cpu, missing = discover_cpu_info()
        assert isinstance(cpu, CPUInfo)
        assert len(cpu.model_name) > 0
        assert cpu.logical_cores >= 1

    def test_mock_linux_cpuinfo_parsing(self) -> None:
        fake_cpuinfo = (
            "processor\t: 0\n"
            "vendor_id\t: GenuineIntel\n"
            "model name\t: Intel(R) Xeon(R) Platinum 8480+\n"
            "core id\t\t: 0\n"
            "physical id\t: 0\n\n"
            "processor\t: 1\n"
            "vendor_id\t: GenuineIntel\n"
            "model name\t: Intel(R) Xeon(R) Platinum 8480+\n"
            "core id\t\t: 1\n"
            "physical id\t: 0\n"
        )
        with (
            patch("platform.system", return_value="Linux"),
            patch("os.cpu_count", return_value=2),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=fake_cpuinfo)),
        ):
            cpu, missing = discover_cpu_info()
            assert "Xeon" in cpu.model_name
            assert cpu.vendor == "Intel"
            assert cpu.logical_cores == 2
            assert cpu.physical_cores == 2

    def test_mock_darwin_cpu_parsing(self) -> None:
        brand_mock = MagicMock(returncode=0, stdout="Apple M3 Max\n")
        cores_mock = MagicMock(returncode=0, stdout="16\n")

        def fake_subprocess_run(cmd: list[str], **kwargs: object) -> MagicMock:
            if "machdep.cpu.brand_string" in cmd:
                return brand_mock
            if "hw.physicalcpu" in cmd:
                return cores_mock
            return MagicMock(returncode=1, stdout="")

        with (
            patch("platform.system", return_value="Darwin"),
            patch("os.cpu_count", return_value=16),
            patch("subprocess.run", side_effect=fake_subprocess_run),
        ):
            cpu, missing = discover_cpu_info()
            assert cpu.model_name == "Apple M3 Max"
            assert cpu.vendor == "Apple"
            assert cpu.physical_cores == 16
            assert cpu.logical_cores == 16

    def test_cpu_discovery_fallback_on_unsupported_os(self) -> None:
        with (
            patch("platform.system", return_value="FreeBSD"),
            patch("platform.processor", return_value="Generic-x86_64"),
            patch("os.cpu_count", return_value=4),
        ):
            cpu, missing = discover_cpu_info()
            assert cpu.model_name == "Generic-x86_64"
            assert cpu.logical_cores == 4
            assert cpu.physical_cores is None
            assert "cpu.physical_cores" in missing


class TestGPUDiscoveryAndTelemetry:
    """Tests for telemetry tool identification and GPU discovery."""

    def test_discover_telemetry_tools_detection(self) -> None:
        with (
            patch(
                "shutil.which",
                side_effect=lambda tool: "/usr/bin/nvidia-smi" if tool == "nvidia-smi" else None,
            ),
            patch("importlib.util.find_spec", return_value=None),
        ):
            telemetry = discover_telemetry_tools()
            assert telemetry.has_nvidia_smi is True
            assert telemetry.nvidia_smi_path == "/usr/bin/nvidia-smi"
            assert telemetry.has_rocm_smi is False
            assert telemetry.has_pynvml is False

    def test_gpu_discovery_when_nvidia_smi_missing(self) -> None:
        telemetry = TelemetryTools(
            has_nvidia_smi=False,
            nvidia_smi_path=None,
            has_rocm_smi=False,
            rocm_smi_path=None,
            has_pynvml=False,
        )
        gpus, missing, warnings = discover_gpus(telemetry)
        assert len(gpus) == 0
        assert "gpu.telemetry_tool" in missing
        assert any("nvidia-smi utility was not found" in w for w in warnings)

    def test_gpu_discovery_mock_nvidia_smi_success(self) -> None:
        fake_csv = (
            "0, NVIDIA A100-SXM4-80GB, 81920, 535.129.03, GPU-1234, 0000:01:00.0, 8.0\n"
            "1, NVIDIA GeForce RTX 4090, 24576, 535.129.03, GPU-5678, 0000:02:00.0, 8.9\n"
        )
        telemetry = TelemetryTools(
            has_nvidia_smi=True,
            nvidia_smi_path="/usr/bin/nvidia-smi",
            has_rocm_smi=False,
            rocm_smi_path=None,
            has_pynvml=True,
        )
        proc_mock = MagicMock(returncode=0, stdout=fake_csv, stderr="")

        with patch("subprocess.run", return_value=proc_mock):
            gpus, missing, warnings = discover_gpus(telemetry)
            assert len(gpus) == 2

            assert gpus[0].index == 0
            assert gpus[0].name == "NVIDIA A100-SXM4-80GB"
            assert gpus[0].total_memory_mb == 81920.0
            assert gpus[0].driver_version == "535.129.03"
            assert gpus[0].compute_capability == "8.0"

            assert gpus[1].index == 1
            assert gpus[1].name == "NVIDIA GeForce RTX 4090"
            assert gpus[1].total_memory_mb == 24576.0
            assert gpus[1].compute_capability == "8.9"

    def test_gpu_discovery_mock_nvidia_smi_failure(self) -> None:
        telemetry = TelemetryTools(has_nvidia_smi=True, nvidia_smi_path="/usr/bin/nvidia-smi")
        proc_mock = MagicMock(
            returncode=12,
            stdout="",
            stderr="NVIDIA-SMI has failed because of driver mismatch",
        )

        with patch("subprocess.run", return_value=proc_mock):
            gpus, missing, warnings = discover_gpus(telemetry)
            assert len(gpus) == 0
            assert any("driver mismatch" in w for w in warnings)

    def test_gpu_discovery_mock_nvidia_smi_timeout(self) -> None:
        telemetry = TelemetryTools(has_nvidia_smi=True, nvidia_smi_path="/usr/bin/nvidia-smi")

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
        ):
            gpus, missing, warnings = discover_gpus(telemetry)
            assert len(gpus) == 0
            assert any("timed out" in w for w in warnings)


class TestHardwareCatalog:
    """Tests for NVIDIA hardware specification catalog lookups, units, and citations."""

    def test_catalog_retrieval_all_variants(self) -> None:
        specs = list_gpu_specs()
        assert (
            len(specs) >= 9
        )  # H100 SXM5/PCIe/NVL, A100 SXM4/PCIe80/PCIe40, RTX 4090, RTX 3090, T4

        expected_ids = [
            "nvidia-h100-sxm5-80gb",
            "nvidia-h100-pcie-80gb",
            "nvidia-h100-nvl-94gb",
            "nvidia-a100-sxm4-80gb",
            "nvidia-a100-pcie-80gb",
            "nvidia-a100-pcie-40gb",
            "nvidia-rtx-4090",
            "nvidia-rtx-3090",
            "nvidia-t4",
        ]
        for spec_id in expected_ids:
            spec = get_gpu_spec(spec_id)
            assert spec is not None
            assert spec.catalog_id == spec_id
            assert spec.memory_capacity_gb > 0
            assert spec.theoretical_memory_bandwidth_gbps > 0
            assert spec.provenance == ProvenanceType.HEURISTIC
            assert len(spec.source_reference) > 0
            assert len(spec.caveats) > 0

    def test_h100_sxm5_specifications(self) -> None:
        spec = get_gpu_spec("nvidia-h100-sxm5-80gb")
        assert spec is not None
        assert spec.architecture == "Hopper"
        assert spec.memory_capacity_gb == 80.0
        assert spec.theoretical_memory_bandwidth_gbps == 3350.0
        assert spec.compute.fp16_dense_tflops == 989.0
        assert spec.compute.fp16_sparse_tflops == 1979.0
        assert spec.compute.fp8_dense_tflops == 1979.0
        assert spec.tdp_watts == 700

    def test_a100_precision_boundaries(self) -> None:
        spec = get_gpu_spec("nvidia-a100-sxm4-80gb")
        assert spec is not None
        assert spec.architecture == "Ampere"
        assert spec.theoretical_memory_bandwidth_gbps == 2039.0
        assert spec.compute.fp16_dense_tflops == 312.0
        # FP8 is not supported on Ampere
        assert spec.compute.fp8_dense_tflops is None
        assert spec.compute.fp8_sparse_tflops is None

    def test_t4_precision_and_sparsity_boundaries(self) -> None:
        spec = get_gpu_spec("nvidia-t4")
        assert spec is not None
        assert spec.architecture == "Turing"
        assert spec.theoretical_memory_bandwidth_gbps == 320.0
        assert spec.compute.fp16_dense_tflops == 65.0
        # Turing does not support 2:4 structural sparsity or BF16/FP8
        assert spec.compute.fp16_sparse_tflops is None
        assert spec.compute.bf16_dense_tflops is None
        assert spec.compute.fp8_dense_tflops is None

    def test_unknown_catalog_id_returns_none(self) -> None:
        assert get_gpu_spec("nvidia-non-existent-gpu") is None
        assert get_gpu_spec("") is None

    def test_match_gpu_name_fuzzy_resolution(self) -> None:
        # H100 variants
        assert match_gpu_name("NVIDIA H100 SXM5 80GB") == H100_SXM5_80GB
        assert match_gpu_name("NVIDIA H100 80GB HBM3") == H100_SXM5_80GB
        assert match_gpu_name("H100 PCIe 80GB") == H100_PCIE_80GB
        assert match_gpu_name("H100 NVL") == H100_NVL_94GB

        # A100 variants
        assert match_gpu_name("NVIDIA A100-SXM4-80GB") == A100_SXM4_80GB
        assert match_gpu_name("A100 PCIe 80GB") == A100_PCIE_80GB
        assert match_gpu_name("A100-PCIE-40GB") == A100_PCIE_40GB

        # Consumer & Inference
        assert match_gpu_name("NVIDIA GeForce RTX 4090") == RTX_4090
        assert match_gpu_name("GeForce RTX 3090") == RTX_3090
        assert match_gpu_name("Tesla T4") == TESLA_T4

        # Non-matching devices
        assert match_gpu_name("Intel(R) UHD Graphics 770") is None
        assert match_gpu_name("AMD Radeon RX 7900 XTX") is None
        assert match_gpu_name("") is None


class TestSimulatedProfilesAndRooflineCeiling:
    """Tests for simulated profiles and roofline ceiling calculation."""

    def test_simulated_h100_profile(self) -> None:
        prof = SIMULATED_H100_SXM5_PROFILE
        assert prof.is_simulated is True
        assert prof.execution_mode == ExecutionMode.SIMULATED
        assert len(prof.gpus) == 1
        assert prof.gpus[0].name == "NVIDIA H100 SXM5 80GB"
        assert len(prof.matched_catalog_specs) == 1
        assert prof.matched_catalog_specs[0].catalog_id == "nvidia-h100-sxm5-80gb"

    def test_simulated_cpu_only_profile(self) -> None:
        prof = SIMULATED_CPU_ONLY_PROFILE
        assert prof.is_simulated is True
        assert prof.execution_mode == ExecutionMode.SIMULATED
        assert len(prof.gpus) == 0
        assert len(prof.matched_catalog_specs) == 0

    def test_simulated_profile_determinism(self) -> None:
        prof1 = get_simulated_profile("nvidia-rtx-4090", num_gpus=2)
        prof2 = get_simulated_profile("nvidia-rtx-4090", num_gpus=2)
        assert prof1.report_id == prof2.report_id
        assert len(prof1.gpus) == 2
        assert prof1.gpus[0].uuid == prof2.gpus[0].uuid
        assert prof1.memory.total_bytes == prof2.memory.total_bytes

    def test_theoretical_roofline_ceiling_derivation(self) -> None:
        prof = SIMULATED_H100_SXM5_PROFILE
        ceiling = prof.get_theoretical_roofline_ceiling("fp16_dense")
        assert ceiling is not None
        assert ceiling.peak_memory_bandwidth_gbps == 3350.0
        assert ceiling.peak_compute_tflops == 989.0
        # Machine balance point = (989.0 * 1000) / 3350.0 = 295.22 FLOPs/byte
        assert ceiling.ridge_point_flops_per_byte == 295.22
        assert ceiling.provenance == ProvenanceType.HEURISTIC
        assert "Theoretical roofline ceiling" in ceiling.notes

    def test_roofline_ceiling_returns_none_for_cpu_or_unsupported_precision(
        self,
    ) -> None:
        # 1. CPU-only profile has no GPU spec -> ceiling is None
        prof_cpu = SIMULATED_CPU_ONLY_PROFILE
        assert prof_cpu.get_theoretical_roofline_ceiling("fp16_dense") is None

        # 2. Ampere profile requested with FP8 -> unsupported -> returns None
        prof_a100 = SIMULATED_A100_SXM4_80GB_PROFILE
        assert prof_a100.get_theoretical_roofline_ceiling("fp8_dense") is None


class TestFullHardwareDiscoveryPipeline:
    """Verify complete physical hardware discovery execution on the host machine."""

    def test_discover_hardware_physical_host(self) -> None:
        report = discover_hardware()
        assert isinstance(report, HardwareCapabilityReport)
        assert report.is_simulated is False
        assert report.execution_mode == ExecutionMode.REAL
        assert report.host.os_name in ("Windows", "Linux", "Darwin")
        assert report.cpu.logical_cores >= 1
        assert report.memory.total_bytes > 0
        assert isinstance(report.warnings, list)
        assert isinstance(report.missing_or_incomplete_fields, list)
