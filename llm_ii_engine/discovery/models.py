"""Domain models for hardware discovery, specification catalog, and capability reporting.

All models strictly separate runtime-detected hardware facts from static catalog
specifications and simulated test environments.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.core.enums import ExecutionMode, ProvenanceType


class HostSystemInfo(BaseModel):
    """Operating system and host platform identification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    os_name: str = Field(description="Operating system name (e.g. 'Windows', 'Linux', 'Darwin')")
    os_version: str = Field(description="Operating system build/kernel version")
    os_release: str = Field(description="Operating system release identifier")
    architecture: str = Field(
        description="Machine architecture (e.g. 'AMD64', 'x86_64', 'aarch64')"
    )
    python_version: str = Field(description="Active Python interpreter version")


class CPUInfo(BaseModel):
    """Central processing unit hardware identification and core counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(description="Raw CPU brand/model name")
    vendor: str | None = Field(
        default=None, description="CPU vendor (e.g. 'Intel', 'AMD', 'Apple')"
    )
    physical_cores: int | None = Field(
        default=None,
        gt=0,
        description="Physical hardware CPU core count if reliably detected",
    )
    logical_cores: int = Field(gt=0, description="Logical CPU execution threads available to OS")
    frequency_mhz: float | None = Field(
        default=None, gt=0.0, description="Nominal CPU clock frequency in MHz"
    )


class MemoryInfo(BaseModel):
    """Host system physical memory (RAM) capacity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_bytes: int = Field(gt=0, description="Total physical RAM in bytes")
    available_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Available physical RAM in bytes at discovery time",
    )

    @property
    def total_mb(self) -> float:
        """Total memory in megabytes (MiB)."""
        return round(self.total_bytes / (1024 * 1024), 2)

    @property
    def total_gb(self) -> float:
        """Total memory in gigabytes (GiB)."""
        return round(self.total_bytes / (1024 * 1024 * 1024), 2)

    @property
    def available_mb(self) -> float | None:
        """Available memory in megabytes (MiB)."""
        if self.available_bytes is None:
            return None
        return round(self.available_bytes / (1024 * 1024), 2)

    @property
    def available_gb(self) -> float | None:
        """Available memory in gigabytes (GiB)."""
        if self.available_bytes is None:
            return None
        return round(self.available_bytes / (1024 * 1024 * 1024), 2)


class DetectedGPU(BaseModel):
    """Physical GPU device discovered on the host at runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0, description="Zero-based device index assigned by driver or OS")
    name: str = Field(
        min_length=1,
        description="Reported device model name (e.g. 'NVIDIA GeForce RTX 4090')",
    )
    vendor: str = Field(default="NVIDIA", description="GPU vendor identifier")
    total_memory_mb: float | None = Field(
        default=None, ge=0.0, description="Total dedicated VRAM capacity in MiB"
    )
    driver_version: str | None = Field(
        default=None, description="Installed GPU driver version string"
    )
    uuid: str | None = Field(default=None, description="Unique GPU hardware UUID if reported")
    pci_bus_id: str | None = Field(default=None, description="PCI bus identifier string")
    compute_capability: str | None = Field(
        default=None,
        description="NVIDIA compute capability version (e.g. '8.9', '9.0')",
    )
    detection_source: str = Field(
        default="nvidia-smi",
        description="Utility or API used to discover the device (e.g. 'nvidia-smi', 'mock')",
    )


class TelemetryTools(BaseModel):
    """Availability of local GPU management and telemetry tools."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    has_nvidia_smi: bool = Field(
        default=False, description="Whether nvidia-smi executable was found on PATH"
    )
    nvidia_smi_path: str | None = Field(
        default=None, description="Resolved filesystem path to nvidia-smi"
    )
    has_rocm_smi: bool = Field(
        default=False, description="Whether rocm-smi executable was found on PATH"
    )
    rocm_smi_path: str | None = Field(
        default=None, description="Resolved filesystem path to rocm-smi"
    )
    has_pynvml: bool = Field(
        default=False, description="Whether pynvml Python library is importable"
    )


class ComputeCapabilitySpec(BaseModel):
    """Documented theoretical peak compute throughput across precisions.

    Values denote theoretical hardware ceilings from vendor whitepapers.
    Dense ratings are distinguished from structural 2:4 sparse ratings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fp32_tflops: float | None = Field(
        default=None, description="Standard or Tensor Core FP32 TFLOPs"
    )
    fp16_dense_tflops: float | None = Field(
        default=None, description="Peak dense FP16 Tensor Core TFLOPs"
    )
    fp16_sparse_tflops: float | None = Field(
        default=None,
        description="Peak FP16 Tensor Core TFLOPs with 2:4 structural sparsity",
    )
    bf16_dense_tflops: float | None = Field(
        default=None, description="Peak dense BF16 Tensor Core TFLOPs"
    )
    bf16_sparse_tflops: float | None = Field(
        default=None,
        description="Peak BF16 Tensor Core TFLOPs with 2:4 structural sparsity",
    )
    fp8_dense_tflops: float | None = Field(
        default=None, description="Peak dense FP8 Tensor Core TFLOPs"
    )
    fp8_sparse_tflops: float | None = Field(
        default=None,
        description="Peak FP8 Tensor Core TFLOPs with 2:4 structural sparsity",
    )
    int8_dense_tops: float | None = Field(
        default=None, description="Peak dense INT8 Tensor Core TOPS"
    )
    int8_sparse_tops: float | None = Field(
        default=None,
        description="Peak INT8 Tensor Core TOPS with 2:4 structural sparsity",
    )


class GPUSpecification(BaseModel):
    """Static catalog entry documenting verified theoretical specifications of a GPU variant.

    All specifications originate from authoritative manufacturer datasheets.
    Provenance is strictly HEURISTIC, reflecting vendor-specified theoretical peaks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_id: str = Field(
        min_length=1,
        description="Normalized canonical identifier (e.g. 'nvidia-h100-sxm5-80gb')",
    )
    name: str = Field(min_length=1, description="Official product name")
    architecture: str = Field(
        min_length=1, description="Microarchitecture family (e.g. 'Hopper', 'Ampere')"
    )
    form_factor: str = Field(
        min_length=1,
        description="Physical form factor (e.g. 'SXM5', 'PCIe', 'Desktop')",
    )
    memory_capacity_gb: float = Field(gt=0.0, description="Nominal dedicated VRAM capacity in GiB")
    memory_type: str = Field(
        min_length=1, description="Memory technology (e.g. 'HBM3', 'HBM2e', 'GDDR6X')"
    )
    memory_bus_width_bits: int | None = Field(
        default=None, gt=0, description="Memory bus width in bits"
    )
    theoretical_memory_bandwidth_gbps: float = Field(
        gt=0.0, description="Documented theoretical peak memory bandwidth in GB/s"
    )
    compute: ComputeCapabilitySpec = Field(
        description="Theoretical peak compute ratings by precision"
    )
    tdp_watts: int | None = Field(
        default=None, gt=0, description="Thermal Design Power rating in Watts"
    )
    compute_capability: str | None = Field(
        default=None, description="NVIDIA compute capability version string"
    )
    source_reference: str = Field(
        min_length=1, description="Authoritative citation (datasheet or whitepaper)"
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Important constraints or architectural footnotes",
    )
    provenance: ProvenanceType = Field(
        default=ProvenanceType.HEURISTIC,
        description="Specifications are theoretical manufacturer ceilings, not runtime measurements",
    )


class TheoreticalRooflineCeiling(BaseModel):
    """Theoretical compute and bandwidth boundaries extracted for future Roofline analysis.

    Notice: This represents theoretical hardware potential, not an empirical workload measurement.
    Actual operational intensity and bottleneck classification require real workload timing
    and model architectural parameters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    peak_memory_bandwidth_gbps: float = Field(
        gt=0.0, description="Theoretical memory bandwidth ceiling in GB/s"
    )
    peak_compute_tflops: float = Field(
        gt=0.0, description="Theoretical peak compute ceiling in TFLOPs"
    )
    precision: str = Field(
        description="Target precision for the compute ceiling (e.g. 'fp16_dense')"
    )
    ridge_point_flops_per_byte: float = Field(
        gt=0.0,
        description="Machine balance point: peak FLOPs / peak memory bandwidth bytes",
    )
    provenance: ProvenanceType = Field(
        default=ProvenanceType.HEURISTIC,
        description="Ceilings are static theoretical limits from hardware specifications",
    )
    notes: str = Field(
        default="Theoretical hardware ceiling only. Does not represent measured operational intensity.",
        description="Clarifying note regarding Roofline boundary",
    )


class HardwareCapabilityReport(BaseModel):
    """Consolidated hardware capability and discovery report.

    Combines detected host hardware, available telemetry tools, matched catalog specifications,
    explicit omissions, and operational warnings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(min_length=1, description="Unique identifier for this discovery report")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when discovery was executed",
    )
    is_simulated: bool = Field(
        default=False,
        description="True if report represents a synthetic/mock environment",
    )
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.REAL,
        description="REAL physical host discovery vs SIMULATED mock profile",
    )
    host: HostSystemInfo = Field(description="Host OS and platform details")
    cpu: CPUInfo = Field(description="Host CPU model and core counts")
    memory: MemoryInfo = Field(description="Host RAM capacity")
    gpus: list[DetectedGPU] = Field(
        default_factory=list, description="List of physical or simulated GPUs detected"
    )
    telemetry: TelemetryTools = Field(
        description="Status of diagnostic and telemetry utilities on the host"
    )
    matched_catalog_specs: list[GPUSpecification] = Field(
        default_factory=list, description="Catalog entries matched to detected GPUs"
    )
    missing_or_incomplete_fields: list[str] = Field(
        default_factory=list, description="Fields that could not be reliably determined"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Actionable warnings regarding incomplete discovery or missing drivers",
    )

    def get_primary_gpu_bandwidth_gbps(self) -> float | None:
        """Return theoretical memory bandwidth of the primary GPU if catalog match exists."""
        if self.matched_catalog_specs:
            return self.matched_catalog_specs[0].theoretical_memory_bandwidth_gbps
        return None

    def get_theoretical_roofline_ceiling(
        self, precision: str = "fp16_dense"
    ) -> TheoreticalRooflineCeiling | None:
        """Derive theoretical roofline ceilings from primary matched GPU specifications.

        Args:
            precision: Precision target ('fp16_dense', 'fp16_sparse', 'bf16_dense', 'fp8_dense', 'int8_dense').

        Returns:
            TheoreticalRooflineCeiling if a matching GPU catalog specification exists, else None.
        """
        if not self.matched_catalog_specs:
            return None

        spec = self.matched_catalog_specs[0]
        bw_gbps = spec.theoretical_memory_bandwidth_gbps

        # Resolve compute rating for requested precision
        compute_map = {
            "fp32": spec.compute.fp32_tflops,
            "fp16_dense": spec.compute.fp16_dense_tflops,
            "fp16_sparse": spec.compute.fp16_sparse_tflops,
            "bf16_dense": spec.compute.bf16_dense_tflops,
            "bf16_sparse": spec.compute.bf16_sparse_tflops,
            "fp8_dense": spec.compute.fp8_dense_tflops,
            "fp8_sparse": spec.compute.fp8_sparse_tflops,
            "int8_dense": spec.compute.int8_dense_tops,
            "int8_sparse": spec.compute.int8_sparse_tops,
        }
        compute_tflops = compute_map.get(precision)
        if compute_tflops is None:
            return None

        # Machine balance point = (TFLOPs * 10^12) / (GB/s * 10^9) = (TFLOPs * 1000) / GB/s
        ridge_point = round((compute_tflops * 1000.0) / bw_gbps, 2)

        return TheoreticalRooflineCeiling(
            peak_memory_bandwidth_gbps=bw_gbps,
            peak_compute_tflops=compute_tflops,
            precision=precision,
            ridge_point_flops_per_byte=ridge_point,
            provenance=ProvenanceType.HEURISTIC,
            notes=(
                f"Theoretical roofline ceiling for {spec.name} ({precision}). "
                "Inputs required for future bottleneck diagnosis: active prompt/decode sequence lengths, "
                "measured TTFT and TPOT, model parameters, and KV cache memory footprint."
            ),
        )
