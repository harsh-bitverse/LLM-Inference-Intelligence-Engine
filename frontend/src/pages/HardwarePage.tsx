import React, { useEffect, useState } from 'react';
import {
  Cpu,
  HardDrive,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Activity,
  Zap,
  BookOpen,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { GPUSpecification, HardwareCapabilityReport } from '../api/types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorMessage } from '../components/common/ErrorMessage';
import { MetricCard } from '../components/common/MetricCard';
import { EvidenceBadge } from '../components/common/EvidenceBadge';

export const HardwarePage: React.FC = () => {
  const [report, setReport] = useState<HardwareCapabilityReport | null>(null);
  const [catalog, setCatalog] = useState<GPUSpecification[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [probing, setProbing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchHardware = async (discover: boolean = false) => {
    if (discover) {
      setProbing(true);
    } else {
      setLoading(true);
    }
    setError(null);

    try {
      const [data, catalogData] = await Promise.all([
        apiClient.getHardware(discover),
        apiClient.getHardwareCatalog().catch(() => [] as GPUSpecification[]),
      ]);
      setReport(data);
      setCatalog(catalogData);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to retrieve hardware capability report');
    } finally {
      setLoading(false);
      setProbing(false);
    }
  };

  useEffect(() => {
    fetchHardware(false);
  }, []);

  if (loading && !report) {
    return <LoadingSpinner message="Querying hardware profiler and GPU telemetry..." size="lg" />;
  }

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
              Hardware Discovery & Profiling
            </h1>
            {report && (
              <EvidenceBadge
                executionMode={report.execution_mode}
                isSynthetic={report.is_simulated}
              />
            )}
          </div>
          <p className="mt-1 text-xs text-slate-400">
            System topology, CPU/RAM detection, accelerator discovery, and matched catalog specifications.
          </p>
        </div>

        <button
          onClick={() => fetchHardware(true)}
          disabled={probing}
          className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-sky-600 hover:bg-sky-500 disabled:bg-sky-950 disabled:text-sky-400 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors self-start sm:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${probing ? 'animate-spin' : ''}`} />
          {probing ? 'Probing Hardware...' : 'Re-Probe Hardware'}
        </button>
      </div>

      {error && <ErrorMessage title="Hardware Query Error" message={error} onRetry={() => fetchHardware(false)} />}

      {report && (
        <>
          {/* Top-Level Capability Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Detected Accelerators"
              value={report.gpus.length}
              unit={report.gpus.length === 1 ? 'GPU' : 'GPUs'}
              subtitle={
                report.gpus.length > 0
                  ? report.gpus[0]?.name || 'Unknown'
                  : 'No active GPUs detected'
              }
              icon={<Zap className="w-4 h-4" />}
            />

            <MetricCard
              title="Host RAM Total"
              value={formatBytes(report.memory.total_bytes)}
              subtitle={`Available: ${formatBytes(report.memory.available_bytes)}`}
              icon={<HardDrive className="w-4 h-4" />}
            />

            <MetricCard
              title="CPU Processor"
              value={report.cpu.physical_cores}
              unit="physical cores"
              subtitle={`${report.cpu.logical_cores} logical threads`}
              icon={<Cpu className="w-4 h-4" />}
            />

            <MetricCard
              title="Report Metadata"
              value={report.report_id.slice(0, 14)}
              subtitle={new Date(report.timestamp_utc).toLocaleString()}
              icon={<Activity className="w-4 h-4" />}
            />
          </div>

          {/* Missing Fields or Warnings */}
          {(report.missing_or_incomplete_fields.length > 0 || report.warnings.length > 0) && (
            <div className="p-4 rounded-xl border border-amber-800/60 bg-amber-950/20 text-xs text-amber-200 space-y-2">
              <div className="flex items-center gap-2 font-semibold text-amber-300">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Discovery Diagnostics & Incomplete Fields
              </div>
              {report.missing_or_incomplete_fields.length > 0 && (
                <p>
                  <span className="font-medium text-amber-400">Missing fields:</span>{' '}
                  {report.missing_or_incomplete_fields.join(', ')}
                </p>
              )}
              {report.warnings.map((warn, i) => (
                <p key={i} className="text-amber-300/80">
                  • {warn}
                </p>
              ))}
            </div>
          )}

          {/* Main Hardware Sections */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Host & CPU Architecture */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                <Cpu className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-semibold text-slate-200">Host & Processor Details</h3>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <span className="text-slate-400 block text-[11px]">Operating System</span>
                  <span className="font-mono text-slate-200 mt-0.5 block truncate">
                    {report.host.os_name} {report.host.os_release || report.host.os_version}
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <span className="text-slate-400 block text-[11px]">Architecture</span>
                  <span className="font-mono text-slate-200 mt-0.5 block">
                    {report.host.architecture}
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <span className="text-slate-400 block text-[11px]">CPU Model</span>
                  <span className="font-mono text-slate-200 mt-0.5 block truncate" title={report.cpu.model_name}>
                    {report.cpu.model_name}
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <span className="text-slate-400 block text-[11px]">CPU Vendor</span>
                  <span className="font-mono text-slate-200 mt-0.5 block">
                    {report.cpu.vendor}
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <span className="text-slate-400 block text-[11px]">Core Topology</span>
                  <span className="font-mono text-slate-200 mt-0.5 block">
                    {report.cpu.physical_cores} Physical / {report.cpu.logical_cores} Logical
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <span className="text-slate-400 block text-[11px]">Nominal Frequency</span>
                  <span className="font-mono text-slate-200 mt-0.5 block">
                    {report.cpu.frequency_mhz ? `${report.cpu.frequency_mhz.toFixed(0)} MHz` : 'Unavailable'}
                  </span>
                </div>
              </div>

              {/* Telemetry Tool Availability */}
              <div className="pt-2">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                  System Telemetry Tools
                </span>
                <div className="flex gap-3">
                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs">
                    {report.telemetry.has_nvidia_smi ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5 text-slate-500" />
                    )}
                    <span className="font-mono">nvidia-smi</span>
                  </div>

                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs">
                    {report.telemetry.has_rocm_smi ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5 text-slate-500" />
                    )}
                    <span className="font-mono">rocm-smi</span>
                  </div>

                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs">
                    {report.telemetry.has_pynvml ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5 text-slate-500" />
                    )}
                    <span className="font-mono">pynvml</span>
                  </div>
                </div>
              </div>
            </div>

            {/* GPU Device Inspection */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                <Zap className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-semibold text-slate-200">Accelerators & GPUs</h3>
              </div>

              {report.gpus.length === 0 ? (
                <div className="p-6 text-center border border-dashed border-slate-800 rounded-lg bg-slate-950/40 space-y-2">
                  <HelpCircle className="w-8 h-8 text-slate-500 mx-auto" />
                  <p className="text-xs font-semibold text-slate-300">
                    No GPU devices reported by detection profiler.
                  </p>
                  <p className="text-[11px] text-slate-500 max-w-sm mx-auto">
                    Scientific honesty notice: Inconclusive detection does not prove hardware absence. If running in a container or CPU-only VM, GPU passthrough may be disabled.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {report.gpus.map((gpu) => (
                    <div
                      key={gpu.index}
                      className="p-4 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2 text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-slate-100 font-mono text-sm">
                          [{gpu.index}] {gpu.name}
                        </span>
                        <span className="font-mono text-sky-400 font-semibold">
                          {(gpu.total_memory_mb / 1024).toFixed(1)} GB VRAM
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-400 pt-1">
                        <div>
                          <span>Vendor: </span>
                          <span className="text-slate-200 font-mono">{gpu.vendor}</span>
                        </div>
                        <div>
                          <span>Detection: </span>
                          <span className="text-slate-200 font-mono">{gpu.detection_source}</span>
                        </div>
                        {gpu.compute_capability && (
                          <div>
                            <span>Compute Cap: </span>
                            <span className="text-slate-200 font-mono">{gpu.compute_capability}</span>
                          </div>
                        )}
                        {gpu.driver_version && (
                          <div>
                            <span>Driver: </span>
                            <span className="text-slate-200 font-mono">{gpu.driver_version}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Matched Catalog Specifications */}
              {report.matched_catalog_specs.length > 0 && (
                <div className="pt-2">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                    Matched Catalog Ceilings
                  </span>
                  <div className="space-y-2">
                    {report.matched_catalog_specs.map((spec) => (
                      <div
                        key={spec.catalog_id}
                        className="p-3 rounded-lg bg-slate-950/50 border border-slate-800/80 text-xs flex justify-between items-center"
                      >
                        <div>
                          <span className="font-bold text-slate-200">{spec.name}</span>
                          <span className="text-slate-400 text-[11px] block">
                            Bandwidth: {spec.theoretical_memory_bandwidth_gbps} GB/s • Memory: {spec.memory_capacity_gb} GB
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-purple-400 font-mono font-semibold">
                            {spec.compute.fp16_dense_tflops
                              ? `${spec.compute.fp16_dense_tflops} TFLOPS (FP16)`
                              : spec.compute.fp32_tflops
                              ? `${spec.compute.fp32_tflops} TFLOPS (FP32)`
                              : 'Compute N/A'}
                          </span>
                          <span className="text-slate-500 text-[10px] block">Catalog ID: {spec.catalog_id}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Reference GPU Catalog */}
          {catalog.length > 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-slate-200">
                    Authoritative GPU Reference Catalog ({catalog.length})
                  </h3>
                </div>
                <span className="text-[11px] px-2 py-0.5 rounded bg-purple-950/80 border border-purple-800/60 text-purple-300 font-mono">
                  Theoretical Specs
                </span>
              </div>

              <p className="text-xs text-slate-400">
                Authoritative specifications sourced from manufacturer whitepapers. These values represent theoretical hardware ceilings used for analytical Roofline analysis and feasibility boundaries, not empirical benchmark measurements.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {catalog.map((gpu) => (
                  <div
                    key={gpu.catalog_id}
                    className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 hover:border-slate-700 text-xs space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-100 font-mono">{gpu.name}</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                        {gpu.architecture}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-1.5 text-[11px] text-slate-400 font-mono">
                      <div>
                        <span className="text-slate-500">VRAM:</span> {gpu.memory_capacity_gb} GB {gpu.memory_type}
                      </div>
                      <div>
                        <span className="text-slate-500">Bandwidth:</span> {gpu.theoretical_memory_bandwidth_gbps} GB/s
                      </div>
                      <div>
                        <span className="text-slate-500">FP16 Dense:</span>{' '}
                        {gpu.compute.fp16_dense_tflops ? `${gpu.compute.fp16_dense_tflops} TF` : 'N/A'}
                      </div>
                      <div>
                        <span className="text-slate-500">TDP:</span> {gpu.tdp_watts ? `${gpu.tdp_watts} W` : 'N/A'}
                      </div>
                    </div>

                    <div className="text-[10px] text-slate-500 truncate pt-1 border-t border-slate-800/60" title={gpu.source_reference}>
                      Ref: {gpu.source_reference}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
