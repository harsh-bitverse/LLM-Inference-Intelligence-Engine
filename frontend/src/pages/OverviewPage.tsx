import React, { useEffect, useState } from 'react';
import {
  Activity,
  Cpu,
  Gauge,
  Stethoscope,
  Sparkles,
  ArrowRight,
  Database,
  Layers,
} from 'lucide-react';
import { apiClient } from '../api/client';
import {
  BenchmarkSuiteResult,
  DiagnosticReport,
  HardwareCapabilityReport,
} from '../api/types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorMessage } from '../components/common/ErrorMessage';
import { MetricCard } from '../components/common/MetricCard';
import { EvidenceBadge } from '../components/common/EvidenceBadge';
import { NavTab } from '../components/layout/Navbar';

interface OverviewPageProps {
  onNavigate: (tab: NavTab) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ onNavigate }) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [hardware, setHardware] = useState<HardwareCapabilityReport | null>(null);
  const [recentBenchmarks, setRecentBenchmarks] = useState<BenchmarkSuiteResult[]>([]);
  const [totalBenchmarks, setTotalBenchmarks] = useState<number>(0);
  const [recentDiagnostics, setRecentDiagnostics] = useState<DiagnosticReport[]>([]);
  const [totalDiagnostics, setTotalDiagnostics] = useState<number>(0);
  const [totalOptimizations, setTotalOptimizations] = useState<number>(0);

  const fetchOverviewData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [hwRes, benchRes, diagRes, optRes] = await Promise.allSettled([
        apiClient.getHardware(false),
        apiClient.listBenchmarks({ limit: 5 }),
        apiClient.listDiagnostics(5),
        apiClient.listOptimizations(5),
      ]);

      if (hwRes.status === 'fulfilled') setHardware(hwRes.value);
      if (benchRes.status === 'fulfilled') {
        setRecentBenchmarks(benchRes.value.items);
        setTotalBenchmarks(benchRes.value.total);
      }
      if (diagRes.status === 'fulfilled') {
        setRecentDiagnostics(diagRes.value.items);
        setTotalDiagnostics(diagRes.value.total);
      }
      if (optRes.status === 'fulfilled') {
        setTotalOptimizations(optRes.value.total);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard overview');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOverviewData();
  }, []);

  if (loading) {
    return <LoadingSpinner message="Polling system state and catalog records..." size="lg" />;
  }

  if (error) {
    return <ErrorMessage title="Dashboard Overview Error" message={error} onRetry={fetchOverviewData} />;
  }

  const gpuName = hardware?.gpus && hardware.gpus.length > 0 ? hardware.gpus[0]?.name : null;

  return (
    <div className="space-y-8 pb-12">
      {/* Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-r from-slate-900 via-slate-900/90 to-sky-950/40 p-6 sm:p-8">
        <div className="max-w-2xl">
          <div className="flex items-center gap-2 text-sky-400 font-semibold text-xs tracking-wider uppercase">
            <Activity className="w-4 h-4" /> System Intelligence Overview
          </div>
          <h1 className="mt-2 text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            LLM Inference Intelligence Engine
          </h1>
          <p className="mt-2 text-sm text-slate-300 leading-relaxed">
            Measurement-driven benchmarking, Roofline efficiency diagnostics, empirical performance
            prediction, and constrained multi-objective Pareto configuration optimization.
          </p>
        </div>
      </div>

      {/* Primary KPI Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Hardware Platform"
          value={gpuName || (hardware?.gpus?.length === 0 ? 'CPU Only' : 'Unknown')}
          subtitle={`${hardware?.host.os_name || 'OS'} • ${hardware?.cpu.physical_cores || '?'} Cores`}
          badge={
            hardware && (
              <EvidenceBadge
                executionMode={hardware.execution_mode}
                isSynthetic={hardware.is_simulated}
              />
            )
          }
          icon={<Cpu className="w-4 h-4" />}
        />

        <MetricCard
          title="Benchmark Suites"
          value={totalBenchmarks}
          unit="runs recorded"
          subtitle={`${recentBenchmarks.length} loaded in view`}
          icon={<Gauge className="w-4 h-4" />}
        />

        <MetricCard
          title="Diagnostic Reports"
          value={totalDiagnostics}
          unit="analyses"
          subtitle="Bottleneck & roofline evaluations"
          icon={<Stethoscope className="w-4 h-4" />}
        />

        <MetricCard
          title="Optimization Runs"
          value={totalOptimizations}
          unit="Pareto searches"
          subtitle="Constrained configuration sizings"
          icon={<Sparkles className="w-4 h-4" />}
        />
      </div>

      {/* Quick Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div
          onClick={() => onNavigate('benchmarks')}
          className="group cursor-pointer rounded-xl border border-slate-800 bg-slate-900/40 p-5 hover:border-sky-500/50 hover:bg-slate-900/80 transition-all"
        >
          <div className="flex items-center justify-between">
            <div className="w-10 h-10 rounded-lg bg-sky-950/80 border border-sky-800/60 flex items-center justify-center text-sky-400 group-hover:scale-105 transition-transform">
              <Gauge className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-sky-400 group-hover:translate-x-1 transition-all" />
          </div>
          <h3 className="mt-4 font-semibold text-slate-100 text-base">Benchmark Runner</h3>
          <p className="mt-1 text-xs text-slate-400 leading-relaxed">
            Execute latency and throughput workloads with real-time SSE token stream telemetry.
          </p>
        </div>

        <div
          onClick={() => onNavigate('diagnostics')}
          className="group cursor-pointer rounded-xl border border-slate-800 bg-slate-900/40 p-5 hover:border-emerald-500/50 hover:bg-slate-900/80 transition-all"
        >
          <div className="flex items-center justify-between">
            <div className="w-10 h-10 rounded-lg bg-emerald-950/80 border border-emerald-800/60 flex items-center justify-center text-emerald-400 group-hover:scale-105 transition-transform">
              <Stethoscope className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 group-hover:translate-x-1 transition-all" />
          </div>
          <h3 className="mt-4 font-semibold text-slate-100 text-base">Bottleneck Diagnostics</h3>
          <p className="mt-1 text-xs text-slate-400 leading-relaxed">
            Examine Roofline operational intensity ceilings and memory-bandwidth bound regimes.
          </p>
        </div>

        <div
          onClick={() => onNavigate('optimizer')}
          className="group cursor-pointer rounded-xl border border-slate-800 bg-slate-900/40 p-5 hover:border-purple-500/50 hover:bg-slate-900/80 transition-all"
        >
          <div className="flex items-center justify-between">
            <div className="w-10 h-10 rounded-lg bg-purple-950/80 border border-purple-800/60 flex items-center justify-center text-purple-400 group-hover:scale-105 transition-transform">
              <Sparkles className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-purple-400 group-hover:translate-x-1 transition-all" />
          </div>
          <h3 className="mt-4 font-semibold text-slate-100 text-base">Configuration Optimizer</h3>
          <p className="mt-1 text-xs text-slate-400 leading-relaxed">
            Search multi-dimensional configurations and discover Pareto-optimal trade-offs.
          </p>
        </div>
      </div>

      {/* Two-column summary of recent activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Benchmarks */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-sky-400" />
              <h3 className="text-sm font-semibold text-slate-200">Recent Benchmark Suites</h3>
            </div>
            <button
              onClick={() => onNavigate('benchmarks')}
              className="text-xs text-sky-400 hover:text-sky-300 font-medium inline-flex items-center gap-1"
            >
              View all <ArrowRight className="w-3 h-3" />
            </button>
          </div>

          {recentBenchmarks.length === 0 ? (
            <div className="py-8 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg">
              No benchmark suites recorded yet. Launch a benchmark to begin.
            </div>
          ) : (
            <div className="space-y-2">
              {recentBenchmarks.map((suite) => (
                <div
                  key={suite.suite_id}
                  className="flex items-center justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 hover:border-slate-700 text-xs"
                >
                  <div className="min-w-0 pr-3">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-slate-200 font-mono truncate">
                        {suite.model_id}
                      </span>
                      <EvidenceBadge executionMode={suite.execution_mode} provenance={suite.provenance} />
                    </div>
                    <div className="mt-1 text-[11px] text-slate-400 flex items-center gap-3">
                      <span>Engine: {suite.engine_type}</span>
                      <span>TTFT p50: {suite.ttft_p50_ms ? `${suite.ttft_p50_ms.toFixed(1)} ms` : 'N/A'}</span>
                      <span>Throughput: {suite.aggregate_output_tokens_per_second ? `${suite.aggregate_output_tokens_per_second.toFixed(1)} tok/s` : 'N/A'}</span>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500 whitespace-nowrap">
                    {new Date(suite.timestamp_utc).toLocaleDateString()}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Recent Diagnostics & Jobs */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-semibold text-slate-200">Recent Diagnostic Reports</h3>
            </div>
            <button
              onClick={() => onNavigate('diagnostics')}
              className="text-xs text-emerald-400 hover:text-emerald-300 font-medium inline-flex items-center gap-1"
            >
              View all <ArrowRight className="w-3 h-3" />
            </button>
          </div>

          {recentDiagnostics.length === 0 ? (
            <div className="py-8 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg">
              No diagnostic reports stored. Run diagnostics on a benchmark suite.
            </div>
          ) : (
            <div className="space-y-2">
              {recentDiagnostics.map((diag) => (
                <div
                  key={diag.report_id}
                  className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 hover:border-slate-700 text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-200 font-mono truncate">
                      {diag.model_id}
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">
                      {new Date(diag.timestamp_utc).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="mt-1 text-[11px] text-slate-400 line-clamp-1">{diag.summary}</p>
                  <div className="mt-2 flex items-center gap-2">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                      {diag.findings.length} findings
                    </span>
                    {diag.roofline_analysis && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800/60">
                        {diag.roofline_analysis.limiting_factor || 'Roofline Analyzed'}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
