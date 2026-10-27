import React, { useEffect, useState } from 'react';
import {
  Stethoscope,
  Play,
  AlertTriangle,
  Info,
  AlertCircle,
  Clock,
  Layers,
  Cpu,
  HardDrive,
  Eye,
} from 'lucide-react';
import { apiClient } from '../api/client';
import {
  BenchmarkSuiteResult,
  DiagnosticReport,
  DiagnosticSeverity,
} from '../api/types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorMessage } from '../components/common/ErrorMessage';
import { EvidenceBadge } from '../components/common/EvidenceBadge';

export const DiagnosticsPage: React.FC = () => {
  const [suiteId, setSuiteId] = useState<string>('');
  const [availableSuites, setAvailableSuites] = useState<BenchmarkSuiteResult[]>([]);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const [activeReport, setActiveReport] = useState<DiagnosticReport | null>(null);
  const [history, setHistory] = useState<DiagnosticReport[]>([]);
  const [loadingHistory, setLoadingHistory] = useState<boolean>(true);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const fetchInitial = async () => {
    setLoadingHistory(true);
    setHistoryError(null);
    try {
      const [diagRes, benchRes] = await Promise.allSettled([
        apiClient.listDiagnostics(50),
        apiClient.listBenchmarks({ limit: 50 }),
      ]);

      if (diagRes.status === 'fulfilled') {
        setHistory(diagRes.value.items);
        if (diagRes.value.items.length > 0 && !activeReport) {
          setActiveReport(diagRes.value.items[0] || null);
        }
      }
      if (benchRes.status === 'fulfilled') {
        setAvailableSuites(benchRes.value.items);
        if (benchRes.value.items.length > 0 && !suiteId) {
          setSuiteId(benchRes.value.items[0]?.suite_id || '');
        }
      }
    } catch (err: unknown) {
      setHistoryError(err instanceof Error ? err.message : 'Failed to load diagnostics');
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    fetchInitial();
  }, []);

  const handleRunDiagnostics = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!suiteId.trim()) {
      setSubmitError('A Benchmark Suite ID is required.');
      return;
    }

    setSubmitting(true);
    setSubmitError(null);

    try {
      const report = await apiClient.runDiagnostics({ suite_id: suiteId.trim() });
      setActiveReport(report);
      setHistory((prev) => [report, ...prev]);
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : 'Diagnostic run failed');
    } finally {
      setSubmitting(false);
    }
  };

  const getSeverityBadge = (severity: DiagnosticSeverity) => {
    switch (severity) {
      case 'critical':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-red-950 text-red-300 border border-red-700/60">
            <AlertCircle className="w-3 h-3 text-red-400" /> Critical
          </span>
        );
      case 'warning':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-950 text-amber-300 border border-amber-700/60">
            <AlertTriangle className="w-3 h-3 text-amber-400" /> Warning
          </span>
        );
      case 'info':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-sky-950 text-sky-300 border border-sky-700/60">
            <Info className="w-3 h-3 text-sky-400" /> Info
          </span>
        );
    }
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
          Evidence-Based Bottleneck Diagnostics
        </h1>
        <p className="mt-1 text-xs text-slate-400">
          Apply analytical Roofline models, memory saturation checks, and latency efficiency diagnostics to measured benchmarks.
        </p>
      </div>

      {/* Launcher & Controls */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
          <Stethoscope className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-200">Run Diagnostics on Benchmark Suite</h3>
        </div>

        {submitError && <ErrorMessage title="Diagnostic Error" message={submitError} />}

        <form onSubmit={handleRunDiagnostics} className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-end text-xs">
          <div className="sm:col-span-8">
            <label className="block text-slate-300 font-medium mb-1">
              Select or Enter Benchmark Suite ID
            </label>
            <div className="flex gap-2">
              <select
                value={suiteId}
                onChange={(e) => setSuiteId(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono text-xs"
              >
                {availableSuites.map((s) => (
                  <option key={s.suite_id} value={s.suite_id}>
                    {s.suite_id} — {s.model_id} ({s.engine_type}, {s.execution_mode})
                  </option>
                ))}
              </select>
              <input
                type="text"
                value={suiteId}
                onChange={(e) => setSuiteId(e.target.value)}
                placeholder="Or custom suite-..."
                className="w-1/2 px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono text-xs"
              />
            </div>
          </div>

          <div className="sm:col-span-4">
            <button
              type="submit"
              disabled={submitting}
              className="w-full py-2 px-4 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-semibold rounded-lg shadow-sm transition-colors flex items-center justify-center gap-2 text-xs"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              {submitting ? 'Analyzing Suite...' : 'Analyze Bottlenecks'}
            </button>
          </div>
        </form>
      </div>

      {/* Main Report View */}
      {activeReport ? (
        <div className="space-y-6">
          {/* Executive Summary Card */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-3">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-bold text-slate-100 font-mono">
                    Report: {activeReport.report_id}
                  </h2>
                  <EvidenceBadge executionMode={activeReport.execution_mode} />
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  Model: {activeReport.model_id} • Engine: {activeReport.engine_type} • Date:{' '}
                  {new Date(activeReport.timestamp_utc).toLocaleString()}
                </p>
              </div>

              {activeReport.hardware_catalog_id && (
                <span className="text-xs font-mono px-2 py-1 rounded bg-slate-950 text-slate-300 border border-slate-800">
                  Target Spec: {activeReport.hardware_catalog_id}
                </span>
              )}
            </div>

            <p className="text-sm text-slate-200 leading-relaxed bg-slate-950/70 p-4 rounded-lg border border-slate-800/80 font-sans">
              {activeReport.summary}
            </p>

            {/* Roofline Model and Memory Ceilings */}
            {(activeReport.roofline_analysis || activeReport.resource_estimate) && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                {/* Roofline Ceilings */}
                {activeReport.roofline_analysis && (
                  <div className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 space-y-3 text-xs">
                    <div className="flex items-center gap-2 text-sky-400 font-semibold">
                      <Cpu className="w-4 h-4" /> Roofline Operational Intensity
                    </div>
                    <div className="grid grid-cols-2 gap-2 font-mono">
                      <div>
                        <span className="text-slate-500 block text-[11px]">OPERATIONAL INTENSITY</span>
                        <span className="text-slate-200 font-bold">
                          {activeReport.roofline_analysis.operational_intensity.toFixed(2)} FLOP/B
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[11px]">RIDGE POINT</span>
                        <span className="text-purple-400 font-bold">
                          {activeReport.roofline_analysis.ridge_point_flops_per_byte.toFixed(2)} FLOP/B
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[11px]">LIMITING FACTOR</span>
                        <span
                          className={`font-bold ${
                            activeReport.roofline_analysis.limiting_factor === 'COMPUTE_BOUND'
                              ? 'text-sky-400'
                              : 'text-amber-400'
                          }`}
                        >
                          {activeReport.roofline_analysis.limiting_factor}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[11px]">ATTAINABLE COMPUTE</span>
                        <span className="text-emerald-400 font-bold">
                          {activeReport.roofline_analysis.attainable_compute_tflops.toFixed(1)} TFLOPs
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Resource Estimation */}
                {activeReport.resource_estimate && (
                  <div className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 space-y-3 text-xs">
                    <div className="flex items-center gap-2 text-emerald-400 font-semibold">
                      <HardDrive className="w-4 h-4" /> Analytical Memory Requirements
                    </div>
                    <div className="grid grid-cols-2 gap-2 font-mono">
                      <div>
                        <span className="text-slate-500 block text-[11px]">WEIGHT STORAGE</span>
                        <span className="text-slate-200 font-bold">
                          {activeReport.resource_estimate.parameter_storage_mb.toFixed(1)} MB ({activeReport.resource_estimate.weight_precision})
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[11px]">KV CACHE / TOKEN</span>
                        <span className="text-slate-200 font-bold">
                          {activeReport.resource_estimate.kv_cache_per_token_bytes != null
                            ? `${activeReport.resource_estimate.kv_cache_per_token_bytes.toFixed(1)} Bytes`
                            : 'N/A'}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[11px]">TOTAL PAYLOAD MEMORY</span>
                        <span className="text-emerald-400 font-bold">
                          {activeReport.resource_estimate.total_payload_memory_mb.toFixed(1)} MB
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[11px]">ESTIMATED RUNTIME VRAM</span>
                        <span className="text-purple-400 font-bold">
                          {activeReport.resource_estimate.estimated_runtime_vram_mb != null
                            ? `${activeReport.resource_estimate.estimated_runtime_vram_mb.toFixed(1)} MB`
                            : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Diagnostic Findings List */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              Typed Bottleneck Findings ({activeReport.findings.length})
            </h3>

            {activeReport.findings.length === 0 ? (
              <div className="p-6 text-center border border-dashed border-slate-800 rounded-lg text-slate-500 text-xs">
                No bottleneck findings generated for this benchmark suite.
              </div>
            ) : (
              activeReport.findings.map((finding) => (
                <div
                  key={finding.finding_id}
                  className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-3 text-xs transition-colors hover:border-slate-700"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-800/80 pb-2.5">
                    <div className="flex items-center gap-2 flex-wrap">
                      {getSeverityBadge(finding.severity)}
                      <span className="font-bold text-slate-100 text-sm">{finding.title}</span>
                    </div>

                    <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
                      <span>Category: {finding.category}</span>
                      <span>•</span>
                      <span className="text-slate-300">Confidence: {finding.confidence}</span>
                    </div>
                  </div>

                  <p className="text-slate-300 leading-relaxed">{finding.message}</p>
                  <p className="text-slate-400 text-[11px] italic">
                    Confidence Rationale: {finding.confidence_rationale}
                  </p>

                  {/* Supporting Observations */}
                  {finding.supporting_observations && finding.supporting_observations.length > 0 && (
                    <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-1.5">
                      <span className="font-semibold text-slate-300 text-[11px] block">
                        Supporting Quantitative Observations:
                      </span>
                      <div className="space-y-1">
                        {finding.supporting_observations.map((obs, idx) => (
                          <div key={idx} className="flex items-center justify-between text-[11px] font-mono">
                            <span className="text-slate-400">{obs.description || 'Metric Observation'}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-slate-200 font-bold">
                                {String(obs.value)} {obs.unit || ''}
                              </span>
                              <EvidenceBadge executionMode={obs.execution_mode} provenance={obs.provenance} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Suggested Experiments */}
                  {finding.suggested_experiments && finding.suggested_experiments.length > 0 && (
                    <div className="pt-1 text-[11px] text-slate-400 space-y-1">
                      <span className="font-medium text-emerald-400">Suggested Verification Experiments:</span>
                      <ul className="list-disc list-inside space-y-0.5 text-slate-300">
                        {finding.suggested_experiments.map((exp, i) => (
                          <li key={i}>{exp}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        <div className="py-12 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-xl">
          Select or run a diagnostic report above to inspect bottleneck findings.
        </div>
      )}

      {/* Historical Diagnostics Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-slate-200">Historical Diagnostic Reports</h3>
          </div>
          <button
            onClick={fetchInitial}
            className="text-xs text-sky-400 hover:text-sky-300 flex items-center gap-1"
          >
            <Clock className="w-3.5 h-3.5" /> Refresh
          </button>
        </div>

        {loadingHistory ? (
          <LoadingSpinner message="Fetching historical reports..." />
        ) : historyError ? (
          <ErrorMessage title="Error loading diagnostic history" message={historyError} onRetry={fetchInitial} />
        ) : history.length === 0 ? (
          <div className="py-8 text-center text-xs text-slate-500">No historical diagnostic reports stored.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300 font-mono">
              <thead className="bg-slate-950/80 text-slate-400 font-semibold border-b border-slate-800 text-[11px] uppercase">
                <tr>
                  <th className="p-2.5">Report ID</th>
                  <th className="p-2.5">Model</th>
                  <th className="p-2.5">Engine</th>
                  <th className="p-2.5">Findings</th>
                  <th className="p-2.5">Date</th>
                  <th className="p-2.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {history.map((h) => (
                  <tr key={h.report_id} className="hover:bg-slate-800/30">
                    <td className="p-2.5 font-bold text-slate-200">{h.report_id}</td>
                    <td className="p-2.5 text-slate-300">{h.model_id}</td>
                    <td className="p-2.5 uppercase text-slate-400">{h.engine_type}</td>
                    <td className="p-2.5 text-emerald-400 font-semibold">{h.findings.length} findings</td>
                    <td className="p-2.5 text-slate-500">{new Date(h.timestamp_utc).toLocaleDateString()}</td>
                    <td className="p-2.5 text-right">
                      <button
                        onClick={() => setActiveReport(h)}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-400 text-xs font-sans inline-flex items-center gap-1"
                      >
                        <Eye className="w-3 h-3" /> View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
