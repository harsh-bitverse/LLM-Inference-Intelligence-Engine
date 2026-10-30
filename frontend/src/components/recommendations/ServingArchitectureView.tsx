import React, { useState, useEffect } from 'react';
import {
  Server,
  Cpu,
  Layers,
  Zap,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
  Play,
  Clock,
  ShieldCheck,
  Split,
  Activity,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import {
  ServingCapabilitiesResponse,
  ServingCapabilityMatrix,
  ServingArchitectureComparison,
  HeadOfLineComparisonObservation,
  ServingArchitectureRecommendation,
  ArchitectureRecommendationRequest,
  CapabilityVerificationStatus,
} from '../../api/types';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorMessage } from '../common/ErrorMessage';

export const ServingArchitectureView: React.FC = () => {
  // Telemetry & Capabilities
  const [capabilitiesData, setCapabilitiesData] = useState<ServingCapabilitiesResponse | null>(null);
  const [comparisons, setComparisons] = useState<ServingArchitectureComparison[]>([]);
  const [holComparisons, setHolComparisons] = useState<HeadOfLineComparisonObservation[]>([]);
  const [loadingInitial, setLoadingInitial] = useState<boolean>(true);
  const [initialError, setInitialError] = useState<string | null>(null);

  // Model filter for comparisons
  const [selectedModel, setSelectedModel] = useState<string>('all');

  // Recommender Form State
  const [recModelId, setRecModelId] = useState<string>('qwen2.5:0.5b');
  const [recWorkloadType, setRecWorkloadType] = useState<string>('mixed_chat_summary');
  const [recConcurrency, setRecConcurrency] = useState<number>(2);
  const [recArrivalRate, setRecArrivalRate] = useState<string>('1.5');
  const [recLatencySlaMs, setRecLatencySlaMs] = useState<string>('2000');
  const [recMemoryBudgetMb, setRecMemoryBudgetMb] = useState<string>('8192');

  const [recommending, setRecommending] = useState<boolean>(false);
  const [recommendation, setRecommendation] = useState<ServingArchitectureRecommendation | null>(null);
  const [recError, setRecError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchInitialData = async () => {
      setLoadingInitial(true);
      setInitialError(null);
      try {
        const [caps, comps, hols] = await Promise.all([
          apiClient.getServingCapabilities(),
          apiClient.getArchitectureComparisons(),
          apiClient.getServingHolComparisons(),
        ]);
        if (isMounted) {
          setCapabilitiesData(caps);
          setComparisons(comps.comparisons || []);
          setHolComparisons(hols.hol_comparisons || []);
        }
      } catch (err: unknown) {
        if (isMounted) {
          setInitialError(err instanceof Error ? err.message : 'Failed to fetch serving architecture data');
        }
      } finally {
        if (isMounted) {
          setLoadingInitial(false);
        }
      }
    };
    fetchInitialData();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleGetRecommendation = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setRecommending(true);
    setRecError(null);
    try {
      const req: ArchitectureRecommendationRequest = {
        model_id: recModelId,
        workload_type: recWorkloadType,
        client_concurrency: recConcurrency,
        arrival_rate_rps: recArrivalRate ? parseFloat(recArrivalRate) : undefined,
        latency_sla_ms: recLatencySlaMs ? parseFloat(recLatencySlaMs) : undefined,
        memory_budget_mb: recMemoryBudgetMb ? parseFloat(recMemoryBudgetMb) : null,
      };
      const res = await apiClient.recommendServingArchitecture(req);
      setRecommendation(res);
    } catch (err: unknown) {
      setRecError(err instanceof Error ? err.message : 'Failed to obtain serving architecture recommendation');
    } finally {
      setRecommending(false);
    }
  };

  const getStatusBadge = (status: CapabilityVerificationStatus) => {
    switch (status) {
      case 'VERIFIED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="w-3 h-3" /> Verified
          </span>
        );
      case 'NOT_SUPPORTED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <XCircle className="w-3 h-3" /> Not Supported
          </span>
        );
      case 'ADVERTISED_UNVERIFIED':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <AlertTriangle className="w-3 h-3" /> Advertised (Unverified)
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-slate-500/10 text-slate-400 border border-slate-500/20">
            <HelpCircle className="w-3 h-3" /> Unknown
          </span>
        );
    }
  };

  const filteredComparisons = selectedModel === 'all'
    ? comparisons
    : comparisons.filter((c) => c.model_id === selectedModel);

  const filteredHol = selectedModel === 'all'
    ? holComparisons
    : holComparisons.filter((h) => h.model_id === selectedModel);

  if (loadingInitial) {
    return (
      <div className="flex flex-col items-center justify-center p-12">
        <LoadingSpinner />
        <p className="mt-4 text-sm text-slate-400">Loading serving architecture telemetry and benchmarks...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {initialError && <ErrorMessage message={initialError} />}

      {/* Header & Host Environment Overview */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <Server className="w-5 h-5 text-sky-400" />
              Serving Architecture & Parallelism Evaluation
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Empirical characterization of multi-slot execution, continuous batching, and queue wait elimination across Ollama and llama.cpp server.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Ollama Daemon (Port {capabilitiesData?.ollama_port ?? 11434})
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Cpu className="w-3.5 h-3.5" />
              llama-server {capabilitiesData?.llama_server_binary_installed ? 'Installed' : 'Missing'}
            </span>
          </div>
        </div>

        {capabilitiesData && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
            <div className="bg-slate-950/50 p-3 rounded-lg border border-slate-800/60">
              <div className="text-slate-500 font-medium mb-1">Host CPU Architecture</div>
              <div className="font-semibold text-slate-200">{capabilitiesData.host_hardware.cpu}</div>
              <div className="text-slate-400 mt-0.5">{capabilitiesData.host_hardware.hardware_threads} hardware execution threads</div>
            </div>
            <div className="bg-slate-950/50 p-3 rounded-lg border border-slate-800/60">
              <div className="text-slate-500 font-medium mb-1">System Memory & KV Partitioning</div>
              <div className="font-semibold text-slate-200">{capabilitiesData.host_hardware.ram_gb} GB Physical RAM</div>
              <div className="text-slate-400 mt-0.5">Partitioned per active parallel execution slot</div>
            </div>
            <div className="bg-slate-950/50 p-3 rounded-lg border border-slate-800/60">
              <div className="text-slate-500 font-medium mb-1">Serving Engine Parity</div>
              <div className="font-semibold text-slate-200">Continuous Batching Enabled</div>
              <div className="text-slate-400 mt-0.5">Iterative token decode across concurrent requests</div>
            </div>
          </div>
        )}
      </div>

      {/* Model Filter Selector */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Layers className="w-4 h-4 text-sky-400" />
          Filter Empirical Architecture Results
        </div>
        <div className="flex items-center gap-1.5">
          {['all', 'qwen2.5:0.5b', 'qwen2.5:1.5b', 'llama3.2:1b'].map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setSelectedModel(m)}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                selectedModel === m
                  ? 'bg-sky-500 text-white shadow-sm'
                  : 'bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700'
              }`}
            >
              {m === 'all' ? 'All Models' : m}
            </button>
          ))}
        </div>
      </div>

      {/* Section 1: Capability Matrix */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 mb-2 flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-sky-400" />
          Inference Serving Capability Matrix (Advertised vs Empirically Observed)
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Rigorous differentiation between client concurrency (multiple outstanding HTTP calls), server execution slots, static batching, and continuous batching.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-950/40 text-slate-400">
                <th className="py-2.5 px-3 font-semibold">Runtime & Architecture</th>
                <th className="py-2.5 px-3 font-semibold">Model</th>
                <th className="py-2.5 px-3 font-semibold text-center">Slots</th>
                <th className="py-2.5 px-3 font-semibold text-center">Parallel Exec</th>
                <th className="py-2.5 px-3 font-semibold text-center">Static Batching</th>
                <th className="py-2.5 px-3 font-semibold text-center">Continuous Batching</th>
                <th className="py-2.5 px-3 font-semibold">KV Cache Memory Behavior</th>
                <th className="py-2.5 px-3 font-semibold">Provenance & Verification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {capabilitiesData?.capability_matrix
                .filter((row) => selectedModel === 'all' || row.model_id === selectedModel)
                .map((row: ServingCapabilityMatrix, idx: number) => (
                  <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-3 font-medium text-slate-200">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-slate-300">{row.architecture_type}</span>
                      </div>
                      <div className="text-[10px] text-slate-500">{row.runtime_name} ({row.runtime_version})</div>
                    </td>
                    <td className="py-2.5 px-3 font-mono text-slate-300">
                      {row.model_id}
                      <span className="block text-[10px] text-slate-500 font-sans">{row.quantization}</span>
                    </td>
                    <td className="py-2.5 px-3 text-center font-mono text-slate-300">
                      {row.configurable_slots}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      {getStatusBadge(row.parallel_request_support)}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      {getStatusBadge(row.batching_support)}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      {getStatusBadge(row.continuous_batching_support)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300 text-[11px] max-w-xs">
                      {row.kv_cache_memory_behavior}
                    </td>
                    <td className="py-2.5 px-3 text-[11px] text-slate-400 max-w-xs">
                      <div className="font-medium text-slate-300">{row.evidence_source}</div>
                      <div className="text-[10px] text-slate-500 line-clamp-1" title={row.verification_notes}>
                        {row.verification_notes}
                      </div>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 2: Head-of-Line Blocking Rescue */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 mb-2 flex items-center gap-2">
          <Clock className="w-4 h-4 text-amber-400" />
          Head-of-Line (HoL) Blocking Mitigation Under Serving Parallelism
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Under single-slot serial execution (Ollama), short interactive requests trapped behind long-running queries suffer severe queueing delay. Continuous batching interleaves generation and eliminates head-of-line blocking.
        </p>

        {filteredHol.length === 0 ? (
          <p className="text-xs text-slate-500">No Head-of-Line blocking observations available for this model filter.</p>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {filteredHol.map((hol) => (
              <div key={hol.experiment_id} className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800/80">
                  <span className="font-mono text-xs font-semibold text-slate-200">{hol.model_id}</span>
                  {hol.sla_rescued_by_parallelism ? (
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      <CheckCircle2 className="w-3 h-3" /> SLA Rescued by Parallelism
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-semibold bg-slate-500/10 text-slate-400 border border-slate-500/20">
                      Standard Improvement
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs mb-3">
                  <div className="p-2.5 rounded-lg bg-rose-950/20 border border-rose-900/30">
                    <div className="text-rose-400 font-semibold mb-1">Ollama (Single Slot)</div>
                    <div className="space-y-0.5 text-slate-300">
                      <div>Queue Wait: <span className="font-mono font-medium text-rose-300">{hol.ollama_consequent_wait_ms.toFixed(1)} ms</span></div>
                      <div>End-to-End: <span className="font-mono text-slate-300">{hol.ollama_consequent_e2e_ms.toFixed(1)} ms</span></div>
                      <div className="text-[11px] font-semibold mt-1">
                        SLA Status: {hol.ollama_sla_met ? (
                          <span className="text-emerald-400">Met</span>
                        ) : (
                          <span className="text-rose-400">FAILED (Breach)</span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-emerald-950/20 border border-emerald-900/30">
                    <div className="text-emerald-400 font-semibold mb-1">llama-server (2 Slots / CB)</div>
                    <div className="space-y-0.5 text-slate-300">
                      <div>Queue Wait: <span className="font-mono font-medium text-emerald-300">{hol.llama_server_consequent_wait_ms.toFixed(1)} ms</span></div>
                      <div>End-to-End: <span className="font-mono text-slate-300">{hol.llama_server_consequent_e2e_ms.toFixed(1)} ms</span></div>
                      <div className="text-[11px] font-semibold mt-1">
                        SLA Status: {hol.llama_server_sla_met ? (
                          <span className="text-emerald-400">MET (Pass)</span>
                        ) : (
                          <span className="text-rose-400">Failed</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-900/60 p-2.5 rounded-lg text-xs space-y-1 text-slate-300">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Queue Wait Reduction:</span>
                    <span className="font-mono font-semibold text-emerald-400">{hol.queue_wait_reduction_ms.toFixed(1)} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Turnaround Speedup:</span>
                    <span className="font-mono font-semibold text-sky-400">{hol.latency_reduction_factor.toFixed(2)}x faster</span>
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">
                    Consequent SLA Target: {hol.consequent_sla_ms} ms | Antecedent: {hol.antecedent_workload}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Section 3: Empirical Concurrency & Architecture Comparisons */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 mb-2 flex items-center gap-2">
          <Activity className="w-4 h-4 text-sky-400" />
          Empirical Architecture Benchmarks (Throughput, Tail Latency & Memory)
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Measured head-to-head performance across concurrency levels under identical prompt and output distributions.
        </p>

        {filteredComparisons.length === 0 ? (
          <p className="text-xs text-slate-500">No architecture comparison data available for this model filter.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {filteredComparisons.map((comp) => (
              <div key={comp.comparison_id} className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-mono text-xs font-semibold text-slate-200">{comp.model_id}</div>
                    <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/20">
                      Concurrency C = {comp.client_concurrency}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-3">{comp.workload_description}</p>

                  <div className="grid grid-cols-3 gap-2 mb-3 text-center">
                    <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                      <div className="text-[10px] uppercase font-semibold text-slate-400">Throughput Gain</div>
                      <div className={`text-sm font-bold font-mono mt-0.5 ${comp.throughput_gain_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {comp.throughput_gain_pct >= 0 ? `+${comp.throughput_gain_pct.toFixed(1)}%` : `${comp.throughput_gain_pct.toFixed(1)}%`}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-0.5">
                        {comp.baseline_throughput_rps.toFixed(2)} → {comp.target_throughput_rps.toFixed(2)} RPS
                      </div>
                    </div>

                    <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                      <div className="text-[10px] uppercase font-semibold text-slate-400">P95 Speedup</div>
                      <div className="text-sm font-bold font-mono mt-0.5 text-sky-400">
                        {comp.latency_p95_speedup_factor.toFixed(2)}x
                      </div>
                      <div className="text-[10px] text-slate-500 mt-0.5">
                        {comp.baseline_p95_ms.toFixed(0)}ms → {comp.target_p95_ms.toFixed(0)}ms
                      </div>
                    </div>

                    <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                      <div className="text-[10px] uppercase font-semibold text-slate-400">Queue Reduction</div>
                      <div className="text-sm font-bold font-mono mt-0.5 text-emerald-400">
                        -{comp.queue_wait_reduction_ms.toFixed(0)}ms
                      </div>
                      <div className="text-[10px] text-slate-500 mt-0.5">
                        Wait: {comp.target_mean_queue_wait_ms.toFixed(0)}ms
                      </div>
                    </div>
                  </div>

                  <div className="text-xs space-y-1.5 bg-slate-900/40 p-2.5 rounded-lg border border-slate-800/60 mb-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">SLA Compliance:</span>
                      <span className="font-mono text-slate-200">
                        {comp.baseline_sla_compliance_pct.toFixed(0)}% → <span className="font-semibold text-emerald-400">{comp.target_sla_compliance_pct.toFixed(0)}%</span>
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Memory Footprint:</span>
                      <span className="font-mono text-slate-200">
                        {comp.baseline_memory_mb.toFixed(0)} MB → {comp.target_memory_mb.toFixed(0)} MB (+{comp.memory_overhead_mb.toFixed(0)} MB)
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Architectures:</span>
                      <span className="font-mono text-[11px] text-slate-400">
                        {comp.baseline_architecture} → {comp.target_architecture}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="text-[10px] text-slate-500 italic border-t border-slate-800/60 pt-2">
                  {comp.scientific_provenance_summary}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Section 4: Interactive Architecture Recommender */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Zap className="w-4 h-4 text-sky-400" />
          <h3 className="text-sm font-semibold text-slate-200">
            Automated Inference Serving Architecture Recommender
          </h3>
        </div>

        <form onSubmit={handleGetRecommendation} className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Target Model</label>
            <select
              value={recModelId}
              onChange={(e) => setRecModelId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option value="qwen2.5:0.5b">qwen2.5:0.5b (Q4_K_M)</option>
              <option value="qwen2.5:1.5b">qwen2.5:1.5b (Q4_K_M)</option>
              <option value="llama3.2:1b">llama3.2:1b (Q8_0)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Workload Profile</label>
            <select
              value={recWorkloadType}
              onChange={(e) => setRecWorkloadType(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option value="mixed_chat_summary">Mixed Chat & Summarization (Interactive)</option>
              <option value="code_generation">Code Generation (Long Context & Decode)</option>
              <option value="high_concurrency_batch">High Concurrency Multi-Client</option>
              <option value="single_stream_interactive">Single Client Serial Stream</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Expected Concurrency (Clients)</label>
            <input
              type="number"
              min={1}
              max={8}
              value={recConcurrency}
              onChange={(e) => setRecConcurrency(parseInt(e.target.value, 10) || 1)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Target Arrival Rate (RPS)</label>
            <input
              type="text"
              value={recArrivalRate}
              onChange={(e) => setRecArrivalRate(e.target.value)}
              placeholder="e.g. 1.5"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Latency SLA (ms)</label>
            <input
              type="text"
              value={recLatencySlaMs}
              onChange={(e) => setRecLatencySlaMs(e.target.value)}
              placeholder="e.g. 2000"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Memory Budget (MB)</label>
            <input
              type="text"
              value={recMemoryBudgetMb}
              onChange={(e) => setRecMemoryBudgetMb(e.target.value)}
              placeholder="e.g. 8192"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div className="md:col-span-3 flex justify-end">
            <button
              type="submit"
              disabled={recommending}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold transition-all disabled:opacity-50"
            >
              {recommending ? <LoadingSpinner /> : <Play className="w-3.5 h-3.5" />}
              Generate Serving Recommendation
            </button>
          </div>
        </form>

        {recError && <ErrorMessage message={recError} />}

        {recommendation && (
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
              <div>
                <span className="text-[10px] uppercase font-semibold text-slate-400">Recommended Architecture</span>
                <div className="text-base font-bold text-sky-400 font-mono flex items-center gap-2">
                  <Split className="w-4 h-4 text-sky-400" />
                  {recommendation.recommended_architecture}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-1 rounded text-xs font-semibold bg-sky-500/10 text-sky-300 border border-sky-500/20 font-mono">
                  {recommendation.recommended_slots} Execution Slots
                </span>
                <span className="px-2.5 py-1 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {recommendation.evidence_tier} Evidence
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/60 text-center">
                <div className="text-[10px] text-slate-400 uppercase font-semibold">Expected Throughput</div>
                <div className="text-lg font-bold text-slate-100 font-mono mt-0.5">
                  {recommendation.expected_throughput_rps.toFixed(2)} RPS
                </div>
              </div>
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/60 text-center">
                <div className="text-[10px] text-slate-400 uppercase font-semibold">Expected P95 Latency</div>
                <div className="text-lg font-bold text-slate-100 font-mono mt-0.5">
                  {recommendation.expected_p95_latency_ms.toFixed(0)} ms
                </div>
              </div>
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/60 text-center">
                <div className="text-[10px] text-slate-400 uppercase font-semibold">SLA Compliance</div>
                <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">
                  {recommendation.expected_sla_compliance_pct.toFixed(0)}%
                </div>
              </div>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <span className="font-semibold text-slate-300">Decision Rationale:</span>
                <p className="text-slate-400 mt-1">{recommendation.rationale}</p>
              </div>

              {recommendation.tradeoffs.length > 0 && (
                <div>
                  <span className="font-semibold text-slate-300">Operational Tradeoffs:</span>
                  <ul className="list-disc list-inside text-slate-400 mt-1 space-y-0.5">
                    {recommendation.tradeoffs.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div>
                <span className="font-semibold text-slate-300">Deployment Command Guidance:</span>
                <pre className="mt-1 p-3 bg-slate-900 rounded-lg text-slate-300 font-mono text-[11px] overflow-x-auto border border-slate-800">
                  {recommendation.actionable_setup_guidance}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
