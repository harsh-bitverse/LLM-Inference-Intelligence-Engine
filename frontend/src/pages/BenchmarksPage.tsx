import React, { useEffect, useState, useRef } from 'react';
import {
  Play,
  StopCircle,
  Clock,
  Eye,
  Terminal,
  Activity,
  Layers,
} from 'lucide-react';
import { apiClient } from '../api/client';
import {
  BenchmarkSubmissionRequest,
  BenchmarkSuiteResult,
  EngineType,
  JobStatus,
  QuantizationType,
  TokenStreamEvent,
} from '../api/types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorMessage } from '../components/common/ErrorMessage';
import { EvidenceBadge } from '../components/common/EvidenceBadge';

export const BenchmarksPage: React.FC = () => {
  // Form State
  const [modelId, setModelId] = useState<string>('meta-llama/Llama-3-8B');
  const [engineType, setEngineType] = useState<EngineType>('mock');
  const [quantization, setQuantization] = useState<QuantizationType>('none');
  const [paramBillion, setParamBillion] = useState<number>(8.0);
  const [promptTokens, setPromptTokens] = useState<number>(64);
  const [outputTokens, setOutputTokens] = useState<number>(32);
  const [concurrency, setConcurrency] = useState<number>(1);
  const [warmupRequests, setWarmupRequests] = useState<number>(0);
  const [sampleMemory, setSampleMemory] = useState<boolean>(true);
  const [endpointUrl, setEndpointUrl] = useState<string>('');

  // Active Job & Streaming State
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJobStatus, setActiveJobStatus] = useState<JobStatus | null>(null);
  const [streamEvents, setStreamEvents] = useState<TokenStreamEvent[]>([]);
  const [totalStreamEventsCount, setTotalStreamEventsCount] = useState<number>(0);
  const [streamConnected, setStreamConnected] = useState<boolean>(false);
  const [activeJobResult, setActiveJobResult] = useState<BenchmarkSuiteResult | null>(null);
  const [cancelling, setCancelling] = useState<boolean>(false);
  const [formError, setFormError] = useState<string | null>(null);

  // History State
  const [history, setHistory] = useState<BenchmarkSuiteResult[]>([]);
  const [loadingHistory, setLoadingHistory] = useState<boolean>(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selectedSuite, setSelectedSuite] = useState<BenchmarkSuiteResult | null>(null);

  // Stream cleanup ref
  const streamCleanupRef = useRef<(() => void) | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeJobIdRef = useRef<string | null>(null);

  const fetchHistory = async () => {
    setLoadingHistory(true);
    setHistoryError(null);
    try {
      const res = await apiClient.listBenchmarks({ limit: 50 });
      setHistory(res.items);
    } catch (err: unknown) {
      setHistoryError(err instanceof Error ? err.message : 'Failed to load benchmark history');
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    fetchHistory();
    return () => {
      if (streamCleanupRef.current) streamCleanupRef.current();
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  // Validation
  const validateForm = (): string | null => {
    if (!modelId.trim()) return 'Model identifier is required.';
    if (paramBillion <= 0) return 'Parameter count must be greater than 0.';
    if (promptTokens <= 0) return 'Prompt tokens must be at least 1.';
    if (outputTokens < 0) return 'Output tokens cannot be negative.';
    if (concurrency < 1) return 'Concurrency must be at least 1.';
    if (warmupRequests < 0) return 'Warmup requests cannot be negative.';
    return null;
  };

  // Launch Benchmark
  const handleLaunchBenchmark = async (e: React.FormEvent) => {
    e.preventDefault();
    const validationErr = validateForm();
    if (validationErr) {
      setFormError(validationErr);
      return;
    }

    setFormError(null);
    setSubmitting(true);
    setStreamEvents([]);
    setTotalStreamEventsCount(0);
    setActiveJobResult(null);

    const payload: BenchmarkSubmissionRequest = {
      model: {
        model_id: modelId.trim(),
        family: modelId.toLowerCase().includes('llama') ? 'llama' : 'unknown',
        parameter_count_billion: paramBillion,
        quantization,
      },
      engine_type: engineType,
      workloads: [
        {
          prompt_tokens: promptTokens,
          output_tokens: outputTokens,
        },
      ],
      concurrency,
      warmup_requests: warmupRequests,
      sample_memory: sampleMemory,
      endpoint_url: endpointUrl.trim() || undefined,
    };

    try {
      const { job_id } = await apiClient.submitBenchmark(payload);
      activeJobIdRef.current = job_id;
      setActiveJobId(job_id);
      setActiveJobStatus({
        job_id,
        status: 'PENDING',
        created_at: new Date().toISOString(),
        model_id: modelId,
        engine_type: engineType,
      });

      // Start SSE stream
      setStreamConnected(true);
      if (streamCleanupRef.current) streamCleanupRef.current();

      streamCleanupRef.current = apiClient.subscribeJobStream(
        job_id,
        (event) => {
          if (activeJobIdRef.current !== job_id) return;
          setStreamEvents((prev) => [event, ...prev.slice(0, 99)]); // Keep last 100
          setTotalStreamEventsCount((prev) => prev + 1);
        },
        () => {
          if (activeJobIdRef.current === job_id) {
            setStreamConnected(false);
          }
        },
        () => {
          if (activeJobIdRef.current === job_id) {
            setStreamConnected(false);
          }
        }
      );

      // Start polling job status
      if (pollingRef.current) clearInterval(pollingRef.current);
      pollingRef.current = setInterval(async () => {
        if (activeJobIdRef.current !== job_id) {
          if (pollingRef.current) clearInterval(pollingRef.current);
          return;
        }
        try {
          const statusRes = await apiClient.getJobStatus(job_id);
          if (activeJobIdRef.current !== job_id) return;
          setActiveJobStatus(statusRes.job);

          if (
            statusRes.job.status === 'COMPLETED' ||
            statusRes.job.status === 'FAILED' ||
            statusRes.job.status === 'CANCELLED'
          ) {
            if (pollingRef.current) clearInterval(pollingRef.current);
            if (statusRes.result) {
              setActiveJobResult(statusRes.result);
              fetchHistory(); // Refresh history
            }
          }
        } catch {
          // Ignore transient poll errors
        }
      }, 500);
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : 'Benchmark submission failed');
    } finally {
      setSubmitting(false);
    }
  };

  // Cancel Job
  const handleCancel = async () => {
    if (!activeJobId) return;
    setCancelling(true);
    try {
      await apiClient.cancelJob(activeJobId);
      const statusRes = await apiClient.getJobStatus(activeJobId);
      if (activeJobIdRef.current === activeJobId) {
        setActiveJobStatus(statusRes.job);
      }
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : 'Failed to cancel job');
    } finally {
      setCancelling(false);
    }
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Page Header */}
      <div className="border-b border-slate-800 pb-5">
        <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
          Benchmark Orchestration & Telemetry
        </h1>
        <p className="mt-1 text-xs text-slate-400">
          Execute isolated inference workloads, monitor real-time token events over SSE, and inspect measured distributions.
        </p>
      </div>

      {/* Main Grid: Launcher + Live Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Workload Configuration Launcher */}
        <div className="lg:col-span-5 rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <Play className="w-4 h-4 text-sky-400" />
            <h3 className="text-sm font-semibold text-slate-200">Workload Configuration</h3>
          </div>

          {formError && <ErrorMessage title="Validation Error" message={formError} />}

          <form onSubmit={handleLaunchBenchmark} className="space-y-4 text-xs">
            <div>
              <label className="block text-slate-300 font-medium mb-1">Model Identifier</label>
              <input
                type="text"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="e.g. meta-llama/Llama-3-8B"
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Engine Runtime</label>
                <select
                  value={engineType}
                  onChange={(e) => setEngineType(e.target.value as EngineType)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono"
                >
                  <option value="mock">mock (Deterministic simulated)</option>
                  <option value="llamacpp">llamacpp (llama.cpp server)</option>
                  <option value="ollama">ollama (Ollama HTTP API)</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Quantization</label>
                <select
                  value={quantization}
                  onChange={(e) => setQuantization(e.target.value as QuantizationType)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono"
                >
                  <option value="none">none (Full FP16/FP32)</option>
                  <option value="fp16">fp16</option>
                  <option value="int8">int8</option>
                  <option value="int4">int4</option>
                  <option value="gguf_q4_k_m">gguf_q4_k_m</option>
                  <option value="gguf_q8_0">gguf_q8_0</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Params (B)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  value={paramBillion}
                  onChange={(e) => setParamBillion(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Prompt Tok</label>
                <input
                  type="number"
                  min="1"
                  value={promptTokens}
                  onChange={(e) => setPromptTokens(parseInt(e.target.value) || 1)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Output Tok</label>
                <input
                  type="number"
                  min="0"
                  value={outputTokens}
                  onChange={(e) => setOutputTokens(parseInt(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Concurrency</label>
                <input
                  type="number"
                  min="1"
                  value={concurrency}
                  onChange={(e) => setConcurrency(parseInt(e.target.value) || 1)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Warmup Requests</label>
                <input
                  type="number"
                  min="0"
                  value={warmupRequests}
                  onChange={(e) => setWarmupRequests(parseInt(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono"
                />
              </div>
            </div>

            {engineType !== 'mock' && (
              <div>
                <label className="block text-slate-300 font-medium mb-1">
                  Engine Endpoint URL (optional override)
                </label>
                <input
                  type="text"
                  value={endpointUrl}
                  onChange={(e) => setEndpointUrl(e.target.value)}
                  placeholder={engineType === 'ollama' ? 'http://localhost:11434' : 'http://localhost:8080'}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 focus:outline-none focus:border-sky-500 font-mono"
                />
              </div>
            )}

            <div className="flex items-center gap-2 pt-1">
              <input
                type="checkbox"
                id="sampleMem"
                checked={sampleMemory}
                onChange={(e) => setSampleMemory(e.target.checked)}
                className="rounded bg-slate-950 border-slate-800 text-sky-600 focus:ring-0"
              />
              <label htmlFor="sampleMem" className="text-slate-400 select-none cursor-pointer">
                Sample process memory & VRAM watermark telemetry
              </label>
            </div>

            <button
              type="submit"
              disabled={submitting || (activeJobStatus !== null && (activeJobStatus.status === 'RUNNING' || activeJobStatus.status === 'PENDING'))}
              className="w-full mt-2 py-2.5 px-4 bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-semibold rounded-lg shadow-sm transition-colors flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4 fill-current" />
              {submitting ? 'Submitting...' : 'Run Benchmark'}
            </button>
          </form>
        </div>

        {/* Live SSE Telemetry Stream */}
        <div className="lg:col-span-7 rounded-xl border border-slate-800 bg-slate-900/60 p-5 flex flex-col h-[520px]">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-semibold text-slate-200">Live Telemetry & Token Stream</h3>
            </div>

            {activeJobStatus && (
              <div className="flex items-center gap-3">
                <span
                  className={`px-2.5 py-0.5 rounded text-xs font-mono font-semibold uppercase ${
                    activeJobStatus.status === 'COMPLETED'
                      ? 'bg-emerald-950 text-emerald-300 border border-emerald-700/60'
                      : activeJobStatus.status === 'RUNNING'
                      ? 'bg-sky-950 text-sky-300 border border-sky-700/60 animate-pulse'
                      : activeJobStatus.status === 'CANCELLED'
                      ? 'bg-zinc-800 text-zinc-300 border border-zinc-700'
                      : activeJobStatus.status === 'FAILED'
                      ? 'bg-red-950 text-red-300 border border-red-700/60'
                      : 'bg-amber-950 text-amber-300 border border-amber-700/60'
                  }`}
                >
                  {activeJobStatus.status}
                </span>

                {(activeJobStatus.status === 'RUNNING' || activeJobStatus.status === 'PENDING') && (
                  <button
                    onClick={handleCancel}
                    disabled={cancelling}
                    className="inline-flex items-center gap-1 px-2.5 py-1 bg-red-950/80 hover:bg-red-900 border border-red-700 text-red-200 text-xs font-medium rounded transition-colors"
                  >
                    <StopCircle className="w-3.5 h-3.5 text-red-400" />
                    {cancelling ? 'Cancelling...' : 'Cancel'}
                  </button>
                )}
              </div>
            )}
          </div>

          {!activeJobId ? (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-500 text-xs text-center p-6 space-y-2">
              <Terminal className="w-8 h-8 text-slate-600" />
              <p>No active benchmark stream.</p>
              <p className="text-slate-600 max-w-xs">
                Launch a workload to observe live Server-Sent Events (SSE), inter-token arrival deltas, and lifecycle events.
              </p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col overflow-hidden pt-3 space-y-3">
              {/* Job Metadata Bar */}
              <div className="grid grid-cols-3 gap-2 text-[11px] p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 font-mono">
                <div>
                  <span className="text-slate-500 block text-[10px]">JOB ID</span>
                  <span className="text-slate-300 truncate block">{activeJobId}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">EVENTS STREAMED</span>
                  <span className="text-emerald-400 font-semibold">{totalStreamEventsCount}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">SSE CONNECTION</span>
                  <span className={streamConnected ? 'text-emerald-400' : 'text-slate-500'}>
                    {streamConnected ? 'ACTIVE' : 'CLOSED'}
                  </span>
                </div>
              </div>

              {activeJobStatus?.status === 'FAILED' && (
                <div className="p-3 rounded-lg bg-red-950/60 border border-red-800 text-xs text-red-200 space-y-1">
                  <div className="font-semibold text-red-300">Benchmark Execution Failed</div>
                  <div className="font-mono text-[11px] text-red-400">
                    {activeJobStatus.error_message || 'An unknown error occurred during execution.'}
                  </div>
                </div>
              )}

              {activeJobResult && (
                <div className="flex items-center justify-between p-2 rounded bg-emerald-950/40 border border-emerald-800/60 text-xs">
                  <span className="text-emerald-300">
                    Benchmark completed: {activeJobResult.total_requests} requests processed.
                  </span>
                  <button
                    onClick={() => setSelectedSuite(activeJobResult)}
                    className="px-2 py-1 rounded bg-emerald-800 hover:bg-emerald-700 text-white font-medium text-[11px] inline-flex items-center gap-1 transition-colors"
                  >
                    <Eye className="w-3 h-3" /> Inspect Results
                  </button>
                </div>
              )}

              {/* Event Log Window */}
              <div className="flex-1 overflow-y-auto bg-slate-950 rounded-lg p-3 border border-slate-800 font-mono text-xs space-y-1.5">
                {streamEvents.length === 0 ? (
                  <div className="text-slate-600 italic py-4 text-center">
                    Awaiting first stream event from inference worker...
                  </div>
                ) : (
                  streamEvents.map((ev, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2 p-1 rounded hover:bg-slate-900/60 text-[11px]"
                    >
                      <span className="text-slate-500 select-none w-8 text-right flex-shrink-0">
                        #{ev.sequence_index}
                      </span>
                      <span
                        className={`px-1 rounded text-[10px] font-semibold flex-shrink-0 uppercase ${
                          ev.event_type === 'token'
                            ? 'bg-sky-950 text-sky-400 border border-sky-800'
                            : ev.event_type === 'chunk'
                            ? 'bg-amber-950 text-amber-400 border border-amber-800'
                            : ev.event_type === 'start'
                            ? 'bg-purple-950 text-purple-400 border border-purple-800'
                            : ev.event_type === 'completion'
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                            : 'bg-red-950 text-red-400 border border-red-800'
                        }`}
                      >
                        {ev.event_type}
                      </span>
                      <span className="text-slate-300 truncate">
                        {ev.token_text !== undefined && ev.token_text !== null ? (
                          <span className="text-slate-100 bg-slate-900 px-1 rounded">
                            {JSON.stringify(ev.token_text)}
                          </span>
                        ) : ev.termination_reason ? (
                          `Reason: ${ev.termination_reason}`
                        ) : ev.error_message ? (
                          ev.error_message
                        ) : (
                          'Request initiation event'
                        )}
                      </span>
                      {ev.delta_ns !== undefined && ev.delta_ns !== null && (
                        <span className="text-slate-500 ml-auto flex-shrink-0">
                          +{(ev.delta_ns / 1_000_000).toFixed(1)} ms
                        </span>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Benchmark History Section */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-400" />
            <h3 className="text-sm font-semibold text-slate-200">Historical Benchmark Suites</h3>
          </div>
          <button
            onClick={fetchHistory}
            className="text-xs text-sky-400 hover:text-sky-300 flex items-center gap-1"
          >
            <Clock className="w-3.5 h-3.5" /> Refresh List
          </button>
        </div>

        {loadingHistory ? (
          <LoadingSpinner message="Fetching historical benchmark suites..." />
        ) : historyError ? (
          <ErrorMessage title="Failed to load history" message={historyError} onRetry={fetchHistory} />
        ) : history.length === 0 ? (
          <div className="py-8 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg">
            No persisted benchmark suites found.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950/80 text-slate-400 font-semibold border-b border-slate-800 uppercase tracking-wider text-[11px]">
                <tr>
                  <th className="py-2.5 px-3">Suite ID</th>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">Engine</th>
                  <th className="py-2.5 px-3">Provenance / Mode</th>
                  <th className="py-2.5 px-3 text-right">TTFT / TTFC p50</th>
                  <th className="py-2.5 px-3 text-right">TPOT Mean</th>
                  <th className="py-2.5 px-3 text-right">Throughput</th>
                  <th className="py-2.5 px-3 text-right">Requests</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {history.map((s) => (
                  <tr key={s.suite_id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-3 font-semibold text-slate-200">{s.suite_id}</td>
                    <td className="py-2.5 px-3 text-slate-200">{s.model_id}</td>
                    <td className="py-2.5 px-3 uppercase text-slate-400">{s.engine_type}</td>
                    <td className="py-2.5 px-3">
                      <EvidenceBadge executionMode={s.execution_mode} provenance={s.provenance} />
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-200">
                      {s.ttft_p50_ms !== undefined && s.ttft_p50_ms !== null
                        ? `${s.ttft_p50_ms.toFixed(1)} ms`
                        : s.ttfc_p50_ms !== undefined && s.ttfc_p50_ms !== null
                        ? `${s.ttfc_p50_ms.toFixed(1)} ms (chunk)`
                        : 'N/A'}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-200">
                      {s.tpot_mean_ms ? `${s.tpot_mean_ms.toFixed(1)} ms` : 'N/A'}
                    </td>
                    <td className="py-2.5 px-3 text-right text-emerald-400 font-semibold">
                      {s.aggregate_output_tokens_per_second
                        ? `${s.aggregate_output_tokens_per_second.toFixed(1)} tok/s`
                        : 'N/A'}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-400">
                      {s.successful_requests}/{s.total_requests}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <button
                        onClick={() => setSelectedSuite(s)}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-400 text-xs font-sans transition-colors inline-flex items-center gap-1"
                      >
                        <Eye className="w-3 h-3" /> Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Suite Detail Modal / Panel */}
      {selectedSuite && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-4xl w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="p-5 border-b border-slate-800 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-bold text-slate-100 font-mono">
                    Suite: {selectedSuite.suite_id}
                  </h2>
                  <EvidenceBadge
                    executionMode={selectedSuite.execution_mode}
                    provenance={selectedSuite.provenance}
                  />
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  Model: {selectedSuite.model_id} • Engine: {selectedSuite.engine_type} • Date:{' '}
                  {new Date(selectedSuite.timestamp_utc).toLocaleString()}
                </p>
              </div>
              <button
                onClick={() => setSelectedSuite(null)}
                className="text-slate-400 hover:text-slate-200 text-sm font-semibold p-1"
              >
                ✕ Close
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto space-y-6 text-xs">
              {/* Aggregate KPI Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block text-[11px]">First Arrival (TTFT / TTFC)</span>
                  <span className="font-mono text-slate-200 font-bold text-sm block mt-1">
                    {selectedSuite.ttft_p50_ms !== undefined && selectedSuite.ttft_p50_ms !== null
                      ? `${selectedSuite.ttft_p50_ms.toFixed(1)} ms (token)`
                      : selectedSuite.ttfc_p50_ms !== undefined && selectedSuite.ttfc_p50_ms !== null
                      ? `${selectedSuite.ttfc_p50_ms.toFixed(1)} ms (chunk)`
                      : 'Unavailable'}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">
                    {selectedSuite.ttft_p50_ms ? 'True token observation' : selectedSuite.ttfc_p50_ms ? 'Text chunk stream' : 'Prefill-only'}
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block text-[11px]">Inter-Arrival (ITL / Chunk)</span>
                  <span className="font-mono text-slate-200 font-bold text-sm block mt-1">
                    {selectedSuite.itl_p50_ms !== undefined && selectedSuite.itl_p50_ms !== null
                      ? `${selectedSuite.itl_p50_ms.toFixed(1)} ms (ITL)`
                      : selectedSuite.chunk_latency_p50_ms !== undefined && selectedSuite.chunk_latency_p50_ms !== null
                      ? `${selectedSuite.chunk_latency_p50_ms.toFixed(1)} ms (chunk)`
                      : 'Unavailable'}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">
                    {selectedSuite.itl_p50_ms ? 'Token decode delta' : selectedSuite.chunk_latency_p50_ms ? 'Stream chunk delta' : 'No decode deltas'}
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block text-[11px]">Output Throughput</span>
                  <span className="font-mono text-emerald-400 font-bold text-sm block mt-1">
                    {selectedSuite.aggregate_output_tokens_per_second
                      ? `${selectedSuite.aggregate_output_tokens_per_second.toFixed(1)} tok/s`
                      : 'N/A'}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">
                    Wall time: {selectedSuite.wall_time_seconds.toFixed(2)}s
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block text-[11px]">Memory Telemetry</span>
                  <span className="font-mono text-purple-400 font-bold text-sm block mt-1">
                    {selectedSuite.memory_watermark?.peak_gpu_vram_mb
                      ? `VRAM: ${selectedSuite.memory_watermark.peak_gpu_vram_mb.toFixed(0)} MB`
                      : selectedSuite.memory_watermark?.peak_process_rss_mb
                      ? `RSS: ${selectedSuite.memory_watermark.peak_process_rss_mb.toFixed(0)} MB`
                      : 'Unavailable'}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-0.5 truncate" title={selectedSuite.memory_watermark?.notes || 'No notes'}>
                    {selectedSuite.memory_watermark?.telemetry_source || 'Unmetered'}
                  </span>
                </div>
              </div>

              {/* Memory Caveats if Present */}
              {selectedSuite.memory_watermark?.notes && (
                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-[11px] text-slate-400">
                  <span className="text-slate-300 font-semibold">Telemetry Note: </span>
                  {selectedSuite.memory_watermark.notes}
                </div>
              )}

              {/* Request-Level Results */}
              <div>
                <h4 className="font-semibold text-slate-300 mb-2">Request-Level Measurements</h4>
                <div className="border border-slate-800 rounded-lg overflow-x-auto max-h-60">
                  <table className="w-full text-left text-[11px] font-mono">
                    <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="p-2">Request ID</th>
                        <th className="p-2">Tokens (In/Out)</th>
                        <th className="p-2">Source</th>
                        <th className="p-2 text-right">TTFT (ms)</th>
                        <th className="p-2 text-right">TTFC (ms)</th>
                        <th className="p-2 text-right">TPOT (ms)</th>
                        <th className="p-2 text-right">Throughput (tok/s)</th>
                        <th className="p-2 text-right">Duration (ms)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {selectedSuite.request_results.map((req) => (
                        <tr key={req.request_id} className="hover:bg-slate-800/40">
                          <td className="p-2 text-slate-200">{req.request_id}</td>
                          <td className="p-2 text-slate-400">
                            {req.prompt_tokens !== null && req.prompt_tokens !== undefined ? req.prompt_tokens : '—'} / {req.output_tokens}
                          </td>
                          <td className="p-2 text-[10px] text-slate-400 uppercase">
                            {req.token_count_source}
                          </td>
                          <td className="p-2 text-right text-slate-200">
                            {req.ttft_ms !== undefined && req.ttft_ms !== null ? req.ttft_ms.toFixed(1) : '-'}
                          </td>
                          <td className="p-2 text-right text-amber-400">
                            {req.ttfc_ms !== undefined && req.ttfc_ms !== null ? req.ttfc_ms.toFixed(1) : '-'}
                          </td>
                          <td className="p-2 text-right text-slate-200">
                            {req.tpot_ms !== undefined && req.tpot_ms !== null ? req.tpot_ms.toFixed(1) : '-'}
                          </td>
                          <td className="p-2 text-right text-emerald-400">
                            {req.output_tokens_per_second ? req.output_tokens_per_second.toFixed(1) : '-'}
                          </td>
                          <td className="p-2 text-right text-slate-400">{req.duration_ms.toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
