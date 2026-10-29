import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Zap,
  CheckCircle2,
  AlertTriangle,
  Play,
  Sliders,
  Clock,
  Shield,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import {
  RuntimeRecommendationRequest,
  RuntimeRecommendationResult,
  RuntimeScenario,
  RuntimeValidationResponse,
  RuntimeOptimizationObjective,
} from '../../api/types';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorMessage } from '../common/ErrorMessage';
import { MetricCard } from '../common/MetricCard';

export const RuntimeOptimizationView: React.FC = () => {
  // Input parameters
  const [promptTokens, setPromptTokens] = useState<number>(32);
  const [outputTokens, setOutputTokens] = useState<number>(16);
  const [concurrency, setConcurrency] = useState<number>(1);
  const [latencySlaMs, setLatencySlaMs] = useState<string>('1200');
  const [throughputSlaTokS, setThroughputSlaTokS] = useState<string>('');
  const [objective, setObjective] = useState<RuntimeOptimizationObjective>('MIN_LATENCY');
  const [modelFilter, setModelFilter] = useState<string>('');

  // Scenarios & State
  const [scenarios, setScenarios] = useState<RuntimeScenario[]>([]);
  const [activeScenarioId, setActiveScenarioId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RuntimeRecommendationResult | null>(null);

  // Live Validation State
  const [validating, setValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<RuntimeValidationResponse | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchScenarios = async () => {
      try {
        const data = await apiClient.listRuntimeScenarios();
        if (isMounted) {
          setScenarios(data);
        }
      } catch {
        // Fallback
      }
    };
    fetchScenarios();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleApplyScenario = (scenario: RuntimeScenario) => {
    setActiveScenarioId(scenario.scenario_id);
    const req = scenario.request;
    setPromptTokens(req.prompt_tokens);
    setOutputTokens(req.expected_output_tokens);
    setConcurrency(req.concurrency || 1);
    setLatencySlaMs(req.latency_sla_ms ? String(req.latency_sla_ms) : '');
    setThroughputSlaTokS(req.throughput_sla_tok_s ? String(req.throughput_sla_tok_s) : '');
    setObjective(req.objective || 'MIN_LATENCY');
    setModelFilter(req.fixed_model_id || '');
    setValidationResult(null);
    setValidationError(null);
  };

  const handleGetRecommendation = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    setValidationResult(null);
    setValidationError(null);

    try {
      const req: RuntimeRecommendationRequest = {
        prompt_tokens: promptTokens,
        expected_output_tokens: outputTokens,
        concurrency: concurrency,
        latency_sla_ms: latencySlaMs ? parseFloat(latencySlaMs) : null,
        throughput_sla_tok_s: throughputSlaTokS ? parseFloat(throughputSlaTokS) : null,
        objective: objective,
        allow_model_selection: !modelFilter,
        fixed_model_id: modelFilter || null,
      };

      const res = await apiClient.recommendRuntime(req);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to obtain runtime recommendation');
    } finally {
      setLoading(false);
    }
  };

  const handleRunValidation = async () => {
    if (!result) return;
    setValidating(true);
    setValidationError(null);

    try {
      const valRes = await apiClient.validateRuntime({
        recommendation_id: result.recommendation_id,
        model_id: result.model_id,
        num_thread: result.recommended_threads,
        num_ctx: result.recommended_num_ctx,
        prompt_tokens: promptTokens,
        expected_output_tokens: outputTokens,
        expected_latency_ms: result.predicted_client_e2e_ms,
        expected_decode_throughput_tok_s: result.predicted_decode_tokens_per_second,
      });
      setValidationResult(valRes);
    } catch (err: unknown) {
      setValidationError(err instanceof Error ? err.message : 'Live validation benchmark failed');
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Workstation Hardware Calibration Banner */}
      <div className="bg-slate-900 border border-slate-700/80 rounded-xl p-5 shadow-lg">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-blue-500/10 border border-blue-500/20 rounded-lg text-blue-400">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-slate-100">
                Hardware-Aware Runtime Calibration
              </h3>
              <p className="text-xs text-slate-400">
                Intel(R) Core(TM) i7-14700 &bull; 20 Cores (8P + 12E, 28 Threads) &bull; Windows 11 &bull; Ollama v0.40.11
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Zap className="w-3.5 h-3.5 mr-1" />
              8 P-Core Saturation Optimal
            </span>
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
              <Clock className="w-3.5 h-3.5 mr-1" />
              Single-Slot Server Queue
            </span>
          </div>
        </div>
        <p className="mt-3 text-xs text-slate-300 bg-slate-800/60 p-3 rounded-lg border border-slate-700/50">
          <strong>Physical Architecture Insight:</strong> On hybrid Raptor Lake Refresh architectures, 
          allocating <strong>8 threads</strong> targets the high-clocked Performance Cores without triggering cross-cluster 
          synchronization barriers. Exceeding 8 threads engages E-cores, accelerating parallel prefill but introducing 
          tail latency in autoregressive token decoding.
        </p>
      </div>

      {/* Operational Scenarios */}
      {scenarios.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            Preset Validation Scenarios (Session 24)
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {scenarios.map((sc) => (
              <button
                key={sc.scenario_id}
                type="button"
                onClick={() => handleApplyScenario(sc)}
                className={`p-3.5 rounded-lg border text-left transition-all ${
                  activeScenarioId === sc.scenario_id
                    ? 'border-blue-500/80 bg-blue-500/10 ring-1 ring-blue-500/30'
                    : 'border-slate-800 bg-slate-800/40 hover:bg-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="text-xs font-medium text-slate-200">{sc.name}</div>
                <div className="text-[11px] text-slate-400 mt-1 line-clamp-2">{sc.description}</div>
                <div className="mt-2 text-[10px] text-blue-400 font-mono">
                  Prompt: {sc.request.prompt_tokens}t | Output: {sc.request.expected_output_tokens}t | SLA: {sc.request.latency_sla_ms || 'N/A'}ms
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Configuration Form */}
      <form onSubmit={handleGetRecommendation} className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Prompt Tokens
            </label>
            <input
              type="number"
              min={1}
              value={promptTokens}
              onChange={(e) => setPromptTokens(parseInt(e.target.value) || 1)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100"
              required
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Expected Output Tokens
            </label>
            <input
              type="number"
              min={1}
              value={outputTokens}
              onChange={(e) => setOutputTokens(parseInt(e.target.value) || 1)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100"
              required
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Client Concurrency
            </label>
            <input
              type="number"
              min={1}
              max={16}
              value={concurrency}
              onChange={(e) => setConcurrency(parseInt(e.target.value) || 1)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Optimization Objective
            </label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value as RuntimeOptimizationObjective)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100"
            >
              <option value="MIN_LATENCY">Minimum Latency (Interactive)</option>
              <option value="MAX_THROUGHPUT">Maximum Decode Throughput</option>
              <option value="MIN_MEMORY">Minimum Memory Footprint</option>
              <option value="CONSERVATIVE_SLA">Conservative SLA Compliance</option>
              <option value="QUALITY_CONSTRAINED_LATENCY">Quality-Constrained Latency</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Latency SLA (ms)
            </label>
            <input
              type="number"
              step="any"
              placeholder="e.g. 1500"
              value={latencySlaMs}
              onChange={(e) => setLatencySlaMs(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Throughput SLA (tokens/s)
            </label>
            <input
              type="number"
              step="any"
              placeholder="e.g. 35"
              value={throughputSlaTokS}
              onChange={(e) => setThroughputSlaTokS(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Pinned Model (Optional)
            </label>
            <select
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100"
            >
              <option value="">Auto-Select Best Model</option>
              <option value="qwen2.5:0.5b">qwen2.5:0.5b (Q4_K_M)</option>
              <option value="llama3.2:1b">llama3.2:1b (Q8_0)</option>
              <option value="qwen2.5:1.5b">qwen2.5:1.5b (Q4_K_M)</option>
            </select>
          </div>
          <div className="flex items-end">
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800 text-white font-medium text-xs py-2 px-4 rounded-lg flex items-center justify-center space-x-2 transition-all shadow-md shadow-blue-900/20"
            >
              {loading ? (
                <>
                  <LoadingSpinner size="sm" />
                  <span>Optimizing...</span>
                </>
              ) : (
                <>
                  <Sliders className="w-4 h-4 mr-1.5" />
                  <span>Recommend Configuration</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {error && <ErrorMessage message={error} />}

      {/* Recommendation Results Card */}
      {result && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex items-start justify-between flex-wrap gap-4 border-b border-slate-800 pb-5">
            <div>
              <div className="flex items-center space-x-3">
                <span className="text-xl font-bold text-slate-100 font-mono">
                  {result.model_id}
                </span>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono">
                  {result.recommended_threads} CPU Threads
                </span>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700 font-mono">
                  num_ctx: {result.recommended_num_ctx}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1 font-mono">
                Digest: {result.model_digest.slice(0, 16)}... &bull; Quant: {result.quantization} &bull; Tier: {result.evidence_tier}
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <span
                className={`px-3 py-1 rounded-full text-xs font-semibold flex items-center space-x-1.5 ${
                  result.conservative_feasibility === 'ROBUST_FEASIBLE'
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : result.conservative_feasibility === 'NOMINAL_FEASIBLE_ROBUST_RISK'
                    ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                    : result.conservative_feasibility === 'UNKNOWN_FEASIBILITY'
                    ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                    : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                }`}
              >
                {result.conservative_feasibility === 'ROBUST_FEASIBLE' && <CheckCircle2 className="w-3.5 h-3.5" />}
                {result.conservative_feasibility === 'NOMINAL_FEASIBLE_ROBUST_RISK' && <AlertTriangle className="w-3.5 h-3.5" />}
                {result.conservative_feasibility === 'UNKNOWN_FEASIBILITY' && <Shield className="w-3.5 h-3.5" />}
                <span>{result.conservative_feasibility}</span>
              </span>
            </div>
          </div>

          {/* Metric Cards Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard
              title="Predicted Latency"
              value={`${result.predicted_client_e2e_ms.toFixed(1)} ms`}
              subtitle={`Prefill: ${result.predicted_prefill_ms.toFixed(0)}ms | Decode: ${result.predicted_decode_ms.toFixed(0)}ms`}
            />
            <MetricCard
              title="Decode Throughput"
              value={`${result.predicted_decode_tokens_per_second.toFixed(1)} tok/s`}
              subtitle="Autoregressive generation rate"
            />
            <MetricCard
              title="Memory Footprint"
              value={`${result.estimated_memory_mb.toFixed(0)} MB`}
              subtitle="Weights + KV slot cache"
            />
            <MetricCard
              title="Pareto Efficiency"
              value={`Rank ${result.pareto_rank}`}
              subtitle="Non-dominated configuration"
            />
          </div>

          {/* Explanation Callout */}
          <div className="bg-slate-800/40 rounded-lg p-4 border border-slate-700/60 text-xs text-slate-300 space-y-2">
            <div className="font-semibold text-slate-200">Recommendation Rationale:</div>
            <p>{result.explanation}</p>
            <div className="text-[11px] text-slate-400 italic pt-1 border-t border-slate-700/40">
              {result.workstation_provenance_caveat}
            </div>
          </div>

          {/* Alternatives Table */}
          {result.tradeoff_alternatives.length > 0 && (
            <div>
              <h5 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                Alternative Viable Configurations
              </h5>
              <div className="overflow-x-auto rounded-lg border border-slate-800">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-800/70 text-slate-400 font-medium">
                    <tr>
                      <th className="py-2 px-3">Model</th>
                      <th className="py-2 px-3">Threads</th>
                      <th className="py-2 px-3">Predicted E2E</th>
                      <th className="py-2 px-3">Decode tok/s</th>
                      <th className="py-2 px-3">Memory</th>
                      <th className="py-2 px-3">Feasibility</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 text-slate-300 font-mono text-[11px]">
                    {result.tradeoff_alternatives.map((alt, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/30">
                        <td className="py-2 px-3 font-semibold">{alt.model_id}</td>
                        <td className="py-2 px-3">{alt.threads} threads</td>
                        <td className="py-2 px-3">{alt.e2e_ms} ms</td>
                        <td className="py-2 px-3">{alt.decode_tps} tok/s</td>
                        <td className="py-2 px-3">{alt.memory_mb} MB</td>
                        <td className="py-2 px-3">{alt.feasibility}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Live Validation Button & Results */}
          <div className="border-t border-slate-800 pt-5">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <h5 className="text-xs font-semibold text-slate-200">
                  Prospective Live Validation
                </h5>
                <p className="text-[11px] text-slate-400">
                  Execute benchmark on Ollama with {result.model_id} ({result.recommended_threads} threads, num_ctx={result.recommended_num_ctx})
                </p>
              </div>
              <button
                type="button"
                onClick={handleRunValidation}
                disabled={validating}
                className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-emerald-800 text-white font-medium text-xs py-2 px-4 rounded-lg flex items-center space-x-1.5 transition-all shadow-md shadow-emerald-900/20"
              >
                {validating ? (
                  <>
                    <LoadingSpinner size="sm" />
                    <span>Executing Benchmark...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-1.5" />
                    <span>Validate on Workstation</span>
                  </>
                )}
              </button>
            </div>

            {validationError && (
              <div className="mt-4">
                <ErrorMessage message={validationError} />
              </div>
            )}

            {validationResult && (
              <div className="mt-4 bg-slate-800/60 border border-slate-700/80 rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between text-xs font-medium">
                  <span className="text-emerald-400 flex items-center">
                    <CheckCircle2 className="w-4 h-4 mr-1.5" />
                    Validation Benchmark Completed
                  </span>
                  <span className="text-slate-400 font-mono text-[11px]">
                    Executed at {validationResult.executed_at.slice(11, 19)} UTC
                  </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="bg-slate-900/60 p-2.5 rounded border border-slate-800">
                    <div className="text-[10px] text-slate-400">Observed E2E</div>
                    <div className="text-sm font-bold text-slate-200 font-mono">
                      {validationResult.observed_client_e2e_duration_ms.toFixed(1)} ms
                    </div>
                    {validationResult.duration_relative_error_pct !== null && (
                      <div className="text-[10px] text-slate-400">
                        Error: {validationResult.duration_relative_error_pct}% ({validationResult.duration_absolute_error_ms}ms)
                      </div>
                    )}
                  </div>
                  <div className="bg-slate-900/60 p-2.5 rounded border border-slate-800">
                    <div className="text-[10px] text-slate-400">Observed Decode Rate</div>
                    <div className="text-sm font-bold text-slate-200 font-mono">
                      {validationResult.observed_server_decode_throughput_tok_s?.toFixed(1) || 'N/A'} tok/s
                    </div>
                    {validationResult.throughput_relative_error_pct !== null && (
                      <div className="text-[10px] text-slate-400">
                        Error: {validationResult.throughput_relative_error_pct}%
                      </div>
                    )}
                  </div>
                  <div className="bg-slate-900/60 p-2.5 rounded border border-slate-800">
                    <div className="text-[10px] text-slate-400">Observed Prefill</div>
                    <div className="text-sm font-bold text-slate-200 font-mono">
                      {validationResult.observed_server_prefill_duration_ms?.toFixed(1) || 'N/A'} ms
                    </div>
                  </div>
                  <div className="bg-slate-900/60 p-2.5 rounded border border-slate-800">
                    <div className="text-[10px] text-slate-400">Observed Decode</div>
                    <div className="text-sm font-bold text-slate-200 font-mono">
                      {validationResult.observed_server_decode_duration_ms?.toFixed(1) || 'N/A'} ms
                    </div>
                  </div>
                </div>
                <p className="text-xs text-slate-300 pt-1">
                  {validationResult.explanation}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
