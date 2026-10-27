import React, { useState, useEffect } from 'react';
import {
  Compass,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  XCircle,
  Play,
  Activity,
  Layers,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import { apiClient } from '../api/client';
import {
  WorkloadRecommendationRequest,
  WorkloadRecommendationResponse,
  ValidationScenario,
  ValidationResponse,
  OptimizationObjective,
} from '../api/types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorMessage } from '../components/common/ErrorMessage';
import { MetricCard } from '../components/common/MetricCard';
import { MultiModelComparisonView } from '../components/recommendations/MultiModelComparisonView';

export const RecommendationsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'single' | 'multi'>('single');

  // Workload Form State
  const [modelId, setModelId] = useState('llama3.2:1b');
  const [promptTokens, setPromptTokens] = useState<number>(32);
  const [outputTokens, setOutputTokens] = useState<number>(16);
  const [concurrency, setConcurrency] = useState<number>(1);
  const [modelResidency, setModelResidency] = useState<'unknown' | 'warm' | 'cold'>('warm');
  const [prefixCache, setPrefixCache] = useState<'unknown' | 'cold_prefix' | 'warm_prefix' | 'partial_prefix'>('cold_prefix');
  const [latencySlaMs, setLatencySlaMs] = useState<string>('2500');
  const [decodeThroughputSlaTokS, setDecodeThroughputSlaTokS] = useState<string>('');
  const [ttftSlaMs, setTtftSlaMs] = useState<string>('');
  const [requireColdStart, setRequireColdStart] = useState<boolean>(false);
  const [objective, setObjective] = useState<OptimizationObjective>('minimize_duration');

  // Scenarios and Recommendation Result State
  const [scenarios, setScenarios] = useState<ValidationScenario[]>([]);
  const [activeScenarioId, setActiveScenarioId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendation, setRecommendation] = useState<WorkloadRecommendationResponse | null>(null);

  // Live Validation State
  const [validating, setValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<ValidationResponse | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Fetch standard validation scenarios on mount
  useEffect(() => {
    let isMounted = true;
    const loadScenarios = async () => {
      try {
        const data = await apiClient.listRecommendationScenarios();
        if (isMounted) {
          setScenarios(data);
        }
      } catch {
        // Non-fatal fallback
      }
    };
    loadScenarios();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleApplyScenario = (scenario: ValidationScenario) => {
    setActiveScenarioId(scenario.scenario_id);
    const req = scenario.request;
    setModelId(req.model_id || 'llama3.2:1b');
    setPromptTokens(req.prompt_tokens);
    setOutputTokens(req.expected_output_tokens);
    setConcurrency(req.concurrency || 1);
    setModelResidency(req.expected_model_residency || 'unknown');
    setPrefixCache(req.expected_prefix_cache || 'unknown');
    setLatencySlaMs(req.latency_sla_ms !== undefined && req.latency_sla_ms !== null ? String(req.latency_sla_ms) : '');
    setDecodeThroughputSlaTokS(
      req.decode_throughput_sla_tok_s !== undefined && req.decode_throughput_sla_tok_s !== null
        ? String(req.decode_throughput_sla_tok_s)
        : ''
    );
    setTtftSlaMs(req.ttft_sla_ms !== undefined && req.ttft_sla_ms !== null ? String(req.ttft_sla_ms) : '');
    setRequireColdStart(!!req.require_cold_start);
    setObjective(req.objective || 'minimize_duration');
    setValidationResult(null);
    setValidationError(null);
  };

  const handleGetRecommendation = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    setValidationResult(null);
    setValidationError(null);

    const reqPayload: WorkloadRecommendationRequest = {
      model_id: modelId,
      prompt_tokens: Number(promptTokens),
      expected_output_tokens: Number(outputTokens),
      concurrency: Number(concurrency),
      expected_model_residency: modelResidency,
      expected_prefix_cache: prefixCache,
      latency_sla_ms: latencySlaMs ? parseFloat(latencySlaMs) : null,
      decode_throughput_sla_tok_s: decodeThroughputSlaTokS ? parseFloat(decodeThroughputSlaTokS) : null,
      ttft_sla_ms: ttftSlaMs ? parseFloat(ttftSlaMs) : null,
      require_cold_start: requireColdStart,
      objective,
    };

    try {
      const resp = await apiClient.recommendWorkload(reqPayload);
      setRecommendation(resp);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to generate workload recommendation';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleValidateLive = async () => {
    if (!recommendation) return;
    setValidating(true);
    setValidationError(null);

    try {
      const val = await apiClient.validateRecommendation({
        recommendation_id: recommendation.recommendation_id,
        model_id: modelId,
        prompt_tokens: promptTokens,
        expected_output_tokens: outputTokens,
        expected_latency_ms: recommendation.predicted_metrics.client_e2e_duration_ms,
        expected_decode_throughput_tok_s: recommendation.predicted_metrics.server_decode_throughput_tok_s,
      });
      setValidationResult(val);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Live validation benchmark failed';
      setValidationError(msg);
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-sky-500/10 border border-sky-500/20 text-sky-400">
            <Compass className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">Workload Configuration Recommendation</h1>
            <p className="text-sm text-slate-400">
              Evidence-backed inference configuration forecasts, physical latency decompositions, and live benchmark validation.
            </p>
          </div>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex border-b border-slate-800 gap-2">
        <button
          type="button"
          onClick={() => setActiveTab('single')}
          className={`px-4 py-2 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 ${
            activeTab === 'single'
              ? 'border-sky-500 text-sky-400 bg-sky-500/10'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Compass className="w-3.5 h-3.5" />
          Single-Model Recommendation
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('multi')}
          className={`px-4 py-2 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 ${
            activeTab === 'multi'
              ? 'border-sky-500 text-sky-400 bg-sky-500/10'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          Multi-Model Comparison
        </button>
      </div>

      {activeTab === 'multi' ? (
        <MultiModelComparisonView />
      ) : (
        <>
          {/* Preset Validation Scenarios Carousel / Badges */}
          {scenarios.length > 0 && (
        <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            Standard Evaluation Scenarios
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {scenarios.map((sc) => (
              <button
                key={sc.scenario_id}
                type="button"
                onClick={() => handleApplyScenario(sc)}
                className={`text-left p-3 rounded-lg border transition-all text-xs ${
                  activeScenarioId === sc.scenario_id
                    ? 'bg-sky-950/60 border-sky-500 text-sky-200'
                    : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="font-semibold text-slate-200 mb-1">{sc.name}</div>
                <div className="text-slate-400 line-clamp-2 leading-relaxed">{sc.description}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Interactive Workload Form */}
      <form onSubmit={handleGetRecommendation} className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-6">
        <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
          <Layers className="w-4 h-4 text-sky-400" />
          Workload Specification & Constraints
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          {/* Model ID */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Model Identity</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
              placeholder="llama3.2:1b"
              required
            />
          </div>

          {/* Prompt Tokens */}
          <div>
            <label htmlFor="promptTokensInput" className="block font-medium text-slate-300 mb-1">Prompt Tokens (P)</label>
            <input
              id="promptTokensInput"
              type="number"
              min="1"
              value={promptTokens}
              onChange={(e) => setPromptTokens(parseInt(e.target.value) || 1)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
              required
            />
          </div>

          {/* Output Tokens */}
          <div>
            <label htmlFor="outputTokensInput" className="block font-medium text-slate-300 mb-1">Expected Output Tokens (O)</label>
            <input
              id="outputTokensInput"
              type="number"
              min="1"
              value={outputTokens}
              onChange={(e) => setOutputTokens(parseInt(e.target.value) || 1)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
              required
            />
          </div>

          {/* Concurrency */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Concurrency</label>
            <input
              type="number"
              min="1"
              max="64"
              value={concurrency}
              onChange={(e) => setConcurrency(parseInt(e.target.value) || 1)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
              required
            />
          </div>

          {/* Model Residency */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Expected Residency</label>
            <select
              value={modelResidency}
              onChange={(e) => setModelResidency(e.target.value as any)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option value="warm">WARM (Pre-loaded in RAM)</option>
              <option value="cold">COLD (Disk-to-RAM load)</option>
              <option value="unknown">UNKNOWN</option>
            </select>
          </div>

          {/* Prefix Cache */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Prefix Cache State</label>
            <select
              value={prefixCache}
              onChange={(e) => setPrefixCache(e.target.value as any)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option value="cold_prefix">COLD_PREFIX (Full Prefill)</option>
              <option value="warm_prefix">WARM_PREFIX (KV Cache Hit)</option>
              <option value="unknown">UNKNOWN</option>
            </select>
          </div>

          {/* Latency SLA */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Latency SLA (ms, optional)</label>
            <input
              type="number"
              step="any"
              value={latencySlaMs}
              onChange={(e) => setLatencySlaMs(e.target.value)}
              placeholder="e.g. 3000"
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          {/* Decode Throughput SLA */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Decode Throughput SLA (tok/s)</label>
            <input
              type="number"
              step="any"
              value={decodeThroughputSlaTokS}
              onChange={(e) => setDecodeThroughputSlaTokS(e.target.value)}
              placeholder="e.g. 20.0"
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          {/* TTFT SLA */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">TTFT SLA (ms, token-level)</label>
            <input
              type="number"
              step="any"
              value={ttftSlaMs}
              onChange={(e) => setTtftSlaMs(e.target.value)}
              placeholder="e.g. 150"
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          {/* Optimization Objective */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Objective</label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value as any)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option value="minimize_duration">Minimize Latency / Duration</option>
              <option value="maximize_decode_throughput">Maximize Decode Speed</option>
              <option value="maximize_throughput">Maximize End-to-End Throughput</option>
            </select>
          </div>

          {/* Cold Start Requirement Checkbox */}
          <div className="flex items-center gap-2 pt-5">
            <input
              type="checkbox"
              id="requireColdStart"
              checked={requireColdStart}
              onChange={(e) => setRequireColdStart(e.target.checked)}
              className="rounded bg-slate-950 border-slate-800 text-sky-500 focus:ring-0"
            />
            <label htmlFor="requireColdStart" className="text-slate-300 cursor-pointer">
              Require Cold Start Evidence (+1,038 ms)
            </label>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-medium text-xs shadow-lg shadow-sky-600/20 transition-all"
          >
            {loading ? <LoadingSpinner size="sm" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            Get Evidence-Backed Recommendation
          </button>
        </div>
      </form>

      {error && <ErrorMessage message={error} onRetry={() => handleGetRecommendation()} />}

      {/* Recommendation Results Display */}
      {recommendation && (
        <div className="space-y-6">
          {/* Status & Rationale Banner */}
          <div
            className={`rounded-xl border p-5 ${
              recommendation.status === 'RECOMMENDED'
                ? 'bg-emerald-950/20 border-emerald-700/40 text-emerald-100'
                : recommendation.status === 'INCONCLUSIVE'
                ? 'bg-amber-950/20 border-amber-700/40 text-amber-100'
                : 'bg-rose-950/20 border-rose-700/40 text-rose-100'
            }`}
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
              <div className="flex items-center gap-2.5">
                {recommendation.status === 'RECOMMENDED' ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                ) : recommendation.status === 'INCONCLUSIVE' ? (
                  <HelpCircle className="w-5 h-5 text-amber-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-rose-400" />
                )}
                <span className="font-bold text-sm tracking-wide uppercase">
                  Recommendation Status: {recommendation.status}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {recommendation.operating_regime && (
                  <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-indigo-950/80 border border-indigo-700/60 text-indigo-300 uppercase">
                    Regime: {recommendation.operating_regime}
                  </span>
                )}
                <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-slate-900 border border-slate-700 text-slate-300 uppercase">
                  Evidence: {recommendation.evidence_source}
                </span>
                <span className="px-2.5 py-0.5 rounded text-xs font-mono bg-slate-900 border border-slate-700 text-slate-400">
                  {recommendation.applicable_dataset}
                </span>
              </div>
            </div>

            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
              {recommendation.explanation_summary}
            </p>
          </div>

          {/* OOD Warnings Alert */}
          {recommendation.ood_warnings && recommendation.ood_warnings.length > 0 && (
            <div className="bg-amber-950/30 border border-amber-700/50 rounded-xl p-4 text-xs text-amber-200 space-y-1">
              <div className="font-semibold flex items-center gap-1.5 text-amber-300">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Extrapolation & Out-Of-Distribution Warnings
              </div>
              <ul className="list-disc list-inside space-y-0.5 pl-1 text-amber-200/90">
                {recommendation.ood_warnings.map((w, idx) => (
                  <li key={idx}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Decomposed Predicted Metric Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <MetricCard
              title="Client E2E Duration"
              value={recommendation.predicted_metrics.client_e2e_duration_ms ?? 'N/A'}
              unit="ms"
              subtitle={
                recommendation.exploratory_prediction_range
                  ? `Exploratory Range: [${recommendation.exploratory_prediction_range[0]}, ${recommendation.exploratory_prediction_range[1]}] ms (~74% empirical coverage)`
                  : recommendation.prediction_uncertainty_interval
                  ? `Exploratory Range: [${recommendation.prediction_uncertainty_interval[0]}, ${recommendation.prediction_uncertainty_interval[1]}] ms (~74% empirical coverage)`
                  : undefined
              }
            />
            <MetricCard
              title="Decode Throughput"
              value={recommendation.predicted_metrics.server_decode_throughput_tok_s ?? 'N/A'}
              unit="tok/s"
              subtitle="Autoregressive Rate"
            />
            <MetricCard
              title="Client Delivery Rate"
              value={recommendation.predicted_metrics.client_e2e_output_throughput_tok_s ?? 'N/A'}
              unit="tok/s"
              subtitle="End-to-End Speed"
            />
            <MetricCard
              title="Prompt Prefill"
              value={recommendation.predicted_metrics.server_prefill_duration_ms ?? 'N/A'}
              unit="ms"
              subtitle="Prompt Embedding"
            />
            <MetricCard
              title="Token Decode"
              value={recommendation.predicted_metrics.server_decode_duration_ms ?? 'N/A'}
              unit="ms"
              subtitle="Token Generation"
            />
            <MetricCard
              title="Model Weight Load"
              value={recommendation.predicted_metrics.server_load_duration_ms ?? 'N/A'}
              unit="ms"
              subtitle={recommendation.cold_start_assumptions}
            />
          </div>

          {/* SLA Feasibility & Trade-offs Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* SLA Assessment Table */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-sky-400" />
                SLA Feasibility Breakdown
              </h3>
              {Object.keys(recommendation.sla_assessments).length === 0 ? (
                <div className="text-xs text-slate-500 py-3">No hard SLAs requested for this workload.</div>
              ) : (
                <div className="space-y-2.5">
                  {Object.entries(recommendation.sla_assessments).map(([key, val]: [string, any]) => (
                    <div key={key} className="p-2.5 rounded bg-slate-950/80 border border-slate-800 text-xs">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-slate-200 capitalize">{key.replace('_', ' ')}</span>
                        <div className="flex items-center gap-1.5">
                          {val.conservative_feasibility && val.conservative_feasibility !== val.status && (
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                                val.conservative_feasibility === 'ROBUST_FEASIBLE'
                                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-700/60'
                                  : val.conservative_feasibility === 'NOMINAL_FEASIBLE_ROBUST_RISK'
                                  ? 'bg-amber-950 text-amber-300 border border-amber-700/60'
                                  : 'bg-zinc-800 text-zinc-300 border border-zinc-600/60'
                              }`}
                            >
                              {val.conservative_feasibility}
                            </span>
                          )}
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                              val.status === 'FEASIBLE'
                                ? 'bg-emerald-950 text-emerald-300 border border-emerald-700/60'
                                : val.status === 'PROVABLY_INFEASIBLE' || val.status === 'PROVEN_INFEASIBLE'
                                ? 'bg-rose-950 text-rose-300 border border-rose-700/60'
                                : val.status === 'PREDICTED_SLA_MISS'
                                ? 'bg-orange-950 text-orange-300 border border-orange-700/60'
                                : 'bg-zinc-800 text-zinc-300 border border-zinc-600/60'
                            }`}
                          >
                            {val.status}
                          </span>
                        </div>
                      </div>
                      <div className="text-slate-400">
                        Target: <span className="text-slate-200 font-mono">{val.target_ms ?? val.target_tok_s}</span>
                        {val.forecast_ms !== undefined && (
                          <>
                            {' '}• Forecast: <span className="text-slate-200 font-mono">{val.forecast_ms} ms</span>
                          </>
                        )}
                        {val.worst_case_forecast_ms !== undefined && (
                          <>
                            {' '}• Worst-Case: <span className="text-amber-300 font-mono">{val.worst_case_forecast_ms} ms</span>
                          </>
                        )}
                        {val.historical_sla_violation_frequency_pct !== undefined && (
                          <>
                            {' '}• Historical Miss Rate: <span className="text-amber-300 font-mono">{val.historical_sla_violation_frequency_pct}%</span>
                          </>
                        )}
                        {val.headroom_ms !== undefined && (
                          <span className="text-emerald-400 ml-2">({val.headroom_ms} ms margin)</span>
                        )}
                        {val.violation_ms !== undefined && (
                          <span className="text-rose-400 ml-2">(exceeds by {val.violation_ms} ms)</span>
                        )}
                        {val.estimated_violation_ms !== undefined && (
                          <span className="text-amber-400 ml-2">(est. exceeds by {val.estimated_violation_ms} ms — unproven)</span>
                        )}
                        {val.deficit_tok_s !== undefined && (
                          <span className="text-rose-400 ml-2">(deficit: {val.deficit_tok_s} tok/s)</span>
                        )}
                        {val.estimated_deficit_tok_s !== undefined && (
                          <span className="text-amber-400 ml-2">(est. deficit: {val.estimated_deficit_tok_s} tok/s — unproven)</span>
                        )}
                      </div>
                      {val.risk_warning && (
                        <div className="mt-1 text-amber-400/90 text-[11px]">{val.risk_warning}</div>
                      )}
                      {val.qualification_note && (
                        <div className="mt-1 text-amber-300/80 text-[11px]">{val.qualification_note}</div>
                      )}
                      {val.physical_proof && (
                        <div className="mt-1 text-rose-300/80 text-[11px] font-mono">{val.physical_proof}</div>
                      )}
                      {val.limitation_note && (
                        <div className="mt-1 text-slate-400 italic text-[11px]">{val.limitation_note}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Alternatives & Influencing Factors */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4 text-sky-400" />
                Physical Factors & Assumptions
              </h3>
              <ul className="space-y-1.5 text-xs text-slate-300 list-disc list-inside mb-4">
                {recommendation.influencing_factors.map((f, idx) => (
                  <li key={idx}>{f}</li>
                ))}
              </ul>

              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">
                Evaluated Alternatives & Trade-offs
              </h4>
              <div className="space-y-2">
                {recommendation.relevant_alternatives.map((alt, idx) => (
                  <div key={idx} className="p-2 rounded bg-slate-950/60 border border-slate-800/80 text-xs">
                    <div className="flex items-center justify-between text-slate-200 font-medium">
                      <span>{String(alt.name || 'Alternative')}</span>
                      <span className="font-mono text-sky-400">{String(alt.expected_duration_ms ?? '')} ms</span>
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">{String(alt.trade_off || '')}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Prospective Validation Benchmark Action */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <Play className="w-4 h-4 text-emerald-400" />
                  Validate Recommendation via Live Inference Benchmark
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Execute a bounded real benchmark against local Ollama to verify actual latency against the saved prospective forecast.
                </p>
              </div>
              <button
                type="button"
                onClick={handleValidateLive}
                disabled={validating}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium text-xs shadow-md shadow-emerald-600/20 transition-all self-start sm:self-auto"
              >
                {validating ? <LoadingSpinner size="sm" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                Run Live Validation Benchmark
              </button>
            </div>

            {validationError && (
              <div className="mt-3 text-xs text-rose-400">
                Validation error: {validationError}
              </div>
            )}

            {validationResult && (
              <div className="mt-4 pt-4 border-t border-slate-800 space-y-3">
                <div className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  Validation Complete ({validationResult.status})
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 rounded bg-slate-950 border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Observed Client E2E</span>
                    <span className="font-mono text-slate-100 font-bold">{validationResult.observed_client_e2e_duration_ms} ms</span>
                  </div>
                  <div className="p-2.5 rounded bg-slate-950 border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Observed Decode Speed</span>
                    <span className="font-mono text-slate-100 font-bold">{validationResult.observed_server_decode_throughput_tok_s ?? 'N/A'} tok/s</span>
                  </div>
                  <div className="p-2.5 rounded bg-slate-950 border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Latency Absolute Error</span>
                    <span className="font-mono text-slate-100 font-bold">
                      {validationResult.duration_absolute_error_ms !== null ? `${validationResult.duration_absolute_error_ms} ms` : 'N/A'}
                    </span>
                  </div>
                  <div className="p-2.5 rounded bg-slate-950 border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Latency Relative Error</span>
                    <span className="font-mono text-emerald-400 font-bold">
                      {validationResult.duration_relative_error_pct !== null ? `${validationResult.duration_relative_error_pct}%` : 'N/A'}
                    </span>
                  </div>
                </div>
                <p className="text-xs text-slate-300 italic">{validationResult.explanation}</p>
              </div>
            )}
          </div>
        </div>
      )}
        </>
      )}
    </div>
  );
};
