import React, { useEffect, useState } from 'react';
import {
  Sparkles,
  Layers,
  AlertTriangle,
  CheckCircle2,
  Sliders,
  Award,
  Clock,
  Eye,
} from 'lucide-react';
import { apiClient } from '../api/client';
import {
  EngineType,
  OptimizationHistoryItem,
  OptimizationObjective,
  OptimizationRequest,
  OptimizationResult,
  QuantizationType,
  SelectionPolicy,
} from '../api/types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorMessage } from '../components/common/ErrorMessage';
import { FeasibilityBadge } from '../components/common/FeasibilityBadge';

export const OptimizerPage: React.FC = () => {
  // Form State
  const [modelId, setModelId] = useState<string>('meta-llama/Llama-3-8B');
  const [paramBillion, setParamBillion] = useState<number>(8.0);
  const [promptTokens, setPromptTokens] = useState<number>(128);
  const [outputTokens, setOutputTokens] = useState<number>(64);
  const [selectedEngines, setSelectedEngines] = useState<EngineType[]>(['mock', 'llamacpp']);
  const [selectedQuantizations, setSelectedQuantizations] = useState<QuantizationType[]>([
    'fp16',
    'int8',
    'int4',
    'gguf_q4_k_m',
  ]);
  const [selectedConcurrencies, setSelectedConcurrencies] = useState<number[]>([1, 2]);
  const [selectedObjectives, setSelectedObjectives] = useState<OptimizationObjective[]>([
    'minimize_ttft',
    'maximize_throughput',
  ]);
  const [hardwareCatalogId, setHardwareCatalogId] = useState<string>('nvidia-rtx-4090');
  const [costPerHour, setCostPerHour] = useState<number>(1.5);
  const [selectionPolicy, setSelectionPolicy] = useState<SelectionPolicy>('balanced_knee_point');
  const [primaryObjective, setPrimaryObjective] = useState<OptimizationObjective>('minimize_ttft');

  // Execution State
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [activeResult, setActiveResult] = useState<OptimizationResult | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  // History State
  const [history, setHistory] = useState<OptimizationHistoryItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState<boolean>(true);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setLoadingHistory(true);
    setHistoryError(null);
    try {
      const res = await apiClient.listOptimizations(20);
      setHistory(res.items);
      if (res.items.length > 0 && res.items[0] && !activeResult) {
        const full = await apiClient.getOptimizationById(res.items[0].optimization_id);
        setActiveResult(full.result);
      }
    } catch (err: unknown) {
      setHistoryError(err instanceof Error ? err.message : 'Failed to load optimization history');
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleSelectHistory = async (optId: string) => {
    try {
      const full = await apiClient.getOptimizationById(optId);
      setActiveResult(full.result);
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : 'Failed to load optimization run details');
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleToggleEngine = (engine: EngineType) => {
    setSelectedEngines((prev) =>
      prev.includes(engine) ? (prev.length > 1 ? prev.filter((e) => e !== engine) : prev) : [...prev, engine]
    );
  };

  const handleToggleQuant = (quant: QuantizationType) => {
    setSelectedQuantizations((prev) =>
      prev.includes(quant) ? (prev.length > 1 ? prev.filter((q) => q !== quant) : prev) : [...prev, quant]
    );
  };

  const handleToggleObjective = (obj: OptimizationObjective) => {
    setSelectedObjectives((prev) =>
      prev.includes(obj) ? (prev.length > 1 ? prev.filter((o) => o !== obj) : prev) : [...prev, obj]
    );
  };

  const handleToggleConcurrency = (conc: number) => {
    setSelectedConcurrencies((prev) =>
      prev.includes(conc) ? (prev.length > 1 ? prev.filter((c) => c !== conc) : prev) : [...prev, conc].sort()
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelId.trim()) {
      setFormError('Model identifier is required.');
      return;
    }
    if (selectedObjectives.length === 0) {
      setFormError('At least one optimization objective must be selected.');
      return;
    }
    if (selectedObjectives.includes('minimize_cost') && (costPerHour === undefined || costPerHour <= 0)) {
      setFormError("Cost objective requires a positive 'Cost per GPU Hour'.");
      return;
    }

    setFormError(null);
    setSubmitting(true);

    const payload: OptimizationRequest = {
      workload: {
        prompt_tokens: promptTokens,
        output_tokens: outputTokens,
      },
      search_space: {
        models: [
          {
            model_id: modelId.trim(),
            parameter_count_billion: paramBillion,
          },
        ],
        quantizations: selectedQuantizations,
        concurrencies: selectedConcurrencies,
        engines: selectedEngines,
        tensor_parallel_sizes: [1],
      },
      objectives: selectedObjectives,
      hardware_catalog_id: hardwareCatalogId.trim() || undefined,
      cost_per_gpu_hour: selectedObjectives.includes('minimize_cost') ? costPerHour : undefined,
      selection_policy: selectionPolicy,
      primary_objective: selectionPolicy === 'prioritize_objective' ? primaryObjective : undefined,
    };

    try {
      const res = await apiClient.recommendOptimization(payload);
      setActiveResult(res);
      fetchHistory();
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : 'Optimization search failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Page Header */}
      <div className="border-b border-slate-800 pb-5">
        <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
          Constrained Configuration Optimizer & Pareto Sizer
        </h1>
        <p className="mt-1 text-xs text-slate-400">
          Search candidate engine and quantization parameters, check physical hardware feasibility, and compute non-dominated Pareto trade-offs.
        </p>
      </div>

      {/* Main Grid: Form + Optimization Outcomes */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Search Specification Form */}
        <div className="lg:col-span-5 rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <Sliders className="w-4 h-4 text-purple-400" />
            <h3 className="text-sm font-semibold text-slate-200">Search Space Configuration</h3>
          </div>

          {formError && <ErrorMessage title="Validation Error" message={formError} />}

          <form onSubmit={handleSubmit} className="space-y-4 text-xs">
            {/* Model Architecture */}
            <div className="grid grid-cols-3 gap-2">
              <div className="col-span-2">
                <label className="block text-slate-300 font-medium mb-1">Target Model</label>
                <input
                  type="text"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>
              <div>
                <label className="block text-slate-300 font-medium mb-1">Params (B)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  value={paramBillion}
                  onChange={(e) => setParamBillion(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>
            </div>

            {/* Workload Spec */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Prompt Tokens</label>
                <input
                  type="number"
                  min="1"
                  value={promptTokens}
                  onChange={(e) => setPromptTokens(parseInt(e.target.value) || 1)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>
              <div>
                <label className="block text-slate-300 font-medium mb-1">Output Tokens</label>
                <input
                  type="number"
                  min="0"
                  value={outputTokens}
                  onChange={(e) => setOutputTokens(parseInt(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>
            </div>

            {/* Target Hardware Catalog ID */}
            <div>
              <label className="block text-slate-300 font-medium mb-1">Hardware Catalog ID</label>
              <select
                value={hardwareCatalogId}
                onChange={(e) => setHardwareCatalogId(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
              >
                <option value="nvidia-rtx-4090">nvidia-rtx-4090 (24 GB, 1008 GB/s)</option>
                <option value="nvidia-a100-sxm4-80gb">nvidia-a100-sxm4-80gb (80 GB, 2039 GB/s)</option>
                <option value="nvidia-h100-sxm5-80gb">nvidia-h100-sxm5-80gb (80 GB, 3350 GB/s)</option>
                <option value="nvidia-t4">nvidia-t4 (16 GB, 320 GB/s)</option>
              </select>
            </div>

            {/* Candidate Engines */}
            <div>
              <label className="block text-slate-300 font-medium mb-1.5">Candidate Engines</label>
              <div className="flex gap-2">
                {(['mock', 'llamacpp', 'ollama'] as EngineType[]).map((eng) => (
                  <button
                    type="button"
                    key={eng}
                    onClick={() => handleToggleEngine(eng)}
                    className={`px-3 py-1.5 rounded-lg font-mono text-xs border transition-colors ${
                      selectedEngines.includes(eng)
                        ? 'bg-purple-950 text-purple-300 border-purple-700/60 font-semibold'
                        : 'bg-slate-950 text-slate-500 border-slate-800'
                    }`}
                  >
                    {eng}
                  </button>
                ))}
              </div>
            </div>

            {/* Candidate Quantizations */}
            <div>
              <label className="block text-slate-300 font-medium mb-1.5">Candidate Quantizations</label>
              <div className="grid grid-cols-2 gap-2">
                {(['fp16', 'int8', 'int4', 'gguf_q4_k_m'] as QuantizationType[]).map((q) => (
                  <button
                    type="button"
                    key={q}
                    onClick={() => handleToggleQuant(q)}
                    className={`px-2.5 py-1.5 rounded-lg font-mono text-xs border text-left transition-colors truncate ${
                      selectedQuantizations.includes(q)
                        ? 'bg-purple-950 text-purple-300 border-purple-700/60 font-semibold'
                        : 'bg-slate-950 text-slate-500 border-slate-800'
                    }`}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>

            {/* Candidate Concurrencies */}
            <div>
              <label className="block text-slate-300 font-medium mb-1.5">Candidate Concurrencies</label>
              <div className="flex gap-2">
                {[1, 2, 4, 8].map((c) => (
                  <button
                    type="button"
                    key={c}
                    onClick={() => handleToggleConcurrency(c)}
                    className={`px-3 py-1 rounded font-mono text-xs border transition-colors ${
                      selectedConcurrencies.includes(c)
                        ? 'bg-sky-950 text-sky-300 border-sky-700/60 font-semibold'
                        : 'bg-slate-950 text-slate-500 border-slate-800'
                    }`}
                  >
                    c={c}
                  </button>
                ))}
              </div>
            </div>

            {/* Optimization Objectives */}
            <div>
              <label className="block text-slate-300 font-medium mb-1.5">Optimization Objectives</label>
              <div className="grid grid-cols-1 gap-1.5">
                {(
                  [
                    'minimize_ttft',
                    'minimize_tpot',
                    'maximize_throughput',
                    'minimize_vram',
                    'minimize_cost',
                  ] as OptimizationObjective[]
                ).map((obj) => (
                  <button
                    type="button"
                    key={obj}
                    onClick={() => handleToggleObjective(obj)}
                    className={`px-3 py-1.5 rounded-lg font-mono text-xs border text-left flex items-center justify-between transition-colors ${
                      selectedObjectives.includes(obj)
                        ? 'bg-sky-950 text-sky-300 border-sky-700/60 font-semibold'
                        : 'bg-slate-950 text-slate-500 border-slate-800'
                    }`}
                  >
                    <span>{obj}</span>
                    {selectedObjectives.includes(obj) && <CheckCircle2 className="w-3.5 h-3.5 text-sky-400" />}
                  </button>
                ))}
              </div>
            </div>

            {/* Selection Policy */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Selection Policy</label>
                <select
                  value={selectionPolicy}
                  onChange={(e) => setSelectionPolicy(e.target.value as SelectionPolicy)}
                  className="w-full px-2.5 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono text-xs"
                >
                  <option value="balanced_knee_point">balanced_knee_point</option>
                  <option value="prioritize_objective">prioritize_objective</option>
                  <option value="lowest_cost">lowest_cost</option>
                  <option value="none">none (Frontier only)</option>
                </select>
              </div>

              {selectionPolicy === 'prioritize_objective' ? (
                <div>
                  <label className="block text-slate-300 font-medium mb-1">Primary Objective</label>
                  <select
                    value={primaryObjective}
                    onChange={(e) => setPrimaryObjective(e.target.value as OptimizationObjective)}
                    className="w-full px-2.5 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono text-xs"
                  >
                    {selectedObjectives.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                </div>
              ) : selectedObjectives.includes('minimize_cost') ? (
                <div>
                  <label className="block text-slate-300 font-medium mb-1">Cost / GPU Hr ($)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    value={costPerHour}
                    onChange={(e) => setCostPerHour(parseFloat(e.target.value) || 0)}
                    className="w-full px-2.5 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono text-xs"
                  />
                </div>
              ) : null}
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="w-full py-2.5 px-4 bg-purple-600 hover:bg-purple-500 disabled:bg-slate-800 text-white font-semibold rounded-lg shadow-sm transition-colors flex items-center justify-center gap-2"
            >
              <Sparkles className="w-4 h-4 fill-current" />
              {submitting ? 'Searching Space...' : 'Compute Pareto Frontier'}
            </button>
          </form>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-7 space-y-6">
          {activeResult ? (
            <div className="space-y-6">
              {/* Summary & Candidate Search KPI Cards */}
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-800 pb-3">
                  <div>
                    <h2 className="text-base font-bold text-slate-100 font-mono">
                      Optimization Run: {activeResult.request_id}
                    </h2>
                    <span className="text-xs text-slate-400">
                      Generated {activeResult.total_candidates_generated} configurations • Evaluated{' '}
                      {activeResult.total_candidates_evaluated}
                    </span>
                  </div>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/70 p-3 rounded-lg border border-slate-800">
                  {activeResult.summary}
                </p>

                {/* KPI Grid */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs">
                    <span className="text-slate-500 block text-[10px]">FEASIBLE CONFIGS</span>
                    <span className="text-emerald-400 font-bold font-mono text-base">
                      {activeResult.feasible_candidates_count}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs">
                    <span className="text-slate-500 block text-[10px]">REJECTED INFEASIBLE</span>
                    <span className="text-red-400 font-bold font-mono text-base">
                      {activeResult.infeasible_candidates_count}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs">
                    <span className="text-slate-500 block text-[10px]">PARETO FRONTIER</span>
                    <span className="text-purple-400 font-bold font-mono text-base">
                      {activeResult.pareto_frontier.length}
                    </span>
                  </div>
                </div>

                {/* Incomparable evidence warnings */}
                {activeResult.has_incomparable_evidence && (
                  <div className="p-3.5 rounded-lg border border-amber-800/80 bg-amber-950/30 text-xs text-amber-200 space-y-1">
                    <div className="flex items-center gap-1.5 font-semibold text-amber-300">
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      Incomparable Evidence Caveat
                    </div>
                    {activeResult.evidence_caveats.map((cav, idx) => (
                      <p key={idx} className="text-amber-200/90">
                        • {cav}
                      </p>
                    ))}
                  </div>
                )}
              </div>

              {/* Recommended Candidate Highlight */}
              {activeResult.recommended_candidate && (
                <div className="rounded-xl border border-emerald-600/50 bg-emerald-950/20 p-5 space-y-3">
                  <div className="flex items-center justify-between border-b border-emerald-900/50 pb-2.5">
                    <div className="flex items-center gap-2">
                      <Award className="w-5 h-5 text-emerald-400" />
                      <h3 className="font-bold text-emerald-200 text-sm">
                        Recommended Candidate ({activeResult.selection_rationale || 'Selected Recipe'})
                      </h3>
                    </div>
                    <FeasibilityBadge
                      status={activeResult.recommended_candidate.feasibility_status}
                    />
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-xs">
                    <div className="p-2 rounded bg-slate-950/80 border border-slate-800">
                      <span className="text-slate-500 block text-[10px]">ENGINE</span>
                      <span className="text-slate-100 font-semibold uppercase">
                        {activeResult.recommended_candidate.config.engine_type}
                      </span>
                    </div>

                    <div className="p-2 rounded bg-slate-950/80 border border-slate-800">
                      <span className="text-slate-500 block text-[10px]">QUANTIZATION</span>
                      <span className="text-slate-100 font-semibold">
                        {activeResult.recommended_candidate.config.quantization}
                      </span>
                    </div>

                    <div className="p-2 rounded bg-slate-950/80 border border-slate-800">
                      <span className="text-slate-500 block text-[10px]">CONCURRENCY</span>
                      <span className="text-slate-100 font-semibold">
                        {activeResult.recommended_candidate.config.concurrency}
                      </span>
                    </div>

                    <div className="p-2 rounded bg-slate-950/80 border border-slate-800">
                      <span className="text-slate-500 block text-[10px]">TENSOR PARALLEL</span>
                      <span className="text-slate-100 font-semibold">
                        TP={activeResult.recommended_candidate.config.tensor_parallel_size}
                      </span>
                    </div>
                  </div>

                  {/* Objective Values for Recommended Candidate */}
                  <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 space-y-1.5 text-xs">
                    <span className="font-semibold text-slate-300 text-[11px] block">
                      Evaluated Objective Values:
                    </span>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-[11px]">
                      {Object.entries(activeResult.recommended_candidate.objective_values).map(
                        ([objName, val]) => (
                          <div key={objName} className="p-1.5 rounded bg-slate-900 border border-slate-800">
                            <span className="text-slate-500 block text-[9px] uppercase truncate">
                              {objName}
                            </span>
                            <span className="text-sky-300 font-bold">
                              {typeof val === 'number' ? val.toFixed(1) : String(val)}{' '}
                              {activeResult.recommended_candidate?.objective_units[objName] || ''}
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>

                  <p className="text-xs text-emerald-300/90 italic">
                    {activeResult.recommended_candidate.trade_off_summary}
                  </p>
                </div>
              )}

              {/* Feasibility Rejections Breakdown */}
              {activeResult.rejections_by_reason && Object.keys(activeResult.rejections_by_reason).length > 0 && (
                <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 space-y-2">
                  <div className="flex items-center gap-2 font-semibold text-xs text-amber-300">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    Feasibility Rejection Reasons ({Object.values(activeResult.rejections_by_reason).reduce((a, b) => a + b, 0)} pruned)
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
                    {Object.entries(activeResult.rejections_by_reason).map(([reason, count]) => (
                      <div key={reason} className="p-2 rounded bg-slate-950 border border-slate-800 flex justify-between items-center">
                        <span className="text-slate-300 truncate mr-2" title={reason}>{reason}</span>
                        <span className="text-amber-400 font-bold">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Pareto Frontier Table */}
              <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-purple-400" />
                    <h3 className="text-sm font-semibold text-slate-200">
                      Pareto-Optimal Frontier ({activeResult.pareto_frontier.length})
                    </h3>
                  </div>
                </div>

                {activeResult.pareto_frontier.length === 0 ? (
                  <div className="p-6 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-lg">
                    No feasible candidate satisfied all search constraints.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {activeResult.pareto_frontier.map((cand, idx) => (
                      <div
                        key={idx}
                        className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 hover:border-slate-700 text-xs space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 flex-wrap font-mono">
                            <span className="font-bold text-slate-200">
                              {cand.config.model_spec.model_id}
                            </span>
                            <span className="text-purple-400">[{cand.config.quantization}]</span>
                            <span className="text-slate-400">c={cand.config.concurrency}</span>
                            <span className="text-slate-400">eng={cand.config.engine_type}</span>
                          </div>
                          <FeasibilityBadge status={cand.feasibility_status} />
                        </div>

                        {/* Objectives Grid */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                          {Object.entries(cand.objective_values).map(([objKey, val]) => (
                            <div key={objKey} className="p-1.5 rounded bg-slate-900/90 border border-slate-800">
                              <span className="text-slate-500 block text-[9px] uppercase truncate">
                                {objKey}
                              </span>
                              <span className="text-slate-100 font-semibold">
                                {typeof val === 'number' ? val.toFixed(1) : String(val)} {cand.objective_units[objKey] || ''}
                              </span>
                              <span className="text-[9px] text-slate-500 block truncate">
                                {cand.evidence_status[objKey]}
                              </span>
                            </div>
                          ))}
                        </div>

                        {cand.trade_off_summary && (
                          <p className="text-[11px] text-slate-400 italic">
                            Trade-off: {cand.trade_off_summary}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 text-xs text-center p-8 border border-dashed border-slate-800 rounded-xl space-y-2 min-h-[300px]">
              <Sparkles className="w-8 h-8 text-slate-600" />
              <p>No active optimization recommendations displayed.</p>
              <p className="text-slate-600 max-w-sm">
                Submit a workload and candidate dimensions to calculate non-dominated Pareto frontier points.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Historical Optimization Runs Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-purple-400" />
            <h3 className="text-sm font-semibold text-slate-200">Historical Optimization Runs</h3>
          </div>
          <button
            onClick={fetchHistory}
            className="text-xs text-sky-400 hover:text-sky-300 flex items-center gap-1"
          >
            <Clock className="w-3.5 h-3.5" /> Refresh
          </button>
        </div>

        {loadingHistory ? (
          <LoadingSpinner message="Fetching optimization history..." />
        ) : historyError ? (
          <ErrorMessage title="Error loading optimization history" message={historyError} onRetry={fetchHistory} />
        ) : history.length === 0 ? (
          <div className="py-8 text-center text-xs text-slate-500">No historical optimization runs recorded.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300 font-mono">
              <thead className="bg-slate-950/80 text-slate-400 font-semibold border-b border-slate-800 text-[11px] uppercase">
                <tr>
                  <th className="p-2.5">Optimization ID</th>
                  <th className="p-2.5">Date</th>
                  <th className="p-2.5">Model & Workload</th>
                  <th className="p-2.5">Feasible</th>
                  <th className="p-2.5">Infeasible / Unknown</th>
                  <th className="p-2.5">Recommended Config</th>
                  <th className="p-2.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {history.map((h) => (
                  <tr key={h.optimization_id} className="hover:bg-slate-800/30">
                    <td className="p-2.5 font-bold text-slate-200 truncate max-w-[140px]" title={h.optimization_id}>
                      {h.optimization_id}
                    </td>
                    <td className="p-2.5 text-slate-500 whitespace-nowrap">
                      {h.created_at ? new Date(h.created_at).toLocaleDateString() : 'Recent'}
                    </td>
                    <td className="p-2.5 text-slate-300">
                      <span className="text-slate-200 font-semibold">{h.model_id}</span>
                      <span className="text-[10px] text-slate-500 block">
                        prompt: {h.prompt_tokens} / out: {h.output_tokens}
                      </span>
                    </td>
                    <td className="p-2.5 text-emerald-400 font-semibold">{h.feasible_candidates_count}</td>
                    <td className="p-2.5 text-slate-400">
                      {h.infeasible_candidates_count} / {h.unknown_feasibility_count}
                    </td>
                    <td className="p-2.5 text-purple-300 truncate max-w-[160px]" title={h.recommended_config_id || 'None'}>
                      {h.recommended_config_id || <span className="text-slate-500">None</span>}
                    </td>
                    <td className="p-2.5 text-right">
                      <button
                        onClick={() => handleSelectHistory(h.optimization_id)}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-purple-400 text-xs font-sans inline-flex items-center gap-1"
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
