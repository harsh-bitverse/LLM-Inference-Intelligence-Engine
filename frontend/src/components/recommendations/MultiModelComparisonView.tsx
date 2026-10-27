import React, { useState, useEffect } from 'react';
import {
  Layers,
  Scale,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
  Info,
  ShieldCheck,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import {
  CandidateModelSpec,
  BenchmarkWorkloadTask,
  MultiModelComparisonReport,
  ModelComparisonEntry,
} from '../../api/types';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorMessage } from '../common/ErrorMessage';

export const MultiModelComparisonView: React.FC = () => {
  const [catalog, setCatalog] = useState<CandidateModelSpec[]>([]);
  const [workloads, setWorkloads] = useState<BenchmarkWorkloadTask[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([
    'llama3.2:1b',
    'qwen2.5:1.5b',
    'qwen2.5:0.5b',
  ]);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);

  // Workload and constraint inputs
  const [promptTokens, setPromptTokens] = useState<number>(64);
  const [outputTokens, setOutputTokens] = useState<number>(64);
  const [latencySlaMs, setLatencySlaMs] = useState<string>('5000');
  const [requireJsonCompliance, setRequireJsonCompliance] = useState<boolean>(false);
  const [requireCodePass, setRequireCodePass] = useState<boolean>(false);
  const [maxMemoryBudgetMb, setMaxMemoryBudgetMb] = useState<string>('');

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<MultiModelComparisonReport | null>(null);

  useEffect(() => {
    let isMounted = true;
    const loadData = async () => {
      try {
        const [catData, workData] = await Promise.all([
          apiClient.getModelCatalog().catch(() => []),
          apiClient.getWorkloadSuite().catch(() => []),
        ]);
        if (isMounted) {
          if (catData.length > 0) setCatalog(catData);
          if (workData.length > 0) setWorkloads(workData);
        }
      } catch {
        // Non-fatal
      }
    };
    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSelectWorkloadTask = (task: BenchmarkWorkloadTask) => {
    setActiveTaskId(task.task_id);
    setPromptTokens(Math.max(16, task.expected_prompt_words * 1));
    setOutputTokens(task.max_output_tokens);
    if (task.task_type === 'STRUCTURED_JSON') {
      setRequireJsonCompliance(true);
      setRequireCodePass(false);
    } else if (task.task_type === 'CODE_GENERATION') {
      setRequireCodePass(true);
      setRequireJsonCompliance(false);
    } else {
      setRequireJsonCompliance(false);
      setRequireCodePass(false);
    }
  };

  const handleToggleModel = (id: string) => {
    setSelectedModelIds((prev) =>
      prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]
    );
  };

  const handleRunComparison = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (selectedModelIds.length === 0) {
      setError('Select at least one candidate model to evaluate.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const rep = await apiClient.compareModels({
        candidate_model_ids: selectedModelIds,
        prompt_tokens: promptTokens,
        expected_output_tokens: outputTokens,
        latency_sla_ms: latencySlaMs ? parseFloat(latencySlaMs) : null,
        quality_constraints: {
          require_json_compliance: requireJsonCompliance,
          require_code_pass: requireCodePass,
          max_memory_budget_mb: maxMemoryBudgetMb ? parseFloat(maxMemoryBudgetMb) : null,
        },
      });
      setReport(rep);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Multi-model comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const renderFeasibilityBadge = (feasibility: string) => {
    switch (feasibility) {
      case 'ROBUST_FEASIBLE':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="w-3 h-3" /> Robust Feasible
          </span>
        );
      case 'NOMINAL_FEASIBLE_ROBUST_RISK':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <AlertTriangle className="w-3 h-3" /> Nominal Only (SLA Risk)
          </span>
        );
      case 'PREDICTED_SLA_MISS':
      case 'SLA_INFEASIBLE':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <XCircle className="w-3 h-3" /> SLA Miss
          </span>
        );
      case 'QUALITY_INFEASIBLE':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <XCircle className="w-3 h-3" /> Quality Infeasible
          </span>
        );
      case 'PROVEN_INFEASIBLE':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <XCircle className="w-3 h-3" /> Infeasible (Resource)
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-slate-800 text-slate-400 border border-slate-700">
            <HelpCircle className="w-3 h-3" /> Unknown Feasibility
          </span>
        );
    }
  };

  const installedCount = report?.candidate_evaluations.filter(
    (c) => c.installation_status === 'INSTALLED'
  ).length ?? catalog.filter((c) => c.is_installed).length;

  return (
    <div className="space-y-6">
      {/* Workload Suite Preset Selector */}
      {workloads.length > 0 && (
        <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <Layers className="w-3.5 h-3.5 text-sky-400" />
            Standard Multi-Model Workload Suite
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
            {workloads.map((task) => (
              <button
                key={task.task_id}
                type="button"
                onClick={() => handleSelectWorkloadTask(task)}
                className={`p-2.5 rounded-lg border text-left transition-all text-xs ${
                  activeTaskId === task.task_id
                    ? 'bg-sky-950/60 border-sky-500 text-sky-200 shadow-sm'
                    : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="font-semibold text-slate-200 truncate">{task.name}</div>
                <div className="text-[10px] text-slate-400 mt-1 uppercase tracking-wider">
                  {task.task_type.replace('_', ' ')}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Comparison Configuration Form */}
      <form
        onSubmit={handleRunComparison}
        className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-6"
      >
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <Scale className="w-4 h-4 text-sky-400" />
            Candidate Selection & Comparative Constraints
          </h2>
          <span className="text-xs text-slate-400">
            {selectedModelIds.length} candidate{selectedModelIds.length === 1 ? '' : 's'} selected
          </span>
        </div>

        {/* Candidate Model Checkboxes */}
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-2">
            Candidate Models to Evaluate
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {catalog.map((m) => {
              const isSelected = selectedModelIds.includes(m.model_id);
              return (
                <div
                  key={m.model_id}
                  onClick={() => handleToggleModel(m.model_id)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all text-xs ${
                    isSelected
                      ? 'bg-sky-950/40 border-sky-500/80 text-sky-100'
                      : 'bg-slate-950/40 border-slate-800 hover:border-slate-700 text-slate-400'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-slate-200">{m.display_name}</span>
                    {m.is_installed ? (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                        INSTALLED
                      </span>
                    ) : (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400 border border-slate-700">
                        NOT INSTALLED
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-slate-400 flex items-center gap-2">
                    <span>{m.parameter_count_billion}B params</span>
                    <span>•</span>
                    <span>{m.default_quantization}</span>
                    <span>•</span>
                    <span>~{m.expected_disk_size_mb} MB</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Workload Inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div>
            <label className="block font-medium text-slate-300 mb-1">Prompt Tokens (P)</label>
            <input
              type="number"
              min="1"
              value={promptTokens}
              onChange={(e) => setPromptTokens(parseInt(e.target.value, 10) || 1)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
              required
            />
          </div>
          <div>
            <label className="block font-medium text-slate-300 mb-1">Target Output Tokens (O)</label>
            <input
              type="number"
              min="1"
              value={outputTokens}
              onChange={(e) => setOutputTokens(parseInt(e.target.value, 10) || 1)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
              required
            />
          </div>
          <div>
            <label className="block font-medium text-slate-300 mb-1">Client E2E SLA (ms)</label>
            <input
              type="number"
              min="1"
              placeholder="e.g. 5000"
              value={latencySlaMs}
              onChange={(e) => setLatencySlaMs(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>
        </div>

        {/* Quality Constraints */}
        <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3 space-y-3">
          <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-purple-400" />
            Quality & Compatibility Constraints
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
            <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={requireJsonCompliance}
                onChange={(e) => setRequireJsonCompliance(e.target.checked)}
                className="rounded border-slate-700 bg-slate-900 text-sky-500 focus:ring-0"
              />
              <span>Require Valid JSON Schema</span>
            </label>
            <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={requireCodePass}
                onChange={(e) => setRequireCodePass(e.target.checked)}
                className="rounded border-slate-700 bg-slate-900 text-sky-500 focus:ring-0"
              />
              <span>Require Code Unit Test Pass (&ge; 80%)</span>
            </label>
            <div className="flex items-center gap-2">
              <span className="text-slate-400 whitespace-nowrap">Max RAM (MiB):</span>
              <input
                type="number"
                placeholder="No limit"
                value={maxMemoryBudgetMb}
                onChange={(e) => setMaxMemoryBudgetMb(e.target.value)}
                className="w-28 bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-sky-500 text-xs"
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs shadow-lg shadow-sky-600/20 disabled:opacity-50 transition-all"
          >
            {loading ? <LoadingSpinner size="sm" /> : <Scale className="w-4 h-4" />}
            Evaluate & Compare Candidates
          </button>
        </div>
      </form>

      {error && <ErrorMessage message={error} />}

      {/* Comparison Results */}
      {report && (
        <div className="space-y-6">
          {/* Workstation Measurement Limitation Notice if only 1 model is installed */}
          {installedCount <= 1 && (
            <div className="bg-amber-950/30 border border-amber-500/30 rounded-xl p-4 text-xs text-amber-200 flex items-start gap-3">
              <Info className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="font-semibold text-amber-300">
                  Single Installed Model Limitation Disclosed
                </div>
                <p className="text-slate-300 leading-relaxed">
                  Only 1 model (<code>llama3.2:1b</code>) is currently installed and benchmarked on this
                  workstation. Alternative candidates (such as <code>qwen2.5:1.5b</code> and{' '}
                  <code>qwen2.5:0.5b</code>) are preserved as unmeasured candidates with{' '}
                  <code>UNKNOWN_FEASIBILITY</code>. Downloading alternative models requires user
                  authorization.
                </p>
              </div>
            </div>
          )}

          {/* Tokenizer Caveat Banner */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-xs flex items-start gap-3">
            <Scale className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-slate-200">Tokenizer Incommensurability Warning: </span>
              <span className="text-slate-400">{report.comparability_caveat}</span>
            </div>
          </div>

          {/* Ranking Summary */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">Comparative Evaluation & Ranking</h3>
                <p className="text-xs text-slate-400 mt-0.5">{report.ranking_rationale}</p>
              </div>
              {report.recommended_model_id && (
                <div className="px-3 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-semibold">
                  Recommended: {report.recommended_model_id}
                </div>
              )}
            </div>

            {/* Comparison Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800 bg-slate-950/60">
                  <tr>
                    <th className="py-2.5 px-3">Rank</th>
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3">Feasibility</th>
                    <th className="py-2.5 px-3">Nominal Latency</th>
                    <th className="py-2.5 px-3">Worst-Case Latency</th>
                    <th className="py-2.5 px-3">Raw Decode</th>
                    <th className="py-2.5 px-3">Chars/sec</th>
                    <th className="py-2.5 px-3">Quality Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {report.candidate_evaluations.map((cand: ModelComparisonEntry) => (
                    <tr
                      key={cand.model_id}
                      className={
                        cand.rank === 1
                          ? 'bg-sky-950/20 hover:bg-sky-950/30'
                          : 'hover:bg-slate-950/40'
                      }
                    >
                      <td className="py-3 px-3 font-semibold text-slate-100">
                        {cand.rank ? `#${cand.rank}` : '—'}
                      </td>
                      <td className="py-3 px-3">
                        <div className="font-medium text-slate-200">{cand.display_name}</div>
                        <div className="text-[10px] text-slate-400">
                          {cand.parameter_count_billion}B params • {cand.quantization}
                        </div>
                      </td>
                      <td className="py-3 px-3">
                        {cand.installation_status === 'INSTALLED' ? (
                          <span className="text-[10px] text-emerald-400 font-medium">Installed</span>
                        ) : (
                          <span className="text-[10px] text-slate-500 font-medium">Uninstalled</span>
                        )}
                      </td>
                      <td className="py-3 px-3">{renderFeasibilityBadge(cand.conservative_feasibility)}</td>
                      <td className="py-3 px-3 font-mono">
                        {cand.nominal_latency_ms !== null && cand.nominal_latency_ms !== undefined
                          ? `${cand.nominal_latency_ms.toFixed(1)} ms`
                          : '—'}
                      </td>
                      <td className="py-3 px-3 font-mono">
                        {cand.worst_case_latency_ms !== null && cand.worst_case_latency_ms !== undefined
                          ? `${cand.worst_case_latency_ms.toFixed(1)} ms`
                          : '—'}
                      </td>
                      <td className="py-3 px-3 font-mono text-slate-400">
                        {cand.raw_decode_tokens_per_second !== null && cand.raw_decode_tokens_per_second !== undefined
                          ? `${cand.raw_decode_tokens_per_second.toFixed(1)} tok/s*`
                          : '—'}
                      </td>
                      <td className="py-3 px-3 font-mono text-sky-300">
                        {cand.characters_per_second !== null && cand.characters_per_second !== undefined
                          ? `${cand.characters_per_second.toFixed(1)} ch/s`
                          : '—'}
                      </td>
                      <td className="py-3 px-3">
                        {cand.quality_status === 'QUALITY_SATISFIED' && (
                          <span className="text-[10px] text-emerald-400 font-medium">Satisfied</span>
                        )}
                        {cand.quality_status === 'QUALITY_VIOLATED' && (
                          <span className="text-[10px] text-rose-400 font-medium">Violated</span>
                        )}
                        {cand.quality_status === 'UNKNOWN_QUALITY_NO_EVALUATION' && (
                          <span className="text-[10px] text-slate-500 font-medium">Unassessed</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Disclaimers & Notes */}
            <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-400 space-y-1">
              <div className="font-medium text-slate-300">Evaluation Notes & Advisories:</div>
              {report.candidate_evaluations.flatMap((c) => c.disclaimers).map((disc, idx) => (
                <div key={idx} className="flex items-center gap-1.5">
                  <span className="w-1 h-1 rounded-full bg-slate-500"></span>
                  <span>{disc}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
