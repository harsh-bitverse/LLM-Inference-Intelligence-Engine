import React, { useEffect, useState } from 'react';
import {
  TrendingUp,
  PlusCircle,
  Clock,
  HelpCircle,
} from 'lucide-react';
import { apiClient } from '../api/client';
import {
  EvaluationResult,
  ExecutionMode,
  PredictionTarget,
} from '../api/types';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorMessage } from '../components/common/ErrorMessage';
import { EvidenceBadge } from '../components/common/EvidenceBadge';

export const PredictionsPage: React.FC = () => {
  const [evaluations, setEvaluations] = useState<EvaluationResult[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [targetFilter, setTargetFilter] = useState<string>('');

  // Form State
  const [showForm, setShowForm] = useState<boolean>(false);
  const [target, setTarget] = useState<PredictionTarget>('ttft_ms');
  const [predictorId, setPredictorId] = useState<string>('ridge-v1');
  const [predictorType, setPredictorType] = useState<string>('RidgeRegressionPredictor');
  const [splitStrategy, setSplitStrategy] = useState<string>('RANDOM');
  const [sampleCount, setSampleCount] = useState<number>(20);
  const [trainSampleCount, setTrainSampleCount] = useState<number>(80);
  const [mae, setMae] = useState<number>(1.2);
  const [rmse, setRmse] = useState<number>(1.8);
  const [mapePercent, setMapePercent] = useState<number>(8.5);
  const [baselineMae, setBaselineMae] = useState<number>(2.4);
  const [baselineRmse, setBaselineRmse] = useState<number>(3.1);
  const [isSynthetic, setIsSynthetic] = useState<boolean>(false);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('REAL');
  const [notes, setNotes] = useState<string>('Empirical cross-validation evaluation record');
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [formError, setFormError] = useState<string | null>(null);

  const fetchEvaluations = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.listEvaluations({
        limit: 50,
        target: targetFilter || undefined,
      });
      setEvaluations(res.items);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load prediction evaluations');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvaluations();
  }, [targetFilter]);

  const handleSubmitEvaluation = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    setSubmitting(true);

    const isStatSufficient = sampleCount >= 10;
    const payload: EvaluationResult = {
      target,
      predictor_id: predictorId.trim(),
      predictor_type: predictorType.trim(),
      split_strategy: splitStrategy,
      sample_count: sampleCount,
      test_sample_count: sampleCount,
      train_sample_count: trainSampleCount,
      unit: target === 'peak_vram_mb' ? 'MB' : target === 'output_throughput_tok_s' ? 'tok/s' : 'ms',
      mae,
      rmse,
      mape_percent: mapePercent,
      baseline_mae: baselineMae,
      baseline_rmse: baselineRmse,
      is_synthetic: isSynthetic,
      execution_mode: executionMode,
      sufficiency_threshold: 10,
      is_statistically_sufficient: isStatSufficient,
      notes,
    };

    try {
      await apiClient.createEvaluation(payload);
      setShowForm(false);
      fetchEvaluations();
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : 'Failed to record prediction evaluation');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
            Empirical Prediction & Generalization Evaluations
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            Validation scores (MAE, RMSE, MAPE), median baseline benchmarks, and out-of-distribution sufficiency checks.
          </p>
        </div>

        <button
          onClick={() => setShowForm(!showForm)}
          className="inline-flex items-center gap-1.5 px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors self-start sm:self-auto"
        >
          <PlusCircle className="w-4 h-4" />
          {showForm ? 'Hide Form' : 'Record Evaluation'}
        </button>
      </div>

      {/* Scientific Honesty Disclaimer Banner */}
      <div className="p-4 rounded-xl border border-blue-900/60 bg-blue-950/20 text-xs text-blue-200 flex items-start gap-3">
        <HelpCircle className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
        <div>
          <span className="font-semibold text-blue-300 block">Scientific Transparency Principle:</span>
          Empirical prediction models must be strictly distinguished from theoretical estimates and simulated mock runs. Evaluations marked <span className="font-mono text-rose-300">SYNTHETIC</span> represent artificial simulation benchmarks and must never be cited as validated real-world performance.
        </div>
      </div>

      {/* Record Evaluation Form Modal/Drawer */}
      {showForm && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <TrendingUp className="w-4 h-4 text-sky-400" />
            <h3 className="text-sm font-semibold text-slate-200">Record Empirical Evaluation Outcome</h3>
          </div>

          {formError && <ErrorMessage title="Submission Error" message={formError} />}

          <form onSubmit={handleSubmitEvaluation} className="space-y-4 text-xs">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Prediction Target</label>
                <select
                  value={target}
                  onChange={(e) => setTarget(e.target.value as PredictionTarget)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                >
                  <option value="ttft_ms">ttft_ms (Time to First Token)</option>
                  <option value="tpot_ms">tpot_ms (Time per Output Token)</option>
                  <option value="output_throughput_tok_s">output_throughput_tok_s</option>
                  <option value="peak_vram_mb">peak_vram_mb</option>
                  <option value="chunk_latency_ms">chunk_latency_ms</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Predictor ID</label>
                <input
                  type="text"
                  value={predictorId}
                  onChange={(e) => setPredictorId(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Predictor Type</label>
                <input
                  type="text"
                  value={predictorType}
                  onChange={(e) => setPredictorType(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">MAE</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={mae}
                  onChange={(e) => setMae(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">RMSE</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={rmse}
                  onChange={(e) => setRmse(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Baseline MAE</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={baselineMae}
                  onChange={(e) => setBaselineMae(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Baseline RMSE</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={baselineRmse}
                  onChange={(e) => setBaselineRmse(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">Test Samples</label>
                <input
                  type="number"
                  min="1"
                  value={sampleCount}
                  onChange={(e) => setSampleCount(parseInt(e.target.value) || 1)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Train Samples</label>
                <input
                  type="number"
                  min="0"
                  value={trainSampleCount}
                  onChange={(e) => setTrainSampleCount(parseInt(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Split Strategy</label>
                <select
                  value={splitStrategy}
                  onChange={(e) => setSplitStrategy(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                >
                  <option value="RANDOM">RANDOM</option>
                  <option value="GROUPED_BY_SUITE">GROUPED_BY_SUITE</option>
                  <option value="GROUPED_BY_MODEL">GROUPED_BY_MODEL</option>
                  <option value="IN_SAMPLE">IN_SAMPLE</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Execution Mode</label>
                <select
                  value={executionMode}
                  onChange={(e) => setExecutionMode(e.target.value as ExecutionMode)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                >
                  <option value="REAL">REAL (Real Hardware)</option>
                  <option value="SIMULATED">SIMULATED (Mock Engine)</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-300 font-medium mb-1">MAPE (%)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  value={mapePercent}
                  onChange={(e) => setMapePercent(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Notes / Description</label>
                <input
                  type="text"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Evaluation observations..."
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 font-mono"
                />
              </div>
            </div>

            <div className="flex items-center gap-2 pt-1">
              <input
                type="checkbox"
                id="synFlag"
                checked={isSynthetic}
                onChange={(e) => setIsSynthetic(e.target.checked)}
                className="rounded bg-slate-950 border-slate-800 text-sky-600 focus:ring-0"
              />
              <label htmlFor="synFlag" className="text-slate-400 select-none cursor-pointer">
                Dataset derived from synthetic virtual clock generation (tag as SYNTHETIC)
              </label>
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="py-2.5 px-5 bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 text-white font-semibold rounded-lg shadow-sm transition-colors"
            >
              {submitting ? 'Recording...' : 'Persist Evaluation Record'}
            </button>
          </form>
        </div>
      )}

      {/* Target Filter Bar */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400">Filter by Target:</span>
          <select
            value={targetFilter}
            onChange={(e) => setTargetFilter(e.target.value)}
            className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-200 font-mono text-xs"
          >
            <option value="">All Targets</option>
            <option value="ttft_ms">ttft_ms</option>
            <option value="tpot_ms">tpot_ms</option>
            <option value="output_throughput_tok_s">output_throughput_tok_s</option>
            <option value="peak_vram_mb">peak_vram_mb</option>
            <option value="chunk_latency_ms">chunk_latency_ms</option>
          </select>
        </div>

        <button
          onClick={fetchEvaluations}
          className="text-xs text-sky-400 hover:text-sky-300 flex items-center gap-1"
        >
          <Clock className="w-3.5 h-3.5" /> Refresh List
        </button>
      </div>

      {/* Evaluations List */}
      {loading ? (
        <LoadingSpinner message="Fetching empirical prediction evaluations..." />
      ) : error ? (
        <ErrorMessage title="Failed to load evaluations" message={error} onRetry={fetchEvaluations} />
      ) : evaluations.length === 0 ? (
        <div className="p-8 text-center border border-dashed border-slate-800 rounded-xl text-slate-500 text-xs">
          No prediction evaluations found. Record an evaluation or run model validation benchmarks.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {evaluations.map((ev, index) => (
            <div
              key={ev.evaluation_id || index}
              className="p-5 rounded-xl border border-slate-800 bg-slate-900/50 space-y-3 text-xs hover:border-slate-700 transition-colors"
            >
              <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                <div>
                  <span className="font-bold text-slate-100 font-mono text-sm block">{ev.target}</span>
                  <span className="text-[11px] text-slate-400 font-mono">{ev.predictor_id}</span>
                </div>
                <div className="flex items-center gap-2">
                  <EvidenceBadge executionMode={ev.execution_mode} isSynthetic={ev.is_synthetic} />
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                      ev.is_statistically_sufficient
                        ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                        : 'bg-amber-950 text-amber-300 border border-amber-800/60'
                    }`}
                  >
                    {ev.is_statistically_sufficient ? 'SUFFICIENT' : 'INSUFFICIENT SAMPLE'}
                  </span>
                </div>
              </div>

              {/* Error Metrics Comparison */}
              <div className="grid grid-cols-3 gap-2 font-mono p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                <div>
                  <span className="text-slate-500 block text-[10px]">MAE (MODEL)</span>
                  <span className="font-bold text-slate-100 text-sm">
                    {ev.mae.toFixed(2)} {ev.unit}
                  </span>
                  {ev.baseline_mae !== undefined && ev.baseline_mae !== null && (
                    <span className="text-[10px] text-slate-500 block">
                      base: {ev.baseline_mae.toFixed(2)}
                    </span>
                  )}
                </div>

                <div>
                  <span className="text-slate-500 block text-[10px]">RMSE (MODEL)</span>
                  <span className="font-bold text-slate-100 text-sm">
                    {ev.rmse.toFixed(2)} {ev.unit}
                  </span>
                  {ev.baseline_rmse !== undefined && ev.baseline_rmse !== null && (
                    <span className="text-[10px] text-slate-500 block">
                      base: {ev.baseline_rmse.toFixed(2)}
                    </span>
                  )}
                </div>

                <div>
                  <span className="text-slate-500 block text-[10px]">MAPE</span>
                  <span className="font-bold text-sky-400 text-sm">
                    {ev.mape_percent !== undefined && ev.mape_percent !== null ? `${ev.mape_percent.toFixed(1)}%` : 'N/A'}
                  </span>
                  <span className="text-[10px] text-slate-500 block">
                    {ev.split_strategy}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <span>
                  Samples: <strong className="text-slate-200">{ev.sample_count}</strong> test /{' '}
                  <strong className="text-slate-200">{ev.train_sample_count || 0}</strong> train
                </span>
                {ev.notes && <span className="text-slate-500 truncate max-w-xs">{ev.notes}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
