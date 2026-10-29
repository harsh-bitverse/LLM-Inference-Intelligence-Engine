import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  HelpCircle,
  Play,
  ShieldAlert,
  ShieldCheck,
  TrendingUp,
  XCircle,
  Zap,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import {
  AdmissionResult,
  CapacityCurvePoint,
  CapacityScenario,
  CapacityScenariosResponse,
  WorkloadCapacityReport,
} from '../../api/types';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorMessage } from '../common/ErrorMessage';
import { MetricCard } from '../common/MetricCard';

export const CapacityPlanningView: React.FC = () => {
  // Scenario and Manifest State
  const [scenariosData, setScenariosData] = useState<CapacityScenariosResponse | null>(null);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string | null>(null);

  // Form State
  const [modelId, setModelId] = useState<string>('qwen2.5:0.5b');
  const [workloadId, setWorkloadId] = useState<string>('workload_v1_interactive_qa');
  const [numThread, setNumThread] = useState<number>(8);
  const [latencySlaMs, setLatencySlaMs] = useState<string>('2000');
  const [currentInFlight, setCurrentInFlight] = useState<number>(0);
  const [currentQueueDepth, setCurrentQueueDepth] = useState<number>(0);
  const [requireCodePass, setRequireCodePass] = useState<boolean>(false);

  // Results State
  const [capacityReport, setCapacityReport] = useState<WorkloadCapacityReport | null>(null);
  const [admissionResult, setAdmissionResult] = useState<AdmissionResult | null>(null);
  const [loadingReport, setLoadingReport] = useState<boolean>(false);
  const [evaluatingAdmission, setEvaluatingAdmission] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch pre-registered scenarios on mount
  useEffect(() => {
    let isMounted = true;
    const loadScenarios = async () => {
      try {
        const data = await apiClient.listCapacityScenarios();
        if (isMounted) {
          setScenariosData(data);
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Failed to fetch capacity scenarios.');
        }
      }
    };
    loadScenarios();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSelectScenario = (sc: CapacityScenario) => {
    setSelectedScenarioId(sc.scenario_id);
    setModelId(sc.model_id);
    setLatencySlaMs(String(sc.target_sla_ms));
    if (sc.scenario_id === 'SCENARIO_A_LOW_TRAFFIC') {
      setCurrentInFlight(0);
      setCurrentQueueDepth(0);
    } else if (sc.scenario_id === 'SCENARIO_B_NEAR_SATURATION') {
      setCurrentInFlight(1);
      setCurrentQueueDepth(2);
    } else {
      setCurrentInFlight(1);
      setCurrentQueueDepth(4);
    }
  };

  const handleGenerateCapacityReport = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoadingReport(true);
    setError(null);
    try {
      const rep = await apiClient.estimateCapacity({
        model_id: modelId,
        num_thread: numThread,
        workload_id: workloadId,
        latency_sla_ms: latencySlaMs ? parseFloat(latencySlaMs) : undefined,
      });
      setCapacityReport(rep);
    } catch (err: any) {
      setError(err.message || 'Failed to compute serving capacity curve.');
    } finally {
      setLoadingReport(false);
    }
  };

  const handleEvaluateAdmission = async () => {
    setEvaluatingAdmission(true);
    setError(null);
    try {
      const res = await apiClient.admitRequest({
        model_id: modelId,
        num_thread: numThread,
        prompt_tokens: 30,
        expected_output_tokens: 20,
        latency_sla_ms: latencySlaMs ? parseFloat(latencySlaMs) : undefined,
        current_in_flight: currentInFlight,
        current_queue_depth: currentQueueDepth,
        require_code_execution_pass: requireCodePass,
      });
      setAdmissionResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to evaluate admission decision.');
    } finally {
      setEvaluatingAdmission(false);
    }
  };

  const getDecisionBadge = (decision: string) => {
    switch (decision) {
      case 'ACCEPT':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <CheckCircle2 className="w-3.5 h-3.5" />
            ACCEPT — Immediate Execution Slot
          </span>
        );
      case 'QUEUE_WITH_BOUNDED_WAIT':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
            <Clock className="w-3.5 h-3.5" />
            QUEUE WITH BOUNDED WAIT — Headroom Monitored
          </span>
        );
      case 'REJECT_OR_DEFER':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/30">
            <XCircle className="w-3.5 h-3.5" />
            REJECT OR DEFER — SLA Breach Inevitable
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-500/10 text-slate-400 border border-slate-500/30">
            <HelpCircle className="w-3.5 h-3.5" />
            INSUFFICIENT EVIDENCE — Uncalibrated Profile
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-sky-400" />
              <h2 className="text-base font-bold text-slate-100">
                Inference Serving Capacity Planning & Adaptive Request Admission
              </h2>
            </div>
            <p className="text-xs text-slate-400 max-w-3xl leading-relaxed">
              Empirical M/G/1 Pollaczek-Khinchine queueing model calibrated on Intel Core i7-14700 under Ollama v0.40.11.
              Accounts for single-slot server queueing serialization, knee saturation detection, and conservative SLA guardrails.
            </p>
          </div>
          {scenariosData?.has_cached_manifest && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-sky-950/60 border border-sky-800/60 text-sky-300 text-xs">
              <ShieldCheck className="w-4 h-4 text-sky-400" />
              <span>Manifest Active: {scenariosData.total_observations} Observations</span>
            </div>
          )}
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {/* Standard Evaluation Scenarios */}
      {scenariosData?.scenarios && (
        <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            Pre-Registered Operational Scenarios (Prospective Validation)
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {scenariosData.scenarios.map((sc) => (
              <button
                key={sc.scenario_id}
                type="button"
                onClick={() => handleSelectScenario(sc)}
                className={`text-left p-3.5 rounded-lg border transition-all text-xs ${
                  selectedScenarioId === sc.scenario_id
                    ? 'bg-sky-950/60 border-sky-500 text-sky-200 shadow-md shadow-sky-950/50'
                    : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="font-semibold text-slate-200 mb-1 flex items-center justify-between">
                  <span>{sc.name}</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                    {sc.expected_decision}
                  </span>
                </div>
                <div className="text-slate-400 text-[11px] leading-relaxed mb-2">{sc.description}</div>
                <div className="flex items-center gap-3 text-[10px] text-slate-400 font-mono">
                  <span>Target SLA: {sc.target_sla_ms}ms</span>
                  <span>Model: {sc.model_id}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Interactive Controls Form */}
      <form onSubmit={handleGenerateCapacityReport} className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-5">
        <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-sky-400" />
          Workload Configuration & Server Queue State
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          {/* Model */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Model Identifier</label>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option value="qwen2.5:0.5b">qwen2.5:0.5b (Q4_K_M)</option>
              <option value="llama3.2:1b">llama3.2:1b (Q8_0)</option>
              <option value="qwen2.5:1.5b">qwen2.5:1.5b (Q4_K_M)</option>
            </select>
          </div>

          {/* Workload Task */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Workload Task</label>
            <select
              value={workloadId}
              onChange={(e) => setWorkloadId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option value="workload_v1_interactive_qa">Interactive QA (Short Burst)</option>
              <option value="workload_v1_structured_json">Structured JSON Extraction</option>
              <option value="workload_v1_long_form_generation">Long-Form Generation</option>
              <option value="workload_v1_long_context_retrieval">Long Context Retrieval</option>
            </select>
          </div>

          {/* CPU Threads */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">CPU Threads: {numThread}</label>
            <input
              type="range"
              min={2}
              max={28}
              step={2}
              value={numThread}
              onChange={(e) => setNumThread(parseInt(e.target.value, 10))}
              className="w-full accent-sky-500 mt-2"
            />
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>2 threads</span>
              <span className="text-sky-400 font-semibold">8 (Peak Decode)</span>
              <span>28 threads</span>
            </div>
          </div>

          {/* Latency SLA */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Target Latency SLA (ms)</label>
            <input
              type="number"
              value={latencySlaMs}
              onChange={(e) => setLatencySlaMs(e.target.value)}
              placeholder="e.g. 2000"
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          {/* Server Queue Depth */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Current Queue Backlog</label>
            <input
              type="number"
              min={0}
              max={20}
              value={currentQueueDepth}
              onChange={(e) => setCurrentQueueDepth(parseInt(e.target.value, 10) || 0)}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>

          {/* In-Flight */}
          <div>
            <label className="block font-medium text-slate-300 mb-1">Current In-Flight Slot</label>
            <select
              value={currentInFlight}
              onChange={(e) => setCurrentInFlight(parseInt(e.target.value, 10))}
              className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option value={0}>0 (Idle Slot Available)</option>
              <option value={1}>1 (Active Inference In-Flight)</option>
            </select>
          </div>

          {/* Session 23 Fail-Closed Code Guardrail */}
          <div className="flex items-center gap-2 pt-5">
            <input
              type="checkbox"
              id="requireCodePass"
              checked={requireCodePass}
              onChange={(e) => setRequireCodePass(e.target.checked)}
              className="rounded bg-slate-950 border-slate-800 text-sky-500 focus:ring-0"
            />
            <label htmlFor="requireCodePass" className="text-slate-300 cursor-pointer text-xs">
              Verify Code Execution Security
            </label>
          </div>
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={handleEvaluateAdmission}
            disabled={evaluatingAdmission}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 font-medium text-xs border border-slate-700 transition-all"
          >
            {evaluatingAdmission ? <LoadingSpinner size="sm" /> : <ShieldCheck className="w-3.5 h-3.5" />}
            Evaluate Live Request Admission
          </button>
          <button
            type="submit"
            disabled={loadingReport}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-medium text-xs shadow-lg shadow-sky-600/20 transition-all"
          >
            {loadingReport ? <LoadingSpinner size="sm" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            Generate M/G/1 Capacity Curve
          </button>
        </div>
      </form>

      {/* Admission Decision Card */}
      {admissionResult && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-sky-400" />
              <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Admission Controller Verdict
              </h3>
            </div>
            {getDecisionBadge(admissionResult.decision)}
          </div>

          {admissionResult.unverified_code_warning && (
            <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 shrink-0 text-amber-400" />
              <span>
                <strong>Security Mode Warning:</strong> Model-generated code correctness is unverified under fail-closed security mode.
              </span>
            </div>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard
              title="Predicted Queue Delay"
              value={`${admissionResult.predicted_queue_delay_ms.toFixed(1)} ms`}
              subtitle="Single-slot server wait"
            />
            <MetricCard
              title="Isolated Service Time"
              value={`${admissionResult.predicted_service_time_ms.toFixed(1)} ms`}
              subtitle="Prefill + Autoregressive Decode"
            />
            <MetricCard
              title="Predicted Total Latency"
              value={`${admissionResult.predicted_total_latency_ms.toFixed(1)} ms`}
              subtitle={`Target SLA: ${latencySlaMs || 'None'} ms`}
            />
            <MetricCard
              title="SLA Compliance Probability"
              value={`${(admissionResult.sla_compliance_probability * 100).toFixed(0)}%`}
              subtitle={`Headroom: ${admissionResult.capacity_headroom_pct.toFixed(0)}%`}
            />
          </div>

          <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg text-xs text-slate-300 leading-relaxed font-mono">
            {admissionResult.rationale}
          </div>
        </div>
      )}

      {/* Capacity Planning Curve & Serving Metrics */}
      {capacityReport && (
        <div className="space-y-4">
          {/* Top Capacity Summary Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard
              title="Max Sustainable Throughput"
              value={`${capacityReport.max_sustainable_rps.toFixed(2)} req/s`}
              subtitle={`${capacityReport.max_sustainable_rpm.toFixed(1)} req/min (μ = 1/E[S])`}
            />
            <MetricCard
              title="Saturation Knee Rate"
              value={`${capacityReport.saturation_knee_rps.toFixed(2)} req/s`}
              subtitle="Queue diverges rapidly beyond knee"
            />
            <MetricCard
              title="Safe Operating Target"
              value={`${capacityReport.safe_operating_rps.toFixed(2)} req/s`}
              subtitle="~75% of knee for bounded tail latency"
            />
            <MetricCard
              title="Mean Isolated E[S]"
              value={`${capacityReport.service_distribution.mean_service_ms.toFixed(1)} ms`}
              subtitle={`Cv: ${capacityReport.service_distribution.coefficient_of_variation.toFixed(3)} (P95: ${capacityReport.service_distribution.p95_service_ms.toFixed(1)}ms)`}
            />
          </div>

          {/* Capacity Curve Table */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-sky-400" />
                M/G/1 Pollaczek-Khinchine Capacity Trajectory
              </h4>
              <span className="text-[10px] text-slate-400 font-mono">
                Model: {capacityReport.model_id} ({capacityReport.num_thread} threads)
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-[11px] font-semibold uppercase text-slate-400">
                    <th className="py-2 px-3">Offered Rate (λ)</th>
                    <th className="py-2 px-3">Throughput</th>
                    <th className="py-2 px-3">Utilization (ρ)</th>
                    <th className="py-2 px-3">Median E2E</th>
                    <th className="py-2 px-3">P95 Tail</th>
                    <th className="py-2 px-3">Queue Wait</th>
                    <th className="py-2 px-3">SLA Met (%)</th>
                    <th className="py-2 px-3">Backlog State</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                  {capacityReport.capacity_curve.map((pt: CapacityCurvePoint, idx: number) => {
                    const isOverload = pt.estimated_utilization >= 1.0;
                    const isNearKnee = pt.estimated_utilization >= 0.80 && pt.estimated_utilization < 1.0;
                    return (
                      <tr
                        key={idx}
                        className={
                          isOverload
                            ? 'bg-rose-950/20 text-rose-300'
                            : isNearKnee
                            ? 'bg-amber-950/20 text-amber-300'
                            : 'hover:bg-slate-800/30 text-slate-300'
                        }
                      >
                        <td className="py-2 px-3 font-semibold">{pt.offered_rps.toFixed(2)} req/s</td>
                        <td className="py-2 px-3">{pt.completed_rps.toFixed(2)} req/s</td>
                        <td className="py-2 px-3">{pt.estimated_utilization.toFixed(2)}</td>
                        <td className="py-2 px-3">{pt.median_e2e_ms.toFixed(1)} ms</td>
                        <td className="py-2 px-3">{pt.p95_e2e_ms.toFixed(1)} ms</td>
                        <td className="py-2 px-3">{pt.mean_queue_delay_ms.toFixed(1)} ms</td>
                        <td className="py-2 px-3">{pt.sla_compliance_pct.toFixed(1)}%</td>
                        <td className="py-2 px-3">
                          {pt.queue_buildup_detected ? (
                            <span className="text-rose-400 font-semibold flex items-center gap-1">
                              <AlertTriangle className="w-3 h-3" /> Backlog Growing
                            </span>
                          ) : isNearKnee ? (
                            <span className="text-amber-400">Near Saturation</span>
                          ) : (
                            <span className="text-emerald-400">Stable</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Modeling Assumptions Footnote */}
            <div className="pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 space-y-1">
              <div className="font-semibold text-slate-300">Explicit Modeling Assumptions:</div>
              <ul className="list-disc list-inside space-y-0.5 text-slate-400">
                {capacityReport.assumptions.map((asm, i) => (
                  <li key={i}>{asm}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
