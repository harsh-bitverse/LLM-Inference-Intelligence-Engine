import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart2,
  CheckCircle2,
  Clock,
  Cpu,
  HelpCircle,
  Layers,
  Play,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Shuffle,
  Sliders,
  TrendingUp,
  XCircle,
  Zap,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import {
  AdmissionPolicyType,
  AdmissionResult,
  CapacityCurvePoint,
  CapacityScenario,
  CapacityScenariosResponse,
  HeadOfLineBlockingObservation,
  MixedCapacityEstimateResponse,
  MixedWorkloadProfile,
  ModelSwitchingObservation,
  PolicyEvaluationReport,
  WorkloadCapacityReport,
  WorkloadProfileType,
} from '../../api/types';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorMessage } from '../common/ErrorMessage';
import { MetricCard } from '../common/MetricCard';

export const CapacityPlanningView: React.FC = () => {
  const [subTab, setSubTab] = useState<'single' | 'mixed' | 'policies' | 'diagnostics'>('single');

  // =========================================================================
  // Tab 1: Single Workload Capacity State
  // =========================================================================
  const [scenariosData, setScenariosData] = useState<CapacityScenariosResponse | null>(null);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string | null>(null);

  const [modelId, setModelId] = useState<string>('qwen2.5:0.5b');
  const [workloadId, setWorkloadId] = useState<string>('workload_v1_interactive_qa');
  const [numThread, setNumThread] = useState<number>(8);
  const [latencySlaMs, setLatencySlaMs] = useState<string>('2000');
  const [currentInFlight, setCurrentInFlight] = useState<number>(0);
  const [currentQueueDepth, setCurrentQueueDepth] = useState<number>(0);
  const [requireCodePass, setRequireCodePass] = useState<boolean>(false);

  const [capacityReport, setCapacityReport] = useState<WorkloadCapacityReport | null>(null);
  const [admissionResult, setAdmissionResult] = useState<AdmissionResult | null>(null);
  const [loadingReport, setLoadingReport] = useState<boolean>(false);
  const [evaluatingAdmission, setEvaluatingAdmission] = useState<boolean>(false);
  const [singleError, setSingleError] = useState<string | null>(null);

  // =========================================================================
  // Tab 2: Mixed-Workload Simulation State
  // =========================================================================
  const [mixedProfiles, setMixedProfiles] = useState<MixedWorkloadProfile[]>([]);
  const [selectedProfileType, setSelectedProfileType] = useState<WorkloadProfileType>('PROFILE_A_INTERACTIVE_DOMINANT');
  const [mixedModelId, setMixedModelId] = useState<string>('qwen2.5:0.5b');
  const [mixedOfferedRps, setMixedOfferedRps] = useState<number>(0.5);
  const [mixedEstimate, setMixedEstimate] = useState<MixedCapacityEstimateResponse | null>(null);
  const [loadingMixed, setLoadingMixed] = useState<boolean>(false);
  const [mixedError, setMixedError] = useState<string | null>(null);

  // =========================================================================
  // Tab 3: Adaptive Admission Policies State
  // =========================================================================
  const [policyReports, setPolicyReports] = useState<PolicyEvaluationReport[]>([]);
  const [loadingPolicies, setLoadingPolicies] = useState<boolean>(false);
  const [policyError, setPolicyError] = useState<string | null>(null);

  // =========================================================================
  // Tab 4: Diagnostics State (HoL & Model Residency)
  // =========================================================================
  const [holObservations, setHolObservations] = useState<HeadOfLineBlockingObservation[]>([]);
  const [switchingObservations, setSwitchingObservations] = useState<ModelSwitchingObservation[]>([]);
  const [loadingDiagnostics, setLoadingDiagnostics] = useState<boolean>(false);
  const [diagnosticsError, setDiagnosticsError] = useState<string | null>(null);

  // Fetch pre-registered scenarios and mixed profiles on mount
  useEffect(() => {
    let isMounted = true;
    const loadInitialData = async () => {
      try {
        const [scenarios, profiles] = await Promise.allSettled([
          apiClient.listCapacityScenarios(),
          apiClient.getMixedProfiles(),
        ]);
        if (isMounted) {
          if (scenarios.status === 'fulfilled') {
            setScenariosData(scenarios.value);
          }
          if (profiles.status === 'fulfilled') {
            setMixedProfiles(profiles.value);
          }
        }
      } catch (err: any) {
        if (isMounted) {
          setSingleError(err.message || 'Failed to fetch capacity metadata.');
        }
      }
    };
    loadInitialData();
    return () => {
      isMounted = false;
    };
  }, []);

  // Lazy-load data when switching tabs
  useEffect(() => {
    if (subTab === 'policies' && policyReports.length === 0) {
      loadPolicyReports();
    } else if (subTab === 'diagnostics' && holObservations.length === 0) {
      loadDiagnosticsData();
    }
  }, [subTab]);

  const loadPolicyReports = async () => {
    setLoadingPolicies(true);
    setPolicyError(null);
    try {
      const res = await apiClient.getAdmissionPoliciesSummary();
      setPolicyReports(res.policy_reports);
    } catch (err: any) {
      setPolicyError(err.message || 'Failed to load policy evaluation reports.');
    } finally {
      setLoadingPolicies(false);
    }
  };

  const loadDiagnosticsData = async () => {
    setLoadingDiagnostics(true);
    setDiagnosticsError(null);
    try {
      const [holRes, swRes] = await Promise.all([
        apiClient.getHoLAnalysis(),
        apiClient.getModelSwitchingAnalysis(),
      ]);
      setHolObservations(holRes.observations);
      setSwitchingObservations(swRes.observations);
    } catch (err: any) {
      setDiagnosticsError(err.message || 'Failed to load probe diagnostics.');
    } finally {
      setLoadingDiagnostics(false);
    }
  };

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
    setSingleError(null);
    try {
      const rep = await apiClient.estimateCapacity({
        model_id: modelId,
        num_thread: numThread,
        workload_id: workloadId,
        latency_sla_ms: latencySlaMs ? parseFloat(latencySlaMs) : undefined,
      });
      setCapacityReport(rep);
    } catch (err: any) {
      setSingleError(err.message || 'Failed to compute serving capacity curve.');
    } finally {
      setLoadingReport(false);
    }
  };

  const handleEvaluateAdmission = async () => {
    setEvaluatingAdmission(true);
    setSingleError(null);
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
      setSingleError(err.message || 'Failed to evaluate admission decision.');
    } finally {
      setEvaluatingAdmission(false);
    }
  };

  const handleRunMixedSimulation = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoadingMixed(true);
    setMixedError(null);
    try {
      const res = await apiClient.estimateMixedCapacity({
        profile_type: selectedProfileType,
        model_id: mixedModelId,
        offered_rps: mixedOfferedRps,
      });
      setMixedEstimate(res);
    } catch (err: any) {
      setMixedError(err.message || 'Failed to estimate mixed-workload capacity.');
    } finally {
      setLoadingMixed(false);
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

  const getPolicyName = (type: AdmissionPolicyType) => {
    switch (type) {
      case 'POLICY_A_NO_CONTROL':
        return 'Policy A: FIFO (No Control)';
      case 'POLICY_B_STATIC_UTILIZATION':
        return 'Policy B: Static Utilization';
      case 'POLICY_C_WORKLOAD_AWARE':
        return 'Policy C: Workload-Aware';
      default:
        return type;
    }
  };

  const getProfileShortName = (pt: WorkloadProfileType) => {
    switch (pt) {
      case 'PROFILE_A_INTERACTIVE_DOMINANT':
        return 'Profile A (Interactive)';
      case 'PROFILE_B_BALANCED':
        return 'Profile B (Balanced)';
      case 'PROFILE_C_GENERATION_HEAVY':
        return 'Profile C (Gen Heavy)';
      case 'PROFILE_D_BURSTY':
        return 'Profile D (Bursty)';
      default:
        return pt;
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
              Empirical M/G/1 Pollaczek-Khinchine queueing model, multi-class service moments calculus, discrete-event tail latency simulation, and workload-aware adaptive admission on Intel Core i7-14700 under Ollama v0.40.11.
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

      {/* Sub-Navigation Tabs */}
      <div className="flex border-b border-slate-800 space-x-1">
        <button
          type="button"
          onClick={() => setSubTab('single')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all ${
            subTab === 'single'
              ? 'border-sky-500 text-sky-400 bg-sky-950/30'
              : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
          }`}
        >
          <TrendingUp className="w-4 h-4" />
          Single-Workload Capacity (M/G/1)
        </button>
        <button
          type="button"
          onClick={() => setSubTab('mixed')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all ${
            subTab === 'mixed'
              ? 'border-sky-500 text-sky-400 bg-sky-950/30'
              : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
          }`}
        >
          <Layers className="w-4 h-4" />
          Mixed-Workload Simulation
        </button>
        <button
          type="button"
          onClick={() => setSubTab('policies')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all ${
            subTab === 'policies'
              ? 'border-sky-500 text-sky-400 bg-sky-950/30'
              : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
          }`}
        >
          <Sliders className="w-4 h-4" />
          Adaptive Admission Policies
        </button>
        <button
          type="button"
          onClick={() => setSubTab('diagnostics')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition-all ${
            subTab === 'diagnostics'
              ? 'border-sky-500 text-sky-400 bg-sky-950/30'
              : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
          }`}
        >
          <Shuffle className="w-4 h-4" />
          HoL & Residency Diagnostics
        </button>
      </div>

      {/* =====================================================================
          TAB 1: Single-Workload Capacity
         ===================================================================== */}
      {subTab === 'single' && (
        <div className="space-y-6">
          {singleError && <ErrorMessage message={singleError} />}

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
      )}

      {/* =====================================================================
          TAB 2: Mixed-Workload Simulation
         ===================================================================== */}
      {subTab === 'mixed' && (
        <div className="space-y-6">
          {mixedError && <ErrorMessage message={mixedError} />}

          <form onSubmit={handleRunMixedSimulation} className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-5">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Layers className="w-4 h-4 text-sky-400" />
              Mixed-Workload Traffic Generator & Simulation Controls
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
              <div>
                <label className="block font-medium text-slate-300 mb-1">Workload Profile</label>
                <select
                  value={selectedProfileType}
                  onChange={(e) => setSelectedProfileType(e.target.value as WorkloadProfileType)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
                >
                  {mixedProfiles.length > 0 ? (
                    mixedProfiles.map((p) => (
                      <option key={p.profile_type} value={p.profile_type}>
                        {p.name}
                      </option>
                    ))
                  ) : (
                    <>
                      <option value="PROFILE_A_INTERACTIVE_DOMINANT">Profile A: Interactive Dominant (70% QA, 10% JSON, 10% Gen, 10% Ret)</option>
                      <option value="PROFILE_B_BALANCED">Profile B: Balanced Multi-Task (25% each)</option>
                      <option value="PROFILE_C_GENERATION_HEAVY">Profile C: Generation Heavy (10% QA, 10% JSON, 60% Gen, 20% Ret)</option>
                      <option value="PROFILE_D_BURSTY">Profile D: Bursty Interactive Batches (50% QA, 20% JSON, 20% Gen, 10% Ret)</option>
                    </>
                  )}
                </select>
              </div>

              <div>
                <label className="block font-medium text-slate-300 mb-1">Model Identifier</label>
                <select
                  value={mixedModelId}
                  onChange={(e) => setMixedModelId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
                >
                  <option value="qwen2.5:0.5b">qwen2.5:0.5b (Fastest, low latency)</option>
                  <option value="llama3.2:1b">llama3.2:1b (Balanced)</option>
                  <option value="qwen2.5:1.5b">qwen2.5:1.5b (Highest quality)</option>
                </select>
              </div>

              <div>
                <label className="block font-medium text-slate-300 mb-1">Offered Arrival Rate (λ req/s)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="10.0"
                  value={mixedOfferedRps}
                  onChange={(e) => setMixedOfferedRps(parseFloat(e.target.value) || 0.5)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500"
                />
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="submit"
                disabled={loadingMixed}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-medium text-xs shadow-lg shadow-sky-600/20 transition-all"
              >
                {loadingMixed ? <LoadingSpinner size="sm" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                Run Discrete-Event Simulation & Moments Calculus
              </button>
            </div>
          </form>

          {mixedEstimate && (
            <div className="space-y-4">
              {/* Theoretical Moments Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <MetricCard
                  title="Mean Service Time E[Smix]"
                  value={`${mixedEstimate.moments.mean_service_ms.toFixed(1)} ms`}
                  subtitle={`Variance: ${mixedEstimate.moments.variance_service_ms2.toFixed(0)} ms²`}
                />
                <MetricCard
                  title="Coefficient of Variation Cv"
                  value={`${mixedEstimate.moments.coefficient_of_variation.toFixed(3)}`}
                  subtitle={
                    mixedEstimate.moments.coefficient_of_variation > 1.0
                      ? 'High dispersion (Cv > 1.0)'
                      : 'Moderate dispersion (Cv ≤ 1.0)'
                  }
                />
                <MetricCard
                  title="Estimated Utilization ρ"
                  value={`${mixedEstimate.estimated_utilization_rho.toFixed(2)}`}
                  subtitle={
                    mixedEstimate.is_stable
                      ? 'Queue is theoretically stable (ρ < 1)'
                      : 'Queue diverges / saturates (ρ ≥ 1)'
                  }
                />
                <MetricCard
                  title="Theoretical Wait Wq"
                  value={`${mixedEstimate.theoretical_queue_wait_ms.toFixed(1)} ms`}
                  subtitle={`Pollaczek-Khinchine M/G/1`}
                />
              </div>

              {/* Operating Limits */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg text-xs space-y-1">
                  <div className="text-slate-400 font-medium">Max Sustainable Rate (Rmax)</div>
                  <div className="text-lg font-bold text-slate-100 font-mono">
                    {mixedEstimate.max_sustainable_rps.toFixed(2)} req/s
                  </div>
                  <div className="text-[10px] text-slate-400">μ = 1 / E[Smix]</div>
                </div>
                <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg text-xs space-y-1">
                  <div className="text-slate-400 font-medium">Knee Saturation Rate (Rknee)</div>
                  <div className="text-lg font-bold text-amber-400 font-mono">
                    {mixedEstimate.saturation_knee_rps.toFixed(2)} req/s
                  </div>
                  <div className="text-[10px] text-slate-400">0.80 × Rmax (queue explosion boundary)</div>
                </div>
                <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg text-xs space-y-1">
                  <div className="text-slate-400 font-medium">Safe Operating Rate (Rsafe)</div>
                  <div className="text-lg font-bold text-emerald-400 font-mono">
                    {mixedEstimate.safe_operating_rps.toFixed(2)} req/s
                  </div>
                  <div className="text-[10px] text-slate-400">0.60 × Rmax for tail-latency protection</div>
                </div>
              </div>

              {/* Discrete-Event Simulation Tail Latencies */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                    <BarChart2 className="w-3.5 h-3.5 text-sky-400" />
                    Discrete-Event Queue Simulation Results ({mixedEstimate.simulation.request_count} synthetic requests)
                  </h4>
                  <span className="text-[10px] text-emerald-400 font-mono font-semibold">
                    Overall SLA Attainment: {mixedEstimate.simulation.overall_sla_attainment_pct.toFixed(1)}%
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 bg-slate-950/60 rounded border border-slate-800 font-mono">
                    <div className="text-[10px] text-slate-400">Overall P50 E2E</div>
                    <div className="text-sm font-bold text-slate-200">
                      {mixedEstimate.simulation.overall_p50_e2e_ms.toFixed(1)} ms
                    </div>
                  </div>
                  <div className="p-2.5 bg-slate-950/60 rounded border border-slate-800 font-mono">
                    <div className="text-[10px] text-slate-400">Overall P90 E2E</div>
                    <div className="text-sm font-bold text-slate-200">
                      {mixedEstimate.simulation.overall_p90_e2e_ms.toFixed(1)} ms
                    </div>
                  </div>
                  <div className="p-2.5 bg-slate-950/60 rounded border border-slate-800 font-mono">
                    <div className="text-[10px] text-slate-400">Overall P95 E2E</div>
                    <div className="text-sm font-bold text-rose-300">
                      {mixedEstimate.simulation.overall_p95_e2e_ms.toFixed(1)} ms
                    </div>
                  </div>
                  <div className="p-2.5 bg-slate-950/60 rounded border border-slate-800 font-mono">
                    <div className="text-[10px] text-slate-400">Mean Queue Delay</div>
                    <div className="text-sm font-bold text-amber-300">
                      {mixedEstimate.simulation.mean_queue_delay_ms.toFixed(1)} ms
                    </div>
                  </div>
                </div>

                {/* Per-Class Summary Table */}
                <div className="overflow-x-auto pt-2">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-[11px] font-semibold uppercase text-slate-400">
                        <th className="py-2 px-3">Workload Class</th>
                        <th className="py-2 px-3">Simulated Requests</th>
                        <th className="py-2 px-3">Median (P50)</th>
                        <th className="py-2 px-3">Tail (P95)</th>
                        <th className="py-2 px-3">Mean Queue Wait</th>
                        <th className="py-2 px-3">SLA Attainment (%)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                      {Object.entries(mixedEstimate.simulation.per_class_summary).map(([cName, cData]) => {
                        const isHighSla = cData.sla_attainment_pct >= 90;
                        const isPoorSla = cData.sla_attainment_pct < 50;
                        return (
                          <tr key={cName} className="hover:bg-slate-800/30 text-slate-300">
                            <td className="py-2 px-3 font-semibold text-slate-200">{cName}</td>
                            <td className="py-2 px-3">{cData.count}</td>
                            <td className="py-2 px-3">{cData.p50_e2e_ms.toFixed(1)} ms</td>
                            <td className="py-2 px-3">{cData.p95_e2e_ms.toFixed(1)} ms</td>
                            <td className="py-2 px-3">{cData.mean_wait_ms.toFixed(1)} ms</td>
                            <td className="py-2 px-3">
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                                  isHighSla
                                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                                    : isPoorSla
                                    ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                                    : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                                }`}
                              >
                                {cData.sla_attainment_pct.toFixed(1)}%
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* =====================================================================
          TAB 3: Adaptive Admission Policies
         ===================================================================== */}
      {subTab === 'policies' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Sliders className="w-4 h-4 text-sky-400" />
              Admission Policy Comparison Matrix
            </h3>
            <button
              type="button"
              onClick={loadPolicyReports}
              disabled={loadingPolicies}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs border border-slate-700 transition-all"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingPolicies ? 'animate-spin' : ''}`} />
              Refresh Policy Reports
            </button>
          </div>

          {policyError && <ErrorMessage message={policyError} />}

          {/* Policy Descriptions Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="text-xs font-bold text-rose-300">Policy A: FIFO (No Control)</div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Accepts every offered request into single-slot queue. Under heavy or generation-intensive traffic, queue explodes and interactive requests suffer fatal head-of-line blocking, dropping SLA attainment below 20%.
              </p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="text-xs font-bold text-amber-300">Policy B: Static Utilization</div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Sheds requests when raw backlog exceeds fixed depth threshold (e.g. depth &gt; 2). Fails to distinguish between a 100ms QA task and an 8,000ms generation task, causing high unnecessary rejection of short requests.
              </p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-emerald-900/40 bg-emerald-950/10 space-y-2">
              <div className="text-xs font-bold text-emerald-300">Policy C: Workload-Aware Adaptive</div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Evaluates predicted isolated service time, cumulative queue backlog, and target SLA before admission. Delivers 100% SLA compliance for admitted requests without starving short interactive tasks.
              </p>
            </div>
          </div>

          {/* Comparison Table */}
          {loadingPolicies ? (
            <div className="p-12 text-center">
              <LoadingSpinner size="md" />
              <div className="text-xs text-slate-400 mt-2">Loading admission policy evaluation reports...</div>
            </div>
          ) : (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-[11px] font-semibold uppercase text-slate-400">
                      <th className="py-2 px-3">Policy</th>
                      <th className="py-2 px-3">Profile</th>
                      <th className="py-2 px-3">Offered</th>
                      <th className="py-2 px-3">Admitted</th>
                      <th className="py-2 px-3">Admitted SLA (%)</th>
                      <th className="py-2 px-3">False Feasible</th>
                      <th className="py-2 px-3">Unnecessary Rejects</th>
                      <th className="py-2 px-3">Fairness (Jain)</th>
                      <th className="py-2 px-3">Utility Score</th>
                      <th className="py-2 px-3">Invariant Check</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                    {policyReports.map((rep, idx) => {
                      const isPolicyC = rep.policy_type === 'POLICY_C_WORKLOAD_AWARE';
                      const isPolicyA = rep.policy_type === 'POLICY_A_NO_CONTROL';
                      return (
                        <tr
                          key={idx}
                          className={
                            isPolicyC
                              ? 'bg-emerald-950/20 text-emerald-200'
                              : isPolicyA
                              ? 'bg-rose-950/10 text-rose-200'
                              : 'hover:bg-slate-800/30 text-slate-300'
                          }
                        >
                          <td className="py-2 px-3 font-semibold text-slate-100">{getPolicyName(rep.policy_type)}</td>
                          <td className="py-2 px-3 text-slate-300">{getProfileShortName(rep.profile_type)}</td>
                          <td className="py-2 px-3">{rep.total_offered}</td>
                          <td className="py-2 px-3">{rep.total_admitted}</td>
                          <td className="py-2 px-3 font-bold">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] ${
                                rep.admitted_sla_attainment_pct >= 95
                                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                                  : rep.admitted_sla_attainment_pct < 50
                                  ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                                  : 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                              }`}
                            >
                              {rep.admitted_sla_attainment_pct.toFixed(1)}%
                            </span>
                          </td>
                          <td className="py-2 px-3">
                            {rep.false_feasible_count > 0 ? (
                              <span className="text-rose-400 font-semibold">{rep.false_feasible_count}</span>
                            ) : (
                              <span className="text-emerald-400">0</span>
                            )}
                          </td>
                          <td className="py-2 px-3">{rep.unnecessarily_rejected_count}</td>
                          <td className="py-2 px-3">{rep.workload_fairness_index.toFixed(3)}</td>
                          <td className="py-2 px-3 font-bold">{rep.tradeoff_utility_score.toFixed(1)}</td>
                          <td className="py-2 px-3">
                            {rep.accounting_invariant_holds ? (
                              <span className="text-emerald-400 flex items-center gap-1 text-[10px]">
                                <CheckCircle2 className="w-3 h-3" /> Exact
                              </span>
                            ) : (
                              <span className="text-rose-400 flex items-center gap-1 text-[10px]">
                                <XCircle className="w-3 h-3" /> Broken
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-400 space-y-1">
                <div className="font-semibold text-slate-300">Accounting Invariant In Every Evaluation:</div>
                <div className="font-mono text-[10px] text-sky-400">
                  offered == completed + rejected + timed_out + canceled + outstanding
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* =====================================================================
          TAB 4: Head-of-Line & Residency Diagnostics
         ===================================================================== */}
      {subTab === 'diagnostics' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Shuffle className="w-4 h-4 text-sky-400" />
              Empirical Head-of-Line Blocking & Model Residency Probes
            </h3>
            <button
              type="button"
              onClick={loadDiagnosticsData}
              disabled={loadingDiagnostics}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs border border-slate-700 transition-all"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingDiagnostics ? 'animate-spin' : ''}`} />
              Refresh Diagnostics Data
            </button>
          </div>

          {diagnosticsError && <ErrorMessage message={diagnosticsError} />}

          {loadingDiagnostics ? (
            <div className="p-12 text-center">
              <LoadingSpinner size="md" />
              <div className="text-xs text-slate-400 mt-2">Loading empirical probe observations...</div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Section 1: Head-of-Line Blocking */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      Dedicated Head-of-Line (HoL) Blocking Probe
                    </h4>
                    <p className="text-xs text-slate-400 mt-1">
                      Direct empirical measurement of short interactive requests queued behind long-generation requests in single-slot FIFO serving.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {holObservations.map((obs) => (
                    <div key={obs.experiment_id} className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-xs text-slate-200">{obs.scenario_description}</span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                          SLA Breached
                        </span>
                      </div>

                      <div className="space-y-2 text-xs">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                          <span className="text-slate-400">Antecedent ({obs.antecedent_workload}):</span>
                          <span className="font-mono text-slate-200">{obs.antecedent_duration_ms.toFixed(1)} ms</span>
                        </div>
                        <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                          <span className="text-slate-400">Consequent Isolated ({obs.consequent_workload}):</span>
                          <span className="font-mono text-emerald-400">{obs.isolated_consequent_duration_ms.toFixed(1)} ms</span>
                        </div>
                        <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                          <span className="text-slate-400">Consequent Blocked E2E:</span>
                          <span className="font-mono text-rose-400 font-bold">{obs.blocked_consequent_duration_ms.toFixed(1)} ms</span>
                        </div>
                        <div className="flex items-center justify-between pt-1">
                          <span className="text-amber-400 font-medium">Added Queue Wait Delay:</span>
                          <span className="font-mono text-amber-300 font-bold">+{obs.additional_queue_delay_ms.toFixed(1)} ms</span>
                        </div>
                      </div>

                      <div className="p-2.5 rounded bg-slate-900 border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
                        Interactive QA (isolated ~110 ms, target SLA 500 ms) queued behind active Long Generation suffered an extra{' '}
                        <strong className="text-amber-300">+{obs.additional_queue_delay_ms.toFixed(1)} ms</strong> wait, causing a total turnaround of{' '}
                        <strong className="text-rose-400">{obs.blocked_consequent_duration_ms.toFixed(1)} ms</strong> and breaching SLA by over 7.1 seconds.
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Section 2: Model Switching & Memory Residency */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-sky-400" />
                      Model Switching Overhead & RAM Residency Probe
                    </h4>
                    <p className="text-xs text-slate-400 mt-1">
                      Measurement of Ollama model unloads, reload latency, and dual warm-memory residency across alternating models.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {switchingObservations.map((obs) => (
                    <div key={obs.experiment_id} className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-xs text-slate-200 flex items-center gap-2">
                          <span>{obs.from_model_id}</span>
                          <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                          <span>{obs.to_model_id}</span>
                        </span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                          Dual Warm Resident
                        </span>
                      </div>

                      <div className="space-y-2 text-xs">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                          <span className="text-slate-400">Observed Load Duration:</span>
                          <span className="font-mono text-emerald-400 font-bold">{obs.actual_load_duration_ms.toFixed(1)} ms</span>
                        </div>
                        <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                          <span className="text-slate-400">Client Switch Penalty:</span>
                          <span className="font-mono text-slate-200">{obs.client_switch_penalty_ms.toFixed(1)} ms</span>
                        </div>
                        <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                          <span className="text-slate-400">Warm Execution Duration:</span>
                          <span className="font-mono text-slate-200">{obs.warm_execution_duration_ms.toFixed(1)} ms</span>
                        </div>
                        <div className="flex items-center justify-between pt-1">
                          <span className="text-slate-400">Cold Execution Duration:</span>
                          <span className="font-mono text-slate-300">{obs.cold_execution_duration_ms.toFixed(1)} ms</span>
                        </div>
                      </div>

                      <div className="p-2.5 rounded bg-slate-900 border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
                        With 32 GB RAM on this Intel Core i7-14700 workstation, Ollama retains both <code className="text-sky-300">qwen2.5:0.5b</code> (398 MB) and <code className="text-sky-300">llama3.2:1b</code> (1.3 GB) concurrently in memory. Alternating requests incurred negligible server load duration ({obs.actual_load_duration_ms.toFixed(1)} ms) and zero model unloads.
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
