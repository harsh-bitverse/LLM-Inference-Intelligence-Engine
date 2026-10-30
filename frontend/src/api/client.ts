/**
 * Centralized, typed API client for the LLM Inference Intelligence Engine REST API.
 * 
 * Handles error normalization, request abort signals, and live SSE telemetry streaming.
 */

import {
  BenchmarkSubmissionRequest,
  BenchmarkSuiteResult,
  DiagnosticReport,
  ErrorDetail,
  EvaluationResult,
  GPUSpecification,
  HardwareCapabilityReport,
  HealthResponse,
  JobStatus,
  ModelSpec,
  OptimizationHistoryItem,
  OptimizationRequest,
  OptimizationResult,
  PaginatedResponse,
  TokenStreamEvent,
  BenchmarkWorkloadTask,
  CandidateModelSpec,
  MultiModelCompareRequest,
  MultiModelComparisonReport,
  ValidationRequest,
  ValidationResponse,
  ValidationScenario,
  WorkloadRecommendationRequest,
  WorkloadRecommendationResponse,
  RuntimeRecommendationRequest,
  RuntimeRecommendationResult,
  RuntimeScenario,
  RuntimeValidationRequest,
  RuntimeValidationResponse,
  CapacityEstimateRequest,
  WorkloadCapacityReport,
  AdmissionRequest,
  AdmissionResult,
  CapacityScenariosResponse,
  ScenarioValidationResponse,
  MixedWorkloadProfile,
  MixedCapacityEstimateRequest,
  MixedCapacityEstimateResponse,
  HeadOfLineBlockingObservation,
  ModelSwitchingObservation,
  PolicyEvaluationReport,
  ServingCapabilitiesResponse,
  ServingComparisonsResponse,
  ServingHolComparisonsResponse,
  ArchitectureRecommendationRequest,
  ServingArchitectureRecommendation,
} from './types';

export class ApiError extends Error {
  public readonly statusCode: number;
  public readonly errorDetail?: ErrorDetail;

  constructor(statusCode: number, message: string, errorDetail?: ErrorDetail) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.errorDetail = errorDetail;
  }
}

export class EngineApiClient {
  private readonly baseUrl: string;

  constructor(baseUrl: string = import.meta.env.VITE_API_BASE_URL || '') {
    // Strip trailing slashes
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
    const headers: Record<string, string> = {
      Accept: 'application/json',
      ...((options.headers as Record<string, string>) || {}),
    };

    if (options.body && !(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }

    let response: Response;
    try {
      response = await fetch(url, {
        ...options,
        headers,
      });
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw err;
      }
      const msg = err instanceof Error ? err.message : 'Network connection failed';
      throw new ApiError(0, `Unable to reach inference engine backend: ${msg}`);
    }

    if (!response.ok) {
      let detail: ErrorDetail | undefined;
      let errMsg = `Request failed with status ${response.status}`;
      try {
        const json = await response.json();
        if (json && typeof json === 'object') {
          const rec = json as Record<string, unknown>;
          if ('error' in rec) {
            const errorVal = String(rec.error);
            const messageVal =
              'message' in rec
                ? String(rec.message)
                : typeof rec.detail === 'string'
                ? rec.detail
                : `Error: ${errorVal}`;
            detail = {
              error: errorVal,
              message: messageVal,
              detail: rec.detail,
            };
            errMsg = messageVal;
          } else if ('detail' in rec) {
            const detailStr = typeof rec.detail === 'string' ? rec.detail : JSON.stringify(rec.detail);
            detail = {
              error: 'HTTP_ERROR',
              message: detailStr,
              detail: rec.detail,
            };
            errMsg = detailStr;
          }
        }
      } catch {
        // Response was not JSON
      }
      throw new ApiError(response.status, errMsg, detail);
    }

    return (await response.json()) as T;
  }

  // --------------------------------------------------------------------------
  // Health Probe
  // --------------------------------------------------------------------------

  public async checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health', { method: 'GET', signal });
  }

  // --------------------------------------------------------------------------
  // Hardware Discovery
  // --------------------------------------------------------------------------

  public async getHardware(discover: boolean = false, signal?: AbortSignal): Promise<HardwareCapabilityReport> {
    const query = discover ? '?discover=true' : '';
    return this.request<HardwareCapabilityReport>(`/api/v1/hardware${query}`, { method: 'GET', signal });
  }

  public async listHardwareReports(
    limit: number = 50,
    offset: number = 0,
    signal?: AbortSignal
  ): Promise<PaginatedResponse<HardwareCapabilityReport>> {
    return this.request<PaginatedResponse<HardwareCapabilityReport>>(
      `/api/v1/hardware/reports?limit=${limit}&offset=${offset}`,
      { method: 'GET', signal }
    );
  }

  public async getHardwareById(reportId: string, signal?: AbortSignal): Promise<HardwareCapabilityReport> {
    return this.request<HardwareCapabilityReport>(`/api/v1/hardware/${encodeURIComponent(reportId)}`, {
      method: 'GET',
      signal,
    });
  }

  public async getHardwareCatalog(signal?: AbortSignal): Promise<GPUSpecification[]> {
    return this.request<GPUSpecification[]>('/api/v1/hardware/catalog', {
      method: 'GET',
      signal,
    });
  }

  // --------------------------------------------------------------------------
  // Benchmarking and Jobs
  // --------------------------------------------------------------------------

  public async submitBenchmark(
    request: BenchmarkSubmissionRequest,
    signal?: AbortSignal
  ): Promise<{ job_id: string; status: string; message: string }> {
    return this.request<{ job_id: string; status: string; message: string }>('/api/v1/benchmarks/run', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  public async listJobs(limit: number = 50, signal?: AbortSignal): Promise<JobStatus[]> {
    return this.request<JobStatus[]>(`/api/v1/benchmarks/jobs?limit=${limit}`, {
      method: 'GET',
      signal,
    });
  }

  public async getJobStatus(
    jobId: string,
    signal?: AbortSignal
  ): Promise<{ job: JobStatus; result: BenchmarkSuiteResult | null }> {
    return this.request<{ job: JobStatus; result: BenchmarkSuiteResult | null }>(
      `/api/v1/benchmarks/jobs/${encodeURIComponent(jobId)}`,
      { method: 'GET', signal }
    );
  }

  public async cancelJob(
    jobId: string,
    signal?: AbortSignal
  ): Promise<{ job_id: string; cancelled: boolean; message: string }> {
    return this.request<{ job_id: string; cancelled: boolean; message: string }>(
      `/api/v1/benchmarks/jobs/${encodeURIComponent(jobId)}/cancel`,
      { method: 'POST', signal }
    );
  }

  public async listBenchmarks(
    params: { limit?: number; offset?: number; model_id?: string; engine_type?: string } = {},
    signal?: AbortSignal
  ): Promise<PaginatedResponse<BenchmarkSuiteResult>> {
    const sp = new URLSearchParams();
    if (params.limit !== undefined) sp.set('limit', String(params.limit));
    if (params.offset !== undefined) sp.set('offset', String(params.offset));
    if (params.model_id) sp.set('model_id', params.model_id);
    if (params.engine_type) sp.set('engine_type', params.engine_type);

    const query = sp.toString() ? `?${sp.toString()}` : '';
    return this.request<PaginatedResponse<BenchmarkSuiteResult>>(`/api/v1/benchmarks${query}`, {
      method: 'GET',
      signal,
    });
  }

  public async getBenchmarkById(suiteId: string, signal?: AbortSignal): Promise<BenchmarkSuiteResult> {
    return this.request<BenchmarkSuiteResult>(`/api/v1/benchmarks/${encodeURIComponent(suiteId)}`, {
      method: 'GET',
      signal,
    });
  }

  /**
   * Subscribe to the live TokenStreamEvent SSE stream for an active benchmark job.
   * Returns a cleanup function that cancels the stream reader and aborts the request.
   */
  public subscribeJobStream(
    jobId: string,
    onEvent: (event: TokenStreamEvent) => void,
    onError?: (err: Error) => void,
    onComplete?: () => void
  ): () => void {
    const abortController = new AbortController();
    const streamUrl = `${this.baseUrl}/api/v1/benchmarks/jobs/${encodeURIComponent(jobId)}/stream`;

    (async () => {
      try {
        const response = await fetch(streamUrl, {
          signal: abortController.signal,
          headers: { Accept: 'text/event-stream' },
        });

        if (!response.ok) {
          throw new ApiError(response.status, `SSE stream failed: HTTP ${response.status}`);
        }

        if (!response.body) {
          onComplete?.();
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        const parseMessage = (msg: string) => {
          const lines = msg.split('\n');
          const dataLines: string[] = [];
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              dataLines.push(line.slice(6));
            } else if (line.startsWith('data:')) {
              dataLines.push(line.slice(5));
            }
          }

          if (dataLines.length > 0) {
            const dataStr = dataLines.join('\n').trim();
            if (dataStr && dataStr !== 'null') {
              try {
                const parsed = JSON.parse(dataStr) as TokenStreamEvent;
                onEvent(parsed);
              } catch {
                // Ignore parse errors on malformed chunks
              }
            }
          }
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          // Normalize CRLF to LF per SSE specification
          buffer = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
          const messages = buffer.split('\n\n');
          buffer = messages.pop() ?? '';

          for (const msg of messages) {
            if (msg.trim()) {
              parseMessage(msg);
            }
          }
        }

        // Flush any remaining buffered message
        if (buffer.trim()) {
          parseMessage(buffer.trim());
        }

        onComplete?.();
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          return; // Expected cancellation
        }
        onError?.(err instanceof Error ? err : new Error(String(err)));
      }
    })();

    return () => {
      abortController.abort();
    };
  }

  // --------------------------------------------------------------------------
  // Bottleneck Diagnostics
  // --------------------------------------------------------------------------

  public async runDiagnostics(
    req: { suite_id: string; model?: ModelSpec },
    signal?: AbortSignal
  ): Promise<DiagnosticReport> {
    return this.request<DiagnosticReport>('/api/v1/diagnostics/run', {
      method: 'POST',
      body: JSON.stringify(req),
      signal,
    });
  }

  public async listDiagnostics(
    limit: number = 50,
    offset: number = 0,
    signal?: AbortSignal
  ): Promise<PaginatedResponse<DiagnosticReport>> {
    return this.request<PaginatedResponse<DiagnosticReport>>(
      `/api/v1/diagnostics?limit=${limit}&offset=${offset}`,
      { method: 'GET', signal }
    );
  }

  public async getDiagnosticReport(reportId: string, signal?: AbortSignal): Promise<DiagnosticReport> {
    return this.request<DiagnosticReport>(`/api/v1/diagnostics/${encodeURIComponent(reportId)}`, {
      method: 'GET',
      signal,
    });
  }

  // --------------------------------------------------------------------------
  // Empirical Prediction Evaluations
  // --------------------------------------------------------------------------

  public async listEvaluations(
    params: { limit?: number; offset?: number; target?: string } = {},
    signal?: AbortSignal
  ): Promise<PaginatedResponse<EvaluationResult>> {
    const sp = new URLSearchParams();
    if (params.limit !== undefined) sp.set('limit', String(params.limit));
    if (params.offset !== undefined) sp.set('offset', String(params.offset));
    if (params.target) sp.set('target', params.target);

    const query = sp.toString() ? `?${sp.toString()}` : '';
    return this.request<PaginatedResponse<EvaluationResult>>(`/api/v1/predictions/evaluations${query}`, {
      method: 'GET',
      signal,
    });
  }

  public async createEvaluation(
    evaluation: EvaluationResult,
    evaluationId?: string,
    signal?: AbortSignal
  ): Promise<{ evaluation_id: string; message: string }> {
    const query = evaluationId ? `?evaluation_id=${encodeURIComponent(evaluationId)}` : '';
    return this.request<{ evaluation_id: string; message: string }>(`/api/v1/predictions/evaluations${query}`, {
      method: 'POST',
      body: JSON.stringify(evaluation),
      signal,
    });
  }

  public async getEvaluationById(evaluationId: string, signal?: AbortSignal): Promise<EvaluationResult> {
    return this.request<EvaluationResult>(
      `/api/v1/predictions/evaluations/${encodeURIComponent(evaluationId)}`,
      { method: 'GET', signal }
    );
  }

  // --------------------------------------------------------------------------
  // Constrained Optimization and Pareto Recommendations
  // --------------------------------------------------------------------------

  public async recommendOptimization(
    request: OptimizationRequest,
    signal?: AbortSignal
  ): Promise<OptimizationResult> {
    return this.request<OptimizationResult>('/api/v1/optimize/recommend', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  public async listOptimizations(
    limit: number = 50,
    offset: number = 0,
    signal?: AbortSignal
  ): Promise<PaginatedResponse<OptimizationHistoryItem>> {
    return this.request<PaginatedResponse<OptimizationHistoryItem>>(
      `/api/v1/optimize/history?limit=${limit}&offset=${offset}`,
      { method: 'GET', signal }
    );
  }

  public async getOptimizationById(
    optimizationId: string,
    signal?: AbortSignal
  ): Promise<{ request: OptimizationRequest; result: OptimizationResult }> {
    return this.request<{ request: OptimizationRequest; result: OptimizationResult }>(
      `/api/v1/optimize/${encodeURIComponent(optimizationId)}`,
      { method: 'GET', signal }
    );
  }

  // --------------------------------------------------------------------------
  // Session 17 Workload Recommendations and Live Validation
  // --------------------------------------------------------------------------

  public async recommendWorkload(
    request: WorkloadRecommendationRequest,
    signal?: AbortSignal
  ): Promise<WorkloadRecommendationResponse> {
    return this.request<WorkloadRecommendationResponse>('/api/v1/recommendation/workload', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  public async listRecommendationScenarios(
    signal?: AbortSignal
  ): Promise<ValidationScenario[]> {
    return this.request<ValidationScenario[]>('/api/v1/recommendation/scenarios', {
      method: 'GET',
      signal,
    });
  }

  public async validateRecommendation(
    request: ValidationRequest,
    signal?: AbortSignal
  ): Promise<ValidationResponse> {
    return this.request<ValidationResponse>('/api/v1/recommendation/validate', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  public async getModelCatalog(signal?: AbortSignal): Promise<CandidateModelSpec[]> {
    return this.request<CandidateModelSpec[]>('/api/v1/models/catalog', {
      method: 'GET',
      signal,
    });
  }

  public async getWorkloadSuite(signal?: AbortSignal): Promise<BenchmarkWorkloadTask[]> {
    return this.request<BenchmarkWorkloadTask[]>('/api/v1/models/workloads', {
      method: 'GET',
      signal,
    });
  }

  public async compareModels(
    request: MultiModelCompareRequest,
    signal?: AbortSignal
  ): Promise<MultiModelComparisonReport> {
    return this.request<MultiModelComparisonReport>('/api/v1/models/compare', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  // --------------------------------------------------------------------------
  // Session 24 Hardware-Aware Runtime Recommendations and Live Validation
  // --------------------------------------------------------------------------

  public async recommendRuntime(
    request: RuntimeRecommendationRequest,
    signal?: AbortSignal
  ): Promise<RuntimeRecommendationResult> {
    return this.request<RuntimeRecommendationResult>('/api/v1/recommendation/runtime', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  public async listRuntimeScenarios(
    signal?: AbortSignal
  ): Promise<RuntimeScenario[]> {
    return this.request<RuntimeScenario[]>('/api/v1/recommendation/runtime/scenarios', {
      method: 'GET',
      signal,
    });
  }

  public async validateRuntime(
    request: RuntimeValidationRequest,
    signal?: AbortSignal
  ): Promise<RuntimeValidationResponse> {
    return this.request<RuntimeValidationResponse>('/api/v1/recommendation/runtime/validate', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  // --------------------------------------------------------------------------
  // Session 25 Inference Capacity Planning & SLA-Aware Admission
  // --------------------------------------------------------------------------

  public async estimateCapacity(
    request: CapacityEstimateRequest,
    signal?: AbortSignal
  ): Promise<WorkloadCapacityReport> {
    return this.request<WorkloadCapacityReport>('/api/v1/capacity/estimate', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  public async admitRequest(
    request: AdmissionRequest,
    signal?: AbortSignal
  ): Promise<AdmissionResult> {
    return this.request<AdmissionResult>('/api/v1/capacity/admit', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  public async listCapacityScenarios(
    signal?: AbortSignal
  ): Promise<CapacityScenariosResponse> {
    return this.request<CapacityScenariosResponse>('/api/v1/capacity/scenarios', {
      method: 'GET',
      signal,
    });
  }

  public async validateCapacityScenario(
    scenarioId: string,
    inFlight: number = 0,
    queueDepth: number = 0,
    signal?: AbortSignal
  ): Promise<ScenarioValidationResponse> {
    return this.request<ScenarioValidationResponse>('/api/v1/capacity/validate', {
      method: 'POST',
      body: JSON.stringify({
        scenario_id: scenarioId,
        current_in_flight: inFlight,
        current_queue_depth: queueDepth,
      }),
      signal,
    });
  }

  // --------------------------------------------------------------------------
  // Session 26 Mixed-Workload Serving, HoL & Policy Analysis
  // --------------------------------------------------------------------------

  public async getMixedProfiles(
    signal?: AbortSignal
  ): Promise<MixedWorkloadProfile[]> {
    return this.request<MixedWorkloadProfile[]>('/api/v1/capacity/mixed-profiles', {
      method: 'GET',
      signal,
    });
  }

  public async estimateMixedCapacity(
    request: MixedCapacityEstimateRequest,
    signal?: AbortSignal
  ): Promise<MixedCapacityEstimateResponse> {
    return this.request<MixedCapacityEstimateResponse>('/api/v1/capacity/estimate-mixed', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }

  public async getHoLAnalysis(
    signal?: AbortSignal
  ): Promise<{ observations: HeadOfLineBlockingObservation[] }> {
    return this.request<{ observations: HeadOfLineBlockingObservation[] }>('/api/v1/capacity/hol-analysis', {
      method: 'GET',
      signal,
    });
  }

  public async getModelSwitchingAnalysis(
    signal?: AbortSignal
  ): Promise<{ observations: ModelSwitchingObservation[] }> {
    return this.request<{ observations: ModelSwitchingObservation[] }>('/api/v1/capacity/model-switching-analysis', {
      method: 'GET',
      signal,
    });
  }

  public async getAdmissionPoliciesSummary(
    signal?: AbortSignal
  ): Promise<{ policy_reports: PolicyEvaluationReport[] }> {
    return this.request<{ policy_reports: PolicyEvaluationReport[] }>('/api/v1/capacity/admission-policies', {
      method: 'GET',
      signal,
    });
  }

  // --------------------------------------------------------------------------
  // Session 27 Serving Architecture & Parallelism
  // --------------------------------------------------------------------------

  public async getServingCapabilities(
    signal?: AbortSignal
  ): Promise<ServingCapabilitiesResponse> {
    return this.request<ServingCapabilitiesResponse>('/api/v1/serving/capabilities', {
      method: 'GET',
      signal,
    });
  }

  public async getArchitectureComparisons(
    modelId?: string,
    signal?: AbortSignal
  ): Promise<ServingComparisonsResponse> {
    const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
    return this.request<ServingComparisonsResponse>(`/api/v1/serving/comparison${query}`, {
      method: 'GET',
      signal,
    });
  }

  public async getServingHolComparisons(
    modelId?: string,
    signal?: AbortSignal
  ): Promise<ServingHolComparisonsResponse> {
    const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
    return this.request<ServingHolComparisonsResponse>(`/api/v1/serving/hol-comparison${query}`, {
      method: 'GET',
      signal,
    });
  }

  public async recommendServingArchitecture(
    request: ArchitectureRecommendationRequest,
    signal?: AbortSignal
  ): Promise<ServingArchitectureRecommendation> {
    return this.request<ServingArchitectureRecommendation>('/api/v1/serving/recommend', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    });
  }
}

export const apiClient = new EngineApiClient();
