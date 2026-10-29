/**
 * Strict TypeScript domain models and enums matching backend Pydantic contracts.
 * 
 * Enforces explicit data provenance, scientific classifications, and strict nullability.
 */

export type EngineType = 'mock' | 'ollama' | 'vllm' | 'llamacpp' | 'transformers' | 'custom';
export type ExecutionMode = 'REAL' | 'SIMULATED';
export type ProvenanceType = 'MEASURED' | 'PREDICTED' | 'HEURISTIC' | 'THEORETICAL';
export type ClockSource = 'MONOTONIC_REAL' | 'SYNTHETIC_VIRTUAL';

export type QuantizationType =
  | 'none'
  | 'fp16'
  | 'bf16'
  | 'int8'
  | 'int4'
  | 'gguf_q4_k_m'
  | 'gguf_q8_0'
  | 'awq_int4'
  | 'gptq_int4';

export type TokenEventType = 'start' | 'token' | 'chunk' | 'completion' | 'error';
export type StreamTerminationReason = 'finished' | 'max_tokens' | 'stop_sequence' | 'cancelled' | 'error';

export type DiagnosticSeverity = 'info' | 'warning' | 'critical';
export type DiagnosticCategory =
  | 'memory_bottleneck'
  | 'compute_bottleneck'
  | 'efficiency_bottleneck'
  | 'configuration_suboptimal'
  | 'hardware_limitation';
export type DiagnosticConfidence = 'high' | 'medium' | 'low';

export type FeasibilityStatus =
  | 'FEASIBLE'
  | 'PROVEN_INFEASIBLE'
  | 'ESTIMATED_INFEASIBLE'
  | 'UNKNOWN_FEASIBILITY';

export type EvidenceStatus =
  | 'MEASURED_EVALUATED'
  | 'PREDICTED_EMPIRICAL'
  | 'THEORETICAL_ANALYTICAL'
  | 'INSUFFICIENT_EVIDENCE';

export type SelectionPolicy =
  | 'none'
  | 'prioritize_objective'
  | 'balanced_knee_point'
  | 'lowest_cost';

export type PredictionTarget =
  | 'ttft_ms'
  | 'tpot_ms'
  | 'chunk_latency_ms'
  | 'output_throughput_tok_s'
  | 'peak_vram_mb'
  | 'duration_ms'
  | 'ttfc_ms'
  | 'server_prompt_eval_duration_ms'
  | 'server_eval_duration_ms'
  | 'client_e2e_duration_ms'
  | 'client_e2e_output_throughput_tok_s'
  | 'server_total_duration_ms'
  | 'server_load_duration_ms'
  | 'server_prefill_duration_ms'
  | 'server_decode_duration_ms'
  | 'server_decode_throughput_tok_s';

export type OptimizationObjective =
  | 'minimize_ttft'
  | 'minimize_tpot'
  | 'maximize_throughput'
  | 'minimize_vram'
  | 'minimize_cost'
  | 'maximize_decode_throughput'
  | 'minimize_duration'
  | 'minimize_client_duration';

export interface HealthResponse {
  status: string;
  database: string;
  version?: string;
}

export interface ErrorDetail {
  error: string;
  message: string;
  detail?: unknown;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface HostSystemInfo {
  os_name: string;
  os_version: string;
  os_release?: string;
  architecture: string;
  python_version: string;
}

export interface CPUInfo {
  model_name: string;
  vendor: string;
  physical_cores: number;
  logical_cores: number;
  frequency_mhz?: number;
}

export interface MemoryInfo {
  total_bytes: number;
  available_bytes: number;
}

export interface DetectedGPU {
  index: number;
  name: string;
  vendor: string;
  total_memory_mb: number;
  driver_version?: string;
  uuid?: string;
  pci_bus_id?: string;
  compute_capability?: string;
  detection_source: string;
}

export interface ComputeCapabilitySpec {
  fp32_tflops?: number;
  fp16_dense_tflops?: number;
  fp16_sparse_tflops?: number;
  bf16_dense_tflops?: number;
  bf16_sparse_tflops?: number;
  fp8_dense_tflops?: number;
  fp8_sparse_tflops?: number;
  int8_dense_tops?: number;
  int8_sparse_tops?: number;
}

export interface GPUSpecification {
  catalog_id: string;
  name: string;
  architecture: string;
  form_factor: string;
  memory_capacity_gb: number;
  memory_type: string;
  memory_bus_width_bits?: number;
  theoretical_memory_bandwidth_gbps: number;
  compute: ComputeCapabilitySpec;
  tdp_watts?: number;
  compute_capability?: string;
  source_reference: string;
  caveats: string[];
  provenance: ProvenanceType;
}

export interface TelemetryTools {
  has_nvidia_smi: boolean;
  nvidia_smi_path?: string;
  has_rocm_smi: boolean;
  rocm_smi_path?: string;
  has_pynvml: boolean;
}

export interface HardwareCapabilityReport {
  report_id: string;
  timestamp_utc: string;
  is_simulated: boolean;
  execution_mode: ExecutionMode;
  host: HostSystemInfo;
  cpu: CPUInfo;
  memory: MemoryInfo;
  gpus: DetectedGPU[];
  telemetry: TelemetryTools;
  matched_catalog_specs: GPUSpecification[];
  missing_or_incomplete_fields: string[];
  warnings: string[];
}

export interface ModelSpec {
  model_id: string;
  family?: string;
  parameter_count_billion?: number;
  context_window?: number;
  num_layers?: number;
  num_attention_heads?: number;
  num_kv_heads?: number;
  hidden_dimension?: number;
  head_dimension?: number;
  vocab_size?: number;
  quantization?: QuantizationType;
}

export interface EngineConfig {
  engine_type: EngineType;
  quantization?: QuantizationType;
  max_model_len?: number;
  tensor_parallel_size?: number;
  gpu_memory_utilization?: number;
  extra_args?: Record<string, unknown>;
}

export interface WorkloadSpec {
  prompt_tokens?: number | null;
  output_tokens: number;
  temperature?: number;
  top_p?: number;
  prompt_text?: string;
  system_prompt?: string;
}

export interface BenchmarkSubmissionRequest {
  model: ModelSpec;
  engine_type?: EngineType;
  engine_config?: EngineConfig;
  workloads: WorkloadSpec[];
  concurrency?: number;
  warmup_requests?: number;
  request_timeout_seconds?: number;
  sample_memory?: boolean;
  endpoint_url?: string;
}

export interface JobStatus {
  job_id: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  suite_id?: string | null;
  model_id: string;
  engine_type: string;
  error_message?: string | null;
}

export interface ServerReportedStats {
  prompt_tokens?: number;
  output_tokens?: number;
  prompt_duration_ms?: number;
  eval_duration_ms?: number;
  total_duration_ms?: number;
  eval_tokens_per_second?: number;
  prompt_eval_cached_count?: number;
  load_duration_ms?: number;
  server_prefill_duration_ms?: number;
  server_decode_duration_ms?: number;
  server_load_duration_ms?: number;
  server_total_duration_ms?: number;
  server_decode_throughput_tok_s?: number;
  raw_stats?: Record<string, unknown>;
}

export interface MemoryWatermarkResult {
  peak_process_rss_mb?: number;
  peak_gpu_vram_mb?: number;
  sample_count: number;
  sample_interval_ms: number;
  telemetry_source: string;
  is_telemetry_available: boolean;
  notes?: string;
  device_index?: number;
  device_name?: string;
}

export interface RequestBenchmarkResult {
  request_id: string;
  success: boolean;
  error_message?: string;
  prompt_tokens?: number | null;
  requested_output_tokens?: number;
  output_tokens: number;
  ttft_ms?: number;
  ttfc_ms?: number;
  tpot_ms?: number;
  duration_ms: number;
  output_tokens_per_second?: number;
  client_e2e_duration_ms?: number;
  client_e2e_output_throughput_tok_s?: number;
  server_total_duration_ms?: number;
  server_load_duration_ms?: number;
  server_prefill_duration_ms?: number;
  server_decode_duration_ms?: number;
  server_decode_throughput_tok_s?: number;
  inter_token_latencies_ms?: number[];
  itl_p50_ms?: number;
  itl_p90_ms?: number;
  itl_p95_ms?: number;
  itl_p99_ms?: number;
  chunk_count?: number;
  chunk_latencies_ms?: number[];
  chunk_latency_p50_ms?: number;
  chunk_latency_p90_ms?: number;
  chunk_latency_p95_ms?: number;
  chunk_latency_p99_ms?: number;
  server_stats?: ServerReportedStats;
  token_count_source: string;
  prompt_token_count_source: string;
  clock_source: ClockSource;
  execution_mode: ExecutionMode;
  provenance: ProvenanceType;
  is_warmup?: boolean;
  incomplete_reason?: string;
}

export interface BenchmarkSuiteResult {
  suite_id: string;
  timestamp_utc: string;
  status: string;
  model_id: string;
  engine_type: EngineType;
  execution_mode: ExecutionMode;
  clock_source: ClockSource;
  provenance: ProvenanceType;
  concurrency: number;
  total_requests: number;
  warmup_requests: number;
  successful_requests: number;
  failed_requests: number;
  wall_time_seconds: number;
  aggregate_tokens_per_second?: number;
  aggregate_output_tokens_per_second?: number;
  request_throughput_per_second?: number;
  ttft_p50_ms?: number;
  ttft_p90_ms?: number;
  ttft_p95_ms?: number;
  ttft_p99_ms?: number;
  ttfc_p50_ms?: number;
  ttfc_p90_ms?: number;
  ttfc_p95_ms?: number;
  ttfc_p99_ms?: number;
  chunk_latency_p50_ms?: number;
  tpot_mean_ms?: number;
  itl_p50_ms?: number;
  itl_p90_ms?: number;
  itl_p95_ms?: number;
  itl_p99_ms?: number;
  memory_watermark?: MemoryWatermarkResult;
  request_results: RequestBenchmarkResult[];
  warmup_results: RequestBenchmarkResult[];
  warnings: string[];
}

export interface TokenStreamEvent {
  request_id: string;
  sequence_index: number;
  event_type: TokenEventType;
  token_index?: number;
  token_text?: string;
  token_id?: number;
  timestamp_ns: number;
  delta_ns?: number;
  clock_source: ClockSource;
  execution_mode: ExecutionMode;
  is_first_token: boolean;
  is_terminal: boolean;
  termination_reason?: StreamTerminationReason;
  error_message?: string;
  server_stats?: ServerReportedStats;
}

export interface Observation<T = unknown> {
  value: T;
  provenance: ProvenanceType;
  execution_mode: ExecutionMode;
  clock_source?: ClockSource;
  unit?: string;
  description?: string;
}

export interface RooflineAnalysisResult {
  hardware_name: string;
  form_factor: string;
  precision: string;
  peak_compute_tflops: number;
  peak_memory_bandwidth_gbps: number;
  ridge_point_flops_per_byte: number;
  operational_intensity: number;
  operational_intensity_provenance: ProvenanceType;
  attainable_compute_tflops: number;
  theoretical_max_tokens_per_second?: number;
  limiting_factor: string;
  provenance: ProvenanceType;
  assumptions: string[];
  notes: string;
}

export interface ModelResourceEstimate {
  model_id: string;
  parameter_count_billion?: number;
  parameter_storage_mb: number;
  weight_precision: QuantizationType;
  bytes_per_parameter: number;
  is_quantization_exact: boolean;
  quantization_notes?: string;
  attention_architecture: string;
  kv_cache_per_token_bytes?: number;
  kv_cache_total_mb?: number;
  kv_cache_precision: QuantizationType;
  batch_size: number;
  sequence_length: number;
  total_payload_memory_mb: number;
  estimated_runtime_vram_mb?: number;
}

export interface DiagnosticFinding {
  finding_id: string;
  category: DiagnosticCategory;
  severity: DiagnosticSeverity;
  title: string;
  message: string;
  confidence: DiagnosticConfidence;
  confidence_rationale: string;
  supporting_observations?: Observation[];
  assumptions?: string[];
  limitations?: string[];
  suggested_experiments?: string[];
}

export interface DiagnosticReport {
  report_id: string;
  timestamp_utc: string;
  model_id: string;
  engine_type: EngineType;
  execution_mode: ExecutionMode;
  hardware_catalog_id?: string;
  summary: string;
  findings: DiagnosticFinding[];
  roofline_analysis?: RooflineAnalysisResult;
  resource_estimate?: ModelResourceEstimate;
}

export interface EvaluationResult {
  evaluation_id?: string;
  target: PredictionTarget;
  predictor_id: string;
  predictor_type: string;
  split_strategy: string;
  sample_count: number;
  test_sample_count?: number;
  train_sample_count?: number;
  unit: string;
  mae: number;
  rmse: number;
  mape_percent?: number;
  baseline_mae?: number;
  baseline_rmse?: number;
  is_synthetic: boolean;
  execution_mode: ExecutionMode;
  sufficiency_threshold?: number;
  is_statistically_sufficient: boolean;
  notes?: string;
}

export interface CandidateConfig {
  model_spec: ModelSpec;
  quantization: QuantizationType;
  concurrency: number;
  engine_type: EngineType;
  tensor_parallel_size: number;
  gpu_memory_utilization?: number;
  extra_args?: Record<string, unknown>;
}

export interface GeneralizationStatus {
  is_out_of_distribution: boolean;
  warnings: string[];
  extrapolated_features: string[];
  unseen_categories: string[];
  missing_features: string[];
}

export interface CandidateEvaluation {
  config: CandidateConfig;
  feasibility_status: FeasibilityStatus;
  rejection_reasons: string[];
  objective_values: Record<string, number>;
  objective_units: Record<string, string>;
  evidence_status: Record<string, EvidenceStatus>;
  provenance: Record<string, ProvenanceType>;
  resource_estimate?: ModelResourceEstimate;
  roofline_result?: RooflineAnalysisResult;
  generalization_status?: GeneralizationStatus;
  has_mixed_evidence: boolean;
  comparability_warning?: string;
  trade_off_summary: string;
  notes: string[];
}

export interface SearchSpace {
  models: ModelSpec[];
  quantizations: QuantizationType[];
  concurrencies: number[];
  engines: EngineType[];
  tensor_parallel_sizes?: number[];
  max_candidates?: number;
}

export interface OptimizationRequest {
  workload: WorkloadSpec;
  search_space: SearchSpace;
  objectives: OptimizationObjective[];
  hardware_catalog_id?: string;
  cost_per_gpu_hour?: number;
  selection_policy?: SelectionPolicy;
  primary_objective?: OptimizationObjective;
}

export interface OptimizationResult {
  request_id: string;
  summary: string;
  created_at?: string;
  timestamp_utc?: string;
  total_candidates_generated: number;
  total_candidates_evaluated: number;
  feasible_candidates_count: number;
  infeasible_candidates_count: number;
  unknown_feasibility_count: number;
  pareto_frontier: CandidateEvaluation[];
  recommended_candidate?: CandidateEvaluation;
  selection_rationale?: string;
  has_incomparable_evidence: boolean;
  evidence_caveats: string[];
  rejections_by_reason: Record<string, number>;
  all_evaluations: CandidateEvaluation[];
}

export interface OptimizationHistoryItem {
  optimization_id: string;
  created_at: string;
  model_id: string;
  prompt_tokens?: number | null;
  output_tokens: number;
  summary: string;
  recommended_config_id?: string;
  has_incomparable_evidence: boolean;
  feasible_candidates_count: number;
  infeasible_candidates_count: number;
  unknown_feasibility_count: number;
}

// --------------------------------------------------------------------------
// Session 17 Workload Recommendation & Prospective Validation Types
// --------------------------------------------------------------------------

export type RecommendationStatus =
  | 'RECOMMENDED'
  | 'INCONCLUSIVE'
  | 'INFEASIBLE'
  | 'UNSUPPORTED'
  | 'REJECTED';

export type RecommendationEvidenceSource =
  | 'DIRECTLY_MEASURED'
  | 'DIRECTLY_MEASURED_CONDITIONAL'
  | 'HELD_OUT_VALIDATED_PREDICTION'
  | 'EXPLORATORY_PREDICTION'
  | 'ANALYTICAL_ESTIMATE'
  | 'INSUFFICIENT_EVIDENCE';

export interface WorkloadRecommendationRequest {
  model_id?: string;
  hardware_catalog_id?: string | null;
  runtime?: string;
  prompt_tokens: number;
  prompt_tokens_provenance?: string;
  expected_output_tokens: number;
  concurrency?: number;
  expected_model_residency?: 'warm' | 'cold' | 'unknown';
  expected_prefix_cache?: 'cold_prefix' | 'warm_prefix' | 'partial_prefix' | 'unknown';
  latency_sla_ms?: number | null;
  decode_throughput_sla_tok_s?: number | null;
  ttft_sla_ms?: number | null;
  objective?: OptimizationObjective;
  require_cold_start?: boolean;
  user_preferences?: Record<string, unknown>;
}

export interface PredictedMetrics {
  client_e2e_duration_ms?: number | null;
  server_decode_throughput_tok_s?: number | null;
  client_e2e_output_throughput_tok_s?: number | null;
  server_prefill_duration_ms?: number | null;
  server_decode_duration_ms?: number | null;
  server_load_duration_ms?: number | null;
}

export type OperatingRegime =
  | 'COLD_START'
  | 'WARM_SHORT_BURST'
  | 'WARM_SUSTAINED_GENERATION'
  | 'PROMPT_PREFIX_REUSE'
  | 'CONSECUTIVE_SATURATED';

export interface WorkloadRecommendationResponse {
  recommendation_id: string;
  created_at: string;
  status: RecommendationStatus;
  workload_summary: Record<string, unknown>;
  selected_configuration?: Record<string, unknown> | null;
  predicted_metrics: PredictedMetrics;
  evidence_source: RecommendationEvidenceSource;
  applicable_dataset: string;
  prediction_uncertainty_interval?: [number, number] | null;
  exploratory_prediction_range?: [number, number] | null;
  confidence_label_status?: string;
  range_coverage_percent?: number | null;
  generalization_status: {
    is_out_of_distribution: boolean;
    warnings: string[];
    extrapolated_features: string[];
    unseen_categories: string[];
    missing_features: string[];
  };
  ood_warnings: string[];
  cold_start_assumptions: string;
  prefix_cache_assumptions: string;
  sla_assessments: Record<string, unknown>;
  relevant_alternatives: Array<Record<string, unknown>>;
  explanation_summary: string;
  influencing_factors: string[];
  validation_benchmark_payload?: Record<string, unknown> | null;
  operating_regime?: OperatingRegime;
  worst_case_latency_ms?: number | null;
  historical_sla_violation_frequency_pct?: number | null;
  conservative_feasibility?: string | null;
  sla_risk_flag?: string | null;
  quality_assessment?: {
    quality_status: 'QUALITY_SATISFIED' | 'QUALITY_VIOLATED' | 'UNKNOWN_QUALITY_NO_EVALUATION';
    json_compliance_rate?: number | null;
    code_pass_rate?: number | null;
    reasoning_accuracy?: number | null;
    task_correctness_mean?: number | null;
    evaluation_notes: string[];
  } | null;
  model_compatibility?: 'COMPATIBLE' | 'UNINSTALLED' | 'INSUFFICIENT_MEMORY' | 'INCOMPATIBLE_RUNTIME';
}

export interface ValidationScenario {
  scenario_id: string;
  name: string;
  description: string;
  request: WorkloadRecommendationRequest;
}

export interface ValidationRequest {
  recommendation_id: string;
  model_id?: string;
  prompt_tokens: number;
  expected_output_tokens: number;
  expected_latency_ms?: number | null;
  expected_decode_throughput_tok_s?: number | null;
}

export interface ValidationResponse {
  validation_id: string;
  recommendation_id: string;
  status: string;
  executed_at: string;
  observed_client_e2e_duration_ms: number;
  observed_server_decode_duration_ms?: number | null;
  observed_server_prefill_duration_ms?: number | null;
  observed_server_load_duration_ms?: number | null;
  observed_server_decode_throughput_tok_s?: number | null;
  observed_client_e2e_output_throughput_tok_s?: number | null;
  predicted_client_e2e_duration_ms?: number | null;
  predicted_server_decode_throughput_tok_s?: number | null;
  duration_absolute_error_ms?: number | null;
  duration_relative_error_pct?: number | null;
  throughput_absolute_error_tok_s?: number | null;
  throughput_relative_error_pct?: number | null;
  explanation: string;
}

export interface CandidateModelSpec {
  model_id: string;
  display_name: string;
  model_family: string;
  parameter_count_billion: number;
  default_quantization: string;
  expected_disk_size_mb: number;
  expected_ram_mb: number;
  context_window: number;
  tokenizer_name: string;
  vocabulary_size: number;
  license: string;
  intended_evaluation_role: string;
  download_command: string;
  installation_status: 'INSTALLED' | 'UNINSTALLED_CANDIDATE' | 'CORRUPTED' | 'UNSUPPORTED';
  is_installed: boolean;
  digest?: string | null;
}

export interface BenchmarkWorkloadTask {
  task_id: string;
  name: string;
  task_type: string;
  prompt: string;
  max_output_tokens: number;
  expected_prompt_words: number;
  description: string;
  version: string;
}

export interface ModelQualityConstraintSpec {
  require_json_compliance?: boolean;
  require_code_pass?: boolean;
  min_task_correctness?: number | null;
  min_json_compliance_rate?: number | null;
  min_code_pass_rate?: number | null;
  min_reasoning_accuracy?: number | null;
  require_json_semantic_correctness?: boolean;
  min_retrieval_accuracy?: number | null;
  min_instruction_compliance?: number | null;
  target_category?: string | null;
  max_memory_budget_mb?: number | null;
  preferred_model_family?: string | null;
}

export interface ModelComparisonEntry {
  model_id: string;
  display_name: string;
  model_family: string;
  parameter_count_billion: number;
  quantization: string;
  installation_status: 'INSTALLED' | 'UNINSTALLED_CANDIDATE' | 'CORRUPTED' | 'UNSUPPORTED';
  compatibility_status: 'COMPATIBLE' | 'UNINSTALLED' | 'INSUFFICIENT_MEMORY' | 'INCOMPATIBLE_RUNTIME';
  evidence_source: string;
  nominal_latency_ms?: number | null;
  worst_case_latency_ms?: number | null;
  raw_decode_tokens_per_second?: number | null;
  characters_per_second?: number | null;
  quality_status: 'QUALITY_SATISFIED' | 'QUALITY_VIOLATED' | 'UNKNOWN_QUALITY_NO_EVALUATION';
  mean_correctness_score?: number | null;
  json_compliance_rate?: number | null;
  code_pass_rate?: number | null;
  category_scores?: Record<string, number>;
  numerical_accuracy?: number | null;
  retrieval_accuracy?: number | null;
  instruction_compliance_rate?: number | null;
  evidence_sample_count?: number;
  conservative_feasibility: string;
  rank?: number | null;
  decision_label: string;
  disclaimers: string[];
}

export interface MultiModelComparisonReport {
  workload_summary: Record<string, unknown>;
  latency_sla_ms?: number | null;
  quality_constraints: ModelQualityConstraintSpec;
  candidate_evaluations: ModelComparisonEntry[];
  recommended_model_id?: string | null;
  comparability_caveat: string;
  ranking_rationale: string;
}

export interface MultiModelCompareRequest {
  candidate_model_ids: string[];
  prompt_tokens: number;
  expected_output_tokens: number;
  latency_sla_ms?: number | null;
  quality_constraints?: ModelQualityConstraintSpec;
}

