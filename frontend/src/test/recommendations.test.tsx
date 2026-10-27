import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { RecommendationsPage } from '../pages/RecommendationsPage';
import { apiClient } from '../api/client';
import {
  WorkloadRecommendationResponse,
  ValidationScenario,
} from '../api/types';

describe('RecommendationsPage Component', () => {
  const mockScenarios: ValidationScenario[] = [
    {
      scenario_id: 'SCENARIO_A',
      name: 'Scenario A — Short Interactive Request',
      description: 'Small prompt and short response.',
      request: {
        model_id: 'llama3.2:1b',
        prompt_tokens: 32,
        expected_output_tokens: 16,
        latency_sla_ms: 2500,
      },
    },
    {
      scenario_id: 'SCENARIO_E',
      name: 'Scenario E — Strict TTFT SLA Request',
      description: 'Requests unmeasurable TTFT SLA.',
      request: {
        model_id: 'llama3.2:1b',
        prompt_tokens: 46,
        expected_output_tokens: 32,
        ttft_sla_ms: 100,
      },
    },
  ];

  const mockRecommendation: WorkloadRecommendationResponse = {
    recommendation_id: 'rec-test-1234',
    created_at: new Date().toISOString(),
    status: 'RECOMMENDED',
    workload_summary: {
      model_id: 'llama3.2:1b',
      prompt_tokens: 32,
      expected_output_tokens: 16,
      concurrency: 1,
      objective: 'minimize_duration',
    },
    selected_configuration: {
      model_id: 'llama3.2:1b',
      quantization: 'q8_0',
      runtime: 'ollama',
      device: 'CPU',
    },
    predicted_metrics: {
      client_e2e_duration_ms: 807.27,
      server_decode_throughput_tok_s: 22.72,
      client_e2e_output_throughput_tok_s: 19.82,
      server_prefill_duration_ms: 96.83,
      server_decode_duration_ms: 704.19,
      server_load_duration_ms: 0.0,
    },
    evidence_source: 'HELD_OUT_VALIDATED_PREDICTION',
    applicable_dataset: 'expansion_llama3_2_1b_v2',
    prediction_uncertainty_interval: [421.4, 1193.2],
    generalization_status: {
      is_out_of_distribution: false,
      warnings: [],
      extrapolated_features: [],
      unseen_categories: [],
      missing_features: [],
    },
    ood_warnings: [],
    cold_start_assumptions: 'Warm residency assumed.',
    prefix_cache_assumptions: 'Cold prefix evaluation.',
    sla_assessments: {
      latency_sla: {
        constraint_type: 'HARD',
        target_ms: 2500,
        forecast_ms: 807.27,
        status: 'FEASIBLE',
        headroom_ms: 1692.7,
      },
    },
    relevant_alternatives: [
      {
        name: 'Cold-Start Baseline',
        expected_duration_ms: 1845.01,
        trade_off: 'Loads model from disk; adds 1,038 ms cold start latency.',
      },
    ],
    explanation_summary: 'Recommended Ollama with llama3.2:1b on Intel Core i7-14700 CPU.',
    influencing_factors: ['Warm memory residency assumes zero model loading delay.'],
    validation_benchmark_payload: {
      model_id: 'llama3.2:1b',
      engine: 'ollama',
    },
  };

  beforeEach(() => {
    vi.spyOn(apiClient, 'listRecommendationScenarios').mockResolvedValue(mockScenarios);
    vi.spyOn(apiClient, 'recommendWorkload').mockResolvedValue(mockRecommendation);
  });

  it('renders page header and preset evaluation scenarios', async () => {
    await act(async () => {
      render(<RecommendationsPage />);
    });

    expect(screen.getByText('Workload Configuration Recommendation')).toBeInTheDocument();
    expect(await screen.findByText('Scenario A — Short Interactive Request')).toBeInTheDocument();
    expect(await screen.findByText('Scenario E — Strict TTFT SLA Request')).toBeInTheDocument();
  });

  it('applies scenario preset when clicked and updates form inputs', async () => {
    await act(async () => {
      render(<RecommendationsPage />);
    });

    const scenarioA = await screen.findByText('Scenario A — Short Interactive Request');
    fireEvent.click(scenarioA);

    const promptInput = screen.getByLabelText(/Prompt Tokens/i) as HTMLInputElement;
    expect(promptInput.value).toBe('32');
  });

  it('fetches and displays recommendation with metrics and evidence classification', async () => {
    await act(async () => {
      render(<RecommendationsPage />);
    });

    const submitBtn = screen.getByRole('button', { name: /Get Evidence-Backed Recommendation/i });
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    expect(await screen.findByText(/Recommendation Status: RECOMMENDED/i)).toBeInTheDocument();
    expect(screen.getByText(/HELD_OUT_VALIDATED_PREDICTION/i)).toBeInTheDocument();
    expect(screen.getByText('807.27')).toBeInTheDocument();
    expect(screen.getByText('22.72')).toBeInTheDocument();
  });

  it('renders unavailable metrics gracefully without substituting zero', async () => {
    const incompleteRec: WorkloadRecommendationResponse = {
      ...mockRecommendation,
      predicted_metrics: {
        client_e2e_duration_ms: null,
        server_decode_throughput_tok_s: null,
        client_e2e_output_throughput_tok_s: null,
        server_prefill_duration_ms: null,
        server_decode_duration_ms: null,
        server_load_duration_ms: null,
      },
    };
    vi.spyOn(apiClient, 'recommendWorkload').mockResolvedValue(incompleteRec);

    await act(async () => {
      render(<RecommendationsPage />);
    });

    const submitBtn = screen.getByRole('button', { name: /Get Evidence-Backed Recommendation/i });
    await act(async () => {
      fireEvent.click(submitBtn);
    });

    const naElements = await screen.findAllByText('N/A');
    expect(naElements.length).toBeGreaterThanOrEqual(1);
  });
});
