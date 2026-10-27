# Workload-Aware Inference Configuration Recommendation Guide

## Overview

The **LLM Inference Intelligence Engine (IIE)** provides an end-to-end, evidence-backed inference configuration recommendation workflow. Rather than relying on generic rule-of-thumb heuristics or theoretical ceilings, IIE evaluates workloads against empirical benchmark observations and physical stage-latency decomposition models.

A user or automated orchestrator provides a workload description (prompt length, expected output length, concurrency, SLAs, residency state) and receives:
1. **A feasible inference runtime configuration**.
2. **Decomposed latency and throughput forecasts**.
3. **The scientific evidence category supporting the forecast**.
4. **Important operational assumptions and physical uncertainties**.
5. **A structured, plain-English explanation of why the configuration was recommended**.
6. **A 1-click live benchmark validation workflow** to verify prospective predictions against genuine ground truth.

---

## 1. Physical Stage Latency Decomposition Model

IIE forecasts end-to-end client latency prospective to execution by decomposing inference into four physical stages:

$$\hat{T}_{\text{client\_e2e}} = \hat{T}_{\text{load}}(\text{residency}) + \hat{T}_{\text{prefill}}(P, \text{cache}) + \hat{T}_{\text{decode}}(O) + \hat{T}_{\text{overhead}}$$

Where:
- $\hat{T}_{\text{load}}$: Model weight loading duration. Modeled as $1,037.74\text{ ms}$ for cold NVMe-to-RAM loading, or $0.00\text{ ms}$ when weights are verified resident in memory.
- $\hat{T}_{\text{prefill}}$: Prompt ingestion and KV cache creation. Evaluated at $\approx 3.026\text{ ms/token}$ on Intel Core i7-14700 CPU for cold prompts ($\approx 330\text{ tok/s}$), or $\approx 0.042\text{ ms/token}$ for warm prefix-cache hits.
- $\hat{T}_{\text{decode}}$: Autoregressive token generation. Modeled at a stable rate of $\approx 22.72\text{ tok/s}$ ($44.012\text{ ms/token}$) for `llama3.2:1b` (Q8_0).
- $\hat{T}_{\text{overhead}}$: HTTP headers, JSON serialization, and local loopback socket framing ($\approx 6.25\text{ ms}$).

---

## 2. Evidence Classification Hierarchy

IIE classifies the provenance of all recommendations into five tiers:

1. **`DIRECTLY_MEASURED`**: Workload exactly matches historical measured benchmarks from empirical datasets.
2. **`HELD_OUT_VALIDATED_PREDICTION`**: In-distribution workload ($P \in [42, 677]$, $O \in [16, 128]$) evaluated by the Workload-Aware Latency Predictor (validated with 17-fold LOCO CV).
3. **`EXPLORATORY_PREDICTION`**: Workload extends beyond empirical bounds ($P > 677$ or $O > 128$). Uncertainty intervals are widened and explicit extrapolation warnings are surfaced.
4. **`ANALYTICAL_ESTIMATE`**: Parameter formulas or theoretical compute/bandwidth rooflines without empirical validation.
5. **`INSUFFICIENT_EVIDENCE`**: Required telemetry is unmeasurable (e.g. token-level TTFT on SSE chunk streams).

---

## 3. REST API Reference

### A. Compute Workload Recommendation
- **Endpoint:** `POST /api/v1/recommendation/workload`
- **Request Schema (`WorkloadRecommendationRequest`):**
  ```json
  {
    "model_id": "llama3.2:1b",
    "prompt_tokens": 46,
    "expected_output_tokens": 32,
    "concurrency": 1,
    "expected_model_residency": "warm",
    "expected_prefix_cache": "cold_prefix",
    "latency_sla_ms": 3000.0,
    "objective": "minimize_duration"
  }
  ```
- **Response Schema (`WorkloadRecommendationResponse`):**
  ```json
  {
    "recommendation_id": "rec-09806eff0018",
    "status": "RECOMMENDED",
    "selected_configuration": {
      "model_id": "llama3.2:1b",
      "quantization": "q8_0",
      "runtime": "ollama",
      "device": "CPU"
    },
    "predicted_metrics": {
      "client_e2e_duration_ms": 1537.5,
      "server_decode_throughput_tok_s": 21.67,
      "client_e2e_output_throughput_tok_s": 20.81,
      "server_prefill_duration_ms": 139.2,
      "server_decode_duration_ms": 1398.3,
      "server_load_duration_ms": 0.0
    },
    "evidence_source": "DIRECTLY_MEASURED",
    "applicable_dataset": "expansion_llama3_2_1b_v2",
    "prediction_uncertainty_interval": [1491.4, 1583.6],
    "explanation_summary": "Recommended Ollama with 'llama3.2:1b' (Q8_0) on Intel Core i7-14700 CPU..."
  }
  ```

### B. List Standard Validation Scenarios
- **Endpoint:** `GET /api/v1/recommendation/scenarios`
- Returns preset specifications for Scenarios A through F.

### C. Live Validation Benchmark
- **Endpoint:** `POST /api/v1/recommendation/validate`
- Executes a bounded live request against the local runtime and compares observed latency and throughput against the prospective forecast.

---

## 4. UI Dashboard Workflow

1. Navigate to the **Recommendations** tab in the top navigation bar.
2. Select a **Standard Evaluation Scenario** or manually enter workload parameters.
3. Click **Get Evidence-Backed Recommendation**.
4. Inspect the decomposed metrics, SLA feasibility badges, and physical assumptions.
5. Click **Run Live Validation Benchmark** to verify real performance and calculate relative error.
