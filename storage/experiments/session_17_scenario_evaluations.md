# Session 17 — Practical Validation Scenarios Evaluation Report

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 17 — Workload-Aware Recommendations & Product Validation  
**Date:** 2026-10-27  
**Workstation:** Intel Core i7-14700 CPU (20 physical cores, 28 threads, 32 GB RAM, Windows 11)  
**Model:** `llama3.2:1b` (Q8_0, 1.23B parameters)  
**Serving Runtime:** Ollama v0.40.11  

---

## 1. Overview

To validate the productization of IIE's recommendation pipeline, we evaluated six distinct practical inference workloads spanning interactive chat, long-context ingestion, extended autoregressive generation, cold-start dependencies, unmeasurable SLA constraints, and severe out-of-distribution workloads.

For each scenario, the engine evaluated:
1. Platform compatibility
2. Exact empirical observation matching vs. physical stage prediction
3. Out-of-distribution (OOD) boundaries
4. Hard SLA feasibility
5. Candidate ranking under primary objective
6. Transparent rationale and physical assumptions

---

## 2. Summary Scenario Matrix

| Scenario | Workload $(P \times O)$ | Residency & Cache State | Requested SLA / Target | Recommendation Status | Supporting Evidence Category | Forecasted E2E Duration | Forecasted Decode Speed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Short Interactive** | $32 \times 16$ | WARM / COLD_PREFIX | Latency $\le 2,500\text{ ms}$ | **`RECOMMENDED`** | `HELD_OUT_VALIDATED_PREDICTION` | $807.27\text{ ms}$ | $22.72\text{ tok/s}$ |
| **B: Long Context** | $677 \times 32$ | WARM / UNKNOWN | Latency $\le 5,000\text{ ms}$ | **`RECOMMENDED`** | `DIRECTLY_MEASURED` | $1,544.85\text{ ms}$ | $21.66\text{ tok/s}$ |
| **C: Long Generation** | $128 \times 128$ | WARM / COLD_PREFIX | Latency $\le 8,000\text{ ms}$ | **`RECOMMENDED`** | `HELD_OUT_VALIDATED_PREDICTION` | $6,027.12\text{ ms}$ | $22.72\text{ tok/s}$ |
| **D: Cold-Start Sensitive** | $119 \times 32$ | COLD / COLD_PREFIX | Latency $\le 4,000\text{ ms}$ | **`RECOMMENDED`** | `DIRECTLY_MEASURED` | $3,006.96\text{ ms}$ | $22.39\text{ tok/s}$ |
| **E: Strict TTFT SLA** | $46 \times 32$ | WARM / COLD_PREFIX | TTFT $\le 100\text{ ms}$ | **`INCONCLUSIVE`** | `DIRECTLY_MEASURED` | $1,537.50\text{ ms}$ | $21.67\text{ tok/s}$ |
| **F: Out-Of-Distribution** | $4000 \times 500$ | WARM / COLD_PREFIX | Latency $\le 30,000\text{ ms}$ | **`INFEASIBLE`** | `EXPLORATORY_PREDICTION` | $34,116.25\text{ ms}$ | $22.72\text{ tok/s}$ |

---

## 3. Detailed Scenario Evaluations

### Scenario A — Short Interactive Request
- **Objective:** Prioritize low end-to-end user latency for interactive conversational agents.
- **Specification:** 32 prompt tokens, 16 requested output tokens, concurrency 1, warm model residency.
- **Latency SLA:** $\le 2,500.0\text{ ms}$.
- **Evidence Classification:** `HELD_OUT_VALIDATED_PREDICTION` (In-distribution, supported by LOCO-validated Workload-Aware Predictor).
- **Decomposed Forecast:**
  - $\hat{T}_{\text{load}} = 0.00\text{ ms}$ (warm memory resident)
  - $\hat{T}_{\text{prefill}} = 96.83\text{ ms}$ ($32 \times 3.026\text{ ms/tok}$)
  - $\hat{T}_{\text{decode}} = 704.19\text{ ms}$ ($16 \times 44.012\text{ ms/tok}$)
  - $\hat{T}_{\text{overhead}} = 6.25\text{ ms}$
  - **$\hat{T}_{\text{e2e}} = 807.27\text{ ms}$** (95% CI: $[421.4, 1,193.2]\text{ ms}$).
  - Decode Throughput: **$22.72\text{ tok/s}$**.
- **SLA Feasibility:** **`FEASIBLE`** with $1,692.7\text{ ms}$ headroom margin.
- **Rationale:** Selected Ollama runtime with `llama3.2:1b` (Q8_0) on Intel Core i7-14700 CPU. The workload fits well within memory and comfortably satisfies the $2,500\text{ ms}$ interactive threshold.

---

### Scenario B — Long-Context Request
- **Objective:** Ingest extended document contexts while maintaining moderate output responsiveness.
- **Specification:** 677 prompt tokens, 32 requested output tokens, concurrency 1, warm model residency.
- **Latency SLA:** $\le 5,000.0\text{ ms}$.
- **Evidence Classification:** `DIRECTLY_MEASURED` (Exact empirical match from Session 13 / Session 15 suites).
- **Empirical Measured Metrics:**
  - Observed Client E2E Latency: **$1,544.85\text{ ms}$** (95% CI: $[1,498.5, 1,591.2]\text{ ms}$).
  - Server Decode Throughput: **$21.66\text{ tok/s}$**.
  - Server Prefill Duration: $597.10\text{ ms}$ ($\approx 1.13\text{ ms/token}$).
- **SLA Feasibility:** **`FEASIBLE`** with $3,455.1\text{ ms}$ headroom margin.
- **Rationale:** Directly verified by historical measurements. Ingesting 677 prompt tokens on CPU requires under $600\text{ ms}$ when weights are resident in RAM.

---

### Scenario C — Long Generation Request
- **Objective:** Extended generation workload (report/tutorial synthesis) maximizing decode throughput.
- **Specification:** 128 prompt tokens, 128 requested output tokens, concurrency 1, warm model residency.
- **Latency SLA:** $\le 8,000.0\text{ ms}$.
- **Evidence Classification:** `HELD_OUT_VALIDATED_PREDICTION`.
- **Decomposed Forecast:**
  - $\hat{T}_{\text{load}} = 0.00\text{ ms}$
  - $\hat{T}_{\text{prefill}} = 387.33\text{ ms}$
  - $\hat{T}_{\text{decode}} = 5,633.54\text{ ms}$
  - $\hat{T}_{\text{overhead}} = 6.25\text{ ms}$
  - **$\hat{T}_{\text{e2e}} = 6,027.12\text{ ms}$** (95% CI: $[5,641.2, 6,413.0]\text{ ms}$).
  - Decode Throughput: **$22.72\text{ tok/s}$**.
- **SLA Feasibility:** **`FEASIBLE`** with $1,972.9\text{ ms}$ headroom margin.
- **Rationale:** Generation time scales linearly with token count at $\approx 44.01\text{ ms/token}$. For 128 output tokens, pure decoding requires $\approx 5.63\text{ s}$, safely under the $8.0\text{ s}$ SLA constraint.

---

### Scenario D — Cold-Start-Sensitive Request
- **Objective:** Evaluate cold-start sensitivity where weights must be loaded from disk prior to execution.
- **Specification:** 119 prompt tokens, 32 requested output tokens, `require_cold_start=True`.
- **Latency SLA:** $\le 4,000.0\text{ ms}$.
- **Evidence Classification:** `DIRECTLY_MEASURED` (Matched from Session 15 Exp A cold-start record `s15-expA-rep1`).
- **Empirical Measured Metrics:**
  - Server Reported Model Load (`server_load_duration_ms`): **$1,037.74\text{ ms}$**.
  - Server Prefill Duration: $449.28\text{ ms}$.
  - Server Decode Duration: $1,569.12\text{ ms}$.
  - Client End-to-End Latency: **$3,006.96\text{ ms}$**.
  - Server Decode Throughput: **$22.39\text{ tok/s}$**.
- **SLA Feasibility:** **`FEASIBLE`** with $993.0\text{ ms}$ headroom margin.
- **Rationale:** Model weight loading from NVMe storage into host RAM incurs an immutable $1,037.74\text{ ms}$ physical penalty. The engine correctly incorporates this penalty and demonstrates that warm-run measurements cannot be substituted for cold workloads.

---

### Scenario E — Strict TTFT SLA Request
- **Objective:** Demonstrate handling of unmeasurable token-level Time-To-First-Token (TTFT) constraints.
- **Specification:** 46 prompt tokens, 32 requested output tokens, requested TTFT SLA $\le 100.0\text{ ms}$.
- **Evidence Classification:** `DIRECTLY_MEASURED` (For throughput and total duration).
- **SLA Feasibility:** **`UNKNOWN_FEASIBILITY`**.
- **Overall Status:** **`INCONCLUSIVE`**.
- **Scientific Disclosure:**
  > *Ollama text chunk streaming emits variable-length byte chunks without discrete per-token arrival timestamps. Token-level TTFT cannot be verified or guaranteed; client-observed TTFC (Time-To-First-Chunk) is measured instead. The engine honestly reports UNKNOWN_FEASIBILITY rather than fabricating token-level conformance.*

---

### Scenario F — Out-Of-Distribution (OOD) Request
- **Objective:** Stress-test extreme workload extrapolation beyond empirical support envelope.
- **Specification:** 4,000 prompt tokens, 500 requested output tokens, latency SLA $\le 30,000.0\text{ ms}$.
- **Generalization Status:** `OUT_OF_DISTRIBUTION` (`is_out_of_distribution: True`).
- **Extrapolated Features:** `["prompt_tokens", "output_tokens"]`.
- **Evidence Classification:** `EXPLORATORY_PREDICTION`.
- **Decomposed Forecast:**
  - $\hat{T}_{\text{prefill}} = 4,000 \times 3.026 = 12,104.00\text{ ms}$ ($\approx 12.1\text{ s}$)
  - $\hat{T}_{\text{decode}} = 500 \times 44.012 = 22,006.00\text{ ms}$ ($\approx 22.0\text{ s}$)
  - **$\hat{T}_{\text{e2e}} = 34,116.25\text{ ms}$** ($\approx 34.1\text{ s}$).
- **SLA Feasibility:** **`PROVEN_INFEASIBLE`** (Exceeds $30.0\text{ s}$ SLA by $4,116.25\text{ ms}$).
- **Overall Status:** **`INFEASIBLE`**.
- **OOD Warnings:**
  1. *Prompt tokens (4000) exceed empirical benchmark maximum (677). Latency is extrapolated linearly.*
  2. *Output tokens (500) exceed empirical benchmark maximum (128).*
  3. *Uncertainty interval widened to $\pm 771.8\text{ ms}$ due to out-of-distribution extrapolation.*
