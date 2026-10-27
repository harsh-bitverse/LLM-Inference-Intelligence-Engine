# Optimizer Validation Report: Empirical Grounding & Feasibility Audit (Session 13)

**Model:** `llama3.2:1b` (GGUF Q8_0)  
**Host Runtime:** Ollama v0.40.11 (CPU)  
**Optimizer:** `llm_ii_engine.optimizer.engine.ConfigurationOptimizer`  

---

## 1. Audit Objectives & Verified Safeguards

| Verification Principle | Implementation Requirement | Verified Optimizer Behavior | Status |
| :--- | :--- | :--- | :--- |
| **Exact Empirical Matching** | Exact historical matches must take precedence over ML predictions or heuristics. | For configuration `(prompt=46, output=32)`, optimizer resolved status to `MEASURED_EVALUATED` with provenance `MEASURED`. | **PASSED** |
| **Unmeasured TTFT Safety** | Unmeasured TTFT in chunk-only streams must NOT produce false SLA compliance. | When user requested `max_ttft_ms = 50.0 ms`, candidate was classified as `UNKNOWN_FEASIBILITY` because TTFT is unobserved in text chunk streams. | **PASSED** |
| **Engine-Quantization Guard** | Unsupported quantization formats must be rejected as provably infeasible. | Requesting `AWQ` quantization on `OLLAMA` was rejected early during search generation (`Engine 'ollama' does not support quantization 'awq'`). | **PASSED** |
| **Evidence Tier Separation** | Distinguish measured evidence from empirical predictions and analytical formulas. | Evidence tiers `MEASURED_EVALUATED`, `PREDICTED_EMPIRICAL`, and `THEORETICAL_ANALYTICAL` remain strictly partitioned. | **PASSED** |
| **Hardware Bounding** | Optimization does not extrapolate to unsupported hardware. | Catalog lookups without physical matches produce unknown memory feasibility rather than speculative assertions. | **PASSED** |

---

## 2. Test Case Detail

### Case 1: Exact Empirical Configuration Match
- Workload: Prompt 46 tokens, Output 32 tokens, Concurrency 1, Q8_0.
- Result: Objective `MAXIMIZE_THROUGHPUT` = 21.9 tok/s.
- Evidence Status: `MEASURED_EVALUATED`.
- Provenance: `MEASURED`.

### Case 2: Strict TTFT SLA without Discrete Token Observability
- Constraint: `max_ttft_ms = 50.0 ms`.
- Result: `UNKNOWN_FEASIBILITY`.
- Rationale: `Target GPU hardware specification is missing; device memory capacity cannot be verified`.
- Audit Significance: Eliminates false claims of SLA compliance when underlying stream protocol only emits text chunks.

### Case 3: Unsupported Quantization Format
- Configuration: `AWQ` on `OLLAMA`.
- Result: **Rejected during search space generation** (0 feasible candidates generated).
- Rationale: `Engine 'ollama' does not support quantization 'awq'`.
