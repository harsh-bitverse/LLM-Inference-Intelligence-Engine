# Session 18 — Scenario Reconciliation & Feasibility Semantics

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 18 — Conservative SLA Decisions & Prospective Prediction Reliability  
**Date:** 2026-10-27  
**Workstation:** Intel Core i7-14700 CPU (20 physical cores, 28 threads, 32 GB RAM, Windows 11)  
**Model:** `llama3.2:1b` (Q8_0, 1.23B parameters)  
**Runtime:** Ollama v0.40.11  

---

## 1. Executive Summary

In Session 17, IIE established an initial workload recommendation engine. However, rigorous post-session audit revealed two semantic flaws that compromised scientific validity:
1. **Scenario F ($P=4000, O=500$, SLA $\le 30,000\text{ ms}$)** was classified as **`PROVEN_INFEASIBLE`**, asserting that an extrapolated linear point forecast exceeding the SLA mathematically proved the workload could not be served.
2. **Scenario B ($P=677, O=32$, SLA $\le 5,000\text{ ms}$)** was classified as unconditional **`DIRECTLY_MEASURED`**, despite the request specifying runtime cache state as `UNKNOWN`, concealing the $36\times\text{--}65\times$ prefill latency variance between cold-prefix ingestion and KV cache replay.

This report establishes the mathematical and physical foundations correcting these classifications, formalizes the conservative feasibility decision contract, and provides explicit reconciliation for all evaluation scenarios.

---

## 2. Scenario F Reconciliation: Why Extrapolation Cannot Prove Infeasibility

### 2.1 The Session 17 Error
In Session 17, Scenario F evaluated:
- Prompt Length ($P$): 4,000 tokens (Empirical training bound: $[42, 677]$ tokens).
- Expected Output ($O$): 500 tokens (Empirical training bound: $[16, 128]$ tokens).
- Latency SLA: $30,000.0\text{ ms}$ ($30.0\text{ s}$).
- Linear Extrapolated Latency: $\hat{T}_{\text{e2e}} = 34,116.25\text{ ms}$ ($34.12\text{ s}$).

The recommendation engine executed the following logic:
```python
# Flawed Session 17 Logic
if pred_e2e_ms > request.latency_sla_ms:
    sla_assessments["latency_sla"]["status"] = FeasibilityStatus.PROVEN_INFEASIBLE.value
    overall_status = RecommendationStatus.INFEASIBLE
```
Because $34,116.25 > 30,000$, the system labeled the SLA violation as **`PROVEN_INFEASIBLE`**.

### 2.2 Mathematical & Physical Proof of Error
An assertion of "proof" requires either:
1. Direct empirical observation of failure under matching physical conditions, or
2. Violation of an immutable physical law or hardware boundary.

Neither condition was satisfied in Scenario F:
1. **Context Window Constraint:** Llama 3.2 1B supports an official context length of $131,072$ tokens. Total sequence length for Scenario F is $4,000 + 500 = 4,500$ tokens, consuming only $3.43\%$ of the model's architectural capacity.
2. **Memory Capacity:** The Q8_0 quantized weights require $1.32\text{ GB}$ of RAM. KV cache for $4,500$ tokens requires $\approx 56.25\text{ MB}$. Total working set ($< 1.4\text{ GB}$) fits easily within the $32\text{ GB}$ host memory.
3. **Theoretical Compute & Bandwidth Bound:** On dual-channel DDR5-5600 RAM ($\sim 89.6\text{ GB/s}$ peak bandwidth), reading $1.32\text{ GB}$ weights requires a theoretical minimum of $14.73\text{ ms}$ per autoregressive decode token. Generating 500 tokens requires at minimum $7,365\text{ ms}$ ($7.37\text{ s}$). Ingesting 4,000 prompt tokens on 20 CPU threads requires $\approx 800\text{ ms}$ at peak AVX2 throughput. Theoretical minimum execution time is $\approx 8.2\text{ s}$, which is far below the $30.0\text{ s}$ SLA constraint.
4. **Extrapolation Uncertainty:** Because $P=4,000$ and $O=500$ lie far beyond the empirical support envelope ($P_{\text{max}}=677, O_{\text{max}}=128$), the variance of the linear estimator is uncalibrated. A workload of this scale could experience non-linear batching efficiency, thread contention, or AVX-512 vectorization variations.

Therefore, an extrapolated point estimate of $34.1\text{ s}$ cannot mathematically or physically prove that the true execution time will exceed $30.0\text{ s}$. Declaring `PROVEN_INFEASIBLE` was a false certainty claim.

### 2.3 Corrected Classification
Under Session 18 rules:
- **Feasibility Status:** **`UNKNOWN_FEASIBILITY`**.
- **Recommendation Status:** **`INCONCLUSIVE`**.
- **Explanation:** *"Workload is out-of-distribution ($P=4000, O=500$); extrapolated latency ($34,116.3\text{ ms}$) exceeds requested SLA ($30,000.0\text{ ms}$) by $4,116.3\text{ ms}$, but SLA miss cannot be mathematically proven without live empirical benchmarking."*

---

## 3. Scenario B Reconciliation: Conditional vs. Unconditional Evidence

### 3.1 The Session 17 Error
In Session 17, Scenario B evaluated:
- Prompt Length: 677 tokens, Output Length: 32 tokens.
- Request constraints: `expected_model_residency=WARM`, `expected_prefix_cache=UNKNOWN`.
- Evidence Category assigned: **`DIRECTLY_MEASURED`**.

### 3.2 Forensic Audit of Historical Records
In the historical empirical dataset (`session_15_cache_controlled_measurements.jsonl`), two distinct measurements exist for $P=677, O=32$:
1. `s15-expC-forward-p677`: Cache state = `partial_prefix` / `cold_prefix`.
   - Measured Prefill Duration: **$2,048.51\text{ ms}$**.
   - Measured Client E2E Duration: **$3,235.02\text{ ms}$**.
2. `s15-expC-shuffled-p677` / `s15-expC-reverse-p677`: Cache state = `warm_prefix` (KV cache hit).
   - Measured Prefill Duration: **$28.21\text{ ms}$**.
   - Measured Client E2E Duration: **$1,544.85\text{ ms}$**.

When the request passed `expected_prefix_cache = UNKNOWN`, the engine matched `s15-expC-shuffled-p677` ($1,544.85\text{ ms}$) and stamped it `DIRECTLY_MEASURED`. This misled callers into assuming that any 677-token prompt would complete in $1.54\text{ s}$, when an unprimed prompt takes $> 3.2\text{ s}$.

### 3.3 Corrected Classification
- When runtime cache or residency state is `UNKNOWN`, the evidence tier must NOT be unconditional `DIRECTLY_MEASURED`.
- It must be classified as **`DIRECTLY_MEASURED_CONDITIONAL`**.
- Disclosed assumption: *"Direct empirical observation matched conditionally; runtime prefix cache state was requested as UNKNOWN and assumed to be warm KV-cache replay ($1,544.85\text{ ms}$). An unprimed cold cache will add $+1,690\text{ ms}$ prefill delay ($3,235.02\text{ ms}$ total)."*

---

## 4. Formal Feasibility Decision Precedence

The revised engine enforces a strict four-tier hierarchy:

```
[Request Input & Hard Physical Bounds]
                │
                ├─ Context Window Exceeded (>131,072) ───────────► PROVABLY_INFEASIBLE (INFEASIBLE)
                ├─ Memory Bandwidth Lower Bound > SLA ───────────► PROVABLY_INFEASIBLE (INFEASIBLE)
                ├─ Direct Empirical Match Observed > SLA ────────► PROVABLY_INFEASIBLE (INFEASIBLE)
                │
                ├─ Workload Out-Of-Distribution (OOD)? ──────────► UNKNOWN_FEASIBILITY (INCONCLUSIVE)
                ├─ Required Metric Unmeasurable (TTFT)? ─────────► UNKNOWN_FEASIBILITY (INCONCLUSIVE)
                │
                ├─ In-Distribution & Forecast > SLA? ────────────► PREDICTED_SLA_MISS (INFEASIBLE)
                │
                └─ In-Distribution & Forecast <= SLA ────────────► FEASIBLE (RECOMMENDED)
                     (If Headroom < Error Margin: Flag Risk Warning)
```

### Table of Scenario Reconciliations

| Scenario | Workload $(P \times O)$ | SLA Target | Session 17 Status | Session 18 Status | Reconciled Feasibility | Key Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Short Interactive** | $46 \times 16$ | $\le 2500\text{ ms}$ | `RECOMMENDED` | `RECOMMENDED` | `FEASIBLE` | In-distribution; $849.6\text{ ms}$ forecast with $1,650.4\text{ ms}$ headroom. |
| **B: Long Context** | $677 \times 32$ | $\le 5000\text{ ms}$ | `RECOMMENDED` (`DIRECTLY_MEASURED`) | `RECOMMENDED` (`DIRECTLY_MEASURED_CONDITIONAL`) | `FEASIBLE` | Matched record is conditional on KV cache replay; disclosed to user. |
| **C: Long Generation** | $119 \times 128$ | $\le 8000\text{ ms}$ | `RECOMMENDED` | `RECOMMENDED` | `FEASIBLE` (Risk Flag) | In-distribution; narrow headroom flagged against sustained decode scaling. |
| **D: Cold Start** | $119 \times 32$ | $\le 4000\text{ ms}$ | `RECOMMENDED` (`DIRECTLY_MEASURED`) | `RECOMMENDED` (`DIRECTLY_MEASURED`) | `FEASIBLE` | Exact match on cold residency and cold cache; $3,007\text{ ms} \le 4,000\text{ ms}$. |
| **E: Strict TTFT** | $46 \times 32$ | TTFT $\le 100\text{ ms}$ | `INCONCLUSIVE` | `INCONCLUSIVE` | `UNKNOWN_FEASIBILITY` | Text-chunk streaming cannot verify token-level arrival boundaries. |
| **F: Out-of-Distribution** | $4000 \times 500$ | $\le 30000\text{ ms}$ | `INFEASIBLE` (`PROVEN_INFEASIBLE`) | `INCONCLUSIVE` | `UNKNOWN_FEASIBILITY` | Extrapolated linear forecast cannot prove SLA violation without live test. |
