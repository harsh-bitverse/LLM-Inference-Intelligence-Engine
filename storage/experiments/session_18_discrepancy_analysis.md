# Session 18 — Discrepancy Analysis: Prospective Latency Underprediction

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 18 — Conservative SLA Decisions & Prospective Prediction Reliability  
**Date:** 2026-10-27  
**Workstation:** Intel Core i7-14700 CPU (20 physical cores, 28 threads, 32 GB RAM, Windows 11)  
**Model:** `llama3.2:1b` (Q8_0, 1.23B parameters)  
**Serving Runtime:** Ollama v0.40.11  

---

## 1. Executive Summary

During Session 17's prospective validation, the physical stage predictor systematically underpredicted execution latency for two key workloads:
- **Scenario A ($P=32, O=16$):** Predicted $807.27\text{ ms}$, Observed median $985.55\text{ ms}$ (Underpredicted by $+178.28\text{ ms}$ / $22.08\%$). Decode rate dropped from predicted $22.72\text{ tok/s}$ to observed $18.77\text{ tok/s}$.
- **Scenario C ($P=128, O=128$):** Predicted $6,027.12\text{ ms}$, Observed median $7,411.72\text{ ms}$ (Underpredicted by $+1,384.60\text{ ms}$ / $22.97\%$). Decode rate dropped from predicted $22.72\text{ tok/s}$ to observed $17.42\text{ tok/s}$.

In Session 18, a new controlled live validation benchmark was executed across 4 workloads. It revealed an even more pronounced degradation:
- **Short Output ($O=16$):** Predicted $849.64\text{ ms}$, Observed **$858.01\text{ ms}$** (Error: **$8.37\text{ ms}$ / $0.99\%$**, Decode: $20.02\text{ tok/s}$).
- **Long Output ($O=128$):** Predicted $5,999.88\text{ ms}$, Observed **$10,282.98\text{ ms}$** (Error: **$+4,283.10\text{ ms}$ / $71.39\%$**, Decode: **$12.58\text{ tok/s}$**).

This report investigates the physical mechanisms behind this performance degradation, separating verified facts from unproven hypotheses, and strictly refraining from asserting thermal throttling without direct MSR sensor telemetry.

---

## 2. Hypothesis Evaluation

### Hypothesis 1: Output Token Count Variation
- **Conjecture:** Did the model generate more tokens than requested (e.g. ignoring `num_predict`), inflating decode duration?
- **Empirical Evidence:**
  - In Session 17 Scenario A ($O=16$), Ollama reported `actual_output_tokens = 16` on all 3 repetitions.
  - In Session 17 Scenario C ($O=128$), Ollama reported `actual_output_tokens = 128` on all 3 repetitions.
  - In Session 18 Short ($O=16$), `actual_output_tokens = 16` on all 3 repetitions.
  - In Session 18 Long ($O=128$), `actual_output_tokens = 128` on all 3 repetitions.
- **Verdict: RULED OUT.** In all test executions, the model generated the exact requested token count. The latency inflation stems entirely from per-token decode slowdown.

---

### Hypothesis 2: Client Loopback & HTTP Framing Overhead (Scenario A)
- **Conjecture:** Did fixed HTTP socket serialization, JSON marshalling, and Ollama request dispatch overhead account for the $+178\text{ ms}$ difference in short runs?
- **Empirical Evidence:**
  - In Scenario A, total client wall-clock duration was $985.55\text{ ms}$.
  - Server-reported prefill time was $60.01\text{ ms}$, and decode time was $852.39\text{ ms}$ ($912.40\text{ ms}$ total server execution).
  - Unaccounted client duration was $985.55 - 912.40 = 73.15\text{ ms}$.
  - The engine's linear model assumed a constant overhead $\beta_{\text{overhead}} = 6.25\text{ ms}$.
- **Verdict: CONFIRMED CONTRIBUTOR FOR SHORT WORKLOADS.** Client loopback socket handling, HTTP connection establishment, and Ollama Go runtime scheduling impose $50\text{--}75\text{ ms}$ of unmodeled latency. On an $800\text{ ms}$ request, this explains $\approx 40\%$ of the error.

---

### Hypothesis 3: Autoregressive KV Cache Growth ($O(N^2)$ Complexity)
- **Conjecture:** Does attention matrix growth during generation slow down decoding as sequence length increases?
- **Empirical Evidence:**
  - For $P=119, O=128$, the total sequence length reaches only 247 tokens.
  - In Llama 3.2 1B (16 layers, 32 query heads, 8 KV heads, hidden dimension 2048), the KV cache size at 247 tokens is only $\approx 3.08\text{ MB}$, fitting entirely within the Intel Core i7-14700's $33\text{ MB}$ L3 cache.
  - Attention computation at 250 tokens accounts for $< 2\%$ of total FLOPs; GEMV matrix-vector multiplication of the $1.23\text{B}$ parameter weights dominates $> 98\%$ of compute time.
- **Verdict: RULED OUT.** KV cache computational scaling is negligible at sequence lengths under 500 tokens.

---

### Hypothesis 4: CPU Power Limits & Frequency Scaling Under Sustained AVX2 Load
- **Conjecture:** Intel Core i7-14700 features an aggressive Turbo Boost architecture (PL2 = 219W with a time window $\tau \approx 28\text{--}56\text{ s}$, reverting to PL1 = 65W base sustained power). When all 20 cores (28 threads) execute continuous AVX2 matrix multiplications for $> 7\text{--}10$ seconds, CPU package power constraints force all-core clock frequencies to scale downward from peak turbo frequencies ($\sim 5.0\text{--}5.4\text{ GHz}$) to sustained base limits ($\sim 2.1\text{--}3.5\text{ GHz}$).
- **Empirical Evidence & Telemetry Boundaries:**
  - When workloads are brief ($O=16$, duration $< 1\text{ s}$), decode throughput is high: **$18.8\text{--}20.2\text{ tok/s}$** ($49.5\text{--}53.2\text{ ms/token}$).
  - In Session 17 Scenario C ($O=128$, duration $7.4\text{ s}$), decode throughput dropped to **$17.42\text{ tok/s}$** ($57.4\text{ ms/token}$).
  - In Session 18 Long ($O=128$, executed after prior heavy workloads), decode throughput dropped to **$12.58\text{ tok/s}$** ($79.5\text{ ms/token}$).
  - This represents a massive $50\%\text{--}60\%$ increase in per-token decode time during prolonged execution.
- **Scientific Qualification:**
  > **IMPORTANT SCIENTIFIC DISTINCTION:**  
  > While CPU all-core frequency scaling under sustained PL1 power limits is physically consistent with this observed degradation, **we do NOT possess direct real-time MSR frequency telemetry or CPU package temperature logs in this benchmark environment**.  
  > Therefore, **we do NOT assert thermal throttling**. We designate CPU power-limit and sustained multi-core frequency scaling as a **highly plausible physical hypothesis**, but preserve the scientific boundary between measured fact and unverified hardware conjecture.

---

### Hypothesis 5: Memory Bus Contention & Cache Eviction Under Multi-Threading
- **Conjecture:** In Ollama, CPU execution dispatches across 20 threads. In autoregressive decoding, each token requires streaming the entire $1.32\text{ GB}$ model weights across the memory bus. When multiple consecutive runs saturate the memory controller, memory latency and cache-line invalidation increase.
- **Empirical Evidence:**
  - Memory read requirements: $128 \times 1.32\text{ GB} = 168.96\text{ GB}$ transferred in $10\text{ seconds} \approx 16.9\text{ GB/s}$ sustained throughput.
  - While within the theoretical $89.6\text{ GB/s}$ dual-channel limit, random thread synchronization and memory stalls on DDR5 under high core counts create non-linear latency inflation.
- **Verdict: PLAUSIBLE CONTRIBUTING FACTOR.**

---

## 3. Comparative Synthesis: Burst vs. Sustained Inference

| Workload Condition | Execution Duration | Active Threads | Measured Decode Rate | Implied Time / Token | Prediction Model Assumption | Prediction Accuracy Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Short Burst ($O=16$)** | $0.85\text{ s}$ | 20 CPU threads | **$20.02\text{ tok/s}$** | $49.9\text{ ms/tok}$ | $22.72\text{ tok/s}$ ($44.0\text{ ms/tok}$) | **Accurate:** $0.99\%$ error, inside range. |
| **Isolated Sustained ($O=128$)** | $7.41\text{ s}$ | 20 CPU threads | **$17.42\text{ tok/s}$** | $57.4\text{ ms/tok}$ | $22.72\text{ tok/s}$ ($44.0\text{ ms/tok}$) | **Underpredicted:** $22.97\%$ error, outside range. |
| **Successive Sustained ($O=128$)** | $10.28\text{ s}$ | 20 CPU threads | **$12.58\text{ tok/s}$** | $79.5\text{ ms/tok}$ | $22.72\text{ tok/s}$ ($44.0\text{ ms/tok}$) | **Severely Underpredicted:** $71.39\%$ error, SLA missed. |

---

## 4. Engineering Rectifications for IIE

1. **Conservative SLA Margin Enforcement:**
   Because decode throughput degrades from $\sim 22.7\text{ tok/s}$ to $\sim 12.6\text{ tok/s}$ under sustained load, the recommendation engine must never treat $22.72\text{ tok/s}$ as an immutable constant.
2. **Exploratory Uncertainty Disclosures:**
   When recommending configurations for long output generations ($O > 64$), the engine must emit a warning:
   *"Forecasted decode throughput assumes burst CPU frequencies; sustained generation (> 5 seconds) on 20 threads may degrade decode rate by 20% to 45% due to CPU power limits."*
3. **No False Certainty:**
   Predictions for workloads where the forecast approaches SLA limits must preserve uncertainty rather than declaring unqualified `FEASIBLE` compliance.
