# SLA Feasibility and Empirical Evidence Policy

**Project:** LLM Inference Intelligence Engine (IIE)  
**Standard Version:** 1.0 (Session 18)  
**Classification:** Authoritative Technical Standard  

---

## 1. Scope and Purpose

Inference engine recommendations must maintain absolute scientific integrity. Overly confident feasibility declarations, unverified performance bounds, and conflation of statistical forecasts with empirical proofs lead to production outages and degraded customer SLAs.

This document establishes the binding engineering contract governing:
1. Proof requirements vs. empirical predictions.
2. Evidence classification tiers and qualification rules.
3. SLA feasibility assessment statuses.
4. Scientific uncertainty representation and disclaimer requirements.

---

## 2. Strict Definitions: Proof vs. Prediction

### 2.1 The Principle of Empirical Proof
A performance or resource claim is **PROVEN** if and only if one of the following criteria is met:
1. **Physical Impossibility Proof:** The requested workload or configuration violates a physical hardware limit or mathematical invariant (e.g., prompt tokens exceed maximum context window, model weights exceed available RAM, or requested response latency is shorter than the theoretical minimum data transfer time across the memory bus).
2. **Direct Empirical Verification Proof:** An identical workload running on identical hardware, serving runtime, quantization format, and memory/cache state was empirically measured and observed to violate the constraint during controlled benchmarking.

### 2.2 The Principle of Predictive Uncertainty
Any numerical value derived from statistical regression, neural models, heuristic scaling formulas, or out-of-distribution extrapolation is a **PREDICTION**, never a proof:
- An in-distribution prediction exceeding an SLA is a **`PREDICTED_SLA_MISS`**, not a proven failure.
- An out-of-distribution prediction exceeding an SLA has uncalibrated uncertainty and must be labeled **`UNKNOWN_FEASIBILITY`**.
- Predictions cannot be used to declare unconditional feasibility when SLA headroom is narrower than the empirical error margin.

---

## 3. Scientific Evidence Classification Tiers

Every recommendation and performance forecast emitted by IIE must be stamped with exactly one evidence tier:

| Evidence Tier | Qualification Criteria | Deployment Trust Level |
| :--- | :--- | :--- |
| **`DIRECTLY_MEASURED`** | Exact match on: Model ID, Quantization, Serving Runtime, Host Hardware, Concurrency, Prompt Tokens ($P$), Output Tokens ($O$), Memory Residency State, AND KV Prefix Cache State. | Authoritative Ground Truth |
| **`DIRECTLY_MEASURED_CONDITIONAL`** | Exact match on Model, Hardware, $P$, and $O$, but runtime state (residency or prefix cache) was requested as `UNKNOWN` or differs from the historical benchmark conditions. | High, with mandatory assumption disclosures |
| **`HELD_OUT_VALIDATED_PREDICTION`** | Workload falls strictly within the empirical interpolation envelope ($P \in [P_{\text{min}}, P_{\text{max}}]$, $O \in [O_{\text{min}}, O_{\text{max}}]$) and is evaluated by a model validated under Leave-One-Configuration-Out (LOCO) cross-validation. | Medium; supported by empirical error margins |
| **`EXPLORATORY_PREDICTION`** | Workload exceeds empirical bounds ($P > P_{\text{max}}$, $O > O_{\text{max}}$, or Concurrency $> 1$). Relies on linear or analytical extrapolation. | Low; uncalibrated exploratory forecast |
| **`ANALYTICAL_ESTIMATE`** | Derived from roofline FLOPS, parameter counts, or memory bandwidth equations without empirical regression fitting. | Informational baseline only |
| **`INSUFFICIENT_EVIDENCE`** | Model is not installed/authorized, or essential runtime hardware specifications are missing. | Rejected / Unsupported |

---

## 4. SLA Feasibility Status Taxonomy

When assessing candidate configurations against hard user SLAs (Latency, Decode Throughput, TTFT), the engine must assign one of four authoritative statuses:

### 4.1 `PROVABLY_INFEASIBLE`
Reserved exclusively for deterministic physical or empirical violations:
1. **Context Window Exceeded:** $P + O > \text{Context Window}$ (e.g., $> 131,072$ for Llama 3.2).
2. **Hardware Memory Exceeded:** Model parameter footprint $> \text{Host Memory Capacity}$.
3. **Memory Bus Bandwidth Bound:** Latency SLA $< \frac{O \times \text{Model Size}}{\text{Peak Bandwidth}}$.
4. **Empirically Measured Failure:** Directly observed execution under exact matching conditions exceeded SLA.

### 4.2 `PREDICTED_SLA_MISS`
Assigned when an in-distribution performance prediction indicates an SLA violation, but no independent physical constraint is broken:
- Workload must be in-distribution ($is\_ood = \text{False}$).
- Recommendation Status is set to **`INFEASIBLE`**.
- The specific forecast deficit must be reported (e.g., *"Forecasted latency of 3,700 ms exceeds SLA of 2,500 ms by 1,200 ms"*).

### 4.3 `UNKNOWN_FEASIBILITY`
Assigned whenever the available evidence is insufficient to guarantee or disprove SLA compliance:
1. **Out-of-Distribution Inputs:** Any workload where $is\_ood = \text{True}$. Extrapolated linear models have high uncertainty and cannot confirm an SLA miss or guarantee satisfaction.
2. **Unmeasurable Metrics:** Requests specifying token-level TTFT on streaming protocols (such as Ollama text-chunk streaming) that emit byte chunks without verified discrete token timestamps.
3. **Overlapping Uncertainty Intervals:** In-distribution forecasts where the prediction interval spans across the SLA threshold.
- Recommendation Status is set to **`INCONCLUSIVE`**.
- User must be presented with a live validation benchmark payload to empirically verify behavior.

### 4.4 `FEASIBLE`
Assigned only when the predicted or directly measured performance satisfies the requested SLA with positive headroom.
- If headroom is narrower than the empirical error margin ($\text{Headroom} < \text{MAE}$), the engine must attach a **`Narrow SLA Headroom Risk Warning`** disclosing that satisfaction is sensitive to runtime jitter.

---

## 5. Uncertainty Representation & Confidence Claims

### 5.1 Prohibition of Misleading "95% Confidence Interval" Labels
Under empirical testing across all 62 historical benchmark observations on Intel Core i7-14700 CPU, evaluating prospective predictions under Leave-One-Configuration-Out (LOCO) cross-validation demonstrated that:
- The interval $[\hat{y} - \text{MAE}, \hat{y} + \text{MAE}]$ ($[\hat{y} - 385.88, \hat{y} + 385.88]\text{ ms}$) achieves **74.19% empirical coverage**, NOT 95%.
- Achieving true 90% empirical coverage requires an error margin of $\pm 699.81\text{ ms}$.
- Achieving true 95% empirical coverage requires an error margin of $\pm 1,481.25\text{ ms}$ (nearly $4\times \text{MAE}$).

### 5.2 Mandatory Labeling Standard
All code, API schemas, UI displays, and technical documentation must:
1. Label prediction intervals as **`exploratory_prediction_range`** or **`empirical_error_margin`**.
2. Explicitly disclose the empirical coverage level (e.g., *"~74% empirical coverage across historical held-out evaluations"*).
3. Strictly forbid the use of the term "95% Confidence Interval" unless supported by a formal, calibrated conformal prediction pipeline.
