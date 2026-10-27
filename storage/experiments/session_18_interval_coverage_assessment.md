# Session 18 — Prediction Interval Coverage & Empirical Error Assessment

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 18 — Conservative SLA Decisions & Prospective Prediction Reliability  
**Date:** 2026-10-27  
**Workstation:** Intel Core i7-14700 CPU (20 physical cores, 28 threads, 32 GB RAM, Windows 11)  
**Model:** `llama3.2:1b` (Q8_0, 1.23B parameters)  
**Reference Dataset:** 62 Real Non-Warmup Observations across 17 Configuration Groups  

---

## 1. Executive Summary

In Sessions 16 and 17, IIE reported prediction uncertainty intervals as:
$$\text{Interval} = [\hat{y} - \text{MAE}, \hat{y} + \text{MAE}] \quad \text{where } \text{MAE} = 385.88\text{ ms}$$
These intervals were displayed in API responses, markdown tables, and the React UI as **"95% Confidence Intervals"**.

This empirical assessment evaluated the actual coverage of this interval across all 62 real observations in the empirical dataset using Leave-One-Configuration-Out (LOCO) cross-validation across 17 distinct workload folds.

### Key Findings
1. **True Empirical Coverage of $\pm 1\times \text{MAE}$ ($385.88\text{ ms}$):** Exactly **`46 / 62 = 74.19%`**, NOT 95%.
2. **True Empirical Coverage of $\pm 2\times \text{MAE}$ ($771.76\text{ ms}$):** **`56 / 62 = 90.32%`**.
3. **Margin Required for True 90% Empirical Coverage:** **$\pm 699.81\text{ ms}$**.
4. **Margin Required for True 95% Empirical Coverage:** **$\pm 1,481.25\text{ ms}$** ($\approx 3.84\times \text{MAE}$).
5. **Conclusion:** Presenting $\pm 385.88\text{ ms}$ as a "95% Confidence Interval" was statistically and empirically unjustified. The system has replaced this label with **`exploratory_prediction_range`**, disclosing its exact empirical coverage of $\sim 74\%$.

---

## 2. Experimental Methodology

### 2.1 Dataset Composition
The assessment incorporates all authoritative, real-hardware benchmark observations collected across Sessions 12B through 15:
- `storage/datasets/pilot_llama3_2_1b.jsonl`: 9 measured records.
- `storage/datasets/expansion_llama3_2_1b_v2.jsonl`: 21 measured records.
- `storage/datasets/session_15_cache_controlled_measurements.jsonl`: 32 measured records.
- **Total Population:** 62 eligible records (all warmup and failure runs excluded).

### 2.2 Cross-Validation Split Strategy
The 62 records span 17 unique configuration groups ($P \times O \times \text{Residency} \times \text{Cache}$). To prevent data leakage, evaluation was conducted via **Leave-One-Configuration-Out (LOCO)**:
- For each group $k \in \{1 \dots 17\}$, all records in group $k$ were held out as the test set.
- The `WorkloadAwareLatencyPredictor` was fitted purely on the remaining 16 groups.
- Prospective predictions $\hat{y}_i$ were generated for each held-out record $i \in k$ without access to post-execution runtime fields (`actual_output_tokens`, `prompt_eval_cached_count`, `duration_ms`).
- Residual errors $e_i = |y_i - \hat{y}_i|$ were computed and evaluated against interval bounds.

---

## 3. Empirical Coverage Results

### 3.1 Interval Coverage Summary

| Interval Formulation | Margin ($\pm \Delta$) | Total Interval Width | Records Covered | Empirical Coverage (%) |
| :--- | :--- | :--- | :--- | :--- |
| **$1\times \text{MAE}$ (Previous "95% CI")** | $\pm 385.88\text{ ms}$ | $771.76\text{ ms}$ | **46 / 62** | **74.19%** |
| **$2\times \text{MAE}$ (Session 17 OOD)** | $\pm 771.76\text{ ms}$ | $1,543.52\text{ ms}$ | **56 / 62** | **90.32%** |
| **Empirical 90th Percentile** | $\pm 699.81\text{ ms}$ | $1,399.62\text{ ms}$ | **56 / 62** | **90.32%** |
| **Empirical 95th Percentile** | $\pm 1,481.25\text{ ms}$ | $2,962.50\text{ ms}$ | **59 / 62** | **95.16%** |

### 3.2 Top 10 Largest Prediction Errors Under Held-Out Evaluation

The 10 worst prediction misses illustrate where linear decomposition experiences empirical stress:

| Request ID | Workload Group | Prompt ($P$) | Output ($O$) | Cache State | Actual ($y_i$) | Predicted ($\hat{y}_i$) | Error ($e_i$) | Root Cause Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `req-0012` | `P677_O64` | 677 | 64 | Unspecified | $5,050.6\text{ ms}$ | $3,168.3\text{ ms}$ | **$1,882.3\text{ ms}$** | Unspecified cache state; measured under cold unprimed prefill. |
| `req-0019` | `P677_O64` | 677 | 64 | Unspecified | $4,979.5\text{ ms}$ | $3,168.3\text{ ms}$ | **$1,811.2\text{ ms}$** | Re-dispatch jitter under unprimed prompt cache. |
| `req-0026` | `P677_O64` | 677 | 64 | Unspecified | $4,967.8\text{ ms}$ | $3,168.3\text{ ms}$ | **$1,799.6\text{ ms}$** | Cold prefill delay ($\sim 2,000\text{ ms}$) unmodeled in base fit. |
| `s15-expC-forward-p677` | `P677_O32` | 677 | 32 | `partial_prefix` | $3,235.0\text{ ms}$ | $1,753.7\text{ ms}$ | **$1,481.2\text{ ms}$** | Forward ordering encountered cold prefix ingestion. |
| `req-0008` | `P99_O128` | 99 | 128 | Unspecified | $7,149.6\text{ ms}$ | $5,762.1\text{ ms}$ | **$1,387.5\text{ ms}$** | Long decode (128 tok) experienced CPU all-core frequency decay. |
| `s15-expB-b1` | `P311_O32` | 311 | 32 | `partial_prefix` | $2,484.0\text{ ms}$ | $1,583.4\text{ ms}$ | **$900.6\text{ ms}$** | Partial prefix branch mismatch forced partial prompt re-evaluation. |
| `req-0011` | `P99_O128` | 99 | 128 | Unspecified | $6,461.9\text{ ms}$ | $5,762.1\text{ ms}$ | **$699.8\text{ ms}$** | Sustained autoregressive decode variance. |
| `req-0004` | `P53_O96` | 53 | 96 | Unspecified | $5,016.3\text{ ms}$ | $4,319.0\text{ ms}$ | **$697.3\text{ ms}$** | Mid-range token generation scaling variance. |
| `s15-expD-pair1-cold` | `P119_O32` | 119 | 32 | `cold_prefix` | $3,173.8\text{ ms}$ | $2,494.8\text{ ms}$ | **$679.0\text{ ms}$** | GGUF weight mapping from NVMe storage into RAM delay. |
| `req-0007` | `P53_O96` | 53 | 96 | Unspecified | $4,984.1\text{ ms}$ | $4,319.0\text{ ms}$ | **$665.1\text{ ms}$** | Thread scheduling and loopback socket variance. |

---

## 4. Root Causes of Coverage Discrepancy

Why does the interval $[\hat{y} \pm \text{MAE}]$ only cover $74.19\%$ rather than $95\%$?

1. **Non-Gaussian Residual Distribution:** In a normal distribution, $\pm 1\times \text{MAE}$ ($\approx \pm 0.798\sigma$) covers approximately $57.5\%$ of observations. The fact that IIE covers $74.19\%$ indicates that the error distribution has a sharp central peak with heavy tails.
2. **Heavy-Tailed Tail Effects from Cache and Residency Transitions:**
   - When a 677-token prompt experiences a cold-cache hit, prefill takes $2,048\text{ ms}$. When it hits the KV cache, prefill takes $28\text{ ms}$.
   - This physical bifurcation of $2,020\text{ ms}$ cannot be absorbed by a single symmetric margin of $\pm 386\text{ ms}$.
3. **Absence of Calibrated Conformal Guarantees:** A true 95% confidence interval requires either a rigorous parametric model with verified normality or distribution-free inductive conformal prediction. IIE had neither.

---

## 5. Architectural & UX Rectifications

1. **Schema Renaming:**
   - Field `prediction_uncertainty_interval` description updated to explicit exploratory disclaimer.
   - Field `exploratory_prediction_range` added to `WorkloadRecommendationResponse`.
   - Field `confidence_label_status` set to `"EXPLORATORY_RANGE"`.
   - Field `range_coverage_percent` set to `74.19` (in-distribution) and `90.32` (widened OOD range).
2. **Frontend UI Display:**
   - In `RecommendationsPage.tsx`, removed all mentions of `"95% CI"`.
   - Replaced with: `Exploratory Range: [lower, upper] ms (~74% empirical coverage)`.
3. **Artifact Persistence:**
   - Saved full evaluated records and group metrics to `storage/experiments/session_18_interval_coverage_assessment.json`.
