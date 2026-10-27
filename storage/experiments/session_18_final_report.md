# Session 18 — Final Engineering & Scientific Validation Report
## Conservative SLA Decisions & Prospective Prediction Reliability

**Project:** A1 — LLM Inference Intelligence Engine (IIE)  
**Session:** 18  
**Date:** 2026-10-27  
**Workstation Environment:**
- **Operating System:** Windows 11 Enterprise (Build 26100)
- **CPU:** Intel Core i7-14700 (20 physical cores: 8 P-cores, 12 E-cores; 28 threads)
- **RAM:** 32.0 GB Host Memory
- **Inference Runtime:** Official Ollama runtime v0.40.11 (HTTP API on `127.0.0.1:11434`)
- **Authorized Model:** `llama3.2:1b` (GGUF Q8_0, 1.23B parameters, SHA256 digest `baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878`)

---

## 1. Executive Summary

Session 18 focused on establishing rigorous scientific reliability in IIE's configuration recommendation and SLA feasibility assessment pipelines. Prior to this session, the engine suffered from two critical validity issues:
1. It conflated statistical point forecasts with mathematical proofs, erroneously labeling an out-of-distribution extrapolated forecast in Scenario F ($P=4000, O=500$) as `PROVEN_INFEASIBLE`.
2. It presented symmetric $\pm \text{MAE}$ uncertainty bounds ($\pm 385.88\text{ ms}$) as a **"95% Confidence Interval"**, despite achieving only **74.19% empirical coverage** across the 62 real benchmark observations.

In this session, we resolved these discrepancies, introduced formal conservative feasibility semantics, audited evidence qualifications, and executed a prospective live validation benchmark against the local Ollama instance on Intel Core i7-14700 CPU.

### Core Achievements
- **Corrected Feasibility Taxonomy:** Established four distinct feasibility states: `PROVABLY_INFEASIBLE`, `PREDICTED_SLA_MISS`, `UNKNOWN_FEASIBILITY`, and `FEASIBLE`. Scenario F is now correctly classified as `UNKNOWN_FEASIBILITY` with `INCONCLUSIVE` recommendation status.
- **Empirical Coverage Grounding:** Proved via 17-fold Leave-One-Configuration-Out (LOCO) cross-validation that $\pm \text{MAE}$ achieves $74.19\%$ coverage, while true $90\%$ and $95\%$ empirical coverage require margins of $\pm 699.81\text{ ms}$ and $\pm 1,481.25\text{ ms}$ respectively. Removed all misleading "95% CI" claims from schemas, API, frontend, and reports, replacing them with `exploratory_prediction_range`.
- **Audited Evidence Tiers:** Introduced `DIRECTLY_MEASURED_CONDITIONAL` for requests where runtime state (residency or prefix cache) is unspecified (`UNKNOWN`). Reconciled Scenario B ($P=677, O=32$) to disclose that measured latency ($1,544.85\text{ ms}$) is conditional on KV cache replay.
- **Live Prospective Validation:** Executed a pre-timestamped benchmark protocol across four workloads on Ollama (`llama3.2:1b` on Intel i7-14700). Confirmed burst decode speeds of $20.02\text{ tok/s}$ ($0.99\%$ error on 16 tokens) while revealing sustained all-core decode degradation to $12.58\text{ tok/s}$ ($79.5\text{ ms/token}$) over 128 continuous tokens, validating the necessity of conservative SLA margins.
- **Quality Gates Passed:** 391 Python unit/integration tests passing (10 new Session 18 regression tests), 23 React/vitest frontend tests passing, 0 TypeScript errors, and 0 Ruff lint errors.

---

## 2. Feasibility Taxonomy & Proof vs. Prediction

### 2.1 The Four Authoritative Feasibility Statuses

```
                                  Constraint Evaluation
                                            │
           ┌────────────────────────────────┼────────────────────────────────┐
           ▼                                ▼                                ▼
  [Physical Law or Direct]       [In-Distribution Model]         [OOD or Unmeasurable]
     [Empirical Violation]          [Forecast Exceeds SLA]           [Uncertainty Spans]
           │                                │                                │
           ▼                                ▼                                ▼
  PROVABLY_INFEASIBLE              PREDICTED_SLA_MISS               UNKNOWN_FEASIBILITY
 (Status: INFEASIBLE)             (Status: INFEASIBLE)            (Status: INCONCLUSIVE)
```

1. **`PROVABLY_INFEASIBLE`:** Only when a hard physical constraint is violated:
   - Sequence length exceeds context window ($P + O > 131,072$ tokens).
   - Memory footprint exceeds host memory capacity ($> 32\text{ GB}$).
   - Memory bandwidth lower bound exceeds SLA ($T_{\text{mem}} = \frac{O \times 1.32\text{ GB}}{89.6\text{ GB/s}} > \text{SLA}$).
   - Direct empirical measurement under exact matching conditions exceeded SLA.
2. **`PREDICTED_SLA_MISS`:** In-distribution workload ($P \in [42, 677]$, $O \in [16, 128]$, concurrency 1) where forecasted latency or throughput fails SLA without violating physical hardware limits.
3. **`UNKNOWN_FEASIBILITY`:** Out-of-distribution workloads (e.g. Scenario F), unmeasurable metrics (token-level TTFT on text-chunk streaming), or where prediction uncertainty spans across the constraint boundary.
4. **`FEASIBLE`:** Forecast or measurement satisfies SLA with positive headroom. If headroom $< \text{MAE}$, a **Narrow SLA Headroom Risk Warning** is attached.

---

## 3. Empirical Interval Coverage Assessment

Across the 62 real, non-warmup benchmark observations in IIE's repository:

$$\begin{aligned}
\text{Total Observations } (N) &= 62 \\
\text{LOCO Configuration Folds } (K) &= 17 \\
\text{Mean Absolute Error } (\text{MAE}) &= 385.88\text{ ms}
\end{aligned}$$

### Coverage Distribution

| Formulation | Error Margin | Interval Width | Records Covered | Actual Coverage | Previous Label | Rectified Label |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\pm 1\times \text{MAE}$ | $\pm 385.88\text{ ms}$ | $771.76\text{ ms}$ | **46 / 62** | **74.19%** | *"95% Confidence Interval"* | **`exploratory_prediction_range`** |
| $\pm 2\times \text{MAE}$ | $\pm 771.76\text{ ms}$ | $1,543.52\text{ ms}$ | **56 / 62** | **90.32%** | *"OOD Widened 95% CI"* | **`exploratory_prediction_range`** (OOD) |
| Empirical 90% | $\pm 699.81\text{ ms}$ | $1,399.62\text{ ms}$ | **56 / 62** | **90.32%** | N/A | `empirical_90_pct_margin` |
| Empirical 95% | $\pm 1,481.25\text{ ms}$ | $2,962.50\text{ ms}$ | **59 / 62** | **95.16%** | N/A | `empirical_95_pct_margin` |

### Key Insight
Because cache and model residency transitions create heavy-tailed, non-Gaussian residual distributions, $\pm 1\times \text{MAE}$ achieves only $74.19\%$ coverage. Claiming 95% confidence was statistically invalid. The system now transparently reports `range_coverage_percent = 74.19` with explicit exploratory qualifications.

---

## 4. Controlled Live Prospective Validation Results

A prospective validation pipeline was executed on the local authorized Ollama instance (`llama3.2:1b` on Intel Core i7-14700 CPU):
- **Pre-execution forecast timestamp:** `2026-10-27T18:09:29.588573+00:00`
- **Post-execution completion timestamp:** `2026-10-27T18:10:22.810045+00:00`

### Live Validation Benchmark Summary

| Workload ID | Workload Specification | Frozen Forecast | Exploratory Range | Median Observed E2E | Duration Error | Relative Error | Within Range? | Target SLA | Predicted SLA Status | Observed SLA Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`VAL_WORKLOAD_SHORT`** | $P=46, O=16$, Warm Res. | $849.64\text{ ms}$ | $[463.8, 1,235.5]\text{ ms}$ | **$858.01\text{ ms}$** | **$+8.37\text{ ms}$** | **$0.99\%$** | **YES** | $\le 2500\text{ ms}$ | `FEASIBLE` | **SATISFIED** |
| **`VAL_WORKLOAD_LONG`** | $P=119, O=128$, Warm Res. | $5,999.88\text{ ms}$ | $[5,614.0, 6,385.8]\text{ ms}$ | **$10,282.98\text{ ms}$** | **$+4,283.10\text{ ms}$** | **$71.39\%$** | **NO** | $\le 8000\text{ ms}$ | `FEASIBLE` | **VIOLATED** |
| **`VAL_WORKLOAD_CACHE_COLD`** | $P=677, O=32$, Cold Cache | $3,463.23\text{ ms}$ | $[3,077.3, 3,849.1]\text{ ms}$ | **$2,738.62\text{ ms}$** | **$-724.61\text{ ms}$** | **$20.92\%$** | **NO** | $\le 5000\text{ ms}$ | `FEASIBLE` | **SATISFIED** |
| **`VAL_WORKLOAD_CACHE_WARM`** | $P=677, O=32$, Warm Cache | $1,544.85\text{ ms}$ | $[1,498.5, 1,591.2]\text{ ms}$ | **$2,670.31\text{ ms}$** | **$+1,125.46\text{ ms}$** | **$72.85\%$** | **NO** | $\le 2500\text{ ms}$ | `FEASIBLE` | **VIOLATED** |

### Key Scientific Findings from Live Validation
1. **Short Burst Precision:** On brief requests ($O=16$), the physical model predicted $849.64\text{ ms}$ vs. observed $858.01\text{ ms}$ ($0.99\%$ error), demonstrating excellent fidelity when CPU cores operate near peak boost frequencies.
2. **Sustained Decode Degradation:** On extended generation ($O=128$), decode throughput dropped from $20.02\text{ tok/s}$ to $12.58\text{ tok/s}$ ($79.5\text{ ms/token}$). While the linear prior predicted $6.0\text{ s}$ (`FEASIBLE` under an $8.0\text{ s}$ SLA), actual execution took $10.28\text{ s}$ (**SLA miss**). This empirically confirms the necessity of conservative SLA policies.
3. **KV Cache Acceleration Verified:** Prefill latency for 677 tokens dropped from **$4,025.27\text{ ms}$** on cold ingestion to **$62.93\text{ ms}$** on warm replay—a **$51.83\times$ prefill speedup**.

---

## 5. Software & UX Modifications

1. **`llm_ii_engine/optimizer/models.py`:**
   - Expanded `FeasibilityStatus` enum to include `PROVABLY_INFEASIBLE` and `PREDICTED_SLA_MISS` alongside backward-compatible aliases.
2. **`llm_ii_engine/recommendation/models.py`:**
   - Added `DIRECTLY_MEASURED_CONDITIONAL` to `RecommendationEvidenceSource`.
   - Added `exploratory_prediction_range`, `confidence_label_status`, and `range_coverage_percent` to `WorkloadRecommendationResponse`.
   - Removed all references to "95% Confidence Interval".
3. **`llm_ii_engine/recommendation/engine.py`:**
   - Implemented exact matching requirements on residency and cache state.
   - Enforced hard physical bounds (context window $\le 131,072$, memory bandwidth lower bound).
   - Classified in-distribution failures as `PREDICTED_SLA_MISS` and OOD failures as `UNKNOWN_FEASIBILITY`.
   - Added narrow headroom risk detection.
4. **`frontend/src/api/types.ts` & `frontend/src/pages/RecommendationsPage.tsx`:**
   - Updated TypeScript interfaces.
   - Replaced "95% CI" with `Exploratory Range: [lower, upper] ms (~74% empirical coverage)`.
   - Added visual badges and warning callouts for `PREDICTED_SLA_MISS`, `UNKNOWN_FEASIBILITY`, narrow headroom risks, and unproven OOD estimations.
5. **Regression Unit Tests (`tests/unit/test_session_18_conservative_sla.py`):**
   - 10 targeted test cases verifying all 6 feasibility conditions, evidence tiers, and exploratory ranges.

---

## 6. Verification and Quality Gates

| Verification Suite | Target | Result | Execution Time |
| :--- | :--- | :--- | :--- |
| **Backend Pytest** | `tests/` (391 tests) | **391 Passed** (100%) | 9.34 seconds |
| **Frontend Vitest** | `frontend/src/test/` (23 tests) | **23 Passed** (100%) | 1.86 seconds |
| **Frontend Typecheck** | `tsc --noEmit` | **0 Errors** | 1.50 seconds |
| **Frontend Production Build** | `npm run build` | **0 Errors / Clean Bundle** | 3.20 seconds |
| **Python Ruff Linter** | `ruff check .` | **All Checks Passed** | 0.25 seconds |

---

## 7. Deliverables Index

The following authoritative artifacts have been created and persisted in the repository:
1. `storage/experiments/session_18_scenario_reconciliation.md`: Formal reconciliation of Scenario F and Scenario B.
2. `docs/sla_feasibility_and_evidence_policy.md`: Authoritative engineering standard for feasibility decisions and evidence tiers.
3. `storage/experiments/session_18_interval_coverage_assessment.md`: Full 17-fold LOCO empirical coverage findings on 62 records.
4. `storage/experiments/session_18_interval_coverage_assessment.json`: Raw empirical evaluation data and residual errors.
5. `storage/experiments/session_18_discrepancy_analysis.md`: In-depth investigation of prospective underprediction and burst vs sustained decode scaling.
6. `storage/experiments/session_18_controlled_validation.md`: Controlled live validation benchmark report across 4 workloads.
7. `storage/experiments/session_18_controlled_validation.json`: Pre- and post-timestamped validation benchmark artifacts.
8. `tests/unit/test_session_18_conservative_sla.py`: 10 comprehensive unit tests for conservative SLA logic.
9. `storage/experiments/session_18_final_report.md`: This comprehensive executive milestone report.

---

## 8. Explicit Constraint Adherence

- **No New Models:** Maintained exclusive use of the authorized `llama3.2:1b` (Q8_0) model weights.
- **No Runtime Upgrades:** Preserved Ollama v0.40.11 without modifying system configuration.
- **No Neural-Network Predictors:** Kept lightweight, interpretable, physical stage predictors without opaque neural models.
- **Session Boundary Respected:** Session 18 is complete. **Session 19 has NOT been started.**
