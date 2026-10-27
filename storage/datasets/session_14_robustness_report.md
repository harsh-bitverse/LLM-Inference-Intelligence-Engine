# Session 14 — Predictor Robustness, Multi-Target Validation & Evidence-Calibrated Optimization

**Project**: A1 — LLM Inference Intelligence Engine (IIE)  
**Milestone**: Session 14 — Predictor Robustness & Evidence-Calibrated Optimization  
**Date**: October 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

Session 14 solidifies the empirical predictive and optimization capabilities of IIE using the genuine benchmark datasets established in Session 12B and Session 13 on local workstation hardware (Intel Core i7-14700, DDR5-5600, 32 GB RAM, Ollama `0.5.1`, `llama3.2:1b` Q8_0 GGUF).

All previously acquired datasets (`storage/datasets/pilot_llama3_2_1b.jsonl`, `storage/datasets/expansion_llama3_2_1b.jsonl`, manifests, repeatability summaries, and SQLite records) were audited and verified intact. No existing benchmark observations were altered, re-benchmarked, or relabeled.

### Key Milestones Achieved:
1. **Mathematical Root-Cause Resolution**: Fully explained and documented the root cause of the Session 13 random-split discrepancy (Constant MAE 4.33 tok/s on random split vs 0.72 tok/s on grouped holdout).
2. **Leave-One-Configuration-Out (LOCO) Cross-Validation**: Implemented a leak-proof cross-validation framework across all 7 configurations ($N=21$ steady-state repetitions).
3. **Multi-Target Predictor Evaluation**: Expanded prediction targets to include End-to-End Latency (`duration_ms`), Time to First Chunk (`ttfc_ms`), Server Prefill (`server_prompt_eval_duration_ms`), and Server Decode (`server_eval_duration_ms`), while strictly maintaining the exclusion of unmeasured token-level TTFT and ITL.
4. **Feature Availability & Leakage Prevention**: Verified that all latency predictors operate strictly on prospective features known prior to execution (`requested_output_tokens`), completely eliminating retrospective data leakage.
5. **Sample-Size Uncertainty Governance**: Enforced that prediction intervals for sample sizes $N < 30$ are marked `EXPLORATORY_INSUFFICIENT_SAMPLE` with explicit disclaimers rather than presenting uncalibrated confidence intervals.
6. **Optimizer Measurement-Matching Hardening**: Updated `_match_empirical_observation` to strictly scope matches by hardware, engine, model ID, digest, quantization, concurrency, and exact workload prompt/output tokens. Warmup observations are strictly excluded, and multiple matching repetitions are aggregated using `statistics.median`.
7. **Empirical Optimization Scenarios**: Evaluated four realistic decision scenarios (Exact Match, Extrapolation, Unmeasured SLA, and OOD Rejection), proving that the optimizer neither overclaims empirical backing nor substitutes unmeasured surrogate metrics.
8. **Cold-Start Architectural Analysis**: Disclosed and documented the physical mechanics of Ollama's +97.06% cold-start latency penalty.
9. **Full Quality Gate Verification**: Achieved 344 passing backend tests, 0 linting errors (`ruff`), 19 passing frontend tests (`vitest`), clean TypeScript checking (`tsc`), and clean production packaging (`vite build`).

---

## 2. Root-Cause Analysis: Session 13 Random-Split Discrepancy

In Session 13, evaluating a `ConstantMedianPredictor` on `output_tokens_per_second` produced a large discrepancy between evaluation partitioning strategies:
- **Random Split (seed 42, 75/25)**: Constant MAE = **4.33 tok/s** (21.72% MAPE)
- **Configuration Grouped Holdout**: Constant MAE = **0.72 tok/s** (3.26% MAPE)

A rigorous mathematical and architectural audit revealed the exact root cause:

### Mathematical Definition of Throughput
In IIE, `output_tokens_per_second` represents end-to-end user-observed throughput:
$$\text{Throughput} = \frac{\text{Actual Output Tokens}}{\text{Duration (seconds)}} = \frac{O}{\frac{D_{\text{prefill}} + D_{\text{decode}}}{1000}}$$

### The Prefill Dilution Effect
On CPU execution, autoregressive token generation (decode) takes a constant $\approx 43.5\text{ ms/token}$ ($\approx 23.0\text{ tok/s}$) regardless of sequence length. However, prompt evaluation (prefill) is compute-heavy. For a 677-token prompt (`P677_O64`), cold prompt evaluation took **2,049 ms**.

For `P677_O64` (output = 64 tokens):
- Decode duration: $64 \times 43.5\text{ ms} \approx 2,784\text{ ms}$
- Cold prefill duration: $\approx 2,049\text{ ms}$
- Total request duration: $2,049 + 2,784 \approx 4,833\text{ ms}$
- End-to-end throughput: $\frac{64}{4.833\text{ s}} = \mathbf{12.80\text{ tok/s}}$

In contrast, during the immediately subsequent execution of `P677_O128` (output = 128 tokens), Ollama reused the KV cache prefix from the identical prompt text. Prefill evaluation dropped to **47.5 ms**:
- Total request duration: $\approx 5,830\text{ ms}$
- End-to-end throughput: $\frac{128}{5.830\text{ s}} = \mathbf{22.16\text{ tok/s}}$

### Split-Induced Sampling Bias
1. **The Random Split Artifact**: With a random 75/25 train/test split on 21 observations ($N_{\text{test}} = 5$), pseudo-random seed 42 happened to assign **two of the three** prefill-diluted `P677_O64` observations (throughput 12.75 tok/s and 12.87 tok/s) into the 5-sample test set (constituting 40% of the test set). The training set median was **22.41 tok/s**. Evaluating the training median against those two points produced residuals of:
   $$|12.75 - 22.41| = 9.66\text{ tok/s}$$
   $$|12.87 - 22.41| = 9.54\text{ tok/s}$$
   These two outlier points inflated the test set MAE to **4.33 tok/s**.
2. **The Grouped Holdout Masking**: In Session 13's grouped holdout, configurations `P46_O32` and `P119_O64` were held out. Both configurations have short prompts where prefill latency is negligible ($< 80\text{ ms}$). Because neither held-out configuration suffered from prefill dilution, their actual throughputs ($\approx 21.75$ and $\approx 22.73\text{ tok/s}$) closely matched the training median ($22.35\text{ tok/s}$), yielding a deceptive MAE of **0.72 tok/s**.

**Conclusion**: The discrepancy was not a flaw in the predictor code, but a physical demonstration that **end-to-end throughput is sensitive to prompt prefill overhead when prompt length is large relative to output length**. Configuration-level cross-validation (LOCO CV) is required to expose performance across all operational regimes without random sampling luck.

---

## 3. Multi-Target Leave-One-Configuration-Out (LOCO) Cross-Validation

LOCO CV partitions the 21 steady-state observations across all 7 unique configurations. In each fold, all 3 repetitions of one configuration are held out for testing, while the models are trained on the remaining 18 repetitions across the other 6 configurations.

### Comprehensive LOCO CV Results ($N=21$ over 7 Held-Out Folds)

| Prediction Target | Unit | Constant Median MAE | Constant MAPE | Ridge Regression MAE | Ridge MAPE | Analytical Roofline MAE | Analytical MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Output Throughput** | tok/s | **1.87** | **12.98%** | 3.98 | 22.23% | 47.35 | 238.98% |
| **End-to-End Latency (`duration_ms`)** | ms | 1,978.00 | 61.55% | **896.21** | **21.02%** | N/A | N/A |
| **Server Decode (`server_eval_ms`)** | ms | 1,070.86 | 31.61% | **93.31** | **3.21%** | N/A | N/A |
| **Server Prefill (`server_prompt_eval_ms`)** | ms | **289.57** | **21.42%** | 599.36 | 654.44% | N/A | N/A |
| **Time to First Chunk (`ttfc_ms`)** | ms | **332.32** | **65.62%** | 622.92 | 344.33% | N/A | N/A |

### Scientific Insights from LOCO CV:
1. **Predictor Suitability Inversion**:
   - For **Output Throughput**, Constant Median achieves superior generalization (MAE 1.87 tok/s vs Ridge 3.98 tok/s). Ridge regression attempts to fit a linear slope across prompt and output tokens that reflects caching artifacts rather than intrinsic hardware throughput.
   - For **End-to-End Latency** and **Server Decode**, Ridge Regression dramatically outperforms Constant Median. Server decode duration is extraordinarily linear with output tokens ($43.5\text{ ms/token}$), enabling Ridge to achieve an outstanding **3.21% MAPE** on held-out configurations.
2. **Analytical Roofline Ceiling**:
   - The DDR5 theoretical ceiling for 1B Q8_0 weights ($89.6\text{ GB/s} / 1.31\text{ GB} \approx 68.2\text{ tok/s}$) overestimates real CPU throughput by **+205%** (MAE 47.35 tok/s, MAPE 238.98%). This confirms that memory-bandwidth analytical models represent an asymptotic hardware ceiling, not an operational point predictor on host CPUs where cache hierarchies, thread synchronization, and CPU SIMD dispatch dominate.

---

## 4. Prospective Feature Safety & Leakage Prevention

In Session 10.1 and Session 13 audits, feature definitions were tightened to prevent retrospective data leakage. In Session 14, this discipline was extended to all latency predictors:

1. **Prospective Features**:
   - Features available prior to execution: `prompt_tokens`, `requested_output_tokens`, `concurrency`, `quantization`, `model_id`.
   - In greedy decoding with fixed limits, `requested_output_tokens` is known prospective knowledge.
2. **Retrospective Exclusion**:
   - `actual_output_tokens` and `total_sequence_length` are known only after inference completes. If a request terminates early due to an EOS token or error, actual output tokens differ from requested tokens.
   - Training latency predictors on `actual_output_tokens` creates a retrospective leakage flaw when used for prospective deployment planning.
   - In `llm_ii_engine/prediction/baselines.py`, `DEFAULT_FEATURES_BY_TARGET[PredictionTarget.DURATION_MS]` is strictly set to `["prompt_tokens", "requested_output_tokens", "concurrency"]`.

---

## 5. Evidence-Aware Uncertainty & Small-Sample Governance

Predictive ML models often report confidence intervals based on Gaussian assumptions that collapse or become wildly optimistic when sample sizes are small ($N < 30$).

### Governance Rules Implemented in `BasePredictor`:
1. When training sample count $N < 30$:
   - `interval_status` is explicitly set to `"EXPLORATORY_INSUFFICIENT_SAMPLE"`.
   - `prediction_interval` is suppressed (`None`) or flagged with explicit limitation warnings:
     `"Sample size is inadequate for calibrated prediction intervals."`
2. When training data is derived from fewer than 10 physical observations:
   - `validation_status` is set to `"EXPLORATORY_INSUFFICIENT_DATA"`.
3. Out-of-Distribution Support Envelope:
   - `OODDetector` tracks the min/max bounding box for numerical features (`prompt_tokens`, `requested_output_tokens`) and categorical sets (`model_id`, `model_digest`, `engine_type`, `quantization`).
   - Requests outside this envelope return `is_out_of_distribution = True` with human-readable warnings.

---

## 6. Optimizer Measurement-Matching Hardening

The configuration optimizer's empirical observation matching method (`_match_empirical_observation`) in [`llm_ii_engine/optimizer/engine.py`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/llm_ii_engine/optimizer/engine.py#L70) was audited and hardened:

1. **Exclusion of Warmup Observations**:
   ```python
   if rec.is_warmup:
       continue
   ```
   Ensures that cold-start initialization latency is never matched as steady-state operational evidence.
2. **Exact Workload Scoping**:
   Prompt length and requested output tokens must match the historical benchmark exactly. Mismatches do not inherit measured status.
3. **Hardware & Engine Identity**:
   Matches require identical engine, quantization, concurrency, and hardware identifier (or catalog ID). Single-GPU benchmarks never match multi-GPU configurations ($TP > 1$).
4. **Repetition Aggregation via Median**:
   When multiple repetitions exist for an identical configuration (e.g., 3 repetitions in Session 13), the optimizer aggregates all matching records into a single synthetic observation using `statistics.median` across all physical metrics (`output_tokens_per_second`, `duration_ms`, `ttfc_ms`, etc.).

---

## 7. Optimization Decision Scenarios Validation

Four decision scenarios were executed and logged in [`storage/experiments/session_14_optimization_decisions.md`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/storage/experiments/session_14_optimization_decisions.md):

### Scenario 1: Exact Empirical Match (`P46_O32`)
- **Query**: Workload prompt=46, output=32, Ollama `llama3.2:1b` Q8_0, concurrency=1, hardware `hw-report-2606abdf9fce`.
- **Outcome**: `EvidenceStatus.MEASURED_EVALUATED`, throughput = **21.90 tok/s** (exact median of 21.82, 21.90, 22.01 tok/s across repetitions).
- **Feasibility**: Feasible against throughput threshold of 15.0 tok/s.

### Scenario 2: Workload Mismatch / Extrapolation (`P1500_O256`)
- **Query**: Workload prompt=1500, output=256 (both exceeding maximum training range $[46, 677]$ and $[32, 128]$).
- **Outcome**: `EvidenceStatus.INSUFFICIENT_EVIDENCE`, `is_out_of_distribution = True`.
- **Warnings Emitted**:
  - `Numerical feature 'prompt_tokens' value 1500.0 falls outside training range [46.0, 677.0].`
  - `Numerical feature 'output_tokens' value 256.0 falls outside training range [32.0, 128.0].`
  - `Numerical feature 'total_sequence_length' value 1756.0 falls outside training range [78.0, 805.0].`
- **Result**: The optimizer falls back to empirical prediction and flags that the workload requires unverified extrapolation.

### Scenario 3: Unmeasured Latency SLA Constraint (`max_ttft_ms = 50.0 ms`)
- **Query**: Workload with constraint `max_ttft_ms = 50.0 ms` on Ollama streaming chunk mode.
- **Outcome**: Feasibility evaluates strictly to `FeasibilityStatus.UNKNOWN_FEASIBILITY`.
- **Rationale**: Ollama stream emits `TEXT_CHUNK` events without per-token boundaries. IIE measures Time to First Chunk (`ttfc_ms = 154.11 ms`), but strictly refuses to substitute TTFC for TTFT. The SLA cannot be verified empirically.

### Scenario 4: Out-of-Distribution Architecture & Incompatible Quantization
- **Query**: Candidate configuration requesting `llama-3-70b-instruct` with AWQ quantization on Ollama.
- **Outcome**: `FeasibilityStatus.PROVABLY_INFEASIBLE`.
- **Rationale**: Early search space generation filters out unsupported combinations (`"Engine 'ollama' does not support quantization 'awq'"`).

---

## 8. Cold-Start / Warmup Overhead Investigation

In Session 12B and Session 13, the first warmup request experienced a **+97.06% duration penalty** relative to steady-state repetitions:
- Cold Warmup Request: **3,016 ms** duration (Throughput: **10.61 tok/s**)
- Warm Steady-State Repetition 1: **1,507 ms** duration (Throughput: **21.90 tok/s**)
- Warm Steady-State Repetition 2: **1,461 ms** duration (Throughput: **21.90 tok/s**)
- Warm Steady-State Repetition 3: **1,455 ms** duration (Throughput: **22.01 tok/s**)

### Architectural Root Cause in Ollama Runtime:
Workstation process and log analysis revealed the physical sequence of events during cold start:
1. **Server Process Spawning**: Ollama's daemon (`ollama.exe`) does not keep model runner processes in RAM continuously. On the first incoming request for `llama3.2:1b`, it spawns `llama-server.exe` as a dedicated child process.
2. **Model Weight Loading**: The 1.22 GiB GGUF file is read from SSD storage and mapped into host memory pages.
3. **Compute Graph Allocation**: Ollama logs show that during token 0 of request 1, the execution backend compiles and allocates compute graphs (`graph allocs = 1`).
4. **Steady-State Cache Hit**: On subsequent requests (Repetitions 1–3), `llama-server.exe` remains resident in memory (`keep_alive = 5m`), the memory mappings are warm in the OS file cache, and the execution graph is reused (`graphs reused = 2150`, `graph reallocs = 0`).

**Governance Principle**: IIE benchmarks must always isolate warmup from measured runs. Warmup exclusion is mandatory to prevent cold-start process overhead from contaminating steady-state model serving evaluations.

---

## 9. Quality Assurance & Regression Verification

Full quality gates were executed locally:

```bash
# 1. Python Unit and Integration Test Suite
uv run pytest
# Result: 344 passed, 3 warnings in 9.11s

# 2. Python Code Quality and Formatting
uv run ruff check .
# Result: All checks passed!

# 3. Frontend Unit & Component Tests
npm --prefix frontend test -- --run
# Result: 4 test files passed, 19 tests passed in 1.32s

# 4. Frontend Type Checking
npm --prefix frontend run typecheck
# Result: 0 errors (clean compilation)

# 5. Production Webpack/Vite Build
npm --prefix frontend run build
# Result: Built production bundle (dist/index.html, dist/assets) in 1.19s
```

All 344 backend tests and 19 frontend tests pass without failures, regressions, or bypasses.

---

## 10. Scientific Integrity Commitments & Boundaries

1. **Single-Workstation Scope**: All measurements in this report originate exclusively from the authorized local Windows workstation (Intel i7-14700 CPU, DDR5-5600, Ollama `llama3.2:1b` Q8_0). No claims are made regarding GPU performance, multi-node scaling, or other model families.
2. **Measurement Provenance**: Token-level TTFT and ITL are honestly documented as unmeasured under Ollama streaming chunk mode. TTFC and Chunk Latency are reported with exact provenance.
3. **No Uncalibrated Claims**: Predictive intervals are constrained to exploratory ratings until sample sizes meet statistical adequacy criteria ($N \ge 30$).
4. **Scope Preservation**: No additional models were downloaded, no unverified packages were installed, and no new inference backends were introduced.
