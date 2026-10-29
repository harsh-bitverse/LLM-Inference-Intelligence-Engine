# Session 22 — Scientific Reconciliation of Session 21 Empirical Claims

**Milestone:** A1 — LLM Inference Intelligence Engine (IIE) — Session 22  
**Date:** October 28, 2026  
**Status:** Approved & Enforced  
**Context:** Audit of Session 21 cross-model benchmarking claims, Pareto frontier definitions, and quality metrics.

---

## 1. Executive Summary

This document formalizes scientific corrections and methodological refinements to the cross-model evaluation findings presented in Session 21. While Session 21 established empirical benchmarking across `llama3.2:1b`, `qwen2.5:1.5b`, and `qwen2.5:0.5b`, seven specific scientific claims required formal reconciliation to prevent overclaiming, confounded attribution, and metric distortion.

---

## 2. Itemized Reconciliation of Scientific Claims

### Issue 1: Unsupported Claim of "Zero Quantization Degradation" for Q8_0
- **Session 21 Statement:** "8-bit quantization (`Q8_0`) guarantees zero quantization degradation on common English language tokens."
- **Scientific Audit:** This claim was ungrounded. Zero degradation can only be demonstrated by running a controlled, side-by-side benchmark comparing an unquantized FP16 or FP32 reference model against the quantized candidate on identical prompts. Because no FP16/FP32 baseline was tested, the claim that Q8_0 exhibits zero degradation is an unsupported empirical assumption. All integer quantizations introduce non-zero roundoff noise in weights and activations.
- **Reconciled Policy:** IIE will not claim zero quantization degradation for `Q8_0` or any quantized format without explicit comparative measurements against a verified full-precision baseline.

---

### Issue 2: Recomputing the Pareto Frontier Using Strictly Measured Objectives
- **Session 21 Statement:** "Llama 3.2 1B is non-dominated on the high-precision / quality constraint boundary."
- **Scientific Audit:** Quantization bit-width is an architectural hyperparameter and execution input, *not* an objective function. In formal multi-objective optimization, a Pareto frontier is defined solely over candidate output objectives (e.g., minimize Latency, minimize Resident Memory, maximize Task Correctness).
  
  Evaluating Session 21 measurements strictly over the objective tuple:
  $$\min(\text{Median Latency}), \quad \min(\text{Weights / Memory Footprint}), \quad \max(\text{Task Correctness Score})$$

  - `qwen2.5:0.5b`: 1,580.9 ms median latency, 398 MB weights footprint, 0.667 mean correctness.
    - **Status:** **Non-dominated.** Lowest latency and lowest memory footprint.
  - `qwen2.5:1.5b`: 2,482.5 ms median latency, 986 MB weights footprint, 0.833 mean correctness.
    - **Status:** **Non-dominated.** Highest mean correctness, 100% code unit test pass rate, faster than Llama 3.2 1B.
  - `llama3.2:1b`: 3,002.8 ms median latency, 1,321 MB weights footprint, 0.700 mean correctness.
    - **Status:** Under this specific 3-objective aggregate tuple, `llama3.2:1b` was strictly *dominated* by `qwen2.5:1.5b` (which had lower latency, smaller memory footprint, and higher overall accuracy).
- **Reconciled Policy:** Pareto dominance must be calculated strictly from measured objective performance metrics (latency, memory, correctness by category) and never from input parameter specifications or quantization bit-widths.

---

### Issue 3: Disentangling Confounded Architectural Variables
- **Session 21 Statement:** Attributing performance or quality differences to architecture without isolating parameter counts or quantization levels.
- **Scientific Audit:** The three deployed models differ across three confounded dimensions simultaneously:
  1. **Parameter count:** 0.49B vs. 1.24B vs. 1.54B.
  2. **Quantization format:** `Q4_K_M` (4-bit block-wise) vs. `Q8_0` (8-bit symmetric).
  3. **Model Family / Tokenizer:** Llama architecture with TikToken (128k vocab) vs. Qwen2 architecture with Qwen BPE (152k vocab).
  
  Consequently, observed latency or accuracy deltas between `llama3.2:1b` and `qwen2.5:1.5b` cannot be attributed solely to the Llama vs. Qwen family, because Qwen had 24% more parameters but 50% fewer bits per weight.
- **Reconciled Policy:** IIE reports comparative outcomes as end-to-end **artifact configurations** (e.g., `llama3.2:1b-Q8_0` vs. `qwen2.5:1.5b-Q4_K_M`) and explicitly notes that family, parameter count, and quantization are confounded.

---

### Issue 4: Distinguishing JSON Syntax Compliance from Semantic Extraction Correctness
- **Session 21 Practice:** An output that parsed with `json.loads` and had expected keys was awarded full score, even if semantic values were not deeply verified against ground truth.
- **Scientific Audit:** Syntax compliance ensures parseability; it does not ensure factual extraction truth. A model that outputs `{"patient_id": "UNKNOWN", "systolic_bp": 999}` is syntactically valid but semantically erroneous.
- **Reconciled Policy:** In Session 22, all structured data evaluations decompose into two distinct, reported metrics:
  1. `json_schema_valid: bool`: Valid JSON syntax adhering to expected top-level schema.
  2. `json_semantic_correctness: float`: Field-by-field value correctness against authoritative answer keys.

---

### Issue 5: Distinguishing Code Syntax Validity from Verified Test Execution
- **Session 21 Practice:** Valid Python AST gave an automatic partial score of 0.20, while unit-test pass yielded 1.00.
- **Scientific Audit:** Conflating syntax parsing with functional code correctness creates ambiguities in quality reporting. Syntax checking is static parsing; functional correctness requires dynamic execution against rigorous unit tests.
- **Reconciled Policy:** Session 22 reports:
  1. `code_syntax_valid: bool`: Static Python AST parse validation.
  2. `code_unit_test_success: bool`: All functional test assertions passed.
  3. `code_test_pass_rate: float`: Proportion of test cases passed (0.0 to 1.0) executed inside a restricted, secure execution sandbox.

---

### Issue 6: Preserving Category-Specific Quality Outcomes vs. Universal Score Collapse
- **Session 21 Practice:** Models were sometimes ranked by a single scalar `mean_correctness_score`.
- **Scientific Audit:** Averaging disparate task scores (e.g. code generation, numerical reasoning, factual QA) into a single scalar conceals critical task-specific strengths and failure modes. For instance, `llama3.2:1b` achieved 100% on factual QA and structured JSON, but failed numerical reasoning and complex coding. Averaging these penalizes a model that is optimal for factual QA.
- **Reconciled Policy:** IIE maintains and displays category-level quality profiles (`factual_qa`, `numerical_reasoning`, `structured_json`, `code_generation`, `context_retrieval`, `long_form_compliance`). Candidate models are filtered and ranked based on the user's workload-specific category constraints.

---

### Issue 7: Formal Definitions for Multi-Objective Selection
- **Reconciled Policy:** When recommending a model, the recommendation engine must provide:
  1. The exact objective set under which the model is optimal or non-dominated.
  2. The category-specific evidence supporting the recommendation.
  3. The unmeasured or violated categories as explicit advisories.
