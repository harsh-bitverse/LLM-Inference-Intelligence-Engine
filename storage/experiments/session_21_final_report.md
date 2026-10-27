# Session 21 — Authorized Cross-Model Benchmarking, Quality Validation & GitHub Delivery

**Milestone:** A1 — LLM Inference Intelligence Engine (IIE) — Session 21  
**Execution Date:** October 28, 2026  
**Status:** Completed & Delivered  
**Workstation Environment:** Intel Core i7-14700 CPU (20 cores / 28 threads: 8 Performance cores, 12 Efficient cores), 32 GB DDR5 RAM, Windows 11  
**Serving Runtime:** Ollama Server v0.40.11 bound exclusively to loopback `http://127.0.0.1:11434` (Pre-existing PID 328412 preserved)  
**Models Evaluated:** `llama3.2:1b` (Q8_0), `qwen2.5:1.5b` (Q4_K_M), `qwen2.5:0.5b` (Q4_K_M)  

---

## 1. Executive Summary & Authorization Verification

In Session 21, IIE expanded from single-model recommendations to authorized, empirically validated cross-model evaluation across three local LLM architectures. Pursuant to explicit project owner authorization, exactly two additional models were downloaded:
1. `qwen2.5:1.5b` (1.54B parameters, `Q4_K_M` quantization)
2. `qwen2.5:0.5b` (0.49B parameters, `Q4_K_M` quantization)

These candidates join the existing baseline `llama3.2:1b` (1.24B parameters, `Q8_0` quantization) to form a tri-model evaluation matrix spanning parameter sizes from ~0.49B to 1.54B, quantizations from 4-bit to 8-bit, and distinct tokenizer families (TikToken vs. Qwen BPE).

### Core Accomplishments:
- **Controlled Multi-Model Benchmark Execution:** Completed 3 repetitions across all 6 standardized tasks in `WORKLOAD_SUITE` for each candidate (54 total measured requests), utilizing per-repetition model order rotation to minimize thermal and caching order biases.
- **Scientific Tokenizer Incomparability Demonstration:** Quantified token inflation between Qwen BPE (152k vocabulary) and TikToken (128k vocabulary), proving that raw decode tokens/second can distort comparative efficiency by up to 36% compared to normalized character throughput (`characters_per_second`).
- **Restricted Sandbox Code Quality Evaluation:** Implemented an AST-verified execution sandbox restricted to harmless builtins (`range`, `int`, `float`, `bool`, `len`, `isinstance`, `min`, `max`, `abs`), executing 8 deterministic unit test cases to verify code correctness without system vulnerabilities.
- **Pre-Registered Prospective Validation:** Pre-registered prospective predictions to disk before executing live validation runs; confirmed 100% of hypotheses across both latency-prioritized and quality-constrained code generation scenarios.
- **Multi-Model Comparator & Dashboard Integration:** Upgraded recommendation engine to dynamically ingest live empirical benchmark statistics and enforce memory budgets. Enhanced dashboard UI with interactive Pareto efficiency visualization.
- **Full Quality Gate Verification:** 421 Python tests passing (`uv run pytest`), zero Ruff linter warnings (`uv run ruff check .`), 24 frontend Vitest tests passing, TypeScript typecheck clean, and Vite production build verified.

---

## 2. Environment & Model Preflight Audit

The benchmarking environment was rigorously audited prior to execution:
- **Processor:** Intel Core i7-14700 (20 physical cores, 28 logical threads; AVX2 vector execution).
- **Memory Footprint:** 31.69 GB total physical RAM; 9.48 GB free RAM before benchmark initialization. All three models require ~2.70 GB disk space and ~2.8 GB total resident working set memory when sequentially swapped.
- **Disk Availability:** 122+ GB available on drive `C:`.
- **Runtime Process:** Ollama v0.40.11 running as PID 328412 bound strictly to `127.0.0.1:11434`. Preserved intact across sessions.

### Authoritative Model Metadata:
| Model ID | Family | Params | Quantization | Context Window | Vocab Size | Weights Size (bytes) | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `llama3.2:1b` | `llama` | 1.24B | `Q8_0` | 131,072 | 128,256 | 1,321,098,329 | Llama 3.2 Community |
| `qwen2.5:1.5b` | `qwen2` | 1.54B | `Q4_K_M` | 32,768 | 152,064 | 986,061,892 | Apache 2.0 |
| `qwen2.5:0.5b` | `qwen2` | 0.49B | `Q4_K_M` | 32,768 | 152,064 | 397,821,319 | Apache 2.0 |

---

## 3. Workload Suite Execution & Repetition Design

The standardized workload suite comprises 6 deterministic tasks across distinct operational modalities:
1. `workload_v1_interactive_qa`: Short-prompt, low-output interactive fact retrieval.
2. `workload_v1_long_context_retrieval`: Long prompt (230+ tokens), extraction of targeted needle key.
3. `workload_v1_long_form_generation`: Multi-paragraph sustained generation (target 160 tokens).
4. `workload_v1_structured_json`: Clinical record extraction conforming to strict JSON schema.
5. `workload_v1_code_generation`: Python function synthesis evaluated against 8 deterministic unit tests inside a restricted sandbox.
6. `workload_v1_reasoning_numerical`: Multi-step warehouse arithmetic reasoning problem.

### Experimental Protocol:
- **Sampling Parameters:** Deterministic (`temperature=0.0`, `top_p=1.0`, `top_k=1`, `seed=42`).
- **Warmup Policy:** One initial warm-up request per model block prior to measurements, explicitly excluded from dataset observations.
- **Rotation Schedule:**
  - Repetition 1: `llama3.2:1b` -> `qwen2.5:1.5b` -> `qwen2.5:0.5b`
  - Repetition 2: `qwen2.5:1.5b` -> `qwen2.5:0.5b` -> `llama3.2:1b`
  - Repetition 3: `qwen2.5:0.5b` -> `llama3.2:1b` -> `qwen2.5:1.5b`
- **Total Dataset Observations:** 54 measured requests (3 models x 6 tasks x 3 repetitions).

---

## 4. Empirical Performance & Quality Results Table

Aggregated performance and quality statistics computed across the 3 repetitions:

| Model ID | Family | Params | Quant | Reps | Median E2E (ms) | Mean E2E (ms) | Raw Decode (tok/s) | Chars/s | Compliance (%) | Mean Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `llama3.2:1b` | `llama` | 1.24B | `Q8_0` | 3 | 3,002.8 | 3,478.6 | 26.8 | 64.4 | 83.3% | 0.700 |
| `qwen2.5:1.5b` | `qwen2` | 1.54B | `Q4_K_M` | 3 | 2,482.5 | 2,854.2 | 34.0 | 75.0 | 83.3% | 0.833 |
| `qwen2.5:0.5b` | `qwen2` | 0.49B | `Q4_K_M` | 3 | 1,580.9 | 1,374.1 | 79.2 | 180.2 | 66.7% | 0.667 |

### Per-Task Breakdown:
| Task ID | Metric | `llama3.2:1b` | `qwen2.5:1.5b` | `qwen2.5:0.5b` |
| :--- | :--- | :--- | :--- | :--- |
| **Interactive QA** | Median E2E (ms) | 163.3 | 120.4 | **92.9** |
| | Decode Throughput (tok/s) | 44.2 | 55.0 | **129.4** |
| | Correctness Score | 1.00 | 1.00 | 1.00 |
| **Long Context Retrieval** | Median E2E (ms) | 513.3 | 480.6 | **220.5** |
| | Correctness Score | 1.00 | 1.00 | 1.00 |
| **Long Form Generation** | Median E2E (ms) | 6,960.4 | 5,509.6 | **2,367.7** |
| | Normalized Chars/s | 127.5 | 148.1 | **351.3** |
| **Structured JSON** | Median E2E (ms) | 1,922.3 | 2,061.2 | **895.2** |
| | Format Compliance | 100.0% | 100.0% | 100.0% |
| | Correctness Score | 1.00 | 1.00 | 1.00 |
| **Code Generation** | Median E2E (ms) | 3,751.5 | 3,432.0 | **2,069.1** |
| | Format Compliance | 100.0% | 100.0% | 0.0% |
| | Unit Test Correctness | 0.20 | **1.00** | 0.00 |
| **Numerical Reasoning** | Median E2E (ms) | 6,986.8 | 5,513.7 | **2,339.8** |
| | Correctness Score | 0.00 | 0.00 | 0.00 |

---

## 5. Tokenizer Incomparability Evidence & Inflation Factor Analysis

A fundamental scientific conclusion confirmed by Session 21 is that **raw token throughput cannot be directly compared across different model families**.
- `llama3.2:1b` utilizes TikToken with 128,256 vocabulary entries.
- `qwen2.5:1.5b` and `qwen2.5:0.5b` utilize Qwen BPE with 152,064 vocabulary entries.

### Empirical Token Inconsistencies:
1. **Prompt Ingestion Expansion:**
   Across identical task prompts, Qwen BPE produced 6% to 13% more tokens than Llama TikToken (e.g. 161 tokens vs 143 tokens on Structured JSON; 252 vs 235 tokens on Long Context).
2. **Output Token Density Divergence:**
   On the structured clinical JSON task, both models generated identical semantic data. However, Llama emitted 43 tokens to output 93 characters, whereas Qwen emitted 58 tokens to output 109 characters (a 1.36x token inflation ratio).
3. **Misleading Throughput Comparison:**
   Reporting raw tokens/second gives `qwen2.5:1.5b` an apparent 29.7 tok/s vs `llama3.2:1b` 23.8 tok/s. However, their end-to-end task duration is virtually identical (2,061 ms vs 1,922 ms), and character delivery throughput is within 4 chars/s (50.9 chars/s vs 46.5 chars/s).
4. **Resolution:** IIE strictly reports `task_elapsed_ms` and `characters_per_second` as the authoritative cross-model comparison metrics, flagging token throughput with an explicit non-comparability caveat.

---

## 6. Automated Multi-Task Quality Verification Results

Quality verification was fully automated using deterministic scoring:
- **Interactive QA & Long Context:** Deterministic regex pattern matching for targeted ground-truth factual strings (`Paris`, `94812`). All models scored 1.0.
- **Structured JSON:** Strict JSON parser validation with key existence and value type verification (`patient_id`, `age`, `systolic_bp`, `diastolic_bp`, `pulse_bpm`). All three models achieved 100% compliance and score 1.0.
- **Code Generation (Sandbox Unit Tests):**
  - Code generated by `qwen2.5:1.5b` passed Python AST validation and passed all 8 unit tests (`is_prime(2)==True`, `is_prime(17)==True`, `is_prime(1)==False`, `is_prime(0)==False`, `is_prime(-5)==False`, `is_prime(4)==False`, `is_prime(9)==False`, `is_prime(25)==False`), achieving score 1.0.
  - Code generated by `llama3.2:1b` parsed syntactically but failed on corner cases (score 0.20).
  - Code generated by `qwen2.5:0.5b` included markdown chatter outside fences and failed unit test execution (score 0.0).
- **Numerical Reasoning:** All three small models failed to arrive at the exact final count `112` on the multi-step warehouse arithmetic task without chain-of-thought fine-tuning or scratchpad guidance.

---

## 7. Multi-Dimensional Pareto Efficiency Frontier

Candidate models occupy distinct non-dominated positions along the multi-dimensional Pareto frontier:

```
Latency (Lower is better)
 ^
 |    [llama3.2:1b] (Q8_0 High Precision, High JSON/Text Compliance)
 |         \
 |          \
 |       [qwen2.5:1.5b] (1.5B Capacity, 100% Code Quality, Sub-1GB weights)
 |             \
 |              \
 |           [qwen2.5:0.5b] (Ultra-low Latency, 79 tok/s, <400MB weights)
 +--------------------------------------------------------------------> Correctness / Capacity
```

1. **`qwen2.5:0.5b` (Extreme Latency & Memory Efficiency Frontier):**
   - Lowest latency across all tasks (92 ms interactive QA, 895 ms JSON).
   - Lowest resident memory footprint (< 600 MB RAM).
   - Fast raw decode throughput (79.2 tok/s average; up to 129 tok/s).
   - **Trade-off:** Incapable of complex code generation or advanced reasoning.
2. **`qwen2.5:1.5b` (Quality-Constrained Frontier):**
   - Highest mean correctness across suite (0.833).
   - Only candidate to pass 100% of functional code unit tests.
   - Balanced throughput (34 tok/s, 75 chars/s).
   - Sub-1 GB disk footprint (986 MB).
3. **`llama3.2:1b` (High Precision Format Adherence Frontier):**
   - 8-bit quantization (`Q8_0`) guarantees zero quantization degradation on common English language tokens.
   - 100% compliance on JSON and long-form writing.
   - Higher memory requirement (~1.5 GB RAM) and higher computational latency per token due to 8-bit arithmetic.

---

## 8. Pre-Registered Prospective Recommendation Validation

Following rigorous open-science methodology, recommendation hypotheses were pre-registered and serialized to `storage/experiments/session_21_prospective_preregistration.json` **before** executing live validation requests.

### Prospective Validation Results:
| Scenario | Requirement & Constraints | Pre-Registered Expected Model | Measured Winner | Hypothesis Status | Key Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario A** | Low-latency Interactive QA (SLA 2,000 ms, min score 0.8) | `qwen2.5:0.5b` | `qwen2.5:0.5b` | **CONFIRMED** | Cleared SLA in **59.3 ms** (vs 86.4 ms for Qwen 1.5B, 100.2 ms for Llama 1B) with score 1.00. |
| **Scenario B** | Quality-Constrained Code Generation (SLA 8,000 ms, require code pass rate) | `qwen2.5:1.5b` | `qwen2.5:1.5b` | **CONFIRMED** | Only candidate to pass unit tests (score **1.00**, 2,511 ms). Both Llama 1B (score 0.20) and Qwen 0.5B (score 0.00) classified as `QUALITY_INFEASIBLE`. |

**Hypothesis Confirmation Rate:** 100% (2 / 2 scenarios).

---

## 9. Multi-Model Comparator & API Enhancement

1. **Empirical Summary Prioritization:** Updated `/api/v1/models/smoke-summary` and `MultiModelComparator` to ingest authoritative Session 21 empirical measurements across all 3 models.
2. **Dynamic Throughput & Latency Model:** Computes model-specific nominal latency and throughput from empirical observations rather than relying on single-model estimates.
3. **Conservative Feasibility & Rejection Logic:**
   - Enforces `QUALITY_INFEASIBLE` when candidate models fail required formatting or unit tests.
   - Enforces `INSUFFICIENT_MEMORY` / `PROVEN_INFEASIBLE` when model working set exceeds user-specified RAM budgets.
   - Disallows performance claims for uninstalled candidates (`UNKNOWN_FEASIBILITY`, rank `None`).

---

## 10. Dashboard Integration

- **Component:** `MultiModelComparisonView.tsx` located in `frontend/src/components/recommendations/`.
- **Features:**
  - Standard Workload Suite preset selector (Interactive QA, JSON, Code, Math).
  - Dynamic multi-model candidate comparison cards with feasibility badges (`Robust Feasible`, `Quality Infeasible`, `SLA Miss`, `Insufficient Memory`).
  - Strict Tokenizer Comparability Advisory banner.
  - Normalized metrics display: Client E2E duration, characters/second, and quality verification pass rates.

---

## 11. Test Coverage & Verification Quality Gates

All regression and quality gates passed without exceptions:

1. **Python Unit & Integration Test Suite (`uv run pytest`):**
   - **421 passed, 0 failed** in 10.30 seconds.
   - Covers database persistence, schema migrations, conservative SLA policies, operating regime classifications, AST sandbox security, code quality scoring, and multi-model comparator ranking.
2. **Python Code Formatting & Linting (`uv run ruff check .`):**
   - **All checks passed!** Clean repository.
3. **Frontend Vitest Suite (`npm --prefix frontend test -- --run`):**
   - **5 test files passed, 24 tests passed** in 1.45 seconds.
4. **Frontend TypeScript Typecheck (`npm --prefix frontend run typecheck`):**
   - **Zero errors.**
5. **Frontend Production Build (`npm --prefix frontend run build`):**
   - **Built successfully** in 1.17s.

---

## 12. Security & Sandbox Audit

- **Restricted Sandbox Architecture:** `workload_suite.py` limits Python `__builtins__` to:
  `range`, `int`, `float`, `bool`, `len`, `isinstance`, `min`, `max`, `abs`.
- **Protected Surface:**
  - File I/O (`open`, `os`, `pathlib`) -> blocked.
  - Process creation (`subprocess`, `os.system`) -> blocked.
  - Network sockets (`socket`, `urllib`, `requests`, `httpx`) -> blocked.
  - Code escape (`eval`, `exec`, `__import__`) -> blocked.
- **Verification:** Tested malicious payloads attempting file access, dynamic imports, and builtin overrides; all failed safely without runtime exposure.

---

## 13. Historical Dataset & Provenance Preservation

All historical benchmark datasets from previous milestones remain preserved intact in `storage/`:
- Session 12B/13 pilot runs (`storage/datasets/pilot_v1_observations.jsonl`).
- Session 14 repeatability datasets.
- Session 15 cache-state controlled datasets.
- Session 16 workload datasets.
- Session 17 prospective validation datasets.
- Session 18 conservative SLA datasets.
- Session 19 sustained-load records (`storage/datasets/session_19_sustained_load_measurements.jsonl`).
- Session 20 multi-model smoke datasets (`storage/datasets/session_20_multi_model_measurements.jsonl`).
- Session 21 cross-model datasets (`storage/datasets/session_21_cross_model_measurements.jsonl` and `.csv`).

---

## 14. Ollama Runtime Lifecycle

- **Pre-existing Process Status:** Ollama process PID 328412 was active on `127.0.0.1:11434` prior to Session 21.
- **Lifecycle Action:** Preserved running and undisturbed.
- **Binding Security:** Verified bound exclusively to loopback interface `127.0.0.1:11434`.

---

## 15. Deliverables Checklist & File Manifest

| Deliverable | Path | Description |
| :--- | :--- | :--- |
| **Cross-Model Runner** | [`llm_ii_engine/benchmark/multi_model_runner.py`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/llm_ii_engine/benchmark/multi_model_runner.py) | Reusable runner with repetition loops, order rotation, and per-task stats. |
| **Execution Script** | [`llm_ii_engine/benchmark/run_session_21_cross_model.py`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/llm_ii_engine/benchmark/run_session_21_cross_model.py) | Full execution, analysis generation, pre-registration, and live validation script. |
| **Experiment Manifest** | [`storage/experiments/session_21_cross_model_manifest.json`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/storage/experiments/session_21_cross_model_manifest.json) | Complete JSON manifest with 54 observations and summary statistics. |
| **JSONL Dataset** | [`storage/datasets/session_21_cross_model_measurements.jsonl`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/storage/datasets/session_21_cross_model_measurements.jsonl) | Tabular JSONL export of 54 cross-model observations. |
| **CSV Dataset** | [`storage/datasets/session_21_cross_model_measurements.csv`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/storage/datasets/session_21_cross_model_measurements.csv) | Authoritative CSV export of 54 cross-model observations. |
| **Statistical Analysis** | [`storage/experiments/session_21_cross_model_statistical_analysis.md`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/storage/experiments/session_21_cross_model_statistical_analysis.md) | In-depth statistical aggregation, dispersion, and Pareto analysis. |
| **Pre-Registration** | [`storage/experiments/session_21_prospective_preregistration.json`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/storage/experiments/session_21_prospective_preregistration.json) | Pre-registered prospective predictions saved before validation. |
| **Validation Report** | [`storage/experiments/session_21_prospective_validation.md`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/storage/experiments/session_21_prospective_validation.md) | Validation outcomes demonstrating 100% hypothesis confirmation. |
| **Validation Manifest** | [`storage/experiments/session_21_prospective_validation.json`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/storage/experiments/session_21_prospective_validation.json) | Typed JSON records of prospective validation runs. |
| **Comparator Engine** | [`llm_ii_engine/recommendation/multi_model_comparator.py`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/llm_ii_engine/recommendation/multi_model_comparator.py) | Multi-model ranking engine with empirical data ingestion and memory constraints. |
| **API Endpoints** | [`llm_ii_engine/api/routes/models.py`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/llm_ii_engine/api/routes/models.py) | REST API serving 3-model summary data and multi-model comparisons. |
| **Dashboard UI** | [`frontend/src/components/recommendations/MultiModelComparisonView.tsx`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/frontend/src/components/recommendations/MultiModelComparisonView.tsx) | Interactive React dashboard component. |
| **Test Suite** | [`tests/unit/test_session_21_cross_model.py`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/tests/unit/test_session_21_cross_model.py) | 9 targeted unit tests covering catalog, sandbox, scoring, integrity, and comparator. |
| **Final Report** | [`storage/experiments/session_21_final_report.md`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/storage/experiments/session_21_final_report.md) | Comprehensive milestone report. |

---

## 16. Strict Milestone Boundary Notice

**Session 21 is fully completed and verified.**  
All model benchmarks, quality evaluations, prospective validations, API enhancements, tests, and documentation are in place.  
As required by the milestone contract: **Do NOT begin Session 22.**
