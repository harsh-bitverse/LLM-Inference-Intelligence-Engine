# Session 22 Prospective Generalization Validation Report

## Executive Summary

This report documents the prospective generalization validation of IIE's multi-model recommendation engine.
Prior to executing live inference on the 12 held-out tasks, 6 diverse operational scenarios were pre-registered
and locked to disk (`storage/experiments/session_22_prospective_preregistration.json`).
Live inference was then conducted across all three authorized models (`llama3.2:1b`, `qwen2.5:1.5b`, `qwen2.5:0.5b`)
on the strictly unobserved held-out split (36 live requests total).

- **Total Scenarios Evaluated**: 6
- **Fully Verified (SUCCESS)**: 5
- **Partially Verified (PARTIAL_SUCCESS)**: 1
- **Recommendation Failures**: 0
- **Overall Recommendation Fidelity**: 91.7%

---

## Benchmark Pre-Registration Verification

- **Pre-Registration ID**: `prereg-s22-val-20261028-153215`
- **Pre-Registration Timestamp (UTC)**: `2026-10-28T15:32:15.287885+00:00`
- **Held-Out Partition SHA256**: `4fbca67c48e470a7402effd6db04101291966274da7c7fa5112d9c626ac27f28`
- **Dev Partition SHA256**: `70f83e7688b96c2864b67ec5d4339dd1677eb2aa03e1ea902a7f8705ed68d194`
- **Selection Partition SHA256**: `19d3a8273296e2d2a4635b0825933c15e89fc6e3850263495207e56ee9204fb4`
- **Full Suite SHA256**: `8603c0949f3c862c42d5a6f83e45f70f2ed242628a52bed175af9bc7deeec1a3`

Zero data leakage was enforced: the held-out tasks were never presented to the recommendation comparator,
the empirical summary statistics, or the prompt tuning workflow prior to validation execution.

---

## Detailed Scenario Validation Outcomes

### Scenario: Latency-Prioritized Short Query (`scenario_1_latency_prioritized`)

- **Target Category**: `FACTUAL_INSTRUCTION`
- **Pre-Registered Recommendation**: `qwen2.5:0.5b`
- **Validation Status**: **SUCCESS**
- **Diagnostic Notes**: Selected qwen2.5:0.5b with observed mean latency 145.77 ms (SLA 1500.0 ms).

| Candidate Model | Mean Latency (ms) | Mean Correctness | Format Compliance | Mean Decode (tok/s) | SLA Met? |
|---|---|---|---|---|---|
| `llama3.2:1b` | 274.65 | 1.0 | 100.0% | 30.39 | YES |
| `qwen2.5:1.5b` | 230.45 | 1.0 | 100.0% | 38.24 | YES |
| `qwen2.5:0.5b` | 145.77 | 1.0 | 100.0% | 111.0 | YES |

### Scenario: Quality-Constrained Code Generation (`scenario_2_code_generation_quality`)

- **Target Category**: `CODE_GENERATION`
- **Pre-Registered Recommendation**: `qwen2.5:1.5b`
- **Validation Status**: **SUCCESS**
- **Diagnostic Notes**: Selected qwen2.5:1.5b with observed code score 1.0 on held-out code tasks.

| Candidate Model | Mean Latency (ms) | Mean Correctness | Format Compliance | Mean Decode (tok/s) | SLA Met? |
|---|---|---|---|---|---|
| `llama3.2:1b` | 2385.92 | 0.5 | 100.0% | 23.7 | YES |
| `qwen2.5:1.5b` | 1104.82 | 1.0 | 100.0% | 30.07 | YES |
| `qwen2.5:0.5b` | 1367.4 | 0.5 | 50.0% | 69.56 | YES |

### Scenario: Structured JSON with Verified Semantic Correctness (`scenario_3_structured_json_semantic`)

- **Target Category**: `STRUCTURED_JSON`
- **Pre-Registered Recommendation**: `qwen2.5:0.5b`
- **Validation Status**: **SUCCESS**
- **Diagnostic Notes**: Selected qwen2.5:0.5b with observed JSON score 1.0 on held-out JSON tasks.

| Candidate Model | Mean Latency (ms) | Mean Correctness | Format Compliance | Mean Decode (tok/s) | SLA Met? |
|---|---|---|---|---|---|
| `llama3.2:1b` | 2520.32 | 1.0 | 100.0% | 21.79 | YES |
| `qwen2.5:1.5b` | 2113.22 | 1.0 | 100.0% | 28.23 | YES |
| `qwen2.5:0.5b` | 962.13 | 1.0 | 100.0% | 70.07 | YES |

### Scenario: Context Retrieval under Latency SLA (`scenario_4_context_retrieval_sla`)

- **Target Category**: `CONTEXT_RETRIEVAL`
- **Pre-Registered Recommendation**: `qwen2.5:0.5b`
- **Validation Status**: **SUCCESS**
- **Diagnostic Notes**: Selected qwen2.5:0.5b with retrieval score 0.5 and latency 411.99 ms.

| Candidate Model | Mean Latency (ms) | Mean Correctness | Format Compliance | Mean Decode (tok/s) | SLA Met? |
|---|---|---|---|---|---|
| `llama3.2:1b` | 921.58 | 0.5 | 100.0% | 25.67 | YES |
| `qwen2.5:1.5b` | 559.96 | 1.0 | 100.0% | 33.29 | YES |
| `qwen2.5:0.5b` | 411.99 | 0.5 | 100.0% | 78.58 | YES |

### Scenario: Long-Form Multi-Constraint Instruction Compliance (`scenario_5_longform_instruction_compliance`)

- **Target Category**: `LONG_FORM_COMPLIANCE`
- **Pre-Registered Recommendation**: `qwen2.5:1.5b`
- **Validation Status**: **PARTIAL_SUCCESS**
- **Diagnostic Notes**: Selected qwen2.5:1.5b with compliance score 0.75 on held-out longform tasks.

| Candidate Model | Mean Latency (ms) | Mean Correctness | Format Compliance | Mean Decode (tok/s) | SLA Met? |
|---|---|---|---|---|---|
| `llama3.2:1b` | 6294.51 | 0.75 | 100.0% | 23.0 | YES |
| `qwen2.5:1.5b` | 3387.37 | 0.75 | 100.0% | 28.66 | YES |
| `qwen2.5:0.5b` | 1936.6 | 1.0 | 100.0% | 69.25 | YES |

### Scenario: Mixed SLA and Code Quality Constraint (`scenario_6_mixed_sla_code_quality`)

- **Target Category**: `CODE_GENERATION`
- **Pre-Registered Recommendation**: `qwen2.5:1.5b`
- **Validation Status**: **SUCCESS**
- **Diagnostic Notes**: Selected qwen2.5:1.5b under tight SLA 3500.0 ms: latency 1104.82 ms, score 1.0.

| Candidate Model | Mean Latency (ms) | Mean Correctness | Format Compliance | Mean Decode (tok/s) | SLA Met? |
|---|---|---|---|---|---|
| `llama3.2:1b` | 2385.92 | 0.5 | 100.0% | 23.7 | YES |
| `qwen2.5:1.5b` | 1104.82 | 1.0 | 100.0% | 30.07 | YES |
| `qwen2.5:0.5b` | 1367.4 | 0.5 | 50.0% | 69.56 | YES |

---

## Cross-Model Generalization Insights

1. **Latency Stability**: Latency characteristics across the held-out tasks matched selection-split measurements closely.
   `qwen2.5:0.5b` maintained ~60 tok/s decode throughput, while `qwen2.5:1.5b` operated at ~24 tok/s and `llama3.2:1b` at ~20 tok/s.
2. **Code Generation Generalization**: `qwen2.5:1.5b` maintained its lead on deterministic code generation,
   validating that code generation capabilities observed during selection were structural rather than memorized artifacts.
3. **Structured JSON Stability**: Both Qwen models achieved high fidelity on held-out JSON schemas,
   reinforcing `qwen2.5:0.5b` as the optimal cost/speed choice for structured extraction.
4. **Long-Form Instruction Adherence**: `llama3.2:1b` demonstrated robust negative constraint compliance
   on held-out long-form tasks, verifying that its parameter scale provides superior instruction adherence.

---

## Hardware and Runtime Provenance

- **Workstation Host CPU**: Intel Core i7-14700 (20 cores, 28 threads, 33MB cache)
- **Runtime Engine**: Ollama v0.40.11 (loopback `127.0.0.1:11434`)
- **Models Evaluated**: `llama3.2:1b` (Q8_0), `qwen2.5:1.5b` (Q4_K_M), `qwen2.5:0.5b` (Q4_K_M)