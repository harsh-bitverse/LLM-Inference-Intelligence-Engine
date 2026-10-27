# Session 14 — Optimization Decision Scenarios & Evidence Calibration

## Overview
This artifact documents the evaluation of four realistic optimization decision scenarios under IIE's
evidence-calibrated optimization pipeline using empirical observations from Session 12B/13.

---

### Scenario 1: Exact Historical Empirical Match
- **Workload**: Prompt = 46 tokens, Output = 32 tokens
- **Runtime**: Ollama, `llama3.2:1b`, Q8_0, Concurrency = 1
- **Feasibility Status**: `UNKNOWN_FEASIBILITY`
- **Evidence Tier**: `MEASURED_EVALUATED`
- **Evaluated Throughput**: `21.9 tok/s`
- **Provenance**: `MEASURED`
- **Evidence Calibration Principle**:
  Exact configuration and workload match aggregates all matching measured repetitions via `statistics.median`.
  Cold-start warmup runs are strictly excluded, ensuring steady-state performance reporting.

---

### Scenario 2: Workload Mismatch / Extrapolation
- **Workload**: Prompt = 1500 tokens, Output = 256 tokens
- **Runtime**: Ollama, `llama3.2:1b`, Q8_0, Concurrency = 1
- **Feasibility Status**: `UNKNOWN_FEASIBILITY`
- **Evidence Tier**: `INSUFFICIENT_EVIDENCE`
- **Evaluated Throughput**: `22.32 tok/s`
- **Out-of-Distribution Detected**: `True`
- **OOD Reasons**: Numerical feature 'prompt_tokens' value 1500.0 falls outside training range [46.0, 677.0]., Numerical feature 'output_tokens' value 256.0 falls outside training range [32.0, 128.0]., Numerical feature 'total_sequence_length' value 1756.0 falls outside training range [78.0, 805.0].
- **Evidence Calibration Principle**:
  When workload parameters deviate from empirically tested configurations, the optimizer must NOT assert
  `MEASURED_EVALUATED`. It falls back to empirical prediction and explicitly flags the prediction as out-of-distribution.

---

### Scenario 3: Unmeasured Latency SLA Constraint
- **Constraint**: `max_ttft_ms = 50.0 ms`
- **Runtime**: Ollama streaming chunk mode (`TokenEventType.TEXT_CHUNK`)
- **Feasibility Status**: `UNKNOWN_FEASIBILITY`
- **Rejection / Unknown Reasons**: ['Target GPU hardware specification is missing; device memory capacity cannot be verified', 'SLA max_ttft_ms (50.00 ms) cannot be verified because TTFT was not evaluated']
- **Audit Notes**: ['Parameter weights: 1.24B parameters at 1.0625 bytes/param', 'GGUF Q8_0 quantization: 32-element INT8 blocks with 16-bit FP16 block scale = (32 + 2)/32 = 1.0625 bytes/param', 'KV cache computed for GQA (16 layers, 8 KV heads, 128 head dim, 2.0 bytes/elem). Factor 2 accounts for Key and Value tensors.', 'Estimated runtime VRAM includes 15% allocator/activation margin and 512 MiB baseline driver/runtime context overhead.', '2 FLOPs per parameter per token for forward pass matrix multiplications (GEMMs)', 'Attention FLOPs included using 4 * num_layers (16) * d_hidden (2048) * S^2 for prefill, and incremental causal attention projections for decode.', 'Prefill operational intensity (86.6 FLOPs/Byte) assumes weights are streamed once and KV cache is written to VRAM for prompt tokens.', 'Decode operational intensity (1.88 FLOPs/Byte) models weight transfer per decode step with batch size 1 and KV cache read operations.', 'Matched real benchmark observation has measured TTFC=154.11 ms, but token-level TTFT is unmeasured in streaming chunk adapter; TTFT SLA cannot be verified.']
- **Evidence Calibration Principle**:
  IIE never substitutes Time-to-First-Chunk (TTFC) for Time-to-First-Token (TTFT). When an SLA requires a metric
  that was not physically measured, feasibility evaluates strictly to `UNKNOWN_FEASIBILITY` rather than falsely passing.

---

### Scenario 4: Out-Of-Distribution Architecture & Engine Incompatibility
- **Candidate Model**: `llama-3-70b-instruct` (70B parameters, AWQ quantization)
- **Target Engine**: Ollama
- **Feasibility Status**: `PROVABLY_INFEASIBLE`
- **Rejection Reasons**: ["Engine 'ollama' does not support quantization 'awq': 1"]
- **Out-of-Distribution Detected**: `True`
- **Evidence Calibration Principle**:
  Provable engine-quantization incompatibilities (AWQ on Ollama) and memory violations trigger provable rejection,
  while architectural dimension shifts trigger OOD disclaimers.