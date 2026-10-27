# Session 16 — Raw-to-Dashboard Metric Lineage Report

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 16 — Measurement Contract Verification & Lineage Tracing  
**Artifact ID:** `report-session-16-metric-lineage`  
**Date:** 2026-10-27  

---

## 1. Lineage Architecture Overview

The IIE pipeline enforces strict semantic traceability from the raw HTTP byte stream emitted by the inference engine to the interactive frontend dashboard. This report traces how each authoritative metric is captured, converted, persisted, exported, predicted, and visualized.

```
[Inference Server (Ollama / llama.cpp)]
                  │
                  ▼ (SSE / HTTP JSON Chunk Stream)
        [Inference Adapter]  <── Captures monotonic perf_counter timestamps & terminal payload
                  │
                  ▼ (StreamEvent & ServerReportedStats)
       [Benchmark Metric Sampler]  <── Aggregates ITL/chunk latencies & duration
                  │
                  ▼ (RequestBenchmarkResult)
      [SQLite Database (request_benchmarks)]  <── Persists raw server_stats_json & client metrics
                  │
                  ▼ (EmpiricalDatasetExporter)
    [EmpiricalObservationRecord (JSONL/CSV)]  <── Normalizes v1 & v2 distinct metrics
                  │
                  ▼ (DatasetBuilder)
           [PredictionRecord]  <── Enforces prospective feature safety & eligibility flags
                  │
                  ▼ (Predictor / Optimizer)
       [PredictionOutput / CandidateEvaluation]  <── Evaluates against validated objectives
                  │
                  ▼ (FastAPI REST Endpoints)
             [REST API JSON]
                  │
                  ▼ (TypeScript Domain Models)
         [Frontend Dashboard UI]
```

---

## 2. End-to-End Field Mapping Matrix

| Authoritative Semantic Metric | Raw Server Payload Key | Adapter Property (`ServerReportedStats`) | Benchmark Result (`RequestBenchmarkResult`) | SQLite Column (`request_benchmarks`) | Dataset Record (`EmpiricalObservationRecord`) | Predictor Target (`PredictionTarget`) | Frontend Interface (`types.ts`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Client E2E Duration** | N/A (client timer) | N/A | `duration_ms` / `client_e2e_duration_ms` | `duration_ms` | `client_e2e_duration_ms` | `CLIENT_E2E_DURATION_MS` | `client_e2e_duration_ms` |
| **Server Total Duration** | `total_duration` (ns) | `total_duration_ms` | `server_stats.total_duration_ms` | `server_stats_json.total_duration_ms` | `server_total_duration_ms` | `SERVER_TOTAL_DURATION_MS` | `server_total_duration_ms` |
| **Server Load Duration** | `load_duration` (ns) | `load_duration_ms` | `server_stats.load_duration_ms` | `server_stats_json.load_duration_ms` | `server_load_duration_ms` | `SERVER_LOAD_DURATION_MS` | `server_load_duration_ms` |
| **Server Prefill Duration** | `prompt_eval_duration` (ns) | `prompt_duration_ms` | `server_stats.prompt_duration_ms` | `server_stats_json.prompt_duration_ms` | `server_prefill_duration_ms` | `SERVER_PREFILL_DURATION_MS` | `server_prefill_duration_ms` |
| **Server Decode Duration** | `eval_duration` (ns) | `eval_duration_ms` | `server_stats.eval_duration_ms` | `server_stats_json.eval_duration_ms` | `server_decode_duration_ms` | `SERVER_DECODE_DURATION_MS` | `server_decode_duration_ms` |
| **Client Output Throughput** | N/A (computed) | N/A | `output_tokens_per_second` | `output_tokens_per_second` | `client_e2e_output_throughput_tok_s` | `CLIENT_E2E_OUTPUT_THROUGHPUT_TOK_S` | `client_e2e_output_throughput_tok_s` |
| **Server Decode Throughput** | N/A (computed) | `eval_tokens_per_second` | `server_stats.eval_tokens_per_second` | `server_stats_json.eval_tokens_per_second` | `server_decode_throughput_tok_s` | `SERVER_DECODE_THROUGHPUT_TOK_S` | `server_decode_throughput_tok_s` |
| **Time-to-First-Chunk (TTFC)** | N/A (client timer) | N/A | `ttfc_ms` | `ttfc_ms` | `ttfc_ms` | `TTFC_MS` | `ttfc_ms` |
| **Prompt Cache Count** | `prompt_eval_cached_count` | `prompt_eval_cached_count` | `server_stats.prompt_eval_cached_count` | `server_stats_json.prompt_eval_cached_count` | `server_prompt_eval_cached_count` | N/A (provenance only) | `prompt_eval_cached_count` |

---

## 3. Unit Conversions & Precision Verification

1. **Nanosecond-to-Millisecond Scaling**:
   - Ollama API emits duration values in nanoseconds (`10^-9` seconds).
   - In `llm_ii_engine/adapters/ollama.py`:
     ```python
     prompt_duration_ms = raw["prompt_eval_duration"] / 1e6
     eval_duration_ms = raw["eval_duration"] / 1e6
     total_duration_ms = raw["total_duration"] / 1e6
     load_duration_ms = raw["load_duration"] / 1e6
     ```
   - Verified: All adapter-level durations are normalized to floating-point milliseconds (`ms`) with $0.001\text{ ms}$ precision before aggregation.

2. **Throughput Scaling**:
   - `eval_tokens_per_second = eval_count / (eval_duration / 1e9)`
   - Verified: Expressed strictly in integer tokens per real second (`tok/s`).

3. **No Intermediate Double-Conversion**:
   - Millisecond values are preserved throughout SQLite, Pydantic domain models, and JSON serialization without round-trip precision degradation.

---

## 4. Verification and Regression Tests

Regression tests in `tests/unit/test_session_16_measurement_contract.py` confirm:
1. End-to-end vs decode throughput separation across all adapters and schemas.
2. Full property accessibility on historical and newly generated datasets.
3. Bidirectional schema mapping consistency between backend Pydantic models and frontend TypeScript types.
