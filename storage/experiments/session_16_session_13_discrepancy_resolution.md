# Session 16 — Session 13 Throughput Discrepancy Forensic Resolution

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 16 — Measurement Contract Verification & Discrepancy Forensic Analysis  
**Artifact ID:** `report-session-16-discrepancy-resolution`  
**Date:** 2026-10-27  

---

## 1. Discrepancy Statement

In Session 13 and Session 14 artifacts, an apparent contradiction was identified for the configuration with 677 prompt tokens and 64 requested output tokens ($P_{677}\_O_{64}$ on `llama3.2:1b` Q8_0):

1. **Manifest & Raw Benchmark Record**: Reported an elapsed request duration of **$\approx 5.00\text{ seconds}$** ($4,999.32\text{ ms}$) with an output throughput of **$12.80\text{ tok/s}$**.
2. **Session 13 Validation Report Table**: Reported an elapsed duration of **$\approx 2.86\text{ seconds}$** ($2,864.12\text{ ms}$) with an output throughput of **$22.35\text{ tok/s}$**.

This forensic audit identifies the exact records, requests, code paths, and root cause of the contradiction.

---

## 2. Evidence Trace & Raw Record Audit

### Request Identifiers
The $P_{677}\_O_{64}$ workload was measured across 3 distinct repetitions in experiment `exp-llama3-2-1b-expansion`:
- **Repetition 1:** `req-0012`
- **Repetition 2:** `req-0019`
- **Repetition 3:** `req-0026`

### Raw Telemetry from SQLite (`request_benchmarks`) & JSONL
Inspecting the exact raw server responses stored in `storage/llm_ii_engine.db` and `storage/datasets/expansion_llama3_2_1b.jsonl`:

| Request ID | Repetition | Client Duration (`duration_ms`) | Server Prefill Duration (`prompt_duration_ms`) | Server Decode Duration (`eval_duration_ms`) | Server Total Duration (`total_duration_ms`) | Output Tokens | Client Throughput (`output_tokens_per_second`) | Server Decode Throughput (`eval_tokens_per_second`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `req-0012` | 1 | 5,164.71 ms | 2,138.83 ms | 2,864.12 ms | 5,093.30 ms | 64 | 12.39 tok/s | 22.35 tok/s |
| `req-0019` | 2 | 4,923.75 ms | 2,012.33 ms | 2,804.81 ms | 4,896.72 ms | 64 | 13.00 tok/s | 22.82 tok/s |
| `req-0026` | 3 | 4,909.51 ms | 1,995.70 ms | 2,785.49 ms | 4,858.91 ms | 64 | 13.04 tok/s | 22.98 tok/s |
| **Mean** | — | **4,999.32 ms** | **2,048.95 ms** | **2,818.14 ms** | **4,949.64 ms** | **64** | **12.80 tok/s** | **22.72 tok/s** |

---

## 3. Forensic Root Cause

The raw benchmark instrumentation and database records were completely accurate and consistent throughout:
1. The **client end-to-end duration** was indeed **$4,999.32\text{ ms}$** ($\approx 5.0\text{ seconds}$), yielding an application throughput of **$12.80\text{ tok/s}$**.
2. The **server decode-only evaluation time** was **$2,818.14\text{ ms}$** ($2,864.12\text{ ms}$ in repetition 1), yielding a decode throughput of **$22.72\text{ tok/s}$** ($22.35\text{ tok/s}$ in repetition 1).
3. The server prompt prefill evaluation phase required **$2,048.95\text{ ms}$** ($\approx 2.05\text{ seconds}$) to ingest and evaluate the 677 prompt tokens on CPU.

### Where the Discrepancy Occurred
In `storage/datasets/expansion_llama3_2_1b_validation_report.md` (lines 78–82), the human author manually compiled a summary table. For configuration $P_{677}\_O_{64}$, the author mistakenly copied the **decode-only evaluation duration** (`eval_duration_ms = 2,864.12 ms`) and **decode-only throughput** (`eval_tokens_per_second = 22.35 tok/s`) into the general "Duration" and "Throughput" columns, completely omitting the $2,049\text{ ms}$ prompt evaluation phase.

---

## 4. Full Timing Reconciliation

$$\text{Client E2E Duration} (4,999.32\text{ ms}) = \text{Prefill} (2,048.95\text{ ms}) + \text{Decode} (2,818.14\text{ ms}) + \text{Loopback/Framing Overhead} (132.23\text{ ms})$$

$$\text{Client Throughput} = \frac{64\text{ tokens}}{4.99932\text{ s}} = \mathbf{12.80\text{ tok/s}}$$

$$\text{Server Decode Throughput} = \frac{64\text{ tokens}}{2.81814\text{ s}} = \mathbf{22.72\text{ tok/s}}$$

---

## 5. Resolution & Architectural Guardrails

To prevent this confusion permanently:
1. **Ambiguous `throughput` and `duration` terms are prohibited** in all tabular reports and database schemas.
2. The system now publishes two explicitly named metrics:
   - `client_e2e_output_throughput_tok_s` ($12.80\text{ tok/s}$)
   - `server_decode_throughput_tok_s` ($22.72\text{ tok/s}$)
3. Derived version 2 dataset (`storage/datasets/expansion_llama3_2_1b_v2.jsonl` and `.csv`) exposes both columns simultaneously, with backward-compatible aliases for historical pipelines.
4. Predictor evaluations and optimizer candidate evaluations explicitly distinguish whether the objective is application-level throughput or hardware decode speed.
