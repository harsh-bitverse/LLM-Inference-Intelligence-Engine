# Authoritative Measurement Contract

## Project: LLM Inference Intelligence Engine (IIE)
**Status:** Authoritative Standard (Session 16)  
**Applies to:** Adapters, Telemetry, Persistence, Dataset Builders, Prediction Models, Optimizer, REST API, Frontend Dashboard

---

## 1. Executive Summary

This document establishes the single authoritative, scientifically defensible measurement contract for the LLM Inference Intelligence Engine (IIE). It eliminates historical semantic ambiguity between client-observed end-to-end metrics and server-reported autoregressive execution metrics.

All IIE components (adapters, collectors, SQLite repositories, export serializers, prediction targets, optimizer objectives, REST schemas, and TypeScript frontend interfaces) **MUST** adhere to the exact naming, units, provenance constraints, and fallback semantics specified herein.

---

## 2. The 10 Core Authoritative Metrics

| Metric Key | Semantic Definition | Authoritative Clock / Source | Physical Unit | Nullability & Preconditions |
| :--- | :--- | :--- | :--- | :--- |
| `client_e2e_duration_ms` | Elapsed monotonic client time from initial request dispatch to terminal chunk consumption. | Client monotonic clock (`time.perf_counter()`) | Milliseconds (`ms`) | **Non-null** on successful requests. |
| `server_total_duration_ms` | Total generation time reported by the inference engine runtime. | Server runtime telemetry (`total_duration` / $10^6$) | Milliseconds (`ms`) | Nullable if server does not emit terminal statistics. |
| `server_load_duration_ms` | Elapsed time spent loading model weights into host memory / GPU VRAM. | Server runtime telemetry (`load_duration` / $10^6$) | Milliseconds (`ms`) | Nullable if model is already memory-resident. |
| `server_prefill_duration_ms` | Elapsed time spent evaluating the prompt token sequence (prompt prefill phase). | Server runtime telemetry (`prompt_eval_duration` / $10^6$) | Milliseconds (`ms`) | Nullable if unobserved or cached out. |
| `server_decode_duration_ms` | Elapsed time spent executing autoregressive token decoding (eval phase). | Server runtime telemetry (`eval_duration` / $10^6$) | Milliseconds (`ms`) | Nullable if zero tokens generated. |
| `client_e2e_output_throughput_tok_s` | Application-level output delivery throughput: `actual_output_tokens` / (`client_e2e_duration_ms` / $1000$). | Client monotonic duration & verified output count | Tokens / sec (`tok/s`) | Nullable if output count is 0 or duration is 0. |
| `server_decode_throughput_tok_s` | Engine-level autoregressive generation throughput: `actual_output_tokens` / (`server_decode_duration_ms` / $1000$). | Server runtime decode duration & output count | Tokens / sec (`tok/s`) | Nullable if decode duration is null or $\le 0$. |
| `ttfc_ms` | Time-to-First-Chunk: elapsed client monotonic time from request dispatch to first text chunk arrival. | Client monotonic stream timestamp | Milliseconds (`ms`) | Nullable for non-streaming requests. |
| `ttft_ms` | Time-to-First-Token: elapsed time from request dispatch to first verified discrete token generation event. | Client monotonic clock with discrete token boundary | Milliseconds (`ms`) | **MUST be null** for chunk-streaming protocols (e.g. Ollama `/api/generate`). |
| `tpot_ms` | Time-Per-Output-Token: mean inter-token arrival latency across verified discrete token events. | Discrete token event timestamps | Milliseconds (`ms`) | **MUST be null** unless true discrete token events with $\ge 2$ tokens are observed. |

---

## 3. Mathematical Timing Decomposition

For any single generation request, total elapsed duration decomposes into distinct physical phases:

$$\text{client\_e2e\_duration\_ms} = \text{server\_total\_duration\_ms} + \epsilon_{\text{client\_overhead}}$$

$$\text{server\_total\_duration\_ms} = \text{server\_load\_duration\_ms} + \text{server\_prefill\_duration\_ms} + \text{server\_decode\_duration\_ms} + \epsilon_{\text{server\_overhead}}$$

where:
1. $\epsilon_{\text{server\_overhead}} \in [0, 60]\text{ ms}$: Internal server tokenization, thread pool scheduling, and terminal JSON serialization.
2. $\epsilon_{\text{client\_overhead}} \in [5, 40]\text{ ms}$: Local TCP loopback socket connection, HTTP parser overhead, and client stream event dispatch.

### Provenance of Residuals
In cold-start inference (e.g., Session 15 `s15-expA-rep1`):
- $\text{client\_e2e\_duration\_ms} = 3,062.53\text{ ms}$
- $\text{server\_prefill\_duration\_ms} = 449.28\text{ ms}$
- $\text{server\_decode\_duration\_ms} = 1,569.12\text{ ms}$
- Sum of prefill + decode = $2,018.40\text{ ms}$
- Difference = $1,044.13\text{ ms}$
- $\text{server\_load\_duration\_ms} = \mathbf{1,037.74\text{ ms}}$
- Residual client loopback overhead = $1,044.13 - 1,037.74 = \mathbf{6.39\text{ ms}}$

**Conclusion:** The $\approx 1,044\text{ ms}$ gap is **100% physically accounted for** by model loading ($1,037.74\text{ ms}$) and loopback serialization ($6.39\text{ ms}$).

---

## 4. Throughput Bifurcation Rule

Engineers and automated pipelines **MUST NEVER** emit an unqualified `throughput` metric without specifying the measurement domain:

1. **Client End-to-End Throughput (`client_e2e_output_throughput_tok_s`)**:
   $$\text{Throughput}_{\text{client}} = \frac{N_{\text{output}}}{T_{\text{client\_e2e}} / 1000}$$
   - Measures end-user experience, incorporating cold load, prompt prefill latency, and network overhead.
   - For $P_{677}\_O_{64}$, this is $\approx \mathbf{12.80\text{ tok/s}}$.

2. **Server Decode Throughput (`server_decode_throughput_tok_s`)**:
   $$\text{Throughput}_{\text{decode}} = \frac{N_{\text{output}}}{T_{\text{server\_decode}} / 1000}$$
   - Measures raw autoregressive execution throughput of the hardware/kernel.
   - For $P_{677}\_O_{64}$, this is $\approx \mathbf{22.72\text{ tok/s}}$.

---

## 5. Cache Telemetry & Provenance Rules

1. `prompt_eval_cached_count`:
   - Populated **strictly** from server-reported responses (`done: true` payload).
   - Provenance is `MEASURED`.
   - If missing from server response, it **MUST remain `None`**. Estimating it from prompt evaluation latency is strictly prohibited.
2. `cache_state` Classification:
   - `WARM_PREFIX`: Verified server-reported cached tokens $> 0$ with matching prefix.
   - `COLD_PREFIX`: Verified server-reported cached tokens $== 0$.
   - `UNKNOWN`: Missing server telemetry or unmetered cache runtime.
3. `model_residency_state` Classification:
   - `COLD`: First execution after process startup, model load duration $> 100\text{ ms}$, or explicitly evicted from memory.
   - `WARM`: Model confirmed memory-resident prior to request dispatch (`load_duration < 10\text{ ms}`).
   - `UNKNOWN`: Uninstrumented residency state.

---

## 6. Prospective Feature Safety Rules

Predictors operating prospectively (before execution) **MUST NOT** consume:
- `actual_output_tokens` or `output_tokens`
- `server_prompt_eval_cached_count`
- `chunk_count`
- Any measured latency or throughput metric

All prospective predictions must rely solely on features known prior to execution: `prompt_tokens`, `requested_output_tokens` (or empirical output ratio), `concurrency`, and declared `model_residency_state`.
Retrospective predictions using `actual_output_tokens` are permitted solely for post-hoc error diagnostic comparisons.
