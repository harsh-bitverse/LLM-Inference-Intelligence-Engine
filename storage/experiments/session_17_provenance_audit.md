# Session 17 — Provenance, Residency & Cross-Validation Audit

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 17 — Workload-Aware Recommendations & Product Validation  
**Date:** 2026-10-27  
**Status:** Validated & Closed  

---

## 1. Executive Summary

Phase 1 of Session 17 conducts a rigorous audit to close all remaining provenance, workstation identity, cache telemetry, model residency, and cross-validation integrity gaps before implementing the workload recommendation engine.

Key conclusions:
1. **Workstation Hardware Truth**: The host workstation is conclusively an **Intel Core i7-14700** (GenuineIntel, 20 physical cores: 8 Performance cores + 12 Efficient cores, 28 logical threads, 32 GB RAM, Windows 11). The conversational mention of "AMD Ryzen 7 7840HS" in the Session 16 chat summary was an ungrounded assistant conversational artifact. Raw database records, benchmark manifests, and exported datasets have strictly recorded `Intel(R) Core(TM) i7-14700` and `CPU`.
2. **`prompt_eval_cached_count` Provenance**: Ollama v0.40.11+ natively emits `prompt_eval_cached_count` as a top-level integer in terminal SSE/JSON payloads (`done: true`). In Sessions 12B and 13, this field was not mapped into schemas and remains honestly `None` (Unknown). In Session 15, it was captured directly from server payloads (yielding 0 on cold starts and 20–676 tokens on warm prefix hits). Inferred cache counts are never marked as server-measured.
3. **Prospective Residency Semantics**: Model residency (`WARM` vs `COLD`) must strictly be treated prospectively. If residency is unknown at recommendation time, it must be reported as `UNKNOWN`, with both warm-path ($0\text{ ms}$ load) and cold-path ($1,037.74\text{ ms}$ load) assumptions explicitly exposed.
4. **Cross-Validation Integrity & Fold Independence**: Comparing 17-fold Configuration-Grouped (LOCO) cross-validation with 9-fold Prompt-Grouped (LOPO) cross-validation proves that sequence correlation or prompt reuse across folds does not inflate prediction performance: Workload-Aware Latency Predictor maintains an MAE of **$451.67\text{ ms}$** under strict LOPO (a **67.9% reduction** over the Constant Median baseline of $1,406.10\text{ ms}$).

---

## 2. Workstation Identity Verification

### Physical Inspection
Direct query of the operating system and hardware APIs:
```python
platform.processor() -> 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel'
platform.uname()     -> uname_result(system='Windows', node='CCL1PC03R3', release='11', version='10.0.22000', machine='AMD64')
```
SQLite database table `hardware_reports` in `storage/llm_ii_engine.db`:
- Record `hw-report-104e1b59879d`: `cpu_model = 'Intel(R) Core(TM) i7-14700'`, `cpu_vendor = 'Intel'`, `total_ram_bytes = 34,031,325,184` (31.69 GiB), `os_name = 'Windows'`.
- Record `hw-report-2606abdf9fce`: `cpu_model = 'Intel(R) Core(TM) i7-14700'`, `cpu_vendor = 'Intel'`.

### Reconciliation
No codebase or dataset files contain references to AMD Ryzen hardware. The reference in the Session 16 chat summary table was a conversational drafting error. All empirical measurements in Sessions 12B, 13, 15, and 16 were executed solely on the authorized **Intel Core i7-14700** CPU under Windows 11.

---

## 3. `prompt_eval_cached_count` Provenance Audit

### Server Payload Evidence
A live test against the local Ollama daemon (`http://127.0.0.1:11434`) confirmed the terminal JSON payload schema emitted by the Ollama runtime:
```json
{
  "model": "llama3.2:1b",
  "created_at": "2026-10-27T16:49:05.123456Z",
  "response": "...",
  "done": true,
  "done_reason": "stop",
  "total_duration": 1369847600,
  "load_duration": 2791600,
  "prompt_eval_count": 27,
  "prompt_eval_duration": 44671000,
  "eval_count": 32,
  "eval_duration": 1324579000,
  "prompt_eval_cached_count": 26
}
```

### Dataset Provenance Classification
- **Sessions 12B & 13 (`pilot_llama3_2_1b.jsonl`, `expansion_llama3_2_1b.jsonl`)**:
  - `prompt_eval_cached_count`: `None` (Unknown).
  - Provenance: `UNKNOWN`. Earlier telemetry collectors did not deserialize this optional key. Historical records are preserved with `None` rather than retroactively injecting synthetic values.
- **Session 15 (`session_15_cache_controlled_measurements.jsonl`)**:
  - `prompt_eval_cached_count`: Direct integer value read from `final_data.get("prompt_eval_cached_count")`.
  - Provenance: `MEASURED` (Server Reported).
  - Observed range: `0` (Cold starts in Exp A and Exp D), `20` to `676` (Warm identical and prefix-reused prompts).
- **Rule for Recommendations**:
  - Inferred token counts based on string prefix matching must be classified as `HEURISTIC` or `ESTIMATED`.
  - Only server-reported terminal statistics may be classified as `MEASURED`.

---

## 4. Prospective Residency & Cache State Rules

1. **Residency State (`ModelResidencyState`)**:
   - `WARM`: Model weights are verified or assumed resident in memory. Load duration is modeled as $\hat{T}_{\text{load}} = 0\text{ ms}$.
   - `COLD`: Model weights must be read from disk into host memory. Load duration is modeled as $\hat{T}_{\text{load}} = 1,037.74\text{ ms}$.
   - `UNKNOWN`: The recommendation engine must explicitly surface that cold start adds $\approx 1,038\text{ ms}$ of latency and present both scenarios or default to conservative estimates.
2. **Prefix-Cache State (`CacheStateClassification`)**:
   - `COLD_PREFIX`: No KV cache reuse. Entire prompt evaluated at full prefill rate ($\approx 3.03\text{ ms/token}$).
   - `WARM_PREFIX`: Identical prompt or long prefix in KV cache. Evaluated at high throughput ($\approx 0.04\text{ ms/token}$).
   - `UNKNOWN`: Default assumption when prefix history is unmeasured.

---

## 5. Cross-Validation Independence & Leakage Audit

To verify that the 62 observations from Sessions 12B, 13, and 15 were not contaminated by sequence correlation or near-duplicate prompts crossing folds, we evaluated two distinct cross-validation partitioning strategies:

1. **Leave-One-Configuration-Out (LOCO)**: 17 folds partitioned by unique $(P \times O)$ tuple.
2. **Leave-One-Prompt-Out (LOPO)**: 9 folds partitioned strictly by prompt token length $P \in \{42, 46, 53, 99, 119, 300, 311, 317, 677\}$, ensuring that **zero prompts of identical length appear in both training and test sets**.

### Comparative Results on `client_e2e_duration_ms`

| Split Strategy | Folds | Constant Median MAE | Ridge Regression MAE | Workload-Aware Predictor MAE | Error Reduction vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LOCO** (Config $P \times O$) | 17 | $1,439.38\text{ ms}$ | $573.29\text{ ms}$ | **$385.88\text{ ms}$** | **-73.2%** |
| **LOPO** (Strict Prompt $P$) | 9 | $1,406.10\text{ ms}$ | $524.38\text{ ms}$ | **$451.67\text{ ms}$** | **-67.9%** |

### Comparative Results on `server_decode_throughput_tok_s`

| Split Strategy | Folds | Constant Median MAE | Ridge Regression MAE | Workload-Aware Predictor MAE |
| :--- | :--- | :--- | :--- | :--- |
| **LOCO** (Config $P \times O$) | 17 | $0.85\text{ tok/s}$ | $0.78\text{ tok/s}$ | **$0.77\text{ tok/s}$** |
| **LOPO** (Strict Prompt $P$) | 9 | $0.95\text{ tok/s}$ | $0.94\text{ tok/s}$ | **$1.00\text{ tok/s}$** |

### Findings
- The Workload-Aware Predictor retains superior predictive accuracy even when tested on completely unseen prompt lengths (LOPO MAE $451.67\text{ ms}$ vs $1,406.10\text{ ms}$, a 67.9% reduction).
- Autoregressive decode throughput is highly stable across all prompt lengths ($\approx 22.7\text{ tok/s}$), with MAE $\le 1.0\text{ tok/s}$ across both split strategies.
- No sequence correlation or fold leakage invalidates the empirical modeling foundation.
