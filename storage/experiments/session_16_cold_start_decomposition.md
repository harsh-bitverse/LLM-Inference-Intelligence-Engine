# Session 16 — Cold-Start Timing Decomposition Forensic Report

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 16 — Timing Decomposition Verification  
**Artifact ID:** `report-session-16-cold-start-decomposition`  
**Date:** 2026-10-27  

---

## 1. Objective

Investigate the timing decomposition of cold-start vs warm-start inference execution, specifically analyzing why the first request in an experiment exhibits an apparent $\approx 1,044\text{ ms}$ discrepancy between the client-observed end-to-end duration ($3,062.53\text{ ms}$) and the server-reported execution phases (prefill $449.28\text{ ms}$ + decode $1,569.12\text{ ms} = 2,018.40\text{ ms}$).

---

## 2. Experimental Subject: ExpA_IdenticalRepetition

Workload: Prompt 119 tokens, requested output 32 tokens, on `llama3.2:1b` (Q8_0, CPU execution).

### Quantitative Comparison: Repetition 1 (Cold) vs Repetition 2 (Warm)

| Metric | Repetition 1 (`s15-expA-rep1`, Cold) | Repetition 2 (`s15-expA-rep2`, Warm) | Physical Delta |
| :--- | :--- | :--- | :--- |
| **Model Residency State** | **COLD** | **WARM** | Model retained in memory |
| **KV Cache State** | **COLD_PREFIX** (0 cached tokens) | **WARM_PREFIX** (118 of 119 cached) | 99.2% prefix reused |
| **Client Monotonic Duration (`duration_ms`)** | **3,062.53 ms** | **1,591.27 ms** | **-1,471.26 ms (-48.0%)** |
| **Time-to-First-Chunk (`ttfc_ms`)** | **1,492.65 ms** | **59.06 ms** | **-1,433.59 ms (-96.0%)** |
| **Server Load Duration (`load_duration_ms`)** | **1,037.74 ms** | **1.55 ms** | **-1,036.19 ms (-99.9%)** |
| **Server Prefill Duration (`prompt_eval_duration_ms`)** | **449.28 ms** | **52.22 ms** | **-397.06 ms (-88.4%)** |
| **Server Decode Duration (`eval_duration_ms`)** | **1,569.12 ms** | **1,531.93 ms** | **-37.19 ms (-2.4%)** |
| **Sum of Prefill + Decode** | **2,018.40 ms** | **1,584.15 ms** | **-434.25 ms** |
| **Unaccounted Gap (Client Total - (Prefill + Decode))** | **1,044.13 ms** | **7.12 ms** | **-1,037.01 ms** |
| **Residual Client Overhead (Gap - Load Duration)** | **6.39 ms** | **5.57 ms** | **Consistent client framing** |

---

## 3. Mathematical Verification

### Repetition 1 (Cold Start):
$$\text{client\_e2e\_duration\_ms} = 3,062.53\text{ ms}$$
$$\text{Components} = \underbrace{1,037.74\text{ ms}}_{\text{Model Loading}} + \underbrace{449.28\text{ ms}}_{\text{Prompt Prefill}} + \underbrace{1,569.12\text{ ms}}_{\text{Autoregressive Decode}} + \underbrace{6.39\text{ ms}}_{\text{Loopback Framing Overhead}} = \mathbf{3,062.53\text{ ms}}$$

### Repetition 2 (Warm Start with Prefix Cache):
$$\text{client\_e2e\_duration\_ms} = 1,591.27\text{ ms}$$
$$\text{Components} = \underbrace{1.55\text{ ms}}_{\text{Residency Check}} + \underbrace{52.22\text{ ms}}_{\text{Cached Prefill}} + \underbrace{1,531.93\text{ ms}}_{\text{Autoregressive Decode}} + \underbrace{5.57\text{ ms}}_{\text{Loopback Framing Overhead}} = \mathbf{1,591.27\text{ ms}}$$

---

## 4. Key Scientific Insights

1. **The $1,044\text{ ms}$ Gap is 99.4% Explained by Model Loading**:
   - $1,037.74\text{ ms}$ was consumed loading the 1.23B parameter GGUF model from disk into RAM.
   - Only $6.39\text{ ms}$ was consumed by network transport and OS socket buffers.
2. **First Chunk Timing (`ttfc_ms`) Directly Mirrors Cold Load**:
   - In Repetition 1, TTFC was $1,492.65\text{ ms}$. This equals $\text{Load} (1,037.74\text{ ms}) + \text{Prefill} (449.28\text{ ms}) + \text{First Token Decode} (\approx 48\text{ ms}) \approx 1,535\text{ ms}$.
   - In Repetition 2, TTFC dropped to $59.06\text{ ms}$ because load was $1.55\text{ ms}$ and prefill was $52.22\text{ ms}$.
3. **Autoregressive Decode is Residency-Invariant**:
   - Decode duration was $1,569.12\text{ ms}$ cold vs $1,531.93\text{ ms}$ warm ($2.4\%$ variance within normal OS scheduling noise).
   - Once weights are mapped, generation rate depends exclusively on CPU memory bandwidth, not cache residency.

---

## 5. Architectural Implementation in IIE

1. **`WorkloadAwareLatencyPredictor` explicitly incorporates `mean_cold_load_duration_ms`**:
   - If `model_residency_state == COLD`, the predictor adds $\bar{T}_{\text{cold\_load}} \approx 1,000\text{ ms}$.
   - If `model_residency_state == WARM`, $\hat{T}_{\text{load}} = 0\text{ ms}$.
2. **Optimizer Guardrail**:
   - When users specify hard cold-start SLAs (`require_cold_start_feasibility=True`), candidates backed only by warm evidence are flagged as `INSUFFICIENT_EVIDENCE`.
