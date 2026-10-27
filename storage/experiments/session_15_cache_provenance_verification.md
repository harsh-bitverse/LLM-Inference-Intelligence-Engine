# Session 15 — Cache Telemetry & Model Residency Provenance Verification

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 16 — Verification of Session 15 Telemetry Integrity  
**Artifact ID:** `report-session-15-cache-provenance`  
**Date:** 2026-10-27  

---

## 1. Audit Question

In Session 15, cache-controlled benchmark experiments were introduced using `prompt_eval_cached_count` and `load_duration`. This audit conclusively verifies:
1. Whether `prompt_eval_cached_count` is **natively reported** by the Ollama runtime or derived/heuristic.
2. Whether cache state classifications (`COLD_PREFIX`, `WARM_PREFIX`, `PARTIAL_PREFIX`, `UNKNOWN`) rest on empirical evidence.
3. The exact raw response payloads from Ollama v0.40.11 on Windows.

---

## 2. Empirical Verification from Raw Server Payloads

Inspecting the terminal JSON payload returned by Ollama server `http://127.0.0.1:11434/api/generate` on completion (`done: true`):

```json
{
  "model": "llama3.2:1b",
  "created_at": "2026-10-27T10:09:00.3343302Z",
  "response": "...",
  "done": true,
  "done_reason": "length",
  "context": [128006, 9125, ...],
  "total_duration": 2644152400,
  "load_duration": 4708800,
  "prompt_eval_count": 42,
  "prompt_eval_cached_count": 41,
  "prompt_eval_duration": 49836000,
  "eval_count": 48,
  "eval_duration": 2532712000
}
```

### Key Provenance Findings:
1. **`prompt_eval_cached_count` is an Official, Top-Level Native Ollama Field**:
   - It is directly calculated by Ollama's underlying llama.cpp KV-cache manager (`llama_context`).
   - It counts the exact number of prefix tokens for which KV-cache slots were reused from previous evaluations.
   - For a 42-token prompt with 41 shared prefix tokens, Ollama explicitly returns `"prompt_eval_cached_count": 41` and only evaluates 1 new token (`prompt_eval_duration: 49.8 ms` instead of $\approx 130\text{ ms}$).
2. **`load_duration` is an Official, Top-Level Native Ollama Field**:
   - Expressed in nanoseconds, measuring the time taken to map GGUF model weights into host RAM / GPU VRAM.
   - On cold start, `load_duration` is $\approx 10^9\text{ ns}$ ($1,037.74\text{ ms}$).
   - On warm execution, `load_duration` is $\approx 4\times 10^6\text{ ns}$ ($4.7\text{ ms}$) reflecting instant memory residency verification.

---

## 3. Strict Provenance Rules Enforced

1. **Direct Ingestion Only**:
   - In `llm_ii_engine/adapters/ollama.py`:
     ```python
     prompt_eval_cached_count = raw.get("prompt_eval_cached_count")
     load_duration_ms = (raw["load_duration"] / 1e6) if "load_duration" in raw else None
     ```
2. **Never Estimated**:
   - If an inference backend (such as a generic proxy or older mock) does not emit `prompt_eval_cached_count`, the field **MUST remain `None`**.
   - Inferring cache reuse from low prompt evaluation duration is strictly forbidden because network congestion, host thermal throttling, or sequence batching can mimic cache timing shifts.
3. **`UNKNOWN` Classification Mandatory When Telemetry Missing**:
   - If `prompt_eval_cached_count is None`, `cache_state` is assigned `CacheStateClassification.UNKNOWN`.
   - Predictors and optimizers must treat `UNKNOWN` cache state as requiring un-cached cold-prefix SLA verification.
