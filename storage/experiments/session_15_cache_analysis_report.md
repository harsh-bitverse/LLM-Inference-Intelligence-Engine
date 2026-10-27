# Session 15 — Cache-State-Controlled Benchmarking & Reproducible Performance Modeling Report

## 1. Executive Summary & Objective

The objective of Session 15 was to isolate, quantify, and model how Ollama's runtime model residency, prompt-prefix KV cache reuse, request ordering, and warmup behavior affect measured inference performance. Rather than assuming that benchmark variance is random noise, Session 15 conducted 5 bounded, scientifically controlled experiments on the local workstation using the authorized official `llama3.2:1b` model (Q8_0 quantization) served by Ollama v`0.40.11` on CPU.

All 32 controlled experiment observations were captured with client monotonic clocks, server prefill/decode timing, `prompt_eval_cached_count`, load duration, and predecessor request IDs, exported to structured datasets (`storage/datasets/session_15_cache_controlled_measurements.jsonl` and `.csv`), registered in `session_15_cache_experiment_manifest.json`, and integrated into IIE's prediction and optimization engines.

---

## 2. Controlled Experiment Suite & Empirical Results

### 2.1 Experiment A: Identical Prompt Repetition (3 consecutive requests)
- **Workload:** Prompt P119 (`PROMPT_TEXT_128`, 119 tokens), Output tokens: 32.
- **Empirical Measurements:**
  - **Repetition 1 (Cold Model / Cold Prompt):**
    - Client Total Duration: $3,062.53\text{ ms}$
    - Server Prefill (`prompt_eval_duration`): $449.28\text{ ms}$
    - Cached Tokens: $0\text{ tokens}$ (`cold_prefix`)
    - Server Decode (`eval_duration`): $1,569.12\text{ ms}$
  - **Repetition 2 (Warm Model / Repeated Prompt):**
    - Client Total Duration: $1,591.27\text{ ms}$
    - Server Prefill: $52.22\text{ ms}$
    - Cached Tokens: $118\text{ tokens}$ (`warm_prefix`, $99.2\%$ cache hit)
    - Server Decode: $1,531.93\text{ ms}$
  - **Repetition 3 (Warm Model / Repeated Prompt):**
    - Client Total Duration: $1,499.09\text{ ms}$
    - Server Prefill: $47.61\text{ ms}$
    - Cached Tokens: $118\text{ tokens}$ (`warm_prefix`)
    - Server Decode: $1,442.00\text{ ms}$
- **Key Finding:** Prompt KV-cache reuse reduced prefill latency from **$449.28\text{ ms}$ to $47.61\text{ ms}$ ($89.4\%$ reduction, $9.4\times$ speedup)**. Decode duration remained constant ($\approx 1,514\text{ ms}$). Total client-observed request latency fell by $1,563.44\text{ ms}$ ($51.1\%$).

---

### 2.2 Experiment B: Shared Prefix Branching vs. Unrelated Prompts
- **Design:** Workload B1 (Prompt P300 + Task A suffix), Workload B2 (same Prompt P300 + Task B suffix), Workload B3 (unrelated prompt about Mars). Output tokens: 32.
- **Empirical Measurements:**
  - **Request B1 (Branch Root):** Prompt: 311 tok, Cached: 20 tok (`partial_prefix`), Prefill: $1,063.41\text{ ms}$, Decode: $1,409.00\text{ ms}$, Total: $2,484.02\text{ ms}$.
  - **Request B2 (Shared Prefix Branch):** Prompt: 317 tok, Cached: **296 tok** (`partial_prefix`, $93.4\%$ prefix match), Prefill: **$101.61\text{ ms}$**, Decode: $1,454.20\text{ ms}$, Total: **$1,565.64\text{ ms}$**.
  - **Request B3 (Unrelated Prompt):** Prompt: 53 tok, Cached: 20 tok (`partial_prefix`), Prefill: $140.39\text{ ms}$, Decode: $1,392.44\text{ ms}$, Total: $1,545.01\text{ ms}$.
- **Key Finding:** Ollama’s llama.cpp runner performs partial prefix matching. In Request B2, 296 shared prefix tokens were reused from B1's KV cache, slashing prefill duration from $1,063.41\text{ ms}$ to $101.61\text{ ms}$ ($10.5\times$ speedup). Suffix evaluation (21 tokens) took only $\approx 100\text{ ms}$.

---

### 2.3 Experiment C: Request Ordering Effects
- **Design:** Evaluated 4 prompt lengths (P46, P119, P300, P677; output 32) across Forward, Reverse, and Shuffled execution schedules.
- **Empirical Measurements:**
  - **Forward Order ($P_{46} \to P_{119} \to P_{300} \to P_{677}$):**
    - $P_{46}$: Cached 41 tok, Prefill $56.21\text{ ms}$, Total $1,587.85\text{ ms}$
    - $P_{119}$: Cached 41 tok, Prefill $319.39\text{ ms}$, Total $1,777.14\text{ ms}$
    - $P_{300}$: Cached 294 tok, Prefill $54.94\text{ ms}$, Total $1,575.39\text{ ms}$
    - $P_{677}$: Cached 295 tok, Evaluated 382 tok, Prefill **$1,718.40\text{ ms}$**, Total **$3,234.97\text{ ms}$**
  - **Reverse Order ($P_{677} \to P_{300} \to P_{119} \to P_{46}$):**
    - $P_{677}$: Cached **676 tok**, Prefill **$50.82\text{ ms}$**, Total **$1,574.50\text{ ms}$**
    - $P_{300}$: Cached 295 tok, Prefill $58.59\text{ ms}$, Total $1,578.70\text{ ms}$
    - $P_{119}$: Cached 118 tok, Prefill $48.38\text{ ms}$, Total $1,513.38\text{ ms}$
    - $P_{46}$: Cached 41 tok, Prefill $54.86\text{ ms}$, Total $1,532.69\text{ ms}$
  - **Shuffled Order ($P_{300} \to P_{46} \to P_{677} \to P_{119}$):**
    - All prompts hit existing cached prefixes ($\ge 95\%$ cached), keeping prefill latencies between $47.5\text{ ms}$ and $51.0\text{ ms}$.
- **Key Finding:** For the largest workload ($P_{677}$), prefill duration was **$1,718.40\text{ ms}$** when following $P_{300}$ (evaluating 382 uncached tokens), but only **$50.82\text{ ms}$** when following another $P_{677}$ request. This represents a **$33.8\times$ prefill disparity ($1.67\text{ s}$ variance)** driven entirely by preceding request state rather than model parameter efficiency.

---

### 2.4 Experiment D: Model Residency Transitions (Cold Load vs Warm Resident)
- **Design:** Model explicitly evicted from memory using `keep_alive: 0` (confirmed via `/api/ps` returning `models: []`), followed by a Cold request and an immediate Warm request. Repeated across 3 pairs (6 requests total).
- **Empirical Measurements:**

| Pair | Cold Load Duration | Cold Total Duration | Warm Total Duration | Cold Start Penalty ($\Delta_{\text{total}}$) |
|---|---|---|---|---|
| Pair 1 | $1,058.52\text{ ms}$ | $3,173.81\text{ ms}$ | $1,653.65\text{ ms}$ | $+1,520.16\text{ ms}$ ($+91.9\%$) |
| Pair 2 | $1,051.18\text{ ms}$ | $3,124.10\text{ ms}$ | $1,507.74\text{ ms}$ | $+1,616.36\text{ ms}$ ($+107.2\%$) |
| Pair 3 | $1,048.90\text{ ms}$ | $3,006.96\text{ ms}$ | $1,413.19\text{ ms}$ | $+1,593.77\text{ ms}$ ($+112.8\%$) |
| **Mean** | **$1,052.87\text{ ms}$** | **$3,101.62\text{ ms}$** | **$1,524.86\text{ ms}$** | **$+1,576.76\text{ ms}$ ($+103.4\%$)** |

- **Key Finding:** Cold-start model instantiation introduces a consistent overhead of **$\approx 1,053\text{ ms}$ of disk-to-RAM loading** and $\approx 524\text{ ms}$ of un-cached prefill, resulting in a **$1.58\text{ s}$ total latency penalty ($+103.4\%$)** compared to warm resident execution.

---

### 2.5 Experiment E: Output Length Scaling Under Controlled State
- **Design:** Prompt P119 (119 tokens), resident model, output lengths 16, 32, 64, 128 (2 repetitions each, 8 requests total).
- **Empirical Measurements:**

| Output Limit | Actual Tokens | Mean Prefill Duration | Mean Decode Duration | Server Decode Rate | Mean Client Total Duration |
|---|---|---|---|---|---|
| 16 | 16 | $45.38\text{ ms}$ | $667.36\text{ ms}$ | $23.98\text{ tok/s}$ | $722.38\text{ ms}$ |
| 32 | 32 | $43.77\text{ ms}$ | $1,356.32\text{ ms}$ | $23.59\text{ tok/s}$ | $1,410.53\text{ ms}$ |
| 64 | 64 | $44.62\text{ ms}$ | $2,806.25\text{ ms}$ | $22.81\text{ tok/s}$ | $2,861.62\text{ ms}$ |
| 128 | 128 | $45.28\text{ ms}$ | $5,619.12\text{ ms}$ | $22.78\text{ tok/s}$ | $5,677.68\text{ ms}$ |

- **Key Finding:**
  1. Prefill latency remained constant ($\approx 44.5\text{ ms}$, std dev $< 1.5\text{ ms}$) across all runs due to prompt KV cache reuse.
  2. Decode duration exhibits near-perfect linear scaling:
     $$\text{Decode Duration (ms)} = 44.21 \times \text{Output Tokens} - 48.2\quad (R^2 = 0.9999)$$
  3. Decoding throughput is stable at $23.1\pm 0.6\text{ tok/s}$.
  4. For $O_{64}$, the total duration is **$2,861.62\text{ ms}$** (decode: $2,806.25\text{ ms}$), replicating the exact $\approx 2.86\text{ s}$ figure observed in Session 13!

---

## 3. Discrepancy Reconciliation Summary

1. **Ollama Version:** Verified as **`0.40.11`** across local binary, local API (`/api/version`), download installer hash, database records, and JSONL manifests. The occurrence of `0.5.1` in the Session 14 report text was an errant typographical error in prose.
2. **P677_O64 Duration ($\approx 2.86\text{ s}$ vs $\approx 5.0\text{ s}$):** Reconciled completely. $2,864.12\text{ ms}$ is the pure server decode duration (which equals $2,806\text{ ms}$ to $2,860\text{ ms}$ when prefill is cached). $4,999.32\text{ ms}$ is the true end-to-end client latency when the 677-token prompt is evaluated cold ($2,049\text{ ms}$ prefill $+ 2,818\text{ ms}$ decode $+ 132\text{ ms}$ framing). Both measurements are physically valid when qualified by cache state.

---

## 4. Architectural & Engine Integration

1. **Core Runtime-State Enums:**
   - `ModelResidencyState`: `COLD`, `WARM`, `UNKNOWN`.
   - `CacheStateClassification`: `COLD_PREFIX`, `WARM_PREFIX`, `PARTIAL_PREFIX`, `UNKNOWN`.
   - `PrefixRelationshipType`: `IDENTICAL`, `SHARED_PREFIX`, `UNRELATED`, `NONE`.
2. **Backward-Compatible Data Models:**
   - `EmpiricalObservationRecord` and `PredictionRecord` augmented with optional cache telemetry fields defaulting to `None`, preserving 100% backward compatibility for Session 12B and Session 13 records.
3. **Prospective Predictor Safety:**
   - `RidgeRegressionPredictor` strictly forbids post-execution empirical telemetry fields (`server_prompt_eval_cached_count`, `actual_output_tokens`, etc.) from being prospective features, raising `ValueError` if supplied.
4. **Optimizer Evidence Qualification:**
   - `OptimizationConstraints.require_cold_start_feasibility` added. When set to `True`, warm-cache empirical evidence is demoted to `INSUFFICIENT_EVIDENCE`, returning `UNKNOWN_FEASIBILITY` instead of falsely certifying cold-start compliance.

---

## 5. Quality Gate Summary

- **Backend Pytest:** 355 passed, 0 failed (including 11 new Session 15 cache-state unit tests).
- **Backend Linting:** `uv run ruff check .` passed with 0 errors.
- **Frontend Vitest:** 19 passed, 0 failed.
- **Frontend Typecheck:** `tsc --noEmit` passed with 0 errors.
- **Frontend Build:** Production build generated cleanly in 1.16s.
