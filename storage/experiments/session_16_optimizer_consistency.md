# Session 16 — Optimizer Evidence-Consistency & Objective Routing Report

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 16 — Optimizer Contract Verification  
**Artifact ID:** `report-session-16-optimizer-consistency`  
**Date:** 2026-10-27  

---

## 1. Executive Summary

This report audits the configuration optimization engine (`llm_ii_engine.optimizer`) to guarantee that all candidate evaluations, Pareto dominance calculations, and constraint checks strictly align with the Session 16 Authoritative Measurement Contract.

---

## 2. Objective Mapping & Semantic Separation

| Optimizer Objective Enum | Target Metric Field | Measurement Domain | Unit | Direction | Physical Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MAXIMIZE_THROUGHPUT` | `client_e2e_output_throughput_tok_s` | Client Application | `tok/s` | Maximize | Effective output delivery rate including cold load, prompt evaluation, and streaming transport. |
| `MAXIMIZE_DECODE_THROUGHPUT` | `server_decode_throughput_tok_s` | Inference Hardware | `tok/s` | Maximize | Autoregressive token decoding speed of the hardware/kernel. |
| `MINIMIZE_DURATION` / `MINIMIZE_CLIENT_DURATION` | `client_e2e_duration_ms` | Client Application | `ms` | Minimize | Total elapsed generation time experienced by user. |
| `MINIMIZE_TTFT` | `ttft_ms` | Client / Stream | `ms` | Minimize | Time to first true token (disallows text chunks). |
| `MINIMIZE_TPOT` | `tpot_ms` | Client / Stream | `ms` | Minimize | Time per output token across discrete events. |
| `MINIMIZE_VRAM` | `peak_vram_mb` | Hardware Host | `MiB` | Minimize | Memory allocation overhead. |
| `MINIMIZE_COST` | Derived from Throughput & Hourly Rate | Financial | `$/1M tok` | Minimize | Cost per million output tokens. |

---

## 3. Evidence Consistency Guardrails

1. **Streaming Text Chunk Audit**:
   - If an observation provides `ttfc_ms` from an HTTP chunk stream without discrete token boundaries, the optimizer **does not fabricate** `ttft_ms`.
   - If the user specifies `max_ttft_ms` constraint on a chunk stream, the optimizer registers:
     > *"Matched real benchmark observation has measured TTFC, but token-level TTFT is unmeasured in streaming chunk adapter; TTFT SLA cannot be verified."*
2. **Cold-Start Feasibility Guardrail**:
   - When `constraints.require_cold_start_feasibility = True`:
     - If the matched observation has `model_residency_state == WARM` or `cache_state == WARM_PREFIX`, the evidence status is downgraded to `INSUFFICIENT_EVIDENCE`.
     - The optimizer flags:
       > *"Matched real observation is from warm-residency/warm-prefix execution; insufficient evidence for cold-start TTFT/duration SLA."*
3. **Disparate Evidence Tier Warning**:
   - If a candidate's objectives mix `MEASURED_EVALUATED` (e.g. from Ollama benchmark) with `THEORETICAL_ANALYTICAL` (e.g. from Roofline ceiling), `has_mixed_evidence` is set to `True`.
   - A prominent disclaimer is returned in the API and rendered in the dashboard.

---

## 4. Pareto Dominance Invariants

In `llm_ii_engine/optimizer/pareto.py`:
1. `OBJECTIVE_DIRECTIONS` explicitly assigns `ObjectiveDirection.MAXIMIZE` to both `MAXIMIZE_THROUGHPUT` and `MAXIMIZE_DECODE_THROUGHPUT`.
2. `MINIMIZE_DURATION` and `MINIMIZE_CLIENT_DURATION` are assigned `ObjectiveDirection.MINIMIZE`.
3. Incomparable candidates with missing metrics are excluded from the non-dominated set, preventing false Pareto claims.
