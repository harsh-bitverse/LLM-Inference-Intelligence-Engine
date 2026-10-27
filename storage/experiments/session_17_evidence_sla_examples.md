# Session 17 — Evidence Classification & SLA Explanation Reference Examples

**Project:** LLM Inference Intelligence Engine (IIE)  
**Session:** 17 — Workload-Aware Recommendations & Product Validation  
**Date:** 2026-10-27  

---

## 1. Evidence Hierarchy & Scientific Taxonomy

IIE enforces an explicit 5-tier evidence classification taxonomy:

```
[Tier 1] DIRECTLY_MEASURED
         Exact match on model, quantization, prompt length, output length, concurrency, and residency.
         Telemetry derived from real monotonic timer counters and verified server payloads.

[Tier 2] HELD_OUT_VALIDATED_PREDICTION
         In-distribution workload ($P \in [42, 677]$, $O \in [16, 128]$) evaluated by
         WorkloadAwareLatencyPredictor with proven LOCO/LOPO cross-validation bounds.

[Tier 3] EXPLORATORY_PREDICTION
         Workload with features outside empirical bounds ($P > 677$ or $O > 128$).
         Evaluated with widened uncertainty intervals and explicit extrapolation callouts.

[Tier 4] ANALYTICAL_ESTIMATE
         Theoretical roofline ceiling or parameter-based memory footprint without empirical fit.

[Tier 5] INSUFFICIENT_EVIDENCE
         Telemetry is unobservable or protocol guarantees are absent (e.g. token-level TTFT on SSE).
```

---

## 2. Structured Reference Examples

### Example 1 — Directly Measured Evidence (Scenario B)
```json
{
  "status": "RECOMMENDED",
  "evidence_source": "DIRECTLY_MEASURED",
  "applicable_dataset": "expansion_llama3_2_1b_v2",
  "predicted_metrics": {
    "client_e2e_duration_ms": 1544.85,
    "server_decode_throughput_tok_s": 21.66,
    "server_prefill_duration_ms": 597.10,
    "server_decode_duration_ms": 1421.30,
    "server_load_duration_ms": 0.0
  },
  "sla_assessments": {
    "latency_sla": {
      "constraint_type": "HARD",
      "target_ms": 5000.0,
      "forecast_ms": 1544.85,
      "status": "FEASIBLE",
      "headroom_ms": 3455.15
    }
  },
  "explanation_summary": "Recommended Ollama with 'llama3.2:1b' (Q8_0) on Intel Core i7-14700 CPU. Workload (677 prompt tokens, 32 output tokens) yields a forecasted client end-to-end duration of 1544.8 ms and stable autoregressive decode throughput of 21.66 tok/s. All requested SLAs are satisfied with positive feasibility margins."
}
```

### Example 2 — Held-Out-Validated Prediction (Scenario A)
```json
{
  "status": "RECOMMENDED",
  "evidence_source": "HELD_OUT_VALIDATED_PREDICTION",
  "applicable_dataset": "expansion_llama3_2_1b_v2",
  "prediction_uncertainty_interval": [421.4, 1193.2],
  "generalization_status": {
    "is_out_of_distribution": false,
    "warnings": []
  },
  "predicted_metrics": {
    "client_e2e_duration_ms": 807.27,
    "server_decode_throughput_tok_s": 22.72,
    "server_prefill_duration_ms": 96.83,
    "server_decode_duration_ms": 704.19,
    "server_load_duration_ms": 0.0
  },
  "sla_assessments": {
    "latency_sla": {
      "constraint_type": "HARD",
      "target_ms": 2500.0,
      "forecast_ms": 807.27,
      "status": "FEASIBLE",
      "headroom_ms": 1692.73
    }
  }
}
```

### Example 3 — Cold-Start-Sensitive Evidence (Scenario D)
```json
{
  "status": "RECOMMENDED",
  "evidence_source": "DIRECTLY_MEASURED",
  "cold_start_assumptions": "Cold start accounted for (+1,037.74 ms penalty).",
  "predicted_metrics": {
    "client_e2e_duration_ms": 3006.96,
    "server_load_duration_ms": 1037.74,
    "server_prefill_duration_ms": 449.28,
    "server_decode_duration_ms": 1569.12
  },
  "sla_assessments": {
    "latency_sla": {
      "constraint_type": "HARD",
      "target_ms": 4000.0,
      "forecast_ms": 3006.96,
      "status": "FEASIBLE",
      "headroom_ms": 993.04
    }
  }
}
```

### Example 4 — Strict TTFT SLA Producing Inconclusive Result (Scenario E)
```json
{
  "status": "INCONCLUSIVE",
  "evidence_source": "DIRECTLY_MEASURED",
  "sla_assessments": {
    "ttft_sla": {
      "constraint_type": "HARD",
      "target_ms": 100.0,
      "forecast_ms": null,
      "status": "UNKNOWN_FEASIBILITY",
      "limitation_note": "Ollama chunk streaming emits text byte chunks without verified discrete token timestamps. True token-level TTFT cannot be verified; client-observed TTFC is recorded instead."
    }
  },
  "explanation_summary": "Recommendation is inconclusive: Token-level TTFT SLA cannot be verified on Ollama streaming protocol (emits text chunks without discrete token timestamps).. Forecasted operational metrics: 1537.5 ms duration, 21.67 tok/s decode throughput. A live validation benchmark is required to verify real-world behavior."
}
```

### Example 5 — Infeasible Latency SLA (Tight Deadline)
```json
{
  "status": "INFEASIBLE",
  "evidence_source": "HELD_OUT_VALIDATED_PREDICTION",
  "sla_assessments": {
    "latency_sla": {
      "constraint_type": "HARD",
      "target_ms": 500.0,
      "forecast_ms": 2509.45,
      "status": "PROVEN_INFEASIBLE",
      "violation_ms": 2009.45
    }
  },
  "explanation_summary": "Configuration is infeasible for the requested SLA: Forecasted latency (2509.45 ms) exceeds requested SLA (500.0 ms) by 2009.45 ms.. The CPU-bound execution on Intel Core i7-14700 cannot satisfy the required constraint."
}
```

### Example 6 — Out-Of-Distribution Workload (Scenario F)
```json
{
  "status": "INFEASIBLE",
  "evidence_source": "EXPLORATORY_PREDICTION",
  "generalization_status": {
    "is_out_of_distribution": true,
    "warnings": [
      "Prompt tokens (4000) exceed empirical benchmark maximum (677). Latency is extrapolated linearly.",
      "Output tokens (500) exceed empirical benchmark maximum (128)."
    ],
    "extrapolated_features": ["prompt_tokens", "output_tokens"]
  },
  "ood_warnings": [
    "Prompt tokens (4000) exceed empirical benchmark maximum (677). Latency is extrapolated linearly.",
    "Output tokens (500) exceed empirical benchmark maximum (128)."
  ],
  "predicted_metrics": {
    "client_e2e_duration_ms": 34116.25,
    "server_prefill_duration_ms": 12104.0,
    "server_decode_duration_ms": 22006.0
  },
  "sla_assessments": {
    "latency_sla": {
      "constraint_type": "HARD",
      "target_ms": 30000.0,
      "forecast_ms": 34116.25,
      "status": "PROVEN_INFEASIBLE",
      "violation_ms": 4116.25
    }
  }
}
```
