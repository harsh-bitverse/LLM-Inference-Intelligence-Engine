# Session 20 — Quality-Aware Recommendation Contract

## 1. Scientific Principles & Design

A pure performance-focused recommendation system may recommend a model solely because it achieves high decode throughput or low latency, even if its outputs are malformed, syntactically invalid, or factually wrong.

Conversely, treating unmeasured model quality as "known" or assuming that benchmark scores from external leaderboards (e.g. HuggingFace Open LLM Leaderboard) apply identically to quantized local CPU executions is scientifically unsound.

### The IIE Quality Contract enforces four rules:
1. **Separation of Concerns:** Performance latency/throughput SLAs and functional quality constraints are evaluated along distinct, transparent dimensions.
2. **Preservation of Unknown Quality:** If a user specifies a quality constraint (e.g. `require_json_compliance=True` or `min_code_pass_rate=0.8`), and IIE has no empirical test suite records for the target model, the system MUST NOT guess or assume compliance. It must report `quality_status = UNKNOWN_QUALITY_NO_EVALUATION` and `conservative_feasibility = UNKNOWN_FEASIBILITY`.
3. **No Unjustified Metric Compression:** IIE strictly prohibits computing a single arbitrary "magic score" (e.g. $S = 0.4 \times \text{speed} + 0.6 \times \text{quality}$). Aggregating incommensurable units into an uncalibrated composite score obscures whether an SLA miss or a quality failure occurred.
4. **Physical Compatibility Gates:** Before evaluating performance or quality, the engine evaluates `ModelCompatibilityStatus`:
   - `COMPATIBLE`: Model is installed and supported on the host workstation.
   - `UNINSTALLED`: Model exists in the catalog but is not installed locally.
   - `INSUFFICIENT_MEMORY`: Model footprint exceeds available RAM or user-specified memory budget.
   - `INCOMPATIBLE_RUNTIME`: Serving engine does not support model format or architecture.

---

## 2. Quality Constraints Specification (`ModelQualityConstraintSpec`)

Users can declare explicit constraints in their recommendation requests:

| Field Name | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `require_json_compliance` | `bool` | Requires valid JSON parseability for extraction tasks. | `False` |
| `require_code_pass` | `bool` | Requires passing sandboxed unit test assertions ($\ge 80\%$). | `False` |
| `min_task_correctness` | `float \| None` | Minimum overall correctness across evaluated tasks ($0.0 \le s \le 1.0$). | `None` |
| `min_json_compliance_rate` | `float \| None` | Minimum percentage of valid JSON outputs. | `None` |
| `min_code_pass_rate` | `float \| None` | Minimum code execution pass rate. | `None` |
| `min_reasoning_accuracy` | `float \| None` | Minimum accuracy on numerical reasoning tasks. | `None` |
| `max_memory_budget_mb` | `float \| None` | Hard upper bound on process memory footprint (MiB). | `None` |
| `preferred_model_family` | `str \| None` | Architectural filter (e.g. `llama`, `qwen2`). | `None` |

---

## 3. Decision Matrix & Feasibility Hierarchy

When quality constraints are active, the overall recommendation outcome follows a strict priority cascade:

```
[Incompatible / Uninstalled]
   │
   ├─► Model not locally installed ──► Status: UNINSTALLED, Feasibility: UNKNOWN_FEASIBILITY
   ├─► Exceeds memory budget ─────────► Status: INSUFFICIENT_MEMORY, Feasibility: PROVEN_INFEASIBLE
   │
[Installed Model]
   │
   ├─► Quality evidence missing ──────► Quality: UNKNOWN_QUALITY, Feasibility: UNKNOWN_FEASIBILITY
   ├─► Quality constraint violated ──► Quality: QUALITY_VIOLATED, Feasibility: QUALITY_INFEASIBLE
   │
   └─► Quality satisfied ─────────────► Evaluates Performance SLA:
                                        ├─► Latency <= Nominal & Worst-Case <= SLA ──► ROBUST_FEASIBLE
                                        ├─► Latency <= Nominal but Worst-Case > SLA ─► NOMINAL_FEASIBLE_ROBUST_RISK
                                        └─► Latency > SLA ───────────────────────────► PREDICTED_SLA_MISS / PROVEN_INFEASIBLE
```

This guarantees that an unverified model cannot be recommended as meeting quality criteria, protecting the user from silent functional degradation.
