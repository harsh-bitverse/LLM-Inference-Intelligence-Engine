# Session 19 — Prospective Reliability Validation Report

**Pre-Execution Timestamp (UTC):** `2026-10-27T18:39:28.370805+00:00`  
**Post-Execution Timestamp (UTC):** `2026-10-27T18:40:09.678048+00:00`  
**Host Platform:** Intel(R) Core(TM) i7-14700 CPU (20 cores, 28 threads, 31.69 GB RAM)  
**Runtime & Model:** Ollama `0.40.11` with `llama3.2:1b` (Q8_0, GGUF)  

---

## 1. Executive Summary & Prospective Integrity

In strict compliance with empirical science standards:
1. All prospective performance forecasts were generated and serialized to disk with UTC timestamps **prior to initiating live inference**.
2. The test suite encompassed all 4 core operating regimes (`WARM_SHORT_BURST`, `WARM_SUSTAINED_GENERATION`, `PROMPT_PREFIX_REUSE`, `COLD_START`).
3. Three discrete SLA thresholds (Generous, Tight, Unachievable) were prospective evaluated per scenario.
4. Live benchmark repetitions (3 per scenario) were executed on the local workstation to test prospective assertions.

---

## 2. Prospective Forecasts vs. Live Measured Latency

| Scenario | Operating Regime | Nominal Forecast (ms) | Worst-Case Bound (ms) | Observed Median (ms) | Observed Worst (ms) | Within Regime Bound? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SCENARIO_1_SHORT_BURST` | `WARM_SHORT_BURST` | 849.64 | 1147.0 | 761.62 | 1885.5 | NO |
| `SCENARIO_2_SUSTAINED_GEN` | `WARM_SUSTAINED_GENERATION` | 5657.3125 | 10242.7 | 5694.84 | 5880.91 | **YES** |
| `SCENARIO_3_PREFIX_REUSE` | `PROMPT_PREFIX_REUSE` | 1544.851 | 2008.3 | 1484.31 | 1514.74 | NO |
| `SCENARIO_4_COLD_START` | `COLD_START` | 3006.964 | 3237.7 | 2965.84 | 2996.31 | NO |

- **Session 19 Regime Interval Coverage:** 25.0%
- **Session 18 Mean Absolute Error:** 56.8 ms
- **Session 19 Mean Interval Residual:** 47.42 ms

---

## 3. Repetition Breakdown per Scenario

### `SCENARIO_1_SHORT_BURST` — Short Output Workload (O=16)

| Repetition | Client E2E (ms) | Decode Duration (ms) | Prefill Duration (ms) | Load Duration (ms) | Decode TPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Rep 1 | 1885.5 | 668.91 | 169.26 | 1040.34 | 23.92 |
| Rep 2 | 750.46 | 693.69 | 49.47 | 2.77 | 23.07 |
| Rep 3 | 761.62 | 706.53 | 48.38 | 2.23 | 22.65 |

### `SCENARIO_2_SUSTAINED_GEN` — Long Output Workload (O=128) — Critical SLA Target

| Repetition | Client E2E (ms) | Decode Duration (ms) | Prefill Duration (ms) | Load Duration (ms) | Decode TPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Rep 1 | 5880.91 | 5588.22 | 285.66 | 2.63 | 22.91 |
| Rep 2 | 5694.84 | 5634.9 | 47.0 | 5.27 | 22.72 |
| Rep 3 | 5595.89 | 5539.3 | 46.92 | 2.07 | 23.11 |

### `SCENARIO_3_PREFIX_REUSE` — Long Prompt Warm Prefix Replay (P=677, O=32)

| Repetition | Client E2E (ms) | Decode Duration (ms) | Prefill Duration (ms) | Load Duration (ms) | Decode TPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Rep 1 | 1514.74 | 1453.82 | 49.23 | 3.42 | 22.01 |
| Rep 2 | 1484.31 | 1424.68 | 48.34 | 2.65 | 22.46 |
| Rep 3 | 1483.82 | 1425.07 | 47.5 | 2.39 | 22.46 |

### `SCENARIO_4_COLD_START` — Cold Start Disk Load Workload (O=32)

| Repetition | Client E2E (ms) | Decode Duration (ms) | Prefill Duration (ms) | Load Duration (ms) | Decode TPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Rep 1 | 2835.33 | 1346.52 | 443.11 | 1041.48 | 23.76 |
| Rep 2 | 2965.84 | 1470.34 | 450.74 | 1039.97 | 21.76 |
| Rep 3 | 2996.31 | 1484.8 | 466.12 | 1038.96 | 21.55 |
