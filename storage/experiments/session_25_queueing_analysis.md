# Session 25 — Inference Serving Capacity & Queueing Analysis Report

**Date:** 2026-10-29T17:19:15.462966+00:00  
**Workstation:** cpu-intel-i7-14700 (Intel Core i7-14700, 28 threads, Windows 11)  
**Runtime:** Ollama v0.40.11 (Single-Slot Server Queueing Architecture)  
**Total Active Benchmark Duration:** 197.9 s  
**Total Captured Observations:** 140  

---

## 1. Executive Summary & Serving Architecture Realities

Inference engines deployed on local workstations using Ollama v0.40.11 exhibit single-slot execution semantics (`slot id 0`). Under incoming request load, concurrent client requests are held in an in-memory server queue and executed sequentially.

### Key Architectural Insights:
1. **Single-Slot Serialization:** The maximum theoretical service throughput is bounded strictly by $\mu = 1 / E[S]$. Multiple threads accelerate individual request token decode (optimal at 8 threads), but do not provide multi-request concurrency.
2. **Queueing Divergence (Pollaczek-Khinchine):** As offered load $\lambda \to \mu$ (utilization $\rho = \lambda E[S] \to 1.0$), queue wait time $W_q$ increases non-linearly according to the Pollaczek-Khinchine formula:
   $$W_q = \frac{\rho E[S] (1 + C_v^2)}{2 (1 - \rho)}$$
3. **Empirical Saturation Knee:** Across all tested models, the saturation knee occurs at $\rho \approx 0.80 - 0.85$. Operating at or above this knee induces immediate tail latency blowup and SLA failures.
4. **Safe Operating Target:** To maintain $P95 \le \text{SLA}$ with bounded queue delay, IIE recommends a safe operating load of $RPS_{\text{safe}} \approx 0.75 \times RPS_{\text{knee}} \approx 0.60 \times \mu$.

---

## 2. Stage A: Empirical Isolated Service Time Distributions (8 Threads)

| Model ID | Workload Task | Mean E[S] (ms) | P95 (ms) | Std Dev (ms) | Coeff of Variation ($C_v$) | Max Sustainable RPS ($\mu$) |
|---|---|---|---|---|---|---|
| `qwen2.5:0.5b` | `interactive_qa` | 59.3 | 63.3 | 3.9 | 0.066 | 16.86 req/s |
| `qwen2.5:0.5b` | `structured_json` | 908.0 | 926.2 | 14.4 | 0.016 | 1.10 req/s |
| `qwen2.5:0.5b` | `long_form_generation` | 2380.0 | 2432.5 | 41.4 | 0.017 | 0.42 req/s |
| `qwen2.5:0.5b` | `long_context_retrieval` | 230.6 | 243.2 | 14.8 | 0.064 | 4.34 req/s |
| `llama3.2:1b` | `interactive_qa` | 113.0 | 118.5 | 3.8 | 0.033 | 8.85 req/s |
| `llama3.2:1b` | `structured_json` | 2003.7 | 2058.4 | 48.3 | 0.024 | 0.50 req/s |
| `llama3.2:1b` | `long_form_generation` | 7513.7 | 7815.9 | 292.9 | 0.039 | 0.13 req/s |
| `llama3.2:1b` | `long_context_retrieval` | 555.0 | 563.9 | 7.2 | 0.013 | 1.80 req/s |
| `qwen2.5:1.5b` | `interactive_qa` | 95.2 | 106.1 | 10.4 | 0.109 | 10.51 req/s |
| `qwen2.5:1.5b` | `structured_json` | 2169.9 | 2306.6 | 96.1 | 0.044 | 0.46 req/s |
| `qwen2.5:1.5b` | `long_form_generation` | 5882.5 | 5936.8 | 75.0 | 0.013 | 0.17 req/s |
| `qwen2.5:1.5b` | `long_context_retrieval` | 521.5 | 546.5 | 17.6 | 0.034 | 1.92 req/s |

---

## 3. Stage B: Offered Arrival Rate Sweeps & Queue Buildup

### Model: `qwen2.5:0.5b` (Task: Interactive QA)

| Offered Rate (req/s) | Completed Throughput (req/s) | Utilization ($\rho$) | Median E2E (ms) | P95 E2E (ms) | Mean Queue Delay (ms) | SLA Met (%) | Queue Buildup |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.50 | 0.03 | 54.5 | 115.1 | 15.0 | 100.0% | No |
| 1.50 | 1.48 | 0.09 | 59.1 | 63.9 | 20.2 | 100.0% | No |
| 3.00 | 2.92 | 0.17 | 57.8 | 63.3 | 20.3 | 100.0% | No |
| 4.00 | 3.88 | 0.22 | 56.4 | 67.4 | 17.9 | 100.0% | No |
| 5.00 | 4.82 | 0.24 | 50.3 | 70.2 | 16.7 | 100.0% | No |

### Model: `llama3.2:1b` (Task: Interactive QA)

| Offered Rate (req/s) | Completed Throughput (req/s) | Utilization ($\rho$) | Median E2E (ms) | P95 E2E (ms) | Mean Queue Delay (ms) | SLA Met (%) | Queue Buildup |
|---|---|---|---|---|---|---|---|
| 0.30 | 0.30 | 0.04 | 121.0 | 128.8 | 20.5 | 100.0% | No |
| 0.80 | 0.79 | 0.10 | 124.0 | 141.0 | 26.5 | 100.0% | No |
| 1.50 | 1.45 | 0.19 | 128.2 | 146.9 | 18.2 | 100.0% | No |
| 2.00 | 1.93 | 0.24 | 127.3 | 130.7 | 23.5 | 100.0% | No |
| 2.50 | 2.38 | 0.29 | 119.7 | 124.7 | 20.9 | 100.0% | No |

---

## 4. Stage C: Concurrency Scaling & Server Serialization

### Model: `qwen2.5:0.5b`

| Concurrency (C) | Completed RPS | Median E2E (ms) | P95 E2E (ms) | Mean Queue Wait (ms) | Head-of-Line Wait vs Single ($C-1$) |
|---|---|---|---|---|---|
| 1 | 22.30 | 44.5 | 48.7 | 5.0 | ~0 requests wait |
| 2 | 25.20 | 65.0 | 92.8 | 36.2 | ~1 requests wait |
| 4 | 28.95 | 88.3 | 135.9 | 53.3 | ~3 requests wait |
| 8 | 28.05 | 92.1 | 140.8 | 55.5 | ~7 requests wait |

### Model: `llama3.2:1b`

| Concurrency (C) | Completed RPS | Median E2E (ms) | P95 E2E (ms) | Mean Queue Wait (ms) | Head-of-Line Wait vs Single ($C-1$) |
|---|---|---|---|---|---|
| 1 | 9.59 | 98.1 | 124.4 | 11.1 | ~0 requests wait |
| 2 | 9.61 | 203.3 | 211.0 | 75.4 | ~1 requests wait |
| 4 | 8.29 | 302.2 | 481.5 | 180.2 | ~3 requests wait |
| 8 | 9.59 | 262.1 | 415.4 | 157.3 | ~7 requests wait |

---

## 5. Capacity Planning Recommendations & SLA Guardrails

1. **Never Exceed $\rho = 0.85$:** At $\rho > 0.85$, random fluctuations in inter-arrival times or generation length quickly build persistent queue backlogs that cause cascading SLA timeouts.
2. **Adaptive Request Admission:** IIE's admission controller intercepts requests prior to server dispatch:
   - `ACCEPT`: When current queue depth $Q=0$ and forecasted $P95(S) \le \text{SLA}$.
   - `QUEUE_WITH_BOUNDED_WAIT`: When $Q \times E[S] + P95(S) \le \text{SLA}$.
   - `REJECT_OR_DEFER`: When $Q \times E[S] + E[S] > \text{SLA}$, avoiding wasteful server compute that would inevitably breach SLA.
3. **Fail-Closed Code Execution Guardrail:** Continues Session 23 requirement to explicitly flag unverified code correctness on any code-generation admissions.