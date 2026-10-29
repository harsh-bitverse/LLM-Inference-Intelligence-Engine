# Session 26 — Mixed-Workload Queueing Model Validation Report

**Date:** 2026-10-29T17:40:47.536937+00:00  
**Hardware:** cpu-intel-i7-14700 (Intel Core i7-14700, 28 threads, Windows 11)  
**Serving Architecture:** Ollama v0.40.11 (Single-Slot Server Queueing Architecture)  

---

## 1. Mixed Service Moments Formulation

For heterogeneous workload mixtures comprising classes $i \in \{1, \dots, K\}$ with arrival fractions $p_i$:
$$E[S_{\text{mix}}] = \sum_{i=1}^K p_i E[S_i]$$
$$E[S_{\text{mix}}^2] = \sum_{i=1}^K p_i E[S_i^2] = \sum_{i=1}^K p_i \left( \text{Var}[S_i] + (E[S_i])^2 \right)$$
$$\text{Var}[S_{\text{mix}}] = E[S_{\text{mix}}^2] - (E[S_{\text{mix}}])^2, \quad C_{v,\text{mix}} = \frac{\sqrt{\text{Var}[S_{\text{mix}}]}}{E[S_{\text{mix}}]}$$

### Profile Analytical Moments (Evaluated on `llama3.2:1b` calibrated distributions):

| Profile Type | Description | $E[S_{\text{mix}}]$ (ms) | $E[S_{\text{mix}}^2]$ ($10^6 \text{ ms}^2$) | $\text{Std}[S_{\text{mix}}]$ (ms) | $C_{v,\text{mix}}$ | Max Sustainable Throughput ($\mu$) |
|---|---|---|---|---|---|---|
| `PROFILE_A_INTERACTIVE_DOMINANT` | Profile A: Interactive Dominant | 766.6 | 3.44 | 1688.8 | 2.203 | 1.30 req/s |
| `PROFILE_B_BALANCED` | Profile B: Balanced Mixture | 2160.3 | 12.52 | 2802.3 | 1.297 | 0.46 req/s |
| `PROFILE_C_GENERATION_HEAVY` | Profile C: Generation Heavy | 4380.6 | 29.48 | 3207.7 | 0.732 | 0.23 req/s |
| `PROFILE_D_BURSTY` | Profile D: Bursty Traffic Mixture | 1963.2 | 14.14 | 3207.9 | 1.634 | 0.51 req/s |

---

## 2. Discrete-Event Simulator vs. Empirical Live Measurements

Because mean queue delay $W_q$ does not capture per-class tail latencies under Head-of-Line blocking, IIE uses a bounded deterministic discrete-event simulator to estimate per-class P95 latencies and SLA attainment.

| Profile Type | Offered Rate | Simulated P50 E2E | Simulated P95 E2E | Live Measured P50 E2E | Live Measured P95 E2E | Tail Estimation Delta |
|---|---|---|---|---|---|---|
| `PROFILE_A_INTERACTIVE_DOMINANT` | 1.20 req/s | 3628.8 ms | 16762.0 ms | 131.4 ms | 2266.5 ms | 639.6% error |
| `PROFILE_B_BALANCED` | 0.80 req/s | 37901.8 ms | 82608.2 ms | 2418.7 ms | 9470.0 ms | 772.3% error |
| `PROFILE_C_GENERATION_HEAVY` | 0.40 req/s | 85257.7 ms | 183740.7 ms | 8514.5 ms | 20022.9 ms | 817.7% error |

---

## 3. Scientific Verification & Assumptions
- **Single-Server FCFS:** Ollama executes requests in strict arrival sequence without preemption.
- **Variance Amplification:** When long generation requests (7,500ms) mix with short requests (113ms), $C_v$ jumps from ~0.03 to >1.5, causing non-linear queue divergence even at moderate utilization $\rho = 0.50$.