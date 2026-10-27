# Session 19 — SLA Failure Latency Phase Decomposition

**Experiment ID:** `exp-session19-sla-failure-reproduction`  
**Execution Timestamp (UTC):** `2026-10-27T18:27:44.114613+00:00`  
**Target Workload:** `VAL_WORKLOAD_LONG` (Prompt Tokens: 119, Requested Output: 128, Concurrency: 1)  
**Host Workstation:** Intel(R) Core(TM) i7-14700 (20 physical cores, 28 threads, 31.69 GB RAM)  
**Inference Runtime:** Ollama `0.40.11` / `llama-server.exe` (PID 251964) with `llama3.2:1b` (Q8_0, 1.2B params, GGUF)

---

## 1. Executive Summary & Root Cause Analysis

In Session 18, prospective validation for `VAL_WORKLOAD_LONG` ($P=119, O=128$, SLA $\le 8,000\text{ ms}$) produced a critical SLA violation:
- **Forecast E2E Latency:** $5,999.88\text{ ms}$ (Forecast decode throughput: $22.72\text{ tok/s}$)
- **Session 18 Observed Median:** $10,282.98\text{ ms}$ (Observed decode throughput: $12.58\text{ tok/s}$)
- **SLA Target:** $8,000.0\text{ ms}$
- **Session 18 Margin of Violation:** $+2,282.98\text{ ms}$ ($+28.5\%$ past SLA; $+71.4\%$ prediction error).

### Phase-by-Phase Latency Breakdown (Session 18 vs Isolated Reproduction)

| Latency Phase | Session 18 Median (ms) | Session 18 Share (%) | Session 19 Isolated Median (ms) | Session 19 Share (%) | Delta (ms) | Attribution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Load (`load_duration`)** | $5.40\text{ ms}$ | $0.05\%$ | $2.78\text{ ms}$ | $0.05\%$ | $-2.62\text{ ms}$ | Zero contribution to SLA failure (model resident in RAM). |
| **Prompt Prefill (`prompt_eval_duration`)** | $88.19\text{ ms}$ | $0.86\%$ | $47.29\text{ ms}$ | $0.82\%$ | $-40.90\text{ ms}$ | Fast prefill ($0.4\text{--}0.7\text{ ms/tok}$); faster than model's $360\text{ ms}$ forecast. |
| **Autoregressive Decode (`eval_duration`)** | **$10,174.39\text{ ms}$** | **$98.94\%$** | **$5,619.71\text{ ms}$** | **$97.82\%$** | **$-4,554.68\text{ ms}$** | **$100\%$ of SLA failure is decode throughput decay.** |
| **Other Server Overhead** | $14.99\text{ ms}$ | $0.15\%$ | $4.89\text{ ms}$ | $0.09\%$ | $-10.10\text{ ms}$ | Negligible internal Go/C++ IPC overhead. |
| **Client/Server Divergence** | $0.01\text{ ms}$ | $<0.01\%$ | $2.21\text{ ms}$ | $0.04\%$ | $+2.20\text{ ms}$ | Loopback HTTP transport is near-zero overhead. |
| **Total Client E2E Duration** | **$10,282.98\text{ ms}$** | **$100.0\%$** | **$5,744.93\text{ ms}$** | **$100.0\%$** | **$-4,538.05\text{ ms}$** | **$44.1\%$ latency reduction in isolated state.** |

---

## 2. Definitive Findings: Why Did the Forecast Miss in Session 18?

1. **Failure Is 100% Decode Phase:** Neither model loading, prompt prefill, nor HTTP transport contributed to the SLA miss. In Session 18, autoregressive token generation accounted for $10,174.39\text{ ms}$ ($98.94\%$) of total execution time.
2. **Decode Throughput Instability Under Sustained Load:**
   - In Session 18, `VAL_WORKLOAD_LONG` ran immediately following `VAL_WORKLOAD_SHORT` and consecutive multi-token workloads. Decode throughput degraded from $22.7\text{ tok/s}$ down to $12.58\text{ tok/s}$ ($79.5\text{ ms/token}$ vs $44.0\text{ ms/token}$).
   - In the isolated Session 19 test (executed from an idle state), the exact same workload achieved $22.78\text{ tok/s}$ ($43.9\text{ ms/token}$), completing in $5,744.93\text{ ms}$ and satisfying the $8,000\text{ ms}$ SLA.
3. **The Static Predictor Flaw:** The Session 18 predictor assumed an invariant decode throughput of $22.72\text{ tok/s}$ across all workloads and load regimes. It failed to account for sustained-load throughput degradation, where continuous generation or consecutive requests drop throughput by up to $45\%$.
4. **Conclusion:** An SLA policy relying solely on nominal/isolated decode throughput ($22.72\text{ tok/s}$) is structurally unsafe. A conservative SLA policy must condition its predictions on the **operating regime** and evaluate against worst-case sustained-generation decode rates ($12.5\text{--}14.5\text{ tok/s}$).

---

## 3. Repetition Details (Session 19 Isolated Run)

- **Repetition 1:** Client E2E: $5,939.65\text{ ms}$ | Decode: $5,589.28\text{ ms}$ ($22.90\text{ tok/s}$) | Prefill: $344.81\text{ ms}$ | Load: $1.56\text{ ms}$ | CPU Util: $55.1\%$ | SLA Violated: False
- **Repetition 2:** Client E2E: $5,744.93\text{ ms}$ | Decode: $5,687.70\text{ ms}$ ($22.50\text{ tok/s}$) | Prefill: $47.29\text{ ms}$ | Load: $2.78\text{ ms}$ | CPU Util: $57.9\%$ | SLA Violated: False
- **Repetition 3:** Client E2E: $5,679.44\text{ ms}$ | Decode: $5,619.71\text{ ms}$ ($22.78\text{ tok/s}$) | Prefill: $47.29\text{ ms}$ | Load: $5.27\text{ ms}$ | CPU Util: $57.6\%$ | SLA Violated: False

**Environmental Telemetry:**
- Host Memory: $9.95\text{ GB}$ available / $31.69\text{ GB}$ total ($68.0\%$ memory load).
- `llama-server.exe` (PID 251964): WorkingSet64: $1,546,194,944\text{ bytes}$ ($\approx 1.55\text{ GB}$), PrivateMemory: $1,627,303,936\text{ bytes}$, 45 active threads.
- `ollama.exe` (PID 192396): WorkingSet64: $34,082,816\text{ bytes}$ ($\approx 34.1\text{ MB}$), 24 active threads.
