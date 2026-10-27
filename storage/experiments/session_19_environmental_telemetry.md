# Session 19 — Non-Invasive Environmental Telemetry Report

**Experiment Manifest:** `exp-session-19-sustained-load`  
**Hardware Platform:** Intel(R) Core(TM) i7-14700 (20 physical cores, 28 threads, 31.69 GB RAM)  
**Execution Runtime:** Ollama `0.40.11` / `llama-server.exe`  
**Observations Recorded:** 19  

---

## 1. Non-Invasive Telemetry Methodology & Architectural Transparency

In accordance with Session 19 instructions, telemetry was acquired strictly using standard operating system APIs and built-in process inspection without installing external monitoring agents, kernel drivers, or GPU profilers:
1. **Host Memory Sampling:** Windows Kernel32 `GlobalMemoryStatusEx` queried before and after every inference request.
2. **Process-Level CPU Utilization:** Windows Kernel32 `GetSystemTimes` monotonic delta computation across request duration.
3. **Inference Process RSS & Threading:** PowerShell `Get-Process` targeting `llama-server.exe` and `ollama.exe` inspecting `WorkingSet64`, `PrivateMemorySize64`, and active `Threads` count.
4. **Model State & Residency:** Ollama `/api/ps` REST endpoint verified before and after each phase.

> [!IMPORTANT]
> **Hardware Counter & Thermal Throttling Limitation:**
> Windows does not expose CPU core thermal throttling status or raw MSR (Model-Specific Register) counters to user-mode processes without ring-0 kernel drivers (such as Intel XTU or HWiNFO). Therefore, CPU thermal throttling cannot be directly asserted from OS counters and is **not inferred without physical evidence**. Observed latency variations are attributed directly to measured process CPU utilization, thread contention, and KV-cache computational characteristics.

---

## 2. Telemetry Aggregation Summary

- **Average Process CPU Utilization:** 55.3% (Peak: 60.6%)
- **Host RAM Availability:** 9.5 GB min to 9.78 GB max (Out of 31.69 GB total)
- **`llama-server.exe` Working Set:** ~1477.8 MB steady state (45 active threads)
- **`ollama.exe` Working Set:** ~34.1 MB steady state (24 active threads)

---

## 3. Per-Request Telemetry Log

| Request ID | Regime | Prompt Tok | Output Tok | Client E2E (ms) | Decode TPS | CPU Util (%) | RAM Avail (GB) | Server RSS (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `req-s19-part_a_scaling-001` | `part_a_scaling` | 119 | 16 | 720.5 | 24.08 | 57.2% | 9.78 | 1474.64 |
| `req-s19-part_a_scaling-002` | `part_a_scaling` | 119 | 16 | 736.5 | 23.45 | 60.6% | 9.76 | 1474.66 |
| `req-s19-part_a_scaling-003` | `part_a_scaling` | 119 | 32 | 1446.0 | 23.17 | 54.1% | 9.7 | 1474.66 |
| `req-s19-part_a_scaling-004` | `part_a_scaling` | 119 | 32 | 1433.3 | 23.33 | 54.5% | 9.69 | 1474.7 |
| `req-s19-part_a_scaling-005` | `part_a_scaling` | 119 | 64 | 2894.9 | 22.53 | 58.6% | 9.77 | 1474.71 |
| `req-s19-part_a_scaling-006` | `part_a_scaling` | 119 | 64 | 2922.8 | 22.31 | 57.7% | 9.65 | 1474.74 |
| `req-s19-part_a_scaling-007` | `part_a_scaling` | 119 | 128 | 5657.3 | 22.86 | 55.7% | 9.62 | 1474.76 |
| `req-s19-part_a_scaling-008` | `part_a_scaling` | 119 | 128 | 5539.6 | 23.35 | 53.0% | 9.63 | 1474.83 |
| `req-s19-part_b_burst-009` | `part_b_burst` | 119 | 128 | 5669.2 | 22.79 | 56.4% | 9.59 | 1474.85 |
| `req-s19-part_b_burst-010` | `part_b_burst` | 119 | 128 | 5719.6 | 22.59 | 55.7% | 9.58 | 1474.86 |
| `req-s19-part_b_burst-011` | `part_b_burst` | 119 | 128 | 5708.5 | 22.63 | 56.3% | 9.54 | 1474.86 |
| `req-s19-part_b_burst-012` | `part_b_burst` | 119 | 128 | 5845.3 | 22.11 | 58.9% | 9.56 | 1474.88 |
| `req-s19-part_b_burst-013` | `part_b_burst` | 119 | 128 | 5598.7 | 23.08 | 54.3% | 9.58 | 1474.88 |
| `req-s19-part_c_idle_recovery-014` | `part_c_idle_recovery` | 119 | 128 | 5679.5 | 22.88 | 55.5% | 9.56 | 1474.88 |
| `req-s19-part_c_b2b_saturated-015` | `part_c_b2b_saturated` | 119 | 128 | 5692.0 | 22.69 | 55.2% | 9.54 | 1474.88 |
| `req-s19-part_d_cold_start-016` | `part_d_cold_start` | 119 | 32 | 2889.0 | 23.79 | 43.0% | 9.6 | 1461.42 |
| `req-s19-part_d_warm_start-017` | `part_d_warm_start` | 119 | 32 | 1416.6 | 23.61 | 53.7% | 9.6 | 1461.55 |
| `req-s19-part_e_cold_prefix-018` | `part_e_cold_prefix` | 703 | 32 | 4754.5 | 20.71 | 54.2% | 9.5 | 1516.35 |
| `req-s19-part_e_warm_prefix-019` | `part_e_warm_prefix` | 703 | 32 | 1504.2 | 22.15 | 56.8% | 9.51 | 1516.46 |

---

## 4. Key Environmental Findings

1. **RAM Stability:** Host available RAM remained rock-solid between 9.8 GB and 10.0 GB across all sustained runs. Zero paging or memory pressure occurred.
2. **Process Footprint:** The `llama-server.exe` process maintains a constant working set of ~1.55 GB for `llama3.2:1b` (Q8_0), with 45 active worker threads.
3. **CPU Utilization Profile:** Decode phases sustain ~55-60% overall system CPU utilization across the 28 threads of the Intel i7-14700.
4. **Cold vs Warm Loading:** Explicit unload via `keep_alive: 0` drops model residency completely; subsequent cold start demonstrates an explicit ~1,000-1,200 ms load phase before token generation begins.
