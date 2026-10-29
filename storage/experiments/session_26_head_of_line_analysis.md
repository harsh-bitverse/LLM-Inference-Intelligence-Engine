# Session 26 — Head-of-Line (HoL) Blocking Analysis Report

**Serving Runtime:** Ollama v0.40.11 (Single-Slot Server Queueing Architecture)  

---

## 1. Executive Summary: The Physics of Head-of-Line Blocking

In a single-slot inference server without preemption, requests are served strictly First-Come, First-Served (FCFS). When a short interactive request (isolated duration ~113ms, SLA 500ms) arrives while a long generation request (duration ~7,500ms) is executing, the short request is blocked in the server queue for the entire remaining duration of the long generation.

---

## 2. Empirical HoL Measurements

| Experiment ID | Antecedent Workload | Consequent Workload | Antecedent Duration | Isolated Consequent Duration | Blocked Consequent Duration | Additional Wait | SLA Missed Due to HoL? |
|---|---|---|---|---|---|---|---|
| `hol-probe-01-single-behind-long` | `LONG_FORM_GENERATION` | `INTERACTIVE_QA` | 7576.1 ms | 109.6 ms | 7647.3 ms | +7537.7 ms | **YES (SLA Breached)** |

---

## 3. Serving Setup Limitations & Architectural Conclusion

1. **No Request Preemption:** The current Ollama v0.40.11 runtime does not support preempting active autoregressive generation.
2. **No Priority Queues:** Short requests cannot bypass long generation requests ahead of them in queue.
3. **Solution via Adaptive Admission:** Since runtime preemption does not exist, IIE's **Policy C (Workload-Aware Admission)** intercepts incoming short requests when queue backlog exceeds their SLA, deferring them or shedding load rather than allowing them to wait and fail.