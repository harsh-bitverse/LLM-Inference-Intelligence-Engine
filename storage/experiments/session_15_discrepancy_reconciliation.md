# Session 15 — Baseline Integrity & Historical Discrepancy Reconciliation

## Executive Summary

As required by Session 15, an exhaustive audit of all historical Session 12B, Session 13, and Session 14 manifests, datasets (JSONL/CSV), SQLite database records (`llm_ii_engine.db`), and generated reports was conducted to reconcile two historical baseline discrepancies:

1. **Ollama Version Discrepancy:** `0.40.11` versus `0.5.1`.
2. **P677_O64 Duration Discrepancy:** Approximately $2.86\text{ s}$ versus approximately $5.0\text{ s}$.

Both discrepancies have been rigorously investigated against immutable raw telemetry, binary inspection, and runtime API queries. No historical observations were deleted or mutated. This document serves as the authoritative, permanent reconciliation record.

---

## 1. Discrepancy 1: Ollama Version `0.40.11` vs `0.5.1`

### 1.1 Reported Inconsistency
- In `storage/experiments/pilot_llama3_2_1b_manifest.json`, `storage/experiments/expansion_llama3_2_1b_manifest.json`, and all exported JSONL rows, `runtime_version` is recorded as `"0.40.11"`.
- In `storage/datasets/session_14_robustness_report.md` (line 12), the narrative text stated: *"Ollama v0.5.1 serving llama3.2:1b"*.

### 1.2 Physical and Telemetry Evidence
1. **Installed Binary Query:**
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version
   # Output: ollama version is 0.40.11
   ```
2. **Local Runtime API Endpoint Query:**
   ```http
   GET http://127.0.0.1:11434/api/version
   HTTP/1.1 200 OK
   {"version":"0.40.11"}
   ```
3. **Installer Archive Check:**
   - The installer `C:\Workspace\downloads\OllamaSetup.exe` (SHA256: `87309d23b423abeae1d8df2f98df95f94bf35659debfdf3761d65b7b62a75508`) corresponds to the official release of Ollama `0.40.11`.
4. **Historical Database & Raw JSONL Records:**
   - In SQLite table `benchmark_suites`, every suite result records `0.40.11`.
   - In `storage/datasets/pilot_llama3_2_1b.jsonl`, all 9 records contain `"runtime_version": "0.40.11"`.
   - In `storage/datasets/expansion_llama3_2_1b.jsonl`, all 21 measured records contain `"runtime_version": "0.40.11"`.

### 1.3 Reconciliation Finding & Root Cause
- The actual, true, and consistently executed runtime version on this workstation has always been **`0.40.11`**.
- The string `"0.5.1"` in `storage/datasets/session_14_robustness_report.md` was an errant typographical error in the markdown prose summary.
- **Authoritative Resolution:** The authoritative runtime version for all experiments (Session 12B, 13, 14, and 15) is **`0.40.11`**.

---

## 2. Discrepancy 2: P677_O64 Duration ($\approx 2.86\text{ s}$ vs $\approx 5.0\text{ s}$)

### 2.1 Reported Inconsistency
- In `storage/experiments/expansion_llama3_2_1b_validation_report.md` (Table 1, Workload 5: P768/O64), the reported total duration was recorded as $2,864.12\text{ ms}$ with a reported throughput of $22.35\text{ tok/s}$.
- In `storage/datasets/expansion_llama3_2_1b.jsonl`, `storage/experiments/expansion_llama3_2_1b_repeatability.json`, and SQLite table `request_benchmarks`, the actual end-to-end client duration is recorded as approximately $5,000\text{ ms}$ ($\approx 5.0\text{ s}$) with client throughput of $12.80\text{ tok/s}$.

### 2.2 Forensic Inspection of Raw Telemetry Records
Inspecting the exact database and JSONL records for Workload 5 (`P677_O64`):

| Repetition | Request ID | Client `duration_ms` | Server Prefill (`prompt_eval`) | Server Decode (`eval_duration`) | Total Server Time | Client Throughput | Server Decode TPS |
|---|---|---|---|---|---|---|---|
| Warmup | `warmup-0005` | $5,000.20\text{ ms}$ | $2,033.45\text{ ms}$ | $2,796.75\text{ ms}$ | $4,830.20\text{ ms}$ | $12.80\text{ tok/s}$ | $22.88\text{ tok/s}$ |
| Repetition 1 | `req-0012` | $5,050.59\text{ ms}$ | $2,052.95\text{ ms}$ | $2,878.88\text{ ms}$ | $4,931.83\text{ ms}$ | $12.67\text{ tok/s}$ | $22.23\text{ tok/s}$ |
| Repetition 2 | `req-0019` | $4,979.52\text{ ms}$ | $2,020.61\text{ ms}$ | $2,789.19\text{ ms}$ | $4,809.80\text{ ms}$ | $12.85\text{ tok/s}$ | $22.95\text{ tok/s}$ |
| Repetition 3 | `req-0026` | $4,967.84\text{ ms}$ | $2,073.30\text{ ms}$ | $2,786.34\text{ ms}$ | $4,859.64\text{ ms}$ | $12.88\text{ tok/s}$ | $22.97\text{ tok/s}$ |
| **Mean** | — | **$4,999.32\text{ ms}$** | **$2,048.95\text{ ms}$** | **$2,818.14\text{ ms}$** | **$4,867.09\text{ ms}$** | **$12.80\text{ tok/s}$** | **$22.72\text{ tok/s}$** |

### 2.3 Reconciliation Finding & Root Cause
1. **The $2,864.12\text{ ms}$ figure:**
   - Corresponds almost exactly to the **server decode duration alone** ($2,818.14\text{ ms}$ server decode plus loopback network framing and chunk-serialization overhead), yielding the reported decode rate of $22.35\text{ tok/s}$ ($64\text{ tokens} / 2.864\text{ s}$).
   - In `storage/experiments/expansion_llama3_2_1b_validation_report.md`, the author mistakenly copied the server-side decode duration into the "Total Duration" column, omitting the **$2,048.95\text{ ms}$ prefill latency** required to evaluate the 677-token prompt.
2. **The $5,000\text{ ms}$ figure:**
   - Represents the **true client-observed end-to-end duration** (Monotonic clock start to finish), which includes both the $2,049\text{ ms}$ prefill phase and the $2,818\text{ ms}$ decode phase:
     $$\text{Total Duration} \approx \text{Prefill Duration } (2,049\text{ ms}) + \text{Decode Duration } (2,818\text{ ms}) + \text{Overhead } (132\text{ ms}) = 4,999\text{ ms} \approx 5.0\text{ s}$$
   - The true client-observed throughput is $64\text{ tokens} / 4.999\text{ s} = 12.80\text{ tok/s}$.
3. **Authoritative Resolution:**
   - Both numbers are scientifically valid measurements of different physical phenomena:
     - **Client End-to-End Latency:** Mean is **$4,999.32\text{ ms}$** ($12.80\text{ tok/s}$).
     - **Server Decode Duration:** Mean is **$2,818.14\text{ ms}$** ($22.72\text{ tok/s}$).
   - The narrative discrepancy in the Session 13 report was caused by conflating decode duration with total generation duration.
   - Moving forward in Session 15, all reports and models explicitly report both components (`server_prompt_eval_duration_ms` and `server_eval_duration_ms`) alongside total client `duration_ms`.

---

## 3. Discrepancy Reconciliation Summary Matrix

| Metric / Attribute | Errant Historical Report Entry | Immutable Raw Data (JSONL/DB) | Reconciled Physical Meaning | Authoritative Value |
|---|---|---|---|---|
| **Ollama Runtime Version** | `0.5.1` (Report prose) | `0.40.11` (Manifest, DB, JSONL) | Local binary and running daemon version | **`0.40.11`** |
| **P677_O64 Duration** | `2,864.12 ms` (Validation report) | `4,999.32 ms` (Req-0012, 0019, 0026 mean) | $2.86\text{s}$ is decode-only; $5.00\text{s}$ is end-to-end (prefill + decode) | **Total: $4,999.32\text{ ms}$**<br>**Decode: $2,818.14\text{ ms}$**<br>**Prefill: $2,048.95\text{ ms}$** |
| **P677_O64 Throughput** | `22.35 tok/s` (Validation report) | `12.80 tok/s` (Req-0012, 0019, 0026 mean) | $22.35\text{ tok/s}$ is decode-only; $12.80\text{ tok/s}$ is end-to-end | **Client: $12.80\text{ tok/s}$**<br>**Decode-only: $22.72\text{ tok/s}$** |
