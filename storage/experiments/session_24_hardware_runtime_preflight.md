# Session 24 — Hardware & Runtime Discovery Preflight

**Date**: 2026-10-29  
**Workstation**: Intel(R) Core(TM) i7-14700 Desktop Workstation  
**Operating System**: Microsoft Windows 11 Pro (Build 26100)  
**Corpus / Project**: `harsh-bitverse/LLM-Inference-Intelligence-Engine`  
**Git Baseline**: Commit `f24019a` on branch `main`

---

## 1. Verified Hardware Fingerprint

Live hardware discovery executed via Windows Management Instrumentation (`Win32_Processor`, `Win32_OperatingSystem`):

| Hardware Attribute | Discovered Value | Specification Details |
|---|---|---|
| **CPU Model** | Intel(R) Core(TM) i7-14700 | 14th Gen Raptor Lake Refresh |
| **Physical Cores** | 20 cores | 8 Performance Cores (P-cores) + 12 Efficient Cores (E-cores) |
| **Logical Processors** | 28 threads | 16 P-core threads (Hyper-Threading) + 12 E-core single threads |
| **Max Clock Speed** | Up to 5.4 GHz (P-core) / 4.2 GHz (E-core) | Base: 2.1 GHz (P-core) / 1.5 GHz (E-core) |
| **Total Physical RAM** | 31.69 GB (~32 GiB DDR5) | Free Physical RAM during preflight: ~8.97 GB |
| **Primary Inference Mode** | CPU-First Host Memory Inference | ggml / llama.cpp AVX2 CPU execution backend |

---

## 2. Verified Runtime Identity

- **Runtime**: Ollama
- **Runtime Version**: `0.40.11`
- **Daemon Binding**: `http://127.0.0.1:11434` (Strict Loopback)
- **Active Process**: Existing healthy daemon reused (PID 328412)
- **Network Isolation**: Loopback calls explicitly bypass ambient corporate HTTP proxy via `NO_PROXY="127.0.0.1,localhost"` and empty Python proxy handlers (`urllib.request.ProxyHandler({})`).

---

## 3. Installed Model Registry & Provenance

Live query to `/api/tags` confirms exact models, digests, and quantization parameters:

| Model ID | File Size (Bytes) | SHA-256 Digest | Parameter Size | Quantization | Context Window | Family |
|---|---|---|---|---|---|---|
| `llama3.2:1b` | 1,321,098,329 | `baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878` | 1.2B | `Q8_0` | 131,072 | `llama` |
| `qwen2.5:1.5b` | 986,061,892 | `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b` | 1.5B | `Q4_K_M` | 32,768 | `qwen2` |
| `qwen2.5:0.5b` | 397,821,319 | `a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67` | 494.03M | `Q4_K_M` | 32,768 | `qwen2` |

*Policy note*: No additional models were downloaded. No existing models were modified.

---

## 4. Runtime Configuration Telemetry & Option Support

Empirical verification conducted against `/api/generate`:

| Control Option | Supported by Ollama v0.40.11 | Verification Mechanism | Observable Effect |
|---|---|---|---|
| `num_thread` | **Yes** (dynamic per-request) | Server execution timing & llama.cpp runner | Modifies prefill and decode thread allocation. Prefill scales up with threads, but autoregressive decode peaks around 8 threads (P-core saturation) and degrades at 14–20 threads. |
| `num_ctx` | **Yes** (dynamic per-request) | Server slot telemetry (`n_ctx_slot`) | Alters context allocation (`n_ctx_slot=2048` vs `4096`). Invalidates slot prompt cache when context size is resized. |
| `num_predict` | **Yes** (dynamic per-request) | Output token count (`eval_count`) | Enforces token generation limit. |
| `keep_alive` | **Yes** (dynamic per-request) | Model residency in memory | Prevents model eviction between consecutive requests. |
| `num_batch` | **Accepted** | Request option accepted | Controls internal batching size during prefill. |
| `concurrency` (Client) | **Yes** (Client-side) | Monotonic arrival and completion timestamps | Client requests are accepted over HTTP. |
| `parallelism` (Server) | **Serialized** (`OLLAMA_NUM_PARALLEL=1`) | Server slot logs (`slot id 0`) | Ollama server defaults to a single execution slot (`slot id 0`). Simultaneous client requests queue behind active requests; client concurrency does not imply parallel GPU/CPU inference slots. |
| `CPU Affinity` | **Unverified / System Default** | OS Process Manager | Windows thread scheduler dispatches threads across P-cores and E-cores based on OS heuristics. |

### Telemetry Provenance Rule
- **Requested vs Effective Settings**: Ollama's HTTP completion payload does not echo back `num_thread` or `num_ctx`. Therefore, benchmarks record `requested_options` explicitly and mark effective verification through server-reported execution durations (`prompt_eval_duration`, `eval_duration`) and server log audit.
