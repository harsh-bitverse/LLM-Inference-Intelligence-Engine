# Session 27 — Parallelism & Continuous Batching Verification Evidence

## 1. Verification of Server Execution Overlap

Client concurrency alone does not prove parallel server execution. We verified genuine server execution overlap using:
1. Monotonic start and end timestamps showing overlapping execution windows.
2. llama.cpp server `/slots` endpoint telemetry confirming multiple active slots (`is_processing: true`).
3. Timing decomposition separating queue delay from pure decode time.

## 2. Capabilities Summary

- **Ollama v0.40.11:** Client requests are accepted via HTTP and queued serially. Server slot count = 1.
- **llama.cpp server v0.7.0-dev:** Configured with `-np 2` or `-np 4` and `-cb` (continuous batching). Real parallel multi-slot execution verified.
