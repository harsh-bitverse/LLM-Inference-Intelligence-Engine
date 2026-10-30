# Session 27 — Serving Architecture Comparison Report

**Host Hardware:** Intel Core i7-14700 (28 threads), 32 GB RAM, Windows 11

## 1. Executive Summary

Controlled benchmarks compared Ollama's default single-slot serial execution against llama.cpp server's multi-slot continuous batching architecture.

## 2. Quantitative Architecture Performance Matrix

| Workload | Model | Concurrency (C) | Baseline P95 | Target P95 | P95 Speedup | Throughput Gain | Mean Wait Reduction | SLA Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Balanced Multi-Client Load (C=2) | `qwen2.5:0.5b` | C=2 | 1235.0ms | 609.0ms | **2.03x** | **+100.0%** | -295.5ms | 100.0% -> **100.0%** |
| Bursty Interactive Traffic (C=4) | `qwen2.5:0.5b` | C=4 | 1797.0ms | 719.0ms | **2.5x** | **+151.2%** | -626.3ms | 75.0% -> **100.0%** |

## 3. Findings

1. **Parallel Execution Overlap:** In llama.cpp server with continuous batching, concurrent requests decode simultaneously rather than queuing. Under C=2, effective server concurrency reached 1.9+, compared to 1.0 on Ollama.
2. **P95 Latency Reduction:** Continuous batching reduced P95 tail latency by up to 2x under multi-client load, directly eliminating the queue serialization delay inherent to Ollama's single-slot server.
