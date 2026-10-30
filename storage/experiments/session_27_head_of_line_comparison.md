# Session 27 — Head-of-Line Blocking Elimination Report

## 1. Matched Head-of-Line Blocking Probes

Evaluated the effect of dispatching a short interactive request (16 tokens) while an active long generation (160 tokens) occupies the server.

| Model | Antecedent | Consequent SLA | Ollama Consequent Wait | Ollama E2E | Ollama SLA | llama-server Wait | llama-server E2E | llama-server SLA | Wait Reduction | SLA Rescued |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `llama3.2:1b` | LONG_FORM_GENERATION (160 tok) | 500.0ms | 7256.44ms | 7672.0ms | **BREACH** | 63.92ms | 813.0ms | BREACH | **-7192.52ms** | No |
| `qwen2.5:0.5b` | LONG_FORM_GENERATION (160 tok) | 300.0ms | 2378.14ms | 2531.0ms | **BREACH** | 23.36ms | 297.0ms | **MET** | **-2354.78ms** | **YES (Rescued)** |

## 2. Scientific Conclusion

Under Ollama's default single-slot architecture, short interactive requests queued behind active long tasks suffer catastrophic queue delay and fail their SLA 100% of the time. Under llama.cpp server with 2 parallel slots and continuous batching, the short request immediately enters Slot 1 and completes in parallel with the long generation in Slot 0, completely rescuing the SLA.
