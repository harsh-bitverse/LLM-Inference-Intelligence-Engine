# Session 28 Slot & Context Optimization Analysis

Answers the central question: **How many simultaneous inference requests can this workstation serve while preserving required context and meeting latency SLAs?**

| Model ID | Context Req | Offered Concurrency | Recommended Slots | Verified Slot Context | Est Total RAM | Expected P95 | Risk | Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen2.5:0.5b` | 512 tok | C=1 | **1 slots** | 512 tok | 554 MB | 2682 ms | LOW | **LATENCY_UNFEASIBLE** |
| `qwen2.5:0.5b` | 1024 tok | C=2 | **2 slots** | 1024 tok | 572 MB | 5592 ms | LOW | **LATENCY_UNFEASIBLE** |
| `llama3.2:1b` | 2048 tok | C=4 | **2 slots** | 2048 tok | 1599 MB | 44918 ms | MODERATE | **LATENCY_UNFEASIBLE** |
| `qwen2.5:1.5b` | 1024 tok | C=2 | **2 slots** | 1024 tok | 1192 MB | 10648 ms | LOW | **LATENCY_UNFEASIBLE** |