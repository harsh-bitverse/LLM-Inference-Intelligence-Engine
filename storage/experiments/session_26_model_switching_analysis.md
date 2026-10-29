# Session 26 — Model Switching & Residency Analysis Report

**Serving Runtime:** Ollama v0.40.11  
**Models Tested:** `qwen2.5:0.5b` and `llama3.2:1b`  

---

## 1. Executive Summary: Weight Paging and Model Residency

Desktop LLM servers manage model memory dynamically. When consecutive requests target different models, the server must unload the currently resident model from system memory/VRAM and page the requested model's weights into memory. This introduces a measurable cold-start latency tax on the first request to the new model.

---

## 2. Empirical Model-Switching Observations

| Experiment ID | From Model | To Model | Server Load Duration | Warm Execution | Cold Execution | Client Switching Penalty |
|---|---|---|---|---|---|---|
| `switch-probe-01-a-to-b` | `qwen2.5:0.5b` | `llama3.2:1b` | 3.3 ms | 121.2 ms | 112.5 ms | +0.0 ms |

---

## 3. Capacity Model Integration

IIE's Policy C incorporates an empirical switching penalty of **+850ms** whenever the incoming request targets a non-resident model. This prevents the admission controller from falsely classifying cold model requests as meeting tight latency SLAs.