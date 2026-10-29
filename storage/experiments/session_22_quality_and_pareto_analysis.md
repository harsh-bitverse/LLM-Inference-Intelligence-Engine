# Session 22 — Quality Analysis & Corrected Pareto Frontier

**Experiment ID:** `exp-s22-quality`  
**Evaluation Scope:** 24 Main Tasks (6 Development + 18 Selection) + 36 Variability Observations across 3 Models = 108 Total Observations  
**Hardware:** Intel Core i7-14700 CPU (20 cores, 28 threads, 32 GB RAM, Windows 11)  
**Runtime:** Ollama Server v0.40.11 on `127.0.0.1:11434`  
**Evaluated Models:** `llama3.2:1b` (Q8_0), `qwen2.5:1.5b` (Q4_K_M), `qwen2.5:0.5b` (Q4_K_M)  

---

## 1. Aggregate Model Performance & Quality Summary

| Model ID | Family | Params | Quant | Obs Count | Median E2E (ms) | Mean Decode (tok/s) | Chars/s | Overall Mean Score | Compliance (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `llama3.2:1b` | `llama` | 1.24B | `Q8_0` | 36 | 2,197.2 | 20.4 | 52.8 | 0.669 | 91.7% |
| `qwen2.5:1.5b` | `qwen2` | 1.54B | `Q4_K_M` | 36 | 1,619.3 | 24.9 | 66.9 | **0.785** | **100.0%** |
| `qwen2.5:0.5b` | `qwen2` | 0.49B | `Q4_K_M` | 36 | **764.9** | **60.3** | **142.7** | 0.488 | 94.4% |

---

## 2. Category-Level Quality Breakdown

To prevent the distortion of universal scalar averaging, quality is evaluated and reported across 6 distinct functional categories:

| Category | Metric | `llama3.2:1b` | `qwen2.5:1.5b` | `qwen2.5:0.5b` | Winner / Key Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Factual & Instruction** | Accuracy (%) | 66.7% | **100.0%** | 33.3% | `qwen2.5:1.5b` (Perfect factual & formatting adherence) |
| | Mean Score | 0.667 | **1.000** | 0.333 | |
| | Median E2E (ms) | 442.3 | 370.6 | **328.7** | |
| **Numerical Reasoning** | Accuracy (%) | **16.7%** | 0.0% | 0.0% | `llama3.2:1b` (Only model to solve multi-step retail profit) |
| | Mean Score | **0.167** | 0.000 | 0.000 | |
| | Median E2E (ms) | 1,920.8 | 1,388.9 | **273.9** | |
| **Structured JSON** | Syntax Compliance | 50.0% | **100.0%** | **100.0%** | `qwen2.5:0.5b` (100% semantic score at lowest latency: 1,432 ms) |
| | Semantic Score | 0.500 | **1.000** | **1.000** | |
| | Median E2E (ms) | 4,405.8 | 3,478.8 | **1,432.8** | |
| **Code Generation** | Syntax Compliance | **100.0%** | **100.0%** | 66.7% | `qwen2.5:1.5b` (100% unit tests passed across all code tasks) |
| | Unit Test Pass Rate | 93.3% | **100.0%** | 0.0% | |
| | Median E2E (ms) | 3,155.4 | 2,566.0 | **2,381.9** | |
| **Context Retrieval** | Accuracy (%) | **83.3%** | **83.3%** | **83.3%** | Tied across models (all 3 extract key needles reliably) |
| | Mean Score | 0.833 | 0.833 | 0.833 | |
| | Median E2E (ms) | 540.3 | 498.9 | **239.0** | `qwen2.5:0.5b` provides fastest retrieval (239 ms) |
| **Long-Form Compliance**| Compliance Score | **0.911** | 0.875 | 0.764 | `llama3.2:1b` (Superior header & structural adherence) |
| | Accuracy (%) | **66.7%** | 50.0% | 33.3% | |
| | Median E2E (ms) | 9,865.0 | 7,919.6 | **2,266.5** | |

---

## 3. Corrected Pareto Efficiency Analysis

In accordance with the Session 22 scientific reconciliation, Pareto dominance is evaluated strictly over explicitly measured objectives, rather than input bit-width or architecture labels.

### Multi-Objective Formulations:

#### Objective Set 1: General Interactive Inference
$$\min(\text{Median Latency}), \quad \min(\text{Resident Memory Footprint}), \quad \max(\text{Mean Correctness Score})$$
- **`qwen2.5:0.5b`**: (765 ms, 398 MB weights, 0.488 score) $\rightarrow$ **Non-Dominated** (Fastest & lowest memory).
- **`qwen2.5:1.5b`**: (1,619 ms, 986 MB weights, 0.785 score) $\rightarrow$ **Non-Dominated** (Highest overall correctness).
- **`llama3.2:1b`**: (2,197 ms, 1,321 MB weights, 0.669 score) $\rightarrow$ **Dominated** by `qwen2.5:1.5b` under this aggregate scalar objective set.

#### Objective Set 2: Code Generation Workload
$$\min(\text{Median Latency}), \quad \max(\text{Unit Test Pass Rate})$$
- **`qwen2.5:1.5b`**: (2,566 ms, 1.000 test pass rate) $\rightarrow$ **Pareto Optimal** (Best test pass rate).
- **`qwen2.5:0.5b`**: (2,381 ms, 0.000 test pass rate) $\rightarrow$ Rejection: Fails mandatory execution constraint.
- **`llama3.2:1b`**: (3,155 ms, 0.933 test pass rate) $\rightarrow$ Dominated by `qwen2.5:1.5b` (slower and lower pass rate).

#### Objective Set 3: Long-Form Instruction & Structural Adherence Workload
$$\min(\text{Median Latency}), \quad \max(\text{Structural Adherence Score})$$
- **`qwen2.5:0.5b`**: (2,266 ms, 0.764 adherence) $\rightarrow$ **Non-Dominated** (Speed frontier).
- **`llama3.2:1b`**: (9,865 ms, 0.911 adherence) $\rightarrow$ **Non-Dominated** (Quality frontier: best structural compliance).
- **`qwen2.5:1.5b`**: (7,919 ms, 0.875 adherence) $\rightarrow$ Balanced trade-off point.

#### Objective Set 4: High-Throughput Structured Data Extraction
$$\min(\text{Median Latency}), \quad \max(\text{JSON Semantic Correctness})$$
- **`qwen2.5:0.5b`**: (1,432 ms, 1.000 semantic score) $\rightarrow$ **Pareto Dominant** (Achieves 100% semantic accuracy while running 2.4x faster than Qwen 1.5B and 3.0x faster than Llama 3.2 1B).
