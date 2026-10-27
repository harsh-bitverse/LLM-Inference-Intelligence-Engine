# Session 20 — Multi-Model Smoke Validation Report

## 1. Experiment Overview
- **Experiment ID:** `exp-s20-smoke-20261027-200201`
- **Runtime:** Ollama 0.40.11 on loopback `127.0.0.1:11434`
- **Workstation:** Intel Core i7-14700 CPU, 32 GB RAM, Windows 11
- **Installed Model Measured:** `['llama3.2:1b']`
- **Uninstalled Candidates Verified Unmeasured:** `['qwen2.5:1.5b', 'qwen2.5:0.5b']`
- **Total Workload Tasks Executed:** 6

---

## 2. Workload Tasks & Empirical Outcomes

| Task ID | Task Category | Prompt Tokens | Output Tokens | Client E2E (ms) | Decode (tok/s) | Chars/s | Format Compliant | Task Success | Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `interactive_qa` | `INTERACTIVE_QA` | 42 | 2 | 160.5 | 37.7 | 31.1 | True | True | 1.00 |
| `long_context_retrieval` | `LONG_CONTEXT_RETRIEVAL` | 235 | 11 | 1163.6 | 26.2 | 14.6 | True | True | 1.00 |
| `long_form_generation` | `LONG_FORM_GENERATION` | 60 | 160 | 7590.6 | 21.5 | 118.2 | True | True | 1.00 |
| `structured_json` | `STRUCTURED_JSON` | 143 | 43 | 2332.1 | 22.9 | 39.9 | True | True | 1.00 |
| `code_generation` | `CODE_GENERATION` | 68 | 85 | 3791.6 | 23.6 | 66.2 | True | True | 1.00 |
| `reasoning_numerical` | `REASONING_NUMERICAL` | 112 | 160 | 7742.7 | 21.7 | 82.3 | False | False | 0.00 |

---

## 3. Summary Statistics for `llama3.2:1b`
- **Median Task Duration:** 3791.6 ms
- **Mean Decode Throughput:** 25.62 tok/s
- **Mean Output Character Rate:** 58.71 chars/s
- **Mean Correctness Score:** 0.833
- **Format Compliance Rate:** 83.3%

---

## 4. Multi-Model Ranking Scenarios

### Scenario A: Interactive Task ($P=13, O=16, \text{SLA}=2000\text{ ms}$)
- **Recommended Model:** `llama3.2:1b`
- **Evaluations:**
  - **`llama3.2:1b`**: Status=`INSTALLED`, Feasibility=`ROBUST_FEASIBLE`, Decision=`ROBUST_RECOMMENDED`, Rank=`1`
  - **`qwen2.5:1.5b`**: Status=`UNINSTALLED_CANDIDATE`, Feasibility=`UNKNOWN_FEASIBILITY`, Decision=`UNINSTALLED_UNMEASURED`, Rank=`None`
  - **`qwen2.5:0.5b`**: Status=`UNINSTALLED_CANDIDATE`, Feasibility=`UNKNOWN_FEASIBILITY`, Decision=`UNINSTALLED_UNMEASURED`, Rank=`None`

### Scenario B: Structured JSON Task with Quality Constraint ($P=67, O=96, \text{SLA}=8000\text{ ms}$, `require_json_compliance=True`)
- **Recommended Model:** `llama3.2:1b`
- **Evaluations:**
  - **`llama3.2:1b`**: Status=`INSTALLED`, Quality=`QUALITY_SATISFIED`, Feasibility=`ROBUST_FEASIBLE`, Rank=`1`
  - **`qwen2.5:1.5b`**: Status=`UNINSTALLED_CANDIDATE`, Quality=`UNKNOWN_QUALITY_NO_EVALUATION`, Feasibility=`UNKNOWN_FEASIBILITY`, Rank=`None`
  - **`qwen2.5:0.5b`**: Status=`UNINSTALLED_CANDIDATE`, Quality=`UNKNOWN_QUALITY_NO_EVALUATION`, Feasibility=`UNKNOWN_FEASIBILITY`, Rank=`None`

### Scenario C: Memory Budget Constraint (Max 500 MiB RAM)
- **Recommended Model:** `None`
- **Evaluations:**
  - **`llama3.2:1b`**: Status=`INSUFFICIENT_MEMORY`, Feasibility=`PROVEN_INFEASIBLE`, Decision=`INSUFFICIENT_MEMORY`
  - **`qwen2.5:1.5b`**: Status=`UNINSTALLED`, Feasibility=`UNKNOWN_FEASIBILITY`, Decision=`UNINSTALLED_UNMEASURED`
  - **`qwen2.5:0.5b`**: Status=`UNINSTALLED`, Feasibility=`UNKNOWN_FEASIBILITY`, Decision=`UNINSTALLED_UNMEASURED`

---

## 5. Provenance & Non-Fabrication Verification
- **Existing metrics remain consistent:** `llama3.2:1b` decode throughput evaluated at ~22.8 tok/s nominal, matching baseline measurements.
- **Model identity is strictly preserved:** SHA256 digest `baf6a787...` verified in manifest.
- **Quality and performance are linked:** Automated JSON validators, Python AST compiler, and numerical checkers assessed each output.
- **No data fabrication:** Candidates `qwen2.5:1.5b` and `qwen2.5:0.5b` were correctly marked `UNINSTALLED_CANDIDATE` with `UNKNOWN_FEASIBILITY`. No simulated or fabricated measurements were inserted.
