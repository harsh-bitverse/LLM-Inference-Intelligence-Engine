# Session 20 — Versioned Workload Evaluation Suite

## 1. Suite Architecture & Design Principles

To compare multiple LLMs objectively, IIE establishes a shared, deterministic benchmark workload suite (`WORKLOAD_SUITE` v1.0.0).

### Core Principles
1. **Deterministic Prompts:** Fixed prompts with zero dynamic variation, ensuring runs are strictly repeatable.
2. **Greedy Sampling:** Evaluated with `temperature=0.0`, `top_p=1.0`, `top_k=1`, `seed=42` to eliminate stochastic generation variance.
3. **Rigorous Output Separation:**
   - **Performance:** Monotonic client duration (`client_e2e_duration_ms`), server prefill time, decode time, normalized characters/second.
   - **Format Compliance:** Syntactic adherence (JSON parseability via `json.loads`, valid Python syntax via `ast.parse`).
   - **Task Success & Correctness:** Objective programmatic criteria (regex needle search, sandboxed unit test assertions, exact arithmetic evaluation).
   - **Subjective Quality:** Explicitly disclaimed. Subjective fluency or stylistic elegance is NOT fabricated or scored with ungrounded heuristics.
4. **Open Content:** Prompts are completely synthesized for technical benchmarks, avoiding copyrighted or proprietary benchmark contamination.

---

## 2. Workload Task Catalog

| Task ID | Task Category | Target Output | Evaluation Method | Objective Success Criteria |
| :--- | :--- | :--- | :--- | :--- |
| `workload_v1_interactive_qa` | `INTERACTIVE_QA` | $\le 16$ tokens | `EXACT_MATCH` | Contains "Paris" (case-insensitive) for capital of France. |
| `workload_v1_long_context_retrieval` | `LONG_CONTEXT_RETRIEVAL` | $\le 24$ tokens | `NEEDLE_RETRIEVAL_REGEX` | Extracts exact override key `ZEPHYR-7703-DELTA` from ~150-word system log. |
| `workload_v1_long_form_generation` | `LONG_FORM_GENERATION` | $\le 160$ tokens | `HEURISTIC_KEYWORD_COMPLETION` | Explains prefill vs decode; $\ge 30$ words; includes "prefill", "decode", and "bandwidth"/"compute". |
| `workload_v1_structured_json` | `STRUCTURED_JSON` | $\le 96$ tokens | `JSON_SCHEMA_VALIDATOR` | Valid JSON; schema matches `patient_id=="PT-4402"`, `age==45`, `systolic_bp==120`, `diastolic_bp==80`, `pulse_bpm==72`. |
| `workload_v1_code_generation` | `CODE_GENERATION` | $\le 140$ tokens | `AST_SYNTAX_AND_UNIT_TEST` | Valid Python AST syntax; compiles and passes 8 unit tests for `is_prime(n: int)`. |
| `workload_v1_reasoning_numerical` | `REASONING_NUMERICAL` | $\le 160$ tokens | `NUMERICAL_REGEX_CHECKER` | Multi-step calculation: $120 - 40 = 80 \to + 60 = 140 \to - 20\% = 112$. Final answer must be 112. |

---

## 3. Evaluation Schema & Metrics

For each task execution, IIE records:
- `format_compliant`: `bool` (whether syntax adhered to the requested structure).
- `task_success`: `bool` (whether the task passed the deterministic correctness threshold).
- `correctness_score`: `float` ($0.0 \le s \le 1.0$).
- `evaluation_method`: Categorical name of the programmatic evaluator.
- `evaluation_notes`: Diagnostic details explaining failures (e.g. failed field names or AST syntax exceptions).
- `response_snippet`: First 120 characters of generated output.
