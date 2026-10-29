# Session 22 — Objective Quality Benchmark Suite Specification

**Version:** 2.2.0  
**Date:** October 28, 2026  
**Total Tasks:** 36 deterministic tasks  
**Partition Splits:** Development (6 tasks), Selection (18 tasks), Held-Out (12 tasks)  
**Categories (6):** `FACTUAL_INSTRUCTION`, `NUMERICAL_REASONING`, `STRUCTURED_JSON`, `CODE_GENERATION`, `CONTEXT_RETRIEVAL`, `LONG_FORM_COMPLIANCE`  

---

## 1. Cryptographic Task-Set Hashes & Data Separation

To ensure scientific integrity and eliminate data contamination, all tasks are strictly partitioned into three isolated splits. No selection rules or model weights are tuned on the held-out split.

| Partition Split | Task Count | Purpose | SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| **Complete Suite** | 36 | Comprehensive evaluation specification | `8603c0949f3c862c42d5a6f83e45f70f2ed242628a52bed175af9bc7deeec1a3` |
| **Development** | 6 | Evaluator calibration and sanity testing | `70f83e7688b96c2864b67ec5d4339dd1677eb2aa03e1ea902a7f8705ed68d194` |
| **Selection** | 18 | Model profiling and recommendation training | `19d3a8273296e2d2a4635b0825933c15e89fc6e3850263495207e56ee9204fb4` |
| **Held-Out** | 12 | Prospective generalization verification only | `4fbca67c48e470a7402effd6db04101291966274da7c7fa5112d9c626ac27f28` |

---

## 2. Complete Task Inventory

| Task ID | Split | Category | Name | Difficulty | Output Limit | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `factual_dev_01` | DEV | Factual | Chemical Symbol Recall | easy | 8 | Exact case-insensitive match |
| `factual_select_01` | SELECTION | Factual | Multi-Instruction Colors | medium | 16 | Punctuation & casing constraints |
| `factual_select_02` | SELECTION | Factual | Moon Landing Year Prefix | easy | 12 | Strict prefix format |
| `factual_select_03` | SELECTION | Factual | Fact Extraction with Units | easy | 16 | Unit adherence required |
| `factual_heldout_01` | HELD_OUT | Factual | Absent Info Abstention | medium | 16 | Must output NOT_SPECIFIED |
| `factual_heldout_02` | HELD_OUT | Factual | Negative Constraint Geography | medium | 12 | Required capital, prohibited distractors |
| `num_dev_01` | DEV | Numerical | Retail Profit Calculation | medium | 48 | Multi-step revenue minus costs |
| `num_select_01` | SELECTION | Numerical | Discount and Sales Tax | medium | 32 | Percentage discount then tax |
| `num_select_02` | SELECTION | Numerical | Distractor Filtering | medium | 64 | Ignores irrelevant building numbers |
| `num_select_03` | SELECTION | Numerical | Speed Rate Calculation | medium | 24 | km/h rate with decimal hours |
| `num_heldout_01` | HELD_OUT | Numerical | Expense Split Table | medium | 32 | Table sum and division |
| `num_heldout_02` | HELD_OUT | Numerical | Algebraic Age Ratio | hard | 48 | System of two linear equations |
| `json_dev_01` | DEV | JSON | Server Config Extraction | easy | 64 | Host, port, ssl_enabled types |
| `json_select_01` | SELECTION | JSON | Nested Clinical Vitals | hard | 96 | Nested demographics and vitals |
| `json_select_02` | SELECTION | JSON | Array Items Extraction | medium | 96 | Item array length and quantity sum |
| `json_select_03` | SELECTION | JSON | Missing Value Null | medium | 64 | Explict null for missing email |
| `json_heldout_01` | HELD_OUT | JSON | Database Float/Bool Config | medium | 80 | Types: int, float, bool, string |
| `json_heldout_02` | HELD_OUT | JSON | Escaped Quotes Support Ticket | hard | 80 | Handles escaped quotation marks |
| `code_dev_01` | DEV | Code | Palindrome String Predicate | medium | 128 | 5 unit tests inside sandbox |
| `code_select_01` | SELECTION | Code | Preserve Order Deduplication | medium | 128 | 4 unit tests (empty, dupes) |
| `code_select_02` | SELECTION | Code | Word Frequency Counter | medium | 128 | 4 unit tests (counts) |
| `code_select_03` | SELECTION | Code | Binary Search Algorithm | hard | 140 | 5 unit tests (found, -1, empty) |
| `code_heldout_01` | HELD_OUT | Code | Sentence Word Reversal | easy | 96 | 4 unit tests (spaces, ordering) |
| `code_heldout_02` | HELD_OUT | Code | Value Clamping Range | easy | 96 | 5 unit tests (low, high, clamp) |
| `retrieval_dev_01` | DEV | Retrieval | Short Context Key Person | easy | 16 | Extracts person name |
| `retrieval_select_01` | SELECTION | Retrieval | Branch Needle in Haystack | medium | 16 | Hidden passcode in 3-branch log |
| `retrieval_select_02` | SELECTION | Retrieval | Superseded Policy Retrieval | medium | 16 | Rejects superseded allowance |
| `retrieval_select_03` | SELECTION | Retrieval | Multi-Fact Patient Aggregation | medium | 16 | Aggregates daily doses for Patient A |
| `retrieval_heldout_01` | HELD_OUT | Retrieval | Unstated Fact Abstention | medium | 16 | Must output ABSENT_FROM_CONTEXT |
| `retrieval_heldout_02` | HELD_OUT | Retrieval | Deep Needle Token | medium | 16 | Extracts node authentication token |
| `longform_dev_01` | DEV | Long-Form | Headings & Length Bounds | medium | 180 | Required headers and word bounds |
| `longform_select_01` | SELECTION | Long-Form | Keyword Inclusion/Prohibition | hard | 180 | Prohibits 'mitochondria' |
| `longform_select_02` | SELECTION | Long-Form | Exact 4-Item List | medium | 160 | Numbered 1-4, no 5th item |
| `longform_select_03` | SELECTION | Long-Form | Release Note Header & Metrics | medium | 160 | Required version & latency claim |
| `longform_heldout_01` | HELD_OUT | Long-Form | Closing Token Constraint | medium | 160 | Ends with '---END OF SUMMARY---' |
| `longform_heldout_02` | HELD_OUT | Long-Form | Pronoun-Free Explanation | hard | 140 | No personal pronouns (we/you/I) |

---

## 3. Evaluation Methodologies by Category

1. **Factual & Instruction-Following:** Regular expression pattern matching, exact case-insensitive extraction, and positive/negative keyword verification.
2. **Numerical Reasoning:** Extract-and-parse numerical evaluation with float tolerance ($\pm 0.01$) against independent mathematical ground truth.
3. **Structured JSON:** Two-tier parsing:
   - Level 1: Syntactic schema parse (`json_schema_valid`).
   - Level 2: Semantic key-value correctness (`json_semantic_correctness`).
4. **Code Generation:**
   - Static AST parsing (`code_syntax_valid`).
   - Sandboxed dynamic unit test execution inside isolated thread with restricted builtins and 2.0s wall-time limits (`code_unit_test_success`, `code_test_pass_rate`).
5. **Context Retrieval:** Factual needle extraction, superseded policy rejection, and explicit abstention detection (`ABSENT_FROM_CONTEXT`).
6. **Long-Form Instruction Compliance:** Word-count bounds, mandatory structural headers, prohibited phrase detection, and mandatory closing tokens.
