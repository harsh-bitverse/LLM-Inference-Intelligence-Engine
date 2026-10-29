# Session 23 — Benchmark Integrity Audit & Dataset Verification Report

**Audit Status**: **PASS** (11/11 checks passed)  
**Audit Timestamp (UTC)**: `2026-10-29T13:22:13.870195+00:00`  
**Verified Manifest Experiment ID**: `exp-s22-quality-20261028-151713`  
**Profiling Observations**: `108`  
**Held-Out Observations**: `36`  
**Total Reconciled Observations**: `144`  

---

## 1. Detailed Verification Checks

| # | Verification Check | Status | Details |
|---|---|---|---|
| 1 | **Artifact File Existence** | `PASS` | All 5 Session 22 benchmark artifacts exist on disk |
| 2 | **Manifest Schema Compliance** | `PASS` | Manifest schema valid (experiment_id: exp-s22-quality-20261028-151713) |
| 3 | **Task Partition Cryptographic Hashes** | `PASS` | SHA-256 suite and partition hashes match authoritative expanded suite exactly |
| 4 | **Profiling Observation Count** | `PASS` | JSONL contains exactly 108 observations (expected 108: 36 tasks x 3 models) |
| 5 | **Observation ID Uniqueness** | `PASS` | All 108 observation IDs are globally unique |
| 6 | **Task Metadata and Split Attribution** | `PASS` | All observations correctly attributed to defined tasks and splits (Splits: {'DEV': 18, 'SELECTION': 90}) |
| 7 | **Model Weight Digest Consistency** | `PASS` | All observations match authoritative SHA256 model digests across all 3 models |
| 8 | **Measurement Timestamp Monotonicity** | `PASS` | All observation timestamps are strictly non-decreasing |
| 9 | **CSV Dataset Parity** | `PASS` | CSV table matches JSONL stream exactly (108 rows) |
| 10 | **Prospective Validation Held-Out Integrity** | `PASS` | Prospective validation confirmed 36 held-out observations across 6 scenarios with matching partition hash |
| 11 | **Total Multi-Model Observation Accounting** | `PASS` | Reconciled 144 observations (108 profiling + 36 held-out = 144 total) |

---

## 2. Partition & Hash Accounting Summary

| Partition | Task Count | Observed Observations (x3 Models) | SHA-256 Partition Digest |
|---|---|---|---|
| **DEV** | 6 tasks | 18 observations | `70f83e7688b96c2864b67ec5d4339dd1677eb2aa03e1ea902a7f8705ed68d194` |
| **SELECTION** | 18 tasks | 54 observations | `19d3a8273296e2d2a4635b0825933c15e89fc6e3850263495207e56ee9204fb4` |
| **HELD_OUT** | 12 tasks | 36 observations (Prospective) | `4fbca67c48e470a7402effd6db04101291966274da7c7fa5112d9c626ac27f28` |
| **TOTAL SUITE** | **36 tasks** | **144 observations** | `8603c0949f3c862c42d5a6f83e45f70f2ed242628a52bed175af9bc7deeec1a3` |

---

## 3. Scientific Reconciliation & Security Attribution Note

- Historical Session 22 records were captured under `RESTRICTED_AST_SANDBOX_UNIT_TEST` with standard subprocess execution.
- In Session 23, all historical raw measurements are **strictly preserved** without in-place mutation.
- For forward-looking and production recommendations, code generation quality is categorized under the fail-closed taxonomy (`EXECUTION_UNVERIFIED` by default) to ensure that untrusted model output is never executed on the host without an authorized OS container.
