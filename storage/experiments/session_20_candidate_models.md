# Session 20 — Candidate Model Catalog & Shortlist

## 1. Candidate Model Architecture & Resource Specifications

To evaluate multi-model inference behavior on the local workstation (Intel Core i7-14700 CPU, 20 cores, 28 threads, 32 GB RAM, 122.1 GB available C: disk), we have curated 6 candidate models ranging from 0.49B to 3.82B parameters.

| Model Identifier | Family | Params (B) | Quantization | Disk Footprint | Expected RAM | Context Window | Tokenizer & Vocab | License | Local Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`llama3.2:1b`** | `llama` | 1.24B | Q8_0 | ~1.32 GB | ~1.55 GB | 131,072 | `tiktoken_llama3` (128,256) | Meta Llama 3.2 Community | **INSTALLED** |
| **`qwen2.5:0.5b`** | `qwen2` | 0.49B | Q4_K_M | ~398 MB | ~0.72 GB | 32,768 | `bpe_qwen2` (152,064) | Apache 2.0 | **UNINSTALLED** |
| **`qwen2.5:1.5b`** | `qwen2` | 1.54B | Q4_K_M | ~986 MB | ~1.75 GB | 32,768 | `bpe_qwen2` (152,064) | Apache 2.0 | **UNINSTALLED** |
| **`qwen2.5:3b`** | `qwen2` | 3.09B | Q4_K_M | ~1.90 GB | ~3.20 GB | 32,768 | `bpe_qwen2` (152,064) | Qwen Research License | **UNINSTALLED** |
| **`llama3.2:3b`** | `llama` | 3.21B | Q4_K_M | ~2.00 GB | ~3.40 GB | 131,072 | `tiktoken_llama3` (128,256) | Meta Llama 3.2 Community | **UNINSTALLED** |
| **`phi3.5:3.8b`** | `phi` | 3.82B | Q4_K_M | ~2.20 GB | ~4.10 GB | 131,072 | `tiktoken_phi3` (32,064) | MIT License | **UNINSTALLED** |

---

## 2. Tokenizer and Architecture Contrasts

1. **Llama 3.2 vs. Qwen 2.5:**
   - Llama 3.2 utilizes a 128k vocabulary TikToken implementation.
   - Qwen 2.5 employs a 152k vocabulary BPE tokenizer optimized for multi-lingual and code tokens.
   - **Empirical effect:** For technical and structured JSON text, Qwen's tokenizer produces fewer tokens than Llama 3.2 for the identical character payload, altering naive tokens-per-second calculations.
2. **Quantization Baseline:**
   - The current baseline `llama3.2:1b` was installed at `Q8_0` (8-bit quantization, 1.32 GB).
   - Standard Ollama distributions for `qwen2.5` use `Q4_K_M` (4-bit medium k-quantization). Comparing `Q8_0` against `Q4_K_M` captures both quantization efficiency and architectural differences.

---

## 3. Recommended Shortlist for User Authorization

Before multi-model benchmarks can execute physical cross-model measurements, the user must explicitly authorize any model downloads. We submit the following minimal, scientifically targeted shortlist:

### Recommendation 1: `qwen2.5:1.5b`
- **Identifier:** `qwen2.5:1.5b`
- **Role:** Direct peer model to `llama3.2:1b` (1.54B vs 1.24B parameters).
- **Download Command:** `ollama pull qwen2.5:1.5b`
- **Download Size:** ~986 MB
- **Expected RAM:** ~1.75 GB
- **License:** Apache 2.0 (fully open, permissive)
- **Scientific Value:** Enables true head-to-head comparison between Meta Llama and Alibaba Qwen families in the ~1B-2B parameter class.

### Recommendation 2: `qwen2.5:0.5b`
- **Identifier:** `qwen2.5:0.5b`
- **Role:** Ultra-compact CPU reference model (0.49B parameters).
- **Download Command:** `ollama pull qwen2.5:0.5b`
- **Download Size:** ~398 MB
- **Expected RAM:** ~720 MB
- **License:** Apache 2.0 (fully open, permissive)
- **Scientific Value:** Establishes the speed/quality trade-off floor for constrained CPU environments.

---

## 4. Strict Guardrail Adherence

**Current Status:** In strict accordance with user instructions, **NO additional models have been downloaded or installed during this session**.
Only `llama3.2:1b` is currently installed and active on the workstation. All uninstalled models remain strictly designated `UNINSTALLED_CANDIDATE` with `UNKNOWN` empirical performance and feasibility.
