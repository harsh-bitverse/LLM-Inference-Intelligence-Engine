# Session 20 — Multi-Model Benchmarking Protocol

## 1. Objective and Scientific Rationale

When benchmarking a single model configuration (e.g. `llama3.2:1b`), reporting metrics in `tokens/second` is sufficient because token length is uniform across runs. However, extending benchmarks to **multiple models** across different model families introduces a critical measurement artifact: **Tokenizer Incommensurability**.

### The Tokenizer Discrepancy Problem
1. **Vocabulary Size and Compression:**
   - Llama 3 / 3.2 uses a TikToken-based BPE tokenizer with a vocabulary of 128,256 tokens.
   - Qwen 2.5 uses a BPE tokenizer with a vocabulary of 152,064 tokens.
   - Gemma / Gemma 2 uses a SentencePiece tokenizer with 256,000 tokens.
   - Older Llama 2 / Mistral models use 32,000 tokens.
2. **Token-to-Text Divergence:**
   - For an identical English paragraph or structured JSON snippet, a model with a larger, more dense vocabulary (e.g. Qwen 2.5 or Gemma 2) may tokenize the prompt into 85 tokens, whereas a model with a 32k or 128k vocabulary may tokenize it into 110 tokens.
   - Similarly, during generation, one model may express the same answer in 40 tokens while another requires 55 tokens.
3. **Flawed Raw Throughput Comparisons:**
   - If Model A decodes at 25 tokens/second and produces 50 tokens (total time: 2.0 s), while Model B decodes at 20 tokens/second but produces only 35 tokens (total time: 1.75 s) for the exact same semantic answer, Model B completes the user's task faster despite having a lower raw "tokens/second" metric!
   - Treating `tokens/sec` as directly comparable across different tokenizers is scientifically invalid.

---

## 2. Normalized Comparison Contract

To provide fair, defensible cross-model evaluations, the IIE Multi-Model Protocol enforces dual reporting:

| Metric Category | Metric Field | Unit | Cross-Model Comparable? | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Normalized Task** | `task_elapsed_ms` | ms | **Yes (Primary)** | Monotonic client wall-clock duration to complete identical prompt task. |
| **Normalized Text** | `characters_per_second` | chars/s | **Yes** | Textual output rate independent of token segmentation. |
| **Normalized Text** | `words_per_second` | words/s | **Yes** | Natural language delivery rate for human reading comprehension. |
| **Model Prefill** | `server_prefill_duration_ms` | ms | **Yes** | Server prompt processing time for the exact same input text. |
| **Raw Prefill Tokens** | `raw_prompt_tokens` | tokens | No | Model-specific tokenizer count for the prompt. |
| **Raw Output Tokens** | `raw_output_tokens` | tokens | No | Model-specific tokenizer count for the generated text. |
| **Raw Decode Throughput** | `raw_decode_tokens_per_second` | tok/s | **No (Flagged)** | Valid only within the same model family and tokenizer. |

---

## 3. Protocol Schema Specification

Every multi-model benchmark observation recorded in IIE captures the following 19 mandatory dimensions:

1. **Model Identity & Digest:** Official tag (e.g. `llama3.2:1b`), weights SHA256 digest (`baf6a787...`), and parent model ID.
2. **Model Family & Architecture:** Family designation (`llama`, `qwen2`, `phi`), transformer layers, attention heads, KV heads.
3. **Parameter Count & Size:** Total parameter count in billions (`parameter_count_billion`) and exact weight size in bytes.
4. **Quantization Scheme:** Weight quantization format (`Q8_0`, `Q4_K_M`, etc.) and block size.
5. **Runtime Engine & Version:** Runtime platform (`ollama`), executable version (`0.40.11`), and API commit hash.
6. **Hardware & Execution Backend:** Hardware platform ID (`cpu-intel-i7-14700`), ISA SIMD instructions (`AVX2`, `AVX-512`), thread count.
7. **Context Length:** Engine configured `num_ctx` and architectural maximum context window.
8. **Actual Prompt Tokens:** Observed prompt tokens from server response metadata.
9. **Actual Generated Tokens:** Observed output tokens from server response metadata.
10. **Concurrency Level:** Number of parallel client requests executing against the engine (enforced at $C=1$ for baseline comparisons).
11. **Sampling Configuration:** Explicitly fixed decoding parameters (`temperature=0.0`, `top_p=1.0`, `top_k=1`, deterministic seed=42).
12. **Model Residency State:** Weight presence in memory prior to request dispatch (`WARM` vs `COLD`).
13. **Prompt Cache State:** Prefix KV cache state (`COLD_PREFIX`, `WARM_PREFIX`).
14. **Warmup Policy:** Explicit exclusion of initial warmup invocation (`1_warmup_excluded`).
15. **Request Order & Position:** Monotonic sequence index within the experiment run.
16. **Client E2E Duration:** Monotonic client wall-clock duration (`client_e2e_duration_ms`).
17. **Server Timing Phases:** Decomposed prefill duration (`server_prefill_duration_ms`) and decode duration (`server_decode_duration_ms`).
18. **Memory Watermark:** Sampled working set RSS (`process_rss_mb`) of the server runtime process.
19. **Measurement Provenance:** Rigorous tag (`EMPIRICAL`, `SIMULATED`, or `ANALYTICAL`) ensuring unmeasured predictions are never disguised as measurements.

---

## 4. Comparability Enforcement Rules

1. **Disallow Naked Token Comparisons:** Any table or chart displaying `tokens/sec` across different model families MUST display `cross_tokenizer_comparable: False` and the disclaimer:
   > *"Raw tokens/second cannot be directly compared across differing tokenizers. Compare task_elapsed_ms or characters_per_second for equivalent workloads."*
2. **Uninstalled Candidate Safeguard:** Candidate models defined in the catalog that are not currently downloaded or installed on the host workstation MUST be labeled `UNINSTALLED_CANDIDATE`. Their performance metrics MUST be reported as `UNKNOWN` rather than interpolated or fabricated.
3. **Execution Feasibility:** Predictions for uninstalled models must remain `EXPLORATORY_PREDICTION` or `UNKNOWN_FEASIBILITY` until empirical benchmarks on the host workstation validate them.
