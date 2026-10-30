# Session 28 Model & Context Equivalence Audit

This audit establishes the degree of model representation equivalence across Ollama and llama.cpp server.

| Model ID | Runtime | Quantization | Tokenizer | Verified Usable Context | Allocation Geometry | Equivalence Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen2.5:0.5b` | Ollama | Q4_K_M | Embedded GGUF Tokenizer (qwen2.5:0.5b) | 2048 tok | 1 slots x 2048 = 2048 tok | **EXACT_EQUIVALENT** |
| `qwen2.5:0.5b` | llama.cpp server | Q4_K_M | Embedded GGUF Tokenizer (qwen2.5:0.5b) | 1024 tok | 2 slots x 1024 = 2048 tok | **EXACT_EQUIVALENT** |
| `llama3.2:1b` | Ollama | Q8_0 | Embedded GGUF Tokenizer (llama3.2:1b) | 2048 tok | 1 slots x 2048 = 2048 tok | **EXACT_EQUIVALENT** |
| `llama3.2:1b` | llama.cpp server | Q8_0 | Embedded GGUF Tokenizer (llama3.2:1b) | 1024 tok | 2 slots x 1024 = 2048 tok | **EXACT_EQUIVALENT** |
| `qwen2.5:1.5b` | Ollama | Q4_K_M | Embedded GGUF Tokenizer (qwen2.5:1.5b) | 2048 tok | 1 slots x 2048 = 2048 tok | **EXACT_EQUIVALENT** |
| `qwen2.5:1.5b` | llama.cpp server | Q4_K_M | Embedded GGUF Tokenizer (qwen2.5:1.5b) | 1024 tok | 2 slots x 1024 = 2048 tok | **EXACT_EQUIVALENT** |

### Key Audit Conclusions:
1. **Weight & Tokenizer Equivalence:** EXACT_EQUIVALENT. Both runtimes evaluate the exact same underlying GGUF model blobs stored in `~/.ollama/models/blobs/`.
2. **Context Geometry Distinction:** Ollama provides a single shared 2048 context. In llama-server, passing `-c 2048 -np 2` allocates 1024 tokens per slot, while `-c 4096 -np 4` allocates 1024 tokens per slot. To maintain context equivalence with Ollama, llama-server must be launched with `-c (slots * context)`.