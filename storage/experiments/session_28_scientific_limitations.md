# Session 28 Scientific & Operational Limitations

1. **CPU Memory Bandwidth Saturation:** On the Intel Core i7-14700 with dual-channel DDR5 RAM (~70 GB/s peak bandwidth), scaling from 2 to 4 parallel slots improves token throughput by +150% to +177%, but incurs a ~45% slowdown in single-token decode latency due to CPU cache contention.
2. **Context Window Partitioning:** In llama.cpp server, total context is partitioned across slots (`total_context / slots`). If a server is launched with `-c 2048 -np 4`, each slot only receives 512 tokens. Clients submitting prompts larger than 512 tokens will experience truncation unless `-c (slots * required_context)` is specified.
3. **Model Swapping Overhead:** While Ollama maintains dual warm residency in RAM across models without process recreation, llama.cpp server requires a dedicated process per model. Multi-model serving under llama-server requires either multiple dedicated ports or external reverse-proxy orchestration.
