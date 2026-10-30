# Session 28 KV-Cache Memory Specification

## 1. Transformer Architecture Hyperparameters

| Model ID | Architecture | Layers ($L$) | Heads ($H$) | KV Heads ($H_{kv}$) | Head Dim ($d_k$) | GQA Ratio | Bytes/Token (FP16) | KB/Token |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen2.5:0.5b` | qwen2 | 24 | 14 | 2 | 64 | 7.0x | 12288 B | 12.00 KB |
| `llama3.2:1b` | llama | 16 | 32 | 8 | 64 | 4.0x | 32768 B | 32.00 KB |
| `qwen2.5:1.5b` | qwen2 | 28 | 12 | 2 | 128 | 6.0x | 28672 B | 28.00 KB |

$$\text{Bytes per Token} = 2 \times L \times H_{kv} \times d_k \times b_{kv}$$

## 2. Theoretical KV-Cache Memory Matrix (Slots x Per-Slot Context)

| Model ID | Slots | Per-Slot Context | Total Context | Theoretical KV Cache | Model Weights | Estimated Total RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen2.5:0.5b` | 1 | 512 tok | 512 tok | 6.0 MB | 398.0 MB | 554.0 MB |
| `qwen2.5:0.5b` | 1 | 1024 tok | 1024 tok | 12.0 MB | 398.0 MB | 560.0 MB |
| `qwen2.5:0.5b` | 1 | 2048 tok | 2048 tok | 24.0 MB | 398.0 MB | 572.0 MB |
| `qwen2.5:0.5b` | 1 | 4096 tok | 4096 tok | 48.0 MB | 398.0 MB | 596.0 MB |
| `qwen2.5:0.5b` | 2 | 512 tok | 1024 tok | 12.0 MB | 398.0 MB | 560.0 MB |
| `qwen2.5:0.5b` | 2 | 1024 tok | 2048 tok | 24.0 MB | 398.0 MB | 572.0 MB |
| `qwen2.5:0.5b` | 2 | 2048 tok | 4096 tok | 48.0 MB | 398.0 MB | 596.0 MB |
| `qwen2.5:0.5b` | 2 | 4096 tok | 8192 tok | 96.0 MB | 398.0 MB | 644.0 MB |
| `qwen2.5:0.5b` | 4 | 512 tok | 2048 tok | 24.0 MB | 398.0 MB | 572.0 MB |
| `qwen2.5:0.5b` | 4 | 1024 tok | 4096 tok | 48.0 MB | 398.0 MB | 596.0 MB |
| `qwen2.5:0.5b` | 4 | 2048 tok | 8192 tok | 96.0 MB | 398.0 MB | 644.0 MB |
| `qwen2.5:0.5b` | 4 | 4096 tok | 16384 tok | 192.0 MB | 398.0 MB | 740.0 MB |
| `llama3.2:1b` | 1 | 512 tok | 512 tok | 16.0 MB | 1321.0 MB | 1487.0 MB |
| `llama3.2:1b` | 1 | 1024 tok | 1024 tok | 32.0 MB | 1321.0 MB | 1503.0 MB |
| `llama3.2:1b` | 1 | 2048 tok | 2048 tok | 64.0 MB | 1321.0 MB | 1535.0 MB |
| `llama3.2:1b` | 1 | 4096 tok | 4096 tok | 128.0 MB | 1321.0 MB | 1599.0 MB |
| `llama3.2:1b` | 2 | 512 tok | 1024 tok | 32.0 MB | 1321.0 MB | 1503.0 MB |
| `llama3.2:1b` | 2 | 1024 tok | 2048 tok | 64.0 MB | 1321.0 MB | 1535.0 MB |