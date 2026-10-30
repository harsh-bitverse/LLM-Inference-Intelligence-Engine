# Session 28 Hardware & Runtime Provenance Report

**Workstation Environment:**
- **CPU:** Intel Core i7-14700 (20 Cores: 8 Performance cores, 12 Efficient cores = 28 hardware threads).
- **Physical Memory:** 32 GB DDR5 RAM (high headroom; model weights + KV cache consume <10% total capacity).
- **Operating System:** Windows 11 Enterprise (Version 10.0.26100).
- **Ollama Runtime:** Version 0.40.11 running persistently on loopback interface `127.0.0.1:11434` (PID 328412).
- **llama.cpp Server Runtime:** Version 0.7.0-dev (build 1, commit `9ff0d7e90`) compiled with Clang 18.1.8 for Windows AMD64. Located inside Ollama's local distribution tree at `C:\Users\ch1251635\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe`.
- **GGUF Model Provenance:**
  - `qwen2.5:0.5b`: `~/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515` (398 MB)
  - `qwen2.5:1.5b`: `~/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4` (986 MB)
  - `llama3.2:1b`: `~/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45` (1.3 GB)
- **Provenance Verdict:** VERIFIED_LOCAL. Zero external binaries, drivers, or models downloaded.
