# Session 23 — Model-Generated Code Evaluation Threat Model & Sandbox Security Architecture

## 1. Executive Summary & Security Baseline

In LLM benchmarking and automated evaluation frameworks (including IIE), model-generated code is inherently **untrusted input**. When an evaluation harness executes Python code produced by a language model, that code executes with the privileges, environment, and network access of the host process unless strictly isolated.

Historically, lightweight benchmark harnesses rely on two common heuristics:
1. Restricted `__builtins__` dictionary inside `exec()` / `eval()`.
2. A subprocess timeout parameter (e.g., `subprocess.run(..., timeout=2.0)`).

**Authoritative Finding**: Neither heuristic constitutes a secure execution boundary. Under standard Windows 11 host execution:
- Python `__builtins__` restrictions are trivial to escape via type introspection (e.g., `().__class__.__bases__[0].__subclasses__()`).
- Subprocess timeouts only terminate the immediate child process, leaving grand-children (orphaned processes) running in the background.
- Child processes inherit current user ambient credentials, environment variables, network sockets, and write access to the user's workspace, repository, and user profile directories.

Consequently, Session 23 establishes a **Fail-Closed Security Architecture**:
Unless an authorized, verified OS-level isolation primitive (such as an AppContainer, Windows Sandbox, or disposable Linux container) is explicitly active, **untrusted code execution is disabled by default**. Syntax and static AST checks still run safely, and the evaluation state is reported transparently as `EXECUTION_UNVERIFIED`.

---

## 2. Windows Host Trust Boundary Analysis

The host workstation is an Intel Core i7-14700 running Windows 11 Pro (`10.0.26100`).

```
+---------------------------------------------------------------------------------+
|                               WINDOWS 11 HOST WORKSTATION                       |
|                                                                                 |
|  +---------------------------------------------------------------------------+  |
|  | User Security Context: ch1251635 (Interactive Desktop Session)            |  |
|  | Ambient Tokens: Git credentials, SSH keys, user home %USERPROFILE%, PATH  |  |
|  +---------------------------------------------------------------------------+  |
|                                     |                                           |
|                                     v                                           |
|  +---------------------------------------------------------------------------+  |
|  | IIE Benchmark Process (Python 3.12 / uv)                                  |  |
|  | Bound ports: 127.0.0.1:11434 (Ollama), 127.0.0.1:8000 (IIE API)           |  |
|  +---------------------------------------------------------------------------+  |
|                                     |                                           |
|                  +------------------+------------------+                        |
|                  | (FAIL-CLOSED)                       | (OPT-IN HARDENED)      |
|                  v                                     v                        |
|  +--------------------------------+   +---------------------------------------+ |
|  | AST Syntax & Static Analysis   |   | Windows Job Object Subprocess         | |
|  | - No bytecode execution        |   | - JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE  | |
|  | - Rejects dangerous AST nodes  |   | - Memory limit: 256 MiB               | |
|  | - Status: EXECUTION_UNVERIFIED |   | - Environment: Sanitized/empty        | |
|  +--------------------------------+   | - Stdout/Stderr capped at 16 KB       | |
|                                       | - Hard timeout: 2.0s                  | |
|                                       +---------------------------------------+ |
+---------------------------------------------------------------------------------+
```

### Trust Boundary Violations Without Isolation
1. **Interactive User Token**: Native subprocesses execute in the same Windows security logon session. Any code run natively has read/write access to files owned by the user.
2. **Loopback Access**: Model code can issue HTTP requests to `http://127.0.0.1:11434` (Ollama management API) to delete models, trigger unmetered model loads, or alter runtime state.
3. **Environment Leakage**: `os.environ` contains system paths, temporary directory locations, and potential user tokens.

---

## 3. Comprehensive Analysis of 11 Threat Vectors

| # | Threat Vector | Mechanism / Exploit Technique | Impact on Host / Benchmark | Mitigation Implemented |
|---|---|---|---|---|
| **1** | **CPU Exhaustion & Infinite Loops** | `while True: pass` or recursive function bomb. | Consumes 100% of an allocated CPU core, freezing benchmark execution. | Hard wall-clock timeout (2.0s) + Job Object CPU rate limit and active process termination. |
| **2** | **Memory Exhaustion (OOM Bomb)** | `s = "a" * (10**9)` or recursive list allocation. | Allocates gigabytes of virtual memory, triggering OS paging or host OOM crashes. | Windows Job Object `JOB_OBJECT_LIMIT_PROCESS_MEMORY` capped at 256 MiB; instant termination upon breach. |
| **3** | **Child Process Spawning / Fork Bomb** | `import subprocess; subprocess.Popen(...)` or `os.fork` equivalent. | Spawns unmanaged processes that outlive the runner. | AST inspection forbids process-spawning imports; Job Object `JOB_OBJECT_LIMIT_ACTIVE_PROCESS` caps total processes to 1. |
| **4** | **Filesystem Data Exfiltration** | Reading sensitive files (`id_rsa`, `.env`, browser profiles, source code). | Loss of confidentiality of user data and workspace source files. | AST static restriction forbidding filesystem access; Fail-Closed execution default. |
| **5** | **Filesystem Mutation / Tampering** | Writing or deleting files (`rmdir`, `os.remove`, overwriting datasets). | Loss of integrity of benchmark logs, datasets, or system configuration. | Fail-Closed mode executes zero bytecode; opt-in execution strips file manipulation builtins and verifies no mutation. |
| **6** | **Environment & Secret Leakage** | `import os; print(os.environ)`. | Exfiltrates secrets, tokens, or system configuration via test output. | Completely scrubbed subprocess environment (`env={}` with only minimal required `SYSTEMROOT`/`COMSPEC` for Windows runtime). |
| **7** | **Localhost & Network SSRF** | Connecting to `127.0.0.1:11434` or internet endpoints via `urllib`/`socket`. | Tampering with Ollama server, model deletion, or external data transmission. | AST import blacklist prohibits `socket`, `http`, `urllib`, `requests`; Fail-Closed execution default. |
| **8** | **Dynamic Metaprogramming Escapes** | `().__class__.__bases__[0].__subclasses__()` to recover `sys.modules` or builtins. | Bypasses Python-level `__builtins__` restrictions in 1-2 lines of standard Python. | Structural AST analysis rejects `__subclasses__`, `__bases__`, `getattr`, `eval`, `exec`; Fail-Closed default prevents execution entirely. |
| **9** | **Native / Interpreter Memory Corruptions** | Invoking `ctypes` or triggering CPython memory bugs. | Host crash or arbitrary native code execution outside CPython VM. | AST prohibits `ctypes`, `cffi`, and binary extensions; memory and execution boundaries enforced by OS Job Object. |
| **10** | **Subprocess-Tree Survival / Orphan Processes** | Spawning a detached child process that does not exit when the parent exits. | Background background tasks persist, leaking memory and CPU indefinitely. | Windows Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. When Job handle closes, Windows kernel terminates entire tree. |
| **11** | **Excessive Output Stream Deadlocks** | Writing 100 MB of data to `sys.stdout` or `sys.stderr`. | Fills OS OS pipe buffers causing `communicate()` deadlock or memory exhaustion in host. | Output streams capped at 16 KB max read; subprocess output strictly truncated. |

---

## 4. Evaluation Taxonomy & Granular Status Schema

To prevent misleading pass/fail binaries when code cannot be securely executed, Session 23 introduces a 7-state evaluation taxonomy:

```
+---------------------------------------------------------------------------------+
|                       EVALUATION TAXONOMY LIFECYCLE                             |
|                                                                                 |
|  [Model Code Output]                                                            |
|          |                                                                      |
|          +---> (Syntax Error) -------------> SYNTAX_INVALID                     |
|          |                                                                      |
|          +---> (AST Parse OK) -------------> SYNTAX_VALID                       |
|                     |                                                           |
|                     +---> (Static Checks Fail) -> STATIC_CHECK_FAILED           |
|                     |                                                           |
|                     +---> (Static Checks OK) ---> STATIC_CHECK_PASSED           |
|                                 |                                               |
|               +-----------------+-----------------+                             |
|               | (Fail-Closed Mode)                | (Opt-In Hardened Sandbox)   |
|               v                                   v                             |
|      EXECUTION_UNVERIFIED               +-------------------+                   |
|      (Default: Zero Bytecode Run)       | Unit Test Runner  |                   |
|                                         +-------------------+                   |
|                                                   |                             |
|                     +-----------------------------+-----------------------+     |
|                     |                             |                       |     |
|                     v                             v                       v     |
|          EXECUTION_TEST_PASSED         EXECUTION_TEST_FAILED      EVALUATION_TIMEOUT
|                                                                   EVALUATOR_ERROR
+---------------------------------------------------------------------------------+
```

### Status Definitions
1. **`SYNTAX_VALID`**: The generated output is valid, parseable Python AST.
2. **`STATIC_CHECK_PASSED`**: The code passes AST structural safety rules (no dangerous imports, no forbidden attribute lookups).
3. **`EXECUTION_TEST_PASSED`**: The code was executed within an authorized, hardened sandbox and passed all deterministic unit tests.
4. **`EXECUTION_TEST_FAILED`**: The code was executed within an authorized, hardened sandbox and failed one or more unit tests or threw a runtime exception.
5. **`EXECUTION_UNVERIFIED`**: Default operational state. The code has valid syntax and static structure, but runtime execution was intentionally withheld because an isolated OS-level sandbox is not authorized or available.
6. **`EVALUATOR_ERROR`**: Evaluation framework or environment error prevented evaluation.
7. **`EVALUATION_TIMEOUT`**: Execution exceeded the strict 2.0s bounded wall-clock limit.

---

## 5. Decision & Recommendation Guardrails

In the recommendation engine (`MultiModelComparator`):
- A model **MUST NOT** be marked `ROBUST_FEASIBLE` under a code pass requirement (`require_code_pass=True` or `min_code_pass_rate > 0`) if its code evaluations are in the `EXECUTION_UNVERIFIED` state.
- When execution evidence is unverified, the engine must assign `UNKNOWN_FEASIBILITY` (or `QUALITY_INFEASIBLE` if strict pass is required) with transparent disclosure:
  > *"Code generation correctness is unverified because untrusted code execution is disabled by fail-closed security policy. Full unit test execution requires an authorized isolated sandbox."*

---

## 6. Future Full OS-Level Sandbox Roadmap

For future sessions where the project owner authorizes isolated execution:
1. **Windows AppContainer Isolation**: Launch execution workers under low-privilege AppContainer SIDs with `SECURITY_CAPABILITY_INTERNET_CLIENT` disabled and no filesystem access outside a temporary scratch directory.
2. **Windows Sandbox / Disposable VM**: Dispatch evaluation scripts into a headless Windows Sandbox container via PowerShell/RDP pipe.
3. **Containerized Linux MicroVM**: Execute untrusted code inside an ephemeral container (e.g., Podman/Docker with `--network none --read-only --memory 256m --cpus 1`).
