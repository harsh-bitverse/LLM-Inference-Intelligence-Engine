# Real-Model Benchmarking & Empirical Dataset Guide (Session 11)

This guide documents how to execute reproducible, evidence-qualified real-model benchmarks
using the **LLM Inference Intelligence Engine (IIE)** and export empirical datasets.

---

## 1. Environmental Prerequisites & Discovery

Before running live benchmarks, verify your workstation hardware and runtime environment:

### System Discovery
Run the following PowerShell commands to check host resources:
```powershell
# CPU model and core counts
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors

# Physical RAM
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory

# Graphics Adapter (Check if dedicated GPU is available)
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion
```

### Hardware Constraints on CPU-Only Workstations
On hosts with integrated GPUs (such as Intel UHD Graphics 770) and no dedicated CUDA/ROCm accelerator:
- Execution operates in **CPU inference mode**.
- Models should be quantized (e.g. `GGUF_Q4_K_M` or `INT4`) and bounded to $\le 3\text{B}$ parameters (e.g., `qwen2.5:0.5b`, `llama3.2:1b`, or `smollm2:135m`).
- Memory telemetry relies on host process RSS (`psutil`) because GPU NVML counters are unavailable.

---

## 2. Setting Up a Local Inference Runtime

### Option A: Ollama (Recommended for Quick Setup)

1. **Download and Install**:
   Download the official Windows installer from [ollama.com/download/windows](https://ollama.com/download/windows) and run the installer.

2. **Verify Installation**:
   Open a new PowerShell window:
   ```powershell
   ollama --version
   ```

3. **Pull a Small CPU-Friendly Model**:
   ```powershell
   ollama pull qwen2.5:0.5b
   # or
   ollama pull llama3.2:1b
   ```

4. **Verify Serving Endpoint**:
   Ensure the daemon is listening on port `11434`:
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
   ```

---

### Option B: llama.cpp (For Fine-Grained Quantization Control)

1. **Download Release**:
   Download the latest pre-built Windows release (e.g. `llama-bXXXX-bin-win-avx2-x64.zip`) from [github.com/ggerganov/llama.cpp/releases](https://github.com/ggerganov/llama.cpp/releases).

2. **Extract**:
   Extract `llama-server.exe` to a known directory (e.g., `C:\llama.cpp\`).

3. **Download a Quantized GGUF Model**:
   Download a GGUF model, for example from Hugging Face:
   - `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`

4. **Launch Server**:
   ```powershell
   .\llama-server.exe -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --port 8080 -c 2048 -ngl 0
   ```
   `-ngl 0` forces pure CPU execution on machines without a dedicated GPU.

---

## 3. Defining a Benchmark Experiment Manifest

An experiment manifest encapsulates all runtime parameters, workload prompts, repetition controls, and failure policies in a reproducible JSON specification:

```python
from llm_ii_engine.benchmark.experiment import (
    BenchmarkExperimentManifest,
    STANDARD_PILOT_SUITE,
)
from llm_ii_engine.core.enums import ExecutionMode, FailurePolicy, QuantizationType

manifest = BenchmarkExperimentManifest.create_standard_pilot(
    model_id="qwen2.5:0.5b",
    runtime_name="ollama",
    quantization=QuantizationType.GGUF_Q4_K_M,
    device_name="CPU",
    cpu_model="Intel Core i7-14700",
    host_ram_gb=32.0,
    warmup_repetitions=1,
    measured_repetitions=3,
    execution_mode=ExecutionMode.REAL,
)

# Save manifest to disk
manifest.save("storage/experiments/pilot_qwen_cpu.json")
```

---

## 4. Executing the Experiment and Exporting Datasets

Execute the experiment using [`ExperimentRunner`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/llm_ii_engine/benchmark/experiment.py) and export the results to JSONL or CSV:

```python
import asyncio
from llm_ii_engine.adapters.ollama import OllamaAdapter
from llm_ii_engine.benchmark.experiment import BenchmarkExperimentManifest, ExperimentRunner
from llm_ii_engine.benchmark.dataset_export import EmpiricalDatasetExporter
from llm_ii_engine.benchmark.repeatability import RepeatabilityAnalyzer
from llm_ii_engine.core.schemas import ModelSpec

async def run_benchmark():
    # 1. Load manifest
    manifest = BenchmarkExperimentManifest.load("storage/experiments/pilot_qwen_cpu.json")

    # 2. Configure adapter and model spec
    adapter = OllamaAdapter(base_url="http://localhost:11434")
    model_spec = ModelSpec(
        model_id=manifest.model_id,
        parameter_count_billion=0.5,
        quantization=manifest.quantization,
    )

    # 3. Execute experiment
    runner = ExperimentRunner(adapter=adapter, model_spec=model_spec)
    result = await runner.run_experiment(manifest)

    # 4. Export empirical dataset
    records = EmpiricalDatasetExporter.from_experiment_result(result)
    
    # Export clean measured records (warmup and failed runs excluded from primary dataset)
    EmpiricalDatasetExporter.export_to_jsonl(records, "storage/datasets/pilot_qwen.jsonl")
    EmpiricalDatasetExporter.export_to_csv(records, "storage/datasets/pilot_qwen.csv")

    # Separately preserve any failed runs for reliability auditing
    EmpiricalDatasetExporter.export_failed_runs_to_jsonl(records, "storage/datasets/pilot_qwen_failures.jsonl")

    # 5. Analyze repeatability and measurement variability
    report = RepeatabilityAnalyzer.analyze_experiment(result)
    print(f"Status: {result.status}")
    print(f"Duration CV: {report.average_duration_cv_percent}%")
    print(f"Warmup Cold-Start Penalty: {report.average_warmup_penalty_percent}%")

asyncio.run(run_benchmark())
```

---

## 5. Ingesting into Prediction Calibration

The exported dataset is fully compatible with [`DatasetBuilder`](file:///c:/Workspace/Projects/LLM%20Inference%20Intelligent%20Engine/llm_ii_engine/prediction/dataset.py):

```python
from llm_ii_engine.prediction.dataset import DatasetBuilder
from llm_ii_engine.prediction.evaluation import DataSplitter

# Ingest empirical records into typed prediction records
pred_dataset = DatasetBuilder.from_empirical_observations(
    observations=records,
    model_spec=model_spec,
)

# Partition with grouping by experiment_id to prevent repetition leakage
train_records, test_records = DataSplitter.grouped_split(
    records=pred_dataset.records,
    group_key="experiment_id",
    test_ratio=0.25,
)
```
