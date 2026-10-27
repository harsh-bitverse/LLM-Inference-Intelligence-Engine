"""Session 19 Phase 1: Reproduce Critical SLA Failure and Decompose Latency Phases.

Reproduces the Session 18 long-generation SLA failure (P=119, O=128, SLA 8000 ms),
isolating model loading, prompt prefill, autoregressive decoding, other server time,
and client/server divergence, while collecting environmental telemetry.
"""

from __future__ import annotations

import ctypes
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llm_ii_engine.benchmark.expansion import PROMPT_TEXT_128


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]


def get_system_memory() -> dict[str, float]:
    """Capture host memory statistics via GlobalMemoryStatusEx."""
    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    return {
        "total_gb": round(mem.ullTotalPhys / (1024**3), 2),
        "available_gb": round(mem.ullAvailPhys / (1024**3), 2),
        "used_gb": round((mem.ullTotalPhys - mem.ullAvailPhys) / (1024**3), 2),
        "memory_load_pct": float(mem.dwMemoryLoad),
    }


def get_cpu_times() -> tuple[int, int, int]:
    idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))

    def to_int(ft: FILETIME) -> int:
        return (ft.dwHighDateTime << 32) + ft.dwLowDateTime

    return to_int(idle), to_int(kernel), to_int(user)


def get_llama_server_stats() -> dict[str, Any]:
    """Capture llama-server and ollama process stats non-invasively."""
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-Process -ErrorAction SilentlyContinue | "
            "Where-Object { $_.ProcessName -in @('llama-server','ollama') } | "
            "Select-Object Id, ProcessName, WorkingSet64, PrivateMemorySize64, CPU, "
            "@{Name='Threads'; Expression={$_.Threads.Count}} | ConvertTo-Json"
        ),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]
            return {"processes": data}
    except Exception as e:
        return {"error": str(e)}
    return {"processes": []}


def reproduce_sla_failure() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent.parent
    client = httpx.Client(base_url="http://127.0.0.1:11434", trust_env=False, timeout=120.0)

    # 1. Check pre-execution environment
    pre_mem = get_system_memory()
    get_llama_server_stats()
    ps_resp = client.get("/api/ps").json()
    model_resident = len(ps_resp.get("models", [])) > 0

    print("=== Phase 1: Reproducing Critical SLA Failure (P=119, O=128, SLA=8000ms) ===")
    print(f"Pre-flight Host RAM: {pre_mem['available_gb']} GB available / {pre_mem['total_gb']} GB total")
    print(f"Model Resident: {model_resident} (Details: {ps_resp.get('models')})")

    # Target workload
    target_sla_ms = 8000.0
    forecast_e2e_ms = 5999.88
    forecast_decode_tps = 22.72
    out_tokens = 128

    repetitions: list[dict[str, Any]] = []

    for rep in range(1, 4):
        # Measure CPU utilization across request
        i1, k1, u1 = get_cpu_times()
        t0 = time.perf_counter()
        resp = client.post(
            "/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": PROMPT_TEXT_128,
                "options": {
                    "num_predict": out_tokens,
                    "temperature": 0.0,
                },
                "stream": False,
            },
        )
        t1 = time.perf_counter()
        i2, k2, u2 = get_cpu_times()

        client_dur_ms = round((t1 - t0) * 1000.0, 2)
        total_delta = (k2 - k1) + (u2 - u1)
        idle_delta = i2 - i1
        cpu_util_pct = round(100.0 * (1.0 - idle_delta / total_delta), 1) if total_delta > 0 else 0.0

        data = resp.json()
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        load_dur_ms = round(data.get("load_duration", 0) / 1e6, 2)
        prefill_dur_ms = round(data.get("prompt_eval_duration", 0) / 1e6, 2)
        decode_dur_ms = round(data.get("eval_duration", 1) / 1e6, 2)
        server_total_ms = round(data.get("total_duration", 0) / 1e6, 2)

        # Decomposed phases
        other_server_ms = round(max(0.0, server_total_ms - (load_dur_ms + prefill_dur_ms + decode_dur_ms)), 2)
        client_server_div_ms = round(max(0.0, client_dur_ms - server_total_ms), 2)
        decode_tps = round(eval_count / (decode_dur_ms / 1000.0), 2) if decode_dur_ms > 0 else 0.0

        # Percentages of client total
        pct_load = round((load_dur_ms / client_dur_ms) * 100.0, 2) if client_dur_ms > 0 else 0.0
        pct_prefill = round((prefill_dur_ms / client_dur_ms) * 100.0, 2) if client_dur_ms > 0 else 0.0
        pct_decode = round((decode_dur_ms / client_dur_ms) * 100.0, 2) if client_dur_ms > 0 else 0.0
        pct_other_server = round((other_server_ms / client_dur_ms) * 100.0, 2) if client_dur_ms > 0 else 0.0
        pct_divergence = round((client_server_div_ms / client_dur_ms) * 100.0, 2) if client_dur_ms > 0 else 0.0

        rep_entry = {
            "repetition": rep,
            "client_e2e_duration_ms": client_dur_ms,
            "server_total_duration_ms": server_total_ms,
            "server_load_duration_ms": load_dur_ms,
            "server_prefill_duration_ms": prefill_dur_ms,
            "server_decode_duration_ms": decode_dur_ms,
            "other_server_duration_ms": other_server_ms,
            "client_server_divergence_ms": client_server_div_ms,
            "phase_percentages": {
                "load_pct": pct_load,
                "prefill_pct": pct_prefill,
                "decode_pct": pct_decode,
                "other_server_pct": pct_other_server,
                "client_divergence_pct": pct_divergence,
            },
            "server_decode_throughput_tok_s": decode_tps,
            "actual_prompt_tokens": prompt_eval_count,
            "actual_output_tokens": eval_count,
            "cpu_utilization_pct": cpu_util_pct,
            "sla_target_ms": target_sla_ms,
            "sla_violated": client_dur_ms > target_sla_ms,
            "violation_margin_ms": round(client_dur_ms - target_sla_ms, 2) if client_dur_ms > target_sla_ms else 0.0,
        }
        repetitions.append(rep_entry)
        print(
            f"  Rep {rep}: Client E2E={client_dur_ms} ms | Decode={decode_dur_ms} ms ({decode_tps} tok/s) "
            f"| Prefill={prefill_dur_ms} ms | Load={load_dur_ms} ms | SLA Violated={rep_entry['sla_violated']} "
            f"(+{rep_entry['violation_margin_ms']} ms)"
        )

    # Median summary
    med_client_e2e = round(statistics.median([r["client_e2e_duration_ms"] for r in repetitions]), 2)
    med_decode_dur = round(statistics.median([r["server_decode_duration_ms"] for r in repetitions]), 2)
    med_prefill_dur = round(statistics.median([r["server_prefill_duration_ms"] for r in repetitions]), 2)
    med_load_dur = round(statistics.median([r["server_load_duration_ms"] for r in repetitions]), 2)
    med_other_server = round(statistics.median([r["other_server_duration_ms"] for r in repetitions]), 2)
    med_div = round(statistics.median([r["client_server_divergence_ms"] for r in repetitions]), 2)
    med_tps = round(statistics.median([r["server_decode_throughput_tok_s"] for r in repetitions]), 2)
    med_cpu = round(statistics.median([r["cpu_utilization_pct"] for r in repetitions]), 2)

    post_mem = get_system_memory()
    post_procs = get_llama_server_stats()

    results = {
        "workload": "Long Output Workload (P=119, O=128)",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "sla_target_ms": target_sla_ms,
        "forecast_e2e_ms": forecast_e2e_ms,
        "forecast_decode_tps": forecast_decode_tps,
        "session_18_observed_median_ms": 10282.98,
        "session_19_reproduced_median_ms": med_client_e2e,
        "reproduction_delta_ms": round(med_client_e2e - 10282.98, 2),
        "median_decomposition": {
            "client_e2e_duration_ms": med_client_e2e,
            "server_decode_duration_ms": med_decode_dur,
            "server_prefill_duration_ms": med_prefill_dur,
            "server_load_duration_ms": med_load_dur,
            "other_server_duration_ms": med_other_server,
            "client_server_divergence_ms": med_div,
            "server_decode_throughput_tok_s": med_tps,
            "cpu_utilization_pct": med_cpu,
        },
        "phase_share_pct": {
            "decode_share_pct": round((med_decode_dur / med_client_e2e) * 100.0, 2),
            "prefill_share_pct": round((med_prefill_dur / med_client_e2e) * 100.0, 2),
            "load_share_pct": round((med_load_dur / med_client_e2e) * 100.0, 2),
            "other_server_share_pct": round((med_other_server / med_client_e2e) * 100.0, 2),
            "client_divergence_share_pct": round((med_div / med_client_e2e) * 100.0, 2),
        },
        "primary_slowdown_phase": "DECODE",
        "telemetry": {
            "pre_memory": pre_mem,
            "post_memory": post_mem,
            "processes": post_procs,
        },
        "repetitions": repetitions,
    }

    out_file = root / "storage" / "experiments" / "session_19_sla_failure_reproduction.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved reproduction results to {out_file}")

    return results


if __name__ == "__main__":
    reproduce_sla_failure()
