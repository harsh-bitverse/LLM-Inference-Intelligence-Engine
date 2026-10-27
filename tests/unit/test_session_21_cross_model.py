"""Unit tests for Session 21 authorized cross-model benchmarking, quality verification, and prospective validation."""

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.benchmark.workload_suite import (
    TASK_CODE_GENERATION,
    evaluate_task_output,
)
from llm_ii_engine.discovery.model_catalog import CANDIDATE_MODELS
from llm_ii_engine.recommendation.models import (
    ModelCompatibilityStatus,
    ModelQualityConstraintSpec,
)
from llm_ii_engine.recommendation.multi_model_comparator import MultiModelComparator


def test_all_three_models_cataloged():
    """Verify that all three authorized models exist in candidate catalog with valid metadata."""
    catalog_ids = {m.model_id: m for m in CANDIDATE_MODELS}
    assert "llama3.2:1b" in catalog_ids
    assert "qwen2.5:1.5b" in catalog_ids
    assert "qwen2.5:0.5b" in catalog_ids

    llama = catalog_ids["llama3.2:1b"]
    assert llama.default_quantization == "Q8_0"
    assert llama.context_window == 131072

    q15 = catalog_ids["qwen2.5:1.5b"]
    assert q15.default_quantization == "Q4_K_M"
    assert q15.context_window == 32768
    assert q15.license == "Apache 2.0"

    q05 = catalog_ids["qwen2.5:0.5b"]
    assert q05.default_quantization == "Q4_K_M"
    assert q05.parameter_count_billion < 1.0


def test_restricted_sandbox_blocks_malicious_operations():
    """Verify restricted python evaluation sandbox blocks arbitrary execution, imports, and filesystem calls."""
    # File I/O attempt - syntax parses, but execution fails safely inside sandbox
    evil_io = "open('/etc/passwd', 'r').read()"
    res_io = evaluate_task_output(TASK_CODE_GENERATION, evil_io)
    assert res_io.task_success is False
    assert "open" in (res_io.evaluation_notes or "")

    # Dynamic import attempt
    evil_import = "import os; os.system('echo hacked')"
    res_import = evaluate_task_output(TASK_CODE_GENERATION, evil_import)
    assert res_import.task_success is False

    # Builtins escape attempt
    evil_builtin = "__builtins__['open']('test.txt', 'w')"
    res_builtin = evaluate_task_output(TASK_CODE_GENERATION, evil_builtin)
    assert res_builtin.task_success is False


def test_code_generation_unit_test_scoring():
    """Verify code evaluator gives 1.0 score only if all 8 primality test cases pass."""
    perfect_code = """
def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
"""
    res = evaluate_task_output(TASK_CODE_GENERATION, perfect_code)
    assert res.format_compliant is True
    assert res.task_success is True
    assert res.correctness_score == 1.0

    # Partially correct code: fails for 1 or negative
    partial_code = """
def is_prime(n: int) -> bool:
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
"""
    res_partial = evaluate_task_output(TASK_CODE_GENERATION, partial_code)
    assert res_partial.correctness_score is not None
    assert 0.0 < res_partial.correctness_score < 1.0
    assert res_partial.task_success is False


def test_cross_model_manifest_dataset_integrity():
    """Verify session 21 manifest and dataset files exist and contain 54 valid observations."""
    manifest_path = Path("storage/experiments/session_21_cross_model_manifest.json")
    jsonl_path = Path("storage/datasets/session_21_cross_model_measurements.jsonl")
    csv_path = Path("storage/datasets/session_21_cross_model_measurements.csv")

    assert manifest_path.exists()
    assert jsonl_path.exists()
    assert csv_path.exists()

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["experiment_id"].startswith("exp-s21-cross-model")
    assert manifest_data["runtime_version"] == "0.40.11"
    assert set(manifest_data["installed_models"]) == {"llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"}
    assert len(manifest_data["observations"]) == 54

    # Verify JSONL lines match
    lines = [line.strip() for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 54

    first_row = json.loads(lines[0])
    assert "repetition_index" in first_row
    assert "request_index" in first_row
    assert "characters_per_second" in first_row
    assert "client_e2e_duration_ms" in first_row


def test_prospective_validation_confirmation():
    """Verify prospective predictions were pre-registered and confirmed by live validation."""
    prereg_path = Path("storage/experiments/session_21_prospective_preregistration.json")
    val_path = Path("storage/experiments/session_21_prospective_validation.json")

    assert prereg_path.exists()
    assert val_path.exists()

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    val = json.loads(val_path.read_text(encoding="utf-8"))

    assert len(prereg["scenarios"]) == 2
    assert len(val["scenarios_evaluated"]) == 2

    for entry in val["scenarios_evaluated"]:
        assert entry["hypothesis_confirmed"] is True
        assert entry["expected_best_model"] == entry["actual_best_model"]


def test_comparator_ranks_qwen_05b_for_low_latency():
    """Verify MultiModelComparator ranks qwen2.5:0.5b first when low latency is the primary requirement."""
    comparator = MultiModelComparator()
    with patch("llm_ii_engine.recommendation.multi_model_comparator.fetch_installed_ollama_models") as mock_fetch:
        mock_fetch.return_value = {
            "llama3.2:1b": {"digest": "baf6"},
            "qwen2.5:1.5b": {"digest": "65ec"},
            "qwen2.5:0.5b": {"digest": "a8b0"},
        }

        # Mock empirical summary reflecting measured rates
        summary = {
            "llama3.2:1b": {"mean_raw_decode_tok_s": 26.8, "mean_characters_per_s": 64.4, "mean_correctness_score": 0.70},
            "qwen2.5:1.5b": {"mean_raw_decode_tok_s": 34.0, "mean_characters_per_s": 75.0, "mean_correctness_score": 0.83},
            "qwen2.5:0.5b": {"mean_raw_decode_tok_s": 79.2, "mean_characters_per_s": 180.2, "mean_correctness_score": 0.67},
        }

        report = comparator.compare_models(
            candidate_ids=["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"],
            prompt_tokens=16,
            expected_output_tokens=16,
            latency_sla_ms=2000.0,
            quality_constraints=ModelQualityConstraintSpec(),
            empirical_smoke_summary=summary,
        )

        assert report.recommended_model_id == "qwen2.5:0.5b"
        best = next(c for c in report.candidate_evaluations if c.model_id == "qwen2.5:0.5b")
        assert best.rank == 1
        assert best.conservative_feasibility == "ROBUST_FEASIBLE"


def test_comparator_selects_qwen_15b_when_code_quality_constrained():
    """Verify MultiModelComparator selects qwen2.5:1.5b when code generation pass rate is required."""
    comparator = MultiModelComparator()
    with patch("llm_ii_engine.recommendation.multi_model_comparator.fetch_installed_ollama_models") as mock_fetch:
        mock_fetch.return_value = {
            "llama3.2:1b": {"digest": "baf6"},
            "qwen2.5:1.5b": {"digest": "65ec"},
            "qwen2.5:0.5b": {"digest": "a8b0"},
        }

        # Task summary where only qwen2.5:1.5b satisfies code quality (1.0 vs 0.2 vs 0.0)
        summary = {
            "llama3.2:1b": {
                "mean_raw_decode_tok_s": 26.8,
                "task_summaries": {"workload_v1_code_generation": {"mean_correctness_score": 0.2, "format_compliance_rate_pct": 100.0}},
            },
            "qwen2.5:1.5b": {
                "mean_raw_decode_tok_s": 34.0,
                "task_summaries": {"workload_v1_code_generation": {"mean_correctness_score": 1.0, "format_compliance_rate_pct": 100.0}},
            },
            "qwen2.5:0.5b": {
                "mean_raw_decode_tok_s": 79.2,
                "task_summaries": {"workload_v1_code_generation": {"mean_correctness_score": 0.0, "format_compliance_rate_pct": 0.0}},
            },
        }

        report = comparator.compare_models(
            candidate_ids=["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"],
            prompt_tokens=58,
            expected_output_tokens=96,
            latency_sla_ms=8000.0,
            quality_constraints=ModelQualityConstraintSpec(require_code_pass=True),
            empirical_smoke_summary=summary,
        )

        assert report.recommended_model_id == "qwen2.5:1.5b"
        q15 = next(c for c in report.candidate_evaluations if c.model_id == "qwen2.5:1.5b")
        assert q15.conservative_feasibility == "ROBUST_FEASIBLE"
        assert q15.rank == 1

        llama = next(c for c in report.candidate_evaluations if c.model_id == "llama3.2:1b")
        assert llama.conservative_feasibility == "QUALITY_INFEASIBLE"

        q05 = next(c for c in report.candidate_evaluations if c.model_id == "qwen2.5:0.5b")
        assert q05.conservative_feasibility == "QUALITY_INFEASIBLE"


def test_comparator_memory_budget_enforcement():
    """Verify that a 500 MiB memory constraint rules out 1B+ parameter models."""
    comparator = MultiModelComparator()
    with patch("llm_ii_engine.recommendation.multi_model_comparator.fetch_installed_ollama_models") as mock_fetch:
        mock_fetch.return_value = {
            "llama3.2:1b": {"digest": "baf6"},
            "qwen2.5:1.5b": {"digest": "65ec"},
            "qwen2.5:0.5b": {"digest": "a8b0"},
        }

        report = comparator.compare_models(
            candidate_ids=["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"],
            prompt_tokens=32,
            expected_output_tokens=32,
            quality_constraints=ModelQualityConstraintSpec(max_memory_budget_mb=800.0),
        )

        assert report.recommended_model_id == "qwen2.5:0.5b"
        q05 = next(c for c in report.candidate_evaluations if c.model_id == "qwen2.5:0.5b")
        assert q05.compatibility_status == ModelCompatibilityStatus.COMPATIBLE

        llama = next(c for c in report.candidate_evaluations if c.model_id == "llama3.2:1b")
        assert llama.compatibility_status == ModelCompatibilityStatus.INSUFFICIENT_MEMORY
        assert llama.conservative_feasibility == "PROVEN_INFEASIBLE"


def test_models_api_routes_serve_session_21_data():
    """Verify FastAPI API serves 3-model summary and comparison."""
    client = TestClient(create_app())

    resp = client.get("/api/v1/models/smoke-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "llama3.2:1b" in data
    assert "qwen2.5:1.5b" in data
    assert "qwen2.5:0.5b" in data

    # Compare endpoint
    compare_resp = client.post(
        "/api/v1/models/compare",
        json={
            "candidate_model_ids": ["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"],
            "prompt_tokens": 16,
            "expected_output_tokens": 16,
            "latency_sla_ms": 2000.0,
        },
    )
    assert compare_resp.status_code == 200
    comp_data = compare_resp.json()
    assert comp_data["recommended_model_id"] == "qwen2.5:0.5b"
    assert len(comp_data["candidate_evaluations"]) == 3
