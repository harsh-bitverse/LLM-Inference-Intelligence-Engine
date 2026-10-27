"""Unit tests for Session 20 multi-model benchmarking foundation and quality contracts."""

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_ii_engine.api.app import create_app
from llm_ii_engine.benchmark.multi_model_protocol import (
    ModelInstallationStatus,
    TaskNormalizedMetrics,
)
from llm_ii_engine.benchmark.workload_suite import (
    TASK_CODE_GENERATION,
    TASK_INTERACTIVE_QA,
    TASK_REASONING_NUMERICAL,
    TASK_STRUCTURED_JSON,
    WORKLOAD_SUITE,
    evaluate_task_output,
)
from llm_ii_engine.discovery.model_catalog import (
    get_model_catalog,
    get_recommended_authorization_shortlist,
)
from llm_ii_engine.recommendation.models import (
    ModelCompatibilityStatus,
    ModelQualityConstraintSpec,
    QualityStatus,
)
from llm_ii_engine.recommendation.multi_model_comparator import MultiModelComparator


def test_model_catalog_metadata():
    """Verify candidate models catalog has valid architectures, licensing, and resource requirements."""
    catalog = get_model_catalog()
    assert len(catalog) >= 6

    ids = [m.model_id for m in catalog]
    assert "llama3.2:1b" in ids
    assert "qwen2.5:1.5b" in ids
    assert "qwen2.5:0.5b" in ids

    for model in catalog:
        assert model.parameter_count_billion > 0.0
        assert model.expected_disk_size_mb > 0.0
        assert model.expected_ram_mb > 0.0
        assert model.context_window > 0
        assert model.tokenizer_name
        assert model.license
        assert model.download_command.startswith("ollama pull ")


def test_recommended_authorization_shortlist():
    """Verify recommended shortlist contains expected candidate peers requiring authorization."""
    shortlist = get_recommended_authorization_shortlist()
    assert len(shortlist) == 2
    shortlist_ids = [m.model_id for m in shortlist]
    assert "qwen2.5:1.5b" in shortlist_ids
    assert "qwen2.5:0.5b" in shortlist_ids
    for m in shortlist:
        assert m.license == "Apache 2.0"


def test_workload_suite_specification():
    """Verify versioned workload suite tasks are deterministic and well-specified."""
    assert len(WORKLOAD_SUITE) == 6
    task_ids = {t.task_id for t in WORKLOAD_SUITE}
    assert len(task_ids) == 6

    for task in WORKLOAD_SUITE:
        assert task.prompt
        assert task.max_output_tokens > 0
        assert task.version == "1.0.0"


def test_quality_evaluator_interactive_qa():
    """Verify interactive QA evaluator accurately validates capital of France."""
    res_correct = evaluate_task_output(TASK_INTERACTIVE_QA, "The capital of France is Paris.")
    assert res_correct.task_success is True
    assert res_correct.correctness_score == 1.0

    res_incorrect = evaluate_task_output(TASK_INTERACTIVE_QA, "The capital of France is London.")
    assert res_incorrect.task_success is False
    assert res_incorrect.correctness_score == 0.0


def test_quality_evaluator_structured_json():
    """Verify JSON schema evaluator detects valid vs invalid JSON extraction."""
    valid_json_text = json.dumps({
        "patient_id": "PT-4402",
        "age": 45,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "pulse_bpm": 72,
    })
    res_valid = evaluate_task_output(TASK_STRUCTURED_JSON, valid_json_text)
    assert res_valid.format_compliant is True
    assert res_valid.task_success is True
    assert res_valid.correctness_score == 1.0

    # Malformed JSON
    res_invalid = evaluate_task_output(TASK_STRUCTURED_JSON, "Patient is 45 years old, no JSON here.")
    assert res_invalid.format_compliant is False
    assert res_invalid.task_success is False
    assert res_invalid.correctness_score == 0.0


def test_quality_evaluator_code_generation():
    """Verify code evaluator tests python syntax and passes functional unit tests."""
    valid_code = """
def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
    res_code = evaluate_task_output(TASK_CODE_GENERATION, valid_code)
    assert res_code.format_compliant is True
    assert res_code.task_success is True
    assert res_code.correctness_score == 1.0

    # Syntax error
    broken_code = "def is_prime(n: int) return False"
    res_broken = evaluate_task_output(TASK_CODE_GENERATION, broken_code)
    assert res_broken.format_compliant is False
    assert res_broken.task_success is False
    assert res_broken.correctness_score == 0.0


def test_quality_evaluator_numerical_reasoning():
    """Verify numerical reasoning evaluator checks for expected outcome 112."""
    correct_math = "120 - 40 = 80. 80 + 60 = 140. 20% of 140 is 28. 140 - 28 = 112. Final Answer: 112"
    res_math = evaluate_task_output(TASK_REASONING_NUMERICAL, correct_math)
    assert res_math.task_success is True
    assert res_math.correctness_score == 1.0

    incorrect_math = "The final answer is 95 racks."
    res_bad = evaluate_task_output(TASK_REASONING_NUMERICAL, incorrect_math)
    assert res_bad.task_success is False
    assert res_bad.correctness_score == 0.0


def test_tokenizer_comparability_contract():
    """Verify normalized metrics flag cross_tokenizer_comparable as False with advisory."""
    metrics = TaskNormalizedMetrics(
        client_e2e_duration_ms=150.0,
        raw_prompt_tokens=32,
        raw_output_tokens=16,
        raw_decode_tokens_per_second=24.5,
        task_elapsed_ms=150.0,
        characters_generated=60,
        words_generated=12,
        characters_per_second=400.0,
        words_per_second=80.0,
        cross_tokenizer_comparable=False,
    )
    assert metrics.cross_tokenizer_comparable is False
    assert "Raw tokens/second cannot be directly compared" in metrics.tokenizer_comparability_advisory


def test_uninstalled_model_handling():
    """Verify uninstalled candidates are preserved as UNINSTALLED with UNKNOWN metrics."""
    comparator = MultiModelComparator()
    with patch("llm_ii_engine.recommendation.multi_model_comparator.fetch_installed_ollama_models") as mock_fetch:
        # Mock only llama3.2:1b as installed
        mock_fetch.return_value = {"llama3.2:1b": {"digest": "baf6a787"}}

        report = comparator.compare_models(
            candidate_ids=["llama3.2:1b", "qwen2.5:1.5b"],
            prompt_tokens=32,
            expected_output_tokens=16,
        )

        assert report.recommended_model_id == "llama3.2:1b"
        qwen_entry = next(c for c in report.candidate_evaluations if c.model_id == "qwen2.5:1.5b")
        assert qwen_entry.installation_status == ModelInstallationStatus.UNINSTALLED_CANDIDATE
        assert qwen_entry.compatibility_status == ModelCompatibilityStatus.UNINSTALLED
        assert qwen_entry.conservative_feasibility == "UNKNOWN_FEASIBILITY"
        assert qwen_entry.nominal_latency_ms is None
        assert qwen_entry.rank is None
        assert "is not installed locally" in qwen_entry.disclaimers[0]


def test_quality_constraint_enforcement():
    """Verify quality constraints enforce QUALITY_VIOLATED and QUALITY_INFEASIBLE when failed."""
    comparator = MultiModelComparator()
    with patch("llm_ii_engine.recommendation.multi_model_comparator.fetch_installed_ollama_models") as mock_fetch:
        mock_fetch.return_value = {"llama3.2:1b": {"digest": "baf6a787"}}

        # Provide smoke summary where json compliance failed
        smoke_summary = {
            "llama3.2:1b": {
                "json_compliance_rate_pct": 50.0,  # Only 50%
                "code_pass_rate_pct": 100.0,
                "mean_correctness_score": 0.5,
            }
        }

        report = comparator.compare_models(
            candidate_ids=["llama3.2:1b"],
            prompt_tokens=32,
            expected_output_tokens=16,
            quality_constraints=ModelQualityConstraintSpec(require_json_compliance=True),
            empirical_smoke_summary=smoke_summary,
        )

        entry = report.candidate_evaluations[0]
        assert entry.quality_status == QualityStatus.QUALITY_VIOLATED
        assert entry.conservative_feasibility == "QUALITY_INFEASIBLE"
        assert entry.decision_label == "REJECTED_QUALITY"


def test_memory_budget_constraint():
    """Verify models exceeding memory budget are flagged as INSUFFICIENT_MEMORY and PROVEN_INFEASIBLE."""
    comparator = MultiModelComparator()
    with patch("llm_ii_engine.recommendation.multi_model_comparator.fetch_installed_ollama_models") as mock_fetch:
        mock_fetch.return_value = {"llama3.2:1b": {"digest": "baf6a787"}}

        report = comparator.compare_models(
            candidate_ids=["llama3.2:1b"],
            prompt_tokens=32,
            expected_output_tokens=16,
            quality_constraints=ModelQualityConstraintSpec(max_memory_budget_mb=500.0),
        )

        entry = report.candidate_evaluations[0]
        assert entry.compatibility_status == ModelCompatibilityStatus.INSUFFICIENT_MEMORY
        assert entry.conservative_feasibility == "PROVEN_INFEASIBLE"
        assert report.recommended_model_id is None


def test_multi_model_api_routes(tmp_path: Path):
    """Verify REST API endpoints for catalog, workloads, and model comparison."""
    app = create_app(database_url=f"sqlite:///{tmp_path}/test_models.db")
    client = TestClient(app)

    # 1. GET /api/v1/models/catalog
    resp_cat = client.get("/api/v1/models/catalog")
    assert resp_cat.status_code == 200
    cat_data = resp_cat.json()
    assert isinstance(cat_data, list)
    assert any(m["model_id"] == "llama3.2:1b" for m in cat_data)

    # 2. GET /api/v1/models/shortlist
    resp_short = client.get("/api/v1/models/shortlist")
    assert resp_short.status_code == 200
    short_data = resp_short.json()
    assert len(short_data) == 2

    # 3. GET /api/v1/models/workloads
    resp_work = client.get("/api/v1/models/workloads")
    assert resp_work.status_code == 200
    work_data = resp_work.json()
    assert len(work_data) == 6

    # 4. POST /api/v1/models/compare
    resp_comp = client.post(
        "/api/v1/models/compare",
        json={
            "candidate_model_ids": ["llama3.2:1b", "qwen2.5:1.5b"],
            "prompt_tokens": 32,
            "expected_output_tokens": 16,
            "latency_sla_ms": 3000.0,
        },
    )
    assert resp_comp.status_code == 200
    comp_data = resp_comp.json()
    assert "candidate_evaluations" in comp_data
    assert comp_data["recommended_model_id"] in ["llama3.2:1b", "qwen2.5:1.5b"]
