"""Unit tests for Session 22 Expanded Objective Quality Suite and Multi-Model Comparator.

Tests:
1. Suite specification, partitioning, and SHA-256 hash integrity.
2. AST sandbox security and timeout enforcement.
3. Category evaluators (JSON semantic, numerical, code execution, retrieval, long-form compliance).
4. Data separation (zero overlap across DEV, SELECTION, and HELD_OUT).
5. MultiModelComparator category filtering, constraint enforcement, and unmeasured quality rejection.
"""

from unittest.mock import patch

import pytest

from llm_ii_engine.benchmark.expanded_quality_suite import (
    DEV_TASKS,
    EXPANDED_QUALITY_SUITE,
    HELDOUT_TASKS,
    SELECTION_TASKS,
    QualityTaskCategory,
    _eval_code,
    compute_suite_hashes,
    evaluate_quality_task,
)
from llm_ii_engine.benchmark.multi_model_protocol import (
    ModelInstallationStatus,
)
from llm_ii_engine.recommendation.models import (
    ModelQualityConstraintSpec,
    QualityStatus,
)
from llm_ii_engine.recommendation.multi_model_comparator import (
    MultiModelComparator,
)


class TestExpandedQualitySuiteIntegrity:
    """Tests for suite specification, split sizes, and SHA-256 cryptographic hashes."""

    def test_partition_counts(self) -> None:
        assert len(EXPANDED_QUALITY_SUITE) == 36
        assert len(DEV_TASKS) == 6
        assert len(SELECTION_TASKS) == 18
        assert len(HELDOUT_TASKS) == 12

    def test_category_distribution(self) -> None:
        categories = [t.category for t in EXPANDED_QUALITY_SUITE]
        for cat in QualityTaskCategory:
            count = sum(1 for c in categories if c == cat)
            assert count == 6, f"Expected 6 tasks for category {cat.value}, found {count}"

    def test_strict_partition_separation_zero_leakage(self) -> None:
        dev_ids = {t.task_id for t in DEV_TASKS}
        sel_ids = {t.task_id for t in SELECTION_TASKS}
        held_ids = {t.task_id for t in HELDOUT_TASKS}

        assert len(dev_ids.intersection(sel_ids)) == 0, "Overlap between DEV and SELECTION"
        assert len(dev_ids.intersection(held_ids)) == 0, "Overlap between DEV and HELD_OUT"
        assert len(sel_ids.intersection(held_ids)) == 0, "Overlap between SELECTION and HELD_OUT"

        dev_prompts = {t.prompt for t in DEV_TASKS}
        sel_prompts = {t.prompt for t in SELECTION_TASKS}
        held_prompts = {t.prompt for t in HELDOUT_TASKS}

        assert len(dev_prompts.intersection(sel_prompts)) == 0, "Prompt overlap between DEV and SELECTION"
        assert len(dev_prompts.intersection(held_prompts)) == 0, "Prompt overlap between DEV and HELD_OUT"
        assert len(sel_prompts.intersection(held_prompts)) == 0, "Prompt overlap between SELECTION and HELD_OUT"

    def test_suite_hashes_reproducibility(self) -> None:
        hashes = compute_suite_hashes()
        assert hashes["total_task_count"] == "36"
        assert hashes["dev_count"] == "6"
        assert hashes["selection_count"] == "18"
        assert hashes["heldout_count"] == "12"
        assert len(hashes["suite_hash"]) == 64
        assert len(hashes["dev_hash"]) == 64
        assert len(hashes["selection_hash"]) == 64
        assert len(hashes["heldout_hash"]) == 64


class TestASTSandboxSecurity:
    """Tests security boundaries, disallowed builtins, and timeout enforcement."""

    def test_sandbox_executes_safe_code(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        code = "def is_palindrome(s: str) -> bool:\n    c = [ch.lower() for ch in s if ch.isalnum()]\n    return c == c[::-1]"
        res = _eval_code(task, code)
        assert res.task_success is True
        assert res.correctness_score == 1.0
        assert "Unit tests passed" in res.evaluation_notes

    def test_sandbox_blocks_disallowed_builtins(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        code = "def is_palindrome(s: str) -> bool:\n    open('test.txt', 'w')\n    return True"
        res = _eval_code(task, code)
        assert res.task_success is False
        assert res.correctness_score == 0.0

    def test_sandbox_handles_syntax_error(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        code = "def is_palindrome(s:\n    return True"
        res = _eval_code(task, code)
        assert res.format_compliant is False
        assert "SyntaxError" in res.evaluation_notes

    def test_sandbox_times_out_on_infinite_loop(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        code = "def is_palindrome(s: str) -> bool:\n    while True:\n        pass\n    return True"
        res = _eval_code(task, code)
        assert res.task_success is False
        assert "timeout" in res.evaluation_notes.lower()


class TestCategoryEvaluators:
    """Tests objective determinism for each evaluation category."""

    def test_structured_json_evaluator(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "json_dev_01")
        # Valid JSON matching schema and types: host: "api.internal", port: 8443, ssl_enabled: True
        valid_response = '```json\n{"host": "api.internal", "port": 8443, "ssl_enabled": true}\n```'
        res = evaluate_quality_task(task, valid_response)
        assert res.format_compliant is True
        assert res.correctness_score == 1.0
        assert res.task_success is True

        # Malformed JSON
        invalid_response = '{"host": "api.internal", "port": 8443, ssl_enabled: true}'
        res_bad = evaluate_quality_task(task, invalid_response)
        assert res_bad.format_compliant is False
        assert res_bad.correctness_score == 0.0

    def test_numerical_reasoning_evaluator(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "num_dev_01")
        # Pattern expects Profit: $? 130
        good_response = "After deducting all operating expenses, the net Profit: $130 for the month."
        res = evaluate_quality_task(task, good_response)
        assert res.task_success is True
        assert res.correctness_score == 1.0

        bad_response = "The profit calculated was 450 dollars."
        res_bad = evaluate_quality_task(task, bad_response)
        assert res_bad.task_success is False
        assert res_bad.correctness_score == 0.0

    def test_code_generation_evaluator(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        valid_code = "```python\ndef is_palindrome(s: str) -> bool:\n    c = [ch.lower() for ch in s if ch.isalnum()]\n    return c == c[::-1]\n```"
        res = evaluate_quality_task(task, valid_code)
        assert res.format_compliant is True
        assert res.task_success is True
        assert res.correctness_score == 1.0

        failing_code = "```python\ndef is_palindrome(s: str) -> bool:\n    return False\n```"
        res_fail = evaluate_quality_task(task, failing_code)
        assert res_fail.task_success is False
        assert res_fail.correctness_score < 1.0

    def test_context_retrieval_evaluator(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "retrieval_dev_01")
        correct_resp = "The verified incident commander was Sandra Ramos at the station."
        res = evaluate_quality_task(task, correct_resp)
        assert res.task_success is True
        assert res.correctness_score == 1.0

        wrong_resp = "The incident commander was John Doe."
        res_bad = evaluate_quality_task(task, wrong_resp)
        assert res_bad.task_success is False
        assert res_bad.correctness_score == 0.0

    def test_long_form_compliance_evaluator(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "longform_dev_01")
        # Headings: ## Benefits, ## Limitations, min 40 words, max 150 words
        resp = """## Benefits
The primary advantage of deploying distributed edge inference nodes is significant latency reduction for end users across all geographic clusters while maintaining operational resilience.

## Limitations
However, distributed hardware clusters require centralized monitoring frameworks, automated failover orchestration mechanisms, and careful management of firmware update distribution across remote physical endpoints."""
        res = evaluate_quality_task(task, resp)
        assert res.format_compliant is True
        assert res.task_success is True
        assert res.correctness_score == 1.0


class TestMultiModelComparatorQualityFiltering:
    """Tests category-aware recommendation logic and unmeasured quality rejection."""

    @pytest.fixture
    def mock_manifest_summary(self) -> dict[str, dict]:
        return {
            "llama3.2:1b": {
                "total_observations": 36,
                "median_duration_ms": 2200.0,
                "mean_decode_tps": 20.35,
                "mean_characters_per_s": 52.8,
                "mean_correctness_score": 0.669,
                "categories": {
                    "CODE_GENERATION": {"mean_correctness_score": 0.933, "accuracy_pct": 83.3},
                    "STRUCTURED_JSON": {"mean_correctness_score": 0.500, "format_compliance_pct": 50.0},
                    "LONG_FORM_COMPLIANCE": {"mean_correctness_score": 0.911, "accuracy_pct": 66.7},
                },
            },
            "qwen2.5:1.5b": {
                "total_observations": 36,
                "median_duration_ms": 1600.0,
                "mean_decode_tps": 24.86,
                "mean_characters_per_s": 66.8,
                "mean_correctness_score": 0.785,
                "categories": {
                    "CODE_GENERATION": {"mean_correctness_score": 1.0, "accuracy_pct": 100.0},
                    "STRUCTURED_JSON": {"mean_correctness_score": 1.0, "format_compliance_pct": 100.0},
                    "LONG_FORM_COMPLIANCE": {"mean_correctness_score": 0.875, "accuracy_pct": 50.0},
                },
            },
            "qwen2.5:0.5b": {
                "total_observations": 36,
                "median_duration_ms": 760.0,
                "mean_decode_tps": 60.31,
                "mean_characters_per_s": 142.7,
                "mean_correctness_score": 0.488,
                "categories": {
                    "CODE_GENERATION": {"mean_correctness_score": 0.0, "accuracy_pct": 0.0},
                    "STRUCTURED_JSON": {"mean_correctness_score": 1.0, "format_compliance_pct": 100.0},
                    "LONG_FORM_COMPLIANCE": {"mean_correctness_score": 0.764, "accuracy_pct": 33.3},
                },
            },
        }

    def test_rejection_of_unmeasured_model_under_quality_constraints(self) -> None:
        comparator = MultiModelComparator()
        # candidate 'llama3.2:3b' is uninstalled
        report = comparator.compare_models(
            candidate_ids=["llama3.2:3b"],
            prompt_tokens=64,
            expected_output_tokens=64,
            quality_constraints=ModelQualityConstraintSpec(require_code_pass=True),
        )
        assert len(report.candidate_evaluations) == 1
        entry = report.candidate_evaluations[0]
        assert entry.installation_status == ModelInstallationStatus.UNINSTALLED_CANDIDATE
        assert entry.quality_status == QualityStatus.UNKNOWN_QUALITY_NO_EVALUATION
        assert entry.conservative_feasibility == "UNKNOWN_FEASIBILITY"
        assert report.recommended_model_id is None

    def test_code_constraint_rejection(self, mock_manifest_summary: dict[str, dict]) -> None:
        comparator = MultiModelComparator()
        with patch(
            "llm_ii_engine.recommendation.multi_model_comparator.fetch_installed_ollama_models",
            return_value={"llama3.2:1b": {}, "qwen2.5:1.5b": {}, "qwen2.5:0.5b": {}},
        ):
            report = comparator.compare_models(
                candidate_ids=["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"],
                prompt_tokens=64,
                expected_output_tokens=64,
                quality_constraints=ModelQualityConstraintSpec(require_code_pass=True),
                empirical_smoke_summary=mock_manifest_summary,
            )
            evals = {e.model_id: e for e in report.candidate_evaluations}
            assert evals["qwen2.5:0.5b"].quality_status == QualityStatus.QUALITY_VIOLATED
            assert evals["qwen2.5:0.5b"].conservative_feasibility == "QUALITY_INFEASIBLE"
            assert evals["qwen2.5:1.5b"].quality_status == QualityStatus.QUALITY_SATISFIED
            assert report.recommended_model_id == "qwen2.5:1.5b"

    def test_semantic_json_constraint(self, mock_manifest_summary: dict[str, dict]) -> None:
        comparator = MultiModelComparator()
        with patch(
            "llm_ii_engine.recommendation.multi_model_comparator.fetch_installed_ollama_models",
            return_value={"llama3.2:1b": {}, "qwen2.5:1.5b": {}, "qwen2.5:0.5b": {}},
        ):
            report = comparator.compare_models(
                candidate_ids=["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"],
                prompt_tokens=64,
                expected_output_tokens=64,
                quality_constraints=ModelQualityConstraintSpec(
                    require_json_semantic_correctness=True
                ),
                empirical_smoke_summary=mock_manifest_summary,
            )
            evals = {e.model_id: e for e in report.candidate_evaluations}
            assert evals["llama3.2:1b"].quality_status == QualityStatus.QUALITY_VIOLATED
            assert evals["llama3.2:1b"].conservative_feasibility == "QUALITY_INFEASIBLE"
            assert report.recommended_model_id == "qwen2.5:0.5b"

    def test_longform_instruction_compliance_constraint(
        self, mock_manifest_summary: dict[str, dict]
    ) -> None:
        comparator = MultiModelComparator()
        with patch(
            "llm_ii_engine.recommendation.multi_model_comparator.fetch_installed_ollama_models",
            return_value={"llama3.2:1b": {}, "qwen2.5:1.5b": {}, "qwen2.5:0.5b": {}},
        ):
            report = comparator.compare_models(
                candidate_ids=["llama3.2:1b", "qwen2.5:1.5b", "qwen2.5:0.5b"],
                prompt_tokens=64,
                expected_output_tokens=64,
                quality_constraints=ModelQualityConstraintSpec(
                    min_instruction_compliance=0.90
                ),
                empirical_smoke_summary=mock_manifest_summary,
            )
            evals = {e.model_id: e for e in report.candidate_evaluations}
            assert evals["qwen2.5:0.5b"].quality_status == QualityStatus.QUALITY_VIOLATED
            assert evals["qwen2.5:1.5b"].quality_status == QualityStatus.QUALITY_VIOLATED
            assert evals["llama3.2:1b"].quality_status == QualityStatus.QUALITY_SATISFIED
            assert report.recommended_model_id == "llama3.2:1b"
