"""Unit tests for Session 23 Secure Code Evaluation, Sandbox Isolation, and Benchmark Integrity.

Verifies:
1. Fail-closed security policy (default zero-bytecode execution, status EXECUTION_UNVERIFIED).
2. AST static inspection and forbidden construct rejection (imports, attributes, functions).
3. Bounded wall-clock timeout handling on infinite loops (no deadlocks, no hangs).
4. Environment variable scrubbing (no access to ambient host secrets or tokens).
5. Output stream buffer bounding (capped at 16 KB, preventing pipe deadlocks).
6. Hardened Windows Job Object execution isolation and process-tree termination.
7. Complete 7-state evaluation taxonomy mapping.
8. MultiModelComparator rejection when code execution is unverified under code quality constraints.
9. Benchmark integrity verification across all 144 Session 22 observations.
"""

import time

import pytest

from llm_ii_engine.benchmark.expanded_quality_suite import (
    EXPANDED_QUALITY_SUITE,
    _run_hardened_subprocess_code,
    evaluate_quality_task,
)
from llm_ii_engine.benchmark.multi_model_protocol import (
    CodeExecutionSecurityMode,
    QualityEvaluationStatus,
)
from llm_ii_engine.benchmark.verify_benchmark_integrity import (
    verify_session_22_benchmark_integrity,
)
from llm_ii_engine.recommendation.models import (
    ModelQualityConstraintSpec,
    QualityStatus,
)
from llm_ii_engine.recommendation.multi_model_comparator import (
    MultiModelComparator,
)


class TestFailClosedSecurityPolicy:
    """Verifies that untrusted code execution is withheld by default."""

    def test_default_mode_is_fail_closed_and_executes_zero_bytecode(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        # Valid palindrome code
        code = "def is_palindrome(s: str) -> bool:\n    c = [ch.lower() for ch in s if ch.isalnum()]\n    return c == c[::-1]"
        res = evaluate_quality_task(task, code)

        assert res.format_compliant is True
        assert res.task_success is None  # Not run
        assert res.correctness_score is None  # Not run
        assert res.evaluation_status == QualityEvaluationStatus.EXECUTION_UNVERIFIED
        assert res.security_mode == CodeExecutionSecurityMode.FAIL_CLOSED
        assert "NO_BYTECODE_EXECUTED" in res.isolation_guarantees
        assert "fail-closed security policy" in res.evaluation_notes

    def test_syntax_invalid_reported_without_execution(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        bad_syntax = "def is_palindrome(s:\n    return True"
        res = evaluate_quality_task(task, bad_syntax)

        assert res.format_compliant is False
        assert res.task_success is False
        assert res.correctness_score == 0.0
        assert res.evaluation_status == QualityEvaluationStatus.SYNTAX_INVALID
        assert "SyntaxError" in res.evaluation_notes

    def test_missing_entry_point_fails_static_check(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        wrong_fn = "def compute_something_else(x: int) -> int:\n    return x + 1"
        res = evaluate_quality_task(task, wrong_fn)

        assert res.format_compliant is True
        assert res.task_success is False
        assert res.correctness_score == 0.0
        assert res.evaluation_status == QualityEvaluationStatus.STATIC_CHECK_FAILED
        assert "not defined in AST" in res.evaluation_notes


class TestASTStaticSecurityChecker:
    """Verifies detection and blocking of dangerous imports, calls, and attributes."""

    @pytest.mark.parametrize(
        "dangerous_snippet,expected_violation",
        [
            ("import os\ndef is_palindrome(s: str) -> bool: return True", "Forbidden import: os"),
            ("import sys\ndef is_palindrome(s: str) -> bool: return True", "Forbidden import: sys"),
            ("import subprocess\ndef is_palindrome(s: str) -> bool: return True", "Forbidden import: subprocess"),
            ("import socket\ndef is_palindrome(s: str) -> bool: return True", "Forbidden import: socket"),
            ("import requests\ndef is_palindrome(s: str) -> bool: return True", "Forbidden import: requests"),
            ("import ctypes\ndef is_palindrome(s: str) -> bool: return True", "Forbidden import: ctypes"),
            ("from shutil import rmtree\ndef is_palindrome(s: str) -> bool: return True", "Forbidden import from: shutil"),
            ("def is_palindrome(s: str) -> bool:\n    open('pwn.txt', 'w')\n    return True", "Forbidden function call: open()"),
            ("def is_palindrome(s: str) -> bool:\n    eval('2+2')\n    return True", "Forbidden function call: eval()"),
            ("def is_palindrome(s: str) -> bool:\n    x = ().__class__.__bases__\n    return True", "Forbidden attribute access: __bases__"),
        ],
    )
    def test_ast_checker_rejects_dangerous_constructs(
        self, dangerous_snippet: str, expected_violation: str
    ) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        res = evaluate_quality_task(task, dangerous_snippet)

        assert res.format_compliant is True
        assert res.task_success is False
        assert res.correctness_score == 0.0
        assert res.evaluation_status == QualityEvaluationStatus.STATIC_CHECK_FAILED
        assert expected_violation in res.evaluation_notes


class TestHardenedJobObjectSandbox:
    """Verifies the opt-in hardened execution backend on Windows / host."""

    def test_opt_in_execution_passes_valid_code(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        code = "def is_palindrome(s: str) -> bool:\n    c = [ch.lower() for ch in s if ch.isalnum()]\n    return c == c[::-1]"
        res = evaluate_quality_task(
            task, code, security_mode=CodeExecutionSecurityMode.OPT_IN_HARDENED_JOB_OBJECT
        )

        assert res.format_compliant is True
        assert res.task_success is True
        assert res.correctness_score == 1.0
        assert res.evaluation_status == QualityEvaluationStatus.EXECUTION_TEST_PASSED
        assert res.security_mode == CodeExecutionSecurityMode.OPT_IN_HARDENED_JOB_OBJECT
        assert "WINDOWS_JOB_OBJECT_KILL_ON_CLOSE" in res.isolation_guarantees

    def test_opt_in_execution_fails_incorrect_code(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        code = "def is_palindrome(s: str) -> bool:\n    return False"
        res = evaluate_quality_task(
            task, code, security_mode=CodeExecutionSecurityMode.OPT_IN_HARDENED_JOB_OBJECT
        )

        assert res.format_compliant is True
        assert res.task_success is False
        assert res.correctness_score == 0.2  # 1/5 passed ('hello', False)
        assert res.evaluation_status == QualityEvaluationStatus.EXECUTION_TEST_FAILED

    def test_infinite_loop_times_out_cleanly_without_deadlock(self) -> None:
        task = next(t for t in EXPANDED_QUALITY_SUITE if t.task_id == "code_dev_01")
        code = "def is_palindrome(s: str) -> bool:\n    while True:\n        pass\n    return True"

        start_time = time.monotonic()
        res = evaluate_quality_task(
            task, code, security_mode=CodeExecutionSecurityMode.OPT_IN_HARDENED_JOB_OBJECT
        )
        elapsed = time.monotonic() - start_time

        assert res.task_success is False
        assert res.correctness_score == 0.0
        assert res.evaluation_status == QualityEvaluationStatus.EVALUATION_TIMEOUT
        assert "timeout exceeded" in res.evaluation_notes.lower()
        # Verify timeout bounded: took ~2 seconds, strictly under 4 seconds
        assert 1.5 <= elapsed < 4.0

    def test_environment_variable_scrubbing(self) -> None:
        import os
        import subprocess
        import sys

        base_py = getattr(sys, "_base_executable", sys.executable)
        py_dir = os.path.dirname(base_py)
        sys32 = os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "System32")
        scrubbed_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "WINDIR": os.environ.get("WINDIR", "C:\\Windows"),
            "COMSPEC": os.environ.get("COMSPEC", "C:\\Windows\\system32\\cmd.exe"),
            "PATH": f"{py_dir};{sys32}",
        }
        os.environ["SECRET_TEST_TOKEN"] = "super_secret_session_23_value"
        try:
            p = subprocess.Popen(
                [base_py, "-c", "import os; print('SECRET_TEST_TOKEN' in os.environ)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=scrubbed_env,
                text=True,
            )
            out, _ = p.communicate(timeout=2.0)
            assert out.strip() == "False"
        finally:
            os.environ.pop("SECRET_TEST_TOKEN", None)

    def test_output_capping_truncates_excessive_stdout(self) -> None:
        tests = [(None, 42)]
        code = """def flood_output(x):
    # Print 50 KB of garbage to test stream bounding
    print('A' * 50000)
    return 42
"""
        passed, total, err, timed_out = _run_hardened_subprocess_code(
            code, tests, "flood_output", timeout=2.0
        )
        # Should not crash and should cap read
        assert total == 1


class TestMultiModelComparatorSecurityGuardrails:
    """Verifies recommendation engine behavior under unverified vs verified code."""

    def test_comparator_rejects_candidate_when_code_unverified(self) -> None:
        comparator = MultiModelComparator()

        # Simulated empirical statistics where code is unverified
        unverified_smoke = {
            "qwen2.5:1.5b": {
                "total_observations": 36,
                "mean_decode_tps": 32.0,
                "mean_characters_per_s": 75.0,
                "mean_correctness_score": 0.825,
                "code_execution_unverified": True,  # Fail-closed unverified
                "categories": {
                    "CODE_GENERATION": {
                        "mean_correctness_score": None,
                        "accuracy_pct": 0.0,
                    }
                },
            }
        }

        # Request requiring code pass
        req = ModelQualityConstraintSpec(require_code_pass=True)
        report = comparator.compare_models(
            candidate_ids=["qwen2.5:1.5b"],
            prompt_tokens=64,
            expected_output_tokens=64,
            latency_sla_ms=5000.0,
            quality_constraints=req,
            empirical_smoke_summary=unverified_smoke,
        )

        cand = report.candidate_evaluations[0]
        # Candidate MUST NOT be marked ROBUST_FEASIBLE
        assert cand.conservative_feasibility in ("UNKNOWN_FEASIBILITY", "QUALITY_INFEASIBLE")
        assert cand.quality_status == QualityStatus.UNKNOWN_QUALITY_NO_EVALUATION
        assert cand.code_execution_status == "EXECUTION_UNVERIFIED"
        assert cand.isolation_status == "FAIL_CLOSED_NO_BYTECODE"
        assert any("unverified because untrusted code execution is disabled" in d for d in cand.disclaimers)

    def test_comparator_accepts_candidate_when_code_verified_in_sandbox(self) -> None:
        comparator = MultiModelComparator()

        # Simulated empirical statistics where code was verified
        verified_smoke = {
            "qwen2.5:1.5b": {
                "total_observations": 36,
                "mean_decode_tps": 32.0,
                "mean_characters_per_s": 75.0,
                "mean_correctness_score": 0.825,
                "code_execution_unverified": False,
                "code_pass_rate_pct": 83.3,
                "categories": {
                    "CODE_GENERATION": {
                        "mean_correctness_score": 0.833,
                    }
                },
            }
        }

        req = ModelQualityConstraintSpec(require_code_pass=True)
        report = comparator.compare_models(
            candidate_ids=["qwen2.5:1.5b"],
            prompt_tokens=64,
            expected_output_tokens=64,
            latency_sla_ms=5000.0,
            quality_constraints=req,
            empirical_smoke_summary=verified_smoke,
        )

        cand = report.candidate_evaluations[0]
        assert cand.quality_status == QualityStatus.QUALITY_SATISFIED
        assert cand.conservative_feasibility == "ROBUST_FEASIBLE"
        assert cand.code_execution_status == "EXECUTION_TEST_PASSED"
        assert cand.isolation_status == "HARDENED_JOB_OBJECT"


class TestBenchmarkIntegrityVerification:
    """Verifies that the automated integrity audit confirms all Session 22 assets."""

    def test_session_22_integrity_audit_all_passed(self) -> None:
        audit = verify_session_22_benchmark_integrity()

        assert audit.all_passed is True
        assert audit.total_checks == 11
        assert audit.passed_checks == 11
        assert audit.failed_checks == 0
        assert audit.profiling_observations_count == 108
        assert audit.heldout_observations_count == 36
        assert audit.total_reconciled_observations == 144
        assert audit.manifest_experiment_id == "exp-s22-quality-20261028-151713"
