"""Deterministic, versioned multi-model workload benchmark suite with automated quality evaluators.

Enforces clear separation between:
- Performance metrics (client E2E duration, prefill, decode)
- Format compliance (JSON parseability, valid Python AST syntax)
- Output correctness (factual exact match, schema field values, unit tests)
- Subjective quality (marked unassessed without human/LLM judge)
"""

import ast
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.benchmark.multi_model_protocol import (
    QualityEvaluationResult,
    TaskEvaluationType,
)


class BenchmarkWorkloadTask(BaseModel):
    """Specification of a versioned, deterministic benchmark workload task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(description="Unique task identifier, e.g. 'workload_v1_json'")
    name: str = Field(description="Descriptive task title")
    task_type: TaskEvaluationType = Field(description="Task category")
    prompt: str = Field(description="Exact prompt string sent to model")
    max_output_tokens: int = Field(gt=0, description="Server output limit (num_predict)")
    expected_prompt_words: int = Field(gt=0, description="Approximate prompt word count")
    description: str = Field(description="Purpose and what this task tests")
    version: str = Field(default="1.0.0", description="SemVer version of workload task")


# --- The 6 Authoritative Workload Tasks ---

TASK_INTERACTIVE_QA = BenchmarkWorkloadTask(
    task_id="workload_v1_interactive_qa",
    name="Interactive Question Answering",
    task_type=TaskEvaluationType.INTERACTIVE_QA,
    prompt="What is the capital of France? Reply with ONLY the city name and nothing else.",
    max_output_tokens=16,
    expected_prompt_words=13,
    description="Tests low-latency short-burst generation and simple factual recall.",
)

TASK_LONG_CONTEXT = BenchmarkWorkloadTask(
    task_id="workload_v1_long_context_retrieval",
    name="Long-Context Needle Retrieval",
    task_type=TaskEvaluationType.LONG_CONTEXT_RETRIEVAL,
    prompt=(
        "SYSTEM TELEMETRY AUDIT LOG - APEX FACILITY 42\n\n"
        "The following operational log records the automated diagnostic status of the high-altitude telemetry relay. "
        "All maintenance engineers must adhere to safety standard ISO-9001-ALPHA. "
        "Power supplies A and B are operating at 99.4% nominal efficiency. "
        "Coolant pressure is stable at 4.2 bar, with liquid helium reservoirs indicating no anomalous pressure deltas. "
        "The primary antenna array was rotated 14 degrees azimuth on 2026-10-15 to align with satellite cluster Leo-8. "
        "During calibration cycle 7, an emergency bypass verification was performed. "
        "The authorized security emergency override key for relay station 42 is ZEPHYR-7703-DELTA. "
        "Secondary sensors in sector 4 show background ionizing radiation within standard ambient baselines. "
        "Backup generator test completed successfully with 48 hours of diesel reserve available.\n\n"
        "Question: According to the audit log above, what is the authorized security emergency override key? "
        "Reply with ONLY the key and nothing else."
    ),
    max_output_tokens=24,
    expected_prompt_words=148,
    description="Tests prompt prefill evaluation speed and factual needle-in-a-haystack retrieval across extended context.",
)

TASK_LONG_FORM = BenchmarkWorkloadTask(
    task_id="workload_v1_long_form_generation",
    name="Long-Form Generation",
    task_type=TaskEvaluationType.LONG_FORM_GENERATION,
    prompt=(
        "Explain the architectural difference between prompt prefill and autoregressive token decode during transformer LLM inference. "
        "Highlight memory bandwidth intensity versus arithmetic compute intensity in exactly three bullet points."
    ),
    max_output_tokens=160,
    expected_prompt_words=29,
    description="Tests sustained autoregressive token generation speed, sustained decode throughput, and technical coherence.",
)

TASK_STRUCTURED_JSON = BenchmarkWorkloadTask(
    task_id="workload_v1_structured_json",
    name="Structured JSON Extraction",
    task_type=TaskEvaluationType.STRUCTURED_JSON,
    prompt=(
        'Extract information from this medical record text:\n'
        '"Patient record PT-4402 indicates a 45-year-old male with resting systolic blood pressure of 120 mmHg, '
        'diastolic blood pressure of 80 mmHg, and resting pulse of 72 beats per minute."\n\n'
        'Output ONLY a valid JSON object matching this schema (do NOT include markdown code fences or commentary):\n'
        '{"patient_id": "PT-4402", "age": 45, "systolic_bp": 120, "diastolic_bp": 80, "pulse_bpm": 72}'
    ),
    max_output_tokens=96,
    expected_prompt_words=67,
    description="Tests syntax compliance, key-value extraction accuracy, and valid JSON serialization.",
)

TASK_CODE_GENERATION = BenchmarkWorkloadTask(
    task_id="workload_v1_code_generation",
    name="Code Generation (Python Prime Function)",
    task_type=TaskEvaluationType.CODE_GENERATION,
    prompt=(
        "Write a Python function named `is_prime(n: int) -> bool` that returns True if n is a prime number and False otherwise. "
        "Include handling for numbers less than 2. Output ONLY executable Python code."
    ),
    max_output_tokens=140,
    expected_prompt_words=32,
    description="Tests programming logic, syntactically valid code emission, and algorithmic correctness.",
)

TASK_REASONING_NUMERICAL = BenchmarkWorkloadTask(
    task_id="workload_v1_reasoning_numerical",
    name="Multi-Step Numerical Reasoning",
    task_type=TaskEvaluationType.REASONING_NUMERICAL,
    prompt=(
        "A warehouse starts Monday morning with 120 server racks in inventory. "
        "On Tuesday, they ship out 40 racks to clients. "
        "On Wednesday, they receive a shipment of 60 fresh racks from the manufacturer. "
        "On Thursday, they install 20 percent of their current inventory into their own internal datacenter. "
        "How many racks remain in warehouse inventory on Friday morning? "
        "Show your calculation and write your final answer as: Final Answer: [number]"
    ),
    max_output_tokens=160,
    expected_prompt_words=73,
    description="Tests multi-step arithmetic reasoning, state tracking, and final answer formatting.",
)

WORKLOAD_SUITE: list[BenchmarkWorkloadTask] = [
    TASK_INTERACTIVE_QA,
    TASK_LONG_CONTEXT,
    TASK_LONG_FORM,
    TASK_STRUCTURED_JSON,
    TASK_CODE_GENERATION,
    TASK_REASONING_NUMERICAL,
]


# --- Quality Evaluation Logic ---


def evaluate_task_output(task: BenchmarkWorkloadTask, raw_output: str) -> QualityEvaluationResult:
    """Evaluate model output against deterministic quality and correctness criteria."""
    text = raw_output.strip()

    if task.task_type == TaskEvaluationType.INTERACTIVE_QA:
        # Check for Paris
        has_paris = "paris" in text.lower()
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=task.task_type,
            format_compliant=True,
            task_success=has_paris,
            correctness_score=1.0 if has_paris else 0.0,
            evaluation_method="EXACT_MATCH",
            evaluation_notes="Contains 'Paris' (case-insensitive)" if has_paris else "Did not contain 'Paris'",
            response_snippet=text[:120],
        )

    elif task.task_type == TaskEvaluationType.LONG_CONTEXT_RETRIEVAL:
        # Check for override key ZEPHYR-7703-DELTA
        has_key = "ZEPHYR-7703-DELTA" in text or "ZEPHYR-7703" in text
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=task.task_type,
            format_compliant=True,
            task_success=has_key,
            correctness_score=1.0 if has_key else 0.0,
            evaluation_method="NEEDLE_RETRIEVAL_REGEX",
            evaluation_notes="Identified key ZEPHYR-7703" if has_key else "Key not found in output",
            response_snippet=text[:120],
        )

    elif task.task_type == TaskEvaluationType.LONG_FORM_GENERATION:
        words = text.split()
        word_count = len(words)
        has_prefill = "prefill" in text.lower() or "prompt" in text.lower()
        has_decode = "decode" in text.lower() or "generation" in text.lower()
        has_concepts = "bandwidth" in text.lower() or "compute" in text.lower() or "arithmetic" in text.lower()

        format_ok = word_count >= 30
        task_ok = format_ok and has_prefill and has_decode and has_concepts
        score = sum([format_ok, has_prefill, has_decode, has_concepts]) / 4.0

        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=task.task_type,
            format_compliant=format_ok,
            task_success=task_ok,
            correctness_score=score,
            evaluation_method="HEURISTIC_KEYWORD_COMPLETION",
            evaluation_notes=f"Words: {word_count}. Keywords: prefill={has_prefill}, decode={has_decode}, compute/bandwidth={has_concepts}",
            response_snippet=text[:120],
        )

    elif task.task_type == TaskEvaluationType.STRUCTURED_JSON:
        # Strip markdown fences if present
        clean_json = text
        if "```" in clean_json:
            clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json, flags=re.MULTILINE)
            clean_json = re.sub(r"\s*```$", "", clean_json, flags=re.MULTILINE)
        clean_json = clean_json.strip()

        # Find first '{' and last '}'
        start = clean_json.find("{")
        end = clean_json.rfind("}")
        parsed: dict[str, Any] | None = None
        format_ok = False
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(clean_json[start : end + 1])
                format_ok = True
            except Exception:
                pass

        if not format_ok or parsed is None:
            return QualityEvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                format_compliant=False,
                task_success=False,
                correctness_score=0.0,
                evaluation_method="JSON_SCHEMA_VALIDATOR",
                evaluation_notes="JSON parsing failed",
                response_snippet=text[:120],
            )

        # Check fields: patient_id == PT-4402, age == 45, systolic_bp == 120, diastolic_bp == 80, pulse_bpm == 72
        checks = [
            str(parsed.get("patient_id", "")).strip() == "PT-4402",
            str(parsed.get("age", "")).strip() == "45",
            str(parsed.get("systolic_bp", "")).strip() == "120",
            str(parsed.get("diastolic_bp", "")).strip() == "80",
            str(parsed.get("pulse_bpm", "")).strip() == "72",
        ]
        score = sum(checks) / float(len(checks))
        task_ok = score >= 0.8

        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=task.task_type,
            format_compliant=True,
            task_success=task_ok,
            correctness_score=score,
            evaluation_method="JSON_SCHEMA_VALIDATOR",
            evaluation_notes=f"JSON valid. Schema match: {sum(checks)}/{len(checks)} fields correct.",
            response_snippet=text[:120],
        )

    elif task.task_type == TaskEvaluationType.CODE_GENERATION:
        # Strip markdown fences
        clean_code = text
        if "```" in clean_code:
            match = re.search(r"```(?:python)?\s*(.*?)\s*```", clean_code, re.DOTALL)
            clean_code = match.group(1) if match else re.sub(r"```", "", clean_code)

        syntax_ok = False
        try:
            tree = ast.parse(clean_code)
            syntax_ok = True
        except SyntaxError:
            tree = None

        if not syntax_ok or tree is None:
            return QualityEvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                format_compliant=False,
                task_success=False,
                correctness_score=0.0,
                evaluation_method="AST_SYNTAX_AND_UNIT_TEST",
                evaluation_notes="Python AST syntax error",
                response_snippet=text[:120],
            )

        # Safely execute function definition in a sandboxed dictionary
        test_env: dict[str, Any] = {}
        correctness_score = 0.0
        try:
            exec(compile(tree, filename="<ast>", mode="exec"), test_env)
            if "is_prime" in test_env and callable(test_env["is_prime"]):
                func = test_env["is_prime"]
                test_cases = [
                    (func(2), True),
                    (func(3), True),
                    (func(4), False),
                    (func(1), False),
                    (func(0), False),
                    (func(-5), False),
                    (func(17), True),
                    (func(25), False),
                ]
                passed = sum(1 for actual, expected in test_cases if actual == expected)
                correctness_score = passed / float(len(test_cases))
        except Exception as e:
            return QualityEvaluationResult(
                task_id=task.task_id,
                task_type=task.task_type,
                format_compliant=True,
                task_success=False,
                correctness_score=0.2,
                evaluation_method="AST_SYNTAX_AND_UNIT_TEST",
                evaluation_notes=f"Syntax valid, execution failed: {e}",
                response_snippet=text[:120],
            )

        task_ok = correctness_score >= 0.875
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=task.task_type,
            format_compliant=True,
            task_success=task_ok,
            correctness_score=correctness_score,
            evaluation_method="AST_SYNTAX_AND_UNIT_TEST",
            evaluation_notes=f"Syntax valid. Unit tests passed: {int(correctness_score * 8)}/8",
            response_snippet=text[:120],
        )

    elif task.task_type == TaskEvaluationType.REASONING_NUMERICAL:
        # Math: 120 - 40 = 80; 80 + 60 = 140; 140 - 28 = 112
        # Check if 112 is present as final answer
        has_112 = bool(re.search(r"\b112\b", text))
        final_answer_formatted = bool(re.search(r"Final Answer:\s*112\b", text, re.IGNORECASE))

        score = 0.0
        if final_answer_formatted:
            score = 1.0
        elif has_112:
            score = 0.8

        task_ok = has_112
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=task.task_type,
            format_compliant=final_answer_formatted or ("final answer" in text.lower()),
            task_success=task_ok,
            correctness_score=score,
            evaluation_method="NUMERICAL_REGEX_CHECKER",
            evaluation_notes="Final answer 112 present" if has_112 else "Final answer 112 not found",
            response_snippet=text[:120],
        )

    return QualityEvaluationResult(
        task_id=task.task_id,
        task_type=task.task_type,
        format_compliant=None,
        task_success=None,
        correctness_score=None,
        evaluation_method="UNASSESSED",
        evaluation_notes="No automated evaluator registered",
        response_snippet=text[:120],
    )
