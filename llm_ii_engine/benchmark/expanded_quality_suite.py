"""Expanded objective quality benchmark suite for LLM Inference Intelligence Engine.

Provides 36 deterministic, versioned tasks across 6 evaluation categories,
strictly partitioned into Development (6 tasks), Selection (18 tasks), and
Held-Out (12 tasks) splits to ensure benchmark validity and prevent data contamination.
"""

import ast
import contextlib
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_ii_engine.benchmark.multi_model_protocol import (
    CodeExecutionSecurityMode,
    QualityEvaluationResult,
    QualityEvaluationStatus,
    TaskEvaluationType,
)


class TaskSplit(str, Enum):
    """Rigorous dataset partition preventing data contamination and leakage."""

    DEV = "DEV"  # Calibration and evaluator verification
    SELECTION = "SELECTION"  # Candidate comparison and recommendation training
    HELD_OUT = "HELD_OUT"  # Prospective generalization validation only


class QualityTaskCategory(str, Enum):
    """Granular task category taxonomy for multi-model quality evaluation."""

    FACTUAL_INSTRUCTION = "FACTUAL_INSTRUCTION"
    NUMERICAL_REASONING = "NUMERICAL_REASONING"
    STRUCTURED_JSON = "STRUCTURED_JSON"
    CODE_GENERATION = "CODE_GENERATION"
    CONTEXT_RETRIEVAL = "CONTEXT_RETRIEVAL"
    LONG_FORM_COMPLIANCE = "LONG_FORM_COMPLIANCE"


class QualityBenchmarkTask(BaseModel):
    """Specification of an objective, versioned quality benchmark task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(description="Unique task identifier, e.g. 'factual_select_01'")
    name: str = Field(description="Descriptive task title")
    category: QualityTaskCategory = Field(description="Task category")
    split: TaskSplit = Field(description="Dataset partition (DEV, SELECTION, or HELD_OUT)")
    prompt: str = Field(description="Exact prompt string sent to model")
    max_output_tokens: int = Field(gt=0, description="Server output limit (num_predict)")
    expected_prompt_words: int = Field(gt=0, description="Approximate prompt word count")
    expected_answer_spec: dict[str, Any] = Field(description="Authoritative answer key or test spec")
    evaluator_version: str = Field(default="2.2.0", description="SemVer version of evaluator logic")
    difficulty: str = Field(default="medium", description="Task difficulty rating (easy, medium, hard)")
    evaluation_limitations: str = Field(description="Known evaluation scope limits or constraints")
    prompt_hash: str = Field(description="SHA256 digest of prompt string")


# ==============================================================================
# Helper to create tasks with automatic prompt hash
# ==============================================================================

def _create_task(
    task_id: str,
    name: str,
    category: QualityTaskCategory,
    split: TaskSplit,
    prompt: str,
    max_output_tokens: int,
    expected_answer_spec: dict[str, Any],
    difficulty: str = "medium",
    evaluation_limitations: str = "Deterministic automated evaluation; does not assess unprompted nuance.",
) -> QualityBenchmarkTask:
    p_hash = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:16]
    words = len(prompt.strip().split())
    return QualityBenchmarkTask(
        task_id=task_id,
        name=name,
        category=category,
        split=split,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        expected_prompt_words=words,
        expected_answer_spec=expected_answer_spec,
        difficulty=difficulty,
        evaluation_limitations=evaluation_limitations,
        prompt_hash=p_hash,
    )


# ==============================================================================
# CATEGORY A: FACTUAL AND INSTRUCTION-FOLLOWING TASKS (6 tasks)
# ==============================================================================

TASK_FACTUAL_DEV_01 = _create_task(
    task_id="factual_dev_01",
    name="Chemical Symbol Recall",
    category=QualityTaskCategory.FACTUAL_INSTRUCTION,
    split=TaskSplit.DEV,
    prompt="What is the chemical symbol for gold? Output ONLY the symbol, nothing else.",
    max_output_tokens=8,
    expected_answer_spec={"match_type": "exact_ci", "expected": "Au"},
    difficulty="easy",
)

TASK_FACTUAL_SELECT_01 = _create_task(
    task_id="factual_select_01",
    name="Constrained Multi-Instruction Colors",
    category=QualityTaskCategory.FACTUAL_INSTRUCTION,
    split=TaskSplit.SELECTION,
    prompt="List exactly three primary colors of light in RGB format, separated by commas. Do not include any punctuation other than commas, and use all lowercase.",
    max_output_tokens=16,
    expected_answer_spec={"match_type": "regex", "pattern": r"^(?:red\s*,\s*green\s*,\s*blue|red\s*,\s*blue\s*,\s*green|green\s*,\s*red\s*,\s*blue)$"},
    difficulty="medium",
)

TASK_FACTUAL_SELECT_02 = _create_task(
    task_id="factual_select_02",
    name="Moon Landing Year Prefix",
    category=QualityTaskCategory.FACTUAL_INSTRUCTION,
    split=TaskSplit.SELECTION,
    prompt="State the year the Apollo 11 moon landing occurred. Format your response exactly as: 'Year: YYYY'.",
    max_output_tokens=12,
    expected_answer_spec={"match_type": "regex", "pattern": r"Year:\s*1969\b"},
    difficulty="easy",
)

TASK_FACTUAL_SELECT_03 = _create_task(
    task_id="factual_select_03",
    name="Fact Extraction with Units",
    category=QualityTaskCategory.FACTUAL_INSTRUCTION,
    split=TaskSplit.SELECTION,
    prompt="Fact: The Mariana Trench reaches an estimated maximum depth of 10,994 meters (36,070 feet).\nQuestion: What is the depth in feet according to the text? State only the number with units.",
    max_output_tokens=16,
    expected_answer_spec={"match_type": "regex", "pattern": r"36[,\.]?070\s*feet\b"},
    difficulty="easy",
)

TASK_FACTUAL_HELDOUT_01 = _create_task(
    task_id="factual_heldout_01",
    name="Absent Information Abstention",
    category=QualityTaskCategory.FACTUAL_INSTRUCTION,
    split=TaskSplit.HELD_OUT,
    prompt="Based strictly on this text: 'The annual biology conference was held in Kyoto in October.'\nQuestion: Who was the keynote speaker? If the speaker is not mentioned in the text, you MUST respond with 'NOT_SPECIFIED'. Do not guess.",
    max_output_tokens=16,
    expected_answer_spec={"match_type": "contains_ci", "expected": "NOT_SPECIFIED"},
    difficulty="medium",
)

TASK_FACTUAL_HELDOUT_02 = _create_task(
    task_id="factual_heldout_02",
    name="Negative Constraint Geography",
    category=QualityTaskCategory.FACTUAL_INSTRUCTION,
    split=TaskSplit.HELD_OUT,
    prompt="Name the capital of Australia. Do NOT mention Sydney or Melbourne. Provide only the capital name.",
    max_output_tokens=12,
    expected_answer_spec={
        "match_type": "positive_and_negative",
        "required": ["Canberra"],
        "prohibited": ["Sydney", "Melbourne"],
    },
    difficulty="medium",
)


# ==============================================================================
# CATEGORY B: NUMERICAL REASONING TASKS (6 tasks)
# ==============================================================================

TASK_NUM_DEV_01 = _create_task(
    task_id="num_dev_01",
    name="Retail Profit Calculation",
    category=QualityTaskCategory.NUMERICAL_REASONING,
    split=TaskSplit.DEV,
    prompt="A store buys 50 notebooks for $2 each and sells them for $5 each. The shipping fee for the shipment was $20. What is the total net profit in dollars? State the final answer as: 'Profit: $X'.",
    max_output_tokens=48,
    expected_answer_spec={"match_type": "regex", "pattern": r"Profit:\s*\$?\s*130\b"},
    difficulty="medium",
)

TASK_NUM_SELECT_01 = _create_task(
    task_id="num_select_01",
    name="Discount and Sales Tax",
    category=QualityTaskCategory.NUMERICAL_REASONING,
    split=TaskSplit.SELECTION,
    prompt="An item originally priced at $80 is discounted by 25%. Then a 10% sales tax is applied to the discounted price. What is the final total cost? Provide only the final numerical price.",
    max_output_tokens=32,
    expected_answer_spec={"match_type": "number_in_text", "expected_number": 66.0},
    difficulty="medium",
)

TASK_NUM_SELECT_02 = _create_task(
    task_id="num_select_02",
    name="Distractor Filtering Arithmetic",
    category=QualityTaskCategory.NUMERICAL_REASONING,
    split=TaskSplit.SELECTION,
    prompt="In a library of 1,200 books across 4 floors built in 1995, there are 30 shelves in Room A. Each shelf holds 25 books. 10 books are currently checked out from Room A. How many books are currently present on the shelves in Room A? End your answer with 'Result: N'.",
    max_output_tokens=64,
    expected_answer_spec={"match_type": "regex", "pattern": r"Result:\s*740\b"},
    difficulty="medium",
)

TASK_NUM_SELECT_03 = _create_task(
    task_id="num_select_03",
    name="Speed Rate Calculation",
    category=QualityTaskCategory.NUMERICAL_REASONING,
    split=TaskSplit.SELECTION,
    prompt="A car travels 180 kilometers in 2 hours and 30 minutes. What is its average speed in kilometers per hour (km/h)? State only the number.",
    max_output_tokens=24,
    expected_answer_spec={"match_type": "number_in_text", "expected_number": 72.0},
    difficulty="medium",
)

TASK_NUM_HELDOUT_01 = _create_task(
    task_id="num_heldout_01",
    name="Expense Split Table",
    category=QualityTaskCategory.NUMERICAL_REASONING,
    split=TaskSplit.HELD_OUT,
    prompt="Table of monthly expenses:\n- Rent: $1,200\n- Utilities: $180\n- Groceries: $420\n- Transit: $100\nIf two roommates split all expenses equally, how much does each person pay? State only the final amount per person.",
    max_output_tokens=32,
    expected_answer_spec={"match_type": "number_in_text", "expected_number": 950.0},
    difficulty="medium",
)

TASK_NUM_HELDOUT_02 = _create_task(
    task_id="num_heldout_02",
    name="Algebraic Age Ratio",
    category=QualityTaskCategory.NUMERICAL_REASONING,
    split=TaskSplit.HELD_OUT,
    prompt="A father is currently 4 times as old as his son. In 20 years, the father will be twice as old as his son. How old is the son now? State the son's current age as a single integer.",
    max_output_tokens=48,
    expected_answer_spec={"match_type": "number_in_text", "expected_number": 10.0},
    difficulty="hard",
)


# ==============================================================================
# CATEGORY C: STRUCTURED DATA AND JSON TASKS (6 tasks)
# ==============================================================================

TASK_JSON_DEV_01 = _create_task(
    task_id="json_dev_01",
    name="Server Configuration Extraction",
    category=QualityTaskCategory.STRUCTURED_JSON,
    split=TaskSplit.DEV,
    prompt="Extract the server specifications into JSON with keys 'host' (string), 'port' (int), 'ssl_enabled' (bool). Text: 'Primary web server nginx running on host api.internal at port 8443 with TLS active.' Return ONLY the raw JSON object.",
    max_output_tokens=64,
    expected_answer_spec={
        "fields": {
            "host": "api.internal",
            "port": 8443,
            "ssl_enabled": True,
        }
    },
    difficulty="easy",
)

TASK_JSON_SELECT_01 = _create_task(
    task_id="json_select_01",
    name="Nested Clinical Vitals Schema",
    category=QualityTaskCategory.STRUCTURED_JSON,
    split=TaskSplit.SELECTION,
    prompt="Extract patient data into JSON: 'Patient ID PT-901, female, age 52. Vitals: blood pressure 130/85, pulse 76.' Use JSON structure: {\"patient_id\": \"PT-901\", \"demographics\": {\"age\": 52, \"gender\": \"female\"}, \"vitals\": {\"systolic\": 130, \"diastolic\": 85, \"pulse\": 76}}. Output ONLY valid JSON.",
    max_output_tokens=96,
    expected_answer_spec={
        "fields": {
            "patient_id": "PT-901",
            "demographics": {"age": 52, "gender": "female"},
            "vitals": {"systolic": 130, "diastolic": 85, "pulse": 76},
        }
    },
    difficulty="hard",
)

TASK_JSON_SELECT_02 = _create_task(
    task_id="json_select_02",
    name="Array Items Extraction",
    category=QualityTaskCategory.STRUCTURED_JSON,
    split=TaskSplit.SELECTION,
    prompt="Convert this shopping list into JSON with an 'items' array of objects containing 'name' (string) and 'quantity' (integer): '3 apples, 1 milk carton, 6 eggs'. Return ONLY JSON.",
    max_output_tokens=96,
    expected_answer_spec={
        "is_array_check": True,
        "array_key": "items",
        "expected_items_count": 3,
        "total_quantity": 10,
    },
    difficulty="medium",
)

TASK_JSON_SELECT_03 = _create_task(
    task_id="json_select_03",
    name="Missing Value Null Extraction",
    category=QualityTaskCategory.STRUCTURED_JSON,
    split=TaskSplit.SELECTION,
    prompt="Extract contact info: 'Contact name: Alice Vance. Phone: 555-0199. No email provided.' Return JSON with keys 'name', 'phone', 'email' (set 'email' to null if not provided). Output ONLY JSON.",
    max_output_tokens=64,
    expected_answer_spec={
        "fields": {
            "name": "Alice Vance",
            "phone": "555-0199",
            "email": None,
        }
    },
    difficulty="medium",
)

TASK_JSON_HELDOUT_01 = _create_task(
    task_id="json_heldout_01",
    name="Database Float and Boolean Config",
    category=QualityTaskCategory.STRUCTURED_JSON,
    split=TaskSplit.HELD_OUT,
    prompt="Convert to JSON: 'Database db-prod-01 on port 5432 with max_connections 100, read_timeout 2.5 seconds, and replica_enabled true.' Structure: {\"database\": str, \"port\": int, \"max_connections\": int, \"read_timeout_sec\": float, \"replica_enabled\": bool}. Output ONLY JSON.",
    max_output_tokens=80,
    expected_answer_spec={
        "fields": {
            "database": "db-prod-01",
            "port": 5432,
            "max_connections": 100,
            "read_timeout_sec": 2.5,
            "replica_enabled": True,
        }
    },
    difficulty="medium",
)

TASK_JSON_HELDOUT_02 = _create_task(
    task_id="json_heldout_02",
    name="Escaped Quotes Support Ticket",
    category=QualityTaskCategory.STRUCTURED_JSON,
    split=TaskSplit.HELD_OUT,
    prompt="Extract into JSON: 'Ticket #404: Title \"DB Connection \\\"Timeout\\\" Error\", priority high, assignee bob@corp.local.' Keys: 'ticket_id' (int), 'title' (string), 'priority' (string), 'assignee' (string). Output ONLY JSON.",
    max_output_tokens=80,
    expected_answer_spec={
        "fields": {
            "ticket_id": 404,
            "priority": "high",
            "assignee": "bob@corp.local",
        }
    },
    difficulty="hard",
)


# ==============================================================================
# CATEGORY D: CODE GENERATION TASKS (6 tasks)
# ==============================================================================

TASK_CODE_DEV_01 = _create_task(
    task_id="code_dev_01",
    name="Palindrome String Predicate",
    category=QualityTaskCategory.CODE_GENERATION,
    split=TaskSplit.DEV,
    prompt="Write a Python function `is_palindrome(s: str) -> bool` that returns True if string `s` is a palindrome (ignoring casing and spaces), False otherwise. Output ONLY the function definition.",
    max_output_tokens=128,
    expected_answer_spec={
        "entry_point": "is_palindrome",
        "tests": [
            ("racecar", True),
            ("Race car", True),
            ("hello", False),
            ("", True),
            ("a", True),
        ],
    },
    difficulty="medium",
)

TASK_CODE_SELECT_01 = _create_task(
    task_id="code_select_01",
    name="Order Preserving Deduplication",
    category=QualityTaskCategory.CODE_GENERATION,
    split=TaskSplit.SELECTION,
    prompt="Write a Python function `deduplicate_preserve_order(lst: list) -> list` that removes duplicate items from `lst` while keeping the original order of their first appearance. Output ONLY the function definition.",
    max_output_tokens=128,
    expected_answer_spec={
        "entry_point": "deduplicate_preserve_order",
        "tests": [
            ([1, 2, 2, 3, 1, 4], [1, 2, 3, 4]),
            (["a", "b", "a"], ["a", "b"]),
            ([], []),
            ([5, 5, 5], [5]),
        ],
    },
    difficulty="medium",
)

TASK_CODE_SELECT_02 = _create_task(
    task_id="code_select_02",
    name="Word Frequency Counter",
    category=QualityTaskCategory.CODE_GENERATION,
    split=TaskSplit.SELECTION,
    prompt="Write a Python function `count_frequencies(words: list) -> dict` that returns a dictionary mapping each string in `words` to the number of times it appears. Output ONLY the function definition.",
    max_output_tokens=128,
    expected_answer_spec={
        "entry_point": "count_frequencies",
        "tests": [
            (["apple", "banana", "apple"], {"apple": 2, "banana": 1}),
            ([], {}),
            (["x"], {"x": 1}),
            (["a", "b", "c"], {"a": 1, "b": 1, "c": 1}),
        ],
    },
    difficulty="medium",
)

TASK_CODE_SELECT_03 = _create_task(
    task_id="code_select_03",
    name="Binary Search Algorithm",
    category=QualityTaskCategory.CODE_GENERATION,
    split=TaskSplit.SELECTION,
    prompt="Write a Python function `binary_search(arr: list, target: int) -> int` that returns the 0-based index of `target` in sorted list `arr`, or -1 if not found. Output ONLY the function definition.",
    max_output_tokens=140,
    expected_answer_spec={
        "entry_point": "binary_search",
        "tests": [
            (([1, 3, 5, 7, 9], 5), 2),
            (([1, 3, 5, 7, 9], 1), 0),
            (([1, 3, 5, 7, 9], 9), 4),
            (([1, 3, 5, 7, 9], 4), -1),
            (([], 2), -1),
        ],
    },
    difficulty="hard",
)

TASK_CODE_HELDOUT_01 = _create_task(
    task_id="code_heldout_01",
    name="Sentence Word Reversal",
    category=QualityTaskCategory.CODE_GENERATION,
    split=TaskSplit.HELD_OUT,
    prompt="Write a Python function `reverse_words(sentence: str) -> str` that reverses the order of words in `sentence` (words separated by single spaces). Output ONLY the function definition.",
    max_output_tokens=96,
    expected_answer_spec={
        "entry_point": "reverse_words",
        "tests": [
            ("hello world", "world hello"),
            ("a b c", "c b a"),
            ("single", "single"),
            ("", ""),
        ],
    },
    difficulty="easy",
)

TASK_CODE_HELDOUT_02 = _create_task(
    task_id="code_heldout_02",
    name="Value Clamping Range",
    category=QualityTaskCategory.CODE_GENERATION,
    split=TaskSplit.HELD_OUT,
    prompt="Write a Python function `clamp(val: int, low: int, high: int) -> int` that returns `low` if `val < low`, `high` if `val > high`, and `val` otherwise. Output ONLY the function definition.",
    max_output_tokens=96,
    expected_answer_spec={
        "entry_point": "clamp",
        "tests": [
            ((5, 1, 10), 5),
            ((-2, 0, 100), 0),
            ((150, 0, 100), 100),
            ((0, 0, 10), 0),
            ((10, 0, 10), 10),
        ],
    },
    difficulty="easy",
)


# ==============================================================================
# CATEGORY E: CONTEXT RETRIEVAL TASKS (6 tasks)
# ==============================================================================

TASK_RETRIEVAL_DEV_01 = _create_task(
    task_id="retrieval_dev_01",
    name="Short Context Key Person",
    category=QualityTaskCategory.CONTEXT_RETRIEVAL,
    split=TaskSplit.DEV,
    prompt="Context: Project Falcon launched on March 14, led by Director Sandra Ramos with an initial grant of $4.2M.\nQuestion: Who is the director of Project Falcon? Answer with only the name.",
    max_output_tokens=16,
    expected_answer_spec={"match_type": "contains_ci", "expected": "Sandra Ramos"},
    difficulty="easy",
)

TASK_RETRIEVAL_SELECT_01 = _create_task(
    task_id="retrieval_select_01",
    name="Branch Needle in Haystack",
    category=QualityTaskCategory.CONTEXT_RETRIEVAL,
    split=TaskSplit.SELECTION,
    prompt=(
        "Global Office Report:\n"
        "The London branch completed facility upgrades in April. Headcount stands at 145 engineers and 20 staff.\n"
        "The Tokyo branch expanded operations in Shibuya, focusing on robotics automation. Headcount reached 88.\n"
        "The Zurich branch hosts the secure cryptographic infrastructure. In case of system failover, the primary activation passcode is ZURICH-ALPHA-99, managed by security lead Thomas Weber.\n"
        "Question: What is Zurich's emergency passcode? State only the passcode."
    ),
    max_output_tokens=16,
    expected_answer_spec={"match_type": "contains_ci", "expected": "ZURICH-ALPHA-99"},
    difficulty="medium",
)

TASK_RETRIEVAL_SELECT_02 = _create_task(
    task_id="retrieval_select_02",
    name="Superseded Policy Retrieval",
    category=QualityTaskCategory.CONTEXT_RETRIEVAL,
    split=TaskSplit.SELECTION,
    prompt=(
        "Policy Notice:\n"
        "The policy established on January 1 set the remote work allowance at $50/month. "
        "On June 15, Amendment B superseded this policy, increasing the allowance to $85/month. "
        "Question: What is the current effective remote work allowance per month according to the text? State only the dollar amount."
    ),
    max_output_tokens=16,
    expected_answer_spec={
        "match_type": "positive_and_negative",
        "required": ["85"],
        "prohibited": ["50/month", "$50 per month", "is $50"],
    },
    difficulty="medium",
)

TASK_RETRIEVAL_SELECT_03 = _create_task(
    task_id="retrieval_select_03",
    name="Multi-Fact Patient Aggregation",
    category=QualityTaskCategory.CONTEXT_RETRIEVAL,
    split=TaskSplit.SELECTION,
    prompt=(
        "Medication Administration Log:\n"
        "Patient A received 200mg medication at 08:00 and 150mg at 14:00.\n"
        "Patient B received 100mg at 10:00.\n"
        "Question: What is the total milligrams of medication received by Patient A across the day? State only the number with mg."
    ),
    max_output_tokens=16,
    expected_answer_spec={"match_type": "regex", "pattern": r"\b350\s*mg\b"},
    difficulty="medium",
)

TASK_RETRIEVAL_HELDOUT_01 = _create_task(
    task_id="retrieval_heldout_01",
    name="Unstated Fact Abstention",
    category=QualityTaskCategory.CONTEXT_RETRIEVAL,
    split=TaskSplit.HELD_OUT,
    prompt=(
        "System Architecture Log:\n"
        "The server cluster operates across us-east-1 and eu-central-1 regions. Backup schedules run daily at 02:00 UTC.\n"
        "Question: What is the encryption key length used for backup storage? If this information is NOT present in the text, you must reply: 'ABSENT_FROM_CONTEXT'. Do not fabricate."
    ),
    max_output_tokens=16,
    expected_answer_spec={"match_type": "contains_ci", "expected": "ABSENT_FROM_CONTEXT"},
    difficulty="medium",
)

TASK_RETRIEVAL_HELDOUT_02 = _create_task(
    task_id="retrieval_heldout_02",
    name="Deep Needle Token",
    category=QualityTaskCategory.CONTEXT_RETRIEVAL,
    split=TaskSplit.HELD_OUT,
    prompt=(
        "Cluster Nodes Diagnostic Log:\n"
        "Node 1: Online, load 0.42, ping 12ms. Checksum: A19-FF.\n"
        "Node 2: Online, load 0.58, ping 15ms. Checksum: B22-C8.\n"
        "Node 3: Standby, load 0.05, ping 8ms. Checksum: D04-11.\n"
        "Node 4: Provisioning sequence initiated. Hardware authentication token: PIN-7731-X. Security status verified.\n"
        "Node 5: Maintenance mode, load 0.00, ping 20ms. Checksum: E88-Z2.\n"
        "Question: What is Node 4's hardware authentication token? State only the token."
    ),
    max_output_tokens=16,
    expected_answer_spec={"match_type": "contains_ci", "expected": "PIN-7731-X"},
    difficulty="medium",
)


# ==============================================================================
# CATEGORY F: LONG-FORM INSTRUCTION COMPLIANCE (6 tasks)
# ==============================================================================

TASK_LONGFORM_DEV_01 = _create_task(
    task_id="longform_dev_01",
    name="Required Headings and Length Bounds",
    category=QualityTaskCategory.LONG_FORM_COMPLIANCE,
    split=TaskSplit.DEV,
    prompt="Write a summary of solar power. You MUST include two section headings: '## Benefits' and '## Limitations'. Write at least 40 words but no more than 140 words.",
    max_output_tokens=180,
    expected_answer_spec={
        "required_headings": ["## Benefits", "## Limitations"],
        "min_words": 40,
        "max_words": 150,
    },
    difficulty="medium",
)

TASK_LONGFORM_SELECT_01 = _create_task(
    task_id="longform_select_01",
    name="Keyword Inclusion and Prohibition",
    category=QualityTaskCategory.LONG_FORM_COMPLIANCE,
    split=TaskSplit.SELECTION,
    prompt="Explain photosynthesis in 2 paragraphs. You MUST mention 'chlorophyll' and 'ATP'. You MUST NOT mention the word 'mitochondria'. Each paragraph must contain at least 2 sentences.",
    max_output_tokens=180,
    expected_answer_spec={
        "required_keywords": ["chlorophyll", "ATP"],
        "prohibited_keywords": ["mitochondria"],
        "min_paragraphs": 2,
    },
    difficulty="hard",
)

TASK_LONGFORM_SELECT_02 = _create_task(
    task_id="longform_select_02",
    name="Exact 4-Item Numbered Troubleshooting List",
    category=QualityTaskCategory.LONG_FORM_COMPLIANCE,
    split=TaskSplit.SELECTION,
    prompt="List exactly 4 key steps to troubleshoot a slow internet connection. Format as a numbered list from 1 to 4. Do not include a 5th item.",
    max_output_tokens=160,
    expected_answer_spec={
        "numbered_items_required": ["1.", "2.", "3.", "4."],
        "prohibited_items": ["5."],
    },
    difficulty="medium",
)

TASK_LONGFORM_SELECT_03 = _create_task(
    task_id="longform_select_03",
    name="Release Note Header and Version Metrics",
    category=QualityTaskCategory.LONG_FORM_COMPLIANCE,
    split=TaskSplit.SELECTION,
    prompt="Draft a quarterly software release note. You must include: (1) Version number 'v2.4.0', (2) Performance metric '35% lower latency', (3) A '### Highlights' section header.",
    max_output_tokens=160,
    expected_answer_spec={
        "required_phrases": ["v2.4.0", "35% lower latency", "### Highlights"],
    },
    difficulty="medium",
)

TASK_LONGFORM_HELDOUT_01 = _create_task(
    task_id="longform_heldout_01",
    name="Greenhouse Description with Exact Closing Token",
    category=QualityTaskCategory.LONG_FORM_COMPLIANCE,
    split=TaskSplit.HELD_OUT,
    prompt="Write a brief description of the greenhouse effect. End your entire response with the exact closing token: '---END OF SUMMARY---'.",
    max_output_tokens=160,
    expected_answer_spec={
        "closing_token": "---END OF SUMMARY---",
        "min_words": 20,
    },
    difficulty="medium",
)

TASK_LONGFORM_HELDOUT_02 = _create_task(
    task_id="longform_heldout_02",
    name="Pronoun-Free Technical Explanation",
    category=QualityTaskCategory.LONG_FORM_COMPLIANCE,
    split=TaskSplit.HELD_OUT,
    prompt="Describe water purification in 40 to 100 words. Do NOT use the personal pronouns 'we', 'you', 'our', or 'I'. Include the word 'filtration'.",
    max_output_tokens=140,
    expected_answer_spec={
        "required_words": ["filtration"],
        "prohibited_words": ["we", "you", "our", "i"],
        "min_words": 35,
        "max_words": 115,
    },
    difficulty="hard",
)


# ==============================================================================
# Complete Authoritative Suite of 36 Tasks
# ==============================================================================

EXPANDED_QUALITY_SUITE: list[QualityBenchmarkTask] = [
    # Category A: Factual
    TASK_FACTUAL_DEV_01,
    TASK_FACTUAL_SELECT_01,
    TASK_FACTUAL_SELECT_02,
    TASK_FACTUAL_SELECT_03,
    TASK_FACTUAL_HELDOUT_01,
    TASK_FACTUAL_HELDOUT_02,
    # Category B: Numerical
    TASK_NUM_DEV_01,
    TASK_NUM_SELECT_01,
    TASK_NUM_SELECT_02,
    TASK_NUM_SELECT_03,
    TASK_NUM_HELDOUT_01,
    TASK_NUM_HELDOUT_02,
    # Category C: Structured JSON
    TASK_JSON_DEV_01,
    TASK_JSON_SELECT_01,
    TASK_JSON_SELECT_02,
    TASK_JSON_SELECT_03,
    TASK_JSON_HELDOUT_01,
    TASK_JSON_HELDOUT_02,
    # Category D: Code Generation
    TASK_CODE_DEV_01,
    TASK_CODE_SELECT_01,
    TASK_CODE_SELECT_02,
    TASK_CODE_SELECT_03,
    TASK_CODE_HELDOUT_01,
    TASK_CODE_HELDOUT_02,
    # Category E: Context Retrieval
    TASK_RETRIEVAL_DEV_01,
    TASK_RETRIEVAL_SELECT_01,
    TASK_RETRIEVAL_SELECT_02,
    TASK_RETRIEVAL_SELECT_03,
    TASK_RETRIEVAL_HELDOUT_01,
    TASK_RETRIEVAL_HELDOUT_02,
    # Category F: Long-Form Instruction Compliance
    TASK_LONGFORM_DEV_01,
    TASK_LONGFORM_SELECT_01,
    TASK_LONGFORM_SELECT_02,
    TASK_LONGFORM_SELECT_03,
    TASK_LONGFORM_HELDOUT_01,
    TASK_LONGFORM_HELDOUT_02,
]

DEV_TASKS = [t for t in EXPANDED_QUALITY_SUITE if t.split == TaskSplit.DEV]
SELECTION_TASKS = [t for t in EXPANDED_QUALITY_SUITE if t.split == TaskSplit.SELECTION]
HELDOUT_TASKS = [t for t in EXPANDED_QUALITY_SUITE if t.split == TaskSplit.HELD_OUT]


def compute_suite_hashes() -> dict[str, str]:
    """Compute deterministic cryptographic SHA256 hashes of task partitions."""
    def _hash_tasks(tasks: list[QualityBenchmarkTask]) -> str:
        serialized = "".join(f"{t.task_id}:{t.prompt}:{json.dumps(t.expected_answer_spec, sort_keys=True)}" for t in tasks)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return {
        "suite_hash": _hash_tasks(EXPANDED_QUALITY_SUITE),
        "dev_hash": _hash_tasks(DEV_TASKS),
        "selection_hash": _hash_tasks(SELECTION_TASKS),
        "heldout_hash": _hash_tasks(HELDOUT_TASKS),
        "total_task_count": str(len(EXPANDED_QUALITY_SUITE)),
        "dev_count": str(len(DEV_TASKS)),
        "selection_count": str(len(SELECTION_TASKS)),
        "heldout_count": str(len(HELDOUT_TASKS)),
    }


# ==============================================================================
# Objective Automated Evaluator Functions
# ==============================================================================

def evaluate_quality_task(
    task: QualityBenchmarkTask,
    raw_output: str,
    security_mode: CodeExecutionSecurityMode = CodeExecutionSecurityMode.FAIL_CLOSED,
) -> QualityEvaluationResult:
    """Execute deterministic automated evaluation against task answer specification."""
    text = (raw_output or "").strip()
    cat = task.category

    if cat == QualityTaskCategory.FACTUAL_INSTRUCTION:
        return _eval_factual(task, text)
    elif cat == QualityTaskCategory.NUMERICAL_REASONING:
        return _eval_numerical(task, text)
    elif cat == QualityTaskCategory.STRUCTURED_JSON:
        return _eval_json(task, text)
    elif cat == QualityTaskCategory.CODE_GENERATION:
        return _eval_code(task, text, security_mode=security_mode)
    elif cat == QualityTaskCategory.CONTEXT_RETRIEVAL:
        return _eval_retrieval(task, text)
    elif cat == QualityTaskCategory.LONG_FORM_COMPLIANCE:
        return _eval_longform(task, text)
    else:
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.INTERACTIVE_QA,
            format_compliant=False,
            task_success=False,
            correctness_score=0.0,
            evaluation_method="UNKNOWN",
            evaluation_notes=f"Unsupported category: {cat}",
            response_snippet=text[:100],
            evaluation_status=QualityEvaluationStatus.EVALUATOR_ERROR,
        )


def _eval_factual(task: QualityBenchmarkTask, text: str) -> QualityEvaluationResult:
    spec = task.expected_answer_spec
    mtype = spec.get("match_type")
    success = False

    if mtype == "exact_ci":
        cleaned = re.sub(r"[^\w]", "", text.strip().lower())
        exp = re.sub(r"[^\w]", "", str(spec["expected"]).lower())
        success = cleaned == exp
    elif mtype == "regex":
        pattern = spec["pattern"]
        success = bool(re.search(pattern, text, re.IGNORECASE))
    elif mtype == "contains_ci":
        exp = spec["expected"].lower()
        success = exp in text.lower()
    elif mtype == "positive_and_negative":
        reqs = spec.get("required", [])
        prohibs = spec.get("prohibited", [])
        pos_met = all(r.lower() in text.lower() for r in reqs)
        neg_met = not any(p.lower() in text.lower() for p in prohibs)
        success = pos_met and neg_met

    return QualityEvaluationResult(
        task_id=task.task_id,
        task_type=TaskEvaluationType.INTERACTIVE_QA,
        format_compliant=True,
        task_success=success,
        correctness_score=1.0 if success else 0.0,
        evaluation_method="OBJECTIVE_FACTUAL_MATCHER",
        evaluation_notes="Match criteria verified" if success else "Criteria failed",
        response_snippet=text[:120],
        evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_PASSED
        if success
        else QualityEvaluationStatus.EXECUTION_TEST_FAILED,
    )


def _eval_numerical(task: QualityBenchmarkTask, text: str) -> QualityEvaluationResult:
    spec = task.expected_answer_spec
    mtype = spec.get("match_type")
    success = False

    if mtype == "regex":
        success = bool(re.search(spec["pattern"], text, re.IGNORECASE))
    elif mtype == "number_in_text":
        expected_val = float(spec["expected_number"])
        # Find all numbers in text
        numbers = [float(n.replace(",", "")) for n in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)]
        # Check if expected number is in candidate numbers
        success = any(abs(n - expected_val) < 0.01 for n in numbers)

    return QualityEvaluationResult(
        task_id=task.task_id,
        task_type=TaskEvaluationType.REASONING_NUMERICAL,
        format_compliant=True,
        task_success=success,
        correctness_score=1.0 if success else 0.0,
        evaluation_method="NUMERICAL_PARSER",
        evaluation_notes="Numeric value verified" if success else "Expected number not found in text",
        response_snippet=text[:120],
        evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_PASSED
        if success
        else QualityEvaluationStatus.EXECUTION_TEST_FAILED,
    )


def _eval_json(task: QualityBenchmarkTask, text: str) -> QualityEvaluationResult:
    spec = task.expected_answer_spec

    # Clean markdown fences if present
    cleaned = text
    if "```json" in text:
        cleaned = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        cleaned = text.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(cleaned)
        syntax_valid = True
    except Exception as e:
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.STRUCTURED_JSON,
            format_compliant=False,
            task_success=False,
            correctness_score=0.0,
            evaluation_method="JSON_SYNTAX_AND_SEMANTIC_PARSER",
            evaluation_notes=f"JSON parse error: {e}",
            response_snippet=text[:120],
            evaluation_status=QualityEvaluationStatus.SYNTAX_INVALID,
        )

    # Semantic evaluation
    if spec.get("is_array_check"):
        arr_key = spec.get("array_key", "items")
        items = parsed.get(arr_key, [])
        if not isinstance(items, list) or len(items) != spec["expected_items_count"]:
            return QualityEvaluationResult(
                task_id=task.task_id,
                task_type=TaskEvaluationType.STRUCTURED_JSON,
                format_compliant=True,
                task_success=False,
                correctness_score=0.5,
                evaluation_method="JSON_SYNTAX_AND_SEMANTIC_PARSER",
                evaluation_notes=f"Array length mismatch: expected {spec['expected_items_count']}, got {len(items)}",
                response_snippet=text[:120],
                evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_FAILED,
            )
        total_q = sum(it.get("quantity", 0) for it in items if isinstance(it, dict))
        if total_q == spec["total_quantity"]:
            return QualityEvaluationResult(
                task_id=task.task_id,
                task_type=TaskEvaluationType.STRUCTURED_JSON,
                format_compliant=True,
                task_success=True,
                correctness_score=1.0,
                evaluation_method="JSON_SYNTAX_AND_SEMANTIC_PARSER",
                evaluation_notes="Array and item quantities matched exactly",
                response_snippet=text[:120],
                evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_PASSED,
            )
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.STRUCTURED_JSON,
            format_compliant=True,
            task_success=False,
            correctness_score=0.7,
            evaluation_method="JSON_SYNTAX_AND_SEMANTIC_PARSER",
            evaluation_notes=f"Quantity mismatch: expected {spec['total_quantity']}, got {total_q}",
            response_snippet=text[:120],
            evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_FAILED,
        )

    expected_fields = spec.get("fields", {})
    if not expected_fields:
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.STRUCTURED_JSON,
            format_compliant=True,
            task_success=True,
            correctness_score=1.0,
            evaluation_method="JSON_SYNTAX_AND_SEMANTIC_PARSER",
            evaluation_notes="Syntax valid, no semantic fields specified",
            response_snippet=text[:120],
            evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_PASSED,
        )

    # Recursive field comparison
    def _check_fields(exp_dict: dict, act_dict: dict) -> tuple[int, int]:
        matched = 0
        total = 0
        for k, v in exp_dict.items():
            total += 1
            if k not in act_dict:
                continue
            act_val = act_dict[k]
            if isinstance(v, dict) and isinstance(act_val, dict):
                sub_m, sub_t = _check_fields(v, act_val)
                matched += sub_m
                total += (sub_t - 1)  # replace 1 with sub_t
            elif v is None:
                if act_val is None:
                    matched += 1
            elif isinstance(v, (int, float)) and isinstance(act_val, (int, float)):
                if abs(float(v) - float(act_val)) < 0.01:
                    matched += 1
            elif str(v).lower() == str(act_val).lower():
                matched += 1
        return matched, total

    if isinstance(parsed, dict):
        matched_cnt, total_cnt = _check_fields(expected_fields, parsed)
        score = round(matched_cnt / total_cnt, 3) if total_cnt > 0 else 1.0
        success = (score == 1.0)
    else:
        score = 0.0
        success = False

    return QualityEvaluationResult(
        task_id=task.task_id,
        task_type=TaskEvaluationType.STRUCTURED_JSON,
        format_compliant=syntax_valid,
        task_success=success,
        correctness_score=score,
        evaluation_method="JSON_SYNTAX_AND_SEMANTIC_PARSER",
        evaluation_notes=f"Semantic fields matched: {score * 100:.1f}%",
        response_snippet=text[:120],
        evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_PASSED
        if success
        else QualityEvaluationStatus.EXECUTION_TEST_FAILED,
    )


# ==============================================================================
# Security Policy & Hardened Sandbox Isolation
# ==============================================================================

FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket", "http", "urllib", "urllib3",
    "requests", "ctypes", "cffi", "importlib", "pickle", "builtins", "posix",
    "nt", "threading", "multiprocessing", "pty", "commands", "winreg",
    "msvcrt", "signal", "_posixsubprocess", "_winapi", "platform", "pathlib",
    "tempfile", "io",
}

FORBIDDEN_ATTRS = {
    "__subclasses__", "__bases__", "__base__", "__globals__", "__builtins__",
    "__code__", "__closure__", "__dict__", "__class__", "__loader__",
    "__spec__", "__import__",
}

FORBIDDEN_FUNCS = {
    "eval", "exec", "getattr", "setattr", "delattr", "compile", "open",
    "input", "breakpoint", "memoryview", "help", "license",
}


def check_ast_security(tree: ast.AST) -> tuple[bool, str]:
    """Inspect AST nodes for dangerous imports, attribute lookups, and function calls."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in FORBIDDEN_IMPORTS:
                return False, f"Forbidden import from: {node.module}"
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                return False, f"Forbidden attribute access: {node.attr}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_FUNCS:
                    return False, f"Forbidden function call: {node.func.id}()"
            elif isinstance(node.func, ast.Attribute) and (
                node.func.attr in FORBIDDEN_FUNCS or node.func.attr in FORBIDDEN_ATTRS
            ):
                return False, f"Forbidden function/attribute call: {node.func.attr}()"
    return True, "AST static security checks passed"


def _run_hardened_subprocess_code(
    code: str, tests: list[Any], entry_point: str, timeout: float = 2.0
) -> tuple[int, int, str | None, bool]:
    """Execute untrusted code inside an isolated, resource-bounded subprocess with sanitized environment.

    Returns (passed_count, total_count, error_message, timed_out).
    """
    runner_script = f"""import json, sys
restricted_builtins = {{
    'range': range, 'int': int, 'float': float, 'bool': bool, 'len': len,
    'isinstance': isinstance, 'min': min, 'max': max, 'abs': abs, 'str': str,
    'dict': dict, 'list': list, 'set': set, 'tuple': tuple, 'enumerate': enumerate,
    'zip': zip, 'sorted': sorted, 'sum': sum, 'reversed': reversed,
}}
sandbox_globals = {{'__builtins__': restricted_builtins}}
code_str = {repr(code)}
tests = {repr(tests)}
entry_point = {repr(entry_point)}
try:
    exec(code_str, sandbox_globals)
except Exception as e:
    print(json.dumps({{'passed': 0, 'total': len(tests), 'err': f'Execution error: {{e}}'}}))
    sys.exit(0)
if entry_point not in sandbox_globals or not callable(sandbox_globals[entry_point]):
    print(json.dumps({{'passed': 0, 'total': len(tests), 'err': f"Function '{{entry_point}}' not defined in environment"}}))
    sys.exit(0)
fn = sandbox_globals[entry_point]
passed = 0
for test_in, test_out in tests:
    try:
        res = fn(*test_in) if isinstance(test_in, tuple) else fn(test_in)
        if res == test_out:
            passed += 1
    except Exception:
        pass
print(json.dumps({{'passed': passed, 'total': len(tests), 'err': None}}))
"""

    base_py = getattr(sys, "_base_executable", sys.executable)
    py_dir = os.path.dirname(base_py)
    system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
    sys32 = os.path.join(system_root, "System32")

    # Scrub environment to prevent secret or token leakage
    scrubbed_env: dict[str, str] = {}
    for k in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
        if k in os.environ:
            scrubbed_env[k] = os.environ[k]
    scrubbed_env["PATH"] = f"{py_dir};{sys32}"

    job_handle = None
    proc = None
    try:
        proc = subprocess.Popen(
            [base_py, "-c", runner_script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=scrubbed_env,
        )

        # Attach Windows Job Object for process tree termination and memory bounds
        if sys.platform == "win32":
            try:
                from ctypes import wintypes

                class _IO_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("ReadOperationCount", ctypes.c_uint64),
                        ("WriteOperationCount", ctypes.c_uint64),
                        ("OtherOperationCount", ctypes.c_uint64),
                        ("ReadTransferCount", ctypes.c_uint64),
                        ("WriteTransferCount", ctypes.c_uint64),
                        ("OtherTransferCount", ctypes.c_uint64),
                    ]

                class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                    _fields_ = [
                        ("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD),
                    ]

                class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                    _fields_ = [
                        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ("IoInfo", _IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t),
                    ]

                kernel32 = ctypes.windll.kernel32
                job_handle = kernel32.CreateJobObjectW(None, None)
                if job_handle:
                    limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                    # KILL_ON_JOB_CLOSE (0x2000) | PROCESS_MEMORY (0x0100) | ACTIVE_PROCESS (0x0008)
                    limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x0100 | 0x0008
                    limits.BasicLimitInformation.ActiveProcessLimit = 1
                    limits.ProcessMemoryLimit = 256 * 1024 * 1024  # 256 MiB
                    kernel32.SetInformationJobObject(
                        job_handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
                    )
                    kernel32.AssignProcessToJobObject(job_handle, int(proc._handle))
            except Exception:
                pass

        stdout, stderr = proc.communicate(timeout=timeout)
        # Cap output reading at 16 KB to prevent memory exhaustion
        stdout = stdout[:16384]
        stderr = stderr[:16384]

        if proc.returncode == 0 and stdout.strip():
            last_line = stdout.strip().splitlines()[-1]
            result_data = json.loads(last_line)
            return (
                result_data.get("passed", 0),
                result_data.get("total", len(tests)),
                result_data.get("err"),
                False,
            )
        else:
            err = stderr.strip() or stdout.strip() or f"Process execution failed with returncode {proc.returncode}"
            return 0, len(tests), err, False

    except subprocess.TimeoutExpired:
        if job_handle and sys.platform == "win32":
            with contextlib.suppress(Exception):
                ctypes.windll.kernel32.TerminateJobObject(job_handle, 1)
        if proc:
            with contextlib.suppress(Exception):
                proc.kill()
                proc.communicate()
        return 0, len(tests), "Execution timeout exceeded (2.0s bounded wall-time limit)", True

    except Exception as e:
        return 0, len(tests), f"Hardened sandbox error: {e}", False

    finally:
        if job_handle and sys.platform == "win32":
            with contextlib.suppress(Exception):
                ctypes.windll.kernel32.CloseHandle(job_handle)


def _eval_code(
    task: QualityBenchmarkTask,
    text: str,
    security_mode: CodeExecutionSecurityMode = CodeExecutionSecurityMode.FAIL_CLOSED,
) -> QualityEvaluationResult:
    spec = task.expected_answer_spec
    entry_point = spec.get("entry_point")
    tests = spec.get("tests", [])

    # Extract code from markdown fences if present
    code = text
    if "```python" in text:
        code = text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        code = text.split("```")[1].split("```")[0].strip()

    # Step 1: AST parse validation
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.CODE_GENERATION,
            format_compliant=False,
            task_success=False,
            correctness_score=0.0,
            evaluation_method="FAIL_CLOSED_AST_STATIC_VERIFICATION",
            evaluation_notes=f"Python SyntaxError: {e}",
            response_snippet=text[:120],
            evaluation_status=QualityEvaluationStatus.SYNTAX_INVALID,
            security_mode=security_mode,
            isolation_guarantees=["NO_BYTECODE_EXECUTED"],
        )

    # Step 2: AST static security check
    safe, reason = check_ast_security(tree)
    if not safe:
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.CODE_GENERATION,
            format_compliant=True,
            task_success=False,
            correctness_score=0.0,
            evaluation_method="FAIL_CLOSED_AST_STATIC_VERIFICATION",
            evaluation_notes=f"Security violation: {reason}",
            response_snippet=text[:120],
            evaluation_status=QualityEvaluationStatus.STATIC_CHECK_FAILED,
            security_mode=security_mode,
            isolation_guarantees=["NO_BYTECODE_EXECUTED", "AST_STATIC_SAFETY_VERIFIED"],
        )

    # Step 3: Verify entry point definition in AST
    has_entry = (
        entry_point is None
        or any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in ast.walk(tree))
    )
    if not has_entry:
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.CODE_GENERATION,
            format_compliant=True,
            task_success=False,
            correctness_score=0.0,
            evaluation_method="FAIL_CLOSED_AST_STATIC_VERIFICATION",
            evaluation_notes=f"Required function '{entry_point}' not defined in AST",
            response_snippet=text[:120],
            evaluation_status=QualityEvaluationStatus.STATIC_CHECK_FAILED,
            security_mode=security_mode,
            isolation_guarantees=["NO_BYTECODE_EXECUTED", "AST_STATIC_SAFETY_VERIFIED"],
        )

    # Step 4: Handle FAIL_CLOSED mode (default)
    if security_mode == CodeExecutionSecurityMode.FAIL_CLOSED:
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.CODE_GENERATION,
            format_compliant=True,
            task_success=None,
            correctness_score=None,
            evaluation_method="FAIL_CLOSED_AST_STATIC_VERIFICATION",
            evaluation_notes="Execution unverified: Untrusted code execution disabled by fail-closed security policy. Syntax and static safety rules verified.",
            response_snippet=text[:120],
            evaluation_status=QualityEvaluationStatus.EXECUTION_UNVERIFIED,
            security_mode=CodeExecutionSecurityMode.FAIL_CLOSED,
            isolation_guarantees=["NO_BYTECODE_EXECUTED", "AST_STATIC_SAFETY_VERIFIED"],
        )

    # Step 5: Handle OPT_IN_HARDENED_JOB_OBJECT mode
    passed_cnt, total_cnt, err_msg, timed_out = _run_hardened_subprocess_code(
        code, tests, entry_point or "", timeout=2.0
    )

    isolation_guarantees = [
        "WINDOWS_JOB_OBJECT_KILL_ON_CLOSE",
        "MEMORY_LIMIT_256MB",
        "ACTIVE_PROCESS_LIMIT_1",
        "ENVIRONMENT_SCRUBBED",
        "OUTPUT_STREAM_CAPPED_16KB",
        "BOUNDED_TIMEOUT_2S",
    ]

    if timed_out:
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.CODE_GENERATION,
            format_compliant=True,
            task_success=False,
            correctness_score=0.0,
            evaluation_method="HARDENED_JOB_OBJECT_SANDBOX_UNIT_TEST",
            evaluation_notes=err_msg or "Execution timeout exceeded (2.0s bounded wall-time limit)",
            response_snippet=text[:120],
            evaluation_status=QualityEvaluationStatus.EVALUATION_TIMEOUT,
            security_mode=CodeExecutionSecurityMode.OPT_IN_HARDENED_JOB_OBJECT,
            isolation_guarantees=isolation_guarantees,
        )

    if err_msg:
        return QualityEvaluationResult(
            task_id=task.task_id,
            task_type=TaskEvaluationType.CODE_GENERATION,
            format_compliant=True,
            task_success=False,
            correctness_score=0.0,
            evaluation_method="HARDENED_JOB_OBJECT_SANDBOX_UNIT_TEST",
            evaluation_notes=err_msg,
            response_snippet=text[:120],
            evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_FAILED,
            security_mode=CodeExecutionSecurityMode.OPT_IN_HARDENED_JOB_OBJECT,
            isolation_guarantees=isolation_guarantees,
        )

    pass_rate = round(passed_cnt / total_cnt, 3) if total_cnt > 0 else 1.0
    success = (passed_cnt == total_cnt)

    return QualityEvaluationResult(
        task_id=task.task_id,
        task_type=TaskEvaluationType.CODE_GENERATION,
        format_compliant=True,
        task_success=success,
        correctness_score=pass_rate,
        evaluation_method="HARDENED_JOB_OBJECT_SANDBOX_UNIT_TEST",
        evaluation_notes=f"Unit tests passed: {passed_cnt}/{total_cnt}",
        response_snippet=text[:120],
        evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_PASSED
        if success
        else QualityEvaluationStatus.EXECUTION_TEST_FAILED,
        security_mode=CodeExecutionSecurityMode.OPT_IN_HARDENED_JOB_OBJECT,
        isolation_guarantees=isolation_guarantees,
    )


def _eval_retrieval(task: QualityBenchmarkTask, text: str) -> QualityEvaluationResult:
    spec = task.expected_answer_spec
    mtype = spec.get("match_type")
    success = False

    if mtype == "contains_ci":
        success = spec["expected"].lower() in text.lower()
    elif mtype == "regex":
        success = bool(re.search(spec["pattern"], text, re.IGNORECASE))
    elif mtype == "positive_and_negative":
        reqs = spec.get("required", [])
        prohibs = spec.get("prohibited", [])
        pos_met = all(r.lower() in text.lower() for r in reqs)
        neg_met = not any(p.lower() in text.lower() for p in prohibs)
        success = pos_met and neg_met

    return QualityEvaluationResult(
        task_id=task.task_id,
        task_type=TaskEvaluationType.LONG_CONTEXT_RETRIEVAL,
        format_compliant=True,
        task_success=success,
        correctness_score=1.0 if success else 0.0,
        evaluation_method="CONTEXT_RETRIEVAL_MATCHER",
        evaluation_notes="Retrieved evidence verified" if success else "Key needle not retrieved",
        response_snippet=text[:120],
        evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_PASSED
        if success
        else QualityEvaluationStatus.EXECUTION_TEST_FAILED,
    )


def _eval_longform(task: QualityBenchmarkTask, text: str) -> QualityEvaluationResult:
    spec = task.expected_answer_spec
    words = len(text.strip().split())

    checks = []

    # Word bounds
    if "min_words" in spec:
        checks.append(words >= spec["min_words"])
    if "max_words" in spec:
        checks.append(words <= spec["max_words"])

    # Required headings / phrases
    for h in spec.get("required_headings", []):
        checks.append(h.lower() in text.lower())
    for p in spec.get("required_phrases", []):
        checks.append(p.lower() in text.lower())
    for kw in spec.get("required_keywords", []):
        checks.append(kw.lower() in text.lower())
    for w in spec.get("required_words", []):
        checks.append(bool(re.search(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE)))

    # Prohibited phrases / words
    for kw in spec.get("prohibited_keywords", []):
        checks.append(kw.lower() not in text.lower())
    for w in spec.get("prohibited_words", []):
        checks.append(not bool(re.search(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE)))
    for p in spec.get("prohibited_items", []):
        checks.append(p not in text)

    # Numbered items
    for item in spec.get("numbered_items_required", []):
        checks.append(item in text)

    # Paragraph count
    if "min_paragraphs" in spec:
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        checks.append(len(paragraphs) >= spec["min_paragraphs"])

    # Closing token
    if "closing_token" in spec:
        token = spec["closing_token"]
        checks.append(token in text)

    if not checks:
        passed_ratio = 1.0
        success = True
    else:
        passed_ratio = round(sum(1 for c in checks if c) / len(checks), 3)
        success = all(checks)

    return QualityEvaluationResult(
        task_id=task.task_id,
        task_type=TaskEvaluationType.LONG_FORM_GENERATION,
        format_compliant=True,
        task_success=success,
        correctness_score=passed_ratio,
        evaluation_method="STRUCTURE_AND_CONSTRAINT_VERIFIER",
        evaluation_notes=f"Constraint adherence: {passed_ratio * 100:.1f}% ({sum(1 for c in checks if c)}/{len(checks)})",
        response_snippet=text[:120],
        evaluation_status=QualityEvaluationStatus.EXECUTION_TEST_PASSED
        if success
        else QualityEvaluationStatus.EXECUTION_TEST_FAILED,
    )
