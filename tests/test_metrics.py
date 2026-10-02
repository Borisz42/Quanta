"""Unit and Regression Tests for Hardened Metrics & Evaluators (Sections 2 & 3).

Verifies:
- Section 2: AI2 ARC-Challenge Science Multiple-Choice Evaluation Hardening:
  - Closing-answer prioritization
  - Prevention of false-positive collision with English article "a" and variable "d"
  - Recognition of parenthesized, boxed, bolded, or isolated choice keys
- Section 3: Complex Long-Context Variable Tracking Numeric-Aware Evaluation:
  - Numeric match score on multi-step reasoning (var_track_256k forensic case)
  - Unit and noun singular/plural stem lenience
  - Distinction between correct calculations and incorrect calculations / mismatched units
  - Hallucination detection coherence with numeric matches
"""

from __future__ import annotations

import pytest

from benchmarks.metrics import BenchmarkMetrics, normalize_answer
from benchmarks.paired_evaluator import PairedEvaluator, EvaluationCondition
from benchmarks.suite_loaders import BenchmarkSample


# =============================================================================
# Section 2: Science Multiple-Choice Evaluation Tests
# =============================================================================

def test_extract_multiple_choice_explicit_anchors():
    """Verifies that explicit answer keywords are accurately detected and prioritized."""
    # Standard closing answer patterns
    assert BenchmarkMetrics.extract_multiple_choice_key(
        "Photosynthesis converts solar photons into chemical energy. Therefore, the answer is (A)."
    ) == "A"

    assert BenchmarkMetrics.extract_multiple_choice_key(
        "The light source is moving away due to redshift. The correct choice is (B)."
    ) == "B"

    assert BenchmarkMetrics.extract_multiple_choice_key(
        "Tungsten has an extremely high melting point.\nAnswer: C"
    ) == "C"

    assert BenchmarkMetrics.extract_multiple_choice_key(
        "Water potential gradient drives osmosis across lipid bilayers.\nAnswer: [D]"
    ) == "D"

    assert BenchmarkMetrics.extract_multiple_choice_key(
        "According to Kepler's Third Law, P^2 is proportional to a^3. Conclusion: A"
    ) == "A"


def test_extract_multiple_choice_closing_priority_over_distractors():
    """Verifies that concluding answers override intermediate mentions of candidate letters."""
    complex_cot = (
        "Let us examine all options:\n"
        "Option A suggests geothermal heat, which is incorrect.\n"
        "Option B suggests greenhouse insulation, which is not the primary purpose.\n"
        "Option D mentions muscle energy breakdown, which is cellular respiration.\n"
        "Therefore, the correct choice is (C)."
    )
    assert BenchmarkMetrics.extract_multiple_choice_key(complex_cot) == "C"

    multi_anchor_cot = (
        "Initially we might consider Choice A.\n"
        "However, upon deeper physical calculation, answer is: B"
    )
    assert BenchmarkMetrics.extract_multiple_choice_key(multi_anchor_cot) == "B"


def test_extract_multiple_choice_no_false_positive_on_prose_lowercase():
    """Verifies that lowercase English article 'a' or variable 'd' in prose do NOT falsely extract A or D."""
    # Lowercase 'a' in natural language explanations
    text_a = "a planet revolves around a distant star in a stable elliptical orbit."
    assert BenchmarkMetrics.extract_multiple_choice_key(text_a) is None

    # Lowercase 'd' as algebraic variable
    text_d = "Given that d = 1.0 m is the distance between the two slits."
    assert BenchmarkMetrics.extract_multiple_choice_key(text_d) is None

    # Combination of lowercase 'a' and 'd' in scientific explanation without choice declaration
    text_ad = "a meteorite traveling with velocity v covers distance d before atmospheric ablation."
    assert BenchmarkMetrics.extract_multiple_choice_key(text_ad) is None


def test_extract_multiple_choice_designated_single_keys():
    """Verifies that isolated or directly designated choice keys still succeed."""
    assert BenchmarkMetrics.extract_multiple_choice_key("A") == "A"
    assert BenchmarkMetrics.extract_multiple_choice_key("(B)") == "B"
    assert BenchmarkMetrics.extract_multiple_choice_key("[C]") == "C"
    assert BenchmarkMetrics.extract_multiple_choice_key("D.") == "D"
    assert BenchmarkMetrics.extract_multiple_choice_key("b") == "B"
    assert BenchmarkMetrics.extract_multiple_choice_key("Answer: a") == "A"


def test_extract_multiple_choice_concluding_lines_and_markdown():
    """Verifies parenthesized, bolded, and boxed options on trailing lines."""
    cot_bold = "1. First principle analysis...\n2. Verification step...\n**B**"
    assert BenchmarkMetrics.extract_multiple_choice_key(cot_bold) == "B"

    cot_boxed = "The calculation yields the result.\n\\boxed{C}"
    assert BenchmarkMetrics.extract_multiple_choice_key(cot_boxed) == "C"

    cot_trailing_line = "Reasoning line 1\nReasoning line 2\n(D)\n"
    assert BenchmarkMetrics.extract_multiple_choice_key(cot_trailing_line) == "D"


# =============================================================================
# Section 3: Numeric-Aware Evaluation & Unit Lenience Tests
# =============================================================================

def test_numeric_match_score_var_track_256k_forensic_case():
    """Forensic verification of the failed 256k long variable tracking sample."""
    quanta_output = (
        "1. Event 1: Dispatched 120 microprocessors (-120)...\n"
        "2. Event 2: Received 250 microprocessors (+250)...\n"
        "3. Event 3: Received 500 microprocessors (+500)...\n"
        "**Calculation:** $-120 + 250 + 500 = 630$.\n"
        "The net change in microprocessor inventory at Warehouse WH-WEST is **630**."
    )
    gold = "630 microprocessors"

    # Verifies numeric quantity + noun stem matching
    assert BenchmarkMetrics.numeric_match_score(quanta_output, gold) is True
    # Verifies that it is not considered a hallucination
    assert BenchmarkMetrics.is_hallucinated(quanta_output, gold) is False


def test_numeric_match_score_unit_lenience():
    """Verifies singular/plural stem lenience across diverse physical/financial units."""
    # Plural in gold, singular in prediction
    gold_credits = "1700 credits"
    pred_credits = (
        "Transactions: +1500, -350, +600, -50.\n"
        "The net balance change for Account ACC-9042 across all recorded transactions is 1700 credit."
    )
    assert BenchmarkMetrics.numeric_match_score(pred_credits, gold_credits) is True

    # Plural in gold, singular stem in prediction
    gold_time = "90 minutes"
    pred_time = "Cluster maintenance log indicates total downtime of **90** minute for Node-07."
    assert BenchmarkMetrics.numeric_match_score(pred_time, gold_time) is True

    # Metric abbreviation
    gold_med = "45 mg"
    pred_med = "Cumulative dosage administered to Patient P-4412 is **45** mg."
    assert BenchmarkMetrics.numeric_match_score(pred_med, gold_med) is True

    # Bare number without unit in gold
    gold_num = "33"
    pred_num = "The total count of unauthorized SSH attempts is 33."
    assert BenchmarkMetrics.numeric_match_score(pred_num, gold_num) is True


def test_numeric_match_score_calculation_anchor():
    """Verifies calculation anchors like '= 630' or 'total is 630'."""
    pred_anchor = "Summing the items: 10 + 20 + 33 = 63."
    assert BenchmarkMetrics.numeric_match_score(pred_anchor, "63 items") is True

    pred_total = "The aggregated total is 1,700 credits."
    assert BenchmarkMetrics.numeric_match_score(pred_total, "1700 credits") is True


def test_numeric_match_score_rejects_errors():
    """Verifies that mathematical errors and mismatched units are rejected."""
    # Wrong calculated value
    pred_wrong_val = (
        "Dispatched 120, received 250, received 500.\n"
        "The net change in microprocessor inventory is **500**."
    )
    assert BenchmarkMetrics.numeric_match_score(pred_wrong_val, "630 microprocessors") is False

    # Correct number but wrong unit / completely mismatched noun
    pred_wrong_unit = (
        "Telemetry review indicates total storage used is **630** gigabytes."
    )
    assert BenchmarkMetrics.numeric_match_score(pred_wrong_unit, "630 microprocessors") is False

    # Non-numeric gold answer
    assert BenchmarkMetrics.numeric_match_score("The key is in the kitchen.", "kitchen") is False


# =============================================================================
# Paired Evaluator Integration Tests for Sections 2 & 3
# =============================================================================

def test_paired_evaluator_evaluates_cot_science():
    """Verifies that PairedEvaluator correctly grades ARC science completions containing reasoning."""
    evaluator = PairedEvaluator(mode="mock")
    sample = BenchmarkSample(
        id="arc_test_01",
        suite="arc_science",
        prompt="Which mechanism allows cellular water balance?\nChoices:\n(A) Osmosis\n(B) Phagocytosis\nConclude your reasoning with 'Answer: [A/B/C/D]'.",
        gold_answer="A",
        expected_tokens=["A", "Osmosis"],
        token_count=50,
    )
    result = evaluator.evaluate_sample(sample)
    # Under mock, Base LLM or Quanta gets the answer
    assert result.quanta_global_result is not None
    assert result.quanta_global_result.is_correct is True
    assert result.quanta_global_result.is_hallucinated is False


def test_paired_evaluator_evaluates_numeric_var_tracking():
    """Verifies that PairedEvaluator grades numeric tracking using numeric_match_score."""
    evaluator = PairedEvaluator(mode="mock")
    sample = BenchmarkSample(
        id="var_track_256k",
        suite="long_variable_tracking",
        prompt="What is the net change in microprocessor inventory?",
        context="Audit logs...",
        gold_answer="630 microprocessors",
        expected_tokens=["630 microprocessors"],
        token_count=1000,
    )
    result = evaluator.evaluate_sample(sample)
    assert result.quanta_global_result is not None
    assert result.quanta_global_result.is_correct is True
    assert result.quanta_global_result.is_hallucinated is False
