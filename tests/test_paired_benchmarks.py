"""Automated Unit and Regression Test Suite for QUANTA Paired Benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from benchmarks.code_evaluator import CodeEvaluator, CodeEvalResult
from benchmarks.latency_profiler import LatencyProfiler
from benchmarks.metrics import BenchmarkMetrics, ComparativeScorecard
from benchmarks.paired_evaluator import PairedEvaluator, EvaluationCondition
from benchmarks.publication_exporter import PublicationExporter
from benchmarks.suite_loaders import BenchmarkSuiteLoader, BenchmarkSample


def test_code_evaluator_pass():
    """Verifies that CodeEvaluator correctly identifies passing code."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        evaluator = CodeEvaluator(timeout_seconds=3.0, sandbox_dir=Path(tmp_dir))
        prompt = "def add_two(a: int, b: int) -> int:\n"
        completion = "    return a + b\n"
        test_code = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n"
        result = evaluator.evaluate_solution("test/add", prompt, completion, test_code, "add_two")
        assert result.passed is True
        assert result.error_message is None
        assert result.execution_time_s >= 0.0


def test_code_evaluator_fail():
    """Verifies that CodeEvaluator catches assertion failures."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        evaluator = CodeEvaluator(timeout_seconds=3.0, sandbox_dir=Path(tmp_dir))
        prompt = "def multiply_two(a: int, b: int) -> int:\n"
        completion = "    return a + b\n"  # Buggy implementation
        test_code = "def check(candidate):\n    assert candidate(2, 3) == 6\n"
        result = evaluator.evaluate_solution("test/mult", prompt, completion, test_code, "multiply_two")
        assert result.passed is False
        assert result.error_message is not None


def test_suite_loaders_schema():
    """Verifies that all benchmark suite loaders return standardized BenchmarkSample schemas."""
    loader = BenchmarkSuiteLoader()

    # 1. HumanEval
    he_samples = loader.load_humaneval(limit=2)
    assert len(he_samples) > 0
    assert he_samples[0].suite == "humaneval"
    assert "entry_point" in he_samples[0].metadata

    # 2. ARC-Challenge Science
    arc_samples = loader.load_arc_science(limit=2)
    assert len(arc_samples) > 0
    assert arc_samples[0].suite == "arc_science"
    assert arc_samples[0].gold_answer in ("A", "B", "C", "D")

    # 3. MuSiQue
    musique_samples = loader.load_musique(limit=2)
    assert len(musique_samples) > 0
    assert musique_samples[0].suite == "musique"
    assert len(musique_samples[0].context) > 0

    # 4. ProofWriter
    pw_samples = loader.load_proofwriter(limit=2)
    assert len(pw_samples) > 0
    assert pw_samples[0].suite == "proofwriter"

    # 5. bAbI
    babi_samples = loader.load_babi_state_tracking(limit=2)
    assert len(babi_samples) > 0
    assert babi_samples[0].suite == "babi"

    # 6. SQuAD Overhead
    squad_samples = loader.load_squad_overhead(limit=2)
    assert len(squad_samples) > 0
    assert squad_samples[0].suite == "squad_overhead"

    # 7. NIAH Long Context
    niah_samples = loader.generate_niah_samples([4000, 16000])
    assert len(niah_samples) == 2
    assert niah_samples[0].token_count >= 2000

    # 8. BABILong Multi-Hop State Tracking
    babilong_samples = loader.load_babilong(limit=2)
    assert len(babilong_samples) == 2
    assert babilong_samples[0].suite == "babilong"
    assert babilong_samples[0].gold_answer in ("bedroom", "cellar", "library", "observatory", "attic")
    assert babilong_samples[0].token_count >= 2000

    # 9. Long Variable Tracking & Aggregation
    var_samples = loader.load_variable_tracking(limit=2)
    assert len(var_samples) == 2
    assert var_samples[0].suite == "long_variable_tracking"
    assert len(var_samples[0].gold_answer) > 0
    assert var_samples[0].token_count >= 2000


def test_latency_profiler_and_extrapolation():
    """Verifies that LatencyProfiler computes regressions and forecasts up to 1M tokens."""
    profiler = LatencyProfiler()
    # Add empirical points
    profiler.record_datapoint(500, 0.05, 1.2, 0.03, 0.025, 0.5, 0.3)
    profiler.record_datapoint(2000, 0.15, 1.4, 0.04, 0.026, 0.7, 0.4)
    profiler.record_datapoint(10000, 0.65, 1.8, 0.09, 0.028, 1.2, 0.5)

    a, b, r2 = profiler.fit_linear_ingestion_regression()
    assert a > 0.0
    assert b > 0.0
    assert 0.0 <= r2 <= 1.0

    forecasts = profiler.generate_extrapolation_forecast([10000, 50000, 100000, 1000000])
    assert len(forecasts) == 4
    assert forecasts[0].target_tokens == 10000
    assert forecasts[3].target_tokens == 1000000
    assert forecasts[3].predicted_ingestion_s > forecasts[0].predicted_ingestion_s


def test_paired_evaluator_3way_mock():
    """Verifies 3-way evaluation in mock mode."""
    loader = BenchmarkSuiteLoader()
    samples = loader.load_arc_science(limit=3)

    evaluator = PairedEvaluator(mode="mock", ablation_mode="3way")
    scorecard = evaluator.evaluate_suite("arc_science", samples)

    assert scorecard.sample_count == 3
    assert scorecard.quanta_global_accuracy_pct >= scorecard.base_accuracy_pct
    assert scorecard.token_compression_pct >= 0.0
    row_md = scorecard.format_markdown_row()
    assert "| **arc_science** |" in row_md


def test_publication_exporter():
    """Verifies LaTeX, BibTeX, and Leaderboard submission generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exporter = PublicationExporter(output_dir=Path(tmp_dir))

        # 1. Statistical significance
        pval, stars = exporter.calculate_statistical_significance([0, 1, 0, 0], [1, 1, 1, 1])
        assert pval <= 0.5
        assert isinstance(stars, str)

        # 2. Scorecard LaTeX export
        sc = ComparativeScorecard(
            suite_name="musique",
            sample_count=20,
            base_accuracy_pct=45.0,
            base_hallucination_pct=25.0,
            base_mean_prompt_tokens=1500,
            base_mean_latency_s=1.25,
            base_mean_tps=45.0,
            quanta_local_accuracy_pct=85.0,
            quanta_local_prompt_tokens=350,
            quanta_local_latency_s=0.85,
            quanta_global_accuracy_pct=95.0,
            quanta_global_hallucination_pct=0.0,
            quanta_global_prompt_tokens=350,
            quanta_global_latency_s=0.80,
            quanta_global_tps=55.0,
            token_compression_pct=76.7,
            speedup_ratio=1.56,
            accuracy_lift_pct=50.0,
            p_value=0.001,
            significance_str="$^{***}$",
        )
        paths = exporter.export_latex_tables([sc])
        assert "main_scorecard" in paths
        assert paths["main_scorecard"].exists()
        content = paths["main_scorecard"].read_text(encoding="utf-8")
        assert "\\textbf{musique}" in content

        # 3. BibTeX export
        bib_p = exporter.generate_bibtex()
        assert bib_p.exists()
        bib_text = bib_p.read_text(encoding="utf-8")
        assert "trivedi2022musique" in bib_text
        assert "chen2021humaneval" in bib_text

        # 4. Leaderboard submissions
        mock_raw = {
            "musique": [{"id": "m1", "quanta_global_answer": "Cambridge"}],
            "humaneval": [{"id": "HumanEval/0", "quanta_global_answer": "    return True"}],
            "arc_science": [{"id": "arc_1", "quanta_global_answer": "A", "gold_answer": "A", "quanta_global_correct": True}],
        }
        subs = exporter.export_leaderboard_submissions(mock_raw)
        assert "musique" in subs
        assert "humaneval" in subs
        assert "arc_science" in subs
        assert subs["musique"].exists()
        musique_json = json.loads(subs["musique"].read_text(encoding="utf-8"))
        assert "musique_ans_preds" in musique_json
        assert musique_json["musique_ans_preds"]["m1"] == "Cambridge"


def test_publication_exporter_formatting_and_sanitization():
    """Verifies that format_reduction_pct, format_significance_latex, and sanitize_musique_answer adhere to Section 5 requirements."""
    # 1. Negative zero suppression
    assert PublicationExporter.format_reduction_pct(0.0) == "0.0\\%"
    assert PublicationExporter.format_reduction_pct(-0.0) == "0.0\\%"
    assert PublicationExporter.format_reduction_pct(0.02) == "0.0\\%"
    assert PublicationExporter.format_reduction_pct(-0.04) == "0.0\\%"
    assert PublicationExporter.format_reduction_pct(76.7) == "-76.7\\%"
    assert PublicationExporter.format_reduction_pct(-12.4) == "+12.4\\%"

    # 2. Statistical significance superscripts (valid LaTeX math mode)
    assert PublicationExporter.format_significance_latex("$^{***}$") == "$^{***}$"
    assert PublicationExporter.format_significance_latex("***") == "$^{***}$"
    assert PublicationExporter.format_significance_latex("**") == "$^{**}$"
    assert PublicationExporter.format_significance_latex("*") == "$^{*}$"
    assert PublicationExporter.format_significance_latex("(n.s.)") == "$^{\\text{n.s.}}$"
    assert PublicationExporter.format_significance_latex("", pval=0.0005) == "$^{***}$"
    assert PublicationExporter.format_significance_latex("", pval=0.008) == "$^{**}$"
    assert PublicationExporter.format_significance_latex("", pval=0.04) == "$^{*}$"
    assert PublicationExporter.format_significance_latex("", pval=0.25) == "$^{\\text{n.s.}}$"

    # 3. MuSiQue CodaLab concise answer extraction
    assert PublicationExporter.sanitize_musique_answer(
        "Based on Document [1] and Document [4], the founder of The Walt Disney Company is **Walt Disney**."
    ) == "Walt Disney"
    assert PublicationExporter.sanitize_musique_answer("**Walt Disney**") == "Walt Disney"
    assert PublicationExporter.sanitize_musique_answer("**Answer:** **Richland County**") == "Richland County"
    assert PublicationExporter.sanitize_musique_answer("**Answer:** Richland County.") == "Richland County"
    assert PublicationExporter.sanitize_musique_answer("Answer: Walt Disney") == "Walt Disney"
    assert PublicationExporter.sanitize_musique_answer("Therefore, the correct answer is Paris.") == "Paris"
    assert PublicationExporter.sanitize_musique_answer("Based on the provided documents, the answer is Cambridge.") == "Cambridge"
    assert PublicationExporter.sanitize_musique_answer("According to the text, the capital of France is Paris.") == "Paris"
    assert PublicationExporter.sanitize_musique_answer("Cambridge") == "Cambridge"


def test_leaderboard_submission_sanitization():
    """Verifies that export_leaderboard_submissions produces sanitized files for HumanEval, MuSiQue, and ARC-Challenge."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exporter = PublicationExporter(output_dir=Path(tmp_dir))

        raw_predictions = {
            "humaneval": [
                {
                    "id": "HumanEval/10",
                    "entry_point": "make_palindrome",
                    "quanta_global": {
                        "answer": "```python\ndef is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n\ndef make_palindrome(string: str) -> str:\n    if not string:\n        return ''\n    return string\n```",
                        "correct": True,
                    },
                },
                {
                    "id": "HumanEval/11",
                    "metadata": {"entry_point": "string_xor"},
                    "quanta_global": {
                        "answer": "    return ''.join('1' if a != b else '0' for a, b in zip(a, b))",
                        "correct": True,
                    },
                },
            ],
            "musique": [
                {
                    "id": "m_1",
                    "quanta_global": {
                        "answer": "Based on Document [3] and Document [5], the answer is **Lexington County**.",
                    },
                },
                {
                    "id": "m_2",
                    "quanta_global": {
                        "answer": "According to the context, the founder is Walt Disney.",
                    },
                },
            ],
            "arc_science": [
                {
                    "id": "ARC_1",
                    "gold_answer": "B",
                    "quanta_global": {
                        "answer": "The correct choice is (B). It indicates redshift.",
                        "correct": True,
                    },
                }
            ],
        }

        subs = exporter.export_leaderboard_submissions(raw_predictions)
        assert "humaneval" in subs
        assert "musique" in subs
        assert "arc_science" in subs

        # Check HumanEval JSONL: bare Python strings without markdown ``` fences
        he_lines = [json.loads(line) for line in subs["humaneval"].read_text(encoding="utf-8").strip().splitlines()]
        assert len(he_lines) == 2
        assert "```" not in he_lines[0]["completion"]
        assert "def make_palindrome" in he_lines[0]["completion"]
        assert "def is_palindrome" in he_lines[0]["completion"]  # helper preserved
        assert he_lines[1]["completion"].startswith("    return")

        # Check MuSiQue CodaLab JSON: concise entity answers
        musique_json = json.loads(subs["musique"].read_text(encoding="utf-8"))
        assert musique_json["musique_ans_preds"]["m_1"] == "Lexington County"
        assert musique_json["musique_ans_preds"]["m_2"] == "Walt Disney"

        # Check ARC-Challenge: choice letter extracted
        arc_json = json.loads(subs["arc_science"].read_text(encoding="utf-8"))
        assert arc_json["ARC_1"]["prediction"] == "B"
        assert arc_json["ARC_1"]["correct"] is True


def test_latex_table_zero_suppression_and_bolding():
    """Verifies that LaTeX tables suppress -0.0% and correctly highlight winning columns in bold."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exporter = PublicationExporter(output_dir=Path(tmp_dir))

        scorecards = [
            # Case 1: 0% reduction, Base wins accuracy
            ComparativeScorecard(
                suite_name="humaneval",
                sample_count=15,
                base_accuracy_pct=100.0,
                base_hallucination_pct=0.0,
                base_mean_prompt_tokens=400,
                base_mean_latency_s=0.50,
                base_mean_tps=50.0,
                quanta_local_accuracy_pct=93.3,
                quanta_local_prompt_tokens=400,
                quanta_local_latency_s=0.48,
                quanta_global_accuracy_pct=93.3,
                quanta_global_hallucination_pct=0.0,
                quanta_global_prompt_tokens=400,
                quanta_global_latency_s=0.45,
                quanta_global_tps=52.0,
                token_compression_pct=0.0,  # Exactly 0
                speedup_ratio=1.11,
                accuracy_lift_pct=-6.7,
                p_value=0.317,
                significance_str="(n.s.)",
            ),
            # Case 2: Positive reduction, QUANTA wins accuracy with significance
            ComparativeScorecard(
                suite_name="musique",
                sample_count=20,
                base_accuracy_pct=45.0,
                base_hallucination_pct=25.0,
                base_mean_prompt_tokens=2500,
                base_mean_latency_s=2.50,
                base_mean_tps=40.0,
                quanta_local_accuracy_pct=80.0,
                quanta_local_prompt_tokens=750,
                quanta_local_latency_s=1.20,
                quanta_global_accuracy_pct=95.0,
                quanta_global_hallucination_pct=0.0,
                quanta_global_prompt_tokens=750,
                quanta_global_latency_s=1.10,
                quanta_global_tps=55.0,
                token_compression_pct=70.0,
                speedup_ratio=2.27,
                accuracy_lift_pct=50.0,
                p_value=0.0005,
                significance_str="$^{***}$",
            ),
        ]

        paths = exporter.export_latex_tables(scorecards)
        main_tex = paths["main_scorecard"].read_text(encoding="utf-8")
        kb_tex = paths["kb_ablation"].read_text(encoding="utf-8")

        # No negative zero in LaTeX
        assert "-0.0\\%" not in main_tex
        assert "+0.0\\%" not in kb_tex
        assert "-0.0\\%" not in kb_tex
        assert "0.0\\%" in main_tex

        # In HumanEval: Base accuracy is 100.0%, QUANTA is 93.3%. Base should be bolded
        assert "\\textbf{100.0\\%}" in main_tex

        # In MuSiQue: QUANTA accuracy is 95.0%, Base is 45.0%. QUANTA should be bolded with $^{***}$
        assert "\\textbf{95.0\\%}$^{***}$" in main_tex

        # MuSiQue reduction should be -70.0%
        assert "-70.0\\%" in main_tex

