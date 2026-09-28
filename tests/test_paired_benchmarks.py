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
