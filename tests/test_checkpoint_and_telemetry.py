"""Tests for Benchmark Runner Resumption, Checkpointing & Telemetry (Section 6)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest

import importlib.util

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_paired_benchmarks", str(REPO_ROOT / "scripts" / "run_paired_benchmarks.py"))
run_paired_benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_paired_benchmarks)
generate_markdown_report = run_paired_benchmarks.generate_markdown_report

from benchmarks.checkpoint import BenchmarkCheckpointManager
from benchmarks.code_evaluator import CodeEvaluator, CodeEvalResult
from benchmarks.latency_profiler import LatencyProfiler, LatencyProfilePoint
from benchmarks.metrics import BenchmarkMetrics, ComparativeScorecard
from benchmarks.paired_evaluator import (
    ConditionResult,
    EvaluationCondition,
    PairedEvaluator,
    PairedResult,
)
from benchmarks.suite_loaders import BenchmarkSample, BenchmarkSuiteLoader


def test_checkpoint_lifecycle_and_atomic_persistence():
    """Verifies that BenchmarkCheckpointManager saves samples incrementally and recovers state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_checkpoint.json"
        mgr = BenchmarkCheckpointManager(checkpoint_path=ckpt_path)

        assert not mgr.has_checkpoint()

        # Initialize run
        mgr.initialize_run(
            execution_mode="mock",
            ablation_mode="3way",
            suite_config={"arc_science": 3, "humaneval": 2},
            resume=False,
        )
        assert mgr.has_checkpoint()

        # Create dummy PairedResult
        cond_a = ConditionResult(
            condition=EvaluationCondition.BASE_LLM,
            answer="A",
            prompt_tokens=100,
            completion_tokens=10,
            prefill_ttft_s=0.03,
            generation_latency_s=0.2,
            total_e2e_latency_s=0.2,
            tokens_per_sec=50.0,
            is_correct=True,
            is_hallucinated=False,
        )
        cond_c = ConditionResult(
            condition=EvaluationCondition.QUANTA_GLOBAL,
            answer="A",
            prompt_tokens=40,
            completion_tokens=10,
            prefill_ttft_s=0.02,
            generation_latency_s=0.1,
            total_e2e_latency_s=0.12,
            tokens_per_sec=60.0,
            is_correct=True,
            is_hallucinated=False,
        )
        pr1 = PairedResult(
            id="arc_1",
            suite="arc_science",
            prompt="What is X?",
            gold_answer="A",
            token_count=100,
            base_result=cond_a,
            quanta_global_result=cond_c,
        )

        # Save single sample incrementally
        mgr.save_sample("arc_science", "arc_1", pr1)

        # Verify completed IDs
        completed = mgr.get_completed_sample_ids("arc_science")
        assert "arc_1" in completed
        assert len(completed) == 1

        # Retrieve cached sample
        retrieved_pr = mgr.get_sample_result("arc_science", "arc_1")
        assert retrieved_pr is not None
        assert retrieved_pr.id == "arc_1"
        assert retrieved_pr.base_result.is_correct is True
        assert retrieved_pr.quanta_global_result.is_correct is True

        # Simulate new manager loading existing checkpoint
        mgr2 = BenchmarkCheckpointManager(checkpoint_path=ckpt_path)
        assert mgr2.has_checkpoint()
        loaded = mgr2.load()
        assert loaded is True
        assert "arc_1" in mgr2.get_completed_sample_ids("arc_science")

        # Clear checkpoint
        mgr.clear()
        assert not mgr.has_checkpoint()


def test_code_evaluator_harness_bug_forensics():
    """Verifies that CodeEvaluator flags [HARNESS BUG] when helper functions in prompt are stripped."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        evaluator = CodeEvaluator(timeout_seconds=3.0, sandbox_dir=Path(tmp_dir))

        # Prompt defines a helper function before non-signature instruction
        # (simulating a harness that fails to extract helper before entry_point)
        prompt = (
            "def is_palindrome(string: str) -> bool:\n"
            "    return string == string[::-1]\n\n\n"
            "# Please implement make_palindrome(string: str) -> str:\n"
            '    """ Find shortest palindrome """\n'
        )
        # Model provides self-contained entry_point that relies on is_palindrome
        completion = (
            "def make_palindrome(string: str) -> str:\n"
            "    if not string:\n"
            "        return ''\n"
            "    beginning_of_suffix = 0\n"
            "    while not is_palindrome(string[beginning_of_suffix:]):\n"
            "        beginning_of_suffix += 1\n"
            "    return string + string[:beginning_of_suffix][::-1]\n"
        )
        test_code = (
            "def check(candidate):\n"
            "    assert candidate('racecar') == 'racecar'\n"
            "    assert candidate('cat') == 'catac'\n"
        )

        result = evaluator.evaluate_solution(
            task_id="HumanEval/10",
            prompt=prompt,
            completion=completion,
            test_code=test_code,
            entry_point="make_palindrome",
        )

        # In current harness, since 'def make_palindrome' is in completion, prompt was stripped
        # which triggers NameError on is_palindrome.
        # But evaluator should have caught and flagged is_harness_bug = True!
        assert result.passed is False
        assert result.is_harness_bug is True
        assert result.harness_bug_detail is not None
        assert "is_palindrome" in result.harness_bug_detail


def test_numeric_match_score_forensics():
    """Verifies that BenchmarkMetrics.numeric_match_score identifies numeric passes with formatting separation."""
    # 1. 256k microprocessor tracking case
    gold_1 = "630 microprocessors"
    pred_1 = (
        "1. Event 1: Dispatched 120 microprocessors (-120)... "
        "2. Event 2: Received 250 microprocessors (+250)... "
        "3. Event 3: Received 500 microprocessors (+500)...\n"
        "**Calculation:** -120 + 250 + 500 = 630.\n"
        "The net change in microprocessor inventory at Warehouse WH-WEST is **630**."
    )
    # String exact match and substring match fail:
    assert not BenchmarkMetrics.exact_match_score(pred_1, gold_1)
    assert gold_1.lower() not in pred_1.lower()
    # Numeric match passes:
    assert BenchmarkMetrics.numeric_match_score(pred_1, gold_1) is True

    # 2. Account credits aggregation case
    gold_2 = "1700 credits"
    pred_2 = "Calculation: 1500 - 400 - 350 + 900 + 600 - 50 = 1700. Verified total balance change is 1700 credits."
    assert BenchmarkMetrics.numeric_match_score(pred_2, gold_2) is True

    # 3. Numeric mismatch should fail
    pred_wrong_val = "Total microprocessor balance is 500 microprocessors."
    assert BenchmarkMetrics.numeric_match_score(pred_wrong_val, gold_1) is False

    # 4. Unit mismatch should fail
    pred_wrong_unit = "Total inventory balance is 630 widgets."
    assert BenchmarkMetrics.numeric_match_score(pred_wrong_unit, gold_1) is False


def test_paired_evaluator_forensic_tagging_in_mock():
    """Verifies that PairedEvaluator tags [HARNESS BUG] and [NUMERIC PASS] in mock mode."""
    loader = BenchmarkSuiteLoader()
    evaluator = PairedEvaluator(mode="mock", ablation_mode="3way")

    # 1. HumanEval/10 should flag [HARNESS BUG] for QUANTA
    can_sol = (
        "    if not string:\n"
        "        return ''\n"
        "    beginning_of_suffix = 0\n"
        "    while not is_palindrome(string[beginning_of_suffix:]):\n"
        "        beginning_of_suffix += 1\n"
        "    return string + string[:beginning_of_suffix][::-1]\n"
    )
    he_sample = BenchmarkSample(
        id="HumanEval/10",
        suite="humaneval",
        prompt=(
            "def is_palindrome(string: str) -> bool:\n"
            "    return string == string[::-1]\n\n\n"
            "# Please implement make_palindrome(string: str) -> str:\n"
        ),
        context="",
        gold_answer="",
        token_count=50,
        metadata={
            "entry_point": "make_palindrome",
            "canonical_solution": can_sol,
            "test_code": "def check(candidate):\n    assert candidate('aba') == 'aba'\n    assert candidate('cat') == 'catac'\n",
        },
    )
    res_he = evaluator.evaluate_sample(he_sample)
    assert res_he.diagnostic_tag == "[HARNESS BUG]"
    assert res_he.base_result.is_correct is True
    assert res_he.quanta_global_result.is_correct is False
    assert res_he.quanta_global_result.diagnostic_tag == "[HARNESS BUG]"

    # 2. var_track_256k should flag [NUMERIC PASS]
    var_sample = BenchmarkSample(
        id="var_track_256k",
        suite="long_variable_tracking",
        prompt="What is the net change in microprocessor inventory at Warehouse WH-WEST?",
        context="Lots of text...",
        gold_answer="630 microprocessors",
        token_count=256000,
    )
    res_var = evaluator.evaluate_sample(var_sample)
    assert res_var.diagnostic_tag == "[NUMERIC PASS]"
    assert res_var.quanta_global_result.is_correct is True
    assert res_var.quanta_global_result.diagnostic_tag == "[NUMERIC PASS]"


def test_generate_markdown_report_includes_forensics_section():
    """Verifies that generate_markdown_report generates Section 5 with forensic details."""
    profiler = LatencyProfiler()
    profiler.record_datapoint(100, 0.01, 1.0, 0.02, 0.02, 0.1, 0.1)

    scorecard = ComparativeScorecard(
        suite_name="humaneval",
        sample_count=1,
        base_accuracy_pct=100.0,
        base_hallucination_pct=0.0,
        base_mean_prompt_tokens=50,
        base_mean_latency_s=0.2,
        base_mean_tps=50.0,
        quanta_local_accuracy_pct=0.0,
        quanta_local_prompt_tokens=30,
        quanta_local_latency_s=0.18,
        quanta_global_accuracy_pct=0.0,
        quanta_global_hallucination_pct=100.0,
        quanta_global_prompt_tokens=30,
        quanta_global_latency_s=0.18,
        quanta_global_tps=55.0,
        token_compression_pct=40.0,
        speedup_ratio=1.1,
        accuracy_lift_pct=-100.0,
    )

    raw_preds = {
        "humaneval": [
            {
                "id": "HumanEval/10",
                "suite": "humaneval",
                "prompt": "def make_palindrome(...)",
                "gold_answer": "make_palindrome",
                "diagnostic_tag": "[HARNESS BUG]",
                "forensic_notes": "Harness dropped is_palindrome helper function",
                "base": {"answer": "def is_palindrome...", "correct": True},
                "quanta_global": {"answer": "def make_palindrome...", "correct": False, "diagnostic_tag": "[HARNESS BUG]"},
            }
        ]
    }

    report = generate_markdown_report(
        scorecards=[scorecard],
        latency_profiler=profiler,
        ablation_mode="3way",
        execution_mode="mock",
        raw_predictions=raw_preds,
    )

    assert "## 5. Evaluation Discrepancies & Forensic Diagnostics" in report
    assert "HumanEval/10" in report
    assert "[HARNESS BUG]" in report
    assert "Harness dropped is_palindrome helper function" in report


def test_checkpoint_resumption_cli():
    """End-to-end test verifying CLI checkpoint creation and --resume skipping."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "bench_ckpt.json"

        # Step 1: Initial run with 2 samples of arc_science
        cmd1 = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_paired_benchmarks.py"),
            "--suite", "arc_science:2",
            "--mode", "mock",
            "--checkpoint-file", str(ckpt_path),
        ]
        res1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert res1.returncode == 0
        assert ckpt_path.exists()

        # Check checkpoint content
        data1 = json.loads(ckpt_path.read_text(encoding="utf-8"))
        assert "arc_science" in data1["results"]
        assert len(data1["results"]["arc_science"]) == 2

        # Step 2: Resume with --resume and expand to 3 samples
        cmd2 = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_paired_benchmarks.py"),
            "--suite", "arc_science:3",
            "--mode", "mock",
            "--checkpoint-file", str(ckpt_path),
            "--resume",
        ]
        res2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert res2.returncode == 0
        assert "[CACHED]" in res2.stdout
        assert "Resuming from checkpoint" in res2.stdout

        # Check that checkpoint now contains 3 samples
        data2 = json.loads(ckpt_path.read_text(encoding="utf-8"))
        assert len(data2["results"]["arc_science"]) == 3

