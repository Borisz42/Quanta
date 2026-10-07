"""Automated test suite for Kev-4B comparative ablation runner (Section 4).

Verifies:
1. Comparative ablation runner executes successfully over gold benchmarks.
2. Approach A meets the minimum accuracy threshold (Macro F1 >= 92.0%).
3. Approach A meets the probability calibration threshold (ECE <= 0.08).
4. Approach A demonstrates 0.0 MB additional VRAM and 0.0 ms adapter switching latency.
5. Markdown report is generated with comprehensive tables and sections.
"""

from pathlib import Path
import sys
import pytest

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.evaluate_kev_approaches import (
    evaluate_kev_approaches,
    generate_relation_benchmark,
    generate_speech_epistemic_benchmark,
    generate_thematic_valency_benchmark,
)


def test_gold_dataset_generation():
    """Verify gold benchmark datasets generate correct sample counts and labels."""
    val_data = generate_thematic_valency_benchmark(100)
    assert len(val_data) == 100
    roles = {s.gold_role for s in val_data}
    assert roles == {"AGENT", "PATIENT", "INSTRUMENT", "NONE"}

    speech_data = generate_speech_epistemic_benchmark(50)
    assert len(speech_data) == 50
    intents = {s.gold_intent for s in speech_data}
    epistemics = {s.gold_epistemic for s in speech_data}
    assert "INFORMATIVE" in intents
    assert "DIRECT_OBSERVATION" in epistemics

    rel_data = generate_relation_benchmark(50)
    assert len(rel_data) == 50
    allens = {s.gold_allen for s in rel_data}
    pearls = {s.gold_pearl for s in rel_data}
    assert "BEFORE" in allens
    assert "MECHANISM_LINK" in pearls


def test_ablation_evaluation_execution_and_thresholds(tmp_path):
    """Run full comparative ablation sweep and verify strict Section 4 quality bars."""
    results = evaluate_kev_approaches(
        server_url=None,
        output_dir=tmp_path,
        num_samples_valency=100,
        num_samples_speech=50,
        num_samples_relations=50,
    )

    metrics_a = results["approach_a"]
    metrics_b = results["approach_b"]
    report_path = Path(results["report_path"])

    # 1. Verification of Approach A Performance Bars (Section 4 requirement: >= 92% F1, <= 0.08 ECE)
    assert metrics_a.overall_macro_f1 >= 0.92, (
        f"Approach A Macro F1 ({metrics_a.overall_macro_f1:.4f}) failed to meet the >= 92% threshold"
    )
    assert metrics_a.overall_ece <= 0.08, (
        f"Approach A ECE ({metrics_a.overall_ece:.4f}) exceeded the <= 0.08 calibration threshold"
    )

    # 2. Hardware and Operational Footprint Verification
    assert metrics_a.vram_overhead_mb == 0.0  # Shared weights require 0.0 MB
    assert metrics_a.adapter_switch_latency_ms == 0.0
    assert metrics_b.vram_overhead_mb == 30.0
    assert metrics_b.adapter_switch_latency_ms > 0.0

    # 3. Report Generation & Structural Verification
    assert report_path.is_file()
    content = report_path.read_text(encoding="utf-8")
    assert "# Kev-4B Non-Autoregressive Relational Engine" in content
    assert "Comparative Matrix: Approach A vs. Approach B" in content
    assert "Pass 1: Thematic Valency & Coreference" in content
    assert "Pass 2: Theory of Mind Speech-Act Intent & Epistemics" in content
    assert "Pass 3: Spatio-Temporal Allen Intervals & Pearl Causal Links" in content
    assert "Belnap Lattice Calibration Readiness" in content
    assert "Architectural Conclusion" in content


def test_production_output_report_exists():
    """Verify standard production report in output/ directory is present and valid."""
    repo_root = Path(__file__).resolve().parents[1]
    prod_report = repo_root / "output" / "kev_approach_ablation_report.md"
    assert prod_report.is_file(), f"Expected report at {prod_report} does not exist"

    text = prod_report.read_text(encoding="utf-8")
    assert "Overall Macro F1" in text
    assert "Expected Calibration Error (ECE)" in text
    assert "Approach A Wins (0 VRAM)" in text
