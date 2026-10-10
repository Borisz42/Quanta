"""Unit and regression tests for Section 5 MuSiQue Representation Benchmark."""

import pytest
from scripts.benchmark_musique_ingestion import (
    load_musique_passages,
    normalize_text,
    compute_em,
    compute_f1,
    generate_reader_answer,
    run_representation_comparison,
    REPRESENTATION_VARIANTS,
)


def test_musique_passages_loading():
    """Verify loading of 16-paragraph MuSiQue multi-hop dataset."""
    q, a, passages = load_musique_passages(corpus="paragraphs")
    assert "Charles Babbage" in q
    assert a == "United Kingdom"
    assert len(passages) == 16
    assert passages[0][0] == "P_babbage_01"


def test_metrics_computation():
    """Verify EM, F1, and normalization logic."""
    assert normalize_text("The United Kingdom.") == "united kingdom"
    assert compute_em("United Kingdom", "United Kingdom") is True
    assert compute_em("Based on the text, it is located in the United Kingdom.", "United Kingdom") is True
    assert compute_em("France", "United Kingdom") is False
    assert compute_f1("United Kingdom", "United Kingdom") == 1.0
    assert compute_f1("The United Kingdom of Great Britain", "United Kingdom") > 0.5


def test_generate_reader_answer_mock():
    """Verify reader answer generator with mock fallback."""
    ans, lat_ms = generate_reader_answer("Where?", "Cambridge is in the United Kingdom.", backend="mock")
    assert ans == "United Kingdom"
    assert lat_ms >= 0.0


def test_representation_variants_registry():
    """Verify all 6 operational permutations are present."""
    assert len(REPRESENTATION_VARIANTS) == 6
    formats = [v["format"] for v in REPRESENTATION_VARIANTS]
    assert formats.count("sexpr_compact") == 2
    assert formats.count("sexpr_positional") == 2
    assert formats.count("json") == 2


def test_run_representation_comparison_mock(tmp_path):
    """Verify execution of representation comparison in mock mode."""
    out_file = tmp_path / "test_report.md"
    results = run_representation_comparison(
        format_arg="sexpr_compact",
        co_decode_arg="off",
        backend_mode="mock",
        corpus="paragraphs",
        slots=2,
        output_path=str(out_file),
    )
    assert "results" in results
    assert len(results["results"]) == 1
    r0 = results["results"][0]
    assert r0["format"] == "sexpr_compact"
    assert r0["co_decoded"] is False
    assert r0["throughput_wps"] > 0
    assert r0["gold_recall_pct"] >= 0.0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "QUANTA High-Throughput S-Expression Ingestion Benchmark Report" in content
