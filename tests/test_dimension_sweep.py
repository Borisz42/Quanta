"""Tests for automated dimension sweep script (src/scripts/run_dimension_sweep.py)."""

import json
from pathlib import Path
import pytest
import numpy as np

from scripts.run_dimension_sweep import (
    benchmark_collision_rate,
    benchmark_rate_distortion_entropy,
    benchmark_solver_grounding_latency,
    benchmark_simd_retrieval_throughput,
    generate_synthetic_dimension_matrix,
    run_full_dimension_sweep,
    export_sweep_artifacts,
)


def test_synthetic_dimension_matrix_projection():
    rng = np.random.default_rng(42)
    base = rng.integers(0, 4, size=(100, 1024), dtype=np.uint8)

    # Down-projection to 64
    d64 = generate_synthetic_dimension_matrix(base, target_dim=64, rng=rng)
    assert d64.shape == (100, 64)
    assert np.all(np.isin(d64, [0, 1, 2, 3]))

    # Up-projection to 2048
    d2048 = generate_synthetic_dimension_matrix(base, target_dim=2048, rng=rng)
    assert d2048.shape == (100, 2048)
    assert np.all(np.isin(d2048, [0, 1, 2, 3]))


def test_collision_rate_benchmark():
    rng = np.random.default_rng(42)
    # Perfectly unique random 1024-d vectors
    mat_1024 = rng.integers(0, 4, size=(500, 1024), dtype=np.uint8)
    res_1024 = benchmark_collision_rate(mat_1024, dim=1024)

    assert res_1024["dimension"] == 1024
    assert res_1024["num_samples"] == 500
    assert res_1024["collision_rate"] == 0.0
    assert res_1024["unique_cids"] == 500

    # Low-dimensional matrix with forced duplicates
    single_col = rng.integers(0, 4, size=(500, 64), dtype=np.uint8)
    single_col[10:20] = single_col[0]  # Force duplicates
    res_64 = benchmark_collision_rate(single_col, dim=64)
    assert res_64["unique_cids"] < 500
    assert res_64["collision_pairs"] > 0


def test_entropy_and_total_correlation_benchmark():
    rng = np.random.default_rng(42)
    mat = rng.integers(0, 4, size=(500, 256), dtype=np.uint8)
    res = benchmark_rate_distortion_entropy(mat, dim=256)

    assert res["dimension"] == 256
    assert res["sum_marginal_entropy_bits"] > 0.0
    assert res["joint_entropy_bits"] > 0.0
    assert res["total_correlation_bits"] >= 0.0
    assert 0.0 <= res["representational_efficiency"] <= 1.0


def test_solver_grounding_latency_benchmark():
    rng = np.random.default_rng(42)
    mat = rng.integers(0, 4, size=(50, 1024), dtype=np.uint8)
    res = benchmark_solver_grounding_latency(mat, dim=1024, num_trials=10)

    assert res["dimension"] == 1024
    assert res["mean_latency_ms"] > 0.0
    assert res["grounding_throughput_qps"] > 0.0


def test_simd_retrieval_throughput_benchmark():
    res = benchmark_simd_retrieval_throughput(dim=1024, num_nodes=50_000, batch_size=10_000)

    assert res["dimension"] == 1024
    assert res["nodes_scanned"] == 50_000
    assert res["throughput_m_nodes_sec"] > 0.0
    assert res["memory_bandwidth_gb_sec"] > 0.0
    assert res["ram_footprint_mb"] > 0.0


def test_end_to_end_sweep_and_export(tmp_path):
    dims = [64, 256, 1024]
    sweep_results = run_full_dimension_sweep(
        dimensions=dims,
        num_samples=100,
        simd_nodes=10_000,
        seed=42,
    )

    assert len(sweep_results["results"]) == 3
    assert 1024 in sweep_results["results"]

    json_p = tmp_path / "test_results.json"
    csv_p = tmp_path / "test_results.csv"
    report_p = tmp_path / "test_report.md"

    export_sweep_artifacts(sweep_results, json_p, csv_p, report_p)

    assert json_p.exists()
    assert csv_p.exists()
    assert report_p.exists()

    with open(json_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "results" in data
    assert "1024" in data["results"]

    report_text = report_p.read_text(encoding="utf-8")
    assert "QUANTA Empirical Dimension Sweep Report" in report_text
    assert "Curve 1: Anchor-Free Collision Rate" in report_text
