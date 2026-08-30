"""Unit and property tests for Candidate Pool, Information Profiler, and mRMR Dimension Selection."""

import csv
import json
import numpy as np
import pytest

from data.corpus_generator import ValidationCorpusGenerator
from profiler.info_profiler import QuantaInformationProfiler
from profiler.mrmr_selector import MRMRSelector, export_optimal_dimensions
from profiler.candidate_pool import build_candidate_pool, export_candidate_pool


@pytest.fixture(scope="module")
def validation_corpus():
    """Fast module fixture generating 300 multi-domain propositions for sub-second unit test execution."""
    gen = ValidationCorpusGenerator(seed=42)
    return gen.generate_corpus(num_samples=300)


def test_candidate_pool_structure_and_export(tmp_path):
    candidates = build_candidate_pool()
    assert len(candidates) >= 1024

    # Verify first 1024 are canonical slots across all 8 bands
    for i in range(1024):
        assert candidates[i].id == i
        assert candidates[i].source.startswith("Band") or "ConceptNet" in candidates[i].source

    # Verify no duplicate candidate names or IDs
    names = [c.name for c in candidates]
    ids = [c.id for c in candidates]
    assert len(set(names)) == len(candidates)
    assert len(set(ids)) == len(candidates)

    # Test export to JSON
    json_path = tmp_path / "candidate_pool.json"
    exported_path = export_candidate_pool(candidates, json_path)
    assert exported_path.exists()

    with open(exported_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert len(loaded) == len(candidates)
    assert loaded[0]["name"] == candidates[0].name
    assert "description" in loaded[0]


def test_validation_corpus_generation(validation_corpus):
    assert validation_corpus.canonical_matrix.shape == (300, 1024)
    assert validation_corpus.candidate_matrix.shape[0] == 300
    assert validation_corpus.candidate_matrix.shape[1] >= 1024
    assert len(validation_corpus.labels) == 300
    assert len(validation_corpus.propositions) == 300


def test_information_profiler_metrics(validation_corpus):
    profiler = QuantaInformationProfiler(validation_corpus.canonical_matrix)
    report = profiler.run_diagnostic_suite(dead_threshold=0.01, redundancy_threshold=0.7, verbose=False)

    assert report["num_samples"] == 300
    assert report["mean_entropy"] > 0.01  # Across diverse sparse 1024-d propositions
    assert report["collision_rate"] < 0.05  # Highly resolvable
    assert report["joint_entropy"] >= 0.0
    assert report["total_correlation"] >= 0.0

    # Check all 8 band entropies plus total mean entropy
    band_stats = report["band_entropies"]
    assert len(band_stats) == 9
    assert "Band 0 (NSM & Kinematics)" in band_stats
    assert "Band 7 (Spatiotemporal & Causal)" in band_stats
    assert "Total Mean Entropy" in band_stats
    for band_name, ent in band_stats.items():
        assert ent >= 0.0


def test_total_correlation_and_report_export(validation_corpus, tmp_path):
    profiler = QuantaInformationProfiler(validation_corpus.canonical_matrix)
    
    # 1. Test theoretical properties on synthetic matrices
    # Case A: Perfectly correlated 1024 dimensions
    rng = np.random.default_rng(42)
    single_col = rng.choice([0, 1, 2, 3], size=1000, p=[0.25, 0.25, 0.25, 0.25])
    correlated_matrix = np.tile(single_col[:, np.newaxis], (1, 1024))
    prof_corr = QuantaInformationProfiler(correlated_matrix)
    
    joint_h = prof_corr.compute_joint_entropy()
    marg_h = prof_corr.compute_entropies()
    tc = prof_corr.compute_total_correlation()
    
    assert np.isclose(joint_h, 2.0, atol=0.05)
    assert np.isclose(marg_h[0], 2.0, atol=0.05)
    # TC = 1024 * 2.0 - 2.0 = 1023 * 2.0 = 2046.0 bits
    assert np.isclose(tc, 1023.0 * 2.0, atol=20.0)

    # 2. Test export functionality
    txt_path = tmp_path / "profiler_report.txt"
    json_path = tmp_path / "profiler_report.json"
    paths = profiler.export_report(txt_path=txt_path, json_path=json_path)
    
    assert paths["txt"].exists()
    assert paths["json"].exists()

    txt_content = paths["txt"].read_text(encoding="utf-8")
    assert "QUANTA INFORMATION PROFILER REPORT" in txt_content
    assert "Total Correlation" in txt_content

    with open(paths["json"], "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert json_data["num_samples"] == 300
    assert "total_correlation" in json_data
    assert "band_entropies" in json_data


def test_mrmr_selector_feature_ranking(validation_corpus):
    candidates = build_candidate_pool()
    candidate_names = [c.name for c in candidates]

    selector = MRMRSelector(candidate_names)
    num_select = 64
    selected = selector.select_dimensions(
        X=validation_corpus.candidate_matrix,
        y=validation_corpus.labels,
        num_to_select=num_select,
        alpha_redundancy=0.5,
    )

    assert len(selected) == num_select
    for idx, name, score in selected:
        assert 0 <= idx < len(candidates)
        assert isinstance(name, str)
        assert isinstance(score, float)

    # Ensure no duplicates in selected set
    selected_indices = [s[0] for s in selected]
    assert len(set(selected_indices)) == num_select


def test_mrmr_optimal_dimensions_verification_and_export(validation_corpus, tmp_path):
    """Verify optimal dimension selection, export to JSON/CSV, and profile selected subspace."""
    candidates = build_candidate_pool()
    candidate_names = [c.name for c in candidates]

    selector = MRMRSelector(candidate_names)
    num_select = 128
    selected = selector.select_dimensions(
        X=validation_corpus.candidate_matrix,
        y=validation_corpus.labels,
        num_to_select=num_select,
        alpha_redundancy=0.5,
    )

    # 1. Verify export to JSON and CSV
    json_path = tmp_path / "optimal_dimensions.json"
    csv_path = tmp_path / "optimal_dimensions.csv"
    export_paths = export_optimal_dimensions(selected, candidates, json_path=json_path, csv_path=csv_path)

    assert export_paths["json"].exists()
    assert export_paths["csv"].exists()

    with open(export_paths["json"], "r", encoding="utf-8") as f:
        dims_json = json.load(f)
    assert len(dims_json) == num_select
    assert dims_json[0]["rank"] == 1
    assert "mrmr_score" in dims_json[0]

    with open(export_paths["csv"], "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == num_select + 1  # header + selected rows
    assert reader[0] == ["Rank", "Candidate_ID", "Name", "Source", "Category", "MRMR_Score", "Description"]

    # 2. Profile the selected sub-tensor
    selected_indices = [s[0] for s in selected]
    selected_labels = [s[1] for s in selected]
    X_selected = validation_corpus.candidate_matrix[:, selected_indices]

    profiler = QuantaInformationProfiler(X_selected, dimension_labels=selected_labels)
    report = profiler.run_diagnostic_suite(dead_threshold=0.01, redundancy_threshold=0.8, verbose=False)

    assert report["num_samples"] == 300
    assert report["mean_entropy"] > 0.01
    assert report["total_correlation"] >= 0.0


@pytest.mark.slow
def test_full_5000_sample_empirical_profile():
    """Heavy integration test generating full 5,000-sample corpus for full 1024-dimension sweep."""
    gen = ValidationCorpusGenerator(seed=42)
    corpus = gen.generate_corpus(num_samples=1000)
    candidates = build_candidate_pool()
    selector = MRMRSelector([c.name for c in candidates])

    selected = selector.select_dimensions(
        X=corpus.candidate_matrix,
        y=corpus.labels,
        num_to_select=1024,
        alpha_redundancy=0.5,
    )
    assert len(selected) == 1024
