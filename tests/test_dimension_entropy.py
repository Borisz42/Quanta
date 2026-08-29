"""Tests for Information Bottleneck profiler, entropy, redundancy, and mRMR selection."""

import numpy as np
import pytest

from data.corpus_generator import ValidationCorpusGenerator
from profiler.info_profiler import QuantaInformationProfiler
from profiler.mrmr_selector import MRMRSelector, export_optimal_dimensions
from profiler.candidate_pool import build_candidate_pool, export_candidate_pool


@pytest.fixture(scope="module")
def validation_corpus():
    gen = ValidationCorpusGenerator(seed=42)
    return gen.generate_corpus(num_samples=5000)


def test_candidate_pool_structure_and_export(tmp_path):
    candidates = build_candidate_pool()
    assert len(candidates) >= 512

    # Verify first 256 are canonical slots
    for i in range(256):
        assert candidates[i].id == i
        assert candidates[i].source.startswith("Band")

    # Verify no duplicate candidate names or IDs
    names = [c.name for c in candidates]
    ids = [c.id for c in candidates]
    assert len(set(names)) == len(candidates)
    assert len(set(ids)) == len(candidates)

    # Test export to JSON
    json_path = tmp_path / "candidate_pool.json"
    exported_path = export_candidate_pool(candidates, json_path)
    assert exported_path.exists()

    import json
    with open(exported_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert len(loaded) == len(candidates)
    assert loaded[0]["name"] == candidates[0].name
    assert "description" in loaded[0]



def test_validation_corpus_generation(validation_corpus):
    assert validation_corpus.canonical_matrix.shape == (5000, 256)
    assert validation_corpus.candidate_matrix.shape[0] == 5000
    assert validation_corpus.candidate_matrix.shape[1] >= 512
    assert len(validation_corpus.labels) == 5000
    assert len(validation_corpus.propositions) == 5000


def test_information_profiler_metrics(validation_corpus):
    profiler = QuantaInformationProfiler(validation_corpus.canonical_matrix)
    report = profiler.run_diagnostic_suite(dead_threshold=0.01, redundancy_threshold=0.7, verbose=False)

    assert report["num_samples"] == 5000
    assert report["mean_entropy"] > 0.1  # Across diverse propositions
    assert report["collision_rate"] < 0.05  # Highly resolvable
    assert report["joint_entropy"] >= 0.0
    assert report["total_correlation"] >= 0.0

    # Check band entropies
    band_stats = report["band_entropies"]
    for band_name, ent in band_stats.items():
        assert ent >= 0.0


def test_total_correlation_and_report_export(validation_corpus, tmp_path):
    profiler = QuantaInformationProfiler(validation_corpus.canonical_matrix)
    
    # 1. Test theoretical properties on synthetic matrices
    # Case A: Perfectly correlated 256 dimensions
    rng = np.random.default_rng(42)
    single_col = rng.choice([0, 1, 2, 3], size=1000, p=[0.25, 0.25, 0.25, 0.25])
    correlated_matrix = np.tile(single_col[:, np.newaxis], (1, 256))
    prof_corr = QuantaInformationProfiler(correlated_matrix)
    
    joint_h = prof_corr.compute_joint_entropy()
    marg_h = prof_corr.compute_entropies()
    tc = prof_corr.compute_total_correlation()
    
    assert np.isclose(joint_h, 2.0, atol=0.05)
    assert np.isclose(marg_h[0], 2.0, atol=0.05)
    # TC = 256 * 2.0 - 2.0 = 255 * 2.0 = 510.0 bits
    assert np.isclose(tc, 255.0 * 2.0, atol=10.0)

    # 2. Test export functionality
    txt_path = tmp_path / "profiler_report.txt"
    json_path = tmp_path / "profiler_report.json"
    paths = profiler.export_report(txt_path=txt_path, json_path=json_path)
    
    assert paths["txt"].exists()
    assert paths["json"].exists()

    txt_content = paths["txt"].read_text(encoding="utf-8")
    assert "QUANTA INFORMATION PROFILER REPORT" in txt_content
    assert "Total Correlation" in txt_content

    import json
    with open(paths["json"], "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert json_data["num_samples"] == 5000
    assert "total_correlation" in json_data
    assert "band_entropies" in json_data



def test_mrmr_selector_feature_ranking(validation_corpus):
    candidates = build_candidate_pool()
    candidate_names = [c.name for c in candidates]

    selector = MRMRSelector(candidate_names)
    selected = selector.select_dimensions(
        X=validation_corpus.candidate_matrix,
        y=validation_corpus.labels,
        num_to_select=256,
        alpha_redundancy=0.5,
    )

    assert len(selected) == 256
    # Each selection is a tuple (index, name, score)
    for idx, name, score in selected:
        assert 0 <= idx < len(candidates)
        assert isinstance(name, str)
        assert isinstance(score, float)

    # Ensure no duplicates in selected set
    selected_indices = [s[0] for s in selected]
    assert len(set(selected_indices)) == 256


def test_mrmr_optimal_dimensions_verification_and_export(validation_corpus, tmp_path):
    """10C.4 Verification: verify selected dimensions, export to JSON/CSV, and profile selected subspace."""
    candidates = build_candidate_pool()
    candidate_names = [c.name for c in candidates]

    selector = MRMRSelector(candidate_names)
    selected = selector.select_dimensions(
        X=validation_corpus.candidate_matrix,
        y=validation_corpus.labels,
        num_to_select=256,
        alpha_redundancy=0.5,
    )

    # 1. Verify export to JSON and CSV
    json_path = tmp_path / "optimal_256_dimensions.json"
    csv_path = tmp_path / "optimal_256_dimensions.csv"
    export_paths = export_optimal_dimensions(selected, candidates, json_path=json_path, csv_path=csv_path)

    assert export_paths["json"].exists()
    assert export_paths["csv"].exists()

    import json
    import csv
    with open(export_paths["json"], "r", encoding="utf-8") as f:
        dims_json = json.load(f)
    assert len(dims_json) == 256
    assert dims_json[0]["rank"] == 1
    assert "mrmr_score" in dims_json[0]

    with open(export_paths["csv"], "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 257  # header + 256 rows
    assert reader[0] == ["Rank", "Candidate_ID", "Name", "Source", "Category", "MRMR_Score", "Description"]

    # 2. Profile the selected 256-dimensional sub-tensor
    selected_indices = [s[0] for s in selected]
    selected_labels = [s[1] for s in selected]
    X_selected = validation_corpus.candidate_matrix[:, selected_indices]

    profiler = QuantaInformationProfiler(X_selected, dimension_labels=selected_labels)
    report = profiler.run_diagnostic_suite(dead_threshold=0.01, redundancy_threshold=0.8, verbose=False)

    assert report["num_samples"] == 5000
    assert report["mean_entropy"] > 0.05
    assert report["collision_rate"] < 0.10
    assert report["total_correlation"] >= 0.0

