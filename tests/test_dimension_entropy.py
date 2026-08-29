"""Tests for Information Bottleneck profiler, entropy, redundancy, and mRMR selection."""

import numpy as np
import pytest

from data.corpus_generator import ValidationCorpusGenerator
from profiler.info_profiler import QuantaInformationProfiler
from profiler.mrmr_selector import MRMRSelector
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

    # Check band entropies
    band_stats = report["band_entropies"]
    for band_name, ent in band_stats.items():
        assert ent >= 0.0


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
