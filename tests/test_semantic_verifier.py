"""Tests for SemanticVectorVerifier, GoldCorpusLoader, and Multi-Objective MRMR Selection."""

import numpy as np
import pytest

from core.slots import CANONICAL_SLOTS
from core.types import QuantaVector
from data.corpus_generator import ValidationCorpusGenerator
from data.gold_corpus import GoldCorpusLoader, GoldPair
from profiler.candidate_pool import build_candidate_pool
from profiler.info_profiler import QuantaInformationProfiler
from profiler.mrmr_selector import MRMRSelector
from profiler.verifier import SemanticVectorVerifier


def test_gold_corpus_loader():
    loader = GoldCorpusLoader()
    pairs = loader.build_gold_validation_suite(max_total=50)

    assert len(pairs) > 0
    for p in pairs:
        assert isinstance(p.gold_vector, QuantaVector)
        assert isinstance(p.source_text, str)
        assert p.domain in ("FOLIO", "CodeAST")


def test_semantic_verifier_gold_alignment():
    verifier = SemanticVectorVerifier()

    # Create synthetic gold and predicted pairs with known differences
    vec_gold1 = QuantaVector.zeros()
    vec_gold1[0] = 1  # NSM_I
    vec_gold1[64] = 1 # VAL_X1_AGENT

    vec_pred1 = QuantaVector.zeros()
    vec_pred1[0] = 1  # Matches
    vec_pred1[64] = 1 # Matches
    vec_pred1[128] = 1 # False positive

    pair1 = GoldPair(
        id="test_1",
        domain="FOLIO",
        source_text="I did it.",
        formal_representation=None,
        gold_vector=vec_gold1,
        predicted_vector=vec_pred1,
    )

    report = verifier.evaluate_gold_alignment([pair1])
    assert report.num_samples == 1
    # Slot 0 should have Precision=1.0, Recall=1.0, F1=1.0
    assert report.precision[0] == 1.0
    assert report.recall[0] == 1.0
    assert report.f1_score[0] == 1.0

    # Slot 128 (FP) should have Precision=0.0
    assert report.precision[128] == 0.0


def test_semantic_verifier_full_suite():
    loader = GoldCorpusLoader()
    pairs = loader.build_gold_validation_suite(max_total=50)

    verifier = SemanticVectorVerifier()
    summary = verifier.run_full_verification(pairs, dead_threshold=0.01)

    assert summary.gold_alignment.num_samples == len(pairs)
    assert 0.0 <= summary.cycle_consistency_score <= 1.0
    assert 0.0 <= summary.symbolic_soundness_rate <= 1.0
    assert len(summary.causal_necessity_scores) == 1024
    assert isinstance(summary.dead_slots_deficiency_type, dict)


def test_multi_objective_mrmr_selection(tmp_path):
    loader = GoldCorpusLoader()
    pairs = loader.build_gold_validation_suite(max_total=50)

    verifier = SemanticVectorVerifier()
    summary = verifier.run_full_verification(pairs)

    gen = ValidationCorpusGenerator(seed=42)
    corpus = gen.generate_corpus(num_samples=100)

    candidates = build_candidate_pool()
    candidate_names = [c.name for c in candidates]

    selector = MRMRSelector(candidate_names)
    selected = selector.select_dimensions_multi_objective(
        X=corpus.candidate_matrix,
        f1_scores=summary.gold_alignment.f1_score,
        causal_necessity=summary.causal_necessity_scores,
        num_to_select=256,
        w_f1=0.35,
        w_entropy=0.25,
        w_necessity=0.20,
        w_redundancy=0.15,
        w_violation=0.05,
    )

    assert len(selected) == 256
    selected_names = [s[1] for s in selected]
    # Verify no duplicate names
    assert len(set(selected_names)) == 256

    # Verify no benchmark-specific pollution in candidate pool
    for name in selected_names:
        assert not name.startswith("PROOFWRITER_")
        assert not name.startswith("FOLIO_")
        assert not name.startswith("CLUTRR_")
        assert not name.startswith("BABI_")
