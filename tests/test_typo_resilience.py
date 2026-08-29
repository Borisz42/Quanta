"""Tests for Typo Resilience and Lexical Normalization in QUANTA."""

import pytest
from parser.typo_normalizer import TypoNormalizer, damerau_levenshtein
from parser.nlp_forward import NLPForwardParser
from pipeline.translator_pipeline import TwoWayTranslationPipeline
from core.types import QuaternaryValue


@pytest.fixture(scope="module")
def normalizer():
    return TypoNormalizer.get_instance()


@pytest.fixture(scope="module")
def nlp_parser():
    return NLPForwardParser()


@pytest.fixture(scope="module")
def pipeline():
    return TwoWayTranslationPipeline()


def test_damerau_levenshtein_distance():
    # Insertions, deletions, substitutions, transpositions
    assert damerau_levenshtein("garden", "graden") == 1
    assert damerau_levenshtein("golden", "glden") == 1
    assert damerau_levenshtein("retriever", "retreiver") == 1
    assert damerau_levenshtein("mailman", "maileman") == 1
    assert damerau_levenshtein("chased", "chsed") == 1
    assert damerau_levenshtein("dog", "dg") == 1
    assert damerau_levenshtein("cat", "ct") == 1


def test_word_and_sentence_normalization(normalizer):
    assert normalizer.correct_word("glden", lang="en") == "golden"
    assert normalizer.correct_word("retreiver", lang="en") == "retriever"
    assert normalizer.correct_word("maileman", lang="en") == "mailman"
    assert normalizer.correct_word("graden", lang="en") == "garden"
    assert normalizer.correct_word("chsed", lang="en") == "chased"
    assert normalizer.correct_word("kichen", lang="en") == "kitchen"
    assert normalizer.correct_word("fatther", lang="en") == "father"
    assert normalizer.correct_word("tigre", lang="en") == "tiger"

    norm_sent = normalizer.normalize_text("A glden retreiver bit the maileman in the graden.", lang="en")
    assert norm_sent == "A golden retriever bit the mailman in the garden."

    norm_compound = normalizer.normalize_text("The bald egle and polr bear saw the post offce.", lang="en")
    assert norm_compound == "The bald eagle and polar bear saw the post office."


def test_typo_asg_construction_and_location_preservation(nlp_parser):
    # Sentence with multiple typos across subject, verb, object, location
    clean_graph = nlp_parser.parse_sentence("A golden retriever bit the mailman in the garden.")
    typo_graph = nlp_parser.parse_sentence("A glden retreiver bit the maileman in the graden.")

    assert clean_graph.root is not None
    assert typo_graph.root is not None

    # Verify root has all 3 relational edges
    assert "VAL_X1_AGENT" in typo_graph.root.edges
    assert "VAL_X2_PATIENT" in typo_graph.root.edges
    assert "VAL_LOCATION_SLOT" in typo_graph.root.edges

    # Check location node
    loc_cid = typo_graph.root.edges["VAL_LOCATION_SLOT"][0]
    loc_node = typo_graph.get_node(loc_cid)
    assert loc_node is not None
    assert "garden" in str(loc_node.literal).lower() or loc_node.anchor == "wn:garden.n.01"
    assert loc_node.get_slot("VAL_LOCATION_SLOT") == QuaternaryValue.TRUE

    # Bit-for-bit Merkle Root invariance between clean and typo inputs
    assert typo_graph.compute_merkle_root() == clean_graph.compute_merkle_root()


def test_typo_translation_pipeline_round_trip(pipeline):
    typo_input = "A glden retreiver bit the maileman in the graden."
    res_en = pipeline.execute_translation(typo_input, target_modality="english", source_modality="english")
    assert res_en.is_success
    assert "garden" in res_en.output_text.lower()
    assert "retriever" in res_en.output_text.lower()
    assert "mailman" in res_en.output_text.lower()
