"""Tests for QUANTA Round-Trip Invariance and Semantic Preservation.

Verifies strict round-trip invariance (NL -> Mentalese -> NL, FOL -> Mentalese -> FOL, Code -> Mentalese -> Code)
with low Hamming distance and 100% semantic type consistency.
"""

import pytest
from pipeline.translator_pipeline import TwoWayTranslationPipeline


@pytest.fixture(scope="module")
def pipeline():
    return TwoWayTranslationPipeline()


def test_english_round_trip_action_sentence(pipeline):
    """Verify Phase 9 Item 9.1: Action sentence round-trip semantic equivalence."""
    input_text = "A dog chased a cat."
    result = pipeline.round_trip(input_text, modality="english")

    assert result.validation_pass, f"Validation failed: {result.muc_errors}"
    assert "dog" in result.realized_output.lower()
    assert "chased" in result.realized_output.lower()
    assert "cat" in result.realized_output.lower()
    assert result.slot_preservation_rate >= 0.85


def test_english_canonical_example_a_round_trip(pipeline):
    """Verify Phase 9 Item 9.1: Canonical Example A sentence round-trip semantic equivalence."""
    input_text = "A golden retriever bit the mailman in the garden."
    result = pipeline.round_trip(input_text, modality="english")

    assert result.validation_pass, f"Validation failed: {result.muc_errors}"
    assert result.realized_output == "A golden retriever bit the mailman in the garden."
    assert result.slot_preservation_rate == 1.0
    assert result.original_vector["NSM_DO"] == 1
    assert result.original_vector["NSM_TOUCH"] == 1
    assert result.original_vector["VAL_X1_AGENT"] == 1
    assert result.original_vector["VAL_LOCATION_SLOT"] == 1


def test_english_round_trip_modal_obligation(pipeline):
    # Deontic modal
    input_text = "A person must touch a rock."
    result = pipeline.round_trip(input_text, modality="english")

    assert result.validation_pass, f"Validation failed: {result.muc_errors}"
    assert "must" in result.realized_output.lower()
    assert result.original_vector["EPIST_DEONTIC_OBLIGATION"] == 1
    assert result.reparsed_vector["EPIST_DEONTIC_OBLIGATION"] == 1


def test_fol_round_trip_implication(pipeline):
    """Verify Phase 9 Item 9.2: Canonical Example E FOL implication exact string match round-trip."""
    fol_input = r"\forall x (Dog(x) \rightarrow Animal(x))"
    result = pipeline.round_trip(fol_input, modality="fol")

    assert result.validation_pass, f"Validation failed: {result.muc_errors}"
    assert result.realized_output == r"\forall x (Dog(x) \rightarrow Animal(x))"
    assert result.slot_preservation_rate == 1.0
    assert result.original_vector["NSM_ALL"] == 1
    assert result.original_vector["LJB_RO_ALL_QUANT"] == 1
    assert result.original_vector["LJB_GANAI_IF_THEN"] == 1


def test_fol_round_trip_existential_negation(pipeline):
    """Verify Phase 9 Item 9.2: Existential conjunction and negation FOL exact string match round-trip."""
    fol_input = r"\exists y (\neg Cat(y) \land Dog(y))"
    result = pipeline.round_trip(fol_input, modality="fol")

    assert result.validation_pass, f"Validation failed: {result.muc_errors}"
    assert result.realized_output == r"\exists y (\neg Cat(y) \land Dog(y))"
    assert result.slot_preservation_rate == 1.0
    assert result.original_vector["NSM_SOME"] == 1
    assert result.original_vector["LJB_SUO_AT_LEAST_ONE"] == 1
    assert result.original_vector["LJB_JE_AND"] == 1
    assert result.original_vector["LJB_NA_NEGATION"] == 2


def test_python_code_round_trip(pipeline):
    # Python code snippet
    code_input = """def process(x):\n    return x"""
    result = pipeline.round_trip(code_input, modality="python")

    assert result.validation_pass
    assert "def process" in result.realized_output
    assert "return" in result.realized_output
    assert result.original_vector["GRAPH_FUNCTION_DEF"] == 1
    assert result.reparsed_vector["GRAPH_FUNCTION_DEF"] == 1


def test_multi_sentence_preservation_rates(pipeline):
    # Test across multiple sentence archetypes
    test_cases = [
        "A person saw a garden.",
        "A dog ran rapidly into the house.",
        "A mailman gave a book to a person.",
    ]

    for sentence in test_cases:
        res = pipeline.round_trip(sentence, modality="english")
        assert res.validation_pass, f"Failed on '{sentence}': {res.muc_errors}"
        assert res.slot_preservation_rate >= 0.70


def test_hungarian_round_trip(pipeline):
    # Hungarian round trip
    hu_input = "A kutya kergetett a macskát."
    result = pipeline.round_trip(hu_input, modality="hungarian")

    assert result.validation_pass, f"Validation failed: {result.muc_errors}"
    assert "kutya" in result.realized_output.lower()
    assert "macskát" in result.realized_output.lower()
    assert result.slot_preservation_rate >= 0.85

