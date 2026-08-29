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


def test_python_code_round_trip_execution_equivalence(pipeline):
    """Verify Phase 9 Item 9.3: Python source -> AST -> ASG -> Code emitter -> exec both and compare output."""
    # 1. Linear process function
    code_identity = "def process(x):\n    return x"
    res_ident = pipeline.round_trip(code_identity, modality="python")
    assert res_ident.validation_pass, f"Validation failed: {res_ident.muc_errors}"
    assert "def process" in res_ident.realized_output
    assert "return" in res_ident.realized_output
    assert res_ident.original_vector["GRAPH_FUNCTION_DEF"] == 1
    assert res_ident.reparsed_vector["GRAPH_FUNCTION_DEF"] == 1
    assert res_ident.slot_preservation_rate == 1.0

    orig_env_1, rt_env_1 = {}, {}
    exec(code_identity, orig_env_1)
    exec(res_ident.realized_output, rt_env_1)
    for test_val in [0, 42, -5, "hello", [1, 2, 3], {"a": 1}]:
        assert orig_env_1["process"](test_val) == rt_env_1["process"](test_val)

    # 2. Canonical Example F: Recursive Factorial
    code_factorial = "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"
    res_fact = pipeline.round_trip(code_factorial, modality="python")
    assert res_fact.validation_pass, f"Validation failed: {res_fact.muc_errors}"
    assert "def factorial" in res_fact.realized_output
    assert "return n * factorial(n - 1)" in res_fact.realized_output
    assert res_fact.original_vector["GRAPH_FUNCTION_DEF"] == 1
    assert res_fact.original_vector["GRAPH_RECURSIVE_REF"] == 1
    assert res_fact.reparsed_vector["GRAPH_RECURSIVE_REF"] == 1
    assert res_fact.slot_preservation_rate == 1.0

    orig_env_2, rt_env_2 = {}, {}
    exec(code_factorial, orig_env_2)
    exec(res_fact.realized_output, rt_env_2)
    for n in [0, 1, 2, 3, 5, 6, 7]:
        assert orig_env_2["factorial"](n) == rt_env_2["factorial"](n)
        assert rt_env_2["factorial"](n) == [1, 1, 2, 6, 120, 720, 5040][[0, 1, 2, 3, 5, 6, 7].index(n)]


def test_quaternary_hamming_distance_zero_canonical_slots(pipeline):
    """Verify Phase 9 Item 9.4: Round-tripped ASG vectors must have Hamming distance = 0 on canonical slots."""
    test_cases = [
        ("A golden retriever bit the mailman in the garden.", "english"),
        (r"\forall x (Dog(x) \rightarrow Animal(x))", "fol"),
        (r"\exists y (\neg Cat(y) \land Dog(y))", "fol"),
        ("def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)", "python"),
        ("A person must touch a rock.", "english"),
    ]

    for input_data, modality in test_cases:
        res = pipeline.round_trip(input_data, modality=modality)
        assert res.validation_pass, f"Validation failed for '{input_data}': {res.muc_errors}"

        orig_v = res.original_vector
        reparsed_v = res.reparsed_vector
        active_slots = orig_v.active_slots()

        # Verify Hamming distance on all active canonical slots is strictly 0
        canonical_hamming = sum(1 for slot_idx, val in active_slots.items() if reparsed_v[slot_idx] != val)
        assert canonical_hamming == 0, f"Canonical slot drift on '{input_data}': {[slot_idx for slot_idx, val in active_slots.items() if reparsed_v[slot_idx] != val]}"


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
    """Hungarian round trip test."""
    hu_input = "A kutya kergetett a macskát."
    result = pipeline.round_trip(hu_input, modality="hungarian")

    assert result.validation_pass, f"Validation failed: {result.muc_errors}"
    assert "kutya" in result.realized_output.lower()
    assert "macskát" in result.realized_output.lower()
    assert result.slot_preservation_rate >= 0.85


def test_cross_lingual_english_hungarian_gold_set_20(pipeline):
    """Verify Phase 9 Item 9.5: Manual gold set of 20 English-Hungarian parallel propositions."""
    gold_pairs = [
        ("A golden retriever bit the mailman in the garden.", "A golden retriever a kertben megharapta a postást."),
        ("The dog did not bite the mailman.", "A kutya nem harapta meg a postást."),
        ("Did the dog perhaps bite a mailman?", "Vajon a kutya megharapott egy postást?"),
        ("A cat chased a dog.", "A macska kergetett egy kutyát."),
        ("The person saw a garden.", "Az ember meglátott egy kertet."),
        ("A dog ran into the house.", "A kutya a házba elfutott."),
        ("The cat was in the house.", "A macska a házban volt."),
        ("A person touched a rock.", "Az ember érintett egy követ."),
        ("The dog walked from the garden.", "A kutya a kertből elsétálott."),
        ("A mailman saw a dog.", "A postás meglátott egy kutyát."),
        ("A dog did not chase a cat.", "A kutya nem kergetett egy macskát."),
        ("The cat ran rapidly.", "A macska gyorsan elfutott."),
        ("Every dog chased a ball.", "Minden kutya kergetett egy labdát."),
        ("Two dogs were in the garden.", "Két kutya a kertben volt."),
        ("A dog lived in the garden.", "A kutya a kertben élt."),
        ("The rock was big.", "A kő nagy volt."),
        ("Did a person see a cat?", "Vajon az ember meglátott egy macskát?"),
        ("The mailman walked into the garden.", "A postás a kertbe elsétálott."),
        ("A person saw a cat in the house.", "Az ember a házban meglátott egy macskát."),
        ("The dog was in the garden.", "A kutya a kertben volt."),
    ]

    assert len(gold_pairs) == 20

    for en_text, expected_hu in gold_pairs:
        res = pipeline.execute_translation(en_text, target_modality="hungarian")
        assert res.is_success, f"Failed on '{en_text}': {res.error_message}"
        assert res.output_text == expected_hu, f"Mismatch on '{en_text}': got '{res.output_text}', expected '{expected_hu}'"


