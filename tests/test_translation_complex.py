"""Comprehensive Test Suite for QUANTA Advanced Stress-Testing Complex Sentences and Scientific Narrative Discourse.

Verifies:
1. Counterfactual Causal Reasoning with Sarcasm & Second-Order Theory of Mind (Slots 244, 170, 165, 161, 166, 18).
2. Mixed Temporal Intervals, Continuous Kinematics, and Spatial Mereotopology (Slots 61, 228, 224, 235, 206, 193).
3. Deep Quantifier Scope Ambiguity with Higher-Order Modal Logic (Slots 90, 91, 111, 163, 248, 75).
4. Metalogical Self-Reference and Deontic Causal Interventions (Slots 103, 101, 198, 242).
5. Multi-Sentence Cohesive Scientific Narrative Paragraph (Coreference Bundles, Spatial Merkle-Folding, Allen Calculus Chain, Epistemic Phase Shift).
6. 4 Language Directions: English-English, English-Hungarian, Hungarian-English, Hungarian-Hungarian.
"""

import pytest
from pipeline.translator_pipeline import TwoWayTranslationPipeline


@pytest.fixture(scope="module")
def pipeline():
    return TwoWayTranslationPipeline()


# ----------------------------------------------------------------------
# 1. Stress Test 1: Counterfactual & Second-Order ToM & Sarcasm
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def stress_1_eng():
    return "Had Alice not falsely pretended to know that Bob believed her investment was secure, the auditor wouldn't have sarcastically remarked that her due diligence was a stroke of genius."


@pytest.fixture(scope="module")
def stress_1_hu():
    return "Ha Alice nem színlelte volna hamisan, hogy tudja, hogy Bob biztonságosnak hitte a befektetését, a könyvvizsgáló nem jegyezte volna meg gúnyosan, hogy az átvilágítása zseniális húzás volt."


def test_stress_1_asg_slots_and_round_trip(pipeline, stress_1_eng):
    """Verify Sentence 1 Mentalese ASG slots and English round-trip invariance."""
    graph, val = pipeline.translate_forward(stress_1_eng, modality="english")
    assert val.is_valid, f"Validation failed: {val.errors}"

    # Verify specific quaternary slots
    assert graph.root.get_slot("CAUSAL_COUNTERFACTUAL_NEC") == 1
    assert graph.root.get_slot("MODALITY_COUNTERFACTUAL") == 1
    assert graph.root.get_slot("ROLE_DECEPTIVE_PROJECTION") == 1
    assert graph.root.get_slot("TOM_BELIEF_SECOND_ORDER") == 1
    assert graph.root.get_slot("ROLE_SARCASM_IRONY") == 1
    assert graph.root.get_slot("NSM_GOOD") == 1

    # Round trip
    rt = pipeline.round_trip(stress_1_eng, modality="english")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert "Alice" in rt.realized_output
    assert "Bob" in rt.realized_output
    assert "auditor" in rt.realized_output


def test_stress_1_cross_lingual_eng_to_hu(pipeline, stress_1_eng, stress_1_hu):
    """Verify Sentence 1 English to Hungarian cross-lingual translation."""
    res = pipeline.execute_translation(stress_1_eng, target_modality="hungarian", source_modality="english")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == stress_1_hu


def test_stress_1_cross_lingual_hu_to_eng(pipeline, stress_1_hu, stress_1_eng):
    """Verify Sentence 1 Hungarian to English cross-lingual translation."""
    res = pipeline.execute_translation(stress_1_hu, target_modality="english", source_modality="hungarian")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == stress_1_eng


def test_stress_1_hungarian_round_trip(pipeline, stress_1_hu):
    """Verify Sentence 1 Hungarian to Hungarian round-trip."""
    rt = pipeline.round_trip(stress_1_hu, modality="hungarian")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert rt.realized_output == stress_1_hu


# ----------------------------------------------------------------------
# 2. Stress Test 2: Kinematics, Temporal Intervals, Spatial Mereotopology
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def stress_2_eng():
    return "While the drone was accelerating into the restricted airspace before dusk, the operator plausibly suspected, but could not deduce with certainty, that the left wingtip was tangentially touching the perimeter wire."


@pytest.fixture(scope="module")
def stress_2_hu():
    return "Miközben a drón szürkület előtt a korlátozott légtérbe gyorsult, a kezelő valószínűsíthetően gyanította, de nem tudta bizonyossággal levezetni, hogy a bal szárnyvég érintőlegesen érintette a kerítésdrótot."


def test_stress_2_asg_slots_and_round_trip(pipeline, stress_2_eng):
    """Verify Sentence 2 Mentalese ASG slots and English round-trip invariance."""
    graph, val = pipeline.translate_forward(stress_2_eng, modality="english")
    assert val.is_valid, f"Validation failed: {val.errors}"

    # Verify specific quaternary slots
    assert graph.root.get_slot("NSM_ACCELERATING_RATE") == 1
    assert graph.root.get_slot("VAL_X3_DESTINATION") == 1
    assert graph.root.get_slot("TEMP_ALLEN_DURING") == 1
    assert graph.root.get_slot("SPATIAL_RCC_TANGENTIAL_PART") == 1
    assert graph.root.get_slot("EPIST_FUZZY_PLAUSIBILITY") == 3
    assert graph.root.get_slot("EPIST_DEDUCTIVE_INFERENCE") == 2

    # Round trip
    rt = pipeline.round_trip(stress_2_eng, modality="english")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert rt.realized_output == stress_2_eng


def test_stress_2_cross_lingual_eng_to_hu(pipeline, stress_2_eng, stress_2_hu):
    """Verify Sentence 2 English to Hungarian cross-lingual translation."""
    res = pipeline.execute_translation(stress_2_eng, target_modality="hungarian", source_modality="english")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == stress_2_hu


def test_stress_2_cross_lingual_hu_to_eng(pipeline, stress_2_hu, stress_2_eng):
    """Verify Sentence 2 Hungarian to English cross-lingual translation."""
    res = pipeline.execute_translation(stress_2_hu, target_modality="english", source_modality="hungarian")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == stress_2_eng


def test_stress_2_hungarian_round_trip(pipeline, stress_2_hu):
    """Verify Sentence 2 Hungarian to Hungarian round-trip."""
    rt = pipeline.round_trip(stress_2_hu, modality="hungarian")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert rt.realized_output == stress_2_hu


# ----------------------------------------------------------------------
# 3. Stress Test 3: Quantifier Scope & Higher-Order Modal Logic
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def stress_3_eng():
    return "Every investigator who doubted that any suspect had necessarily committed every crime secretly wanted someone to prove the absolute impossibility of an accomplice's alibi."


@pytest.fixture(scope="module")
def stress_3_hu():
    return "Minden nyomozó, aki kételkedett abban, hogy bármelyik gyanúsított szükségszerűen elkövetett minden bűncselekményt, titokban azt akarta, hogy valaki bebizonyítsa egy bűntárs alibijének abszolút lehetetlenségét."


def test_stress_3_asg_slots_and_round_trip(pipeline, stress_3_eng):
    """Verify Sentence 3 Mentalese ASG slots and English round-trip invariance."""
    graph, val = pipeline.translate_forward(stress_3_eng, modality="english")
    assert val.is_valid, f"Validation failed: {val.errors}"

    # Verify specific quaternary slots
    assert graph.root.get_slot("LJB_RO_ALL_QUANT") == 1
    assert graph.root.get_slot("LJB_SUO_AT_LEAST_ONE") == 1
    assert graph.root.get_slot("GRAPH_VARIABLE_BIND") == 1
    assert graph.root.get_slot("TOM_DESIRE") == 1
    assert graph.root.get_slot("LOGIC_NECESSITY_BOX") == 1
    assert graph.root.get_slot("LJB_NA_NEGATION") == 2

    # Round trip
    rt = pipeline.round_trip(stress_3_eng, modality="english")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert rt.realized_output == stress_3_eng


def test_stress_3_cross_lingual_eng_to_hu(pipeline, stress_3_eng, stress_3_hu):
    """Verify Sentence 3 English to Hungarian cross-lingual translation."""
    res = pipeline.execute_translation(stress_3_eng, target_modality="hungarian", source_modality="english")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == stress_3_hu


def test_stress_3_cross_lingual_hu_to_eng(pipeline, stress_3_hu, stress_3_eng):
    """Verify Sentence 3 Hungarian to English cross-lingual translation."""
    res = pipeline.execute_translation(stress_3_hu, target_modality="english", source_modality="hungarian")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == stress_3_eng


def test_stress_3_hungarian_round_trip(pipeline, stress_3_hu):
    """Verify Sentence 3 Hungarian to Hungarian round-trip."""
    rt = pipeline.round_trip(stress_3_hu, modality="hungarian")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert rt.realized_output == stress_3_hu


# ----------------------------------------------------------------------
# 4. Stress Test 4: Metalogical Self-Reference & Deontic Prevention
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def stress_4_eng():
    return "By declaring this very decree to be legally void, the council obligated the commissioner to prevent its future enforcement unless the clause could recursively validate its own origin."


@pytest.fixture(scope="module")
def stress_4_hu():
    return "Azzal, hogy ezt a rendeletet jogilag semmisnek nyilvánította, a tanács kötelezte a biztost a jövőbeli végrehajtás megakadályozására, hacsak a záradék rekurzívan nem tudta igazolni saját eredetét."


def test_stress_4_asg_slots_and_round_trip(pipeline, stress_4_eng):
    """Verify Sentence 4 Mentalese ASG slots and English round-trip invariance."""
    graph, val = pipeline.translate_forward(stress_4_eng, modality="english")
    assert val.is_valid, f"Validation failed: {val.errors}"

    # Verify specific quaternary slots
    assert graph.root.get_slot("GRAPH_CYCLIC_BACKLINK") == 1
    assert graph.root.get_slot("GRAPH_RECURSIVE_REF") == 1
    assert graph.root.get_slot("EPIST_DEONTIC_OBLIGATION") == 1
    assert graph.root.get_slot("CAUSAL_PREVENTIVE_BLOCK") == 1

    # Round trip
    rt = pipeline.round_trip(stress_4_eng, modality="english")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert rt.realized_output == stress_4_eng


def test_stress_4_cross_lingual_eng_to_hu(pipeline, stress_4_eng, stress_4_hu):
    """Verify Sentence 4 English to Hungarian cross-lingual translation."""
    res = pipeline.execute_translation(stress_4_eng, target_modality="hungarian", source_modality="english")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == stress_4_hu


def test_stress_4_cross_lingual_hu_to_eng(pipeline, stress_4_hu, stress_4_eng):
    """Verify Sentence 4 Hungarian to English cross-lingual translation."""
    res = pipeline.execute_translation(stress_4_hu, target_modality="english", source_modality="hungarian")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == stress_4_eng


def test_stress_4_hungarian_round_trip(pipeline, stress_4_hu):
    """Verify Sentence 4 Hungarian to Hungarian round-trip."""
    rt = pipeline.round_trip(stress_4_hu, modality="hungarian")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert rt.realized_output == stress_4_hu


# ----------------------------------------------------------------------
# 5. Scientific Narrative Paragraph: Merkle Folding, Cohesion & Allen Chain
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def scientific_paragraph_eng():
    return (
        "Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. "
        "She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, which strongly suggested an unobserved phase transition. "
        "Although her supervisor initially doubted the validity of the discovery, Eleanor verified the hypothesis three hours later by replicating the transformation within the same vessel. "
        "The resulting polymer retained its structural integrity throughout the afternoon, prompting the laboratory director to prohibit all competing tests until her synthesis protocol could be formally audited."
    )


@pytest.fixture(scope="module")
def scientific_paragraph_hu():
    return (
        "Dr. Eleanor Vance hajnalban egy illékony szintetikus vegyületet izolált a kriogén tárolócellában. "
        "Azonnal megjegyezte, hogy ez a minta rendellenes kristályrács-tágulást mutatott, ami határozottan egy megfigyeletlen fázisátmenetre utalt. "
        "Bár a témavezetője kezdetben kételkedett a felfedezés érvényességében, Eleanor három órával később igazolta a hipotézist a transformáció ugyanazon tartályban történő megismétlésével. "
        "A keletkező polimer egész délután megőrizte szerkezeti integritását, ami arra késztette a laboratórium igazgatóját, hogy tiltsa meg az összes versengő tesztet, amíg a szintézis protokollját hivatalosan felül nem vizsgálják."
    )


def test_scientific_paragraph_asg_structural_profile(pipeline, scientific_paragraph_eng):
    """Verify scientific discourse graph contains coreference bundles, Merkle-fold points, Allen chains, and epistemic transitions."""
    graph, val = pipeline.translate_forward(scientific_paragraph_eng, modality="english")
    assert val.is_valid, f"Validation failed: {val.errors}"

    # Master Root Discourse Properties
    root = graph.root
    assert root is not None
    assert root.get_slot("GRAPH_ROOT_NODE") == 1
    assert root.get_slot("GRAPH_ORDERED_SEQ") == 1
    assert root.get_slot("GRAPH_COREF_BUNDLE") == 1
    assert root.get_slot("GRAPH_MERKLE_FOLD_POINT") == 1
    assert root.get_slot("SOLVER_PROOF_VALIDATED") == 1
    assert root.get_slot("CAUSAL_DIRECT_MECHANISM") == 1
    assert root.get_slot("EPIST_DEONTIC_PROHIBITION") == 1
    assert root.get_slot("SPATIAL_RCC_NON_TANG_PART") == 1

    # Check State nodes have Allen intervals
    nodes = list(graph.nodes.values())
    assert any(n.get_slot("TEMP_ALLEN_MEETS") == 1 for n in nodes)
    assert any(n.get_slot("TEMP_ALLEN_BEFORE") == 1 for n in nodes)
    assert any(n.get_slot("TEMP_ALLEN_DURING") == 1 for n in nodes)
    assert any(n.get_slot("LOGIC_TEMPORAL_UNTIL_U") == 1 for n in nodes)

    # Check Coreference Bundles in entities
    coref_nodes = [n for n in graph.nodes.values() if n.get_slot("GRAPH_COREF_BUNDLE") == 1]
    assert len(coref_nodes) >= 2, "Expected Agent and Compound coreference bundles"

    # Check Spatial Merkle Fold Point
    fold_nodes = [n for n in graph.nodes.values() if n.get_slot("GRAPH_MERKLE_FOLD_POINT") == 1]
    assert len(fold_nodes) >= 2, "Expected Cryogenic Containment Cell Merkle fold point reused at vessel"


def test_scientific_paragraph_english_round_trip(pipeline, scientific_paragraph_eng):
    """Verify full scientific paragraph English round-trip invariance."""
    rt = pipeline.round_trip(scientific_paragraph_eng, modality="english")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert rt.realized_output == scientific_paragraph_eng


def test_scientific_paragraph_cross_lingual_eng_to_hu(pipeline, scientific_paragraph_eng, scientific_paragraph_hu):
    """Verify full scientific paragraph English to Hungarian cross-lingual translation."""
    res = pipeline.execute_translation(scientific_paragraph_eng, target_modality="hungarian", source_modality="english")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == scientific_paragraph_hu


def test_scientific_paragraph_cross_lingual_hu_to_eng(pipeline, scientific_paragraph_hu, scientific_paragraph_eng):
    """Verify full scientific paragraph Hungarian to English cross-lingual translation."""
    res = pipeline.execute_translation(scientific_paragraph_hu, target_modality="english", source_modality="hungarian")
    assert res.is_success, f"Failed: {res.error_message}"
    assert res.validation.is_valid
    assert res.output_text == scientific_paragraph_eng


def test_scientific_paragraph_hungarian_round_trip(pipeline, scientific_paragraph_hu):
    """Verify full scientific paragraph Hungarian to Hungarian round-trip."""
    rt = pipeline.round_trip(scientific_paragraph_hu, modality="hungarian")
    assert rt.validation_pass
    assert rt.slot_preservation_rate >= 0.90
    assert rt.realized_output == scientific_paragraph_hu
