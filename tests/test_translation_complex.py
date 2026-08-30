"""Comprehensive Test Suite for QUANTA Advanced Stress-Testing Complex Sentences and Scientific Narrative Discourse.

Verifies:
1. Counterfactual Causal Reasoning with Sarcasm & Second-Order Theory of Mind (Slots 244, 170, 165, 161, 166, 18).
2. Mixed Temporal Intervals, Continuous Kinematics, and Spatial Mereotopology (Slots 61, 228, 224, 235, 206, 193).
3. Deep Quantifier Scope Ambiguity with Higher-Order Modal Logic (Slots 90, 91, 111, 163, 248, 75).
4. Metalogical Self-Reference and Deontic Causal Interventions (Slots 103, 101, 198, 242).
5. Multi-Sentence Cohesive Scientific Narrative Paragraph (Coreference Bundles, Spatial Merkle-Folding, Allen Calculus Chain, Epistemic Phase Shift).
6. English-English Round-Trip Invariance and Slot Preservation.
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


# ----------------------------------------------------------------------
# 2. Stress Test 2: Kinematics, Temporal Intervals, Spatial Mereotopology
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def stress_2_eng():
    return "While the drone was accelerating into the restricted airspace before dusk, the operator plausibly suspected, but could not deduce with certainty, that the left wingtip was tangentially touching the perimeter wire."


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


# ----------------------------------------------------------------------
# 3. Stress Test 3: Quantifier Scope & Higher-Order Modal Logic
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def stress_3_eng():
    return "Every investigator who doubted that any suspect had necessarily committed every crime secretly wanted someone to prove the absolute impossibility of an accomplice's alibi."


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


# ----------------------------------------------------------------------
# 4. Stress Test 4: Metalogical Self-Reference & Deontic Prevention
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def stress_4_eng():
    return "By declaring this very decree to be legally void, the council obligated the commissioner to prevent its future enforcement unless the clause could recursively validate its own origin."


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


def test_punctuation_nodes_and_delimiters(pipeline):
    """Verify punctuation tokens are cleanly anchored as punct:* rather than ConceptNet pseudo-concepts."""
    sentence = "Had Alice not falsely pretended to know that Bob believed her investment was secure, the auditor wouldn't have sarcastically remarked that her due diligence was a stroke of genius."
    graph, val = pipeline.translate_forward(sentence, modality="english")
    assert val.is_valid

    punct_nodes = [n for n in graph.nodes.values() if n.anchor and n.anchor.startswith("punct:")]
    assert len(punct_nodes) >= 2, "Expected punctuation nodes for comma and period"
    for pn in punct_nodes:
        assert pn.anchor in ("punct:.", "punct:,", "punct:!", "punct:?")
        assert not pn.anchor.startswith("cn:en:.")
        assert not pn.anchor.startswith("cn:en:,")


def test_negative_contractions_wiring(pipeline):
    """Verify negative contractions (wouldn't, didn't) wire auxiliary verbs to negation nodes."""
    sentence = "The auditor wouldn't have remarked that."
    graph, val = pipeline.translate_forward(sentence, modality="english")
    assert val.is_valid

    # Find auxiliary node (would) and negation node (n't)
    would_node = next((n for n in graph.nodes.values() if n.literal == "would"), None)
    nt_node = next((n for n in graph.nodes.values() if n.literal == "n't"), None)

    assert would_node is not None, "Expected 'would' token node"
    assert nt_node is not None, "Expected 'n\'t' token node"
    assert nt_node.get_slot("LJB_NA_NEGATION") == 2

    # Check connection between auxiliary and negation
    assert nt_node.cid in would_node.edges.get("GRAPH_IS_SUB_EXP", []) or nt_node.cid in would_node.edges.get("LJB_NA_NEGATION", []) or nt_node.cid in graph.root.edges.get("LJB_NA_NEGATION", [])


def test_possessive_apostrophe_wiring(pipeline):
    """Verify possessive apostrophe constructions (Alice's cat) wire MEREOLOGY_POSSESSIVE relations."""
    sentence = "Alice's cat saw Bob's drone."
    graph, val = pipeline.translate_forward(sentence, modality="english")
    assert val.is_valid

    poss_markers = [n for n in graph._node_list if n.literal == "'s"]
    assert len(poss_markers) == 2, "Expected two 's token nodes"
    for pm in poss_markers:
        assert pm.anchor == "gram:case:possessive"
        assert pm.get_slot("NSM_HAVE") == 1

    # Check round trip
    rt = pipeline.round_trip(sentence, modality="english")
    assert rt.validation_pass
    assert rt.realized_output == "Alice's cat saw Bob's drone."


def test_auxiliary_contractions_and_titles(pipeline):
    """Verify auxiliary contractions ('s, 're, 've, 'll, 'd) and titles (Dr., Prof.) are properly grounded."""
    sentence = "Dr. Vance said that they're ready and she'll arrive."
    graph, val = pipeline.translate_forward(sentence, modality="english")
    assert val.is_valid

    dr_node = next((n for n in graph.nodes.values() if n.literal == "Dr."), None)
    assert dr_node is not None
    assert dr_node.anchor == "gram:title:dr"

    re_node = next((n for n in graph.nodes.values() if n.literal == "'re"), None)
    assert re_node is not None
    assert re_node.anchor == "cn:en:be (v)"

    ll_node = next((n for n in graph.nodes.values() if n.literal == "'ll"), None)
    assert ll_node is not None
    assert ll_node.anchor == "cn:en:will (v)"

    # Check round trip
    rt = pipeline.round_trip(sentence, modality="english")
    assert rt.validation_pass
    assert "they're" in rt.realized_output
    assert "she'll" in rt.realized_output

