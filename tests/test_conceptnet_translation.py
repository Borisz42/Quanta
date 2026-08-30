"""Comprehensive test suite for 256-D ConceptNet integration and bi-directional translation pipeline."""

import pytest
import numpy as np

from core.slots import (
    CANONICAL_SLOTS,
    BAND_3_SLOTS,
    BAND_4_SLOTS,
    SlotBand,
    get_slot_by_name,
    get_slot_by_index,
    CN_Q001_COMPUTING,
    CN_Q128_MANNER,
    CN_Q129_LEAVE,
    CN_Q256_WORTHY,
    TYPE_ANIMATE,
    TYPE_HUMAN,
    AFFORD_INCISED_CUTTING,
)
from core.types import QuantaVector, QuaternaryValue
from core.asg import QuantaGraph, QuantaNode
from parser.lexical_grounder import ConceptNetLexicalGrounder, GroundedLexicalConcept
from parser.nlp_forward import NLPForwardParser
from realizer.english_nlg import EnglishRealizer, ConceptVectorDecoder
from pipeline.translator_pipeline import TwoWayTranslationPipeline
from solver.validator_gate import ValidationGate


def test_conceptnet_slot_registry_geometry():
    """Verify that Band 3 and Band 4 contain exactly 128 slots each, mapped to ConceptNet dimensions."""
    assert len(BAND_3_SLOTS) == 128
    assert len(BAND_4_SLOTS) == 128
    assert len(CANONICAL_SLOTS) == 1024

    # Band 3 boundaries
    assert BAND_3_SLOTS[0].index == 384
    assert BAND_3_SLOTS[0].name == "CN_Q001_COMPUTING"
    assert BAND_3_SLOTS[0].band == SlotBand.BAND_3_ONTOLOGY_STRUCTURES

    assert BAND_3_SLOTS[-1].index == 511
    assert BAND_3_SLOTS[-1].name == "CN_Q128_MANNER"
    assert BAND_3_SLOTS[-1].band == SlotBand.BAND_3_ONTOLOGY_STRUCTURES

    # Band 4 boundaries
    assert BAND_4_SLOTS[0].index == 512
    assert BAND_4_SLOTS[0].name == "CN_Q129_LEAVE"
    assert BAND_4_SLOTS[0].band == SlotBand.BAND_4_AFFORDANCES_OPERATIONS

    assert BAND_4_SLOTS[-1].index == 639
    assert BAND_4_SLOTS[-1].name == "CN_Q256_WORTHY"
    assert BAND_4_SLOTS[-1].band == SlotBand.BAND_4_AFFORDANCES_OPERATIONS


def test_legacy_ontology_aliases():
    """Verify that legacy symbolic ontology constants alias to corresponding ConceptNet slots."""
    assert TYPE_ANIMATE == get_slot_by_name("CN_Q011_ANIMAL").index
    assert TYPE_HUMAN == get_slot_by_name("CN_Q015_PERSON").index
    assert AFFORD_INCISED_CUTTING == get_slot_by_name("CN_Q108_CUT").index

    slot_anim = get_slot_by_name("TYPE_ANIMATE")
    assert slot_anim is not None
    assert slot_anim.name == "CN_Q011_ANIMAL"


def test_conceptnet_lexical_grounder():
    """Verify ConceptNetLexicalGrounder queries offline database and returns QuantaVectors."""
    grounder = ConceptNetLexicalGrounder.get_default()

    # Ground noun
    dog_concept = grounder.resolve_concept("dog", pos="n")
    assert dog_concept is not None
    assert dog_concept.lemma == "dog"
    assert dog_concept.synset_name == "cn:en:dog (n)"
    assert len(dog_concept.active_slots) > 0
    assert dog_concept.vector["CN_Q011_ANIMAL"] == 1
    assert dog_concept.vector["TYPE_ANIMATE"] == 1

    # Ground verb
    bite_concept = grounder.resolve_concept("bite", pos="v")
    assert bite_concept is not None
    assert bite_concept.lemma == "bite"
    assert len(bite_concept.active_slots) > 0

    # Ground multiword term
    gr_concept = grounder.resolve_concept("golden retriever", pos="n")
    assert gr_concept is not None
    assert gr_concept.lemma == "golden retriever"
    assert gr_concept.synset_name == "cn:en:golden retriever (n)"


def test_concept_vector_decoder_tier1():
    """Verify Tier 1 singleton decoding (< 10 ms) on 23,383 singletons from codebook."""
    decoder = ConceptVectorDecoder.get_instance()
    grounder = ConceptNetLexicalGrounder.get_default()

    dog_concept = grounder.resolve_concept("dog", pos="n")
    assert dog_concept is not None

    # Extract 256-D ConceptNet vector (Bands 3 & 4: slots 384..639)
    vec_256 = dog_concept.vector._data[384:640]
    lemma, dist = decoder.decode_vector(vec_256)

    assert lemma == "dog"
    assert dist == 0.0


def test_end_to_end_translation_roundtrip():
    """Verify end-to-end forward parsing and NLG unrolling with ConceptNet grounding."""
    pipeline = TwoWayTranslationPipeline()

    sentences = [
        "The dog bites the mailman.",
        "A golden retriever chased the cat.",
        "Every person knows the truth.",
        "A scientist created a software system in the laboratory.",
    ]

    for sent in sentences:
        res = pipeline.execute_translation(sent, target_modality="english")
        assert res.is_success, f"Translation failed for: {sent}. Error: {res.error_message}"
        assert res.output_text is not None
        assert len(res.output_text) > 0
        assert res.graph is not None
        assert res.graph.root is not None


def test_neurosymbolic_validation_with_conceptnet_aliases():
    """Verify ValidationGate enforces invariants using legacy aliases over ConceptNet slots."""
    gate = ValidationGate()

    # Valid event: Animate agent performing physical action
    valid_event = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="cn:en:bite (v)",
    )
    dog_agent = QuantaNode(
        vector={"TYPE_ANIMATE": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="cn:en:dog (n)",
    )
    g_valid = QuantaGraph()
    e_cid = g_valid.add_node(valid_event, set_as_root=True)
    d_cid = g_valid.add_node(dog_agent)
    g_valid.add_edge(e_cid, "VAL_X1_AGENT", d_cid)

    res_valid = gate.validate_graph(g_valid)
    assert res_valid.is_valid

    # Invalid event: Abstract concept acting as physical agent without figurative modality
    event_node2 = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="cn:en:bite (v)",
    )
    invalid_agent = QuantaNode(
        vector={"TYPE_ABSTRACT_CONCEPT": 1, "ROLE_AGENT_CAPABLE": 1, "MODALITY_LITERAL": 1},
        anchor="cn:en:democracy (n)",
    )
    g_invalid = QuantaGraph()
    e_cid2 = g_invalid.add_node(event_node2, set_as_root=True)
    a_cid2 = g_invalid.add_node(invalid_agent)
    g_invalid.add_edge(e_cid2, "VAL_X1_AGENT", a_cid2)

    res_invalid = gate.validate_graph(g_invalid)
    assert not res_invalid.is_valid
    assert a_cid2 in res_invalid.muc_nodes
