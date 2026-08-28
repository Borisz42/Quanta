"""Tests for neuro-symbolic ASP validator gate and Minimal Unsatisfiable Core (MUC) extraction."""

import pytest
from core.asg import QuantaNode, QuantaGraph
from solver.validator_gate import ValidationGate


@pytest.fixture
def validator():
    return ValidationGate()


def test_valid_proposition_graph(validator):
    # "A golden retriever bit the mailman."
    event_node = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="wn:bite.v.01",
    )
    dog_node = QuantaNode(
        vector={"TYPE_ANIMATE": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:golden_retriever.n.01",
    )
    mailman_node = QuantaNode(
        vector={"TYPE_HUMAN": 1, "VAL_EXPERIENCER": 1},
        anchor="wn:mailman.n.01",
    )

    graph = QuantaGraph()
    event_cid = graph.add_node(event_node, set_as_root=True)
    dog_cid = graph.add_node(dog_node)
    mailman_cid = graph.add_node(mailman_node)

    graph.add_edge(event_cid, "VAL_X1_AGENT", dog_cid)
    graph.add_edge(event_cid, "VAL_X2_PATIENT", mailman_cid)

    result = validator.validate_graph(graph)
    assert result.is_valid
    assert len(result.errors) == 0


def test_abstract_concept_agent_violation_muc(validator):
    # An abstract concept (e.g. "Democracy") cannot be agent-capable without figurative modality
    invalid_node = QuantaNode(
        vector={
            "TYPE_ABSTRACT_CONCEPT": 1,
            "ROLE_AGENT_CAPABLE": 1,
            "MODALITY_LITERAL": 1,
        },
        anchor="wn:democracy.n.01",
    )

    result = validator.validate_node(invalid_node)
    assert not result.is_valid
    assert len(result.muc_slots) > 0

    # Ensure MUC accurately flagged the conflicting slots
    violated_slot_names = [slot for _, slot, _ in result.muc_slots]
    assert "TYPE_ABSTRACT_CONCEPT" in violated_slot_names
    assert "ROLE_AGENT_CAPABLE" in violated_slot_names


def test_figurative_modality_allows_metaphorical_agency(validator):
    # With MODALITY_FIGURATIVE, metaphorical agency is valid ("Time flies", "Democracy speaks")
    figurative_node = QuantaNode(
        vector={
            "TYPE_ABSTRACT_CONCEPT": 1,
            "ROLE_AGENT_CAPABLE": 1,
            "MODALITY_FIGURATIVE": 1,
        },
        anchor="wn:democracy.n.01",
    )

    result = validator.validate_node(figurative_node)
    assert result.is_valid


def test_inanimate_sentience_contradiction_muc(validator):
    # A stone cannot be sentient
    rock_node = QuantaNode(
        vector={
            "TYPE_INANIMATE_PHYSICAL": 1,
            "ROLE_SENTIENT": 1,
        },
        anchor="wn:rock.n.01",
    )

    result = validator.validate_node(rock_node)
    assert not result.is_valid
    violated_slots = [slot for _, slot, _ in result.muc_slots]
    assert "TYPE_INANIMATE_PHYSICAL" in violated_slots
    assert "ROLE_SENTIENT" in violated_slots


def test_deontic_obligation_prohibition_contradiction(validator):
    # Simultaneous obligation and prohibition is logically impossible
    contradictory_node = QuantaNode(
        vector={
            "EPIST_DEONTIC_OBLIGATION": 1,
            "EPIST_DEONTIC_PROHIBITION": 1,
            "TYPE_PROPOSITION": 1,
        },
    )

    result = validator.validate_node(contradictory_node)
    assert not result.is_valid
    violated_slots = [slot for _, slot, _ in result.muc_slots]
    assert "EPIST_DEONTIC_OBLIGATION" in violated_slots
    assert "EPIST_DEONTIC_PROHIBITION" in violated_slots


def test_allen_temporal_contradiction(validator):
    # Simultaneous Before and During is logically impossible
    temporal_conflict = QuantaNode(
        vector={
            "TEMP_ALLEN_BEFORE": 1,
            "TEMP_ALLEN_DURING": 1,
            "TYPE_TEMPORAL_INTERVAL": 1,
        },
    )

    result = validator.validate_node(temporal_conflict)
    assert not result.is_valid
    violated_slots = [slot for _, slot, _ in result.muc_slots]
    assert "TEMP_ALLEN_BEFORE" in violated_slots
    assert "TEMP_ALLEN_DURING" in violated_slots


def test_rcc8_spatial_contradiction(validator):
    # Disconnected and Congruent Eq is spatially contradictory
    spatial_conflict = QuantaNode(
        vector={
            "SPATIAL_RCC_DISCONNECTED": 1,
            "SPATIAL_RCC_CONGRUENT_EQ": 1,
            "TYPE_SPATIAL_REGION": 1,
        },
    )

    result = validator.validate_node(spatial_conflict)
    assert not result.is_valid
    violated_slots = [slot for _, slot, _ in result.muc_slots]
    assert "SPATIAL_RCC_DISCONNECTED" in violated_slots
    assert "SPATIAL_RCC_CONGRUENT_EQ" in violated_slots

