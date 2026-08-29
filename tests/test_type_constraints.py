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


def test_validate_valency_function():
    """Verify fast validate_valency function for ontological constraints."""
    from src.parser.lexical_grounder import validate_valency

    # Predicate: THINK
    think_node = QuantaNode(
        vector={"NSM_THINK": 1, "MODALITY_LITERAL": 1},
        anchor="wn:think.v.01"
    )

    # Valid agent: Human
    human_node = QuantaNode(
        vector={"TYPE_HUMAN": 1, "TYPE_ANIMATE": 1, "ROLE_SENTIENT": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:human.n.01"
    )

    # Invalid agent: Rock (inanimate physical)
    rock_node = QuantaNode(
        vector={"TYPE_INANIMATE_PHYSICAL": 1, "WN_OBJECT_NATURAL": 1},
        anchor="wn:rock.n.01"
    )

    # THINK(human) -> True
    assert validate_valency(think_node, "VAL_X1_AGENT", human_node) is True

    # THINK(rock) -> False (inanimate physical cannot be agent of cognitive verb)
    assert validate_valency(think_node, "VAL_X1_AGENT", rock_node) is False

    # Variable binding bypasses check
    var_node = QuantaNode(
        vector={"GRAPH_VARIABLE_BIND": 1},
        literal="x"
    )
    assert validate_valency(think_node, "VAL_X1_AGENT", var_node) is True


def test_pearl_causal_contradictions(validator):
    """Test Pearl's causal hierarchy exclusivity rules (7B.3)."""
    # Direct mechanism vs Preventive blocker
    causal_conflict = QuantaNode(
        vector={
            "CAUSAL_DIRECT_MECHANISM": 1,
            "CAUSAL_PREVENTIVE_BLOCK": 1,
        },
    )
    res1 = validator.validate_node(causal_conflict)
    assert not res1.is_valid
    slots1 = [s for _, s, _ in res1.muc_slots]
    assert "CAUSAL_DIRECT_MECHANISM" in slots1
    assert "CAUSAL_PREVENTIVE_BLOCK" in slots1

    # Common Confounder vs Collider Effect
    confounder_collider = QuantaNode(
        vector={
            "CAUSAL_COMMON_CONFOUNDER": 1,
            "CAUSAL_COLLIDER_EFFECT": 1,
        },
    )
    res2 = validator.validate_node(confounder_collider)
    assert not res2.is_valid
    slots2 = [s for _, s, _ in res2.muc_slots]
    assert "CAUSAL_COMMON_CONFOUNDER" in slots2
    assert "CAUSAL_COLLIDER_EFFECT" in slots2


def test_structural_topology_invariants(validator):
    """Test Band 1 structural topology rules (7B.4)."""
    # Root vs Leaf on same node
    root_leaf_conflict = QuantaNode(
        vector={
            "GRAPH_ROOT_NODE": 1,
            "GRAPH_LEAF": 1,
        },
    )
    res1 = validator.validate_node(root_leaf_conflict)
    assert not res1.is_valid
    slots1 = [s for _, s, _ in res1.muc_slots]
    assert "GRAPH_ROOT_NODE" in slots1
    assert "GRAPH_LEAF" in slots1

    # Branch then vs else
    branch_conflict = QuantaNode(
        vector={
            "GRAPH_BRANCH_THEN": 1,
            "GRAPH_BRANCH_ELSE": 1,
        },
    )
    res2 = validator.validate_node(branch_conflict)
    assert not res2.is_valid

    # Logical connectives: AND vs XOR
    connective_conflict = QuantaNode(
        vector={
            "LJB_JE_AND": 1,
            "LJB_JON_XOR": 1,
        },
    )
    res3 = validator.validate_node(connective_conflict)
    assert not res3.is_valid


def test_rock_thinks_asp_graph_rejection(validator):
    """Test 7A.4: 'The rock thinks' ASP graph triggers violation; 'The human thinks' passes."""
    # 1. "The rock thinks" -> Rejection
    think_event = QuantaNode(
        vector={"NSM_THINK": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="wn:think.v.01",
    )
    rock_entity = QuantaNode(
        vector={"TYPE_INANIMATE_PHYSICAL": 1, "TYPE_NATURAL_OBJECT": 1},
        anchor="wn:rock.n.01",
    )
    g_invalid = QuantaGraph()
    t_cid = g_invalid.add_node(think_event, set_as_root=True)
    r_cid = g_invalid.add_node(rock_entity)
    g_invalid.add_edge(t_cid, "VAL_X1_AGENT", r_cid)

    res_invalid = validator.validate_graph(g_invalid)
    assert not res_invalid.is_valid
    assert len(res_invalid.errors) > 0

    # 2. "The human thinks" -> Valid
    think_event_valid = QuantaNode(
        vector={"NSM_THINK": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="wn:think.v.01",
    )
    human_entity = QuantaNode(
        vector={"TYPE_HUMAN": 1, "TYPE_ANIMATE": 1, "ROLE_AGENT_CAPABLE": 1, "ROLE_SENTIENT": 1},
        anchor="wn:human.n.01",
    )
    g_valid = QuantaGraph()
    t2_cid = g_valid.add_node(think_event_valid, set_as_root=True)
    h_cid = g_valid.add_node(human_entity)
    g_valid.add_edge(t2_cid, "VAL_X1_AGENT", h_cid)

    res_valid = validator.validate_graph(g_valid)
    assert res_valid.is_valid


def test_multi_node_isolated_muc_pinpointing(validator):
    """Test 7C.4: Create a graph with one invalid node among valid ones; verify MUC pinpoints exactly the invalid node."""
    event_node = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="wn:chase.v.01",
    )
    dog_node = QuantaNode(
        vector={"TYPE_ANIMATE": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:dog.n.01",
    )
    ball_node = QuantaNode(
        vector={"TYPE_INANIMATE_PHYSICAL": 1, "TYPE_ARTIFACT": 1},
        anchor="wn:ball.n.01",
    )
    garden_node = QuantaNode(
        vector={"TYPE_SPATIAL_REGION": 1, "WN_LOCATION_PLACE": 1},
        anchor="wn:garden.n.01",
    )
    # The single flawed node
    flawed_node = QuantaNode(
        vector={"TEMP_ALLEN_BEFORE": 1, "TEMP_ALLEN_DURING": 1},
    )

    graph = QuantaGraph()
    e_cid = graph.add_node(event_node, set_as_root=True)
    d_cid = graph.add_node(dog_node)
    b_cid = graph.add_node(ball_node)
    g_cid = graph.add_node(garden_node)
    flawed_cid = graph.add_node(flawed_node)

    graph.add_edge(e_cid, "VAL_X1_AGENT", d_cid)
    graph.add_edge(e_cid, "VAL_X2_PATIENT", b_cid)
    graph.add_edge(e_cid, "VAL_LOCATION_SLOT", g_cid)
    graph.add_edge(e_cid, "VAL_TIME_SLOT", flawed_cid)

    is_valid, muc = validator.validate(graph)
    assert not is_valid
    assert muc is not None
    assert flawed_cid in muc
    # Valid nodes should not be blamed in the MUC
    assert d_cid not in muc
    assert b_cid not in muc
    assert g_cid not in muc


def test_validator_gate_alias_and_unpacking():
    """Test ValidatorGate alias (7C.1) and tuple return API (7C.3)."""
    from solver import ValidatorGate, ValidationGate

    assert ValidatorGate is ValidationGate
    gate = ValidatorGate()

    g = QuantaGraph()
    node = QuantaNode(vector={"NSM_DO": 1, "TYPE_EVENT": 1})
    g.add_node(node)

    is_valid, muc = gate.validate(g)
    assert is_valid is True
    assert muc is None



