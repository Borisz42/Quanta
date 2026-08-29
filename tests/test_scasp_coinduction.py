"""Tests for s(CASP) / SWI-Prolog bridge and coinductive ASG reasoning (Phase 7D)."""

import pytest
from core.asg import QuantaNode, QuantaGraph
from parser.ast_parser import ASTForwardParser
from solver.scasp_bridge import SCaspBridge


@pytest.fixture
def bridge():
    return SCaspBridge()


def test_prolog_fact_generation(bridge):
    """Test generating valid Prolog facts from a QuantaGraph."""
    node = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="wn:bite.v.01",
    )
    dog = QuantaNode(
        vector={"TYPE_ANIMATE": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:dog.n.01",
    )
    g = QuantaGraph()
    n_cid = g.add_node(node, set_as_root=True)
    d_cid = g.add_node(dog)
    g.add_edge(n_cid, "VAL_X1_AGENT", d_cid)

    facts = bridge.graph_to_prolog_facts(g)
    assert f"node('{node.cid}')." in facts
    assert f"node('{dog.cid}')." in facts
    assert "slot(" in facts
    assert "TYPE_EVENT" in facts
    assert f"edge('{node.cid}', 'VAL_X1_AGENT', '{dog.cid}')." in facts

    program = bridge.generate_prolog_program(g)
    assert "valid_asg :- not false." in program
    assert "TYPE_ABSTRACT_CONCEPT" in program


def test_coinductive_recursive_factorial_ast(bridge):
    """Test 7D.4: Validate cyclic graph (recursive factorial ASG) via coinductive reasoning."""
    source_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    parser = ASTForwardParser()
    graph = parser.parse_ast_node(source_code)

    # Validate coinductive rational tree
    coind_res = bridge.validate_coinductive_cycles(graph)
    assert coind_res.is_valid
    assert len(coind_res.cycles_detected) > 0
    assert len(coind_res.coinductive_hypotheses) > 0
    assert len(coind_res.errors) == 0

    # End-to-end bridge validation
    val_res = bridge.validate(graph)
    assert val_res.is_valid


def test_coinductive_rejection_of_ungrounded_cycle(bridge):
    """Test that an ungrounded infinite loop without function head or base condition is rejected."""
    # Create a cyclic graph with 2 nodes that are plain concepts (not a function, no branches)
    n1 = QuantaNode(vector={"TYPE_INANIMATE_PHYSICAL": 1, "WN_OBJECT_NATURAL": 1})
    n2 = QuantaNode(vector={"TYPE_INANIMATE_PHYSICAL": 1, "WN_OBJECT_NATURAL": 1})

    g = QuantaGraph()
    cid1 = g.add_node(n1, set_as_root=True)
    cid2 = g.add_node(n2)

    g.add_edge(cid1, "RELATED_TO", cid2)
    g.add_edge(cid2, "RELATED_TO", cid1)

    coind_res = bridge.validate_coinductive_cycles(g)
    assert not coind_res.is_valid
    assert len(coind_res.errors) > 0

    val_res = bridge.validate(g)
    assert not val_res.is_valid


def test_coinductive_acyclic_graph_passes(bridge):
    """Test that standard acyclic proposition graphs pass coinductive validation."""
    node = QuantaNode(vector={"NSM_DO": 1, "TYPE_EVENT": 1})
    g = QuantaGraph()
    g.add_node(node)

    coind_res = bridge.validate_coinductive_cycles(g)
    assert coind_res.is_valid
    assert len(coind_res.cycles_detected) == 0

    val_res = bridge.validate(g)
    assert val_res.is_valid
