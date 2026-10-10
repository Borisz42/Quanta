"""Unit tests for Multi-Scale Abstract Syntax Graph (ASG) and passage upgrade (§Phase 4).

Covers:
- QuantaNode parent_macro_id preservation and serialization
- BinaryNodeTable parent macro side table (128-byte layout preservation)
- Mixed-graph construction (micro nodes + macro nodes + concept anchors)
- In-place passage upgrade (upgrade_passage) preserving existing edges
"""

import pytest

from core.asg import QuantaGraph, QuantaNode
from core.binary_node import BinaryNodeTable, QuantaSemanticNodeStruct
from core.types import QuantaVector


def test_quanta_node_parent_macro_id():
    """Verifies parent_macro_id storage, binding, cloning, and dict serialization in QuantaNode."""
    node = QuantaNode(
        anchor="test_entity",
        passage_id="P_micro_101",
        parent_macro_id="P_macro_1",
        span_start=10,
        span_end=35,
    )
    assert node.parent_macro_id == "P_macro_1"
    assert node.passage_id == "P_micro_101"

    # Test bind_passage with parent_macro_id
    node.bind_passage("P_micro_102", 50, 80, confidence=0.95, parent_macro_id="P_macro_2")
    assert node.passage_id == "P_micro_102"
    assert node.parent_macro_id == "P_macro_2"
    assert node.confidence == 0.95

    # Test clone
    cloned = node.clone()
    assert cloned.parent_macro_id == "P_macro_2"
    assert cloned.passage_id == "P_micro_102"

    # Test serialization round-trip
    d = node.to_dict()
    assert d["parent_macro_id"] == "P_macro_2"
    assert d["passage_id"] == "P_micro_102"

    restored = QuantaNode.from_dict(d)
    assert restored.parent_macro_id == "P_macro_2"
    assert restored.passage_id == "P_micro_102"


def test_binary_node_table_parent_macro_side_table():
    """Verifies that BinaryNodeTable stores parent macro references in side tables without modifying 128-byte structs."""
    table = BinaryNodeTable()

    # Append a 128-byte node struct
    struct = QuantaSemanticNodeStruct.create(
        node_id=1,
        passage_id=101,
        span_start=0,
        span_end=20,
        concept_code=500,
    )
    table.append(struct)

    assert len(table) == 1
    # Ensure raw struct size is strictly 128 bytes
    assert QuantaSemanticNodeStruct.NODE_SIZE == 128

    # Register side table mappings
    table.register_parent_macro(node_id=1, parent_macro_id="macro_block_A")
    table.register_passage_parent_macro(passage_id=101, parent_macro_id="macro_block_A")

    assert table.get_parent_macro(1) == "macro_block_A"
    assert table.get_passage_parent_macro(101) == "macro_block_A"
    assert table.get_parent_macro(999) is None


def test_mixed_graph_construction():
    """Verifies mixed-scale graph construction with macro blocks, micro passages, and concept anchors."""
    graph = QuantaGraph()

    # 1. Add a coarse macro node
    macro_vec = QuantaVector.zeros()
    macro_vec["TYPE_PROPOSITION"] = 1
    macro_node = QuantaNode(
        vector=macro_vec,
        anchor="macro:section_thermodynamics",
        passage_id="macro_01",
        node_type="macro_passage",
    )
    macro_cid = graph.add_node(macro_node)

    # 2. Add fine-grained micro nodes
    micro1_vec = QuantaVector.zeros()
    micro1_vec["TYPE_ABSTRACT_CONCEPT"] = 1
    micro1 = QuantaNode(
        vector=micro1_vec,
        anchor="carnot_engine",
        passage_id="micro_01",
        parent_macro_id="macro_01",
    )
    m1_cid = graph.add_node(micro1)

    micro2_vec = QuantaVector.zeros()
    micro2_vec["TYPE_PROPOSITION"] = 1
    micro2 = QuantaNode(
        vector=micro2_vec,
        anchor="isothermal_expansion",
        passage_id="micro_01",
        parent_macro_id="macro_01",
    )
    m2_cid = graph.add_node(micro2)

    # 3. Add inter-scale structural edges
    graph.add_edge(m1_cid, "INTER_SCALE_PARENT", macro_cid)
    graph.add_edge(m2_cid, "INTER_SCALE_PARENT", macro_cid)
    graph.add_edge(m1_cid, "VAL_X1_AGENT", m2_cid)

    # 4. Add concept anchor for macro block
    concept_anchor = graph.add_concept_anchor("macro_01", concept_code=3042)
    assert concept_anchor.node_type == "concept_anchor"
    assert concept_anchor.concept_code == 3042

    # Verify graph connectivity and indices
    assert "macro_01" in graph.passage_to_concept_anchors
    assert concept_anchor.cid in graph.passage_to_concept_anchors["macro_01"]

    # Copy graph and verify integrity
    cloned_graph = graph.copy()
    assert len(cloned_graph.nodes) == len(graph.nodes)
    assert "macro_01" in cloned_graph.passage_to_concept_anchors


def test_upgrade_passage_keeps_existing_edges():
    """Verifies that upgrade_passage merges fine-grained subgraphs without breaking existing graph edges."""
    graph = QuantaGraph()

    # Step 1: External node from previous chunk
    n_prev_vec = QuantaVector.zeros()
    n_prev_vec["TYPE_ABSTRACT_CONCEPT"] = 1
    n_prev = QuantaNode(vector=n_prev_vec, anchor="refrigerant_cycle", passage_id="chunk_0")
    n_prev_cid = graph.add_node(n_prev)

    # Step 2: Unprocessed coarse passage node for chunk_1
    coarse_node_vec = QuantaVector.zeros()
    coarse_node_vec["TYPE_PROPOSITION"] = 1
    coarse_node = QuantaNode(
        vector=coarse_node_vec,
        anchor="passage:chunk_1",
        passage_id="chunk_1",
        node_type="coarse_passage",
    )
    coarse_cid = graph.add_node(coarse_node)

    # External edge pointing from previous chunk to coarse node
    graph.add_edge(n_prev_cid, "TEMP_ALLEN_BEFORE", coarse_cid)

    # Concept anchor attached to chunk_1
    anchor_node = graph.add_concept_anchor("chunk_1", concept_code=772)

    # Verify coarse graph state
    assert len(graph.nodes) == 3

    # Step 3: Now simulate delayed transduction producing a fine-grained subgraph for chunk_1
    subgraph = QuantaGraph()
    fine_event_vec = QuantaVector.zeros()
    fine_event_vec["TYPE_PROPOSITION"] = 1
    fine_event = QuantaNode(
        vector=fine_event_vec,
        anchor="compressor_activation",
        passage_id="chunk_1",
        node_type="event",
    )
    sub_root_cid = subgraph.add_node(fine_event, set_as_root=True)

    fine_entity_vec = QuantaVector.zeros()
    fine_entity_vec["TYPE_ABSTRACT_CONCEPT"] = 1
    fine_entity = QuantaNode(
        vector=fine_entity_vec,
        anchor="valve_pressure",
        passage_id="chunk_1",
        node_type="entity",
    )
    sub_entity_cid = subgraph.add_node(fine_entity)
    subgraph.add_edge(sub_root_cid, "CAUSAL_LEADS_TO", sub_entity_cid)

    # Step 4: Upgrade passage in main graph
    graph.upgrade_passage("chunk_1", subgraph)

    # Step 5: Verify that:
    # (a) Fine nodes are integrated
    assert graph.get_node(sub_root_cid) is not None
    assert graph.get_node(sub_entity_cid) is not None

    # (b) Internal edges in subgraph are preserved
    event_node = graph.get_node(sub_root_cid)
    assert "CAUSAL_LEADS_TO" in event_node.edges

    # (c) Concept anchor is connected to newly transduced nodes
    assert "CONCEPT_ANCHOR" in event_node.edges or any(
        "CONCEPT_ANCHOR" in n.edges for n in graph.get_nodes_for_passage("chunk_1")
    )

    # (d) External edge from n_prev still exists and was forwarded
    prev_node = graph.get_node(n_prev_cid)
    assert "TEMP_ALLEN_BEFORE" in prev_node.edges
    target_nodes = [graph.get_node(t) for t in prev_node.edges["TEMP_ALLEN_BEFORE"]]
    assert any(n is not None and n.anchor in ("passage:chunk_1", "compressor_activation") for n in target_nodes)
