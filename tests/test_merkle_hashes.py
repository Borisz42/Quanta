"""Tests for BLAKE3 Content Identifier (CID) generation and Merkle ASG graphs."""

import pytest
from quanta.core.asg import QuantaNode, QuantaGraph
from quanta.core.types import QuantaVector


def test_deterministic_node_cid():
    node1 = QuantaNode(
        vector={"NSM_DO": 1, "LJB_PU_PAST_TENSE": 1, "TYPE_EVENT": 1},
        anchor="wn:bite.v.01",
    )
    node2 = QuantaNode(
        vector={"NSM_DO": 1, "LJB_PU_PAST_TENSE": 1, "TYPE_EVENT": 1},
        anchor="wn:bite.v.01",
    )

    # Identical content must produce identical 256-bit hex hash
    assert node1.cid == node2.cid
    assert len(node1.cid) == 64  # 256 bits = 64 hex characters


def test_cid_sensitivity_to_mutation():
    node = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1},
        anchor="wn:bite.v.01",
    )
    original_cid = node.cid

    # Mutating slot changes CID
    node.set_slot("LJB_PU_PAST_TENSE", 1)
    new_cid = node.cid
    assert original_cid != new_cid

    # Mutating anchor changes CID
    node.anchor = "wn:eat.v.01"
    node.invalidate_cache()
    assert node.cid != new_cid


def test_asg_merkle_folding_and_tamper_detection():
    # Build a small graph: Agent -> Action -> Patient
    event_node = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="wn:bite.v.01",
    )
    dog_node = QuantaNode(
        vector={"TYPE_ANIMATE": 1, "ROLE_AGENT_CAPABLE": 1},
        anchor="wn:golden_retriever.n.01",
    )
    postman_node = QuantaNode(
        vector={"TYPE_HUMAN": 1, "VAL_EXPERIENCER": 1},
        anchor="wn:mailman.n.01",
    )

    graph = QuantaGraph()
    event_cid = graph.add_node(event_node, set_as_root=True)
    dog_cid = graph.add_node(dog_node)
    postman_cid = graph.add_node(postman_node)

    graph.add_edge(event_cid, "VAL_X1_AGENT", dog_cid)
    graph.add_edge(event_cid, "VAL_X2_PATIENT", postman_cid)

    merkle_root_1 = graph.compute_merkle_root()
    assert len(merkle_root_1) == 64

    # Validate structural integrity
    valid, errors = graph.validate_integrity()
    assert valid
    assert len(errors) == 0

    # Tamper with dog node vector without updating graph
    tampered_node = graph.get_node(dog_cid)
    tampered_node.vector[0] = 2  # Mutate vector
    tampered_node.invalidate_cache()

    # Integrity check should detect payload/CID mismatch
    valid_after_tamper, tamper_errors = graph.validate_integrity()
    assert not valid_after_tamper
    assert len(tamper_errors) > 0


def test_graph_serialization_roundtrip():
    event_node = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1},
        anchor="wn:run.v.01",
    )
    graph = QuantaGraph()
    graph.add_node(event_node, set_as_root=True)

    json_str = graph.to_json()
    reconstructed = QuantaGraph.from_json(json_str)

    assert reconstructed.root_cid == graph.root_cid
    assert reconstructed.compute_merkle_root() == graph.compute_merkle_root()
    assert len(reconstructed) == len(graph)
