"""Tests for BLAKE3 Content Identifier (CID) generation and Merkle ASG graphs."""

import json
import pytest
from core.asg import QuantaNode, QuantaGraph
from core.types import QuantaVector


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


def test_subgraph_folding_and_unfolding_roundtrip():
    """Verify that folding a subtree collapses active node count and unfolding restores exact graph."""
    event_node = QuantaNode(
        vector={"NSM_DO": 1, "TYPE_EVENT": 1, "MODALITY_LITERAL": 1},
        anchor="wn:bite.v.01",
    )
    dog_node = QuantaNode(
        vector={"TYPE_ANIMATE": 1, "WN_ANIMAL_FAUNA": 1},
        anchor="wn:golden_retriever.n.01",
    )
    postman_node = QuantaNode(
        vector={"TYPE_HUMAN": 1, "WN_PERSON_HUMAN": 1},
        anchor="wn:mailman.n.01",
    )

    graph = QuantaGraph()
    event_cid = graph.add_node(event_node, set_as_root=True)
    dog_cid = graph.add_node(dog_node)
    postman_cid = graph.add_node(postman_node)

    graph.add_edge(event_cid, "VAL_X1_AGENT", dog_cid)
    graph.add_edge(event_cid, "VAL_X2_PATIENT", postman_cid)

    initial_merkle = graph.compute_merkle_root()
    initial_node_count = len(graph)
    assert initial_node_count == 3

    # Fold dog node subtree
    storage = {}
    pointer_cid, sub_merkle = graph.fold_subgraph(dog_cid, storage=storage)

    assert sub_merkle in storage
    # Active nodes in main graph is now 2 (event_node + pointer_node, postman) -> dog_node replaced by pointer
    assert len(graph) == 3
    assert graph.get_node(dog_cid) is None
    pointer_node = graph.get_node(pointer_cid)
    assert pointer_node is not None
    assert pointer_node.get_slot("GRAPH_MERKLE_FOLD_POINT") == 1
    assert pointer_node.anchor == f"merkle:{sub_merkle}"

    # Unfold dog node subtree
    restored_root = graph.unfold_subgraph(pointer_cid, storage=storage)
    assert restored_root == dog_cid
    assert len(graph) == initial_node_count
    assert graph.get_node(dog_cid) is not None
    assert graph.compute_merkle_root() == initial_merkle


def test_quantanode_aliases_and_polymorphic_edges():
    """Verify QuantaNode aliases (semantic_vector, concept_label, node_cid, edge_table) and polymorphic edge init."""
    child_cid_bytes = bytes.fromhex("1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")
    child_cid_hex = child_cid_bytes.hex()

    node = QuantaNode(
        semantic_vector={"NSM_DO": 1, "TYPE_EVENT": 1},
        concept_label="wn:chase.v.01",
        edges=[("VAL_X1_AGENT", child_cid_hex), ("VAL_X2_PATIENT", child_cid_bytes)],
    )

    # Verify properties and aliases
    assert node.concept_label == "wn:chase.v.01"
    assert node.semantic_vector["NSM_DO"] == 1
    assert node.node_cid == node.cid
    assert isinstance(node.node_cid_bytes, bytes)
    assert len(node.node_cid_bytes) == 32
    assert node.node_cid_bytes.hex() == node.cid
    assert ("VAL_X1_AGENT", child_cid_hex) in node.edge_table
    assert ("VAL_X2_PATIENT", child_cid_hex) in node.edge_table

    # Verify setters
    old_cid = node.cid
    node.concept_label = "wn:run.v.01"
    assert node.anchor == "wn:run.v.01"
    assert node.cid != old_cid

    # Verify QuantaGraph tree_aggregate_vector and aggregate_vector aliases
    g = QuantaGraph()
    g.add_node(node)
    assert g.tree_aggregate_vector["NSM_DO"] == 1
    assert g.aggregate_vector()["NSM_DO"] == 1


def test_example_a_canonical_affirmative_graph():
    """Verify Phase 3 Item 3B.6: Canonical Example A (Affirmative Action with Modifiers & Spatial Location).

    Sentence: 'A golden retriever bit the mailman in the garden.'
    """
    # 1. Root Predicate Node (wn:bite.v.01, literal: "bit")
    root_node = QuantaNode(
        vector={
            "NSM_DO": 1,
            "NSM_TOUCH": 1,
            "NSM_TRUE": 1,
            "VAL_X1_AGENT": 1,
            "VAL_X2_PATIENT": 1,
            "VAL_LOCATION_SLOT": 1,
            "LJB_PU_PAST_TENSE": 1,
            "GRAPH_ROOT_NODE": 1,
            "TYPE_EVENT": 1,
            "MODALITY_LITERAL": 1,
            "WN_ACT_ACTION": 1,
            "EPIST_DIRECT_OBSERVATION": 1,
            "EPIST_PROB_CERTAIN": 1,
        },
        anchor="wn:bite.v.01",
        literal="bit",
    )

    # 2. Agent Child Node (wn:golden_retriever.n.01, literal: "golden retriever")
    agent_node = QuantaNode(
        vector={
            "NSM_ONE": 1,
            "VAL_X1_AGENT": 1,
            "GRAPH_LEAF": 1,
            "TYPE_ANIMATE": 1,
            "ROLE_AGENT_CAPABLE": 1,
            "ROLE_SENTIENT": 1,
            "ROLE_MOVEABLE": 1,
            "WN_ANIMAL_FAUNA": 1,
            "EPIST_PROB_CERTAIN": 1,
        },
        anchor="wn:golden_retriever.n.01",
        literal="golden retriever",
    )

    # 3. Patient Child Node (wn:mailman.n.01, literal: "mailman")
    patient_node = QuantaNode(
        vector={
            "NSM_THIS": 1,
            "VAL_X2_PATIENT": 1,
            "GRAPH_LEAF": 1,
            "TYPE_HUMAN": 1,
            "TYPE_ANIMATE": 1,
            "ROLE_COMMUNICATOR": 1,
            "ROLE_SENTIENT": 1,
            "ROLE_PATIENT_TARGET": 1,
            "WN_PERSON_HUMAN": 1,
            "EPIST_PROB_CERTAIN": 1,
        },
        anchor="wn:mailman.n.01",
        literal="mailman",
    )

    # 4. Location Child Node (wn:garden.n.01, literal: "garden")
    location_node = QuantaNode(
        vector={
            "NSM_THIS": 1,
            "NSM_INSIDE": 1,
            "VAL_LOCATION_SLOT": 1,
            "GRAPH_LEAF": 1,
            "TYPE_SPATIAL_REGION": 1,
            "WN_LOCATION_PLACE": 1,
            "SPATIAL_RCC_NON_TANG_PART": 1,
        },
        anchor="wn:garden.n.01",
        literal="garden",
    )

    graph = QuantaGraph()
    root_cid = graph.add_node(root_node, set_as_root=True)
    agent_cid = graph.add_node(agent_node)
    patient_cid = graph.add_node(patient_node)
    loc_cid = graph.add_node(location_node)

    graph.add_edge(root_cid, "VAL_X1_AGENT", agent_cid)
    graph.add_edge(root_cid, "VAL_X2_PATIENT", patient_cid)
    graph.add_edge(root_cid, "VAL_LOCATION_SLOT", loc_cid)

    assert len(graph) == 4
    assert graph.root_cid is not None

    # Verify children retrieval
    agent_children = graph.get_children(graph.root_cid, relation="VAL_X1_AGENT")
    assert len(agent_children) == 1
    assert agent_children[0].anchor == "wn:golden_retriever.n.01"

    patient_children = graph.get_children(graph.root_cid, relation="VAL_X2_PATIENT")
    assert len(patient_children) == 1
    assert patient_children[0].anchor == "wn:mailman.n.01"

    location_children = graph.get_children(graph.root_cid, relation="VAL_LOCATION_SLOT")
    assert len(location_children) == 1
    assert location_children[0].anchor == "wn:garden.n.01"

    # Compute whole-tree proposition vector via lattice join (⊔_k)
    v_tree = graph.to_proposition_vector()

    # Verify Band 0 slots
    assert v_tree["NSM_DO"] == 1
    assert v_tree["NSM_TOUCH"] == 1
    assert v_tree["NSM_TRUE"] == 1
    assert v_tree["NSM_ONE"] == 1
    assert v_tree["NSM_THIS"] == 1
    assert v_tree["NSM_INSIDE"] == 1

    # Verify Band 1 slots
    assert v_tree["VAL_X1_AGENT"] == 1
    assert v_tree["VAL_X2_PATIENT"] == 1
    assert v_tree["VAL_LOCATION_SLOT"] == 1
    assert v_tree["LJB_PU_PAST_TENSE"] == 1
    assert v_tree["GRAPH_ROOT_NODE"] == 1
    assert v_tree["GRAPH_LEAF"] == 1

    # Verify Band 2 slots
    assert v_tree["TYPE_EVENT"] == 1
    assert v_tree["TYPE_ANIMATE"] == 1
    assert v_tree["TYPE_HUMAN"] == 1
    assert v_tree["TYPE_SPATIAL_REGION"] == 1
    assert v_tree["ROLE_AGENT_CAPABLE"] == 1
    assert v_tree["ROLE_SENTIENT"] == 1
    assert v_tree["ROLE_COMMUNICATOR"] == 1
    assert v_tree["ROLE_PATIENT_TARGET"] == 1
    assert v_tree["ROLE_MOVEABLE"] == 1
    assert v_tree["MODALITY_LITERAL"] == 1
    assert v_tree["WN_ACT_ACTION"] == 1
    assert v_tree["WN_ANIMAL_FAUNA"] == 1
    assert v_tree["WN_PERSON_HUMAN"] == 1
    assert v_tree["WN_LOCATION_PLACE"] == 1

    # Verify Band 3 slots
    assert v_tree["EPIST_DIRECT_OBSERVATION"] == 1
    assert v_tree["EPIST_PROB_CERTAIN"] == 1
    assert v_tree["SPATIAL_RCC_NON_TANG_PART"] == 1


def test_example_b_canonical_negated_graph():
    """Verify Phase 3 Item 3B.7: Canonical Example B (Explicit Epistemic Negation).

    Sentence: 'The dog did not bite the mailman.'
    """
    # 1. Root Predicate Node (wn:bite.v.01, literal: "did not bite", Polarity 2)
    root_node = QuantaNode(
        vector={
            "NSM_DO": 2,
            "NSM_TOUCH": 2,
            "VAL_X1_AGENT": 1,
            "VAL_X2_PATIENT": 1,
            "LJB_NA_NEGATION": 2,
            "LJB_PU_PAST_TENSE": 1,
            "GRAPH_ROOT_NODE": 1,
            "TYPE_EVENT": 1,
            "MODALITY_LITERAL": 1,
            "WN_ACT_ACTION": 1,
            "EPIST_DIRECT_OBSERVATION": 1,
            "EPIST_PROB_CERTAIN": 1,
        },
        anchor="wn:bite.v.01",
        literal="did not bite",
    )

    # 2. Agent Child Node (wn:dog.n.01, literal: "dog")
    agent_node = QuantaNode(
        vector={
            "NSM_THIS": 1,
            "VAL_X1_AGENT": 1,
            "GRAPH_LEAF": 1,
            "TYPE_ANIMATE": 1,
            "ROLE_AGENT_CAPABLE": 1,
            "ROLE_SENTIENT": 1,
            "WN_ANIMAL_FAUNA": 1,
            "EPIST_PROB_CERTAIN": 1,
        },
        anchor="wn:dog.n.01",
        literal="dog",
    )

    # 3. Patient Child Node (wn:mailman.n.01, literal: "mailman")
    patient_node = QuantaNode(
        vector={
            "NSM_THIS": 1,
            "VAL_X2_PATIENT": 1,
            "GRAPH_LEAF": 1,
            "TYPE_HUMAN": 1,
            "TYPE_ANIMATE": 1,
            "ROLE_COMMUNICATOR": 1,
            "ROLE_SENTIENT": 1,
            "WN_PERSON_HUMAN": 1,
            "EPIST_PROB_CERTAIN": 1,
        },
        anchor="wn:mailman.n.01",
        literal="mailman",
    )

    graph = QuantaGraph()
    root_cid = graph.add_node(root_node, set_as_root=True)
    agent_cid = graph.add_node(agent_node)
    patient_cid = graph.add_node(patient_node)

    graph.add_edge(root_cid, "VAL_X1_AGENT", agent_cid)
    graph.add_edge(root_cid, "VAL_X2_PATIENT", patient_cid)

    assert len(graph) == 3
    assert graph.root_cid is not None

    # Compute whole-tree proposition vector via lattice join (⊔_k)
    v_tree = graph.to_proposition_vector()

    # Verify Band 0 slots (explicit negation value 2)
    assert v_tree["NSM_DO"] == 2
    assert v_tree["NSM_TOUCH"] == 2
    assert v_tree["NSM_THIS"] == 1

    # Verify Band 1 slots
    assert v_tree["VAL_X1_AGENT"] == 1
    assert v_tree["VAL_X2_PATIENT"] == 1
    assert v_tree["LJB_NA_NEGATION"] == 2
    assert v_tree["LJB_PU_PAST_TENSE"] == 1
    assert v_tree["GRAPH_ROOT_NODE"] == 1
    assert v_tree["GRAPH_LEAF"] == 1

    # Verify Band 2 slots
    assert v_tree["TYPE_EVENT"] == 1
    assert v_tree["TYPE_ANIMATE"] == 1
    assert v_tree["TYPE_HUMAN"] == 1
    assert v_tree["ROLE_AGENT_CAPABLE"] == 1
    assert v_tree["ROLE_SENTIENT"] == 1
    assert v_tree["ROLE_COMMUNICATOR"] == 1
    assert v_tree["MODALITY_LITERAL"] == 1
    assert v_tree["WN_ACT_ACTION"] == 1
    assert v_tree["WN_ANIMAL_FAUNA"] == 1
    assert v_tree["WN_PERSON_HUMAN"] == 1

    # Verify Band 3 slots
    assert v_tree["EPIST_DIRECT_OBSERVATION"] == 1
    assert v_tree["EPIST_PROB_CERTAIN"] == 1


def test_example_c_canonical_uncertainty_graph():
    """Verify Phase 3 Item 3B.8: Canonical Example C (Epistemic Uncertainty & Question Query).

    Sentence: 'Did the dog perhaps bite a mailman?'
    """
    # 1. Root Predicate Node (wn:bite.v.01, literal: "bite", Polarity 3)
    root_node = QuantaNode(
        vector={
            "NSM_DO": 3,
            "NSM_TOUCH": 3,
            "NSM_MAYBE": 3,
            "VAL_X1_AGENT": 1,
            "VAL_X2_PATIENT": 1,
            "GRAPH_QUERY_TARGET": 3,
            "LJB_PU_PAST_TENSE": 1,
            "GRAPH_ROOT_NODE": 1,
            "TYPE_EVENT": 1,
            "TYPE_PROPOSITION": 3,
            "MODALITY_HYPOTHETICAL": 3,
            "EPIST_PROB_MARGINAL": 3,
            "EPIST_FUZZY_PLAUSIBILITY": 3,
        },
        anchor="wn:bite.v.01",
        literal="bite",
    )

    # 2. Agent Child Node (wn:dog.n.01, literal: "dog")
    agent_node = QuantaNode(
        vector={
            "NSM_THIS": 1,
            "VAL_X1_AGENT": 1,
            "GRAPH_LEAF": 1,
            "TYPE_ANIMATE": 1,
            "ROLE_AGENT_CAPABLE": 1,
            "ROLE_SENTIENT": 1,
            "WN_ANIMAL_FAUNA": 1,
            "EPIST_PROB_CERTAIN": 1,
        },
        anchor="wn:dog.n.01",
        literal="dog",
    )

    # 3. Patient Child Node (wn:mailman.n.01, literal: "mailman")
    patient_node = QuantaNode(
        vector={
            "NSM_ONE": 1,
            "VAL_X2_PATIENT": 1,
            "GRAPH_LEAF": 1,
            "TYPE_HUMAN": 1,
            "TYPE_ANIMATE": 1,
            "ROLE_COMMUNICATOR": 1,
            "ROLE_SENTIENT": 1,
            "WN_PERSON_HUMAN": 1,
            "EPIST_PROB_CERTAIN": 1,
        },
        anchor="wn:mailman.n.01",
        literal="mailman",
    )

    graph = QuantaGraph()
    root_cid = graph.add_node(root_node, set_as_root=True)
    agent_cid = graph.add_node(agent_node)
    patient_cid = graph.add_node(patient_node)

    graph.add_edge(root_cid, "VAL_X1_AGENT", agent_cid)
    graph.add_edge(root_cid, "VAL_X2_PATIENT", patient_cid)

    assert len(graph) == 3
    assert graph.root_cid is not None

    # Compute whole-tree proposition vector via lattice join (⊔_k)
    v_tree = graph.to_proposition_vector()

    # Verify Band 0 slots (uncertainty/modal value 3)
    assert v_tree["NSM_DO"] == 3
    assert v_tree["NSM_TOUCH"] == 3
    assert v_tree["NSM_MAYBE"] == 3
    assert v_tree["NSM_THIS"] == 1
    assert v_tree["NSM_ONE"] == 1

    # Verify Band 1 slots
    assert v_tree["VAL_X1_AGENT"] == 1
    assert v_tree["VAL_X2_PATIENT"] == 1
    assert v_tree["GRAPH_QUERY_TARGET"] == 3
    assert v_tree["LJB_PU_PAST_TENSE"] == 1
    assert v_tree["GRAPH_ROOT_NODE"] == 1
    assert v_tree["GRAPH_LEAF"] == 1

    # Verify Band 2 slots
    assert v_tree["TYPE_EVENT"] == 1
    assert v_tree["TYPE_ANIMATE"] == 1
    assert v_tree["TYPE_HUMAN"] == 1
    assert v_tree["TYPE_PROPOSITION"] == 3
    assert v_tree["ROLE_AGENT_CAPABLE"] == 1
    assert v_tree["ROLE_SENTIENT"] == 1
    assert v_tree["ROLE_COMMUNICATOR"] == 1
    assert v_tree["MODALITY_HYPOTHETICAL"] == 3
    assert v_tree["WN_ANIMAL_FAUNA"] == 1
    assert v_tree["WN_PERSON_HUMAN"] == 1

    # Verify Band 3 slots
    assert v_tree["EPIST_PROB_CERTAIN"] == 1
    assert v_tree["EPIST_PROB_MARGINAL"] == 3
    assert v_tree["EPIST_FUZZY_PLAUSIBILITY"] == 3


def test_multilevel_subgraph_folding_and_unfolding():
    """Verify that multi-level (depth >= 3) subtrees fold and unfold cleanly without dangling edges."""
    g = QuantaGraph()
    root_node = QuantaNode(vector={"NSM_THINK": 1, "TYPE_EVENT": 1}, anchor="wn:investigate.v.01")
    agent_node = QuantaNode(vector={"TYPE_HUMAN": 1, "ROLE_AGENT_CAPABLE": 1}, anchor="wn:detective.n.01")
    equip_node = QuantaNode(vector={"TYPE_ARTIFACT": 1, "VAL_X5_INSTRUMENT": 1}, anchor="wn:magnifying_glass.n.01")
    lens_node = QuantaNode(vector={"TYPE_INANIMATE_PHYSICAL": 1, "NSM_SEE": 1}, anchor="wn:lens.n.01")
    loc_node = QuantaNode(vector={"NSM_INSIDE": 1, "VAL_LOCATION_SLOT": 1}, anchor="wn:room.n.01")

    r_cid = g.add_node(root_node, set_as_root=True)
    a_cid = g.add_node(agent_node)
    e_cid = g.add_node(equip_node)
    l_cid = g.add_node(lens_node)
    loc_cid = g.add_node(loc_node)

    g.add_edge(r_cid, "VAL_X1_AGENT", a_cid)
    g.add_edge(a_cid, "VAL_X5_INSTRUMENT", e_cid)
    g.add_edge(e_cid, "HAS_PART", l_cid)
    g.add_edge(r_cid, "VAL_LOCATION_SLOT", loc_cid)

    # Initial graph integrity
    valid, errors = g.validate_integrity()
    assert valid, f"Initial graph invalid: {errors}"
    initial_merkle = g.compute_merkle_root()
    assert g.root_cid == g.root.cid

    # Fold equipment subtree (depth 2 subtree: equip -> lens)
    storage = {}
    ptr_cid, sub_merkle = g.fold_subgraph(e_cid, storage=storage)

    # Verify structural integrity after folding
    valid_fold, fold_errors = g.validate_integrity()
    assert valid_fold, f"Graph invalid after fold: {fold_errors}"
    assert g.root_cid == g.root.cid
    assert g.get_node(e_cid) is None
    assert g.get_node(l_cid) is None

    # Semantic proposition vector must preserve lens & equipment slots
    prop_vec = g.to_proposition_vector()
    assert prop_vec["TYPE_ARTIFACT"] == 1
    assert prop_vec["TYPE_INANIMATE_PHYSICAL"] == 1
    assert prop_vec["NSM_SEE"] == 1

    # Unfold equipment subtree
    restored_e = g.unfold_subgraph(ptr_cid, storage=storage)
    assert restored_e == equip_node.cid
    valid_unfold, unfold_errors = g.validate_integrity()
    assert valid_unfold, f"Graph invalid after unfold: {unfold_errors}"
    assert g.compute_merkle_root() == initial_merkle
    assert g.root_cid == g.root.cid
    assert g.get_node(restored_e) is not None
    assert g.get_node(lens_node.cid) is not None


def test_nested_subgraph_folding():
    """Verify nested folding (folding a subgraph inside another subgraph and unfolding in stages)."""
    from core.asg import fold_subgraph, unfold_subgraph

    g = QuantaGraph()
    root = QuantaNode(vector={"NSM_DO": 1}, anchor="wn:manage.v.01")
    sub_a = QuantaNode(vector={"TYPE_HUMAN": 1}, anchor="wn:supervisor.n.01")
    sub_b = QuantaNode(vector={"TYPE_HUMAN": 1}, anchor="wn:worker.n.01")
    sub_c = QuantaNode(vector={"TYPE_ARTIFACT": 1}, anchor="wn:tool.n.01")

    r_cid = g.add_node(root, set_as_root=True)
    a_cid = g.add_node(sub_a)
    b_cid = g.add_node(sub_b)
    c_cid = g.add_node(sub_c)

    g.add_edge(r_cid, "VAL_X1_AGENT", a_cid)
    g.add_edge(a_cid, "SUPERVISES", b_cid)
    g.add_edge(b_cid, "USES", c_cid)

    initial_merkle = g.compute_merkle_root()
    storage = {}

    # 1. Fold innermost: sub_b -> sub_c
    ptr_b, merkle_b = fold_subgraph(g, b_cid, storage=storage)
    assert g.validate_integrity()[0]

    # 2. Fold enclosing: sub_a -> ptr_b
    ptr_a, merkle_a = fold_subgraph(g, a_cid, storage=storage)
    assert g.validate_integrity()[0]
    assert len(g) == 2  # root + ptr_a

    # 3. Unfold enclosing sub_a
    restored_a = unfold_subgraph(ptr_a, storage, target_graph=g)
    assert restored_a == sub_a.cid
    assert g.validate_integrity()[0]
    assert g.get_node(ptr_b) is not None

    # 4. Unfold innermost sub_b
    restored_b = unfold_subgraph(ptr_b, storage, target_graph=g)
    assert restored_b == sub_b.cid
    assert g.validate_integrity()[0]
    assert g.compute_merkle_root() == initial_merkle


def test_merkle_tamper_detection_on_subtree():
    """Verify that tampering with any node attribute in a subtree alters the folded Merkle CID."""
    from core.asg import fold_subgraph

    def make_graph():
        g = QuantaGraph()
        event = QuantaNode(vector={"NSM_DO": 1, "TYPE_EVENT": 1}, anchor="wn:chase.v.01")
        dog = QuantaNode(vector={"TYPE_ANIMATE": 1, "WN_ANIMAL_FAUNA": 1}, anchor="wn:dog.n.01")
        cat = QuantaNode(vector={"TYPE_ANIMATE": 1, "WN_ANIMAL_FAUNA": 1}, anchor="wn:cat.n.01", literal={"color": "black"})
        e_cid = g.add_node(event, set_as_root=True)
        d_cid = g.add_node(dog)
        c_cid = g.add_node(cat)
        g.add_edge(e_cid, "VAL_X1_AGENT", d_cid)
        g.add_edge(d_cid, "CHASES", c_cid)
        return g, d_cid, c_cid

    g_orig, d_orig, c_orig = make_graph()
    storage_orig = {}
    ptr_orig, merkle_orig = fold_subgraph(g_orig, d_orig, storage=storage_orig)

    # 1. Tamper slot in subtree
    g_tamper1, d_tamper1, c_tamper1 = make_graph()
    g_tamper1.get_node(c_tamper1).set_slot("NSM_MOVE", 1)
    storage_tamper1 = {}
    ptr_tamper1, merkle_tamper1 = fold_subgraph(g_tamper1, d_tamper1, storage=storage_tamper1)
    assert merkle_orig != merkle_tamper1
    assert ptr_orig != ptr_tamper1

    # 2. Tamper anchor in subtree
    g_tamper2, d_tamper2, c_tamper2 = make_graph()
    g_tamper2.get_node(c_tamper2).anchor = "wn:kitten.n.01"
    g_tamper2.get_node(c_tamper2).invalidate_cache()
    storage_tamper2 = {}
    ptr_tamper2, merkle_tamper2 = fold_subgraph(g_tamper2, d_tamper2, storage=storage_tamper2)
    assert merkle_orig != merkle_tamper2
    assert ptr_orig != ptr_tamper2

    # 3. Tamper literal in subtree
    g_tamper3, d_tamper3, c_tamper3 = make_graph()
    g_tamper3.get_node(c_tamper3).literal = {"color": "white"}
    g_tamper3.get_node(c_tamper3).invalidate_cache()
    storage_tamper3 = {}
    ptr_tamper3, merkle_tamper3 = fold_subgraph(g_tamper3, d_tamper3, storage=storage_tamper3)
    assert merkle_orig != merkle_tamper3
    assert ptr_orig != ptr_tamper3


def test_unfold_tamper_rejection():
    """Verify that attempting to unfold a corrupted/tampered storage payload is rejected with ValueError."""
    from core.asg import fold_subgraph, unfold_subgraph

    g = QuantaGraph()
    root = QuantaNode(vector={"NSM_DO": 1}, anchor="wn:run.v.01")
    child = QuantaNode(vector={"TYPE_ANIMATE": 1}, anchor="wn:horse.n.01")
    r_cid = g.add_node(root, set_as_root=True)
    c_cid = g.add_node(child)
    g.add_edge(r_cid, "VAL_X1_AGENT", c_cid)

    storage = {}
    ptr_cid, sub_merkle = fold_subgraph(g, c_cid, storage=storage)

    # Tamper with storage payload by mutating a node vector in the serialized dict
    tampered_storage = {k: json.loads(json.dumps(v)) for k, v in storage.items()}
    for n_data in tampered_storage[sub_merkle]["nodes"].values():
        n_data["anchor"] = "wn:unicorn.n.01"

    with pytest.raises(ValueError, match="Merkle CID mismatch during unfold"):
        unfold_subgraph(ptr_cid, tampered_storage, target_graph=g)


def test_module_level_fold_unfold_api():
    """Verify standalone module-level functions fold_subgraph and unfold_subgraph."""
    from core.asg import fold_subgraph, unfold_subgraph

    g = QuantaGraph()
    event = QuantaNode(vector={"NSM_DO": 1}, anchor="wn:fly.v.01")
    bird = QuantaNode(vector={"TYPE_ANIMATE": 1, "WN_ANIMAL_FAUNA": 1}, anchor="wn:bird.n.01")
    e_cid = g.add_node(event, set_as_root=True)
    b_cid = g.add_node(bird)
    g.add_edge(e_cid, "VAL_X1_AGENT", b_cid)

    storage = {}
    ptr_cid, sub_merkle = fold_subgraph(g, b_cid, storage=storage)
    assert ptr_cid in g.nodes
    assert sub_merkle in storage

    # Standalone extraction directly from storage without modifying g
    isolated_subgraph = unfold_subgraph(sub_merkle, storage)
    assert isinstance(isolated_subgraph, QuantaGraph)
    assert len(isolated_subgraph) == 1
    assert isolated_subgraph.root_cid == b_cid

    # In-place restoration into g
    restored_cid = unfold_subgraph(ptr_cid, storage, target_graph=g)
    assert restored_cid == b_cid
    assert g.get_node(b_cid) is not None



