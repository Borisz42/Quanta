"""Unit tests for Polymorphic 2-Bit Typing per Band across the QUANTA architecture."""

import pytest
import numpy as np

from core.types import (
    BandContract,
    QuaternaryValue,
    EpistemicValue,
    StructuralValue,
    RoutingValue,
    RegisterValue,
    QuantaVector,
)
from core.slots import (
    SlotBand,
    BAND_CONTRACTS,
    get_band_contract,
    get_slot_contract,
    get_slot_by_name,
    get_slot_by_index,
)
from core.asg import QuantaGraph, QuantaNode, fold_subgraph, unfold_subgraph
from solver.scasp_bridge import SCaspBridge


def test_band_contracts_mapping():
    """Verify that every band maps to its exact polymorphic contract."""
    assert get_band_contract(SlotBand.BAND_0_NSM_KINEMATICS) == BandContract.EPISTEMIC
    assert get_band_contract(SlotBand.BAND_1_VALENCIES_TOPOLOGY) == BandContract.STRUCTURAL
    assert get_band_contract(SlotBand.BAND_2_LOGIC_VARIABLES) == BandContract.REGISTER
    assert get_band_contract(SlotBand.BAND_3_ONTOLOGY_STRUCTURES) == BandContract.EPISTEMIC
    assert get_band_contract(SlotBand.BAND_4_AFFORDANCES_OPERATIONS) == BandContract.EPISTEMIC
    assert get_band_contract(SlotBand.BAND_5_TOM_PRAGMATICS) == BandContract.EPISTEMIC
    assert get_band_contract(SlotBand.BAND_6_PROOF_DEONTICS) == BandContract.EPISTEMIC
    assert get_band_contract(SlotBand.BAND_7_SPATIOTEMPORAL_CAUSAL) == BandContract.EPISTEMIC

    # Slot contract lookups
    assert get_slot_contract("NSM_KNOW") == BandContract.EPISTEMIC
    assert get_slot_contract("VAL_X1_AGENT") == BandContract.STRUCTURAL
    assert get_slot_contract("GRAPH_FUNCTION_DEF") == BandContract.STRUCTURAL
    assert get_slot_contract("QUANT_UNIVERSAL_FORALL") == BandContract.REGISTER
    assert get_slot_contract("CN_Q001_COMPUTING") == BandContract.EPISTEMIC


def test_polymorphic_enums_and_values():
    """Verify properties of StructuralValue, RegisterValue, and EpistemicValue."""
    # StructuralValue
    assert StructuralValue.INACTIVE == 0
    assert StructuralValue.ACTIVE_LOCAL == 1
    assert StructuralValue.ACTIVE_EXTERNAL == 2
    assert StructuralValue.ACTIVE_MERKLE == 3
    assert not StructuralValue.INACTIVE.is_active
    assert StructuralValue.ACTIVE_LOCAL.is_active
    assert StructuralValue.ACTIVE_EXTERNAL.is_active
    assert StructuralValue.ACTIVE_MERKLE.is_active
    assert RoutingValue == StructuralValue

    # RegisterValue
    assert RegisterValue.UNBOUND == 0
    assert RegisterValue.BOUND_LOCAL == 1
    assert RegisterValue.BOUND_EXTERNAL == 2
    assert RegisterValue.QUERY_TARGET == 3
    assert not RegisterValue.UNBOUND.is_bound
    assert RegisterValue.BOUND_LOCAL.is_bound
    assert RegisterValue.BOUND_EXTERNAL.is_bound
    assert RegisterValue.QUERY_TARGET.is_query

    # EpistemicValue
    assert EpistemicValue.IRRELEVANT == 0
    assert EpistemicValue.TRUE == 1
    assert EpistemicValue.FALSE == 2
    assert EpistemicValue.UNKNOWN == 3
    assert EpistemicValue == QuaternaryValue


def test_polymorphic_lattice_join_meet():
    """Verify that QuantaVector.join() and meet() use polymorphic tables per band."""
    v1 = QuantaVector.zeros(1024)
    v2 = QuantaVector.zeros(1024)

    # Band 0 (Epistemic): TRUE (1) join FALSE (2) -> UNKNOWN (3)
    v1["NSM_KNOW"] = EpistemicValue.TRUE       # 1
    v2["NSM_KNOW"] = EpistemicValue.FALSE      # 2

    # Band 1 (Structural): ACTIVE_LOCAL (1) join ACTIVE_EXTERNAL (2) -> ACTIVE_EXTERNAL (2)
    v1["VAL_X1_AGENT"] = StructuralValue.ACTIVE_LOCAL     # 1
    v2["VAL_X1_AGENT"] = StructuralValue.ACTIVE_EXTERNAL  # 2

    # Band 1 (Structural): ACTIVE_LOCAL (1) join ACTIVE_MERKLE (3) -> ACTIVE_MERKLE (3)
    v1["VAL_X2_PATIENT"] = StructuralValue.ACTIVE_LOCAL    # 1
    v2["VAL_X2_PATIENT"] = StructuralValue.ACTIVE_MERKLE   # 3

    # Band 2 (Register): BOUND_LOCAL (1) join QUERY_TARGET (3) -> QUERY_TARGET (3)
    v1["VAR_SLOT_X0"] = RegisterValue.BOUND_LOCAL          # 1
    v2["VAR_SLOT_X0"] = RegisterValue.QUERY_TARGET         # 3

    # Perform Join
    v_joined = v1.join(v2)

    # In Band 0 (Epistemic): 1 ? 2 = 3 (Belnap contradiction/uncertainty)
    assert v_joined["NSM_KNOW"] == EpistemicValue.UNKNOWN

    # In Band 1 (Structural): 1 ? 2 = 2 (External pointer priority, NOT Merkle 3!)
    assert v_joined.get_structural_slot("VAL_X1_AGENT") == StructuralValue.ACTIVE_EXTERNAL
    assert v_joined.get_structural_slot("VAL_X2_PATIENT") == StructuralValue.ACTIVE_MERKLE

    # In Band 2 (Register): 1 ? 3 = 3
    assert v_joined.get_register_slot("VAR_SLOT_X0") == RegisterValue.QUERY_TARGET

    # Perform Meet
    v_met = v1.meet(v2)
    # Band 0: 1 ? 2 = 0 (Belnap no shared information)
    assert v_met["NSM_KNOW"] == EpistemicValue.IRRELEVANT
    # Band 1: 1 ? 2 = 1 (Local meets External -> Local)
    assert v_met.get_structural_slot("VAL_X1_AGENT") == StructuralValue.ACTIVE_LOCAL


def test_typed_slot_accessors():
    """Verify typed slot accessors on QuantaVector and QuantaNode."""
    node = QuantaNode()
    node.set_structural_slot("VAL_X1_AGENT", StructuralValue.ACTIVE_EXTERNAL)
    node.set_register_slot("VAR_SLOT_X0", RegisterValue.BOUND_LOCAL)
    node.set_slot("NSM_THINK", EpistemicValue.TRUE)

    assert node.get_structural_slot("VAL_X1_AGENT") == StructuralValue.ACTIVE_EXTERNAL
    assert node.get_routing_type("VAL_X1_AGENT") == StructuralValue.ACTIVE_EXTERNAL
    assert node.get_register_slot("VAR_SLOT_X0") == RegisterValue.BOUND_LOCAL
    assert node.get_epistemic_slot("NSM_THINK") == EpistemicValue.TRUE
    assert node.get_slot("VAL_X1_AGENT") == 2
    assert node.get_slot("VAR_SLOT_X0") == 1
    assert node.get_slot("NSM_THINK") == 1


def test_semantic_similarity_masking():
    """Verify that semantic_similarity compares only Epistemic bands."""
    v1 = QuantaVector.zeros(1024)
    v2 = QuantaVector.zeros(1024)

    # Set identical epistemic slots in Band 0
    v1["NSM_THINK"] = 1
    v2["NSM_THINK"] = 1
    v1["NSM_DO"] = 1
    v2["NSM_DO"] = 1

    # Differ only in Band 1 structural routing (Local vs Merkle)
    v1["VAL_X1_AGENT"] = StructuralValue.ACTIVE_LOCAL   # 1
    v2["VAL_X1_AGENT"] = StructuralValue.ACTIVE_MERKLE  # 3

    # Total Hamming distance sees the structural difference
    assert v1.hamming_distance(v2) == 1
    assert v1.similarity(v2) < 1.0

    # Semantic similarity ignores structural/register differences
    assert v1.semantic_similarity(v2) == 1.0


def test_merkle_fold_routing_integration():
    """Verify that Merkle folding marks structural slots as ACTIVE_MERKLE."""
    graph = QuantaGraph()
    root = QuantaNode(literal="parent")
    root.set_slot("NSM_DO", 1)

    child1 = QuantaNode(literal="child_agent")
    child1.set_slot("TYPE_HUMAN", 1)

    child2 = QuantaNode(literal="child_sub")
    child2.set_slot("NSM_SEE", 1)

    graph.add_node(root, set_as_root=True)
    graph.add_edge(root, "VAL_X1_AGENT", child1)
    graph.add_edge(child1, "VAL_X2_PATIENT", child2)

    storage = {}
    ptr_cid, merkle_cid = fold_subgraph(graph, child1.cid, storage=storage)

    ptr_node = graph.get_node(ptr_cid)
    assert ptr_node is not None
    assert ptr_node.get_structural_slot("GRAPH_MERKLE_FOLD_POINT") == StructuralValue.ACTIVE_MERKLE
    assert ptr_node.get_structural_slot("GRAPH_EXT_REFERENCE") == StructuralValue.ACTIVE_MERKLE


def test_scasp_bridge_polymorphic_facts():
    """Verify that SCaspBridge generates differentiated facts for epistemic, structural, and register bands."""
    graph = QuantaGraph()
    node = QuantaNode(literal="test_node")
    node.set_slot("NSM_KNOW", EpistemicValue.TRUE)
    node.set_slot("VAL_X1_AGENT", StructuralValue.ACTIVE_EXTERNAL)
    node.set_slot("VAR_SLOT_X0", RegisterValue.BOUND_LOCAL)
    graph.add_node(node, set_as_root=True)

    bridge = SCaspBridge()
    facts = bridge.graph_to_prolog_facts(graph)

    assert f"epistemic_slot('{node.cid}', 'NSM_KNOW', true)." in facts
    assert f"structural_routing('{node.cid}', 'VAL_X1_AGENT', external)." in facts
    assert f"register_scope('{node.cid}', 'VAR_SLOT_X0', bound_local)." in facts
