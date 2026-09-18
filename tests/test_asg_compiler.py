"""Unit and integration tests for QUANTA Symbolic ASG Compiler (Phase 4).

Verifies:
4.1 ASGCompiler class initialization, API, and error handling.
4.2 Entity compilation: ExtractedEntity -> QuantaNode with ConceptNet grounding,
    Band 2 register assignment, and deterministic BLAKE3 CID computation.
4.3 Event compilation: ExtractedEvent -> predicate QuantaNode with Band 0 NSM primes,
    Band 1 valency edges to entity CIDs, and Band 5/6 epistemic/deontic slots.
4.4 Spatio-temporal & causal edge wiring: Allen interval relations and Pearl causal mechanisms.
4.5 Clingo ASP ValidatorGate integration and Minimal Unsatisfiable Core (MUC) extraction.
4.6 Gold standard Eleanor Vance benchmark: exactly 5 entity nodes, 6 event nodes,
    0 punctuation nodes, correct BLAKE3 CIDs, and 100% Clingo validation pass rate.
"""

import json
from pathlib import Path
import pytest

from core.asg import QuantaGraph, QuantaNode
from core.types import RegisterValue, StructuralValue
from parser.asg_compiler import ASGCompiler, ASGCompilationError
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from parser.transducer import CANONICAL_ELEANOR_VANCE_FIXTURE
from solver.validator_gate import ValidationResult, ValidatorGate


@pytest.fixture
def compiler() -> ASGCompiler:
    """Fixture providing a configured ASGCompiler instance."""
    return ASGCompiler()


# ---------------------------------------------------------------------------
# 4.6: Gold Standard Eleanor Vance ASG Compilation Tests
# ---------------------------------------------------------------------------

def test_eleanor_vance_gold_standard_compilation(compiler: ASGCompiler):
    """Verify compiling Eleanor Vance fixture yields exactly 5 entities, 6 events,

    0 punctuation nodes, valid BLAKE3 CIDs, and 100% Clingo validation pass rate.
    """
    graph = compiler.compile(CANONICAL_ELEANOR_VANCE_FIXTURE, validate=True)

    # 1. Assert exactly 5 entity nodes and 6 event nodes = 11 nodes total
    assert len(graph.nodes) == 11, f"Expected 11 total nodes, got {len(graph.nodes)}"

    entity_cids = []
    event_cids = []
    for cid, node in graph.nodes.items():
        # Check node type
        if node.get_slot("TYPE_EVENT") == 1:
            event_cids.append(cid)
        else:
            entity_cids.append(cid)

    assert len(entity_cids) == 5, f"Expected exactly 5 entity nodes, got {len(entity_cids)}"
    assert len(event_cids) == 6, f"Expected exactly 6 event nodes, got {len(event_cids)}"

    # 2. Assert zero punctuation or raw token nodes
    for cid, node in graph.nodes.items():
        assert not (node.anchor and node.anchor.startswith("punct:")), f"Found punctuation node: {node}"
        assert not (node.anchor and node.anchor.startswith("token:")), f"Found token node: {node}"

    # 3. Assert structural integrity and BLAKE3 CIDs
    struct_valid, struct_errors = graph.validate_integrity()
    assert struct_valid, f"Graph structural integrity failed: {struct_errors}"
    assert len(struct_errors) == 0

    for cid, node in graph.nodes.items():
        assert len(cid) == 64, f"CID should be a 64-char hexadecimal BLAKE3 string, got '{cid}'"
        assert cid == node.compute_cid(), f"Node CID mismatch: {cid} vs {node.compute_cid()}"

    # 4. Assert 100% Clingo ASP validation pass rate
    assert hasattr(graph, "validation")
    val_res: ValidationResult = graph.validation
    assert val_res.is_valid, f"Clingo validation failed with errors: {val_res.errors}"
    assert len(val_res.errors) == 0
    assert len(val_res.muc_slots) == 0


# ---------------------------------------------------------------------------
# 4.2: Entity Compilation & Register Scoping Tests
# ---------------------------------------------------------------------------

def test_entity_compilation_registers_and_slots(compiler: ASGCompiler):
    """Verify entity compilation sets Band 2 registers and category slots."""
    e1 = ExtractedEntity(
        id="E1",
        canonical_name="Dr. Eleanor Vance",
        category="PERSON",
        surface_aliases=["Eleanor"],
    )
    node = compiler.compile_entity(e1, register_index=0)

    # Band 2 register allocation
    assert node.get_register_slot("VAR_SLOT_X0") == RegisterValue.BOUND_LOCAL
    assert node.get_slot("GRAPH_VARIABLE_BIND") == 1
    assert node.get_slot("GRAPH_LEAF") == 1

    # Ontological slots for PERSON
    assert node.get_slot("TYPE_HUMAN") == 1
    assert node.get_slot("TYPE_ANIMATE") == 1
    assert node.get_slot("ROLE_AGENT_CAPABLE") == 1
    assert node.get_slot("ROLE_SENTIENT") == 1
    assert node.get_slot("TYPE_NATURAL_OBJECT") == 1
    assert node.get_slot("TYPE_ARTIFACT") == 0
    assert node.get_slot("TYPE_INANIMATE_PHYSICAL") == 0

    # Test SUBSTANCE entity
    e2 = ExtractedEntity(
        id="E2",
        canonical_name="synthetic compound",
        category="SUBSTANCE",
    )
    node2 = compiler.compile_entity(e2, register_index=1)
    assert node2.get_register_slot("VAR_SLOT_X1") == RegisterValue.BOUND_LOCAL
    assert node2.get_slot("CN_Q072_SUBSTANCE") == 1
    assert node2.get_slot("TYPE_INANIMATE_PHYSICAL") == 1
    assert node2.get_slot("ROLE_SENTIENT") == 0

    # Test LOCATION entity
    e3 = ExtractedEntity(
        id="E3",
        canonical_name="containment cell",
        category="LOCATION",
    )
    node3 = compiler.compile_entity(e3, register_index=2)
    assert node3.get_register_slot("VAR_SLOT_X2") == RegisterValue.BOUND_LOCAL
    assert node3.get_slot("TYPE_SPATIAL_REGION") == 1
    assert node3.get_slot("WN_LOCATION_PLACE") == 1


# ---------------------------------------------------------------------------
# 4.3: Event Compilation & Valency Edge Tests
# ---------------------------------------------------------------------------

def test_event_compilation_nsm_primes_and_valencies(compiler: ASGCompiler):
    """Verify event compilation sets Band 0 NSM primes and attaches thematic valency edges."""
    ev = ExtractedEvent(
        id="Ev1",
        predicate="isolate",
        agent_id="E1",
        patient_id="E2",
        location_id="E3",
        tense="PAST",
        polarity=True,
    )
    props = [
        ExtractedProposition(id="P1", claim_text="observation", epistemic_status="OBSERVATION", event_id="Ev1")
    ]
    node = compiler.compile_event(ev, entity_cids={"E1": "cid1", "E2": "cid2", "E3": "cid3"}, propositions=props)

    # Band 0 NSM primes
    assert node.get_slot("TYPE_EVENT") == 1
    assert node.get_slot("WN_ACT_ACTION") == 1
    assert node.get_slot("NSM_DO") == 1
    assert node.get_slot("NSM_TRUE") == 1

    # Band 1 Tense
    assert node.get_slot("LJB_PU_PAST_TENSE") == 1

    # Band 5 Epistemic observation from proposition
    assert node.get_slot("EPIST_DIRECT_OBSERVATION") == 1

    # Predicate anchor
    assert node.anchor is not None
    assert "isolate" in node.anchor


def test_event_cognitive_and_deontic_slots(compiler: ASGCompiler):
    """Verify cognitive, doubt, verification, and prohibition events get appropriate Band 5/6 slots."""
    ev_doubt = ExtractedEvent(id="Ev3", predicate="doubt", agent_id="E4", tense="PAST")
    props_doubt = [ExtractedProposition(id="P3", claim_text="doubted", epistemic_status="DOUBTED", event_id="Ev3")]
    node_doubt = compiler.compile_event(ev_doubt, {}, props_doubt)
    assert node_doubt.get_slot("NSM_THINK") == 1
    assert node_doubt.get_slot("NSM_MAYBE") == 3
    assert node_doubt.get_slot("TOM_FIRST_ORDER_BELIEF") == 1

    ev_ver = ExtractedEvent(id="Ev4", predicate="verify", agent_id="E1", tense="PAST")
    props_ver = [ExtractedProposition(id="P4", claim_text="verified", epistemic_status="FACT", event_id="Ev4")]
    node_ver = compiler.compile_event(ev_ver, {}, props_ver)
    assert node_ver.get_slot("NSM_THINK") == 1
    assert node_ver.get_slot("NSM_TRUE") == 1
    assert node_ver.get_slot("SOLVER_PROOF_VALIDATED") == 1

    ev_proh = ExtractedEvent(id="Ev6", predicate="prohibit", agent_id="E5", tense="PAST")
    props_proh = [ExtractedProposition(id="P5", claim_text="prohibited", epistemic_status="PROHIBITED", event_id="Ev6")]
    node_proh = compiler.compile_event(ev_proh, {}, props_proh)
    assert node_proh.get_slot("NSM_SAY") == 1
    assert node_proh.get_slot("EPIST_DEONTIC_PROHIBITION") == 1


# ---------------------------------------------------------------------------
# 4.4: Spatio-Temporal & Causal Edge Wiring Tests
# ---------------------------------------------------------------------------

def test_spatiotemporal_and_causal_relations_wiring(compiler: ASGCompiler):
    """Verify Allen intervals and Pearl causal links are properly wired into the graph."""
    graph = compiler.compile(CANONICAL_ELEANOR_VANCE_FIXTURE, validate=False)

    # Find event nodes
    nodes_by_literal: dict = {}
    for node in graph.nodes.values():
        if node.literal:
            nodes_by_literal[str(node.literal)] = node

    ev1_node = next(n for n in graph.nodes.values() if n.literal and "isolated a volatile" in str(n.literal))
    ev2_node = next(n for n in graph.nodes.values() if n.literal and "immediately noted" in str(n.literal))
    ev4_node = next(n for n in graph.nodes.values() if n.literal and "verified the hypothesis" in str(n.literal))
    ev5_node = next(n for n in graph.nodes.values() if n.literal and "retained its structural" in str(n.literal))
    ev6_node = next(n for n in graph.nodes.values() if n.literal and "prohibit all competing" in str(n.literal))

    # Ev1 -> Ev2 (TEMP_ALLEN_MEETS)
    assert "TEMP_ALLEN_MEETS" in ev1_node.edges
    assert ev2_node.cid in ev1_node.edges["TEMP_ALLEN_MEETS"]

    # Ev1 -> Ev4 (TEMP_ALLEN_BEFORE)
    assert "TEMP_ALLEN_BEFORE" in ev1_node.edges
    assert ev4_node.cid in ev1_node.edges["TEMP_ALLEN_BEFORE"]

    # Ev5 -> Ev5 self-relation: TEMP_ALLEN_DURING compiled as node slot
    assert ev5_node.get_slot("TEMP_ALLEN_DURING") == 1

    # Ev5 -> Ev6 (CAUSAL_MECHANISM_LINK)
    assert "CAUSAL_MECHANISM_LINK" in ev5_node.edges
    assert ev6_node.cid in ev5_node.edges["CAUSAL_MECHANISM_LINK"]
    assert ev5_node.get_slot("CAUSAL_DIRECT_MECHANISM") == 1


# ---------------------------------------------------------------------------
# 4.5 & 4.6.2: Input Polymorphism & Foreign-Key Integrity Tests
# ---------------------------------------------------------------------------

def test_input_polymorphism(compiler: ASGCompiler):
    """Verify compiler accepts DiscourseExtractionResult, JSON string, and dict."""
    fixture = CANONICAL_ELEANOR_VANCE_FIXTURE

    # 1. Direct object
    g1 = compiler.compile(fixture, validate=False)

    # 2. JSON string
    json_str = fixture.to_json()
    g2 = compiler.compile(json_str, validate=False)

    # 3. Python dict
    dict_data = fixture.to_dict()
    g3 = compiler.compile(dict_data, validate=False)

    assert len(g1.nodes) == len(g2.nodes) == len(g3.nodes) == 11
    assert g1.compute_merkle_root() == g2.compute_merkle_root() == g3.compute_merkle_root()


def test_dangling_foreign_key_detection(compiler: ASGCompiler):
    """Verify that dangling foreign keys in input JSON trigger ASGCompilationError."""
    bad_data = {
        "entities": [
            {"id": "E1", "canonical_name": "Alice", "category": "PERSON"}
        ],
        "events": [
            {"id": "Ev1", "predicate": "chase", "agent_id": "E999", "tense": "PAST"}
        ],
        "relations": [],
        "propositions": [],
    }

    with pytest.raises(ASGCompilationError, match="Foreign-key validation failed"):
        compiler.compile(bad_data)


def test_unsupported_input_type(compiler: ASGCompiler):
    """Verify compiler raises TypeError on unsupported input."""
    with pytest.raises(TypeError, match="Unsupported extraction input type"):
        compiler.compile(12345)  # type: ignore


# ---------------------------------------------------------------------------
# 4.5: MUC Detection on Ontological Contradiction
# ---------------------------------------------------------------------------

def test_muc_detection_on_ontological_contradiction(compiler: ASGCompiler):
    """Verify that an intentional ontological contradiction triggers Clingo validation error with MUC."""
    # Construct a valid extraction then corrupt a node's slots
    data = {
        "entities": [
            {"id": "E1", "canonical_name": "Justice", "category": "OBJECT"}
        ],
        "events": [
            {"id": "Ev1", "predicate": "bite", "agent_id": "E1", "tense": "PAST"}
        ],
        "relations": [],
        "propositions": [],
    }

    # If compiled with default entity compiler, E1 is OBJECT and has GRAPH_VARIABLE_BIND=1.
    # To trigger a pure ontological clash in Clingo (Rule 1: abstract concept + agent capable):
    gate = ValidatorGate()
    bad_node = QuantaNode(literal="abstract agent")
    bad_node.set_slot("TYPE_ABSTRACT_CONCEPT", 1)
    bad_node.set_slot("ROLE_AGENT_CAPABLE", 1)
    bad_node.set_slot("TYPE_ANIMATE", 0)
    bad_node.set_slot("TYPE_HUMAN", 0)

    val_res = gate.validate_node(bad_node)
    assert not val_res.is_valid
    assert len(val_res.errors) > 0
    assert len(val_res.muc_slots) > 0


# ---------------------------------------------------------------------------
# Graph Serialization Roundtrip
# ---------------------------------------------------------------------------

def test_compiled_graph_serialization_roundtrip(compiler: ASGCompiler):
    """Verify compiled QuantaGraph serializes to and deserializes from JSON without loss."""
    graph = compiler.compile(CANONICAL_ELEANOR_VANCE_FIXTURE, validate=False)
    original_merkle = graph.compute_merkle_root()

    json_repr = graph.to_json()
    restored = QuantaGraph.from_json(json_repr)

    assert len(restored.nodes) == 11
    assert restored.compute_merkle_root() == original_merkle
    assert restored.root_cid == graph.root_cid

    struct_valid, struct_errors = restored.validate_integrity()
    assert struct_valid
    assert len(struct_errors) == 0


# ---------------------------------------------------------------------------
# S-Expression AST and String Direct Compilation Tests
# ---------------------------------------------------------------------------

def test_compile_from_sexpr_ast_and_string(compiler: ASGCompiler):
    """Verify compiler directly compiles S-expression strings and SExprList ASTs."""
    from parser.sexpr_parser import SExprLexer, SExprParser, to_sexpr

    sexpr_str = to_sexpr(CANONICAL_ELEANOR_VANCE_FIXTURE)

    # 1. Compile directly from S-expression string
    g_str = compiler.compile(sexpr_str, validate=True)
    assert isinstance(g_str, QuantaGraph)
    assert len(g_str.nodes) == 11
    assert g_str.validation.is_valid is True
    assert g_str.merkle_root is not None
    assert len(g_str.merkle_root) == 64

    # 2. Compile directly from parsed SExprList AST
    tokens = SExprLexer(sexpr_str).tokenize()
    ast = SExprParser(tokens).parse()
    g_ast = compiler.compile(ast, validate=True)
    assert isinstance(g_ast, QuantaGraph)
    assert len(g_ast.nodes) == 11
    assert g_ast.validation.is_valid is True
    assert g_ast.merkle_root == g_str.merkle_root


def test_direct_ast_clause_compilation(compiler: ASGCompiler):
    """Verify compile_entity and compile_event compile S-expression AST clauses directly."""
    from parser.sexpr_parser import SExprLexer, SExprParser

    # Entity: HUMAN
    tokens1 = SExprLexer('(entity :id e1 :type HUMAN :label "Eleanor Vance" :surface "Dr. Eleanor Vance")').tokenize()
    ast_ent1 = SExprParser(tokens1).parse()
    node1 = compiler.compile_entity(ast_ent1, register_index=0)
    assert node1.get_slot("TYPE_HUMAN") == 1
    assert node1.get_slot("ROLE_AGENT_CAPABLE") == 1
    assert node1.get_register_slot("VAR_SLOT_X0") == RegisterValue.BOUND_LOCAL

    # Entity: TOPIC
    tokens2 = SExprLexer('(entity :id e2 :type TOPIC :label "quantum_mechanics" :surface "quantum mechanics")').tokenize()
    ast_ent2 = SExprParser(tokens2).parse()
    node2 = compiler.compile_entity(ast_ent2, register_index=1)
    assert node2.get_slot("TYPE_ABSTRACT_CONCEPT") == 1

    # Entity: UNIVERSITY
    tokens3 = SExprLexer('(entity :id e3 :type UNIVERSITY :label "Columbia" :surface "Columbia University")').tokenize()
    ast_ent3 = SExprParser(tokens3).parse()
    node3 = compiler.compile_entity(ast_ent3, register_index=2)
    assert node3.get_slot("TYPE_ORGANIZATION") == 1
    assert node3.get_slot("ROLE_AGENT_CAPABLE") == 1

    # Event: PROFESSOR_OF
    tokens4 = SExprLexer('(event :id ev1 :pred PROFESSOR_OF :agent e1 :theme e2 :location e3 :polarity TRUE :tense PRESENT)').tokenize()
    ast_ev = SExprParser(tokens4).parse()
    node_ev = compiler.compile_event(ast_ev)
    assert node_ev.get_slot("TYPE_EVENT") == 1
    assert node_ev.get_slot("LJB_CA_PRESENT_TENSE") == 1
    assert len(node_ev.cid) == 64


def test_merkle_root_persistence(compiler: ASGCompiler):
    """Verify compiler stores valid merkle_root on graph."""
    graph = compiler.compile(CANONICAL_ELEANOR_VANCE_FIXTURE, validate=True)
    assert hasattr(graph, "merkle_root")
    assert graph.merkle_root == graph.compute_merkle_root()
    assert len(graph.merkle_root) == 64
