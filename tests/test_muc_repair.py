"""Unit and integration tests for QUANTA ClingoVerificationGate and MUCRepairManager (Phase 4).

Verifies:
1. Intentional ontological conflict (e.g. abstract concept as agent) produces an explicit, readable [REPAIR REQUEST] diagnostic.
2. Intentional Allen temporal contradiction produces an ordering conflict diagnostic.
3. Intentional causal contradiction produces a causal exclusivity diagnostic.
4. Mock repair simulation succeeds when the transducer fixes the violation on attempt 1.
5. Repair cap halts execution after exactly 2 failed attempts and returns informative error status.
6. Valid S-expressions pass without repair (attempts == 0, success == True).
7. Diagnostic string conversion and prompt formatting fidelity.
"""

from __future__ import annotations

import pytest

from core.asg import QuantaGraph, QuantaNode
from parser.asg_compiler import ASGCompiler
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from parser.sexpr_parser import parse_sexpr, to_sexpr
from parser.unsloth_transducer import MockUnslothTransducer
from verification.clingo_gate import (
    ClingoVerificationGate,
    MUCDiagnostic,
    MUCDiagnosticResult,
    MUCRepairManager,
    RepairResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gate() -> ClingoVerificationGate:
    return ClingoVerificationGate()


@pytest.fixture
def repair_manager(gate) -> MUCRepairManager:
    return MUCRepairManager(gate=gate, max_repair_attempts=2)


@pytest.fixture
def valid_sexpr() -> str:
    """Valid S-expression where human researcher pressurizes argon cylinder."""
    return """(graph :chunk-id "chunk_valid"
  (entity :id E1 :type PERSON :label "Dr. Marcus Vance" :surface "Marcus")
  (entity :id E2 :type OBJECT :label "argon cylinder" :surface "cylinder")
  (event :id Ev1 :pred pressurize :agent E1 :patient E2 :tense PAST :polarity TRUE :raw-text "Dr. Marcus Vance pressurized the argon cylinder.")
)"""


@pytest.fixture
def ontological_invalid_sexpr() -> str:
    """Invalid S-expression: Democracy (abstract concept) acts as physical agent in pressurize."""
    return """(graph :chunk-id "chunk_invalid_ontological"
  (entity :id E1 :type OBJECT :label "Democracy" :surface "Democracy" :props (:abstract TRUE))
  (entity :id E2 :type OBJECT :label "argon cylinder" :surface "cylinder")
  (event :id Ev1 :pred pressurize :agent E1 :patient E2 :tense PAST :polarity TRUE :raw-text "Democracy pressurized the argon cylinder.")
)"""


@pytest.fixture
def temporal_invalid_sexpr() -> str:
    """Invalid S-expression: Event has conflicting Allen temporal relations (BEFORE and DURING)."""
    return """(graph :chunk-id "chunk_invalid_temporal"
  (entity :id E1 :type PERSON :label "Dr. Marcus Vance" :surface "Marcus")
  (entity :id E2 :type OBJECT :label "cylinder" :surface "cylinder")
  (event :id Ev1 :pred pressurize :agent E1 :patient E2 :tense PAST :polarity TRUE)
  (event :id Ev2 :pred inspect :agent E1 :patient E2 :tense PAST :polarity TRUE)
  (relation :type TEMP_ALLEN_BEFORE :source Ev1 :target Ev2)
  (relation :type TEMP_ALLEN_DURING :source Ev1 :target Ev2)
)"""


@pytest.fixture
def causal_invalid_sexpr() -> str:
    """Invalid S-expression: Simultaneous direct causation and preventive blocking."""
    return """(graph :chunk-id "chunk_invalid_causal"
  (entity :id E1 :type PERSON :label "Dr. Marcus Vance" :surface "Marcus")
  (entity :id E2 :type OBJECT :label "valve" :surface "valve")
  (event :id Ev1 :pred open :agent E1 :patient E2 :tense PAST :polarity TRUE)
  (relation :type CAUSAL_MECHANISM_LINK :source Ev1 :target Ev1)
  (relation :type CAUSAL_PREVENTIVE_BLOCK :source Ev1 :target Ev1)
)"""


# ---------------------------------------------------------------------------
# 1. Ontological Conflict Diagnostic Tests
# ---------------------------------------------------------------------------

def test_ontological_conflict_diagnostic_graph(gate):
    """Acceptance Criteria 1:
    Intentional ontological conflict (e.g. abstract concept as agent) produces
    an explicit, readable [REPAIR REQUEST] diagnostic.
    """
    # Construct an invalid graph: Democracy (abstract concept) acts as agent in literal event
    graph = QuantaGraph()
    agent_node = QuantaNode(
        vector={"TYPE_ABSTRACT_CONCEPT": 1, "ROLE_AGENT_CAPABLE": 0},
        literal="Democracy",
        anchor="wn:democracy.n.01",
    )
    event_node = QuantaNode(
        vector={"TYPE_EVENT": 1, "MODALITY_LITERAL": 1, "NSM_DO": 1},
        literal="pressurize",
        anchor="cn:en:pressurize (v)",
    )
    a_cid = graph.add_node(agent_node)
    e_cid = graph.add_node(event_node, set_as_root=True)
    graph.add_edge(e_cid, "VAL_X1_AGENT", a_cid)

    # Validate through gate
    diag_res = gate.validate_graph(graph)
    assert not diag_res.is_valid
    assert diag_res.diagnostic is not None

    diag = diag_res.diagnostic
    assert diag.category == "ontological"
    assert "Democracy" in diag.summary or "abstract" in diag.summary
    assert "agent" in diag.summary.lower()

    # Verify [REPAIR REQUEST] prompt format
    prompt = diag.format_repair_request()
    assert "[REPAIR REQUEST]" in prompt
    assert "CONFLICT:" in prompt
    assert "Regenerate S-expression resolving the conflict." in prompt
    assert str(diag) == prompt


def test_ontological_conflict_via_compiled_sexpr(gate, ontological_invalid_sexpr):
    """Verify ontological conflict diagnosis when compiling directly from S-expression."""
    # Note: ASGCompiler automatically grounds "Democracy" to ConceptNet / WordNet
    # where it receives TYPE_ABSTRACT_CONCEPT=1 and not agent-capable.
    compiler = ASGCompiler(validator_gate=gate.validator_gate)
    extraction = parse_sexpr(ontological_invalid_sexpr)
    graph = compiler.compile(extraction, validate=False)

    diag_res = gate.validate_graph(graph, extraction_result=extraction)
    assert not diag_res.is_valid
    assert diag_res.diagnostic is not None
    diag = diag_res.diagnostic
    assert diag.category == "ontological"
    assert "Democracy" in diag.summary or "abstract" in diag.summary
    prompt = diag.format_repair_request()
    assert "[REPAIR REQUEST]" in prompt
    assert "CONFLICT:" in prompt


# ---------------------------------------------------------------------------
# 2. Allen Temporal Contradiction Diagnostic Tests
# ---------------------------------------------------------------------------

def test_temporal_contradiction_diagnostic(gate):
    """Acceptance Criteria 2:
    Intentional Allen temporal contradiction produces an ordering conflict diagnostic.
    """
    # Create graph with mutually exclusive Allen temporal relations on the same node
    graph = QuantaGraph()
    ev_node = QuantaNode(
        vector={
            "TYPE_EVENT": 1,
            "TEMP_ALLEN_BEFORE": 1,
            "TEMP_ALLEN_DURING": 1,
        },
        literal="observe",
    )
    graph.add_node(ev_node, set_as_root=True)

    diag_res = gate.validate_graph(graph)
    assert not diag_res.is_valid
    assert diag_res.diagnostic is not None

    diag = diag_res.diagnostic
    assert diag.category == "temporal"
    assert "temporal" in diag.summary.lower()
    prompt = diag.format_repair_request()
    assert "[REPAIR REQUEST]" in prompt
    assert "Allen temporal" in prompt or "temporal" in prompt
    assert "CONFLICT:" in prompt


# ---------------------------------------------------------------------------
# 3. Causal Contradiction Diagnostic Tests
# ---------------------------------------------------------------------------

def test_causal_contradiction_diagnostic(gate):
    """Verify intentional causal contradiction produces a causal exclusivity diagnostic."""
    graph = QuantaGraph()
    ev_node = QuantaNode(
        vector={
            "TYPE_EVENT": 1,
            "CAUSAL_DIRECT_MECHANISM": 1,
            "CAUSAL_PREVENTIVE_BLOCK": 1,
        },
        literal="regulate",
    )
    graph.add_node(ev_node, set_as_root=True)

    diag_res = gate.validate_graph(graph)
    assert not diag_res.is_valid
    assert diag_res.diagnostic is not None

    diag = diag_res.diagnostic
    assert diag.category == "causal"
    assert "causal" in diag.summary.lower()
    prompt = diag.format_repair_request()
    assert "[REPAIR REQUEST]" in prompt
    assert "causal" in prompt.lower()
    assert "CONFLICT:" in prompt


# ---------------------------------------------------------------------------
# 4. Mock Repair Simulation (Success on Attempt 1)
# ---------------------------------------------------------------------------

def test_mock_repair_simulation_success_on_attempt_1(repair_manager, ontological_invalid_sexpr, valid_sexpr):
    """Acceptance Criteria 3:
    Mock repair simulation succeeds when the transducer fixes the violation on attempt 1.
    """
    # Transducer returns invalid sexpr initially, then returns valid sexpr on repair request
    mock_transducer = MockUnslothTransducer()
    test_text = "The operator pressurized the argon cylinder."

    # Seed initial invalid response
    mock_transducer.register_fixture(test_text, ontological_invalid_sexpr)
    # Register repair fixture to be returned when repair_request is received
    mock_transducer.register_repair_fixture(test_text, valid_sexpr)

    result = repair_manager.repair_chunk(
        text=test_text,
        transducer=mock_transducer,
        initial_sexpr=ontological_invalid_sexpr,
    )

    assert result.success is True
    assert result.attempts == 1
    assert result.graph is not None
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].category == "ontological"
    assert result.error_message is None

    # Verify resulting graph is valid and has expected structure
    valid, errors = result.graph.validate_integrity()
    assert valid is True
    assert errors == []


# ---------------------------------------------------------------------------
# 5. Strict Repair Cap Enforcement (Halt after <= 2 attempts)
# ---------------------------------------------------------------------------

def test_repair_cap_halts_execution_after_exactly_2_failed_attempts(repair_manager, ontological_invalid_sexpr):
    """Acceptance Criteria 4:
    Repair cap halts execution after exactly 2 failed attempts and returns informative error status.
    """
    # Stubborn transducer that always returns an invalid S-expression
    class StubbornTransducer:
        def __init__(self, bad_sexpr: str):
            self.bad_sexpr = bad_sexpr
            self.call_count = 0
            self.prompts_received: list[str] = []

        def transduce_raw(self, text: str, **kwargs) -> str:
            self.call_count += 1
            repair_req = kwargs.get("repair_request")
            if repair_req:
                self.prompts_received.append(repair_req)
            return self.bad_sexpr

    stubborn = StubbornTransducer(ontological_invalid_sexpr)
    test_text = "Democracy pressurized the argon cylinder."

    result = repair_manager.repair_chunk(
        text=test_text,
        transducer=stubborn,
        initial_sexpr=ontological_invalid_sexpr,
    )

    # Must halt cleanly without crashing
    assert result.success is False
    # Max repair attempts cap is 2
    assert result.attempts == 2
    assert stubborn.call_count == 2
    assert len(result.diagnostics) == 3  # initial (attempt 0) + attempt 1 + attempt 2
    assert result.error_message is not None
    assert "cap exceeded" in result.error_message.lower() or "2 attempts" in result.error_message.lower()

    # All received prompts were valid [REPAIR REQUEST] blocks
    assert len(stubborn.prompts_received) == 2
    for p in stubborn.prompts_received:
        assert "[REPAIR REQUEST]" in p
        assert "CONFLICT:" in p


# ---------------------------------------------------------------------------
# 6. Valid S-Expression Passthrough (Zero Repair Attempts)
# ---------------------------------------------------------------------------

def test_valid_sexpr_passes_without_repair(repair_manager, valid_sexpr):
    """Verify already valid S-expression succeeds immediately with 0 repair attempts."""
    mock_transducer = MockUnslothTransducer()
    result = repair_manager.verify_and_repair_sexpr(
        sexpr=valid_sexpr,
        text="Dr. Marcus Vance pressurized the cylinder.",
        transducer=mock_transducer,
    )

    assert result.success is True
    assert result.attempts == 0
    assert len(result.diagnostics) == 0
    assert result.graph is not None
    assert result.error_message is None


# ---------------------------------------------------------------------------
# 7. Unpacking and Compatibility Helpers
# ---------------------------------------------------------------------------

def test_muc_diagnostic_result_unpacking(gate, valid_sexpr, ontological_invalid_sexpr):
    """Verify MUCDiagnosticResult supports tuple unpacking (is_valid, diagnostic)."""
    compiler = ASGCompiler(validator_gate=gate.validator_gate)

    # 1. Valid unpacking
    ext_valid = parse_sexpr(valid_sexpr)
    graph_valid = compiler.compile(ext_valid, validate=False)
    is_valid, diag = gate.validate_graph(graph_valid)
    assert is_valid is True
    assert diag is None

    # 2. Invalid unpacking
    ext_invalid = parse_sexpr(ontological_invalid_sexpr)
    graph_invalid = compiler.compile(ext_invalid, validate=False)
    is_valid, diag = gate.validate_graph(graph_invalid, extraction_result=ext_invalid)
    assert is_valid is False
    assert diag is not None
    assert isinstance(diag, MUCDiagnostic)
