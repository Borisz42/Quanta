"""End-to-End Integration and Regression Tests for QUANTA CognitivePipeline (Phase 5).

Verifies:
1. End-to-end execution of a single-chunk and multi-chunk narrative:
   text -> S-expression -> QuantaGraph ASG -> English NLG.
2. Unified pipeline process() method with automated chunking, transduction,
   stitching, compilation, Clingo ASP verification, and Merkle folding.
3. Multi-target realization methods:
   - to_english(graph)
   - to_fol(graph)
   - to_code(graph)
   - to_sexpr(graph)
4. S-expression export roundtrip fidelity.
5. Closed-loop MUC repair integration (transducer resolves violation on attempt 1).
6. Strict repair cap enforcement (halts cleanly after <= 2 attempts on stubborn invalid output).
7. Zero-attention graph query answering within < 10ms.
8. Bounded active canvas execution (len(active_canvas) <= capacity) and state reset.
"""

from __future__ import annotations

import time
import pytest

from core.asg import QuantaGraph, QuantaNode
from parser.asg_compiler import ASGCompilationError
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from parser.sexpr_parser import parse_sexpr, to_sexpr
from parser.transducer import (
    CANONICAL_ELEANOR_VANCE_FIXTURE,
    CANONICAL_ELEANOR_VANCE_TEXT,
    MockTransducer,
)
from parser.unsloth_transducer import MockUnslothTransducer
from pipeline.cognitive_pipeline import CognitivePipeline
from verification.clingo_gate import ClingoVerificationGate, MUCRepairManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_unsloth() -> MockUnslothTransducer:
    return MockUnslothTransducer()


@pytest.fixture
def pipeline_with_mock(mock_unsloth) -> CognitivePipeline:
    return CognitivePipeline(
        transducer=mock_unsloth,
        canvas_capacity=512,
    )


@pytest.fixture
def valid_pressurize_sexpr() -> str:
    return """(graph :chunk-id "chunk_valid_repair"
  (entity :id E1 :type PERSON :label "Dr. Marcus Vance" :surface "Marcus")
  (entity :id E2 :type OBJECT :label "argon cylinder" :surface "cylinder")
  (event :id Ev1 :pred pressurize :agent E1 :patient E2 :tense PAST :polarity TRUE :raw-text "Dr. Marcus Vance pressurized the argon cylinder.")
)"""


@pytest.fixture
def invalid_ontological_sexpr() -> str:
    return """(graph :chunk-id "chunk_invalid_ontological"
  (entity :id E1 :type OBJECT :label "Democracy" :surface "Democracy" :props (:abstract TRUE))
  (entity :id E2 :type OBJECT :label "argon cylinder" :surface "cylinder")
  (event :id Ev1 :pred pressurize :agent E1 :patient E2 :tense PAST :polarity TRUE :raw-text "Democracy pressurized the argon cylinder.")
)"""


# ---------------------------------------------------------------------------
# 1. Single-Chunk End-to-End Narrative Execution
# ---------------------------------------------------------------------------

def test_single_chunk_end_to_end_pipeline(pipeline_with_mock):
    """Verify single-chunk execution through process():
    text -> S-expression -> QuantaGraph ASG -> English NLG realization.
    """
    graph = pipeline_with_mock.process(CANONICAL_ELEANOR_VANCE_TEXT)

    assert isinstance(graph, QuantaGraph)
    assert graph.root_cid is not None
    assert graph.root is not None

    # 1. Verify exact node count: 5 entities + 6 events = 11 nodes total
    assert len(graph.nodes) == 11, f"Expected 11 total nodes, got {len(graph.nodes)}"
    entity_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 0]
    event_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 1]
    assert len(entity_nodes) == 5
    assert len(event_nodes) == 6

    # 2. Structural & cryptographic BLAKE3 integrity
    valid, errors = graph.validate_integrity()
    assert valid is True
    assert errors == []
    for cid, node in graph.nodes.items():
        assert len(cid) == 64
        assert cid == node.compute_cid()

    # 3. Formal Clingo ASP validation passed
    assert hasattr(graph, "validation")
    assert graph.validation.is_valid is True
    assert len(graph.validation.errors) == 0

    # 4. English NLG realization
    realized = pipeline_with_mock.to_english(graph)
    assert realized, "English realization is empty!"
    assert "Dr. Eleanor Vance" in realized
    assert "synthetic compound" in realized or "compound" in realized
    assert "cryogenic containment cell" in realized or "containment cell" in realized

    pipeline_with_mock.close()


# ---------------------------------------------------------------------------
# 2. Multi-Target Reverse Realizers (English, FOL, Code, S-Expression)
# ---------------------------------------------------------------------------

def test_all_export_methods(pipeline_with_mock):
    """Verify to_english, to_fol, to_code, and to_sexpr export methods."""
    graph = pipeline_with_mock.process(CANONICAL_ELEANOR_VANCE_TEXT)

    # 1. English
    english = pipeline_with_mock.to_english(graph)
    assert isinstance(english, str)
    assert len(english) > 20
    # Also verify realize() alias
    assert pipeline_with_mock.realize(graph) == english

    # 2. S-Expression
    sexpr = pipeline_with_mock.to_sexpr(graph)
    assert isinstance(sexpr, str)
    assert sexpr.strip().startswith("(graph")
    assert "(entity" in sexpr
    assert "(event" in sexpr

    # S-expression roundtrip fidelity
    parsed = parse_sexpr(sexpr)
    assert len(parsed.entities) == 5
    assert len(parsed.events) == 6

    # 3. FOL formula
    fol = pipeline_with_mock.to_fol(graph)
    assert isinstance(fol, str)
    assert len(fol) > 0

    # 4. Python code
    code = pipeline_with_mock.to_code(graph)
    assert isinstance(code, str)
    assert len(code) > 0

    pipeline_with_mock.close()


# ---------------------------------------------------------------------------
# 3. Multi-Chunk Narrative Stitching Integration
# ---------------------------------------------------------------------------

def test_multi_chunk_narrative_stitching():
    """Verify multi-chunk text processing with GraphStitcher global entity resolution."""
    # Chunk 1: Dr. Eleanor Vance isolates synthetic compound
    c1 = (
        "Dr. Eleanor Vance isolated a volatile synthetic compound inside the "
        "cryogenic containment cell at dawn."
    )
    # Chunk 2: Eleanor notes lattice expansion; supervisor doubts
    c2 = (
        "She immediately noted that this specimen exhibited anomalous lattice expansion. "
        "Her supervisor initially doubted the discovery."
    )
    # Chunk 3: Eleanor verifies hypothesis three hours later
    c3 = (
        "Eleanor verified the hypothesis three hours later within the same vessel. "
        "The resulting polymer retained its structural integrity."
    )

    full_narrative = f"{c1}\n\n***\n\n{c2}\n\n***\n\n{c3}"

    mock = MockUnslothTransducer()
    # Register fixtures for each individual chunk
    f1 = DiscourseExtractionResult(
        chunk_id="chk_1",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Dr. Eleanor Vance", category="PERSON", surface_aliases=["Dr. Vance"]),
            ExtractedEntity(id="E2", canonical_name="synthetic compound", category="SUBSTANCE", surface_aliases=["specimen"]),
            ExtractedEntity(id="E3", canonical_name="cryogenic containment cell", category="LOCATION", surface_aliases=["cell"]),
        ],
        events=[
            ExtractedEvent(id="Ev1", predicate="isolate", agent_id="E1", patient_id="E2", location_id="E3", tense="PAST", polarity=True),
        ],
    )
    f2 = DiscourseExtractionResult(
        chunk_id="chk_2",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Eleanor", category="PERSON", surface_aliases=["she"]),
            ExtractedEntity(id="E2", canonical_name="specimen", category="SUBSTANCE", surface_aliases=["compound"]),
            ExtractedEntity(id="E3", canonical_name="supervisor", category="PERSON", surface_aliases=["her supervisor"]),
        ],
        events=[
            ExtractedEvent(id="Ev1", predicate="note", agent_id="E1", patient_id="E2", tense="PAST", polarity=True),
            ExtractedEvent(id="Ev2", predicate="doubt", agent_id="E3", theme_id="E2", tense="PAST", polarity=True),
        ],
    )
    f3 = DiscourseExtractionResult(
        chunk_id="chk_3",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Dr. Vance", category="PERSON", surface_aliases=["Eleanor"]),
            ExtractedEntity(id="E2", canonical_name="polymer", category="SUBSTANCE", surface_aliases=["synthetic compound"]),
            ExtractedEntity(id="E3", canonical_name="containment cell", category="LOCATION", surface_aliases=["vessel"]),
        ],
        events=[
            ExtractedEvent(id="Ev1", predicate="verify", agent_id="E1", location_id="E3", temporal_anchor="three hours later", tense="PAST", polarity=True),
            ExtractedEvent(id="Ev2", predicate="retain", agent_id="E2", tense="PAST", polarity=True),
        ],
    )

    mock.register_fixture(c1, f1)
    mock.register_fixture(c2, f2)
    mock.register_fixture(c3, f3)

    pipeline = CognitivePipeline(transducer=mock)
    graph = pipeline.process(full_narrative)

    assert isinstance(graph, QuantaGraph)
    assert graph.validation.is_valid is True

    # Global entity unification:
    # E1: Dr. Eleanor Vance (Eleanor, Dr. Vance, she)
    # E2: synthetic compound (specimen, polymer)
    # E3: containment cell (vessel)
    # E4: supervisor
    # Total canonical entities = 4
    entity_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 0]
    assert len(entity_nodes) == 4

    # Events: 1 in c1 + 2 in c2 + 2 in c3 = 5 total events
    event_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 1]
    assert len(event_nodes) == 5

    # Inter-chunk relations synthesized
    assert hasattr(graph, "extraction_result")
    relations = graph.extraction_result.relations
    meets_rels = [r for r in relations if r.relation_type == "TEMP_ALLEN_MEETS"]
    before_rels = [r for r in relations if r.relation_type == "TEMP_ALLEN_BEFORE"]
    assert len(meets_rels) >= 1  # between chunk 1 and chunk 2
    assert len(before_rels) >= 1  # between chunk 2 and chunk 3 ("three hours later")

    pipeline.close()


# ---------------------------------------------------------------------------
# 4. Closed-Loop MUC Repair Integration
# ---------------------------------------------------------------------------

def test_closed_loop_muc_repair_success(invalid_ontological_sexpr, valid_pressurize_sexpr):
    """Verify closed-loop MUC repair is triggered and resolves conflict on attempt 1."""
    mock = MockUnslothTransducer()
    test_text = "The operator pressurized the argon cylinder."

    # Initial invalid output (Democracy as agent)
    mock.register_fixture(test_text, invalid_ontological_sexpr)
    # Corrected output on repair request
    mock.register_repair_fixture(test_text, valid_pressurize_sexpr)

    pipeline = CognitivePipeline(transducer=mock, max_repair_attempts=2)
    graph = pipeline.process(test_text)

    assert isinstance(graph, QuantaGraph)
    assert graph.validation.is_valid is True
    assert len(graph.validation.errors) == 0

    # Realization expresses corrected entity
    realized = pipeline.to_english(graph)
    assert "Dr. Marcus Vance" in realized or "Marcus" in realized

    pipeline.close()


def test_closed_loop_muc_repair_cap_halt(invalid_ontological_sexpr):
    """Verify closed-loop repair halts cleanly after <= 2 attempts on stubborn invalid output."""
    class StubbornTransducer:
        def __init__(self, bad_sexpr: str):
            self.bad_sexpr = bad_sexpr
            self.call_count = 0

        def transduce_raw(self, text: str, **kwargs) -> str:
            self.call_count += 1
            return self.bad_sexpr

    stubborn = StubbornTransducer(invalid_ontological_sexpr)
    pipeline = CognitivePipeline(transducer=stubborn, max_repair_attempts=2)

    with pytest.raises(ASGCompilationError) as exc_info:
        pipeline.process("Democracy pressurized the argon cylinder.")

    assert "repair cap exceeded" in str(exc_info.value).lower() or "clingo" in str(exc_info.value).lower()
    assert stubborn.call_count == 3

    pipeline.close()


# ---------------------------------------------------------------------------
# 5. Zero-Attention Query Answering & Virtual Memory Bounds
# ---------------------------------------------------------------------------

def test_zero_attention_query_answering(pipeline_with_mock):
    """Verify graph query answering resolves in < 10ms without hallucination."""
    graph = pipeline_with_mock.process(CANONICAL_ELEANOR_VANCE_TEXT)

    t0 = time.perf_counter()
    ans = pipeline_with_mock.answer_query("What did Eleanor Vance verify?", target_graph=graph)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert ans == "The hypothesis"
    assert latency_ms < 10.0, f"Query latency {latency_ms:.2f}ms exceeded 10ms limit"

    # Verify active canvas is strictly bounded
    assert len(pipeline_with_mock.active_canvas) <= pipeline_with_mock.active_canvas.capacity

    pipeline_with_mock.close()


# ---------------------------------------------------------------------------
# 6. Lifecycle & Reset Tests
# ---------------------------------------------------------------------------

def test_pipeline_reset_and_close(pipeline_with_mock):
    """Verify pipeline reset clears active working memory and close releases resources."""
    pipeline_with_mock.process(CANONICAL_ELEANOR_VANCE_TEXT)

    assert len(pipeline_with_mock.active_canvas) > 0
    assert len(pipeline_with_mock.entity_engine.manifest.all_active()) > 0

    pipeline_with_mock.reset()

    assert len(pipeline_with_mock.active_canvas) == 0
    assert len(pipeline_with_mock.entity_engine.manifest.all_active()) == 0

    pipeline_with_mock.close()
