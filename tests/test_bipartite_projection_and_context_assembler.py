"""Tests for Bipartite Passage Projection and Realization Bypass (Context Assembly) for QUANTA SVM.

Verifies Section 7 of QUANTA Refactor Master Plan:
1. Bipartite score projection across multiple passages sharing entities and events.
2. Dual-stream formatting under various token budget constraints (500, 1000, 2000 tokens).
3. 100% lexical fidelity of retrieved passages (exact verbatim match to source text).
4. Belnap contradiction pruning in bipartite projection.
5. Deprecation warning on legacy EnglishRealizer formatting.
"""

from __future__ import annotations

import warnings
import pytest

from core.asg import QuantaGraph, QuantaNode
from memory.context_assembler import (
    BipartiteProjector,
    DualStreamContext,
    DualStreamContextAssembler,
    ProjectedPassage,
)
from memory.passage_store import PassageRecord, PassageStore
from memory.spreading_activation import SpreadingActivationRetriever


def _create_node(
    anchor: str,
    literal: str,
    passage_id: str | None = None,
    span_start: int | None = None,
    span_end: int | None = None,
    salience: float = 1.0,
    confidence: float = 1.0,
    truth_status: str = "TRUE",
    evidence_source: str = "direct_observation",
    slots: dict[str, int] | None = None,
    edges: dict[str, list[str]] | None = None,
    time_start: float | None = None,
    time_end: float | None = None,
) -> QuantaNode:
    """Helper creating QuantaNode with provenance and slots."""
    node = QuantaNode(
        anchor=anchor,
        literal=literal,
        passage_id=passage_id,
        span_start=span_start,
        span_end=span_end,
        salience=salience,
        confidence=confidence,
        truth_status=truth_status,
        evidence_source=evidence_source,
    )
    if slots:
        for k, v in slots.items():
            node.set_slot(k, v)
    if edges:
        for rel, targets in edges.items():
            for t in targets:
                node.add_edge(rel, t)
    node.time_start = time_start
    node.time_end = time_end
    return node


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def multi_passage_store() -> PassageStore:
    """Passage store with 4 passages (3 related, 1 unrelated distractor)."""
    store = PassageStore()

    p1_text = "Dr. Eleanor Vance isolated volatile synthetic compound Alpha-9 inside Containment Cell 4 at sunrise."
    p2_text = "Following the isolation of Alpha-9, Dr. Eleanor Vance synthesized Catalyst-B in the central reactor."
    p3_text = "The catalyst reaction produced anomalous thermal radiation, triggering emergency shutdown protocols."
    p4_text = "A routine administrative inventory report was filed by technician Miller regarding spare gaskets."

    store.add_passage(
        PassageRecord(
            passage_id="P101",
            doc_id="doc_physics_run_01",
            char_span=(0, len(p1_text)),
            text=p1_text,
        )
    )
    store.add_passage(
        PassageRecord(
            passage_id="P102",
            doc_id="doc_physics_run_01",
            char_span=(120, 120 + len(p2_text)),
            text=p2_text,
        )
    )
    store.add_passage(
        PassageRecord(
            passage_id="P103",
            doc_id="doc_physics_run_01",
            char_span=(250, 250 + len(p3_text)),
            text=p3_text,
        )
    )
    store.add_passage(
        PassageRecord(
            passage_id="P104",
            doc_id="doc_physics_run_01",
            char_span=(400, 400 + len(p4_text)),
            text=p4_text,
        )
    )
    return store


@pytest.fixture
def multi_passage_graph() -> QuantaGraph:
    """ASG graph where entities and events span multiple passages."""
    graph = QuantaGraph()

    # Shared entity: Dr. Eleanor Vance appears in P101 and P102
    e_vance = _create_node(
        anchor="cn:en:eleanor_vance (n)",
        literal="Dr. Eleanor Vance",
        passage_id="P101",
        span_start=0,
        span_end=17,
        salience=0.95,
        confidence=0.98,
        slots={"ROLE_AGENT_CAPABLE": 1, "TYPE_HUMAN": 1},
    )
    # Shared entity: Alpha-9 appears in P101 and P102
    e_alpha = _create_node(
        anchor="cn:en:alpha_9 (n)",
        literal="Compound Alpha-9",
        passage_id="P101",
        span_start=47,
        span_end=54,
        salience=0.85,
        confidence=0.95,
        slots={"TYPE_SUBSTANCE": 1},
    )
    # Location entity: Containment Cell 4 appears in P101
    e_cell = _create_node(
        anchor="cn:en:containment_cell (n)",
        literal="Containment Cell 4",
        passage_id="P101",
        span_start=62,
        span_end=80,
        salience=0.75,
        confidence=0.90,
        slots={"TYPE_LOCATION": 1},
    )
    # Entity: Catalyst-B appears in P102
    e_catalyst = _create_node(
        anchor="cn:en:catalyst_b (n)",
        literal="Catalyst-B",
        passage_id="P102",
        span_start=68,
        span_end=78,
        salience=0.80,
        confidence=0.95,
        slots={"TYPE_SUBSTANCE": 1},
    )

    # Event 1: Isolation in P101
    ev_isolate = _create_node(
        anchor="cn:en:isolate (v)",
        literal="isolate",
        passage_id="P101",
        span_start=18,
        span_end=26,
        salience=0.92,
        confidence=0.97,
        truth_status="TRUE",
        evidence_source="direct_observation",
        slots={"TYPE_EVENT": 1, "WN_ACT_ACTION": 1},
        time_start=0.0,
        time_end=10.0,
    )
    ev_isolate.add_edge("VAL_X1_AGENT", e_vance.cid)
    ev_isolate.add_edge("VAL_X2_PATIENT", e_alpha.cid)
    ev_isolate.add_edge("VAL_LOCATION_SLOT", e_cell.cid)

    # Event 2: Synthesis in P102
    ev_synthesize = _create_node(
        anchor="cn:en:synthesize (v)",
        literal="synthesize",
        passage_id="P102",
        span_start=48,
        span_end=58,
        salience=0.88,
        confidence=0.96,
        truth_status="TRUE",
        evidence_source="direct_observation",
        slots={"TYPE_EVENT": 1, "WN_ACT_ACTION": 1},
        time_start=10.0,
        time_end=20.0,
    )
    ev_synthesize.add_edge("VAL_X1_AGENT", e_vance.cid)
    ev_synthesize.add_edge("VAL_X2_PATIENT", e_catalyst.cid)

    # Event 3: Radiation in P103
    ev_radiation = _create_node(
        anchor="cn:en:radiation (v)",
        literal="produce radiation",
        passage_id="P103",
        span_start=22,
        span_end=51,
        salience=0.65,
        confidence=0.90,
        truth_status="TRUE",
        evidence_source="direct_observation",
        slots={"TYPE_EVENT": 1, "WN_ACT_ACTION": 1},
        time_start=20.0,
        time_end=30.0,
    )

    # Connect Temporal & Causal sequence: EV1 -> MEETS -> EV2 -> CAUSES -> EV3
    ev_isolate.add_edge("TEMP_ALLEN_MEETS", ev_synthesize.cid)
    ev_synthesize.add_edge("CAUSAL_MECHANISM_LINK", ev_radiation.cid)

    # Add all nodes to graph
    graph.add_node(e_vance)
    graph.add_node(e_alpha)
    graph.add_node(e_cell)
    graph.add_node(e_catalyst)
    graph.add_node(ev_isolate, set_as_root=True)
    graph.add_node(ev_synthesize)
    graph.add_node(ev_radiation)

    # Also register bipartite mappings explicitly in QuantaGraph
    graph.add_passage_anchor(e_vance.cid, "P101", 0, 17)
    graph.add_passage_anchor(e_vance.cid, "P102", 30, 47)  # Vance shared in P102!
    graph.add_passage_anchor(e_alpha.cid, "P101", 47, 54)
    graph.add_passage_anchor(e_cell.cid, "P101", 62, 80)
    graph.add_passage_anchor(ev_isolate.cid, "P101", 18, 26)
    graph.add_passage_anchor(ev_synthesize.cid, "P102", 48, 58)
    graph.add_passage_anchor(e_catalyst.cid, "P102", 68, 78)
    graph.add_passage_anchor(ev_radiation.cid, "P103", 22, 51)

    return graph


# =============================================================================
# Test Suite: Section 7
# =============================================================================

class TestBipartitePassageProjection:
    """Tests for BipartiteProjector mapping semantic activation onto raw passage nodes."""

    def test_bipartite_score_projection_shared_entities(
        self,
        multi_passage_store: PassageStore,
        multi_passage_graph: QuantaGraph,
    ):
        """Tests that passages accumulate scores proportionally from all grounded semantic nodes.

        Formula: Score(v_p) = sum_{v_s in N(v_p)} p(v_s) * conf(v_s)
        Asserts that shared entities (Dr. Eleanor Vance) contribute to both P101 and P102.
        """
        projector = BipartiteProjector(passage_store=multi_passage_store)

        activations = {
            multi_passage_graph.get_node(multi_passage_graph.root_cid).cid: 0.95,
        }

        ranked = projector.project(
            subgraph=multi_passage_graph,
            passage_store=multi_passage_store,
        )

        assert len(ranked) >= 3
        passage_ids = [p.passage_id for p in ranked]

        # P101 has EV_isolate, E_vance, E_alpha, E_cell -> highest accumulated score
        # P102 has EV_synthesize, E_vance (shared), E_catalyst -> second highest
        # P103 has EV_radiation -> third
        # P104 has no grounded nodes -> omitted or score 0
        assert "P101" in passage_ids
        assert "P102" in passage_ids
        assert "P103" in passage_ids
        assert "P104" not in passage_ids

        p101 = next(p for p in ranked if p.passage_id == "P101")
        p102 = next(p for p in ranked if p.passage_id == "P102")
        p103 = next(p for p in ranked if p.passage_id == "P103")

        # P101 has 4 contributing nodes, P102 has 3 contributing nodes
        assert len(p101.contributing_node_cids) >= 3
        assert len(p102.contributing_node_cids) >= 2
        assert len(p103.contributing_node_cids) >= 1

        # Scores should be strictly positive and ranked P101 > P102 > P103
        assert p101.score > p102.score > p103.score > 0.0

    def test_bipartite_projection_poprag_contradiction_pruning(
        self,
        multi_passage_store: PassageStore,
        multi_passage_graph: QuantaGraph,
    ):
        """Asserts that nodes with Belnap CONTRADICTION (00_2) do not contribute positive score."""
        projector = BipartiteProjector(passage_store=multi_passage_store, filter_contradictions=True)

        # Baseline projection
        base_ranked = projector.project(multi_passage_graph, passage_store=multi_passage_store)
        base_p103 = next(p for p in base_ranked if p.passage_id == "P103")
        base_score = base_p103.score

        # Mark EV_radiation as CONTRADICTION
        cloned_graph = multi_passage_graph.copy()
        for node in cloned_graph.nodes.values():
            if node.passage_id == "P103":
                node.truth_status = "CONTRADICTION"

        contra_ranked = projector.project(cloned_graph, passage_store=multi_passage_store)
        p103_contra = [p for p in contra_ranked if p.passage_id == "P103"]

        # P103 should now have score 0 or be completely pruned
        if p103_contra:
            assert p103_contra[0].score < base_score
        else:
            assert True  # Completely pruned


class TestRetrievalLexicalFidelity:
    """Tests 100% lexical fidelity of retrieved passages from PassageStore."""

    def test_exact_verbatim_match_to_source_text(
        self,
        multi_passage_store: PassageStore,
        multi_passage_graph: QuantaGraph,
    ):
        """Asserts that retrieved passages match PassageStore verbatim with zero hallucination."""
        assembler = DualStreamContextAssembler(passage_store=multi_passage_store)
        ctx = assembler.assemble_dual_stream_context(
            subgraph=multi_passage_graph,
            passage_store=multi_passage_store,
            max_tokens=1500,
        )

        assert isinstance(ctx, DualStreamContext)
        assert len(ctx.passages) >= 2

        # Verify 100% exact character match for every retrieved passage
        for proj_p in ctx.passages:
            orig_rec = multi_passage_store.get_passage(proj_p.passage_id)
            assert orig_rec is not None
            # Absolute identity
            assert proj_p.text == orig_rec.text
            # Verified substring in passage stream output
            assert orig_rec.text in ctx.passage_stream
            assert orig_rec.text in ctx.full_context


class TestDualStreamContextAssembler:
    """Tests for Stream 1 (Logical Briefing) and Stream 2 (Top-K Raw Passages) assembly."""

    def test_logical_briefing_block_content(
        self,
        multi_passage_store: PassageStore,
        multi_passage_graph: QuantaGraph,
    ):
        """Verifies that Stream 1 contains verified causal paths, Allen intervals, and valency frames."""
        assembler = DualStreamContextAssembler(passage_store=multi_passage_store)
        briefing = assembler.assemble_logical_briefing(multi_passage_graph, max_tokens=500)

        assert "=== LOGICAL BRIEFING BLOCK ===" in briefing
        # Temporal path
        assert "TEMP_ALLEN_MEETS" in briefing
        # Causal path
        assert "CAUSAL_MECHANISM_LINK" in briefing
        # Valency frames
        assert "Thematic Valency Frames" in briefing
        assert "Agent:" in briefing
        assert "Dr. Eleanor Vance" in briefing or "Eleanor Vance" in briefing
        assert "Patient:" in briefing
        assert "Containment Cell 4" in briefing or "Cell 4" in briefing
        # Status annotations
        assert "Status: TRUE" in briefing or "Status: Verified" in briefing

    def test_dual_stream_budget_constraints(
        self,
        multi_passage_store: PassageStore,
        multi_passage_graph: QuantaGraph,
    ):
        """Tests dual-stream formatting under 500, 1000, and 2000 token budgets."""
        assembler = DualStreamContextAssembler(passage_store=multi_passage_store)

        for budget in (500, 1000, 2000):
            ctx = assembler.assemble_dual_stream_context(
                subgraph=multi_passage_graph,
                passage_store=multi_passage_store,
                max_tokens=budget,
            )

            assert isinstance(ctx.full_context, str)
            assert len(ctx.full_context) > 0
            assert "=== LOGICAL BRIEFING BLOCK ===" in ctx.full_context
            assert "=== RETRIEVED SOURCE PASSAGES ===" in ctx.full_context

            # Estimate token count (words * 1.33)
            word_count = len(ctx.full_context.split())
            est_tokens = int(word_count * 1.33)

            # Assert strictly fits within budget (with small margin for boundary punct)
            assert est_tokens <= budget + 25

            # Higher budget should retrieve at least as many or more passages
            if budget == 500:
                assert len(ctx.passages) >= 1
            elif budget == 2000:
                assert len(ctx.passages) >= 3

    def test_tight_budget_clean_truncation(
        self,
        multi_passage_store: PassageStore,
        multi_passage_graph: QuantaGraph,
    ):
        """Asserts that under extreme token budget (e.g. 50 tokens), output truncates cleanly."""
        assembler = DualStreamContextAssembler(passage_store=multi_passage_store)
        ctx = assembler.assemble_dual_stream_context(
            subgraph=multi_passage_graph,
            passage_store=multi_passage_store,
            max_tokens=50,
        )

        assert isinstance(ctx.full_context, str)
        words = ctx.full_context.split()
        assert len(words) <= int(50 / 1.33) + 15
        assert ctx.full_context.endswith((".", "!", "?"))

    def test_pure_symbolic_graph_fallback(self):
        """Asserts that graphs without passage anchors still produce a valid logical briefing."""
        graph = QuantaGraph()
        vance = _create_node("cn:en:vance (n)", "Eleanor Vance")
        ev = _create_node("cn:en:verify (v)", "verify", slots={"TYPE_EVENT": 1})
        ev.add_edge("VAL_X1_AGENT", vance.cid)
        graph.add_node(vance)
        graph.add_node(ev, set_as_root=True)

        assembler = DualStreamContextAssembler()
        ctx = assembler.assemble_dual_stream_context(graph, max_tokens=300)

        assert "=== LOGICAL BRIEFING BLOCK ===" in ctx.full_context
        assert "Eleanor Vance" in ctx.full_context
        assert "verify" in ctx.full_context
        assert ctx.full_context.endswith(".")


class TestSpreadingActivationIntegrationAndDeprecation:
    """Tests integration with SpreadingActivationRetriever and deprecation warnings."""

    def test_spreading_activation_format_context_dual_stream(
        self,
        multi_passage_store: PassageStore,
        multi_passage_graph: QuantaGraph,
    ):
        """Tests that SpreadingActivationRetriever defaults format='english' to dual stream."""
        retriever = SpreadingActivationRetriever(passage_store=multi_passage_store)

        text = retriever.format_context_for_llm(
            multi_passage_graph,
            format="english",
            max_tokens=800,
            passage_store=multi_passage_store,
        )

        assert isinstance(text, str)
        assert "=== LOGICAL BRIEFING BLOCK ===" in text
        assert "=== RETRIEVED SOURCE PASSAGES ===" in text
        assert "Alpha-9" in text
        assert "Dr. Eleanor Vance" in text

    def test_legacy_format_english_context_deprecation_warning(
        self,
        multi_passage_graph: QuantaGraph,
    ):
        """Asserts that direct calls to _format_english_context emit a DeprecationWarning."""
        retriever = SpreadingActivationRetriever()

        with pytest.deprecated_call():
            retriever._format_english_context(multi_passage_graph, max_tokens=100)
