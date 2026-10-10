"""Unit tests for Multi-Scale HippoRAG Personalized PageRank (PPR) and PoP-RAG gating (§Phase 4).

Covers:
- PassageStore schema migration on legacy SQLite databases
- PoPRAGGating calibrated inter_scale_weight and coarse_node_weight modulation
- Strict Belnap 00_2 CONTRADICTION preservation
- Pass-through PPR equivalence to baseline
- Multi-scale associative bridge spreading activation
"""

import sqlite3
import tempfile
from pathlib import Path
import pytest
import numpy as np

from core.asg import QuantaGraph, QuantaNode
from core.binary_node import BelnapValue
from core.types import QuantaVector
from memory.passage_store import (
    PassageRecord,
    PassageStore,
    INGESTION_STATUS_COARSE_ONLY,
    INGESTION_STATUS_PENDING,
    INGESTION_STATUS_FULL,
)
from memory.poprag_gating import PoPRAGGating
from memory.hipporag_ppr import HippoRAGRetriever


def test_passage_store_migration_on_legacy_db():
    """Verifies that PassageStore successfully migrates legacy SQLite tables without data loss."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "legacy_passages.db"

        # 1. Manually initialize a legacy schema (pre-Phase 4)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE passages (
                passage_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                span_start INTEGER NOT NULL,
                span_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)",
            ("leg_01", "doc_chem", 0, 45, "Entropy increases in isolated systems.", 1600000000.0),
        )
        conn.execute(
            "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)",
            ("leg_02", "doc_chem", 46, 95, "Second law of thermodynamics governs spontaneity.", 1600000001.0),
        )
        conn.commit()
        conn.close()

        # 2. Open with modern PassageStore which must detect and migrate schema
        store = PassageStore(db_path=db_path)
        assert len(store) == 2

        # 3. Check that legacy rows were loaded with modern defaults
        p1 = store.get_passage("leg_01")
        assert p1 is not None
        assert p1.text == "Entropy increases in isolated systems."
        assert p1.granularity == "MICRO"
        assert p1.parent_macro_id is None
        assert p1.concept_codes == ()

        # 4. Check that new table passage_ingestion_state was created
        store.set_ingestion_state("leg_01", INGESTION_STATUS_FULL)
        store.set_ingestion_state("leg_02", INGESTION_STATUS_PENDING)

        assert store.get_ingestion_state("leg_01") == INGESTION_STATUS_FULL
        assert store.get_ingestion_state("leg_02") == INGESTION_STATUS_PENDING
        assert len(store.get_passages_by_status(INGESTION_STATUS_FULL)) == 1

        # 5. Add a new macro passage with multi-scale fields
        store.add_passage(
            PassageRecord(
                passage_id="macro_chem_01",
                doc_id="doc_chem",
                char_span=(0, 200),
                text="Thermodynamics overview block.",
                granularity="MACRO",
                concept_codes=(101, 202),
            ),
            initial_status=INGESTION_STATUS_COARSE_ONLY,
        )

        assert len(store.get_macro_passages("doc_chem")) == 1
        assert store.get_ingestion_state("macro_chem_01") == INGESTION_STATUS_COARSE_ONLY

        store.close()

        # 6. Re-open and verify persistence
        store2 = PassageStore(db_path=db_path)
        assert len(store2) == 3
        m_rec = store2.get_passage("macro_chem_01")
        assert m_rec is not None
        assert m_rec.granularity == "MACRO"
        assert m_rec.concept_codes == (101, 202)
        assert store2.get_ingestion_state("macro_chem_01") == INGESTION_STATUS_COARSE_ONLY
        store2.close()


def test_poprag_coarse_node_weight_and_inter_scale_weight():
    """Verifies that PoPRAGGating properly modulates inter-scale edge weights and coarse node damping."""
    gating_default = PoPRAGGating()
    gating_calibrated = PoPRAGGating(inter_scale_weight=0.6, coarse_node_weight=0.35)

    fine_node = QuantaNode(anchor="fine_step", node_type="entity", truth_status="TRUE")
    macro_node = QuantaNode(anchor="macro_chunk", node_type="macro_passage", truth_status="TRUE")
    contra_node = QuantaNode(anchor="clash", node_type="entity", truth_status="CONTRADICTION")

    # 1. Inter-scale edge weight test
    w_default = gating_default.gate_edge(fine_node, macro_node, "INTER_SCALE_PARENT")
    w_calibrated = gating_calibrated.gate_edge(fine_node, macro_node, "INTER_SCALE_PARENT")

    assert w_default == 1.0  # Nominal default
    # Calibrated applies inter_scale_weight (0.6) * coarse_damping (0.35)
    assert pytest.approx(w_calibrated, rel=1e-3) == (0.6 * 0.35)

    # 2. Concept anchor edge weight test
    concept_node = QuantaNode(anchor="concept:42", node_type="concept_anchor", truth_status="TRUE")
    w_concept_def = gating_default.gate_edge(fine_node, concept_node, "CONCEPT_ANCHOR")
    w_concept_cal = gating_calibrated.gate_edge(fine_node, concept_node, "CONCEPT_ANCHOR")

    assert w_concept_def == 1.0
    assert pytest.approx(w_concept_cal, rel=1e-3) == 0.6

    # 3. Belnap contradiction (00_2) MUST strictly prune to 0.0 regardless of calibration
    w_contra = gating_calibrated.gate_edge(contra_node, macro_node, "INTER_SCALE_PARENT")
    assert w_contra == 0.0

    w_contra_target = gating_calibrated.gate_edge(fine_node, contra_node, "INTER_SCALE_PARENT")
    assert w_contra_target == 0.0


def test_hipporag_ppr_passthrough_matches_baseline():
    """Verifies that pass-through (uncalibrated) HippoRAG PPR produces bit-for-bit identical output to baseline."""
    graph = QuantaGraph()

    n1 = QuantaNode(anchor="node_alpha")
    n2 = QuantaNode(anchor="node_beta")
    n3 = QuantaNode(anchor="node_gamma")

    c1 = graph.add_node(n1)
    c2 = graph.add_node(n2)
    c3 = graph.add_node(n3)

    graph.add_edge(c1, "CAUSAL_LEADS_TO", c2)
    graph.add_edge(c2, "VAL_X1_AGENT", c3)
    graph.add_edge(c3, "TEMP_ALLEN_BEFORE", c1)

    # Base retriever (uncalibrated)
    base_retriever = HippoRAGRetriever()
    # Explicit pass-through retriever (None parameters)
    passthrough_retriever = HippoRAGRetriever(inter_scale_weight=None, coarse_node_weight=None)

    W_base, map_base, _ = base_retriever.build_transition_matrix(graph)
    W_pass, map_pass, _ = passthrough_retriever.build_transition_matrix(graph)

    # Transition matrix equivalence
    assert (W_base != W_pass).nnz == 0
    np.testing.assert_allclose(W_base.toarray(), W_pass.toarray(), atol=1e-12)

    # Power iteration steady state equivalence
    p0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    p_base, iters_base, _ = base_retriever.power_iteration(W_base, p0, 0.15, 1e-6, 50)
    p_pass, iters_pass, _ = passthrough_retriever.power_iteration(W_pass, p0, 0.15, 1e-6, 50)

    assert iters_base == iters_pass
    np.testing.assert_allclose(p_base, p_pass, atol=1e-12)


def test_hipporag_ppr_with_multi_scale_coarse_bridges():
    """Verifies that coarse macro nodes and concept anchors act as associative bridges across separated clusters."""
    graph = QuantaGraph()

    # Cluster A (fine nodes)
    a1 = QuantaNode(anchor="A_hypothesis", passage_id="pA")
    a2 = QuantaNode(anchor="A_experiment", passage_id="pA")
    c_a1 = graph.add_node(a1)
    c_a2 = graph.add_node(a2)
    graph.add_edge(c_a1, "CAUSAL_LEADS_TO", c_a2)

    # Cluster B (fine nodes)
    b1 = QuantaNode(anchor="B_conclusion", passage_id="pB")
    b2 = QuantaNode(anchor="B_application", passage_id="pB")
    c_b1 = graph.add_node(b1)
    c_b2 = graph.add_node(b2)
    graph.add_edge(c_b1, "CAUSAL_LEADS_TO", c_b2)

    # Coarse macro bridge node connecting cluster A and cluster B via concept anchor
    macro_bridge = QuantaNode(anchor="macro_bridge_doc", node_type="macro_passage", passage_id="macro_AB")
    c_macro = graph.add_node(macro_bridge)

    graph.add_edge(c_a2, "INTER_SCALE_PARENT", c_macro)
    graph.add_edge(c_macro, "INTER_SCALE_PARENT", c_b1)

    concept_anchor = graph.add_concept_anchor("macro_AB", concept_code=999)
    graph.add_edge(c_a1, "CONCEPT_ANCHOR", concept_anchor.cid)
    graph.add_edge(concept_anchor.cid, "CONCEPT_ANCHOR", c_b2)

    # Run HippoRAG personalized spreading activation seeded at Cluster A (A_hypothesis)
    retriever = HippoRAGRetriever(
        inter_scale_weight=0.7,
        coarse_node_weight=0.5,
    )
    W, node_to_idx, idx_to_node = retriever.build_transition_matrix(graph)

    N = len(idx_to_node)
    p0 = np.zeros(N, dtype=np.float32)
    p0[node_to_idx[a1.cid]] = 1.0

    p_star, iters, _ = retriever.power_iteration(W, p0, alpha=0.15, convergence_tol=1e-6, max_iter=50)

    # Spreading activation must cross from Cluster A to Cluster B via the macro/concept bridge
    score_b1 = p_star[node_to_idx[b1.cid]]
    score_b2 = p_star[node_to_idx[b2.cid]]
    score_macro = p_star[node_to_idx[macro_bridge.cid]]

    assert score_b1 > 0.001
    assert score_b2 > 0.001
    assert score_macro > 0.001
