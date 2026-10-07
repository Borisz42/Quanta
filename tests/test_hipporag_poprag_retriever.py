"""Tests for Section 6: HippoRAG 2 Spreading Activation & PoP-RAG Epistemic Gating.

Verifies:
1. PoP-RAG Epistemic Gating:
   - 00_2 (Contradiction) -> edge weight set to 0.0 (path completely pruned).
   - 01_2 (True) -> weight maintained at nominal value.
   - 11_2 (Unknown) -> weight scaled by calibrated confidence P.
   - 10_2 (False) -> inverted or suppressed based on query target polarity.
2. HippoRAG 2 Personalized PageRank:
   - Power iteration convergence with alpha in [0.15, 0.20] and tolerance epsilon = 10^-6.
   - Activation distribution over complex 4-hop graphs with zero multi-hop semantic drift.
3. Attenuation and Pruning:
   - Contradictory edges (00_2) completely pruned from activation path.
   - Unknown edges (11_2) have attenuated influence compared to verified true edges (01_2).
4. Scalability & Latency:
   - Traversal latency over 100,000 nodes strictly < 5.0 ms.
5. Integration:
   - SpreadingActivationRetriever seamless integration with algorithm="hipporag".
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pytest
import scipy.sparse as sp

from core.asg import QuantaGraph, QuantaNode
from core.binary_node import BelnapValue
from memory.hipporag_ppr import HippoRAGRetriever
from memory.page_table import PageTable
from memory.poprag_gating import DEFAULT_RELATION_WEIGHTS, PoPRAGGating
from memory.spreading_activation import SpreadingActivationRetriever


# =============================================================================
# Helper Utilities
# =============================================================================

def _make_node(
    anchor: str,
    literal: Any,
    truth_status: str = "TRUE",
    confidence: float = 1.0,
    edges: Dict[str, List[str]] | None = None,
    slots: Dict[str, int] | None = None,
) -> QuantaNode:
    """Helper to create a QuantaNode with Belnap truth status, confidence, and edges."""
    node = QuantaNode(
        anchor=anchor,
        literal=literal,
        truth_status=truth_status,
        confidence=confidence,
        edges=edges or {},
    )
    if slots:
        for s_name, val in slots.items():
            node.set_slot(s_name, val)
    node.compute_cid()
    return node


# =============================================================================
# Task 6.1: PoP-RAG Epistemic Gating Tests
# =============================================================================

class TestPoPRAGEpistemicGating:
    """Verifies edge weight modulation by Belnap 4-valued logic lattice states."""

    @pytest.fixture
    def gating(self) -> PoPRAGGating:
        return PoPRAGGating()

    def test_gate_factor_contradiction_pruning(self, gating: PoPRAGGating):
        """00_2 (Contradiction) must yield factor 0.0 (path completely pruned)."""
        # Test various representations of Contradiction
        assert gating.compute_gate_factor(BelnapValue.CONTRADICTION) == 0.0
        assert gating.compute_gate_factor("CONTRADICTION") == 0.0
        assert gating.compute_gate_factor("contra") == 0.0
        assert gating.compute_gate_factor(0) == 0.0

        # Gated edge weight must be strictly 0.0 regardless of nominal weight
        w_causal = gating.gate_edge_weight(1.2, BelnapValue.CONTRADICTION, confidence=1.0)
        assert w_causal == 0.0

    def test_gate_factor_true_nominal_maintenance(self, gating: PoPRAGGating):
        """01_2 (True) must maintain nominal base weight."""
        assert gating.compute_gate_factor(BelnapValue.TRUE) == 1.0
        assert gating.compute_gate_factor("TRUE") == 1.0
        assert gating.compute_gate_factor("fact") == 1.0
        assert gating.compute_gate_factor(1) == 1.0

        # Base weights preserved
        w_causal = gating.gate_edge_weight(1.2, BelnapValue.TRUE)
        assert pytest.approx(w_causal, 1e-5) == 1.2

        w_temporal = gating.gate_edge_weight(1.0, BelnapValue.TRUE)
        assert pytest.approx(w_temporal, 1e-5) == 1.0

    def test_gate_factor_unknown_confidence_scaling(self, gating: PoPRAGGating):
        """11_2 (Unknown) must scale weight by calibrated confidence P."""
        assert pytest.approx(gating.compute_gate_factor(BelnapValue.UNKNOWN, confidence=0.75), 1e-5) == 0.75
        assert pytest.approx(gating.compute_gate_factor("UNKNOWN", confidence=0.40), 1e-5) == 0.40
        assert pytest.approx(gating.compute_gate_factor("maybe", confidence=0.20), 1e-5) == 0.20

        # Scaled edge weight
        base = 1.2
        w_attenuated = gating.gate_edge_weight(base, BelnapValue.UNKNOWN, confidence=0.5)
        assert pytest.approx(w_attenuated, 1e-5) == 0.60

    def test_gate_factor_false_polarity_handling(self, gating: PoPRAGGating):
        """10_2 (False) is suppressed on positive queries and inverted on negative queries."""
        # Positive query: suppressed
        assert gating.compute_gate_factor(BelnapValue.FALSE, confidence=0.9, query_polarity="positive") == 0.0
        assert gating.gate_edge_weight(1.0, "FALSE", confidence=0.8, query_polarity="positive") == 0.0

        # Negative / refutation query: active with refutation strength
        factor_neg = gating.compute_gate_factor(BelnapValue.FALSE, confidence=0.9, query_polarity="negative")
        assert factor_neg > 0.0

    def test_gate_weights_vectorized(self, gating: PoPRAGGating):
        """Vectorized NumPy evaluation matches scalar logic across all Belnap states."""
        base_weights = np.array([1.2, 1.0, 1.0, 1.2], dtype=np.float32)
        states = np.array([
            BelnapValue.TRUE,
            BelnapValue.CONTRADICTION,
            BelnapValue.UNKNOWN,
            BelnapValue.FALSE,
        ], dtype=np.uint8)
        confidences = np.array([1.0, 0.95, 0.45, 0.85], dtype=np.float32)

        # Positive query polarity
        gated_pos = gating.gate_weights_vectorized(base_weights, states, confidences, query_polarity="positive")
        assert pytest.approx(gated_pos[0], 1e-5) == 1.2         # TRUE: maintained
        assert pytest.approx(gated_pos[1], 1e-5) == 0.0         # CONTRA: pruned
        assert pytest.approx(gated_pos[2], 1e-5) == 1.0 * 0.45  # UNKNOWN: scaled
        assert pytest.approx(gated_pos[3], 1e-5) == 0.0         # FALSE: suppressed


# =============================================================================
# Task 6.2: HippoRAG 2 Personalized PageRank Core Tests
# =============================================================================

class TestHippoRAGPersonalizedPageRank:
    """Verifies Personalized PageRank computation, convergence, and 4-hop propagation."""

    @pytest.fixture
    def retriever(self) -> HippoRAGRetriever:
        return HippoRAGRetriever(alpha=0.15, convergence_tol=1e-6, max_iter=50)

    def test_ppr_convergence_and_distribution_4hop(self, retriever: HippoRAGRetriever):
        """Verifies PPR convergence and activation distribution over a 4-hop chain.
        
        Chain: E0 -> E1 -> EV2 -> EV3 -> E4
        """
        graph = QuantaGraph()

        e4 = _make_node("cn:en:target_entity", "Target Concept E4")
        ev3 = _make_node("cn:en:event_3", "Event 3", edges={"VAL_X2_PATIENT": [e4.cid]})
        ev2 = _make_node("cn:en:event_2", "Event 2", edges={"TEMP_ALLEN_MEETS": [ev3.cid]})
        e1 = _make_node("cn:en:entity_1", "Intermediate E1", edges={"CAUSAL_MECHANISM_LINK": [ev2.cid]})
        e0 = _make_node("cn:en:query_seed", "Query Seed E0", edges={"VAL_X1_AGENT": [e1.cid]})

        # Add distractor node not connected to the chain
        distractor = _make_node("cn:en:unrelated", "Unrelated Distractor")

        for n in [e0, e1, ev2, ev3, e4, distractor]:
            graph.add_node(n)

        scores = retriever.compute_ppr(seed_cids=[e0.cid], graph=graph)

        # 1. Total probability mass must sum to 1.0 (L1 conservation)
        total_mass = sum(scores.values())
        assert pytest.approx(total_mass, 1e-4) == 1.0

        # 2. All chain nodes from 0 to 4 hops must be reachable
        assert e0.cid in scores
        assert e1.cid in scores
        assert ev2.cid in scores
        assert ev3.cid in scores
        assert e4.cid in scores

        # 3. Activation scores decay along the relational chain
        assert scores[e0.cid] > 0.15
        assert scores[e1.cid] > scores[ev2.cid]
        assert scores[ev2.cid] > scores[ev3.cid]
        assert scores[ev3.cid] > scores[e4.cid]
        assert scores[e4.cid] > 0.0

        # Also verify directed traversal where activation flows strictly away from seed
        retriever_dir = HippoRAGRetriever(alpha=0.15, bidirectional=False)
        scores_dir = retriever_dir.compute_ppr(seed_cids=[e0.cid], graph=graph)
        assert scores_dir[e0.cid] > scores_dir[e1.cid] > scores_dir[ev2.cid] > scores_dir[ev3.cid] > scores_dir[e4.cid]

        # 4. Zero semantic drift: completely disconnected distractor receives 0.0 activation
        assert scores.get(distractor.cid, 0.0) == 0.0

    def test_contradictory_edge_complete_pruning(self, retriever: HippoRAGRetriever):
        """Verifies that contradictory edges (00_2) are completely pruned from the activation path."""
        graph = QuantaGraph()

        # Seed node
        seed = _make_node("cn:en:seed", "Seed Query Concept")

        # Path A (Verified True): Seed -> A1 -> TargetA
        target_a = _make_node("cn:en:target_a", "Target A (Verified)", truth_status="TRUE")
        node_a1 = _make_node(
            "cn:en:node_a1", "Node A1",
            truth_status="TRUE",
            edges={"CAUSAL_LEADS_TO": [target_a.cid]},
        )
        seed.edges["VAL_X1_AGENT"] = [node_a1.cid]

        # Path B (Contradictory): Seed -> B1 (CONTRADICTION) -> TargetB
        target_b = _make_node("cn:en:target_b", "Target B (Contradicted)", truth_status="TRUE")
        node_b1 = _make_node(
            "cn:en:node_b1", "Node B1 (Invalid)",
            truth_status="CONTRADICTION",  # 00_2
            confidence=0.95,
            edges={"CAUSAL_LEADS_TO": [target_b.cid]},
        )
        seed.edges["VAL_X2_PATIENT"] = [node_b1.cid]

        for n in [seed, node_a1, target_a, node_b1, target_b]:
            graph.add_node(n)

        scores = retriever.compute_ppr(seed_cids=[seed.cid], graph=graph)

        # Target A must receive strong activation
        assert scores.get(target_a.cid, 0.0) > 0.01

        # Target B and Node B1 must receive STRICTLY 0.0 activation
        assert scores.get(target_b.cid, 0.0) == 0.0
        assert scores.get(node_b1.cid, 0.0) == 0.0

    def test_unknown_edge_attenuated_influence(self, retriever: HippoRAGRetriever):
        """Verifies that unknown edges (11_2) have attenuated influence compared to verified true edges (01_2)."""
        graph = QuantaGraph()
        seed = _make_node("cn:en:seed", "Seed Concept")

        # Path True: Seed -> TrueNode -> TargetTrue
        target_true = _make_node("cn:en:target_true", "Target True")
        node_true = _make_node(
            "cn:en:node_true", "Node True",
            truth_status="TRUE",
            confidence=1.0,
            edges={"CAUSAL_LEADS_TO": [target_true.cid]},
        )

        # Path Unknown: Seed -> UnknownNode -> TargetUnknown
        target_unknown = _make_node("cn:en:target_unknown", "Target Unknown")
        node_unknown = _make_node(
            "cn:en:node_unknown", "Node Unknown",
            truth_status="UNKNOWN",
            confidence=0.30,  # Attenuated confidence
            edges={"CAUSAL_LEADS_TO": [target_unknown.cid]},
        )

        seed.edges["VAL_X1_AGENT"] = [node_true.cid]
        seed.edges["VAL_X2_PATIENT"] = [node_unknown.cid]

        for n in [seed, node_true, target_true, node_unknown, target_unknown]:
            graph.add_node(n)

        scores = retriever.compute_ppr(seed_cids=[seed.cid], graph=graph)

        # TargetTrue must have strictly higher activation than TargetUnknown
        score_true = scores.get(target_true.cid, 0.0)
        score_unknown = scores.get(target_unknown.cid, 0.0)

        assert score_true > score_unknown
        assert score_true >= 1.5 * score_unknown, (
            f"Verified fact ({score_true}) must significantly out-activate unverified fact ({score_unknown})"
        )

    def test_retrieve_subgraph_preserves_provenance_and_sets_salience(self, retriever: HippoRAGRetriever):
        """Verifies that retrieve_subgraph returns QuantaGraph with updated salience and intact provenance."""
        graph = QuantaGraph()
        n1 = _make_node("cn:en:source", "Source Entity", truth_status="TRUE")
        n1.bind_passage("P101", 10, 35, confidence=0.98)
        n2 = _make_node("cn:en:target", "Target Entity", truth_status="TRUE")
        n2.bind_passage("P101", 40, 60, confidence=0.95)
        n1.edges["VAL_X1_AGENT"] = [n2.cid]

        graph.add_node(n1)
        graph.add_node(n2)

        subgraph = retriever.retrieve_subgraph(seed_cids=[n1.cid], graph=graph, threshold=0.001)

        assert isinstance(subgraph, QuantaGraph)
        assert len(subgraph.nodes) == 2

        sub_n1 = subgraph.get_node(n1.cid)
        sub_n2 = subgraph.get_node(n2.cid)
        assert sub_n1 is not None and sub_n2 is not None

        # Provenance attributes preserved
        assert sub_n1.passage_id == "P101"
        assert sub_n1.span_start == 10
        assert sub_n1.span_end == 35

        # Salience reflects PPR score
        assert sub_n1.salience > sub_n2.salience
        assert sub_n1.salience > 0.0


# =============================================================================
# Task 6.3: 100,000-Node Traversal Latency Benchmark (< 5 ms)
# =============================================================================

class TestHippoRAGScalingLatency:
    """Verifies that HippoRAG PPR traversal executes in < 5.0 ms over 100,000 nodes."""

    def test_100k_node_traversal_latency(self):
        """Constructs synthetic 100,000-node sparse graph and benchmarks PPR traversal latency."""
        N = 100000
        nnz = 300000  # Average degree 3 (realistic sparse multi-hop graph)

        rng = np.random.RandomState(42)
        sources = rng.randint(0, N, nnz)
        targets = rng.randint(0, N, nnz)
        weights = rng.uniform(0.8, 1.2, nnz).astype(np.float32)

        # Adjacency matrix in CSR format (row: target, col: source)
        adj = sp.csr_matrix((weights, (targets, sources)), shape=(N, N), dtype=np.float32)
        col_sums = np.array(adj.sum(axis=0)).flatten()
        non_zero = col_sums > 0.0
        inv_sums = np.zeros(N, dtype=np.float32)
        inv_sums[non_zero] = 1.0 / col_sums[non_zero]
        W = adj.dot(sp.diags(inv_sums, format="csr", dtype=np.float32)).tocsr()

        node_to_idx = {f"node_{i:06x}": i for i in range(N)}
        idx_to_node = [f"node_{i:06x}" for i in range(N)]

        retriever = HippoRAGRetriever(alpha=0.15, max_iter=25, localized_threshold_nodes=10000)

        # Query seeds (5 seeds)
        seed_cids = [idx_to_node[42], idx_to_node[100], idx_to_node[500], idx_to_node[1000], idx_to_node[5000]]

        # Warmup pass
        _ = retriever.compute_ppr(
            seed_cids=seed_cids,
            W=W,
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
        )

        # Benchmark 15 iterations
        latencies: List[float] = []
        for _ in range(15):
            t0 = time.perf_counter()
            scores = retriever.compute_ppr(
                seed_cids=seed_cids,
                W=W,
                node_to_idx=node_to_idx,
                idx_to_node=idx_to_node,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

        min_lat = float(np.min(latencies))
        mean_lat = float(np.mean(latencies))

        print(
            f"\n[Section 6 HippoRAG Benchmark] 100,000-Node Traversal Latency: "
            f"min={min_lat:.3f} ms, mean={mean_lat:.3f} ms"
        )

        assert len(scores) > 0
        # Strict latency requirement: Traversal latency over 100,000 nodes must be < 5.0 ms
        assert min_lat < 5.0, f"Min latency must be < 5.0 ms, got {min_lat:.3f} ms"
        assert mean_lat < 5.0, f"Mean latency must be < 5.0 ms, got {mean_lat:.3f} ms"


# =============================================================================
# Task 6.4: SpreadingActivationRetriever Integration Tests
# =============================================================================

class TestSpreadingActivationHippoRAGIntegration:
    """Verifies that SpreadingActivationRetriever executes seamlessly with algorithm='hipporag'."""

    def test_spreading_activation_hipporag_mode(self, tmp_path: Path):
        """Verifies traverse_subgraph with algorithm='hipporag' over PageTable."""
        db_path = tmp_path / "hipporag_pt_test.db"
        pt = PageTable(db_path)

        retriever = SpreadingActivationRetriever(algorithm="hipporag")

        # Create 3-hop chain: ev1 -> ev2 -> ev3
        ev3 = _make_node("cn:en:ev3", "Result Event", slots={"TYPE_EVENT": 1})
        ev2 = _make_node("cn:en:ev2", "Inter Event", slots={"TYPE_EVENT": 1}, edges={"TEMP_ALLEN_MEETS": [ev3.cid]})
        ev1 = _make_node("cn:en:ev1", "Start Event", slots={"TYPE_EVENT": 1}, edges={"CAUSAL_LEADS_TO": [ev2.cid]})

        pt.store_node(ev3)
        pt.store_node(ev2)
        pt.store_node(ev1)

        subgraph = retriever.traverse_subgraph(
            seed_cids=[ev1.cid],
            page_table=pt,
            max_depth=3,
            algorithm="hipporag",
        )

        assert isinstance(subgraph, QuantaGraph)
        assert ev1.cid in subgraph.nodes
        assert ev2.cid in subgraph.nodes
        assert ev3.cid in subgraph.nodes

        pt.close()

    def test_spreading_activation_contradiction_blocked_in_hipporag_mode(self, tmp_path: Path):
        """Verifies that contradictory paths are blocked when queried through SpreadingActivationRetriever."""
        db_path = tmp_path / "hipporag_contra_test.db"
        pt = PageTable(db_path)
        retriever = SpreadingActivationRetriever(algorithm="hipporag")

        # Valid branch
        ev_good = _make_node("cn:en:good", "Good Fact", truth_status="TRUE")
        # Contradictory branch
        ev_contra = _make_node("cn:en:bad", "Contradicted Fact", truth_status="CONTRADICTION")

        seed = _make_node(
            "cn:en:start", "Start Seed",
            edges={
                "CAUSAL_LEADS_TO": [ev_good.cid],
                "CAUSAL_MECHANISM_LINK": [ev_contra.cid],
            },
        )

        pt.store_node(ev_good)
        pt.store_node(ev_contra)
        pt.store_node(seed)

        subgraph = retriever.traverse_subgraph(
            seed_cids=[seed.cid],
            page_table=pt,
            algorithm="hipporag",
        )

        assert ev_good.cid in subgraph.nodes
        assert ev_contra.cid not in subgraph.nodes, "Contradictory node must not be admitted to retrieved subgraph"

        pt.close()
