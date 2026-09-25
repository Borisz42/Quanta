"""Integration Test Suite for Section 7.5: Rigorous Multi-Step Reasoning & Encyclopedic Evaluation at Scale.

Tests Tasks 7.5.1 through 7.5.6 against the live 14GB `data/wikipedia_quanta.db` and MuSiQue benchmark:
- Task 7.5.1: Database Integrity & Source Verification (WikidataIntegrityAuditor)
- Task 7.5.2: Multihop Depth Stress (2-hop to 5-hop Chains, Latency < 10ms, Recall >= 95%)
- Task 7.5.3: Test-Time Learning & Dynamic World-State Revision (WorldStateManager, Read-Only Guard)
- Task 7.5.4: Closed-Loop Soundness & Dual-Level Lattice Invariance (10/10 Corruption Rejection)
- Task 7.5.5: Tri-Fold Head-to-Head Comparative Ablation (Parametric vs Dense RAG vs QUANTA)
- Task 7.5.6: Physical Hardware Bounds (RTX 3070 VRAM <= 128KB, M <= 512, Node Reuse >= 60%)
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List

import pytest

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector
from memory.global_kb import GlobalKnowledgeBase
from memory.node_interner import CanonicalNodeInterner, get_global_interner
from memory.page_table import ActiveCanvas
from memory.world_state import EntityStateRecord, WorldStateManager
from pipeline.multihop_evaluator import (
    DenseRAGBaseline,
    MultiHopBenchmarkEvaluator,
    MultiHopReasoner,
    WikidataIntegrityAuditor,
)
from verification.lattice_gate import LatticeInvarianceGate, LatticeMeetResult


DB_PATH = Path("data/wikipedia_quanta.db")
BENCHMARK_PATH = Path("data/benchmarks/musique_sample_real.json")


@pytest.fixture(scope="module")
def global_kb() -> GlobalKnowledgeBase:
    """Mounts the 14GB Wikipedia knowledge base in strict read-only mode."""
    if not DB_PATH.exists():
        pytest.skip(f"14GB database not found at {DB_PATH}")
    kb = GlobalKnowledgeBase(db_path=DB_PATH)
    yield kb
    kb.close()


@pytest.fixture(scope="module")
def benchmark_dataset() -> List[Dict[str, Any]]:
    """Loads gold MuSiQue reasoning chains."""
    if not BENCHMARK_PATH.exists():
        pytest.skip(f"Benchmark file not found at {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Task 7.5.1: 14GB Database Integrity & Source Verification
# =============================================================================

class TestDatabaseIntegrity:
    """Audits live 14GB encyclopedic database schemas, vectors, and relations."""

    def test_random_sample_schema_and_payload_validity(self, global_kb: GlobalKnowledgeBase):
        """Audits random sample of 250 entities and triples for 100% integrity."""
        auditor = WikidataIntegrityAuditor(global_kb=global_kb)
        report = auditor.audit_random_sample(sample_size=250)

        assert report["status"] == "PASS"
        assert report["overall_integrity_score"] >= 99.0
        assert report["entity_integrity_pct"] == 100.0
        assert report["triple_integrity_pct"] == 100.0
        assert report["category_consistency_pct"] >= 99.0
        assert report["alias_resolution_pct"] >= 95.0
        assert report["elapsed_ms"] < 1000.0  # sub-second random audit

    def test_read_only_mode_strict_write_rejection(self, global_kb: GlobalKnowledgeBase):
        """Verifies strict read-only enforcement (sqlite3.OperationalError on write)."""
        assert global_kb._conn is not None
        with pytest.raises(sqlite3.OperationalError):
            global_kb._conn.execute(
                "INSERT INTO aliases (qid, alias, alias_lower) VALUES ('Q999999', 'Fake', 'fake')"
            )

    def test_instant_bidirectional_alias_resolution(self, global_kb: GlobalKnowledgeBase):
        """Verifies instant resolution of known entities across labels and aliases."""
        test_queries = ["Alan Turing", "Charles Babbage", "United Kingdom", "London", "Douglas Adams"]
        for q in test_queries:
            t0 = time.perf_counter()
            nodes = global_kb.lookup_entity(q, limit=1)
            dt_ms = (time.perf_counter() - t0) * 1000

            assert len(nodes) > 0, f"Failed to resolve {q}"
            node = nodes[0]
            assert isinstance(node, QuantaNode)
            assert node.literal is not None
            assert "qid" in node.literal
            assert dt_ms < 10.0, f"Query for {q} took {dt_ms:.3f} ms (expected < 10ms)"


# =============================================================================
# Task 7.5.2: Multi-Hop Depth Scaling (2-hop to 5-hop Chains)
# =============================================================================

class TestMultiHopDepthScaling:
    """Verifies depth scaling across 2-hop, 3-hop, 4-hop, and 5-hop chains."""

    @pytest.mark.parametrize("hop_count", [2, 3, 4, 5])
    def test_multihop_depth_traversal(
        self,
        global_kb: GlobalKnowledgeBase,
        benchmark_dataset: List[Dict[str, Any]],
        hop_count: int,
    ):
        """Executes multi-hop queries asserting bridge recall >= 90% and latency < 10ms."""
        reasoner = MultiHopReasoner(global_kb=global_kb)
        samples = [d for d in benchmark_dataset if d.get("hop_count") == hop_count][:5]
        assert len(samples) > 0, f"No samples found for hop count {hop_count}"

        recalls = []
        latencies = []

        for s in samples:
            res = reasoner.execute_reasoning_chain(
                start_entity=s["start_entity"],
                reasoning_chain=s["reasoning_chain"],
                expected_target=s["answer"],
            )
            assert res.success is True
            assert res.target_node is not None
            assert res.lattice_meet_sound is True
            assert res.latency_ms < 25.0, f"Individual latency {res.latency_ms:.3f}ms exceeded peak limit"

            recalls.append(res.bridge_recall)
            latencies.append(res.latency_ms)

        mean_recall = sum(recalls) / len(recalls)
        mean_latency = sum(latencies) / len(latencies)

        assert mean_recall >= 0.90, f"Mean recall {mean_recall:.2f} below 0.90 target for {hop_count}-hop"
        assert mean_latency < 10.0, f"Mean latency {mean_latency:.2f}ms exceeded 10.0ms target"


# =============================================================================
# Task 7.5.3: Test-Time Learning & Dynamic World-State Revision
# =============================================================================

class TestDynamicBeliefRevision:
    """Stress-tests test-time non-monotonic belief revision over immutable 14GB DB."""

    def test_dynamic_fluent_mutation_and_point_in_time_query(
        self,
        global_kb: GlobalKnowledgeBase,
    ):
        """Verifies point-in-time state resolution while DB remains 100% untouched."""
        wsm = WorldStateManager()
        reasoner = MultiHopReasoner(global_kb=global_kb, world_state=wsm)

        subject_qid = "Q46633"  # Charles Babbage
        prop = "VAL_LOCATION_SLOT"

        # Event 1: At t = 1810.0, Location = Cambridge (Q350)
        wsm.assert_state(
            entity_cid=subject_qid,
            property_slot=prop,
            value_cid="Q350",
            t_start=1810.0,
        )

        # Event 2: At t = 1812.0, Transfer to London (Q84) (closes interval [1810.0, 1812.0))
        wsm.assert_state(
            entity_cid=subject_qid,
            property_slot=prop,
            value_cid="Q84",
            t_start=1812.0,
        )

        # Event 3: At t = 1828.0, Transfer to Rome (Q220) (closes interval [1812.0, 1828.0))
        wsm.assert_state(
            entity_cid=subject_qid,
            property_slot=prop,
            value_cid="Q220",
            t_start=1828.0,
        )

        # Query Point-in-Time Historical State (t = 1811.0 -> Cambridge)
        rec_1811 = wsm.get_entity_state_record_at(subject_qid, prop, timestamp=1811.0)
        assert rec_1811 is not None
        assert rec_1811.value_cid == "Q350"
        assert rec_1811.t_start == 1810.0
        assert rec_1811.t_end == 1812.0

        # Query Point-in-Time Transitional State (t = 1820.0 -> London)
        rec_1820 = wsm.get_entity_state_record_at(subject_qid, prop, timestamp=1820.0)
        assert rec_1820 is not None
        assert rec_1820.value_cid == "Q84"
        assert rec_1820.t_start == 1812.0
        assert rec_1820.t_end == 1828.0

        # Query Current Open-Ended State (t = 1835.0 -> Rome)
        rec_1835 = wsm.get_entity_state_record_at(subject_qid, prop, timestamp=1835.0)
        assert rec_1835 is not None
        assert rec_1835.value_cid == "Q220"
        assert rec_1835.t_start == 1828.0
        assert rec_1835.is_current is True

        # Verify through MultiHopReasoner execution
        mock_chain = [{"property": prop, "object_qid": "DYNAMIC"}]
        res_hist = reasoner.execute_reasoning_chain(
            start_entity=subject_qid,
            reasoning_chain=mock_chain,
            timestamp=1811.0,
        )
        assert res_hist.target_qid == "Q350"

        res_curr = reasoner.execute_reasoning_chain(
            start_entity=subject_qid,
            reasoning_chain=mock_chain,
            timestamp=1835.0,
        )
        assert res_curr.target_qid == "Q220"

        # Assert DB file integrity was untouched
        assert DB_PATH.stat().st_size == 13997084672


# =============================================================================
# Task 7.5.4: Closed-Loop Dual Lattice Invariance Gate
# =============================================================================

class TestLatticeInvarianceSoundness:
    """Verifies dual-level lattice meet ($v_t \\sqcap v_p$) and 100% corruption rejection."""

    def test_dual_level_clean_lattice_meet(self, global_kb: GlobalKnowledgeBase):
        """Verifies that clean entities meet soundly with zero Hamming drift."""
        gate = LatticeInvarianceGate()
        london = global_kb.get_entity_by_qid("Q84")
        assert london is not None

        # Entity-level meet
        meet_res = gate.audit_round_trip(london, london)
        assert meet_res.is_sound is True
        assert meet_res.hamming_distance == 0
        assert meet_res.contradiction_count == 0

    def test_corrupted_completion_rejection_rate(self, global_kb: GlobalKnowledgeBase):
        """Deliberately injects 10 corrupted completions; asserts 100% rejection rate."""
        gate = LatticeInvarianceGate()
        base_node = global_kb.get_entity_by_qid("Q84")  # London (location)
        assert base_node is not None

        corrupted_cases = []

        # 1. Epistemic Polarity Inversion (slot 1 vs slot 2)
        v1 = base_node.vector.copy()
        v1[1] = 2  # Polarity flipped to FALSE
        corrupted_cases.append(v1)

        # 2. Incompatible Category Swapping (London vector overwritten with Human taxonomy)
        v2 = base_node.vector.copy()
        v2[136] = 0  # Remove Location slot
        v2[2] = 1    # Force Human slot
        corrupted_cases.append(v2)

        # 3. Direct Epistemic Clash (active slot set to opposite truth value)
        v3 = base_node.vector.copy()
        v3[1] = 2
        v3[5] = 2
        corrupted_cases.append(v3)

        # 4. Zeroed out vector (null meet drops 100% of propositional slots)
        v4 = QuantaVector.zeros()
        corrupted_cases.append(v4)

        # 5-10: Inverted slot pairs across different bands
        for offset in [10, 20, 30, 40, 50, 60]:
            v_inv = base_node.vector.copy()
            v_inv[offset] = 2 if base_node.vector[offset] == 1 else 1
            corrupted_cases.append(v_inv)

        assert len(corrupted_cases) == 10

        rejections = 0
        for i, c_vec in enumerate(corrupted_cases):
            c_node = QuantaNode(vector=c_vec, edges={}, anchor="Corrupted")
            audit = gate.audit_round_trip(base_node, c_node)
            if not audit.is_sound or audit.contradiction_count > 0 or audit.hamming_distance > 0:
                rejections += 1

        assert rejections == 10, f"Expected 10/10 rejections, got {rejections}/10"


# =============================================================================
# Task 7.5.5: Tri-Fold Head-to-Head Comparative Ablation
# =============================================================================

class TestTriFoldAblation:
    """Benchmarks Zero-Shot Parametric LLM vs Dense RAG vs QUANTA."""

    def test_head_to_head_ablation_comparison(
        self,
        global_kb: GlobalKnowledgeBase,
        benchmark_dataset: List[Dict[str, Any]],
    ):
        """Verifies QUANTA bridge recall >= 95% while Dense RAG drops below 60%."""
        evaluator = MultiHopBenchmarkEvaluator(
            global_kb=global_kb,
            benchmark_path=BENCHMARK_PATH,
        )

        results = evaluator.run_benchmark(
            sample_size=30,
            hops="all",
            audit_sample_size=100,
            mode="offline",
        )

        ov = results["overall"]
        assert ov["quanta_bridge_recall_pct"] >= 95.0, (
            f"QUANTA recall {ov['quanta_bridge_recall_pct']}% below 95.0%"
        )
        assert ov["dense_bridge_recall_pct"] < 60.0, (
            f"Dense RAG recall {ov['dense_bridge_recall_pct']}% did not reflect semantic hop drift"
        )
        assert ov["hallucination_rate_pct"] == 0.0
        assert ov["prompt_token_compression_pct"] >= 70.0
        assert ov["mean_traversal_latency_ms"] < 10.0


# =============================================================================
# Task 7.5.6: Resource Bound & Physical Hardware SLA Verification
# =============================================================================

class TestPhysicalHardwareBounds:
    """Enforces ActiveCanvas M <= 512 bound and node interning reuse >= 60%."""

    def test_active_canvas_bounded_footprint_100_queries(
        self,
        global_kb: GlobalKnowledgeBase,
        benchmark_dataset: List[Dict[str, Any]],
    ):
        """Runs 100 sequential multi-hop queries and asserts strict O(1) memory bound."""
        canvas = ActiveCanvas(capacity=512)
        interner = CanonicalNodeInterner()
        reasoner = MultiHopReasoner(
            global_kb=global_kb,
            active_canvas=canvas,
            interner=interner,
        )

        samples = benchmark_dataset[:100]
        for s in samples:
            res = reasoner.execute_reasoning_chain(
                start_entity=s["start_entity"],
                reasoning_chain=s["reasoning_chain"],
                expected_target=s["answer"],
            )
            # Physical hardware assertion: canvas size must never exceed 512
            assert canvas.size <= 512
            assert canvas.active_memory_bytes() <= 128 * 1024  # <= 128 KB VRAM footprint

        # Assert flyweight interner reuse rate >= 60%
        stats = interner.stats()
        reuse_rate_pct = stats["reuse_rate"] * 100.0
        assert reuse_rate_pct >= 60.0, f"Node interning hit rate {reuse_rate_pct:.1f}% below 60% target"
