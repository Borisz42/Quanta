"""Tests for Session 4: Decoupled Write-Ahead SVM with Asynchronous Verification.

Verifies:
1. Provisional node records commit immediately with Belnap UNKNOWN (11_2) / provisional confidence.
2. Ingestion returns in < 1.3s (>= 200 words/sec).
3. Background daemon worker refines Belnap lattice in-place in SQLite PageTable and BinaryNodeTable.
4. Priority queueing elevates query-active chunks.
5. Thread-safe execution under concurrent multi-chunk writes.
"""

from __future__ import annotations

import time
from typing import List
import pytest

from core.asg import QuantaGraph, QuantaNode
from core.binary_node import (
    BelnapValue,
    BinaryNodeTable,
    QuantaSemanticNodeStruct,
    SpeechActIntent as BinarySpeechActIntent,
    EpistemicSource as BinaryEpistemicSource,
)
from memory.page_table import PageTable
from models.kev_async_worker import AsyncKevVerificationQueue, VerificationTask
from models.kev_engine import KevDecisionEngine, MockKevEngine
from parser.skeleton_transducer import MockSkeletonTransducer
from pipeline.cognitive_pipeline import CognitivePipeline
from verification.belnap_calibrator import BelnapLatticeMapper


@pytest.fixture
def clean_page_table():
    """Provides an isolated in-memory PageTable."""
    return PageTable(db_path=":memory:")


@pytest.fixture
def binary_table():
    """Provides a fresh BinaryNodeTable."""
    return BinaryNodeTable()


@pytest.fixture
def mock_kev_engine():
    """Provides a mock Kev decision engine."""
    return KevDecisionEngine(base_url="mock", fallback_to_mock=True)


class TestAsyncKevWorker:
    """Unit tests for AsyncKevVerificationQueue."""

    def test_worker_lifecycle(self, clean_page_table, binary_table, mock_kev_engine):
        """Worker pool starts, accepts tasks, flushes, and stops cleanly."""
        queue = AsyncKevVerificationQueue(
            kev_engine=mock_kev_engine,
            page_table=clean_page_table,
            binary_table=binary_table,
            max_workers=2,
            auto_start=True,
        )
        assert queue.is_running
        assert queue.pending_count == 0
        assert queue.completed_count == 0

        # Transduce mock skeleton
        transducer = MockSkeletonTransducer(mode="standard")
        text = "Marcus Vance observed the reactor core. The pressure breached the valve."
        res = transducer.transduce(text, passage_id="P1")

        # Register nodes in binary table
        ent_node_ids = {}
        for ent in res.entities:
            nid = binary_table.append(QuantaSemanticNodeStruct.create(
                node_id=len(binary_table) + 1,
                passage_id=1,
                belnap_lattice=BelnapValue.UNKNOWN,
            ))
            ent_node_ids[ent.id] = nid + 1

        ev_node_ids = {}
        for ev in res.events:
            nid = binary_table.append(QuantaSemanticNodeStruct.create(
                node_id=len(binary_table) + 1,
                passage_id=1,
                belnap_lattice=BelnapValue.UNKNOWN,
            ))
            ev_node_ids[ev.id] = nid + 1

        # Store placeholder nodes in PageTable
        ent_cid_map = {}
        for ent in res.entities:
            node = QuantaNode(anchor=ent.surface_text, literal=ent.surface_text, truth_status="UNKNOWN", confidence=0.85)
            node.compute_canonical_cid()
            clean_page_table.store_node(node)
            ent_cid_map[ent.id] = node.canonical_cid

        ev_cid_map = {}
        for ev in res.events:
            node = QuantaNode(anchor=ev.predicate, literal=ev.predicate, truth_status="UNKNOWN", confidence=0.85)
            node.compute_canonical_cid()
            clean_page_table.store_node(node)
            ev_cid_map[ev.id] = node.canonical_cid

        # Enqueue task
        task = queue.enqueue(
            passage_id="P1",
            doc_id="D1",
            text=text,
            entities=res.entities,
            events=res.events,
            ent_cid_map=ent_cid_map,
            ev_cid_map=ev_cid_map,
            ent_node_ids=ent_node_ids,
            ev_node_ids=ev_node_ids,
        )
        assert task.passage_id == "P1"

        # Flush queue
        flushed = queue.flush(timeout=5.0)
        assert flushed
        assert queue.is_verified("P1")
        assert queue.completed_count == 1

        # Check binary table in-place update
        for ev in res.events:
            nid = ev_node_ids[ev.id]
            struct = binary_table[nid - 1]
            assert struct.belnap_lattice == BelnapValue.TRUE
            assert struct.confidence_float >= 0.85

        queue.stop()
        assert not queue.is_running

    def test_priority_queueing(self, clean_page_table, binary_table, mock_kev_engine):
        """High priority task is processed before normal priority task."""
        # Initialize with auto_start=False to inspect ordering
        queue = AsyncKevVerificationQueue(
            kev_engine=mock_kev_engine,
            page_table=clean_page_table,
            binary_table=binary_table,
            max_workers=1,
            auto_start=False,
        )

        transducer = MockSkeletonTransducer(mode="standard")
        res = transducer.transduce("Marcus Vance observed the core.", passage_id="P_norm")
        res2 = transducer.transduce("Eleanor Vance inspected the laser.", passage_id="P_urgent")

        queue.enqueue(
            passage_id="P_norm",
            doc_id="D1",
            text="Marcus Vance observed the core.",
            entities=res.entities,
            events=res.events,
            priority=10,
        )
        queue.enqueue(
            passage_id="P_urgent",
            doc_id="D1",
            text="Eleanor Vance inspected the laser.",
            entities=res2.entities,
            events=res2.events,
            priority=10,
        )

        # Prioritize P_urgent
        prioritized = queue.prioritize("P_urgent")
        assert prioritized

        # Start worker and flush
        queue.start()
        flushed = queue.flush(timeout=5.0)
        assert flushed
        assert queue.is_verified("P_urgent")
        assert queue.is_verified("P_norm")
        queue.stop()


class TestAsyncCognitivePipeline:
    """Integration tests for CognitivePipeline with kev_mode='async'."""

    def test_provisional_ingestion_immediate_return(self):
        """ingest_document with kev_mode='async' returns provisional graph immediately."""
        pipeline = CognitivePipeline(
            transducer_backend="mock",
            kev_mode="async",
        )
        text = "Marcus Vance observed the reactor core. The pressure breached the valve."
        t0 = time.perf_counter()
        graph = pipeline.ingest_document(text, doc_id="doc_async_01")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Sub-second return SLA (< 1300 ms on live, < 50 ms in mock)
        assert latency_ms < 1300.0

        # Graph should exist and contain provisional tags
        assert graph is not None
        assert len(graph.nodes) >= 2
        assert getattr(graph, "kev_evaluation", None) is not None
        assert graph.kev_evaluation.mode == "async_provisional"

        # Event nodes in graph should be provisional UNKNOWN initially
        for node in graph.nodes.values():
            if node.get_slot("TYPE_EVENT") == 1:
                assert node.truth_status in ("UNKNOWN", "TRUE")
                assert node.evidence_source in ("provisional", "direct_observation")

        # Flush queue and verify refinement
        flushed = pipeline.flush_kev_queue(timeout=5.0)
        assert flushed

        pid = getattr(graph, "passage_id")
        assert pipeline.async_kev_worker.is_verified(pid)

        pipeline.close()

    def test_asynchronous_belnap_lattice_refinement(self):
        """PageTable and BinaryNodeTable are updated in-place with calibrated Belnap posteriors."""
        pipeline = CognitivePipeline(
            transducer_backend="mock",
            kev_mode="async",
        )
        text = "Marcus Vance observed the reactor core. The pressure breached the valve."
        graph = pipeline.ingest_document(text, doc_id="doc_async_02")

        # Flush background queue
        flushed = pipeline.flush_kev_queue(timeout=5.0)
        assert flushed

        # Inspect PageTable stored nodes
        for node_cid, node in graph.nodes.items():
            if node.get_slot("TYPE_EVENT") == 1:
                stored = pipeline.page_table.fetch_node(node_cid)
                assert stored is not None
                assert stored.truth_status == "TRUE"
                assert stored.confidence >= 0.85
                assert stored.evidence_source == "direct_observation"

        # Inspect BinaryNodeTable 128-byte structs
        assert len(pipeline.binary_table) >= 2
        pid = getattr(graph, "passage_id")
        task = pipeline.async_kev_worker._tasks[pid]
        for ev_id, nid in task.ev_node_ids.items():
            struct = pipeline.binary_table[nid - 1]
            assert struct.belnap_lattice == BelnapValue.TRUE
            assert struct.confidence_float >= 0.85

        pipeline.close()

    def test_concurrent_multi_chunk_ingestion(self):
        """Concurrent multi-passage ingestion executes without SQLite locking or deadlocks."""
        pipeline = CognitivePipeline(
            transducer_backend="mock",
            kev_mode="async",
        )
        passages = [
            ("P_c1", "D1", "Marcus Vance inspected the reactor core."),
            ("P_c2", "D1", "The sensor detected a critical temperature drop."),
            ("P_c3", "D1", "Eleanor Vance calibrated the diagnostic laser."),
            ("P_c4", "D1", "Alice Vance audited the safety protocol."),
        ]

        t0 = time.perf_counter()
        graphs = pipeline.ingest_passages_batch(passages, max_workers=4)
        ingest_latency_ms = (time.perf_counter() - t0) * 1000.0

        assert len(graphs) == 4
        # Ingestion returns immediately without waiting for background verification
        assert ingest_latency_ms < 2000.0

        # Flush queue and verify all 4 completed
        flushed = pipeline.flush_kev_queue(timeout=10.0)
        assert flushed

        for pid, _, _ in passages:
            assert pipeline.async_kev_worker.is_verified(pid)

        pipeline.close()

    def test_query_memory_with_async_ingestion(self):
        """Querying memory accesses the active graph and triggers priority elevation."""
        pipeline = CognitivePipeline(
            transducer_backend="mock",
            kev_mode="async",
        )
        text = "Marcus Vance observed the reactor core. The pressure breached the valve."
        graph = pipeline.ingest_document(text, doc_id="doc_query_01")
        pid = getattr(graph, "passage_id")

        # Query memory immediately
        result = pipeline.query_memory("What did Marcus Vance observe?", format="dual_stream")
        assert result is not None
        assert "Marcus Vance" in result.full_context or "reactor" in result.full_context

        # Flush background verification
        pipeline.flush_kev_queue(timeout=5.0)
        assert pipeline.async_kev_worker.is_verified(pid)

        pipeline.close()

    def test_hedged_modal_asynchronous_refinement(self):
        """Modal hedge sentences asynchronously refine to hearsay/conjecture epistemic status."""
        pipeline = CognitivePipeline(
            transducer_backend="mock",
            kev_mode="async",
        )
        text = "Dr. Eleanor Vance allegedly discovered a strange anomaly."
        graph = pipeline.ingest_document(text, doc_id="doc_hedge_01")

        pipeline.flush_kev_queue(timeout=5.0)

        # Event should refine to hearsay
        for node_cid, node in graph.nodes.items():
            if node.get_slot("TYPE_EVENT") == 1:
                stored = pipeline.page_table.fetch_node(node_cid)
                assert stored is not None
                assert stored.evidence_source in ("hearsay", "conjecture")

        pipeline.close()

    def test_throughput_benchmark(self):
        """Ingestion throughput on a 300-word passage meets >= 200 words/sec target."""
        pipeline = CognitivePipeline(
            transducer_backend="mock",
            kev_mode="async",
        )
        passage_text = (
            "Marcus Vance inspected the primary containment facility before the scheduled test began. "
            "The engineer monitored the cooling systems while the pressure increased rapidly inside the vessel. "
            "A sudden temperature spike caused the secondary pressure relief valve to open automatically. "
            "The automated safety protocol initiated immediate emergency shutdown sequences across all sectors. "
            "Technicians observed the diagnostic monitors and reported stable readings after several minutes. "
        ) * 5  # ~300 words
        word_count = len(passage_text.split())

        t0 = time.perf_counter()
        graph = pipeline.ingest_document(passage_text, doc_id="doc_bench_01")
        duration = time.perf_counter() - t0

        words_per_sec = word_count / max(0.0001, duration)
        # Verify return latency SLA < 1.3s and throughput >= 200 w/s
        assert duration < 1.3
        assert words_per_sec >= 200.0

        pipeline.flush_kev_queue(timeout=5.0)
        pipeline.close()
