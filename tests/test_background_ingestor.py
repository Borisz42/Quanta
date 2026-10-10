"""Unit tests for Dynamic Multi-Scale Ingestion Phase 5: Background Completion (§multi_scale_plan.md).

Verifies:
1. Off-switch leaves no worker threads running (QUANTA_BACKGROUND_INGESTION=0 / background.enabled=False).
2. Items are processed in priority order (calibrated filter score ranking + top-priority elevation).
3. Pause/resume works under simulated foreground load (Policy BG-B foreground lock and BG-C slot polling).
4. Upgrades land in the in-memory QuantaGraph and SQLite passage_ingestion_state table.
5. End-to-end integration via CognitivePipeline.
"""

from __future__ import annotations

import os
import threading
import time
from typing import List
import pytest

from config.multi_scale_config import MultiScaleConfig
from core.asg import QuantaGraph, QuantaNode
from core.binary_node import BelnapValue, BinaryNodeTable
from memory.page_table import PageTable
from memory.passage_store import PassageRecord, PassageStore
from models.kev_async_worker import AsyncKevVerificationQueue, BackgroundIngestor, VerificationTask
from models.kev_engine import KevDecisionEngine, MockKevEngine
from parser.skeleton_transducer import MockSkeletonTransducer
from pipeline.cognitive_pipeline import CognitivePipeline


@pytest.fixture
def clean_page_table():
    """Isolated in-memory SQLite PageTable."""
    return PageTable(db_path=":memory:")


@pytest.fixture
def clean_passage_store():
    """Isolated in-memory SQLite PassageStore."""
    return PassageStore(db_path=":memory:")


@pytest.fixture
def binary_table():
    """Fresh BinaryNodeTable."""
    return BinaryNodeTable()


@pytest.fixture
def mock_transducer():
    """Mock skeleton transducer in co-decoded mode."""
    return MockSkeletonTransducer(mode="co_decoded")


class TestBackgroundIngestor:
    """Test suite for BackgroundIngestor and AsyncKevVerificationQueue Phase 5 features."""

    def test_off_switch_leaves_no_threads_running(self):
        """When disabled, the worker pool starts no threads and pipeline clean teardown leaves 0 threads."""
        # 1. Direct queue with enabled=False
        queue = BackgroundIngestor(
            enabled=False,
            auto_start=True,
        )
        assert not queue.is_running
        assert not queue.enabled
        assert queue.pending_count == 0

        # Enqueuing when disabled returns None and spawns no threads
        task = queue.enqueue_deferred(
            passage_id="P_disabled",
            doc_id="D1",
            text="Marcus Vance inspected the reactor.",
        )
        assert task is None
        assert not queue.is_running

        queue.stop(wait=True)
        assert not queue.is_running

        # 2. CognitivePipeline with QUANTA_BACKGROUND_INGESTION=0
        old_env = os.environ.get("QUANTA_BACKGROUND_INGESTION")
        try:
            os.environ["QUANTA_BACKGROUND_INGESTION"] = "0"
            pipeline = CognitivePipeline(
                transducer_backend="mock",
                multi_scale_config=MultiScaleConfig(runtime_overrides={"background.enabled": False}),
            )
            assert not pipeline.background_enabled
            assert pipeline.background_ingestor is None

            pipeline.reset()
            pipeline.close()

            # Confirm no leftover AsyncKevWorker or BackgroundIngestor threads
            active_names = [t.name for t in threading.enumerate() if t.is_alive()]
            assert not any("AsyncKevWorker" in name for name in active_names)
        finally:
            if old_env is not None:
                os.environ["QUANTA_BACKGROUND_INGESTION"] = old_env
            else:
                os.environ.pop("QUANTA_BACKGROUND_INGESTION", None)

    def test_items_processed_in_priority_order(self, mock_transducer):
        """Items are popped in descending order of filter_score; prioritize() elevates to top."""
        queue = BackgroundIngestor(
            transducer=mock_transducer,
            max_workers=1,
            auto_start=False,  # Keep stopped to inspect heap ordering
            enabled=True,
        )

        # Enqueue with various filter scores: 0.15, 0.95, 0.40, 0.80
        t1 = queue.enqueue_deferred("P_low", "D1", "Text low", filter_score=0.15)
        t2 = queue.enqueue_deferred("P_urgent", "D1", "Text urgent", filter_score=0.95)
        t3 = queue.enqueue_deferred("P_med", "D1", "Text med", filter_score=0.40)
        t4 = queue.enqueue_deferred("P_high", "D1", "Text high", filter_score=0.80)

        # Inspect priority values (smaller key = higher priority)
        assert t2.priority < t4.priority < t3.priority < t1.priority

        # Enqueue very low score task and prioritize it
        t_tail = queue.enqueue_deferred("P_tail", "D1", "Text tail", filter_score=0.01)
        assert t_tail.priority > t1.priority

        prioritized = queue.prioritize("P_tail")
        assert prioritized
        assert t_tail.priority == -1e9

        # Start worker and flush
        queue.start()
        flushed = queue.flush(timeout=5.0)
        assert flushed
        assert queue.completed_count == 5
        assert queue.is_verified("P_urgent")
        assert queue.is_verified("P_tail")

        queue.stop(wait=True)
        assert not queue.is_running

    def test_pause_resume_under_foreground_load_policy_b(self, mock_transducer):
        """Pause Policy BG-B: background worker pauses during in-process foreground execution."""
        queue = BackgroundIngestor(
            transducer=mock_transducer,
            pause_policy="foreground_lock",
            max_workers=1,
            auto_start=True,
            enabled=True,
        )
        assert queue.is_running

        # Mark foreground active before enqueuing
        queue.set_foreground_active(True)
        assert queue.is_foreground_active()

        task = queue.enqueue_deferred(
            passage_id="P_paused_b",
            doc_id="D1",
            text="Marcus Vance measured the cooling array.",
            filter_score=0.90,
        )
        assert task is not None

        # Give small window: task should not complete while foreground is held
        time.sleep(0.1)
        assert queue.get_status("P_paused_b") in ("pending", "processing")
        assert not queue.is_verified("P_paused_b")

        # Release foreground lock
        queue.set_foreground_active(False)
        assert not queue.is_foreground_active()

        # Flush queue
        flushed = queue.flush(timeout=5.0)
        assert flushed
        assert queue.is_verified("P_paused_b")

        queue.stop(wait=True)
        assert not queue.is_running

    def test_pause_resume_under_foreground_load_policy_c(self, mock_transducer):
        """Pause Policy BG-C: background worker pauses when busy slots >= threshold."""
        simulated_slots = [2]  # Busy slots = 2 (above threshold 1)

        def check_slots() -> int:
            return simulated_slots[0]

        queue = BackgroundIngestor(
            transducer=mock_transducer,
            pause_policy="slot_polling",
            busy_slot_threshold=1,
            slot_checker=check_slots,
            max_workers=1,
            auto_start=True,
            enabled=True,
        )
        assert queue.is_running

        task = queue.enqueue_deferred(
            passage_id="P_paused_c",
            doc_id="D1",
            text="Eleanor Vance energized the laser chamber.",
            filter_score=0.90,
        )
        assert task is not None

        # Verify worker paused
        time.sleep(0.1)
        assert queue.get_status("P_paused_c") in ("pending", "processing")
        assert not queue.is_verified("P_paused_c")

        # Simulate slots becoming idle (0 busy slots)
        simulated_slots[0] = 0

        # Flush queue
        flushed = queue.flush(timeout=5.0)
        assert flushed
        assert queue.is_verified("P_paused_c")

        queue.stop(wait=True)
        assert not queue.is_running

    def test_upgrade_lands_in_graph_and_passage_state_table(
        self,
        clean_page_table,
        clean_passage_store,
        binary_table,
        mock_transducer,
    ):
        """Upgrades land in QuantaGraph and transition passage_ingestion_state to FULL."""
        graph = QuantaGraph()
        # Create coarse macro passage node
        coarse_node = QuantaNode(
            anchor="passage:P_macro_01",
            literal="passage:P_macro_01",
            truth_status="TRUE",
            confidence=1.0,
            passage_id="P_macro_01",
            node_type="coarse_passage",
        )
        coarse_node.compute_canonical_cid()
        graph.add_node(coarse_node)

        # Register P_macro_01 in PassageStore as PENDING
        clean_passage_store.add_passage(
            PassageRecord(
                passage_id="P_macro_01",
                doc_id="D1",
                char_span=(0, 80),
                text="Dr. Eleanor Vance synthesized the polymer in the chamber.",
                granularity="MICRO",
            )
        )
        clean_passage_store.set_ingestion_state("P_macro_01", "PENDING")
        assert clean_passage_store.get_ingestion_state("P_macro_01") == "PENDING"

        queue = BackgroundIngestor(
            transducer=mock_transducer,
            page_table=clean_page_table,
            binary_table=binary_table,
            passage_store=clean_passage_store,
            max_workers=1,
            auto_start=True,
            enabled=True,
        )

        task = queue.enqueue_deferred(
            passage_id="P_macro_01",
            doc_id="D1",
            text="Dr. Eleanor Vance synthesized the polymer in the chamber.",
            filter_score=0.88,
            graph=graph,
            concept_codes=[101, 102],
        )
        assert task is not None

        flushed = queue.flush(timeout=5.0)
        assert flushed
        assert queue.is_verified("P_macro_01")

        # Verify PassageStore transitioned to FULL
        assert clean_passage_store.get_ingestion_state("P_macro_01") == "FULL"

        # Verify graph contains fine-grained upgraded nodes
        nodes = graph.get_nodes_for_passage("P_macro_01")
        assert len(nodes) >= 2  # coarse node + at least 1 event + 1 entity
        subgraph = queue.get_subgraph("P_macro_01")
        assert subgraph is not None
        assert len(subgraph.nodes) >= 2

        queue.stop(wait=True)
        assert not queue.is_running

    def test_pipeline_integration_deferred_and_flush(self):
        """CognitivePipeline enqueues deferred chunks under hot_transduce and upgrades in background."""
        cfg = MultiScaleConfig(
            runtime_overrides={
                "background.enabled": True,
                "background.pause_policy": "none",
                "fast_path.mode": "hot_transduce",
                "fast_path.hot_transduce_n": 1,
            }
        )

        pipeline = CognitivePipeline(
            transducer_backend="mock",
            multi_scale_config=cfg,
        )
        assert pipeline.background_enabled
        assert pipeline.background_ingestor is not None
        assert pipeline.background_ingestor.is_running

        prompt = (
            "Passage 1: Marcus Vance inspected the reactor core.\n\n"
            "Passage 2: Eleanor Vance monitored the radiation levels.\n\n"
            "Passage 3: The coolant pressure dropped rapidly.\n\n"
            "Question: Who monitored the radiation levels?"
        )

        res = pipeline.answer_long_context(prompt, cold_start=True)
        assert res.mode == "hot_transduce"
        # Since hot_transduce_n=1 and there are multiple chunks/units, deferred_units >= 1
        assert res.units_transduced == 1
        assert res.deferred_units >= 1

        # Background worker flushes
        flushed = pipeline.flush_background_queue(timeout=5.0)
        assert flushed

        # Verify all deferred chunks transitioned to FULL
        all_passages = pipeline.passage_store.get_all_passages()
        assert len(all_passages) >= 2
        for p in all_passages:
            assert pipeline.passage_store.get_ingestion_state(p.passage_id) == "FULL"

        pipeline.close()
        assert pipeline.background_ingestor is None
