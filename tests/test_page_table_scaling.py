"""Tests for Phase 6: Virtual Page-Table Attention & O(1) VRAM Memory Offloading.

Verifies:
- 6.1: Disk-backed PageTable storing QuantaNode instances and string interning tables with SQLite persistence.
- 6.2: SIMD-accelerated bitwise Hamming distance top-K search with 100% mathematical fidelity to QuantaVector.
- 6.3: ActiveCanvas fixed-buffer manager (M=64 to 512 nodes) with strict O(1) bounds, LRU eviction, and pinning.
- 6.4: SemanticPageFaultHandler: page faults on dormant nodes, cache hits on hot nodes, neighbor prefetching, and vector query paging.
- 6.5: 100,000-node scaling benchmark: verifies active memory stays strictly <= 512 nodes and top-K retrieval executes in < 5 ms.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import numpy as np
import pytest

from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, RegisterValue, StructuralValue, pack_quaternary_array
from memory.page_table import (
    ActiveCanvas,
    PageTable,
    SemanticPageFaultHandler,
    SimdHammingIndex,
    StringInternTable,
    batch_quaternary_hamming,
    topk_hamming_search,
)


# =============================================================================
# Helper Utilities
# =============================================================================

def _create_synthetic_node(
    index: int,
    anchor: str = "cn:en:entity (n)",
    edges: dict | None = None,
    literal: dict | None = None,
) -> QuantaNode:
    """Generates a reproducible synthetic QuantaNode with 1024-dim quaternary vector."""
    # Seeded pseudo-random pattern based on index
    rng = np.random.RandomState(index)
    # Sparse slots (e.g. 10-30 active slots per node)
    slots = {}
    active_indices = rng.choice(1024, size=20, replace=False)
    for idx in active_indices:
        slots[int(idx)] = int(rng.randint(1, 4))

    return QuantaNode(
        vector=slots,
        edges=edges,
        anchor=f"{anchor}_{index % 50}",
        literal=literal or {"id": f"Node_{index}", "index": index},
    )


# =============================================================================
# Checklist 6.1: Disk-Backed PageTable & String Interning Tables
# =============================================================================

class TestDiskBackedPageTable:
    """Tests for 6.1: SQLite persistence, QuantaNode CRUD, and StringInternTable."""

    def test_string_interning_deduplication(self, tmp_path: Path):
        """Verifies that frequent strings are deduplicated into unique integer IDs."""
        db_path = tmp_path / "intern_test.db"
        pt = PageTable(db_path)

        id1 = pt.interner.intern("cn:en:eleanor_vance (n)")
        id2 = pt.interner.intern("cn:en:eleanor_vance (n)")
        id3 = pt.interner.intern("VAL_X1_AGENT")

        assert id1 is not None
        assert id1 == id2, "Identical strings must yield the exact same interned ID"
        assert id1 != id3, "Distinct strings must have distinct IDs"

        assert pt.interner.resolve(id1) == "cn:en:eleanor_vance (n)"
        assert pt.interner.resolve(id3) == "VAL_X1_AGENT"

        pt.close()

        # Reopen database and verify persistent resolution
        pt2 = PageTable(db_path)
        assert pt2.interner.resolve(id1) == "cn:en:eleanor_vance (n)"
        assert pt2.interner.intern("cn:en:eleanor_vance (n)") == id1
        pt2.close()

    def test_node_disk_persistence_round_trip(self, tmp_path: Path):
        """Verifies storing nodes to disk, closing DB, and reconstructing with full fidelity."""
        db_path = tmp_path / "persistence_test.db"
        pt = PageTable(db_path)

        node = QuantaNode(
            vector={
                "TYPE_HUMAN": 1,
                "TYPE_ANIMATE": 1,
                "ROLE_AGENT_CAPABLE": 1,
                "VAR_SLOT_X0": int(RegisterValue.BOUND_LOCAL),
            },
            anchor="cn:en:scientist (n)",
            literal={"name": "Dr. Vance", "title": "Lead Chemist"},
            edges={"VAL_X1_AGENT": ["cid_child_123"]},
        )
        orig_cid = node.compute_cid()
        pt.store_node(node)
        pt.close()

        # Reopen and fetch
        pt2 = PageTable(db_path)
        fetched = pt2.fetch_node(orig_cid)
        assert fetched is not None, "Node must be retrieved from disk storage"
        assert fetched.cid == orig_cid
        assert fetched.anchor == "cn:en:scientist (n)"
        assert fetched.literal == {"name": "Dr. Vance", "title": "Lead Chemist"}
        assert fetched.edges == {"VAL_X1_AGENT": ["cid_child_123"]}
        assert fetched.vector["TYPE_HUMAN"] == 1
        assert fetched.vector["VAR_SLOT_X0"] == int(RegisterValue.BOUND_LOCAL)
        assert fetched.compute_cid() == orig_cid
        pt2.close()

    def test_batch_store_and_fetch_nodes(self, tmp_path: Path):
        """Verifies storing and fetching batches of nodes in SQLite."""
        pt = PageTable(tmp_path / "batch_test.db")
        nodes = [_create_synthetic_node(i) for i in range(25)]

        cids = pt.store_nodes(nodes)
        assert len(cids) == 25
        assert pt.count_nodes() == 25

        fetched_nodes = pt.fetch_nodes(cids)
        assert len(fetched_nodes) == 25
        for orig, fetched in zip(nodes, fetched_nodes):
            assert fetched is not None
            assert fetched.cid == orig.cid
            assert fetched.vector.hamming_distance(orig.vector) == 0

        # Test deletion
        assert pt.delete_node(cids[0]) is True
        assert pt.has_node(cids[0]) is False
        assert pt.count_nodes() == 24
        pt.close()

    def test_merkle_subgraph_storage_compatibility(self, tmp_path: Path):
        """Verifies PageTable preserves full Phase 5 Merkle sub-graph storage semantics."""
        pt = PageTable(tmp_path / "subgraph_test.db")
        graph = QuantaGraph()
        n1 = QuantaNode(vector={"TYPE_EVENT": 1}, anchor="cn:en:investigate (v)")
        n2 = QuantaNode(vector={"TYPE_HUMAN": 1}, anchor="cn:en:investigator (n)")
        graph.add_node(n1, set_as_root=True)
        graph.add_node(n2)
        merkle_root = graph.compute_merkle_root()

        pt.store_subgraph(merkle_root, graph)
        retrieved_graph = pt.retrieve_subgraph(merkle_root)
        assert retrieved_graph.compute_merkle_root() == merkle_root
        assert len(retrieved_graph.nodes) == 2
        pt.close()


# =============================================================================
# Checklist 6.2: SIMD-Accelerated Bitwise Hamming Distance Top-K Search
# =============================================================================

class TestSimdHammingSearch:
    """Tests for 6.2: Mathematical fidelity of SWAR popcount and SIMD top-K search."""

    def test_swar_popcount_mathematical_fidelity(self):
        """Asserts that SWAR SIMD popcount exactly matches QuantaVector.hamming_distance across 100 tests."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            # Generate two random 1024-dim quaternary vectors in {0, 1, 2, 3}
            v1_arr = rng.randint(0, 4, size=1024, dtype=np.uint8)
            v2_arr = rng.randint(0, 4, size=1024, dtype=np.uint8)

            qv1 = QuantaVector(v1_arr)
            qv2 = QuantaVector(v2_arr)
            expected_dist = qv1.hamming_distance(qv2)

            # SWAR SIMD calculation
            u64_1 = np.frombuffer(qv1.to_bytes(), dtype=np.uint64).reshape(1, 32)
            u64_2 = np.frombuffer(qv2.to_bytes(), dtype=np.uint64).reshape(32)

            calculated_dist = int(batch_quaternary_hamming(u64_1, u64_2)[0])
            assert calculated_dist == expected_dist, (
                f"SWAR popcount mismatch: got {calculated_dist}, expected {expected_dist}"
            )

    def test_simd_index_topk_ranking(self):
        """Verifies SimdHammingIndex correctly ranks and retrieves top-K nearest neighbors."""
        index = SimdHammingIndex()
        base_vec = QuantaVector({"TYPE_HUMAN": 1, "ROLE_AGENT_CAPABLE": 1, "NSM_SEE": 1})

        # Create neighbors with known increasing differences
        nodes = []
        for i in range(20):
            v = base_vec.copy()
            # Flip i additional slots
            for j in range(i):
                v[100 + j] = 2
            cid = f"test_cid_{i:03d}"
            index.add(cid, v)
            nodes.append((cid, i))

        # Query with base_vec
        results = index.search(base_vec, top_k=5)
        assert len(results) == 5
        # The closest should be test_cid_000 with distance 0
        assert results[0][0] == "test_cid_000"
        assert results[0][1] == 0
        # Ascending distance order
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_simd_index_dynamic_reallocation_and_removal(self):
        """Verifies SimdHammingIndex capacity expansion and O(1) removal."""
        index = SimdHammingIndex(initial_capacity=128)
        cids = [f"cid_{i}" for i in range(300)]
        zero_vec = QuantaVector.zeros()

        # Batch add 300 vectors (exceeding initial 128 capacity)
        index.add_batch([(cid, zero_vec) for cid in cids])
        assert len(index) == 300
        assert "cid_150" in index

        # Test removal
        assert index.remove("cid_150") is True
        assert len(index) == 299
        assert "cid_150" not in index
        assert index.remove("nonexistent_cid") is False

        index.clear()
        assert len(index) == 0


# =============================================================================
# Checklist 6.3: ActiveCanvas Fixed-Buffer Manager (M = 64 to 512)
# =============================================================================

class TestActiveCanvas:
    """Tests for 6.3: Fixed memory buffer, LRU eviction policy, and node pinning."""

    def test_capacity_strictly_bounded(self):
        """Verifies ActiveCanvas never exceeds its fixed capacity M=64."""
        capacity = 64
        canvas = ActiveCanvas(capacity=capacity)

        # Insert 200 nodes
        for i in range(200):
            node = _create_synthetic_node(i)
            canvas.put(node)
            assert len(canvas) <= capacity, f"Canvas exceeded capacity at step {i}: {len(canvas)}"

        assert len(canvas) == capacity
        # Memory strictly bounded: 64 * 256 bytes = 16,384 bytes = 16 KB
        assert canvas.active_memory_bytes() == capacity * 256

    def test_lru_eviction_order(self):
        """Verifies least recently used nodes are evicted first."""
        canvas = ActiveCanvas(capacity=3)
        n1 = _create_synthetic_node(1)
        n2 = _create_synthetic_node(2)
        n3 = _create_synthetic_node(3)
        n4 = _create_synthetic_node(4)

        canvas.put(n1)
        canvas.put(n2)
        canvas.put(n3)

        # Access n1 to make it most recently used: recency order becomes [n2, n3, n1]
        _ = canvas.get(n1.cid)

        # Inserting n4 must evict n2 (the LRU unpinned node)
        evicted = canvas.put(n4)
        assert evicted is not None
        assert evicted.cid == n2.cid
        assert n2.cid not in canvas
        assert n1.cid in canvas
        assert n3.cid in canvas
        assert n4.cid in canvas

    def test_node_pinning_protection(self):
        """Verifies pinned nodes are shielded from LRU eviction."""
        canvas = ActiveCanvas(capacity=3)
        n1 = _create_synthetic_node(1)
        n2 = _create_synthetic_node(2)
        n3 = _create_synthetic_node(3)
        n4 = _create_synthetic_node(4)

        canvas.put(n1)
        canvas.put(n2)
        canvas.put(n3)

        # Pin n1 and n2
        canvas.pin(n1.cid)
        canvas.pin(n2.cid)
        assert canvas.is_pinned(n1.cid)

        # Inserting n4 must evict n3, because n1 and n2 are protected
        evicted = canvas.put(n4)
        assert evicted is not None
        assert evicted.cid == n3.cid
        assert n1.cid in canvas
        assert n2.cid in canvas
        assert n4.cid in canvas

        # Unpin n1
        canvas.unpin(n1.cid)
        assert not canvas.is_pinned(n1.cid)


# =============================================================================
# Checklist 6.4: Semantic Page-Fault Handler
# =============================================================================

class TestSemanticPageFaultHandler:
    """Tests for 6.4: Semantic page faults, cache hits, prefetching, and query paging."""

    def test_page_fault_on_dormant_node_and_cache_hit(self):
        """Verifies that accessing an unpaged node increments page faults and subsequent access hits cache."""
        pt = PageTable(":memory:")
        canvas = ActiveCanvas(capacity=4)
        handler = SemanticPageFaultHandler(canvas, pt)

        node = _create_synthetic_node(10)
        cid = pt.store_node(node)

        # Node not in canvas initially
        assert cid not in canvas
        assert handler.page_fault_count == 0

        # First access: Page Fault!
        retrieved = handler.access(cid)
        assert retrieved.cid == cid
        assert cid in canvas
        assert handler.page_fault_count == 1
        assert canvas.stats()["hits"] == 0
        assert canvas.stats()["misses"] == 1

        # Second access: Cache Hit!
        retrieved_again = handler.access(cid)
        assert retrieved_again.cid == cid
        assert handler.page_fault_count == 1  # No new page fault
        assert canvas.stats()["hits"] == 1

    def test_subgraph_relation_neighbor_prefetching(self):
        """Verifies that prefetch_depth automatically pages outbound relation children."""
        pt = PageTable(":memory:")
        canvas = ActiveCanvas(capacity=10)
        handler = SemanticPageFaultHandler(canvas, pt)

        # Construct parent event and child entities
        agent = _create_synthetic_node(1, anchor="cn:en:agent (n)")
        patient = _create_synthetic_node(2, anchor="cn:en:patient (n)")
        agent_cid = pt.store_node(agent)
        patient_cid = pt.store_node(patient)

        event = _create_synthetic_node(
            3,
            anchor="cn:en:synthesize (v)",
            edges={"VAL_X1_AGENT": [agent_cid], "VAL_X2_PATIENT": [patient_cid]},
        )
        event_cid = pt.store_node(event)

        # Access event with prefetch_depth=1
        _ = handler.access(event_cid, prefetch_depth=1)

        # Event, Agent, and Patient should all now be present in canvas!
        assert event_cid in canvas
        assert agent_cid in canvas
        assert patient_cid in canvas

    def test_query_and_page_nearest(self):
        """Verifies that query_and_page finds top-K semantic nodes and loads them into canvas."""
        pt = PageTable(":memory:")
        canvas = ActiveCanvas(capacity=5)
        handler = SemanticPageFaultHandler(canvas, pt)

        target_node = QuantaNode(
            vector={"TYPE_HUMAN": 1, "ROLE_AGENT_CAPABLE": 1, "NSM_DO": 1},
            anchor="cn:en:eleanor_vance (n)",
        )
        target_cid = pt.store_node(target_node)

        # Add other distractors
        for i in range(10):
            pt.store_node(_create_synthetic_node(i + 100))

        # Query using target_node's vector
        paged_nodes = handler.query_and_page(target_node.vector, top_k=3)
        assert len(paged_nodes) == 3
        assert paged_nodes[0].cid == target_cid
        assert target_cid in canvas


# =============================================================================
# Checklist 6.5: 100,000-Node Scaling Benchmark
# =============================================================================

class TestPageTable100kScaling:
    """Tests for 6.5: Ingesting 100,000 nodes, verifying strictly <= 512 active nodes,

    and asserting sub-5ms top-K retrieval latency on CPU (> 20 M nodes/sec throughput).
    """

    def test_100k_node_scaling_and_sub_5ms_retrieval(self, tmp_path: Path):
        """Ingests 100,000 nodes; asserts O(1) canvas bounds and sub-5ms SIMD top-K search."""
        db_path = tmp_path / "scaling_100k.db"
        pt = PageTable(db_path)
        canvas = ActiveCanvas(capacity=512)
        handler = SemanticPageFaultHandler(canvas, pt)

        N = 100000
        print(f"\n[Phase 6.5 Benchmark] Generating and ingesting {N} synthetic QuantaNodes...")

        # Generate 100,000 packed vectors directly in batch chunks for rapid test setup
        rng = np.random.RandomState(1337)
        chunk_size = 10000
        all_cids: list[str] = []

        t_start_ingest = time.perf_counter()
        for c in range(N // chunk_size):
            chunk_rows = []
            chunk_vecs = []
            now = time.time()
            for i in range(chunk_size):
                global_idx = c * chunk_size + i
                # Generate 256 random bytes representing 1024-dim quaternary vector
                vec_bytes = rng.bytes(256)
                cid = f"{global_idx:016x}" + "0" * 48
                anchor_id = pt.interner.intern(f"concept_{global_idx % 200}")
                literal_str = json.dumps({"idx": global_idx})
                edges_str = "{}"
                chunk_rows.append((cid, vec_bytes, anchor_id, literal_str, None, edges_str, now, 0, now))
                chunk_vecs.append((cid, vec_bytes))
                all_cids.append(cid)

            # Insert directly via PageTable transaction
            with pt._lock, pt._conn:
                pt._conn.executemany(
                    """
                    INSERT INTO nodes 
                    (cid, vector_bytes, anchor_id, literal, parent_cid, edges, created_at, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    chunk_rows,
                )
            pt.vector_index.add_batch(chunk_vecs)

        t_end_ingest = time.perf_counter()
        ingest_ms = (t_end_ingest - t_start_ingest) * 1000
        print(f"[Phase 6.5 Benchmark] Ingested {N} nodes in {ingest_ms:.2f} ms ({N / (t_end_ingest - t_start_ingest):.0f} nodes/sec)")

        assert pt.count_nodes() == N
        assert len(pt.vector_index) == N

        # 1. Verify SIMD top-K search over 100,000 keys is strictly < 5 ms
        query_bytes = rng.bytes(256)
        query_u64 = np.frombuffer(query_bytes, dtype=np.uint64)

        # Warmup
        _ = pt.vector_index.search(query_bytes, top_k=10)

        search_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            matches = pt.vector_index.search(query_bytes, top_k=10)
            t1 = time.perf_counter()
            search_times.append((t1 - t0) * 1000)

        mean_latency_ms = float(np.mean(search_times))
        min_latency_ms = float(np.min(search_times))
        throughput_mps = (N / (min_latency_ms / 1000)) / 1e6

        print(
            f"[Phase 6.5 Benchmark] Top-10 SIMD Hamming search over {N} keys: "
            f"mean={mean_latency_ms:.3f} ms, min={min_latency_ms:.3f} ms. "
            f"Throughput: {throughput_mps:.2f} M nodes/sec"
        )
        assert len(matches) == 10
        # STRICT REQUIREMENT: Retrieval completes in < 5.0 ms
        assert min_latency_ms < 5.0, f"Top-K search must complete in < 5.0 ms, got {min_latency_ms:.3f} ms"

        # 2. Verify ActiveCanvas memory remains strictly <= 512 nodes during intensive access
        print(f"[Phase 6.5 Benchmark] Simulating access pattern across 5,000 cold/hot queries...")
        test_access_cids = rng.choice(all_cids, size=5000, replace=True)

        for cid in test_access_cids:
            node = handler.access(cid)
            assert node is not None
            assert len(canvas) <= 512, (
                f"ActiveCanvas strictly bounded to <= 512 nodes; violated with {len(canvas)}"
            )

        # Physical memory footprint strictly <= 512 * 256 bytes = 128 KB
        active_bytes = canvas.active_memory_bytes()
        assert active_bytes <= 512 * 256, f"Memory footprint must be <= 128 KB, got {active_bytes} bytes"

        stats = canvas.stats()
        print(
            f"[Phase 6.5 Benchmark] Canvas Stats: active_nodes={stats['active_nodes']}, "
            f"memory={stats['memory_bytes'] / 1024:.1f} KB, hits={stats['hits']}, "
            f"misses={stats['misses']}, evictions={stats['evictions']}, hit_rate={stats['hit_rate']:.2%}"
        )

        pt.close()
