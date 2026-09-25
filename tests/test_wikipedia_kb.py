"""Tests for Section 7: Phase 10 Global Knowledge Base Mount (Wikipedia & Wikidata Pre-Compilation).

Verifies:
- Task 7.1: Streaming Wikidata Dump Ingester (stream_wikidata_dump, parse_raw_wikidata_entity)
            and offline 100,000-entity synthetic slice generator (generate_synthetic_wikidata_slice).
- Task 7.2: Entity & Property Mapper (WikidataEntityMapper) converting Q-IDs and P-ID triples
            into QuantaNodes with 1024-D QuantaVectors and 256-bit BLAKE3 CIDs.
- Task 7.3: Persistent SQLite Compiler (WikidataSqliteCompiler) creating `nodes`, `aliases`,
            and `triples` tables with B-Tree indexes.
- Task 7.4: Global Knowledge Base Mount Interface (GlobalKnowledgeBase) in read-only mode (`mode=ro`),
            integrating with PageTable and bounding memory via ActiveCanvas (M <= 512 nodes, <= 128 KB).
- Task 7.5: Encyclopedic multi-hop benchmark tests executing in < 10 ms with 0.000000% hallucination.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from core.asg import QuantaNode
from core.types import QuantaVector
from data.wikidata_ingester import (
    SALIENT_PROPERTIES,
    WikidataEntityMapper,
    WikidataSqliteCompiler,
    classify_entity_category,
    generate_synthetic_wikidata_slice,
    parse_raw_wikidata_entity,
    stream_wikidata_dump,
)
from memory.global_kb import GlobalKnowledgeBase
from memory.page_table import ActiveCanvas, PageTable


# =============================================================================
# Task 7.1: Streaming Wikidata Dump & Synthetic Slice Generator Tests
# =============================================================================

class TestWikidataIngestion:
    """Tests for Task 7.1: Streaming Wikidata Dump Ingestion & Synthetic Generation."""

    def test_parse_raw_wikidata_entity(self):
        """Verifies normalization of raw Wikidata JSON records."""
        raw_entity = {
            "id": "Q42",
            "labels": {"en": {"value": "Douglas Adams"}},
            "descriptions": {"en": {"value": "English author and humorist"}},
            "aliases": {
                "en": [
                    {"value": "Douglas Noel Adams"},
                    {"value": "DNA"},
                ]
            },
            "claims": {
                "P31": [
                    {"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}
                ],
                "P19": [
                    {"mainsnak": {"datavalue": {"value": {"id": "Q350"}}}}
                ],
                "P27": [
                    {"mainsnak": {"datavalue": {"value": {"id": "Q145"}}}}
                ],
                "P800": [
                    {"mainsnak": {"datavalue": {"value": {"id": "Q25169"}}}}
                ],
            },
        }

        parsed = parse_raw_wikidata_entity(raw_entity, filter_salient=True)
        assert parsed is not None
        assert parsed["qid"] == "Q42"
        assert parsed["label"] == "Douglas Adams"
        assert parsed["category"] == "human"
        assert "DNA" in parsed["aliases"]
        assert "Douglas Noel Adams" in parsed["aliases"]
        assert parsed["claims"]["P31"] == ["Q5"]
        assert parsed["claims"]["P19"] == ["Q350"]
        assert parsed["claims"]["P800"] == ["Q25169"]

    def test_streaming_dump_reader(self, tmp_path: Path):
        """Verifies streaming dump reader on compressed JSONL files with array brackets and commas."""
        dump_file = tmp_path / "sample_dump.jsonl.gz"
        entities = [
            {"id": "Q101", "labels": {"en": {"value": "Item 1"}}, "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}]}},
            {"id": "Q102", "labels": {"en": {"value": "Item 2"}}, "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q515"}}}}]}},
            {"id": "Q103", "labels": {"en": {"value": "Item 3"}}, "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q43229"}}}}]}},
        ]

        with gzip.open(dump_file, "wt", encoding="utf-8") as f:
            f.write("[\n")
            for i, ent in enumerate(entities):
                comma = "," if i < len(entities) - 1 else ""
                f.write(json.dumps(ent) + comma + "\n")
            f.write("]\n")

        streamed = list(stream_wikidata_dump(dump_file, filter_salient=False))
        assert len(streamed) == 3
        assert [e["qid"] for e in streamed] == ["Q101", "Q102", "Q103"]
        assert streamed[0]["category"] == "human"
        assert streamed[1]["category"] == "location"
        assert streamed[2]["category"] == "organization"

    def test_synthetic_slice_generation_deterministic(self):
        """Verifies that generate_synthetic_wikidata_slice is deterministic and reproducible."""
        slice_1 = list(generate_synthetic_wikidata_slice(num_entities=100, seed=42))
        slice_2 = list(generate_synthetic_wikidata_slice(num_entities=100, seed=42))

        assert len(slice_1) == 100
        assert len(slice_2) == 100
        for e1, e2 in zip(slice_1, slice_2):
            assert e1["qid"] == e2["qid"]
            assert e1["label"] == e2["label"]
            assert e1["claims"] == e2["claims"]

        # Verify presence of core landmark entities
        qids = {e["qid"] for e in slice_1}
        assert "Q42" in qids    # Douglas Adams
        assert "Q25169" in qids # HHGTTG
        assert "Q350" in qids   # Cambridge
        assert "Q145" in qids   # United Kingdom
        assert "Q84" in qids    # London


# =============================================================================
# Task 7.2: Entity & Property Mapper Tests
# =============================================================================

class TestEntityMapper:
    """Tests for Task 7.2: Mapping Wikidata Entities to 1024-D QuantaNodes."""

    def test_map_human_entity_to_node(self):
        """Verifies QuantaNode construction, vector slots, and BLAKE3 CID for human entity."""
        mapper = WikidataEntityMapper()
        entity = {
            "qid": "Q42",
            "label": "Douglas Adams",
            "description": "English author and humorist",
            "aliases": ["DNA"],
            "category": "human",
            "claims": {
                "P31": ["Q5"],
                "P19": ["Q350"],
                "P800": ["Q25169"],
            },
        }

        node = mapper.map_entity_to_node(entity)
        assert isinstance(node, QuantaNode)
        assert node.anchor == "Douglas Adams"
        assert node.literal["qid"] == "Q42"
        assert node.literal["category"] == "human"

        # Check Band 0 substantive prime NSM_SOMEONE (slot 2)
        assert node.get_slot(2) == 1  # TRUE
        # Check Band 3 concept slots
        assert node.get_slot(460) == 1  # CN_Q077_WORKER
        assert node.get_slot(463) == 1  # CN_Q080_FAMILY

        # Check relation edges
        assert "PLACE_OF_BIRTH" in node.edges
        assert node.edges["PLACE_OF_BIRTH"] == ["Q350"]
        assert "NOTABLE_WORK" in node.edges
        assert node.edges["NOTABLE_WORK"] == ["Q25169"]

        # Check deterministic 256-bit BLAKE3 CID
        cid1 = node.compute_cid()
        assert len(cid1) == 64
        assert all(c in "0123456789abcdef" for c in cid1)

        # Mapping identical entity again yields identical CID
        node2 = mapper.map_entity_to_node(entity)
        assert node2.compute_cid() == cid1

    def test_map_location_entity_to_node(self):
        """Verifies QuantaNode vector slots for location entity."""
        mapper = WikidataEntityMapper()
        entity = {
            "qid": "Q350",
            "label": "Cambridge",
            "description": "University city",
            "aliases": ["Cambridge, UK"],
            "category": "location",
            "claims": {
                "P31": ["Q515"],
                "P17": ["Q145"],
            },
        }

        node = mapper.map_entity_to_node(entity)
        assert node.anchor == "Cambridge"
        # Check Band 1 spatial frame VAL_LOCATION_SLOT (slot 136)
        assert node.get_slot(136) == 1
        # Check Band 3 city/country slots
        assert node.get_slot(464) == 1  # Country
        assert node.get_slot(474) == 1  # City
        assert node.edges["COUNTRY"] == ["Q145"]


# =============================================================================
# Task 7.3: Persistent SQLite Compiler Tests
# =============================================================================

class TestSqliteCompiler:
    """Tests for Task 7.3: Compiling SQLite Knowledge Base."""

    def test_compile_database(self, tmp_path: Path):
        """Compiles a small batch of synthetic entities and validates schema and indexes."""
        db_path = tmp_path / "test_wikipedia_quanta.db"
        compiler = WikidataSqliteCompiler()

        entities = list(generate_synthetic_wikidata_slice(num_entities=500, seed=123))
        stats = compiler.compile_database(entities, db_path, batch_size=200)

        assert stats["total_nodes"] == 500
        assert stats["total_triples"] > 500
        assert stats["total_aliases"] > 0
        assert db_path.exists()

        # Connect to verify SQLite schema directly
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        # Check nodes table
        cur.execute("SELECT COUNT(*) FROM nodes")
        assert cur.fetchone()[0] == 500

        # Check Douglas Adams
        cur.execute("SELECT qid, label, category, vector_bytes FROM nodes WHERE qid = 'Q42'")
        row = cur.fetchone()
        assert row is not None
        assert row[1] == "Douglas Adams"
        assert row[2] == "human"
        assert len(row[3]) == 256  # 256-byte packed quaternary vector

        # Check aliases
        cur.execute("SELECT alias FROM aliases WHERE qid = 'Q42'")
        aliases = [r[0] for r in cur.fetchall()]
        assert "DNA" in aliases

        # Check triples
        cur.execute("SELECT property_name, object_qid FROM triples WHERE subject_qid = 'Q42'")
        triples = dict(cur.fetchall())
        assert triples.get("PLACE_OF_BIRTH") == "Q350"
        assert triples.get("NOTABLE_WORK") == "Q25169"

        # Check indexes exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        index_names = {r[0] for r in cur.fetchall()}
        assert "idx_nodes_label_lower" in index_names
        assert "idx_nodes_cid" in index_names
        assert "idx_triples_sub_prop" in index_names
        conn.close()


# =============================================================================
# Task 7.4: Global Knowledge Base Mount Interface & ActiveCanvas Tests
# =============================================================================

class TestGlobalKnowledgeBaseMount:
    """Tests for Task 7.4: Global Knowledge Base Read-Only Mount & Memory Bounds."""

    @pytest.fixture
    def test_kb(self, tmp_path: Path) -> GlobalKnowledgeBase:
        """Fixture that compiles and mounts a 2,000-entity KB."""
        db_path = tmp_path / "fixture_wikipedia_quanta.db"
        compiler = WikidataSqliteCompiler()
        entities = list(generate_synthetic_wikidata_slice(num_entities=2000, seed=42))
        compiler.compile_database(entities, db_path, batch_size=500)

        kb = GlobalKnowledgeBase(db_path=db_path)
        yield kb
        kb.close()

    def test_read_only_enforcement(self, test_kb: GlobalKnowledgeBase):
        """Verifies database is mounted in mode=ro and rejects write operations."""
        assert test_kb.is_mounted()

        # Attempt write operation; should raise sqlite3.OperationalError
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            test_kb._conn.execute(
                "INSERT INTO nodes (qid, cid, label, label_lower, category, description, vector_bytes, payload) "
                "VALUES ('Q999999', 'cid_test', 'Fake', 'fake', 'other', '', X'00', '{}')"
            )

    def test_lookup_entity_exact_and_alias(self, test_kb: GlobalKnowledgeBase):
        """Verifies entity lookups by exact label, lowercase label, alias, and Q-ID."""
        # 1. Exact name
        nodes = test_kb.lookup_entity("Douglas Adams")
        assert len(nodes) >= 1
        node = nodes[0]
        assert node.anchor == "Douglas Adams"
        assert node.literal["qid"] == "Q42"

        # 2. Case-insensitive lowercase
        nodes_lower = test_kb.lookup_entity("douglas adams")
        assert len(nodes_lower) >= 1
        assert nodes_lower[0].anchor == "Douglas Adams"

        # 3. Alias lookup
        nodes_alias = test_kb.lookup_entity("DNA")
        assert len(nodes_alias) >= 1
        assert any(n.literal["qid"] == "Q42" for n in nodes_alias)

        # 4. Q-ID direct lookup
        nodes_qid = test_kb.lookup_entity("Q42")
        assert len(nodes_qid) == 1
        assert nodes_qid[0].anchor == "Douglas Adams"

        # 5. BLAKE3 CID lookup
        cid = node.cid
        fetched_node = test_kb.get_entity_by_cid(cid)
        assert fetched_node is not None
        assert fetched_node.literal["qid"] == "Q42"

    def test_o1_vram_active_canvas_bounded_memory(self, test_kb: GlobalKnowledgeBase):
        """Verifies that querying into ActiveCanvas enforces strict O(1) memory bound (M <= 64)."""
        capacity = 64
        canvas = ActiveCanvas(capacity=capacity)
        test_kb.active_canvas = canvas

        # Page 200 distinct entities into the 64-node active canvas
        for i in range(200):
            qid = f"Q{200000 + i}"
            test_kb.query_and_page_to_canvas(qid)

        # Strict assertion: Canvas size NEVER exceeds capacity M=64
        assert len(canvas) <= capacity
        assert canvas.stats()["active_nodes"] <= capacity

        # Strict memory bound: 64 nodes * 256 bytes = 16,384 bytes (< 128 KB)
        memory_bytes = canvas.active_memory_bytes()
        assert memory_bytes <= capacity * 256
        assert memory_bytes <= 128 * 1024  # strictly <= 128 KB

        # Verify LRU eviction occurred
        assert canvas.evictions > 0
        assert canvas.evictions >= 100

    def test_page_table_seamless_integration(self, test_kb: GlobalKnowledgeBase):
        """Verifies PageTable transparently resolves unknown CIDs via mounted GlobalKnowledgeBase."""
        pt = PageTable(":memory:")
        pt.mount_global_kb(test_kb)

        # Get an entity CID from the GlobalKnowledgeBase
        adams_node = test_kb.get_entity_by_qid("Q42")
        assert adams_node is not None
        cid = adams_node.cid

        # Node is NOT in PageTable's local storage
        assert not pt.has_node(cid)

        # Calling pt.fetch_node(cid) should seamlessly fallback to global KB
        resolved_node = pt.fetch_node(cid)
        assert resolved_node is not None
        assert resolved_node.anchor == "Douglas Adams"
        assert resolved_node.literal["qid"] == "Q42"
        assert resolved_node.get_slot(2) == 1  # NSM_SOMEONE preserved

        # Batch fetch_nodes should also resolve seamlessly
        batch_nodes = pt.fetch_nodes([cid, "non_existent_cid_999"])
        assert len(batch_nodes) == 2
        assert batch_nodes[0] is not None
        assert batch_nodes[0].anchor == "Douglas Adams"
        assert batch_nodes[1] is None

    def test_concurrent_multi_thread_queries(self, test_kb: GlobalKnowledgeBase):
        """Verifies thread-safety with 10 concurrent reader threads executing queries."""
        errors: List[Exception] = []

        def worker_task(thread_id: int):
            try:
                for _ in range(50):
                    nodes = test_kb.lookup_entity("Douglas Adams")
                    assert len(nodes) >= 1
                    triples = test_kb.get_triples(subject_qid="Q42")
                    assert len(triples) >= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors encountered: {errors}"


# =============================================================================
# Task 7.5: Encyclopedic Multi-Hop Benchmark Tests (< 10 ms & 0% Hallucination)
# =============================================================================

class TestMultiHopBenchmark:
    """Tests for Task 7.5: Encyclopedic Multi-Hop Traversal Accuracy & Sub-10ms Benchmark."""

    @pytest.fixture(scope="class")
    def scaled_100k_kb(self, tmp_path_factory) -> GlobalKnowledgeBase:
        """Creates and compiles an offline 100,000-entity SQLite knowledge base."""
        tmp_dir = tmp_path_factory.mktemp("kb_100k")
        db_path = tmp_dir / "wikipedia_quanta_100k.db"

        print("\n[Section 7 Scaling] Compiling 100,000-entity knowledge base...")
        t0 = time.perf_counter()

        compiler = WikidataSqliteCompiler()
        entities = generate_synthetic_wikidata_slice(num_entities=100_000, seed=42)
        stats = compiler.compile_database(entities, db_path, batch_size=10_000)

        t_compile = time.perf_counter() - t0
        print(
            f"[Section 7 Scaling] Compiled {stats['total_nodes']:,} nodes, "
            f"{stats['total_triples']:,} triples into {stats['file_size_bytes'] / (1024*1024):.1f} MB "
            f"in {t_compile:.2f} s ({stats['nodes_per_second']:,.0f} nodes/s)"
        )

        kb = GlobalKnowledgeBase(db_path=db_path)
        yield kb
        kb.close()

    def test_landmark_multi_hop_traversal_accuracy(self, scaled_100k_kb: GlobalKnowledgeBase):
        """Verifies 1-hop, 2-hop, 3-hop, and 4-hop queries execute with 100% ground-truth accuracy."""
        kb = scaled_100k_kb

        # Hop 1: Creative Work -> AUTHOR
        # "The Hitchhiker's Guide to the Galaxy" (Q25169) -> AUTHOR -> Douglas Adams (Q42)
        hop1 = kb.multi_hop_query("The Hitchhiker's Guide to the Galaxy", ["AUTHOR"])
        assert len(hop1) == 1
        assert hop1[0].anchor == "Douglas Adams"
        assert hop1[0].literal["qid"] == "Q42"

        # Hop 2: Creative Work -> AUTHOR -> PLACE_OF_BIRTH
        # HHGTTG -> Douglas Adams -> Cambridge (Q350)
        hop2 = kb.multi_hop_query("The Hitchhiker's Guide to the Galaxy", ["AUTHOR", "PLACE_OF_BIRTH"])
        assert len(hop2) == 1
        assert hop2[0].anchor == "Cambridge"
        assert hop2[0].literal["qid"] == "Q350"

        # Hop 3: Creative Work -> AUTHOR -> PLACE_OF_BIRTH -> COUNTRY
        # HHGTTG -> Douglas Adams -> Cambridge -> United Kingdom (Q145)
        hop3 = kb.multi_hop_query("The Hitchhiker's Guide to the Galaxy", ["AUTHOR", "PLACE_OF_BIRTH", "COUNTRY"])
        assert len(hop3) == 1
        assert hop3[0].anchor == "United Kingdom"
        assert hop3[0].literal["qid"] == "Q145"

        # Hop 4: Creative Work -> AUTHOR -> PLACE_OF_BIRTH -> COUNTRY -> CAPITAL
        # "What is the capital of the country where the author of The Hitchhiker's Guide was born?"
        # HHGTTG -> Douglas Adams -> Cambridge -> United Kingdom -> London (Q84)
        hop4 = kb.multi_hop_query("The Hitchhiker's Guide to the Galaxy", ["AUTHOR", "PLACE_OF_BIRTH", "COUNTRY", "CAPITAL"])
        assert len(hop4) == 1
        assert hop4[0].anchor == "London"
        assert hop4[0].literal["qid"] == "Q84"

        # Turing: Alan Turing -> PLACE_OF_BIRTH -> COUNTRY -> CAPITAL
        turing_cap = kb.multi_hop_query("Alan Turing", ["PLACE_OF_BIRTH", "COUNTRY", "CAPITAL"])
        assert len(turing_cap) == 1
        assert turing_cap[0].anchor == "London"

        # Curie: Marie Curie -> PLACE_OF_BIRTH -> COUNTRY -> CAPITAL
        # Warsaw -> Poland -> Warsaw
        curie_cap = kb.multi_hop_query("Marie Curie", ["PLACE_OF_BIRTH", "COUNTRY", "CAPITAL"])
        assert len(curie_cap) == 1
        assert curie_cap[0].anchor == "Warsaw"

    def test_100k_node_query_latency_sub_10ms_benchmark(self, scaled_100k_kb: GlobalKnowledgeBase):
        """Strict benchmark: executes 100 4-hop queries over 100k nodes and asserts latency < 10.0 ms."""
        kb = scaled_100k_kb

        # Warmup query
        _ = kb.multi_hop_query("The Hitchhiker's Guide to the Galaxy", ["AUTHOR", "PLACE_OF_BIRTH", "COUNTRY", "CAPITAL"])

        latencies_ms: List[float] = []
        queries = [
            ("The Hitchhiker's Guide to the Galaxy", ["AUTHOR", "PLACE_OF_BIRTH", "COUNTRY", "CAPITAL"]),
            ("Douglas Adams", ["NOTABLE_WORK", "AUTHOR", "PLACE_OF_BIRTH"]),
            ("Alan Turing", ["PLACE_OF_BIRTH", "COUNTRY", "CAPITAL"]),
            ("Marie Curie", ["PLACE_OF_BIRTH", "COUNTRY"]),
            ("Albert Einstein", ["PLACE_OF_BIRTH", "COUNTRY", "CAPITAL"]),
        ]

        trials_per_query = 20
        for entity, path in queries:
            for _ in range(trials_per_query):
                t0 = time.perf_counter()
                results = kb.multi_hop_query(entity, path)
                dt_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(dt_ms)
                assert len(results) >= 1

        min_lat = float(np.min(latencies_ms))
        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        p99_lat = float(np.percentile(latencies_ms, 99))

        print(
            f"\n[Task 7.5 Benchmark] 100,000-Node Encyclopedic Multi-Hop Query Latency: "
            f"min={min_lat:.3f} ms, mean={mean_lat:.3f} ms, p95={p95_lat:.3f} ms, p99={p99_lat:.3f} ms"
        )

        # STRICT REQUIREMENT: Traverses graph in < 10.0 ms
        assert min_lat < 10.0, f"Min multi-hop latency must be < 10.0 ms, got {min_lat:.3f} ms"
        assert mean_lat < 10.0, f"Mean multi-hop latency must be < 10.0 ms, got {mean_lat:.3f} ms"
        assert p95_lat < 10.0, f"P95 multi-hop latency must be < 10.0 ms, got {p95_lat:.3f} ms"
