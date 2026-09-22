"""Unit and Performance Benchmark Tests for Memory-Mapped Lexical Grounder and Ingestion Pipeline (Section 2).

Verifies:
1. MmapLexicalGrounder initialization, zero-copy binary loading, and index resolution.
2. Binary equivalence: Zero Hamming drift (dH = 0) between MmapLexicalGrounder and SQLite ConceptNet.
3. Sub-0.1ms single-concept resolution latency benchmark (target < 0.01 ms).
4. Pre-warmed GBNF grammar hash caching and payload verification.
5. High-throughput ingestion benchmark over 10,000 words (assert throughput > 150 words/sec).
6. 3-stage asynchronous pipelining fidelity and Merkle fold validation.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import List
import pytest
import numpy as np

from core.types import QuantaVector
from parser.asg_compiler import ASGCompiler
from parser.lexical_grounder import ConceptNetLexicalGrounder
from parser.mmap_grounder import MmapLexicalGrounder
from parser.unsloth_transducer import MockUnslothTransducer, UnslothTransducer
from pipeline.cognitive_pipeline import CognitivePipeline


@pytest.fixture
def mmap_grounder() -> MmapLexicalGrounder:
    grounder = MmapLexicalGrounder.get_default()
    assert grounder.is_available(), "MmapLexicalGrounder could not load binary codebook or index."
    return grounder


@pytest.fixture
def sqlite_grounder() -> ConceptNetLexicalGrounder:
    return ConceptNetLexicalGrounder.get_default()


# ---------------------------------------------------------------------------
# 1. Initialization and Index Verification
# ---------------------------------------------------------------------------

def test_mmap_grounder_initialization(mmap_grounder: MmapLexicalGrounder):
    """Verify that MmapLexicalGrounder initializes with expected concepts and index entries."""
    stats = mmap_grounder.stats()
    assert stats["num_concepts"] >= 50000, f"Expected >= 50,000 concepts, got {stats['num_concepts']}"
    assert stats["index_keys"] >= 100000, f"Expected >= 100,000 index keys, got {stats['index_keys']}"

    # Verify basic concept resolutions
    v_slang = mmap_grounder.resolve_concept_vector("slang")
    assert v_slang is not None
    assert isinstance(v_slang, QuantaVector)

    c_slang = mmap_grounder.resolve_concept("slang", pos="n")
    assert c_slang is not None
    assert c_slang.lemma == "slang"
    assert "CN_Q055_LANGUAGE" in c_slang.active_slots or "CN_Q127_DIALECT" in c_slang.active_slots


# ---------------------------------------------------------------------------
# 2. Binary Equivalence & Zero Hamming Drift (dH = 0)
# ---------------------------------------------------------------------------

def test_mmap_vs_sqlite_fidelity(mmap_grounder: MmapLexicalGrounder, sqlite_grounder: ConceptNetLexicalGrounder):
    """Verify exact binary and quaternary equivalence between MmapLexicalGrounder and SQLite."""
    test_concepts = [
        ("slang", "n"),
        ("mineral", "n"),
        ("medicine", "n"),
        ("zoology", "n"),
        ("chemistry", "n"),
        ("organic compound", "n"),
        ("anatomy", "n"),
        ("computing", "n"),
        ("biology", "n"),
        ("person", "n"),
        ("physics", "n"),
        ("botany", "n"),
        ("music", "n"),
    ]

    for lemma, pos in test_concepts:
        mmap_res = mmap_grounder.resolve_concept(lemma, pos=pos)
        sqlite_res = sqlite_grounder.resolve_concept(lemma, pos=pos)

        assert mmap_res is not None, f"Mmap grounder missed concept: {lemma} ({pos})"
        assert sqlite_res is not None, f"SQLite grounder missed concept: {lemma} ({pos})"

        # Compare packed bytes
        mmap_bytes = mmap_res.vector.to_bytes()
        sqlite_bytes = sqlite_res.vector.to_bytes()
        assert mmap_bytes == sqlite_bytes, f"Binary mismatch for '{lemma}': mmap bytes != sqlite bytes"

        # Compare Hamming distance
        mmap_arr = mmap_res.vector.to_numpy()
        sqlite_arr = sqlite_res.vector.to_numpy()
        hamming_dist = int(np.count_nonzero(mmap_arr != sqlite_arr))
        assert hamming_dist == 0, f"Hamming drift detected for '{lemma}': dH = {hamming_dist}"


# ---------------------------------------------------------------------------
# 3. Sub-0.1ms Latency Benchmark (Target < 0.01 ms)
# ---------------------------------------------------------------------------

def test_single_concept_lookup_latency(mmap_grounder: MmapLexicalGrounder):
    """Benchmark 10,000 lookups and assert mean latency is strictly < 0.1 ms."""
    lookup_pool = [
        "slang", "mineral", "medicine", "chemistry", "anatomy",
        "computing", "biology", "person", "physics", "botany",
        "music", "legal", "animal", "plant", "food",
        "organic compound", "county seat", "historical",
    ]

    num_iterations = 10000
    t0 = time.perf_counter()
    for i in range(num_iterations):
        target = lookup_pool[i % len(lookup_pool)]
        vec = mmap_grounder.resolve_concept_vector(target)
        assert vec is not None

    total_time = time.perf_counter() - t0
    mean_latency_ms = (total_time / num_iterations) * 1000.0

    print(f"\n[BENCHMARK] 10,000 Mmap Lookups: Total = {total_time:.4f}s, Mean = {mean_latency_ms:.4f} ms ({mean_latency_ms * 1000.0:.2f} µs)")
    assert mean_latency_ms < 0.1, f"Mean latency {mean_latency_ms:.4f} ms exceeded threshold 0.1 ms"


# ---------------------------------------------------------------------------
# 4. GBNF Grammar State Machine & Pre-Warmed Caching
# ---------------------------------------------------------------------------

def test_gbnf_grammar_hash_and_prewarming():
    """Verify that UnslothTransducer caches grammar hash and injects it into payloads."""
    transducer = UnslothTransducer(fallback_to_mock=True)

    assert hasattr(transducer, "grammar_hash")
    assert isinstance(transducer.grammar_hash, str)
    assert len(transducer.grammar_hash) == 64  # SHA-256 hex string

    # Verify warm_grammar method
    warmed = transducer.warm_grammar()
    assert warmed is True
    assert transducer._grammar_warmed is True

    # Verify payload contains grammar_hash
    payload = transducer._build_payload("Test chunk for grammar injection")
    assert "grammar_hash" in payload
    assert payload["grammar_hash"] == transducer.grammar_hash
    assert "grammar_hash" in payload["extra_body"]
    assert payload["extra_body"]["grammar_hash"] == transducer.grammar_hash


# ---------------------------------------------------------------------------
# 5. High-Throughput Ingestion Benchmark (> 150 words/sec over 10,000 words)
# ---------------------------------------------------------------------------

def test_end_to_end_ingestion_throughput_10k_words():
    """Ingest a 10,000-word narrative through CognitivePipeline and assert throughput > 150 words/sec."""
    mock_transducer = MockUnslothTransducer()
    compiler = ASGCompiler(mmap_grounder=MmapLexicalGrounder.get_default())
    pipeline = CognitivePipeline(
        transducer=mock_transducer,
        compiler=compiler,
        canvas_capacity=512,
    )

    # Build a 10,000-word narrative corpus consisting of 50 structured paragraphs
    paragraph_template = (
        "Dr. Marcus Vance pressurized the argon cylinder inside the test chamber at noon. "
        "The automated diagnostic sensors verified that the pressure valve held stable under extreme thermal stress. "
        "Alice carefully inspected the primary coolant manifold and confirmed that the liquid nitrogen lines were unobstructed. "
        "Meanwhile, Bob monitored the secondary electrical substation to prevent voltage fluctuations across the laboratory grid. "
        "The telemetry showed no anomalous sensor readings during the entire forty-minute operational test phase. "
        "Every engineer signed the compliance logbook before the scheduled maintenance shutdown. "
    )
    words_per_para = len(paragraph_template.split())
    num_paras = int(np.ceil(10000 / words_per_para))  # ~115 paragraphs to reach >= 10,000 words
    narrative_text = "\n\n".join(f"Chapter paragraph {i+1}. {paragraph_template}" for i in range(num_paras))
    total_words = len(narrative_text.split())

    assert total_words >= 10000, f"Expected >= 10,000 words, got {total_words}"

    # Ingest narrative through pipelined process_narrative
    t0 = time.perf_counter()
    episode_graphs, chapter_fold = pipeline.process_narrative(
        narrative_text,
        chapter_id="benchmark_chapter",
        validate=False,  # Measure pure ingestion & compilation throughput
    )
    elapsed = time.perf_counter() - t0

    throughput_wps = total_words / elapsed

    print(f"\n[BENCHMARK] Ingestion of {total_words:,} words: {elapsed:.2f}s -> {throughput_wps:.1f} words/sec")
    print(f"[BENCHMARK] Generated {len(episode_graphs)} episode graphs, chapter root CID: {chapter_fold.cid[:16]}...")

    assert len(episode_graphs) > 0
    assert chapter_fold is not None
    assert throughput_wps > 150.0, f"Ingestion throughput {throughput_wps:.1f} w/s fell below 150 w/s threshold"


# ---------------------------------------------------------------------------
# 6. Asynchronous Multi-Stage Pipelining Validation
# ---------------------------------------------------------------------------

def test_async_process_narrative():
    """Verify that process_narrative_async runs concurrently without errors."""
    async def _run():
        mock_transducer = MockUnslothTransducer()
        pipeline = CognitivePipeline(
            transducer=mock_transducer,
            canvas_capacity=512,
        )

        narrative = (
            "Alice investigated the anomalous energy spike in sector four. "
            "She discovered that the plasma conduit had cracked due to structural fatigue.\n\n"
            "Bob arrived with the repair kit and replaced the fractured ceramic seal. "
            "They recalibrated the magnetic containment field together."
        )

        graphs, chapter_node = await pipeline.process_narrative_async(
            narrative,
            chapter_id="async_test_ch",
            validate=False,
        )

        assert len(graphs) >= 1
        assert chapter_node is not None
        assert chapter_node.cid is not None
        assert len(pipeline.active_canvas) <= pipeline.active_canvas.capacity

    asyncio.run(_run())
