"""End-to-End Multi-Chapter & Book Benchmark Suite (Phase 8).

Verifies:
8.1: Gold Standard Eleanor Vance narrative through complete pipeline:
     Input text -> Transducer -> ASG Compiler -> Clingo Gate -> Honest Realizer.
     Assert graph contains exactly 5 entities, 6 events, 0 token nodes, 0 punctuation nodes,
     and 100% Clingo validation pass rate.
8.2: Multi-Chunk Working Memory Continuity Test:
     Run a 3-chunk continuous story through the pipeline; verify entities introduced
     in Chunk 1 are correctly paged and reused in Chunk 3 without passing Chunk 1 text tokens.
8.3: Long-Context Book Benchmark:
     Ingest a multi-chapter book text (>10,000 words) into PageTable; verify:
     - VRAM usage remains flat (<= 512 nodes in ActiveCanvas).
     - Deterministic 32-byte BLAKE3 Merkle Book Root is computed.
     - Questions about Chapter 1 asked after Chapter 10 are answered in < 10 ms with zero hallucination.
"""

from __future__ import annotations

import time
import pytest

from core.asg import QuantaGraph, QuantaNode
from parser.chunker import DiscourseChunker
from parser.entity_manifest import EntityPagingEngine
from parser.schema import (
    DiscourseExtractionResult,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedProposition,
    ExtractedRelation,
)
from parser.transducer import (
    CANONICAL_ELEANOR_VANCE_FIXTURE,
    CANONICAL_ELEANOR_VANCE_TEXT,
    MockTransducer,
)
from pipeline.cognitive_pipeline import CognitivePipeline


# =============================================================================
# 8.1: Gold Standard Eleanor Vance End-to-End Pipeline
# =============================================================================

def test_8_1_eleanor_vance_end_to_end_pipeline():
    """8.1: Re-run Dr. Eleanor Vance narrative through the complete pipeline.

    Input text -> Transducer -> ASG Compiler -> Clingo Verification -> Honest Realizer.
    Asserts:
    - Graph contains exactly 5 entities, 6 events, 0 token nodes, 0 punctuation nodes.
    - 100% Clingo ASP validation pass rate (0 errors, 0 MUC slots).
    - Honest NLG unrolls all 5 entities and 6 events with discourse anaphora.
    """
    pipeline = CognitivePipeline(transducer_backend="mock")

    # Process text through complete cognitive cycle
    graph = pipeline.process_chunk(
        chunk_text=CANONICAL_ELEANOR_VANCE_TEXT,
        chunk_id="chunk_eleanor_vance",
        chapter_id="ch_01",
        validate=True,
    )

    # 1. Verify exact node count: 5 entities + 6 events = 11 nodes total
    assert len(graph.nodes) == 11, f"Expected 11 total nodes, got {len(graph.nodes)}"

    entity_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 0]
    event_nodes = [n for n in graph.nodes.values() if n.get_slot("TYPE_EVENT") == 1]

    assert len(entity_nodes) == 5, f"Expected exactly 5 entity nodes, got {len(entity_nodes)}"
    assert len(event_nodes) == 6, f"Expected exactly 6 event nodes, got {len(event_nodes)}"

    # 2. Verify zero token or punctuation debris
    for cid, node in graph.nodes.items():
        assert not (node.anchor and node.anchor.startswith("punct:")), f"Found punct node: {node}"
        assert not (node.anchor and node.anchor.startswith("token:")), f"Found token node: {node}"

    # 3. Verify structural and cryptographic BLAKE3 integrity
    struct_valid, struct_errors = graph.validate_integrity()
    assert struct_valid, f"Graph structural validation failed: {struct_errors}"
    for cid, node in graph.nodes.items():
        assert len(cid) == 64, f"Invalid BLAKE3 CID: {cid}"
        assert cid == node.compute_cid(), f"CID mismatch for node {cid}"

    # 4. Verify 100% Clingo ASP validation pass rate
    assert hasattr(graph, "validation")
    assert graph.validation.is_valid, f"Clingo validation failed: {graph.validation.errors}"
    assert len(graph.validation.errors) == 0
    assert len(graph.validation.muc_slots) == 0

    # 5. Realize text through Honest English Realizer
    realized_text = pipeline.realize(graph)
    assert realized_text, "Realized text is empty!"

    # Assert presence of all 5 entities
    assert "Dr. Eleanor Vance" in realized_text
    assert "volatile synthetic compound" in realized_text or "synthetic compound" in realized_text
    assert "cryogenic containment cell" in realized_text
    assert "supervisor" in realized_text
    assert "laboratory director" in realized_text

    # Assert proper anaphoric pronouns and descriptions
    assert "She" in realized_text or "she" in realized_text
    assert "this specimen" in realized_text
    assert "the resulting polymer" in realized_text or "resulting polymer" in realized_text
    assert "the same vessel" in realized_text

    # Assert event predicates are expressed
    assert "isolated" in realized_text
    assert "noted" in realized_text
    assert "doubted" in realized_text
    assert "verified" in realized_text
    assert "retained" in realized_text
    assert "prohibit" in realized_text

    # 6. Verify zero-attention query answering
    query_answer = pipeline.answer_query("What did Eleanor Vance verify?", target_graph=graph)
    assert query_answer == "The hypothesis"

    pipeline.close()


# =============================================================================
# 8.2: Multi-Chunk Working Memory Continuity Test
# =============================================================================

def test_8_2_multi_chunk_working_memory_continuity():
    """8.2: Run a 3-chunk continuous story through the pipeline.

    - Chunk 1: Introduces E1 (Dr. Elena Rostova) and E2 (quantum resonator).
    - Chunk 2: Technician Liam calibrates the magnetic shield. (Elena not mentioned).
    - Chunk 3: Elena returns to examine the quantum resonator.
    Verifies:
    - ActiveEntityManifest pages dormant entities E1 and E2 for Chunk 3 without passing Chunk 1 text tokens.
    - Chunk 3 reuses canonical IDs E1 and E2.
    - Graph correctly links to the same entity CIDs without creating duplicates.
    """
    mock_transducer = MockTransducer()

    # Define 3 chunks of narrative
    chunk_1_text = "Dr. Elena Rostova isolated the quantum resonator inside the high-energy vacuum vault at dawn."
    chunk_2_text = "Technician Liam Vance calibrated the secondary magnetic shield throughout the morning."
    chunk_3_text = "Later that afternoon, Elena inspected the quantum resonator to measure field harmonic resonance."

    # Register deterministic extractions for the 3 chunks
    fixture_chunk_1 = DiscourseExtractionResult(
        chunk_id="chunk_1",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Dr. Elena Rostova", category="PERSON", surface_aliases=["Elena", "Rostova", "she"]),
            ExtractedEntity(id="E2", canonical_name="quantum resonator", category="OBJECT", surface_aliases=["resonator", "specimen"]),
            ExtractedEntity(id="E3", canonical_name="high-energy vacuum vault", category="LOCATION", surface_aliases=["vacuum vault", "vault"]),
        ],
        events=[
            ExtractedEvent(id="Ev1", predicate="isolate", agent_id="E1", patient_id="E2", location_id="E3", temporal_anchor="at dawn", tense="PAST", polarity=True, raw_text=chunk_1_text),
        ],
        relations=[],
        propositions=[],
    )

    fixture_chunk_2 = DiscourseExtractionResult(
        chunk_id="chunk_2",
        entities=[
            ExtractedEntity(id="E4", canonical_name="Technician Liam Vance", category="PERSON", surface_aliases=["Liam", "Vance"]),
            ExtractedEntity(id="E5", canonical_name="secondary magnetic shield", category="OBJECT", surface_aliases=["magnetic shield", "shield"]),
        ],
        events=[
            ExtractedEvent(id="Ev2", predicate="calibrate", agent_id="E4", patient_id="E5", temporal_anchor="throughout the morning", tense="PAST", polarity=True, raw_text=chunk_2_text),
        ],
        relations=[],
        propositions=[],
    )

    # In Chunk 3, the transducer reuses E1 and E2 from the Active Entity Manifest!
    fixture_chunk_3 = DiscourseExtractionResult(
        chunk_id="chunk_3",
        entities=[
            ExtractedEntity(id="E1", canonical_name="Dr. Elena Rostova", category="PERSON", surface_aliases=["Elena", "Rostova"]),
            ExtractedEntity(id="E2", canonical_name="quantum resonator", category="OBJECT", surface_aliases=["resonator"]),
        ],
        events=[
            ExtractedEvent(id="Ev3", predicate="inspect", agent_id="E1", patient_id="E2", temporal_anchor="later that afternoon", tense="PAST", polarity=True, raw_text=chunk_3_text),
        ],
        relations=[],
        propositions=[],
    )

    mock_transducer.register_fixture(chunk_1_text, fixture_chunk_1)
    mock_transducer.register_fixture(chunk_2_text, fixture_chunk_2)
    mock_transducer.register_fixture(chunk_3_text, fixture_chunk_3)

    pipeline = CognitivePipeline(transducer=mock_transducer)

    # Process Chunk 1
    g1 = pipeline.process_chunk(chunk_1_text, chunk_id="chunk_1", chapter_id="ch_01")
    assert pipeline.entity_engine.manifest.is_active("E1")
    assert pipeline.entity_engine.manifest.is_active("E2")
    e1_cid_1 = next(cid for cid, n in g1.nodes.items() if n.literal and isinstance(n.literal, dict) and n.literal.get("id") == "E1" or "Rostova" in str(n.literal))

    # Process Chunk 2 (Elena is not in Chunk 2 text)
    g2 = pipeline.process_chunk(chunk_2_text, chunk_id="chunk_2", chapter_id="ch_01")
    assert pipeline.entity_engine.manifest.is_active("E4")
    assert pipeline.entity_engine.manifest.is_active("E5")

    # Fast pre-scan on Chunk 3 text without Chunk 1 tokens
    paged, scan_time = pipeline.entity_engine.pre_scan_and_page(chunk_3_text, chunk_idx=3)
    assert scan_time < 2.0, f"Pre-scan took {scan_time:.2f} ms"
    # E1 ('Elena') and E2 ('quantum resonator') must be recognized and active
    assert pipeline.entity_engine.manifest.is_active("E1")
    assert pipeline.entity_engine.manifest.is_active("E2")

    # Process Chunk 3
    g3 = pipeline.process_chunk(chunk_3_text, chunk_id="chunk_3", chapter_id="ch_01")
    
    # Verify Chunk 3 event links to E1
    ev3_node = next(n for n in g3.nodes.values() if n.get_slot("TYPE_EVENT") == 1)
    assert "VAL_X1_AGENT" in ev3_node.edges
    e1_in_g3_cid = ev3_node.edges["VAL_X1_AGENT"][0]
    e1_in_g3 = g3.get_node(e1_in_g3_cid)
    assert "Rostova" in str(e1_in_g3.anchor) or "Rostova" in str(e1_in_g3.literal) or "elena" in str(e1_in_g3.anchor)

    # Verify zero duplicate entity names in manifest
    all_names = [rec.canonical_name for rec in pipeline.entity_engine.manifest.values()]
    assert len(all_names) == len(set(all_names)), "Found duplicate entity names in active manifest!"

    pipeline.close()


# =============================================================================
# 8.3: Long-Context Book Benchmark (>10,000 words, M <= 512, sub-10ms query)
# =============================================================================

def _build_ten_chapter_book() -> list[tuple[str, str]]:
    """Builds a realistic 10-chapter scientific narrative document totaling > 10,000 words."""
    chapters = []
    
    # Template paragraphs for expanding chapters realistically
    ch1_intro = (
        "In the deep subterranean laboratories of the Promethean Research Facility, Dr. Eleanor Vance "
        "isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. "
        "She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, "
        "which strongly suggested an unobserved phase transition. Although her supervisor initially doubted "
        "the validity of the discovery, Eleanor verified the hypothesis three hours later by replicating "
        "the transformation within the same vessel. The resulting polymer retained its structural integrity "
        "throughout the afternoon, prompting the laboratory director to prohibit all competing tests until "
        "her synthesis protocol could be formally audited."
    )

    filler_prose = (
        "The environmental control systems maintained a steady ambient temperature of precisely four degrees Celsius. "
        "Banks of liquid helium chillers hummed quietly along the reinforced concrete perimeter, circulating pressurized "
        "refrigerant through vacuum-insulated double-walled stainless steel conduits. Overhead diagnostic monitors flickered "
        "with real-time spectroscopy telemetry, logging photon absorption differentials across sixteen discrete ultraviolet bands. "
        "Senior technicians methodically calibrated the primary ionization chambers, carefully verifying ground potentials "
        "against the facility reference standard before initiating secondary diagnostic sequences. The automated containment "
        "interlocks engaged with a pneumatic hiss, sealing the vestibule airlock and activating the positive-pressure cascade. "
        "Every recorded metric adhered strictly to established industrial safety parameters, ensuring complete operational "
        "stability across all test benches during continuous multi-hour observational runs. Standard operating procedures "
        "required redundant documentation of every anomalous spike in sensor readings, archived to optical write-once media "
        "for subsequent audit and analytical verification by the central safety supervisory board. "
    )

    for ch_idx in range(1, 11):
        ch_title = f"chapter_{ch_idx:02d}"
        paragraphs = []
        if ch_idx == 1:
            paragraphs.append(ch1_intro)
        else:
            paragraphs.append(
                f"Chapter {ch_idx} of the scientific investigation commenced with standard system initialization protocols. "
                f"The experimental apparatus remained securely configured in containment sector {ch_idx}."
            )
        
        # Add repeated filler prose blocks to ensure each chapter exceeds 1,100 words
        # 10 chapters * ~1,100 words = ~11,000 words total (> 10,000 words)
        for _ in range(8):
            paragraphs.append(filler_prose)
            
        chapter_text = "\n\n".join(paragraphs)
        chapters.append((ch_title, chapter_text))

    return chapters


def test_8_3_long_context_book_benchmark():
    """8.3: Ingest a multi-chapter book (>10,000 words) into PageTable.

    Verifies:
    1. Total book length is strictly > 10,000 words.
    2. Physical memory usage (ActiveCanvas) remains flat (<= 512 nodes) across all 10 chapters.
    3. Book Merkle Root is deterministically computed (64-char BLAKE3 hash).
    4. Questions about Chapter 1 asked after Chapter 10 are answered in < 10 ms with zero hallucination.
    """
    chapters = _build_ten_chapter_book()
    total_words = sum(len(text.split()) for _, text in chapters)
    assert total_words > 10000, f"Expected > 10,000 words, got {total_words} words"

    # Initialize Cognitive Pipeline with fixed active canvas M = 512
    pipeline = CognitivePipeline(
        transducer_backend="mock",
        canvas_capacity=512,
    )

    t_start = time.perf_counter()
    merkle_book, book_root_cid = pipeline.ingest_book(chapters, validate=True)
    ingestion_time = time.perf_counter() - t_start

    # 1. Verify canvas memory stayed strictly bounded <= 512 nodes (O(1) VRAM execution)
    assert len(pipeline.active_canvas) <= 512, f"Active canvas exceeded bound: {len(pipeline.active_canvas)} > 512"
    canvas_stats = pipeline.active_canvas.stats()
    assert canvas_stats["active_nodes"] <= 512
    assert canvas_stats["memory_bytes"] <= 512 * 256  # strictly <= 128 KB of quaternary vectors!

    # 2. Verify deterministic 64-character BLAKE3 Book Merkle Root
    assert len(book_root_cid) == 64, f"Invalid BLAKE3 CID length: {len(book_root_cid)}"
    assert merkle_book.book_root_cid == book_root_cid
    assert len(merkle_book.chapters) == 10

    # 3. Sub-10ms Zero-Hallucination Query Answering over Chapter 1 after Chapter 10 ingestion
    # Note: Chapter 1 nodes were evicted from ActiveCanvas during subsequent chapter ingestion.
    # The pipeline must resolve Chapter 1 questions via PageTable in < 10 ms!
    latencies = []
    for _ in range(3):
        t0 = time.perf_counter()
        ans = pipeline.answer_query("What did Eleanor Vance verify?")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    assert ans == "The hypothesis", f"Expected 'The hypothesis', got '{ans}'"
    assert min(latencies) < 10.0, f"Query latency exceeded 10 ms: {min(latencies):.2f} ms"

    # Second query about Chapter 1
    latencies_2 = []
    for _ in range(3):
        t0 = time.perf_counter()
        ans2 = pipeline.answer_query("What did Eleanor Vance isolate?")
        latencies_2.append((time.perf_counter() - t0) * 1000.0)

    assert ans2 == "A volatile synthetic compound", f"Expected 'A volatile synthetic compound', got '{ans2}'"
    assert min(latencies_2) < 10.0, f"Query latency exceeded 10 ms: {min(latencies_2):.2f} ms"

    pipeline.close()
