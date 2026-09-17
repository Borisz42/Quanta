"""Unit and integration tests for Working Memory Entity Manifest & Paging Engine (Phase 2)."""

import time
import pytest

from parser.chunker import DiscourseChunker
from parser.schema import ExtractedEntity
from parser.entity_manifest import (
    ActiveEntityManifest,
    EntityMatcher,
    EntityPagingEngine,
    EntityRecord,
    EntityStorage,
    BAND_2_REGISTERS,
)


# ---------------------------------------------------------------------------
# 2.1: EntityRecord Unit Tests
# ---------------------------------------------------------------------------

def test_entity_record_creation_and_defaults():
    """Verify basic creation and field defaults of EntityRecord."""
    rec = EntityRecord(
        canonical_id="E1",
        canonical_name="Dr. Eleanor Vance",
        category="PERSON",
    )
    assert rec.canonical_id == "E1"
    assert rec.canonical_name == "Dr. Eleanor Vance"
    assert rec.category == "PERSON"
    assert rec.surface_aliases == []
    assert rec.register_binding is None
    assert rec.last_seen_chunk == 0
    assert rec.salience_score == 1.0
    assert rec.mention_count == 1
    assert rec.cid is None
    assert rec.properties == {}


def test_entity_record_alias_management():
    """Verify alias addition, deduplication, and case-insensitivity."""
    rec = EntityRecord(canonical_id="E1", canonical_name="Dr. Eleanor Vance")
    
    # Adding new aliases
    assert rec.add_alias("Eleanor Vance") is True
    assert rec.add_alias("Eleanor") is True
    assert rec.add_alias("Vance") is True
    assert len(rec.surface_aliases) == 3

    # Reject duplicate or case-insensitive duplicate
    assert rec.add_alias("eleanor") is False
    assert rec.add_alias("Vance") is False
    # Reject matching canonical name
    assert rec.add_alias("Dr. Eleanor Vance") is False
    assert rec.add_alias("") is False
    assert len(rec.surface_aliases) == 3

    # Test display aliases
    display_aliases = rec.get_display_aliases()
    assert "Eleanor" in display_aliases
    assert "Vance" in display_aliases
    assert "Dr. Eleanor Vance" not in display_aliases


def test_entity_record_salience_decay_and_boost():
    """Verify exponential salience decay across chunks and mention boost."""
    rec = EntityRecord(canonical_id="E1", canonical_name="Eleanor", salience_score=2.0, last_seen_chunk=0)

    # Seen immediately in chunk 1 (gap = 1)
    new_score = rec.update_salience(chunk_idx=1, boost=1.0, decay_rate=0.9)
    # expected: 2.0 * 0.9^1 + 1.0 = 1.8 + 1.0 = 2.8
    assert pytest.approx(new_score, rel=1e-3) == 2.8
    assert rec.last_seen_chunk == 1
    assert rec.mention_count == 2

    # Not seen until chunk 4 (gap = 3)
    new_score = rec.update_salience(chunk_idx=4, boost=1.0, decay_rate=0.9)
    # expected: 2.8 * (0.9^3) + 1.0 = 2.8 * 0.729 + 1.0 = 2.0412 + 1.0 = 3.0412
    assert pytest.approx(new_score, rel=1e-3) == 3.0412
    assert rec.last_seen_chunk == 4
    assert rec.mention_count == 3


def test_entity_record_serialization():
    """Verify dictionary serialization and deserialization roundtrip."""
    rec = EntityRecord(
        canonical_id="E2",
        canonical_name="synthetic compound",
        category="OBJECT",
        surface_aliases=["polymer", "specimen"],
        register_binding="VAR_SLOT_X1",
        last_seen_chunk=3,
        salience_score=2.5,
        mention_count=4,
        cid="bafkreib3abcdef",
        properties={"viscosity": "high", "temperature": 80},
    )

    data = rec.to_dict()
    restored = EntityRecord.from_dict(data)

    assert restored.canonical_id == rec.canonical_id
    assert restored.canonical_name == rec.canonical_name
    assert restored.category == rec.category
    assert restored.surface_aliases == rec.surface_aliases
    assert restored.register_binding == rec.register_binding
    assert restored.last_seen_chunk == rec.last_seen_chunk
    assert restored.salience_score == rec.salience_score
    assert restored.mention_count == rec.mention_count
    assert restored.cid == rec.cid
    assert restored.properties == rec.properties


# ---------------------------------------------------------------------------
# 2.2: ActiveEntityManifest & LRU Eviction Unit Tests
# ---------------------------------------------------------------------------

def test_manifest_capacity_and_lru_eviction(tmp_path):
    """Test that ActiveEntityManifest evicts LRU entities to storage when max_size is exceeded."""
    db_file = tmp_path / "test_lru.db"
    storage = EntityStorage(db_file)
    manifest = ActiveEntityManifest(storage=storage, min_size=2, max_size=3)

    # Insert 3 entities (fill to max_size)
    e1 = EntityRecord(canonical_id="E1", canonical_name="Entity 1", salience_score=1.0, last_seen_chunk=1)
    e2 = EntityRecord(canonical_id="E2", canonical_name="Entity 2", salience_score=2.0, last_seen_chunk=2)
    e3 = EntityRecord(canonical_id="E3", canonical_name="Entity 3", salience_score=3.0, last_seen_chunk=3)

    manifest.add_or_update(e1, chunk_idx=1, boost_salience=False)
    manifest.add_or_update(e2, chunk_idx=2, boost_salience=False)
    manifest.add_or_update(e3, chunk_idx=3, boost_salience=False)

    assert len(manifest) == 3
    assert manifest.is_active("E1")
    assert manifest.is_active("E2")
    assert manifest.is_active("E3")

    # Adding 4th entity must evict the lowest salience entity (E1)
    e4 = EntityRecord(canonical_id="E4", canonical_name="Entity 4", salience_score=4.0, last_seen_chunk=4)
    manifest.add_or_update(e4, chunk_idx=4, boost_salience=False)

    assert len(manifest) == 3
    assert not manifest.is_active("E1")  # E1 was evicted
    assert manifest.is_active("E2")
    assert manifest.is_active("E3")
    assert manifest.is_active("E4")

    # Verify E1 still exists safely in SQLite storage
    stored_e1 = storage.get_entity("E1")
    assert stored_e1 is not None
    assert stored_e1.canonical_name == "Entity 1"

    storage.close()


def test_manifest_register_allocation():
    """Verify Band 2 register allocation (VAR_SLOT_X0 to X7) and release on eviction."""
    storage = EntityStorage(":memory:")
    manifest = ActiveEntityManifest(storage=storage, min_size=2, max_size=10)

    records = []
    for i in range(8):
        rec = EntityRecord(canonical_id=f"E{i+1}", canonical_name=f"Entity {i+1}")
        manifest.add_or_update(rec, chunk_idx=i)
        records.append(rec)

    # All 8 entities should have a distinct Band 2 register
    assigned_regs = [r.register_binding for r in manifest.all_active()]
    assert len(set(assigned_regs)) == 8
    for reg in BAND_2_REGISTERS:
        assert reg in assigned_regs

    # 9th entity: registers are full, so it receives register only if it displaces lower salience
    rec9 = EntityRecord(canonical_id="E9", canonical_name="Entity 9", salience_score=0.5)
    manifest.add_or_update(rec9, chunk_idx=8, boost_salience=False)
    # rec9 has lowest salience, so remains unbound
    assert rec9.register_binding is None

    # Evict E1: its register should be freed and available
    freed_reg = records[0].register_binding
    manifest.evict("E1")
    assert not manifest.is_active("E1")

    # Update rec9: it can now take the freed register
    manifest._allocate_register(rec9)
    assert rec9.register_binding == freed_reg


# ---------------------------------------------------------------------------
# 2.3: Fast Pre-Scan Entity Matcher Unit Tests (< 1 ms requirement)
# ---------------------------------------------------------------------------

def test_entity_matcher_basic_matching():
    """Test that EntityMatcher accurately identifies registered aliases in chunk text."""
    matcher = EntityMatcher()
    matcher.build_index({
        "Dr. Eleanor Vance": ["E1"],
        "Eleanor": ["E1"],
        "Vance": ["E1"],
        "synthetic compound": ["E2"],
        "polymer": ["E2"],
        "cryogenic containment cell": ["E3"],
    })

    chunk_text = (
        "Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. "
        "She immediately noted that this polymer was stable."
    )

    matches = matcher.match_chunk(chunk_text)
    assert "E1" in matches
    assert "E2" in matches
    assert "E3" in matches
    assert any("Eleanor" in s for s in matches["E1"])
    assert any("polymer" in s or "synthetic compound" in s for s in matches["E2"])


def test_entity_matcher_ignores_generic_pronouns():
    """Verify that generic pronouns like 'she', 'it', 'they' do NOT trigger entity match."""
    matcher = EntityMatcher()
    matcher.build_index({
        "she": ["E1"],
        "it": ["E2"],
        "Eleanor": ["E1"],
    })

    chunk_text = "She observed it carefully throughout the afternoon."
    matches = matcher.match_chunk(chunk_text)
    # 'she' and 'it' are ignored tokens, so no match occurs
    assert len(matches) == 0

    chunk_text_with_name = "She observed Eleanor carefully."
    matches = matcher.match_chunk(chunk_text_with_name)
    assert "E1" in matches


def test_entity_matcher_sub_millisecond_speed():
    """Assert that pre-scanning a typical 300-400 word chunk completes in < 1.0 ms."""
    matcher = EntityMatcher()
    # Populate with 100 realistic entity aliases
    alias_map = {}
    for i in range(100):
        alias_map[f"Character {i}"] = [f"E{i}"]
        alias_map[f"Dr. Researcher {i}"] = [f"E{i}"]
        alias_map[f"Laboratory Equipment {i}"] = [f"E{i+100}"]
    alias_map["Dr. Eleanor Vance"] = ["E1"]
    alias_map["synthetic compound"] = ["E2"]
    alias_map["Marcus Wright"] = ["E3"]
    matcher.build_index(alias_map)

    paragraph = (
        "Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn. "
        "She immediately noted that this specimen exhibited anomalous crystalline lattice expansion, which "
        "strongly suggested an unobserved phase transition. Although her supervisor initially doubted the validity "
        "of the discovery, Eleanor verified the hypothesis three hours later by replicating the transformation "
        "within the same vessel. The resulting polymer retained its structural integrity throughout the afternoon, "
        "prompting the laboratory director to prohibit all competing tests until her synthesis protocol could be formally audited."
    )

    # Warm-up compile
    matcher.match_chunk(paragraph)

    # Benchmark over 500 iterations
    iterations = 500
    t0 = time.perf_counter()
    for _ in range(iterations):
        matcher.match_chunk(paragraph)
    elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
    avg_elapsed_ms = elapsed_total_ms / iterations

    # Average match time must be strictly under 1.0 ms (typically < 0.1 ms)
    assert avg_elapsed_ms < 1.0, f"Average scan time {avg_elapsed_ms:.4f} ms exceeded 1.0 ms limit"


# ---------------------------------------------------------------------------
# 2.4: Prompt Formatter Unit Tests
# ---------------------------------------------------------------------------

def test_prompt_block_formatting():
    """Verify prompt formatting matches the exact specification in Item 2.4."""
    storage = EntityStorage(":memory:")
    manifest = ActiveEntityManifest(storage=storage)

    # Register E1
    e1 = EntityRecord(
        canonical_id="E1",
        canonical_name="Dr. Eleanor Vance",
        category="PERSON",
        surface_aliases=["Eleanor", "Vance"],
        salience_score=3.0,
        last_seen_chunk=1,
    )
    # Register E2
    e2 = EntityRecord(
        canonical_id="E2",
        canonical_name="synthetic compound",
        category="OBJECT",
        surface_aliases=["polymer", "specimen"],
        salience_score=2.0,
        last_seen_chunk=1,
    )
    manifest.add_or_update(e1, chunk_idx=1, boost_salience=False)
    manifest.add_or_update(e2, chunk_idx=1, boost_salience=False)

    engine = EntityPagingEngine(target_size=10)
    engine.manifest = manifest

    block = engine.format_prompt_block()
    expected = (
        "ACTIVE ENTITIES:\n"
        "- E1: Dr. Eleanor Vance (aliases: Eleanor, Vance)\n"
        "- E2: synthetic compound (aliases: polymer, specimen)"
    )
    assert block.strip() == expected.strip()


def test_prompt_block_formatting_no_aliases():
    """Verify prompt formatting when entities have no secondary aliases."""
    engine = EntityPagingEngine()
    engine.register_new_entity(name="Containment Unit", category="OBJECT")

    block = engine.format_prompt_block()
    assert "- E1: Containment Unit" in block
    assert "aliases:" not in block


# ---------------------------------------------------------------------------
# 2.5: 5-Chunk Story Simulation Integration Test
# ---------------------------------------------------------------------------

def test_five_chunk_story_paging_simulation():
    """Simulate a 5-chunk narrative:
    
    - Chunk 1: E1 (Dr. Eleanor Vance) is introduced.
    - Chunks 2-4: Many new characters/objects are introduced, causing E1 to be evicted to SQLite.
    - Chunk 5: Eleanor is mentioned again; fast pre-scan resurrects E1 back into the active manifest.
    - Assert E1 preserves its canonical ID, merged aliases, and updated recency.
    """
    # Create engine with tight working set: target=3, max=4 to test paging pressure
    engine = EntityPagingEngine(target_size=3, min_size=2, max_size=4)

    # -------------------------------------------------------------------------
    # Chunk 1: Dr. Eleanor Vance introduced
    # -------------------------------------------------------------------------
    chunk_1 = "Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn."
    
    # Pre-scan (no entities registered yet)
    paged, scan_time = engine.pre_scan_and_page(chunk_1, chunk_idx=1)
    assert scan_time < 1.0
    assert len(paged) == 0

    # Simulate Neural Transducer extraction for Chunk 1
    engine.update_from_extraction([
        {
            "canonical_name": "Dr. Eleanor Vance",
            "category": "PERSON",
            "surface_aliases": ["Eleanor", "Vance"],
        },
        {
            "canonical_name": "synthetic compound",
            "category": "OBJECT",
            "surface_aliases": ["specimen", "polymer"],
        },
    ], chunk_idx=1)

    assert engine.manifest.is_active("E1")
    assert engine.manifest.is_active("E2")
    assert engine.manifest.get("E1").canonical_name == "Dr. Eleanor Vance"

    # -------------------------------------------------------------------------
    # Chunks 2, 3, 4: Introduce flood of other entities
    # -------------------------------------------------------------------------
    # Chunk 2 introduces E3 (Marcus Wright) and E4 (mass spectrometer)
    chunk_2 = "Marcus Wright checked the mass spectrometer calibration."
    engine.pre_scan_and_page(chunk_2, chunk_idx=2)
    engine.update_from_extraction([
        {"canonical_name": "Marcus Wright", "category": "PERSON", "surface_aliases": ["Marcus"]},
        {"canonical_name": "mass spectrometer", "category": "OBJECT", "surface_aliases": ["spectrometer"]},
    ], chunk_idx=2)

    # Active set is now at max_size (4 entities: E1, E2, E3, E4)
    assert len(engine.manifest) == 4

    # Chunk 3 introduces E5 (Director Harrison) and E6 (audit committee)
    chunk_3 = "Director Harrison convened the audit committee to review procedure."
    engine.pre_scan_and_page(chunk_3, chunk_idx=3)
    engine.update_from_extraction([
        {"canonical_name": "Director Harrison", "category": "PERSON", "surface_aliases": ["Harrison"]},
        {"canonical_name": "audit committee", "category": "ORGANIZATION", "surface_aliases": ["committee"]},
    ], chunk_idx=3)

    # Chunk 4 introduces E7 (nitrogen tank)
    chunk_4 = "The technician refilled the liquid nitrogen tank before the meeting."
    engine.pre_scan_and_page(chunk_4, chunk_idx=4)
    engine.update_from_extraction([
        {"canonical_name": "liquid nitrogen tank", "category": "OBJECT", "surface_aliases": ["nitrogen tank"]},
    ], chunk_idx=4)

    # At this point, E1 has NOT been seen in chunks 2, 3, or 4!
    # Due to max_size=4 and LRU eviction, E1 must be safely evicted from active memory
    assert not engine.manifest.is_active("E1"), "E1 should have been evicted from working memory"

    # BUT E1 must exist in the SQLite Page Table!
    stored_e1 = engine.storage.get_entity("E1")
    assert stored_e1 is not None
    assert stored_e1.canonical_name == "Dr. Eleanor Vance"
    assert "Eleanor" in stored_e1.surface_aliases

    # -------------------------------------------------------------------------
    # Chunk 5: Eleanor returns! Pre-scan must resurrect E1 from SQLite
    # -------------------------------------------------------------------------
    chunk_5 = "Later that evening, Eleanor returned to the laboratory to examine the results."
    
    # Fast pre-scan runs on Chunk 5 text
    paged_5, scan_time_5 = engine.pre_scan_and_page(chunk_5, chunk_idx=5)

    # Must complete in < 1.0 ms
    assert scan_time_5 < 1.0, f"Pre-scan took {scan_time_5:.4f} ms, expected < 1.0 ms"

    # E1 must be detected by the alias 'Eleanor' and paged back into the active manifest!
    assert any(e.canonical_id == "E1" for e in paged_5)
    assert engine.manifest.is_active("E1")

    # Verify E1 properties were preserved
    resurrected_e1 = engine.manifest.get("E1")
    assert resurrected_e1.canonical_id == "E1"
    assert resurrected_e1.canonical_name == "Dr. Eleanor Vance"
    assert resurrected_e1.last_seen_chunk == 5
    assert resurrected_e1.mention_count >= 2

    # Verify prompt block output contains E1 again
    prompt_block = engine.format_prompt_block()
    assert "E1: Dr. Eleanor Vance" in prompt_block

    engine.close()


# ---------------------------------------------------------------------------
# 2.6: Multi-Chapter Coreference & Paging Integration Tests
# ---------------------------------------------------------------------------

MULTI_CHAPTER_BOOK_TEXT = """# Chapter 1: The Discovery
Dr. Eleanor Vance isolated a volatile synthetic compound inside the cryogenic containment cell at dawn.
She immediately noted that this polymer was stable and exhibited anomalous crystalline expansion.

# Chapter 2: The Parallel Inquiry
In the secondary facility across the campus, Dr. Marcus Wright calibrated the high-resolution spectrometer.
Director Harrison convened the regulatory oversight panel to evaluate the instrumentation safety standards.
The audit committee reviewed protocol documentation all afternoon while the technicians monitored temperatures.

# Chapter 3: The Validation
At nine o'clock the following morning, Eleanor presented her verified findings to the entire department.
She confirmed that the synthetic compound retained its anomalous density across all measurement cycles.
"""


def test_multi_chapter_coreference_across_chapters():
    """Verify that a multi-chapter narrative segmented by DiscourseChunker preserves entity IDs
    and resurrects dormant entities across chapters via EntityPagingEngine.
    """
    chunker = DiscourseChunker(min_words=10, max_words=60, use_spacy=False)
    chunks = chunker.chunk_document(MULTI_CHAPTER_BOOK_TEXT)
    assert len(chunks) >= 3

    # Verify chapters are recognized on chunks
    ch_ids = [c.chapter_id for c in chunks if c.chapter_id]
    assert any("chapter_1" in cid for cid in ch_ids)
    assert any("chapter_2" in cid for cid in ch_ids)
    assert any("chapter_3" in cid for cid in ch_ids)

    # Tight capacity: target=2, max=3 so Chapter 2 forces Chapter 1 entities out of active RAM
    engine = EntityPagingEngine(target_size=2, min_size=2, max_size=3)

    # Process Chapter 1 chunk(s)
    ch1_chunks = [c for c in chunks if c.chapter_id and "chapter_1" in c.chapter_id]
    assert len(ch1_chunks) >= 1
    for c in ch1_chunks:
        engine.pre_scan_and_page(c.text, chunk_idx=1)
        engine.update_from_extraction([
            {"canonical_name": "Dr. Eleanor Vance", "category": "PERSON", "surface_aliases": ["Eleanor", "Vance"]},
            {"canonical_name": "synthetic compound", "category": "OBJECT", "surface_aliases": ["polymer"]},
        ], chunk_idx=1)

    assert engine.manifest.is_active("E1")
    assert engine.manifest.is_active("E2")
    assert engine.manifest.get("E1").canonical_name == "Dr. Eleanor Vance"

    # Process Chapter 2 chunk(s) - floods working memory with new entities
    ch2_chunks = [c for c in chunks if c.chapter_id and "chapter_2" in c.chapter_id]
    assert len(ch2_chunks) >= 1
    for idx, c in enumerate(ch2_chunks, start=2):
        engine.pre_scan_and_page(c.text, chunk_idx=idx)
        engine.update_from_extraction([
            {"canonical_name": "Dr. Marcus Wright", "category": "PERSON", "surface_aliases": ["Marcus"]},
            {"canonical_name": "high-resolution spectrometer", "category": "OBJECT", "surface_aliases": ["spectrometer"]},
            {"canonical_name": "Director Harrison", "category": "PERSON", "surface_aliases": ["Harrison"]},
            {"canonical_name": "audit committee", "category": "ORGANIZATION", "surface_aliases": ["panel"]},
        ], chunk_idx=idx)

    # In Chapter 2, Eleanor (E1) was NOT mentioned, so LRU eviction pushed E1 to SQLite
    assert not engine.manifest.is_active("E1"), "E1 should be evicted during Chapter 2"
    stored_e1 = engine.storage.get_entity("E1")
    assert stored_e1 is not None
    assert stored_e1.canonical_name == "Dr. Eleanor Vance"

    # Process Chapter 3 chunk(s) - Eleanor and the synthetic compound return!
    ch3_chunks = [c for c in chunks if c.chapter_id and "chapter_3" in c.chapter_id]
    assert len(ch3_chunks) >= 1
    for idx, c in enumerate(ch3_chunks, start=10):
        paged, scan_time = engine.pre_scan_and_page(c.text, chunk_idx=idx)
        assert scan_time < 1.0, f"Pre-scan took {scan_time:.4f} ms, expected < 1.0 ms"
        # Pre-scan MUST detect alias 'Eleanor' and resurrect E1!
        assert any(e.canonical_id == "E1" for e in paged)
        assert engine.manifest.is_active("E1")

        # Entity properties preserved across chapters
        e1_rec = engine.manifest.get("E1")
        assert e1_rec.canonical_id == "E1"
        assert e1_rec.canonical_name == "Dr. Eleanor Vance"
        assert e1_rec.last_seen_chunk == idx
        assert "Eleanor" in e1_rec.surface_aliases

    # Render prompt block for Chapter 3: must show E1 restored
    prompt_block = engine.format_prompt_block()
    assert "E1: Dr. Eleanor Vance" in prompt_block

    engine.close()


def test_entity_matcher_punctuation_aliases():
    """Verify that EntityMatcher accurately recognizes aliases with trailing/embedded punctuation."""
    matcher = EntityMatcher()
    matcher.build_index({
        "U.S.": ["E1"],
        "Ph.D.": ["E2"],
        "St. Jude": ["E3"],
        "Dr. Vance": ["E4"],
    })

    text = "She earned her Ph.D. in the U.S. before joining St. Jude to work alongside Dr. Vance on research."
    matches = matcher.match_chunk(text)

    assert "E1" in matches, "Should match 'U.S.'"
    assert "E2" in matches, "Should match 'Ph.D.'"
    assert "E3" in matches, "Should match 'St. Jude'"
    assert "E4" in matches, "Should match 'Dr. Vance'"


def test_entity_record_blake3_cid_computation():
    """Verify deterministic BLAKE3 256-bit Merkle CID computation on EntityRecord."""
    rec1 = EntityRecord(canonical_id="E1", canonical_name="Eleanor Vance", category="PERSON")
    cid1 = rec1.compute_cid()
    assert isinstance(cid1, str)
    assert len(cid1) == 64  # 256-bit hex digest

    # Calling compute_cid again returns cached CID
    assert rec1.compute_cid() == cid1

    # Identical name & category produce identical CID
    rec2 = EntityRecord(canonical_id="E99", canonical_name="Eleanor Vance", category="PERSON")
    assert rec2.compute_cid() == cid1

    # Different category produces different CID
    rec3 = EntityRecord(canonical_id="E1", canonical_name="Eleanor Vance", category="ORGANIZATION")
    assert rec3.compute_cid() != cid1

    # to_dict includes cid
    d = rec1.to_dict()
    assert d["cid"] == cid1


def test_storage_batch_persistence_and_wal(tmp_path):
    """Verify save_entities batch insertion and automatic parent directory creation with WAL mode."""
    nested_db = tmp_path / "deep" / "nested" / "dir" / "entities.db"
    assert not nested_db.parent.exists()

    storage = EntityStorage(nested_db)
    assert nested_db.exists()

    # Batch save entities
    records = [
        EntityRecord(canonical_id=f"E{i}", canonical_name=f"Entity {i}", surface_aliases=[f"Alias {i}"])
        for i in range(10)
    ]
    storage.save_entities(records)
    assert storage.count() == 10

    # Retrieve an entity and check alias lookup
    e5 = storage.get_entity("E5")
    assert e5 is not None
    assert e5.canonical_name == "Entity 5"

    found = storage.find_by_alias("Alias 5")
    assert len(found) == 1
    assert found[0].canonical_id == "E5"

    storage.close()


def test_polymorphic_extraction_update():
    """Verify update_from_extraction seamlessly accepts ExtractedEntity dataclasses as well as dicts."""
    engine = EntityPagingEngine()

    # Pass ExtractedEntity dataclass instances
    dataclass_entities = [
        ExtractedEntity(
            id="E1",
            canonical_name="Dr. Eleanor Vance",
            category="PERSON",
            surface_aliases=["Eleanor"],
        ),
        ExtractedEntity(
            id="E2",
            canonical_name="synthetic compound",
            category="OBJECT",
            surface_aliases=["polymer"],
        ),
    ]
    records = engine.update_from_extraction(dataclass_entities, chunk_idx=1)
    assert len(records) == 2
    assert engine.manifest.is_active("E1")
    assert engine.manifest.is_active("E2")
    assert engine.manifest.get("E1").canonical_name == "Dr. Eleanor Vance"

    # Pass plain dictionaries
    dict_entities = [
        {"canonical_id": "E1", "canonical_name": "Dr. Eleanor Vance", "surface_aliases": ["Vance"]},
    ]
    records2 = engine.update_from_extraction(dict_entities, chunk_idx=2)
    assert len(records2) == 1
    assert "Vance" in engine.manifest.get("E1").surface_aliases
    assert engine.manifest.get("E1").last_seen_chunk == 2

    engine.close()


def test_active_manifest_collision_prevention():
    """Verify that registering an external entity with explicit ID advances next_id_num to prevent collisions."""
    storage = EntityStorage(":memory:")
    manifest = ActiveEntityManifest(storage=storage)

    # Default mint starts at E1
    assert manifest.mint_entity_id() == "E1"

    # Add entity with explicit E10
    e10 = EntityRecord(canonical_id="E10", canonical_name="High ID Entity")
    manifest.add_or_update(e10)

    # Next minted ID must be E11 to avoid colliding with E10!
    next_id = manifest.mint_entity_id()
    assert next_id == "E11"

    storage.close()


def test_manifest_prompt_formatting_options_and_token_estimate():
    """Verify prompt block rendering flags and token estimation."""
    engine = EntityPagingEngine()
    engine.register_new_entity("Dr. Eleanor Vance", category="PERSON", aliases=["Eleanor"])
    engine.register_new_entity("synthetic compound", category="OBJECT", aliases=["polymer"])

    # Standard block
    standard = engine.format_prompt_block()
    assert "- E1: Dr. Eleanor Vance (aliases: Eleanor)" in standard
    assert "VAR_SLOT" not in standard

    # With registers and categories
    detailed = engine.format_prompt_block(include_registers=True, include_categories=True)
    assert "[VAR_SLOT_X0]" in detailed
    assert "[PERSON]" in detailed
    assert "[OBJECT]" in detailed

    # Token estimation
    tokens = engine.estimate_manifest_tokens()
    assert 5 < tokens < 100

    engine.close()

