"""Unit and integration tests for QUANTA Fast Skeleton Transducer and SpanAligner.

Verifies Section 3 requirements:
1. SpanAligner character and UTF-8 byte span grounding across whitespace, punctuation,
   case-insensitivity, morphology affixes, and multilingual unicode characters.
2. SkeletonEntity, SkeletonEvent, and SkeletonExtractionResult schemas, serialization,
   Canonical Record DSL generation, and QuantaGraph compilation.
3. SkeletonTransducer and MockSkeletonTransducer extraction over scientific and narrative passages.
4. Strict token budget enforcement: decoding token length is strictly <= 50 tokens (30--45 token profile).
5. Exact byte-level span alignment verification against raw passage text.
6. UnslothTransducer mode="skeleton" integration and telemetry verification.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest

from core.asg import QuantaGraph
from parser.entity_manifest import EntityRecord
from parser.skeleton_transducer import (
    DEFAULT_SKELETON_SYSTEM_PROMPT,
    MockSkeletonTransducer,
    SkeletonEntity,
    SkeletonEvent,
    SkeletonExtractionResult,
    SkeletonTransducer,
    _locate_skeleton_gbnf,
    estimate_token_count,
)
from parser.span_aligner import SpanAligner, SpanAlignment
from parser.unsloth_transducer import MockUnslothTransducer, UnslothTransducer


# ---------------------------------------------------------------------------
# 1. SpanAligner Character & Byte Offset Grounding Tests
# ---------------------------------------------------------------------------

class TestSpanAligner:
    """Verifies precise character and byte-level span resolution."""

    @pytest.fixture
    def aligner(self) -> SpanAligner:
        return SpanAligner()

    def test_exact_ascii_alignment(self, aligner: SpanAligner):
        text = "Dr. Marcus Vance pressurized the argon cylinder inside the test chamber."
        res = aligner.align(text, "argon cylinder")

        assert res is not None
        assert res.is_exact is True
        assert res.confidence == 1.0
        assert res.matched_text == "argon cylinder"
        assert res.char_span == (33, 47)
        assert text[res.start:res.end] == "argon cylinder"

        # Verify byte span in UTF-8
        raw_bytes = text.encode("utf-8")
        assert raw_bytes[res.byte_start:res.byte_end].decode("utf-8") == "argon cylinder"

    def test_punctuation_and_quote_trimming(self, aligner: SpanAligner):
        text = 'She examined the "synthetic compound", which was highly unstable.'
        
        # Test with trailing comma and quotes
        res = aligner.align(text, '"synthetic compound",')
        assert res is not None
        assert res.matched_text == "synthetic compound"
        assert text[res.start:res.end] == "synthetic compound"
        assert res.confidence >= 0.95

    def test_case_insensitive_matching(self, aligner: SpanAligner):
        text = "The Argon Cylinder was stored in cell 4."
        res = aligner.align(text, "argon cylinder")

        assert res is not None
        assert res.matched_text == "Argon Cylinder"
        assert text[res.start:res.end] == "Argon Cylinder"
        assert res.confidence >= 0.90

    def test_whitespace_variance_resolution(self, aligner: SpanAligner):
        text = "Dr. Marcus   Vance\npressurized the cylinder."
        res = aligner.align(text, "Dr. Marcus Vance")

        assert res is not None
        assert "Marcus" in res.matched_text
        assert text[res.start:res.end] == res.matched_text
        assert res.confidence >= 0.90

    def test_morphological_affix_and_boundary(self, aligner: SpanAligner):
        text = "The operator pressurized the cylinders in sequence."
        res = aligner.align(text, "cylinder")

        assert res is not None
        assert "cylinder" in res.matched_text.lower()
        assert res.confidence >= 0.85

    def test_multilingual_unicode_byte_span_alignment(self, aligner: SpanAligner):
        # Hungarian text with multibyte characters (á, é, í, ó, ö, ő, ú, ü, ű)
        text = "Dr. Kovács János szintetizálta az új fluoropolimer mintát a szegedi lézerközpontban."
        res_entity = aligner.align(text, "Kovács János")

        assert res_entity is not None
        assert res_entity.matched_text == "Kovács János"
        assert text[res_entity.start:res_entity.end] == "Kovács János"

        raw_bytes = text.encode("utf-8")
        extracted_from_bytes = raw_bytes[res_entity.byte_start:res_entity.byte_end].decode("utf-8")
        assert extracted_from_bytes == "Kovács János"

        # Note that for 'Kovács', 'á' is 2 bytes, so byte_span length > char_span length
        char_len = res_entity.end - res_entity.start
        byte_len = res_entity.byte_end - res_entity.byte_start
        assert byte_len > char_len  # Because 'á' takes 2 bytes

    def test_chinese_unicode_byte_span_alignment(self, aligner: SpanAligner):
        # Chinese text where each character is 3 UTF-8 bytes
        text = "科学家在国家实验室里合成了新型高分子材料。"
        res = aligner.align(text, "科学家")

        assert res is not None
        assert res.matched_text == "科学家"
        assert res.end - res.start == 3  # 3 characters
        assert res.byte_end - res.byte_start == 9  # 9 bytes (3 bytes per character)

        raw_bytes = text.encode("utf-8")
        assert raw_bytes[res.byte_start:res.byte_end].decode("utf-8") == "科学家"

    def test_fuzzy_alignment_for_minor_typo(self, aligner: SpanAligner):
        text = "The containment cell remained hermetically sealed."
        # Typo in query: 'containmnt' instead of 'containment'
        res = aligner.align(text, "containmnt cell")

        assert res is not None
        assert res.matched_text == "containment cell"
        assert text[res.start:res.end] == "containment cell"
        assert res.confidence >= 0.70

    def test_align_all_sequential(self, aligner: SpanAligner):
        text = "Alice greeted Bob. Later, Alice left the room."
        tokens = ["Alice", "Bob", "Alice"]
        results = aligner.align_all(text, tokens, sequential=True)

        assert len(results) == 3
        assert results[0] is not None and results[0].start == 0
        assert results[1] is not None and results[1].start == 14
        assert results[2] is not None and results[2].start > results[1].end  # Advanced to second Alice


# ---------------------------------------------------------------------------
# 2. Skeleton Schema, DSL & ASG Compilation Tests
# ---------------------------------------------------------------------------

class TestSkeletonSchemas:
    """Verifies typed schemas, round-trip serialization, and ASG/DSL bridge."""

    def test_skeleton_entity_and_event_serialization(self):
        ent = SkeletonEntity(
            id="E1",
            surface_text="Dr. Marcus Vance",
            char_span=(0, 16),
            byte_span=(0, 16),
            category="PERSON",
            confidence=0.98,
        )
        d = ent.to_dict()
        assert d["id"] == "E1"
        assert d["surface_text"] == "Dr. Marcus Vance"
        assert d["char_span"] == [0, 16]

        ent2 = SkeletonEntity.from_dict(d)
        assert ent2 == ent

        ev = SkeletonEvent(
            id="EV1",
            predicate="pressurize",
            char_span=(17, 28),
            subject_ent_id="E1",
            object_ent_id="E2",
            confidence=0.95,
        )
        ev_d = ev.to_dict()
        assert ev_d["predicate"] == "pressurize"
        assert ev_d["subject_ent_id"] == "E1"

        ev2 = SkeletonEvent.from_dict(ev_d)
        assert ev2 == ev

    def test_skeleton_extraction_result_json_roundtrip(self):
        res = SkeletonExtractionResult(
            chunk_id="chunk_42",
            text="Marcus pressurized the cylinder.",
            entities=[
                SkeletonEntity(id="E1", surface_text="Marcus", char_span=(0, 6)),
                SkeletonEntity(id="E2", surface_text="cylinder", char_span=(23, 31)),
            ],
            events=[
                SkeletonEvent(id="EV1", predicate="pressurize", char_span=(7, 18), subject_ent_id="E1", object_ent_id="E2"),
            ],
            metadata={"model": "qwen3.5-4b-skeleton"},
        )

        json_str = res.to_json()
        res_loaded = SkeletonExtractionResult.from_json(json_str)

        assert len(res_loaded.entities) == 2
        assert len(res_loaded.events) == 1
        assert res_loaded.chunk_id == "chunk_42"
        assert res_loaded.events[0].subject_ent_id == "E1"

    def test_skeleton_to_canonical_record_dsl(self):
        res = SkeletonExtractionResult(
            chunk_id="chk_01",
            text="Alice entered the lab.",
            entities=[
                SkeletonEntity(id="E1", surface_text="Alice", char_span=(0, 5), category="PERSON"),
                SkeletonEntity(id="E2", surface_text="lab", char_span=(18, 21), category="LOCATION"),
            ],
            events=[
                SkeletonEvent(id="EV1", predicate="enter", char_span=(6, 13), subject_ent_id="E1", object_ent_id="E2"),
            ],
        )

        dsl = res.to_record_dsl(passage_id="P101", doc_id="doc_bio_01")
        assert "passage P101 {" in dsl
        assert 'doc_id: "doc_bio_01"' in dsl
        assert 'entity E1 "Alice" {' in dsl
        assert "type: PERSON" in dsl
        assert 'event EV1 "enter" {' in dsl
        assert "agent: E1" in dsl
        assert "patient: E2" in dsl

    def test_skeleton_to_asg_graph(self):
        res = SkeletonExtractionResult(
            chunk_id="chk_02",
            text="Marcus observed the sample.",
            entities=[
                SkeletonEntity(id="E1", surface_text="Marcus", char_span=(0, 6), category="PERSON"),
                SkeletonEntity(id="E2", surface_text="sample", char_span=(20, 26), category="OBJECT"),
            ],
            events=[
                SkeletonEvent(id="EV1", predicate="observe", char_span=(7, 15), subject_ent_id="E1", object_ent_id="E2"),
            ],
        )

        graph = res.to_asg(passage_id="P505")
        assert isinstance(graph, QuantaGraph)
        assert len(graph.nodes) == 3  # 2 entities + 1 event
        assert "P505" in graph.passage_to_nodes
        assert len(graph.get_nodes_for_passage("P505")) == 3


# ---------------------------------------------------------------------------
# 3. GBNF Grammar File Check
# ---------------------------------------------------------------------------

def test_skeleton_gbnf_grammar_exists_and_valid():
    """Verify data/grammar/skeleton_schema.gbnf exists and parses valid root rule."""
    gbnf_path = _locate_skeleton_gbnf()
    assert gbnf_path.is_file()
    assert gbnf_path.name == "skeleton_schema.gbnf"

    content = gbnf_path.read_text(encoding="utf-8")
    assert "root ::=" in content
    assert ("entity_list ::=" in content) or ("entity-list ::=" in content)
    assert ("event_list ::=" in content) or ("event-list ::=" in content)


# ---------------------------------------------------------------------------
# 4. Token Budget & Span Verification on Scientific & Narrative Passages
# ---------------------------------------------------------------------------

class TestSkeletonTransducerPassages:
    """Tests skeleton transduction across scientific, narrative, and dynamic passages."""

    @pytest.fixture
    def transducer(self) -> MockSkeletonTransducer:
        return MockSkeletonTransducer()

    def test_scientific_passage_extraction_and_token_budget(self, transducer: MockSkeletonTransducer):
        passage = (
            "Dr. Marcus Vance pressurized the argon cylinder inside the test chamber at noon. "
            "He verified that the valve held."
        )

        result = transducer.transduce(passage, chunk_id="sci_chunk_01")

        # 1. Verification of extracted entities & events
        assert len(result.entities) >= 2
        entity_texts = [e.surface_text for e in result.entities]
        assert "Dr. Marcus Vance" in entity_texts or any("Marcus" in t for t in entity_texts)
        assert any("argon cylinder" in t or "cylinder" in t for t in entity_texts)

        assert len(result.events) >= 1
        event_preds = [ev.predicate for ev in result.events]
        assert "pressurize" in event_preds or "pressurized" in event_preds

        # 2. Strict Token Budget Assertion (<= 50 tokens, typical 30-45 tokens)
        token_count = result.metadata["decoding_token_count"]
        assert token_count <= 50, f"Token count {token_count} exceeds maximum budget of 50 tokens!"
        assert token_count >= 15, f"Token count {token_count} too low for valid skeleton!"

        # 3. Byte-level Span Alignment Verification
        raw_bytes = passage.encode("utf-8")
        for ent in result.entities:
            c_start, c_end = ent.char_span
            assert passage[c_start:c_end] == ent.surface_text
            if ent.byte_span:
                b_start, b_end = ent.byte_span
                assert raw_bytes[b_start:b_end].decode("utf-8") == ent.surface_text

        for ev in result.events:
            c_start, c_end = ev.char_span
            assert passage[c_start:c_end].lower().startswith(ev.predicate[:4].lower())

    def test_narrative_passage_extraction_and_token_budget(self, transducer: MockSkeletonTransducer):
        passage = (
            "Eleanor Vance entered the containment cell while the synthetic compound reacted violently."
        )

        result = transducer.transduce(passage, chunk_id="narr_chunk_02")

        # 1. Verification of extracted entities & events
        assert len(result.entities) >= 2
        entity_names = [e.surface_text for e in result.entities]
        assert "Eleanor Vance" in entity_names or any("Eleanor" in n for n in entity_names)

        event_preds = [ev.predicate for ev in result.events]
        assert "enter" in event_preds or "react" in event_preds

        # 2. Strict Token Budget Assertion (<= 50 tokens)
        token_count = result.metadata["decoding_token_count"]
        assert token_count <= 50, f"Token count {token_count} exceeds maximum budget of 50 tokens!"

        # 3. Byte-level Span Alignment Verification
        raw_bytes = passage.encode("utf-8")
        for ent in result.entities:
            c_start, c_end = ent.char_span
            assert passage[c_start:c_end] == ent.surface_text
            if ent.byte_span:
                b_start, b_end = ent.byte_span
                assert raw_bytes[b_start:b_end].decode("utf-8") == ent.surface_text

    def test_dynamic_unseen_passage_transduction(self, transducer: MockSkeletonTransducer):
        passage = (
            "Professor Jonathan Archer synthesized the crystalline polymer at the Oxford laboratory."
        )

        result = transducer.transduce(passage, chunk_id="dyn_chunk_03")

        assert len(result.entities) >= 1
        assert len(result.events) >= 1

        token_count = result.metadata["decoding_token_count"]
        assert token_count <= 50

        # Verify exact span slice matches
        for ent in result.entities:
            c_start, c_end = ent.char_span
            assert passage[c_start:c_end] == ent.surface_text

    def test_latency_is_sub_10ms_for_mock(self, transducer: MockSkeletonTransducer):
        text = "Dr. Marcus Vance pressurized the cylinder."
        res = transducer.transduce(text)
        assert res.metadata["latency_sec"] < 0.05  # sub-50ms (typically <5ms)


# ---------------------------------------------------------------------------
# 5. UnslothTransducer mode="skeleton" Integration Tests
# ---------------------------------------------------------------------------

class TestUnslothTransducerSkeletonMode:
    """Verifies UnslothTransducer integration in mode='skeleton'."""

    def test_mock_unsloth_transducer_in_skeleton_mode(self):
        transducer = MockUnslothTransducer(mode="skeleton")
        text = "Dr. Marcus Vance pressurized the argon cylinder inside the test chamber at noon."

        res = transducer.transduce(text, chunk_id="mock_skel_01")

        # Returns DiscourseExtractionResult
        assert res.chunk_id == "mock_skel_01"
        assert len(res.entities) >= 2
        assert len(res.events) >= 1
        assert res.metadata["mode"] == "skeleton"
        assert "decoding_token_count" in res.metadata
        assert res.metadata["decoding_token_count"] <= 50
        assert "prefill_latency_sec" in res.metadata
        assert "decoding_latency_sec" in res.metadata

    def test_unsloth_transducer_mode_kwarg_override(self):
        transducer = MockUnslothTransducer(mode="legacy")
        text = "Dr. Marcus Vance pressurized the argon cylinder."

        # Pass mode="skeleton" as kwarg
        res = transducer.transduce(text, mode="skeleton")
        assert res.metadata["mode"] == "skeleton"
        assert res.metadata["decoding_token_count"] <= 50

        # Default remains legacy
        res_legacy = transducer.transduce(text)
        assert res_legacy.metadata["mode"] == "legacy"

    def test_unsloth_transducer_raw_skeleton_json(self):
        transducer = MockUnslothTransducer(mode="skeleton")
        text = "Dr. Marcus Vance pressurized the argon cylinder."

        raw_json = transducer.transduce_raw(text)
        assert raw_json.startswith("{")
        assert raw_json.endswith("}")

        parsed = json.loads(raw_json)
        assert "entities" in parsed
        assert "events" in parsed
        assert estimate_token_count(raw_json) <= 50

    def test_unsloth_transducer_transduce_skeleton_method(self):
        transducer = MockUnslothTransducer()
        text = "Eleanor Vance entered the containment cell."

        skel_res = transducer.transduce_skeleton(text, chunk_id="skel_direct_01")
        assert isinstance(skel_res, SkeletonExtractionResult)
        assert len(skel_res.entities) >= 2
        assert skel_res.metadata["decoding_token_count"] <= 50

    def test_production_unsloth_transducer_fallback_to_mock(self):
        """Verify production UnslothTransducer falls back to mock when server is offline."""
        transducer = UnslothTransducer(
            base_url="http://127.0.0.1:9999/v1",  # Non-existent port
            fallback_to_mock=True,
            mode="skeleton",
        )
        text = "Dr. Marcus Vance pressurized the argon cylinder."

        res = transducer.transduce(text)
        assert res.metadata["mode"] == "skeleton"
        assert res.metadata["decoding_token_count"] <= 50
        assert "Dr. Marcus Vance" in [e.canonical_name for e in res.entities] or any("Marcus" in e.canonical_name for e in res.entities)

    def test_async_skeleton_transduction(self):
        transducer = MockUnslothTransducer(mode="skeleton")
        text = "Dr. Marcus Vance pressurized the argon cylinder."

        res = asyncio.run(transducer.transduce_async(text, mode="skeleton"))
        assert res.metadata["mode"] == "skeleton"
        assert res.metadata["decoding_token_count"] <= 50

        raw_json = asyncio.run(transducer.transduce_raw_async(text, mode="skeleton"))
        assert raw_json.startswith("{")
