"""Unit tests for HierarchicalChunker and MacroBlock (§2.1)."""

import pytest

from config.multi_scale_config import MultiScaleConfig
from parser.chunker import DiscourseChunk
from parser.multi_scale_chunker import HierarchicalChunker, MacroBlock


def test_macro_block_dataclass_and_serialization():
    micro = DiscourseChunk(
        chunk_id="chunk_0000",
        text="The quantum circuit operates at cryogenic temperatures.",
        global_offset=0,
        global_end_offset=56,
        word_count=7,
        token_count_estimate=9,
        parent_macro_id="macro_0000",
    )
    macro = MacroBlock(
        macro_id="macro_0000",
        char_span=(0, 56),
        text="The quantum circuit operates at cryogenic temperatures.",
        micro_chunks=[micro],
        concept_codes=[12, 45, 99],
        surface_entities=["Quantum Circuit"],
        token_count_estimate=9,
    )

    data = macro.to_dict()
    assert data["macro_id"] == "macro_0000"
    assert data["char_span"] == [0, 56]
    assert len(data["micro_chunks"]) == 1
    assert data["micro_chunks"][0]["parent_macro_id"] == "macro_0000"
    assert data["concept_codes"] == [12, 45, 99]
    assert data["surface_entities"] == ["Quantum Circuit"]

    restored = MacroBlock.from_dict(data)
    assert restored.macro_id == macro.macro_id
    assert restored.char_span == macro.char_span
    assert restored.text == macro.text
    assert len(restored.micro_chunks) == 1
    assert restored.micro_chunks[0].chunk_id == "chunk_0000"
    assert restored.micro_chunks[0].parent_macro_id == "macro_0000"
    assert restored.concept_codes == [12, 45, 99]
    assert restored.surface_entities == ["Quantum Circuit"]


def test_hierarchical_chunker_empty_and_short():
    chunker = HierarchicalChunker(macro_target_tokens=500, micro_target_words=100)

    macros, micros = chunker.chunk_document("")
    assert macros == []
    assert micros == []

    short_text = "Albert Einstein formulated the theory of general relativity in Berlin, Germany."
    macros, micros = chunker.chunk_document(short_text)
    assert len(micros) >= 1
    assert len(macros) == 1
    assert macros[0].macro_id == "macro_0000"
    assert macros[0].micro_chunks[0].parent_macro_id == "macro_0000"
    assert "Albert Einstein" in macros[0].surface_entities or "Berlin" in macros[0].surface_entities


def test_hierarchical_chunker_multi_macro_partition():
    # Construct a multi-paragraph long document
    paragraphs = []
    for i in range(12):
        p = (
            f"The thermodynamic analysis of unit {i + 1} for closed-cycle gas turbines exhibits stability. "
            f"Superconducting magnetic coils maintained at 4.2 Kelvin stabilize the plasma torus. "
            f"Researchers at Laboratory {chr(65 + i)} evaluated the heat dissipation under high vacuum. "
            f"The pressure gradient across the containment boundary remained strictly below 10 millipascals. "
            f"Detailed telemetry confirmed monotonic temperature equilibrium across all sensor channels."
        )
        paragraphs.append(p)
    doc = "\n\n".join(paragraphs)

    # Use a small macro target to force multiple MacroBlocks
    chunker = HierarchicalChunker(macro_target_tokens=150, micro_target_words=50)
    macros, micros = chunker.chunk_document(doc)

    assert len(micros) > 1
    assert len(macros) > 1

    # Verify all micro chunks have valid parent_macro_id matching a MacroBlock
    macro_ids = {m.macro_id for m in macros}
    for micro in micros:
        assert micro.parent_macro_id is not None
        assert micro.parent_macro_id in macro_ids

    # Verify spans and text continuity
    for m in macros:
        assert m.char_span[0] < m.char_span[1]
        assert len(m.micro_chunks) >= 1
        assert m.micro_chunks[0].global_offset == m.char_span[0]
        assert m.micro_chunks[-1].global_end_offset == m.char_span[1]

    # Verify profiling stats
    stats = chunker.profiling_stats()
    assert "total_profiling_time_ms" in stats
    assert stats["total_concept_queries"] >= 0


def test_hierarchical_chunker_reads_config():
    config = MultiScaleConfig(runtime_overrides={
        "chunker.macro_target_tokens": 800,
        "chunker.micro_target_words": 180,
    })
    chunker = HierarchicalChunker(config=config)
    assert chunker.macro_target_tokens == 800
    assert chunker.micro_target_words == 180
