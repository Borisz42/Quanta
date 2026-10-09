"""Tests for Long-Context Benchmark Builder and Telemetry (§0.2).

Verifies:
1. Determinism: identical seed produces bit-identical files.
2. Gold-position bookkeeping: gold paragraphs and needles reside at recorded positions.
3. End-to-end mock pipeline execution: ingestion and dual-stream retrieval on long context.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.build_long_context_bench import LongContextBenchBuilder, LongContextItem
from pipeline.cognitive_pipeline import CognitivePipeline
from pipeline.tracer import PipelineExecutionTracer


def test_builder_determinism(tmp_path: Path):
    """Running builder with identical seed produces bit-identical output splits."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    builder1 = LongContextBenchBuilder(seed=42)
    items1 = builder1.build_dataset(length_grid=[2000, 4000], samples_per_cell=2)
    dev1, test1 = builder1.write_splits(items1, dir1)

    builder2 = LongContextBenchBuilder(seed=42)
    items2 = builder2.build_dataset(length_grid=[2000, 4000], samples_per_cell=2)
    dev2, test2 = builder2.write_splits(items2, dir2)

    with open(dev1, "r", encoding="utf-8") as f1, open(dev2, "r", encoding="utf-8") as f2:
        assert f1.read() == f2.read()

    with open(test1, "r", encoding="utf-8") as f1, open(test2, "r", encoding="utf-8") as f2:
        assert f1.read() == f2.read()


def test_gold_position_bookkeeping(tmp_path: Path):
    """Verifies that gold positions correctly point to the paragraphs containing target evidence."""
    builder = LongContextBenchBuilder(seed=123)
    items = builder.build_dataset(length_grid=[2000], samples_per_cell=2)

    for itm in items:
        # Context is split into numbered paragraphs/sections/blocks
        sections = itm.context.split("\n\n")

        assert len(itm.gold_positions) > 0, f"Item {itm.id} has no gold positions recorded"

        if itm.task_family == "niah":
            needle_text = itm.metadata.get("needle_text", "")
            for pos in itm.gold_positions:
                assert pos < len(sections), f"Position {pos} out of range in {itm.id}"
                sec_text = sections[pos]
                assert needle_text in sec_text, (
                    f"Needle '{needle_text}' not found in section {pos} of {itm.id}:\n{sec_text}"
                )
                assert itm.gold_answer.lower() in sec_text.lower()

        elif itm.task_family == "musique":
            found_golds = 0
            for pos in itm.gold_positions:
                assert pos < len(sections), f"Position {pos} out of range in {itm.id}"
                sec_text = sections[pos]
                for g in itm.gold_passages:
                    if g in sec_text:
                        found_golds += 1
                        break
            assert found_golds == len(itm.gold_passages), (
                f"Item {itm.id}: expected {len(itm.gold_passages)} gold passages, found {found_golds}"
            )

        elif itm.task_family == "babilong":
            for pos in itm.gold_positions:
                assert pos < len(sections), f"Position {pos} out of range in {itm.id}"
                sec_text = sections[pos]
                assert "State Update:" in sec_text


def test_mock_pipeline_end_to_end_on_bench_sample():
    """Harness runs end-to-end in mock mode with stage telemetry recording."""
    builder = LongContextBenchBuilder(seed=42)
    items = builder.build_musique_long(target_tokens=2000, count=1)
    assert len(items) == 1
    item = items[0]

    tracer = PipelineExecutionTracer.get_instance()
    tracer.reset()

    pipeline = CognitivePipeline(
        transducer_backend="mock",
        kev_mode="co_decoded",
        tracer=tracer,
    )

    # 1. Ingest document
    graph = pipeline.ingest_document(item.context, doc_id=item.id, validate=False)
    assert graph is not None

    # 2. Query memory
    ctx = pipeline.query_memory(item.prompt, format="dual_stream", max_tokens=1000)
    assert ctx is not None
    assert hasattr(ctx, "full_context")
    assert len(ctx.full_context) > 0

    # 3. Telemetry verification
    stage_timings = tracer.get_stage_timings()
    assert "transduction" in stage_timings
    assert "Kev" in stage_timings
    assert "PPR" in stage_timings
    assert "context assembly" in stage_timings

    # All stage durations must be positive
    for st, dur in stage_timings.items():
        assert dur >= 0.0, f"Stage {st} duration negative: {dur}"
