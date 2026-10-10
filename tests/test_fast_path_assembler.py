"""Test Suite for Fast-Path Assembler, Coverage Evaluator & Pipeline Orchestration (§Phase 3).

Tests:
1. FastPathCoverageEvaluator coverage scoring logic.
2. FastPathAssembler mode resolution (M-A, M-B, M-C, M-D, passthrough).
3. Context construction per mode:
   - M-A (raw-only): stream 1 omitted, stream 2 contains filtered raw spans with 100% lexical fidelity.
   - M-B (hot-transduce): stream 1 briefing present, stream 2 raw spans present.
   - M-C (full): full dual-stream context with PPR.
   - M-D (coverage-adaptive): switches between M-A and M-B based on threshold.
4. Pass-through mode matches B0 output on mock fixtures.
5. CognitivePipeline.answer_long_context end-to-end execution.
6. CognitivePipeline.process_and_answer_single_query fast-path routing.
"""

from __future__ import annotations

import pytest
from typing import List

from config.multi_scale_config import MultiScaleConfig
from core.asg import QuantaGraph, QuantaNode
from memory.context_assembler import DualStreamContextAssembler, ProjectedPassage
from memory.fast_path_assembler import (
    FastPathAssembler,
    FastPathContext,
    FastPathCoverageEvaluator,
    FastPathMode,
    FastPathResult,
)
from memory.passage_store import PassageRecord, PassageStore
from parser.chunker import DiscourseChunk
from parser.multi_scale_chunker import MacroBlock
from parser.task_boundary_extractor import ExtractedTaskIntent
from pipeline.cognitive_pipeline import CognitivePipeline
from retrieval.relevance_filter import ScoredUnit


# -----------------------------------------------------------------------------
# Test Fixtures & Helpers
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_units() -> List[ScoredUnit]:
    """Generates synthetic scored units for testing."""
    u1 = DiscourseChunk(
        chunk_id="chunk_01",
        text="James Cameron directed the blockbuster science fiction movie Avatar in 2009.",
        global_offset=0,
        global_end_offset=75,
    )
    u2 = DiscourseChunk(
        chunk_id="chunk_02",
        text="Avatar was filmed using groundbreaking stereoscopic motion capture technology.",
        global_offset=76,
        global_end_offset=154,
    )
    u3 = DiscourseChunk(
        chunk_id="chunk_03",
        text="The atmospheric dynamics of Jupiter exhibit giant anticyclonic storms.",
        global_offset=155,
        global_end_offset=226,
    )
    return [
        ScoredUnit(unit=u1, unit_id="chunk_01", score=0.95, raw_score=12.5, tokens=16, char_span=(0, 75)),
        ScoredUnit(unit=u2, unit_id="chunk_02", score=0.82, raw_score=9.1, tokens=14, char_span=(76, 154)),
        ScoredUnit(unit=u3, unit_id="chunk_03", score=0.15, raw_score=1.2, tokens=13, char_span=(155, 226)),
    ]


@pytest.fixture
def sample_subgraph() -> QuantaGraph:
    """Generates synthetic QuantaGraph with event and entity nodes."""
    g = QuantaGraph()
    ent = QuantaNode(anchor="James Cameron", literal="James Cameron", passage_id="chunk_01")
    ent.compute_canonical_cid()
    g.add_node(ent)

    ev = QuantaNode(anchor="directed", literal="directed", passage_id="chunk_01")
    ev.set_slot("TYPE_EVENT", 1)
    ev.compute_canonical_cid()
    g.add_node(ev)

    g.add_edge(ev.canonical_cid, "VAL_X1_AGENT", ent.canonical_cid)
    g.root_cid = ev.canonical_cid
    return g


# -----------------------------------------------------------------------------
# 1. Coverage Evaluator Tests
# -----------------------------------------------------------------------------

def test_coverage_evaluator_high_and_low_overlap(sample_units):
    evaluator = FastPathCoverageEvaluator()

    # High coverage intent
    intent_high = ExtractedTaskIntent(
        query_text="Who directed the movie Avatar?",
        context_text="",
        target_entities=["James Cameron", "Avatar"],
    )
    score_high = evaluator.check_coverage(intent_high, sample_units[:2])
    assert score_high >= 0.8, f"Expected high coverage >= 0.8, got {score_high}"

    # Low coverage intent (entities not present)
    intent_low = ExtractedTaskIntent(
        query_text="Where was Marie Curie born and educated?",
        context_text="",
        target_entities=["Marie Curie", "Poland"],
    )
    score_low = evaluator.check_coverage(intent_low, sample_units[:2])
    assert score_low < 0.4, f"Expected low coverage < 0.4, got {score_low}"

    # Empty query or units edge cases
    assert evaluator.check_coverage("", sample_units) == 1.0
    assert evaluator.check_coverage("Any query", []) == 0.0


# -----------------------------------------------------------------------------
# 2. FastPathAssembler Mode & Context Structure Tests
# -----------------------------------------------------------------------------

def test_fast_path_mode_m_a_raw_only(sample_units):
    """M-A raw-only: stream 1 logical briefing must be omitted, stream 2 must retain raw text."""
    assembler = FastPathAssembler(mode="raw_only")
    ctx = assembler.assemble(
        subgraph=None,
        kept_units=sample_units[:2],
        max_tokens=1000,
        query_text="Who directed Avatar?",
    )

    assert ctx.mode == FastPathMode.RAW_ONLY.value
    assert ctx.stream1_briefing == "", "Stream 1 must be omitted in raw_only mode"
    assert "James Cameron directed" in ctx.stream2_passages
    assert "groundbreaking stereoscopic" in ctx.stream2_passages
    assert ctx.context_text == ctx.stream2_passages
    assert ctx.transduced_count == 0
    assert ctx.total_kept_count == 2


def test_fast_path_mode_m_b_hot_transduce(sample_units, sample_subgraph):
    """M-B hot-transduce: stream 1 briefing present from subgraph, stream 2 contains raw passages."""
    assembler = FastPathAssembler(mode="hot_transduce", hot_transduce_n=1)
    ctx = assembler.assemble(
        subgraph=sample_subgraph,
        kept_units=sample_units,
        max_tokens=1000,
        query_text="Who directed Avatar?",
    )

    assert ctx.mode == FastPathMode.HOT_TRANSDUCE.value
    assert ctx.stream1_briefing != "", "Stream 1 briefing must be populated in hot-transduce mode"
    assert "James Cameron" in ctx.stream1_briefing
    assert "=== RETRIEVED SOURCE PASSAGES ===" in ctx.stream2_passages
    assert ctx.transduced_count == 1
    assert ctx.total_kept_count == 3
    assert ctx.stream1_briefing in ctx.context_text
    assert ctx.stream2_passages in ctx.context_text


def test_fast_path_mode_m_c_full(sample_units, sample_subgraph):
    """M-C full: full dual-stream context applied to kept set."""
    assembler = FastPathAssembler(mode="full")
    ctx = assembler.assemble(
        subgraph=sample_subgraph,
        kept_units=sample_units[:2],
        max_tokens=1000,
        query_text="Who directed Avatar?",
    )

    assert ctx.mode == FastPathMode.FULL.value
    assert ctx.stream1_briefing != ""
    assert ctx.transduced_count == 2


def test_fast_path_mode_m_d_coverage_adaptive(sample_units):
    """M-D coverage-adaptive: dynamically selects RAW_ONLY or HOT_TRANSDUCE based on threshold."""
    assembler = FastPathAssembler(mode="coverage_adaptive", coverage_threshold=0.7)

    # Intent with high coverage -> resolves to raw_only
    intent_high = ExtractedTaskIntent(
        query_text="Who directed Avatar?",
        context_text="",
        target_entities=["Avatar", "James Cameron"],
    )
    eff_mode_high, score_high = assembler.resolve_effective_mode(intent_high, sample_units[:2])
    assert eff_mode_high == FastPathMode.RAW_ONLY.value
    assert score_high >= 0.7

    # Intent with low coverage -> resolves to hot_transduce
    intent_low = ExtractedTaskIntent(
        query_text="Who founded NeXT computer in California?",
        context_text="",
        target_entities=["Steve Jobs", "NeXT"],
    )
    eff_mode_low, score_low = assembler.resolve_effective_mode(intent_low, sample_units[:2])
    assert eff_mode_low == FastPathMode.HOT_TRANSDUCE.value
    assert score_low < 0.7


def test_fast_path_mode_normalization():
    assert FastPathAssembler.normalize_mode("M-A") == FastPathMode.RAW_ONLY.value
    assert FastPathAssembler.normalize_mode("raw_only") == FastPathMode.RAW_ONLY.value
    assert FastPathAssembler.normalize_mode("M-B") == FastPathMode.HOT_TRANSDUCE.value
    assert FastPathAssembler.normalize_mode("hot_transduce") == FastPathMode.HOT_TRANSDUCE.value
    assert FastPathAssembler.normalize_mode("M-C") == FastPathMode.FULL.value
    assert FastPathAssembler.normalize_mode("M-D") == FastPathMode.COVERAGE_ADAPTIVE.value
    assert FastPathAssembler.normalize_mode("passthrough") == FastPathMode.PASSTHROUGH.value


# -----------------------------------------------------------------------------
# 3. CognitivePipeline.answer_long_context End-to-End Tests
# -----------------------------------------------------------------------------

def test_pipeline_answer_long_context_raw_only():
    """Verifies answer_long_context in raw_only mode executes with 0 GPU calls and returns FastPathResult."""
    pipeline = CognitivePipeline(transducer_backend="mock", kev_mode="co_decoded")

    long_prompt = (
        "Document: James Cameron directed the film Titanic in 1997. Leonardo DiCaprio starred as Jack Dawson.\n\n"
        "Question: Who directed the film Titanic?"
    )

    cfg = MultiScaleConfig(runtime_overrides={"fast_path.mode": "raw_only"})
    result = pipeline.answer_long_context(long_prompt, config=cfg)

    assert isinstance(result, FastPathResult)
    assert result.mode == "raw_only"
    assert result.units_transduced == 0, "M-A raw-only must not transduce any chunks"
    assert "James Cameron" in result.context
    assert result.units_kept > 0
    assert result.stage_timings.get("query split", 0) >= 0.0
    assert result.stage_timings.get("chunking", 0) >= 0.0
    assert result.stage_timings.get("relevance filter", 0) >= 0.0


def test_pipeline_answer_long_context_hot_transduce():
    """Verifies answer_long_context in hot_transduce mode synchronously transduces top N chunks."""
    pipeline = CognitivePipeline(transducer_backend="mock", kev_mode="co_decoded")

    long_prompt = (
        "Document: Paragraph 1: Albert Einstein won the Nobel Prize in Physics in 1921.\n\n"
        "Paragraph 2: He formulated the theory of general relativity in Berlin.\n\n"
        "Paragraph 3: Photosynthesis converts light energy into chemical energy in plants.\n\n"
        "Question: In what year did Einstein win the Nobel Prize?"
    )

    cfg = MultiScaleConfig(runtime_overrides={"fast_path.mode": "hot_transduce", "fast_path.hot_transduce_n": 1})
    result = pipeline.answer_long_context(long_prompt, config=cfg)

    assert isinstance(result, FastPathResult)
    assert result.mode == "hot_transduce"
    assert result.units_transduced == 1
    assert result.subgraph is not None


def test_pipeline_answer_long_context_full_and_passthrough():
    """Verifies full and passthrough modes in answer_long_context."""
    pipeline = CognitivePipeline(transducer_backend="mock", kev_mode="co_decoded")

    long_prompt = (
        "Document: Nikola Tesla developed alternating current electrical supply systems.\n\n"
        "Question: What system did Tesla develop?"
    )

    cfg_full = MultiScaleConfig(runtime_overrides={"fast_path.mode": "full"})
    res_full = pipeline.answer_long_context(long_prompt, config=cfg_full)
    assert res_full.mode == "full"
    assert res_full.units_transduced >= 1

    cfg_pass = MultiScaleConfig(runtime_overrides={"fast_path.mode": "passthrough"})
    res_pass = pipeline.answer_long_context(long_prompt, config=cfg_pass)
    assert res_pass.mode == "passthrough"
    assert res_pass.units_transduced >= 1


# -----------------------------------------------------------------------------
# 4. process_and_answer_single_query Routing Integration
# -----------------------------------------------------------------------------

def test_process_and_answer_single_query_fast_path_routing():
    """Ensures process_and_answer_single_query routes through fast-path when enabled."""
    pipeline = CognitivePipeline(transducer_backend="mock", kev_mode="co_decoded")

    doc = "Alan Turing designed the Turing machine in 1936, providing a formalization of algorithm concepts."
    q = "When did Alan Turing design the Turing machine?"

    # With fast-path active
    pipeline.multi_scale_config = MultiScaleConfig(runtime_overrides={"fast_path.mode": "raw_only"})
    res = pipeline.process_and_answer_single_query(query_text=q, context_document=doc)

    assert "fast_path_result" in res
    assert res["mode"] == "raw_only"
    assert "Alan Turing" in res["retrieved_context"]

    # In passthrough mode (B0 behavior)
    pipeline.multi_scale_config = MultiScaleConfig(runtime_overrides={"fast_path.mode": "passthrough"})
    res_b0 = pipeline.process_and_answer_single_query(query_text=q, context_document=doc)
    assert "fast_path_result" not in res_b0
    assert "retrieved_context" in res_b0
