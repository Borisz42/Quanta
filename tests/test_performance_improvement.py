"""Verification tests for QUANTA Performance Improvement Plan:
Multi-Hop Retrieval & Adaptive Context Budgeting.
"""

from __future__ import annotations

import pytest

from parser.chunker import DiscourseChunk
from parser.task_boundary_extractor import (
    ExtractedTaskIntent,
    QUESTION_STOPWORDS,
    TaskBoundaryExtractor,
    clean_core_query,
    extract_target_entities,
)
from retrieval.relevance_filter import RelevanceFilter


def test_musique_query_sanitization():
    """Verify clean_core_query strips directive boilerplate while preserving core multi-hop question."""
    raw_prompt = (
        "Where do Greyhound buses leave from in the city where Arna Selznick's employer is headquartered?\n\n"
        "Instructions:\n"
        "1. In 2 to 3 concise sentences, explain the factual bridge connecting the relevant documents. Do not summarize or list irrelevant documents.\n"
        "2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'."
    )
    cleaned = clean_core_query(raw_prompt)
    assert cleaned == "Where do Greyhound buses leave from in the city where Arna Selznick's employer is headquartered?"
    assert "Instructions:" not in cleaned
    assert "factual bridge" not in cleaned
    assert "Answer:" not in cleaned


def test_stopword_filtering_excludes_false_overlap():
    """Verify words like 'leave', 'do', 'where', 'from' are recognized as stopwords."""
    for sw in ("leave", "leaves", "do", "does", "where", "from", "in", "the", "is", "all"):
        assert sw in QUESTION_STOPWORDS, f"Expected '{sw}' to be in QUESTION_STOPWORDS"


def test_musique_gold_chunks_rank_top():
    """Verify that gold evidence chunks for MuSiQue rank #1 and #2 over distractor chunks."""
    raw_prompt = (
        "Where do Greyhound buses leave from in the city where Arna Selznick's employer is headquartered?\n\n"
        "Instructions:\n"
        "1. In 2 to 3 concise sentences, explain the factual bridge connecting the relevant documents. Do not summarize or list irrelevant documents.\n"
        "2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'."
    )

    # Gold chunks
    c18 = DiscourseChunk(
        chunk_id="chunk_0018",
        text="Arna Selznick directed Nelvana's 1985 animated film The Care Bears Movie. Nelvana is an animation studio headquartered in Toronto.",
        global_offset=0,
        global_end_offset=128,
        word_count=21,
        token_count_estimate=28,
    )
    c14 = DiscourseChunk(
        chunk_id="chunk_0014",
        text="Toronto Coach Terminal was the central bus station for inter-city services in Toronto, Ontario, leased to Greyhound Canada.",
        global_offset=129,
        global_end_offset=248,
        word_count=18,
        token_count_estimate=24,
    )

    # Distractor chunks that share unigrams like 'leave', 'employer', 'city'
    d1 = DiscourseChunk(
        chunk_id="distractor_0001",
        text="Maternity leave policies in the United Kingdom allow eligible employees to take statutory maternity leave from their employer after childbirth.",
        global_offset=249,
        global_end_offset=380,
        word_count=19,
        token_count_estimate=25,
    )
    d2 = DiscourseChunk(
        chunk_id="distractor_0002",
        text="The statutory school leaving age determines the minimum age at which a young person can leave full-time secondary education in England.",
        global_offset=381,
        global_end_offset=510,
        word_count=20,
        token_count_estimate=27,
    )
    d3 = DiscourseChunk(
        chunk_id="distractor_0003",
        text="A national UK employer must register with HM Revenue and Customs when operating business offices across any major commercial city.",
        global_offset=511,
        global_end_offset=640,
        word_count=19,
        token_count_estimate=25,
    )

    extractor = TaskBoundaryExtractor(strategy="QE-B")
    intent = extractor.extract(raw_prompt)

    assert "Arna Selznick" in intent.target_entities
    assert "Greyhound" in intent.target_entities

    rf = RelevanceFilter(strategy="F-A")
    scored = rf.score(intent, [d1, d2, d3, c14, c18])

    # Rank units by score descending
    ranked = sorted(scored, key=lambda s: s.score, reverse=True)
    top_ids = [s.unit_id for s in ranked[:2]]

    assert "chunk_0018" in top_ids, f"Expected chunk_0018 in top 2, got {top_ids}"
    assert "chunk_0014" in top_ids, f"Expected chunk_0014 in top 2, got {top_ids}"

    # Verify gold scores are strictly higher than all distractors
    c18_score = next(s.score for s in scored if s.unit_id == "chunk_0018")
    c14_score = next(s.score for s in scored if s.unit_id == "chunk_0014")
    d_max_score = max(s.score for s in scored if s.unit_id.startswith("distractor"))

    assert c18_score > d_max_score, f"chunk_0018 ({c18_score}) must exceed distractor max ({d_max_score})"
    assert c14_score > d_max_score, f"chunk_0014 ({c14_score}) must exceed distractor max ({d_max_score})"


def test_variable_tracking_elastic_budget_expansion():
    """Verify that multi-needle transactions with identical technical variable IDs (ACC-9042)
    elastically expand the 1,500 token budget to keep all 4 transaction chunks (1,829 tokens total).
    """
    query = "What is the net balance change for Account ACC-9042 across all recorded transactions?"

    # 4 transactions for ACC-9042 totaling 1,829 tokens
    t1 = DiscourseChunk(
        chunk_id="chunk_0001",
        text="Audit Record: Account ACC-9042 recorded a credit deposit of 1500 credits. " * 35,
        global_offset=0,
        global_end_offset=2000,
        word_count=350,
        token_count_estimate=466,
    )
    t2 = DiscourseChunk(
        chunk_id="chunk_0003",
        text="Audit Record: Account ACC-9042 recorded a debit transfer of 350 credits. " * 35,
        global_offset=2001,
        global_end_offset=4000,
        word_count=345,
        token_count_estimate=459,
    )
    t3 = DiscourseChunk(
        chunk_id="chunk_0005",
        text="Audit Record: Account ACC-9042 recorded a credit deposit of 600 credits. " * 35,
        global_offset=4001,
        global_end_offset=6000,
        word_count=346,
        token_count_estimate=461,
    )
    t4 = DiscourseChunk(
        chunk_id="chunk_0006",
        text="Audit Record: Account ACC-9042 recorded a debit fee of 50 credits. " * 35,
        global_offset=6001,
        global_end_offset=8000,
        word_count=332,
        token_count_estimate=443,
    )

    # Distractor transactions for different accounts
    d_other1 = DiscourseChunk(
        chunk_id="chunk_0002",
        text="Audit Record: Account ACC-3110 recorded a debit payment of 400 credits. " * 30,
        global_offset=8001,
        global_end_offset=9500,
        word_count=300,
        token_count_estimate=400,
    )
    d_other2 = DiscourseChunk(
        chunk_id="chunk_0004",
        text="Audit Record: Account ACC-7801 recorded a credit deposit of 900 credits. " * 30,
        global_offset=9501,
        global_end_offset=11000,
        word_count=300,
        token_count_estimate=400,
    )

    all_chunks = [t1, d_other1, t2, d_other2, t3, t4]
    target_tokens_sum = t1.token_count_estimate + t2.token_count_estimate + t3.token_count_estimate + t4.token_count_estimate
    assert target_tokens_sum == 1829, f"Target tokens sum is {target_tokens_sum}"

    # Filter with rigid base budget of 1500 tokens
    rf = RelevanceFilter(strategy="F-A", keep_budget_tokens=1500)
    scored = rf.score(query, all_chunks)
    kept, discarded = rf.apply_keep_policy(scored)

    kept_ids = {u.unit_id for u in kept}
    # All 4 target chunks must be retained despite 1829 > 1500
    assert "chunk_0001" in kept_ids, "chunk_0001 (+1500 credits) must not be dropped!"
    assert "chunk_0003" in kept_ids, "chunk_0003 (-350 credits) must not be dropped!"
    assert "chunk_0005" in kept_ids, "chunk_0005 (+600 credits) must not be dropped!"
    assert "chunk_0006" in kept_ids, "chunk_0006 (-50 credits) must not be dropped!"

    total_kept_tokens = sum(u.tokens for u in kept)
    assert total_kept_tokens >= 1829, f"Expected at least 1829 tokens, got {total_kept_tokens}"

    # Distractor chunks should be pruned
    assert "chunk_0002" not in kept_ids
    assert "chunk_0004" not in kept_ids


def test_answer_long_context_musique_end_to_end():
    """Verify answer_long_context retrieves gold evidence chunks for MuSiQue."""
    from pipeline.cognitive_pipeline import CognitivePipeline
    pipeline = CognitivePipeline(transducer_backend="mock")

    doc18 = "Paragraph [18]: Arna Selznick directed Nelvana's 1985 animated film The Care Bears Movie. Nelvana is an animation studio headquartered in Toronto."
    doc14 = "Paragraph [14]: Toronto Coach Terminal was the central bus station for inter-city services in Toronto, Ontario, leased to Greyhound Canada."
    doc_distractors = "\n\n".join([
        "Paragraph [1]: Maternity leave policies in the United Kingdom allow eligible employees to take statutory maternity leave from their employer.",
        "Paragraph [2]: The statutory school leaving age determines the minimum age at which a young person can leave full-time secondary education.",
        "Paragraph [3]: A national UK employer must register with HM Revenue and Customs when operating business offices across any commercial city.",
    ])

    full_prompt = (
        "Where do Greyhound buses leave from in the city where Arna Selznick's employer is headquartered?\n\n"
        "Instructions:\n"
        "1. In 2 to 3 concise sentences, explain the factual bridge connecting the relevant documents.\n"
        "2. Conclude immediately on a new line in the exact format: 'Answer: <final answer>'.\n\n"
        f"Context:\n{doc_distractors}\n\n{doc18}\n\n{doc14}"
    )

    res = pipeline.answer_long_context(full_prompt, max_context_tokens=2500)
    ctx = res.context
    assert "Arna Selznick" in ctx
    assert "Toronto Coach Terminal" in ctx


def test_answer_long_context_variable_tracking_end_to_end():
    """Verify answer_long_context retains all 4 transaction events for ACC-9042."""
    from pipeline.cognitive_pipeline import CognitivePipeline
    pipeline = CognitivePipeline(transducer_backend="mock")

    event1 = "ENTERPRISE TELEMETRY EVENT: Audit Record: Account ACC-9042 recorded a credit deposit of 1500 credits. " * 35
    event2 = "ENTERPRISE TELEMETRY EVENT: Audit Record: Account ACC-3110 recorded a debit payment of 400 credits. " * 30
    event3 = "ENTERPRISE TELEMETRY EVENT: Audit Record: Account ACC-9042 recorded a debit transfer of 350 credits. " * 35
    event4 = "ENTERPRISE TELEMETRY EVENT: Audit Record: Account ACC-7801 recorded a credit deposit of 900 credits. " * 30
    event5 = "ENTERPRISE TELEMETRY EVENT: Audit Record: Account ACC-9042 recorded a credit deposit of 600 credits. " * 35
    event6 = "ENTERPRISE TELEMETRY EVENT: Audit Record: Account ACC-9042 recorded a debit fee of 50 credits. " * 35

    doc = f"{event1}\n\n{event2}\n\n{event3}\n\n{event4}\n\n{event5}\n\n{event6}"
    prompt = (
        f"Context:\n{doc}\n\n"
        "Question: What is the net balance change for Account ACC-9042 across all recorded transactions?"
    )

    res = pipeline.answer_long_context(prompt, max_context_tokens=2500)
    ctx = res.context
    assert "1500 credits" in ctx, "Event 1 (+1500) must be present in retrieved context"
    assert "350 credits" in ctx, "Event 3 (-350) must be present in retrieved context"
    assert "600 credits" in ctx, "Event 5 (+600) must be present in retrieved context"
    assert "50 credits" in ctx, "Event 6 (-50) must be present in retrieved context"

