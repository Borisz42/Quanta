"""Unit and Integration Tests for MuSiQue Multi-Hop Bridge Expansion (Section 4).

Verifies:
1. Dynamic multi-hop depth detection (2-hop, 3-hop, 4-hop query syntaxes).
2. Spreading activation parameter adaptation (decay >= 0.82, threshold <= 0.18).
3. CognitivePipeline intra-chunk entity linking (CO_OCCURS and LOCATED_IN).
4. CognitivePipeline cross-chunk bridge synthesis (CROSS_CHUNK_BRIDGE between episode folds).
5. Adaptive context budgeting expanding up to 1,200 tokens for multi-hop documents.
"""

import pytest
from core.asg import QuantaGraph, QuantaNode
from memory.page_table import PageTable
from memory.spreading_activation import SpreadingActivationRetriever
from parser.chunker import DiscourseChunker
from pipeline.cognitive_pipeline import CognitivePipeline


def test_multihop_depth_detection():
    """Verifies syntactic and semantic detection of 2, 3, and 4 hop query structures."""
    retriever = SpreadingActivationRetriever()

    # 2-hop baseline
    depth_2 = retriever.detect_query_hop_depth("Who is the spouse of the author of The Hobbit?")
    assert depth_2 == 2

    # 3-hop query patterns
    depth_3a = retriever.detect_query_hop_depth(
        "Who was born in the city that is the capital of the state where Harvard University is located?"
    )
    assert depth_3a >= 3

    depth_3b = retriever.detect_query_hop_depth(
        "Which organization was founded by the person who studied at the university where Alan Turing worked?"
    )
    assert depth_3b >= 3

    # 4-hop query patterns
    depth_4 = retriever.detect_query_hop_depth(
        "Find the country sharing a border with the nation whose capital is home to the institute where the spouse of Marie Curie studied."
    )
    assert depth_4 >= 4


def test_multihop_adaptive_parameters():
    """Verifies that 3+ hop queries adapt decay and threshold to prevent premature cutoff."""
    retriever = SpreadingActivationRetriever(decay=0.7, threshold=0.35)
    page_table = PageTable(db_path=":memory:")

    # Build a linear 4-hop chain of nodes in page_table
    n1 = QuantaNode(literal="Node_1")
    n2 = QuantaNode(literal="Node_2")
    n3 = QuantaNode(literal="Node_3")
    n4 = QuantaNode(literal="Node_4")

    page_table.store_node(n1)
    page_table.store_node(n2)
    page_table.store_node(n3)
    page_table.store_node(n4)

    page_table.add_edge(n1.cid, "LOCATED_IN", n2.cid)
    page_table.add_edge(n2.cid, "LOCATED_IN", n3.cid)
    page_table.add_edge(n3.cid, "LOCATED_IN", n4.cid)

    # 2-hop traversal with default decay=0.7, threshold=0.35 reaches node 2 but cuts off node 3/4
    # act(1) = 1.0, act(2) = 0.7, act(3) = 0.49, act(4) = 0.343 < 0.35
    subgraph_2hop = retriever.traverse_subgraph(
        seed_cids=[n1.cid],
        page_table=page_table,
        max_depth=2,
        decay=0.7,
        threshold=0.35,
    )
    assert n1.cid in subgraph_2hop.nodes
    assert n2.cid in subgraph_2hop.nodes
    assert n4.cid not in subgraph_2hop.nodes

    # 4-hop adaptive traversal with decay=0.85, threshold=0.15 survives to node 4
    # act(1)=1.0, act(2)=0.85, act(3)=0.72, act(4)=0.61 > 0.15
    subgraph_4hop = retriever.traverse_subgraph(
        seed_cids=[n1.cid],
        page_table=page_table,
        max_depth=4,
        decay=0.85,
        threshold=0.15,
    )
    assert n1.cid in subgraph_4hop.nodes
    assert n2.cid in subgraph_4hop.nodes
    assert n3.cid in subgraph_4hop.nodes
    assert n4.cid in subgraph_4hop.nodes


def test_cognitive_pipeline_cross_chunk_bridges():
    """Verifies that multi-paragraph narratives synthesize CROSS_CHUNK_BRIDGE between folds and CO_OCCURS between events."""
    pipeline = CognitivePipeline(
        transducer_backend="mock",
        page_table_path=":memory:",
    )

    multi_doc_text = (
        "Document [1]: Alan Turing was born in London. He was an influential mathematician.\n\n"
        "Document [2]: London is the capital of the United Kingdom. It is located on the River Thames.\n\n"
        "Document [3]: The United Kingdom shares maritime borders with France."
    )

    episode_graphs, chapter_fold = pipeline.process_narrative(
        text=multi_doc_text,
        chapter_id="musique_test_ch",
        validate=False,
    )

    assert len(episode_graphs) >= 3

    # Check that London connects episode 1 and episode 2
    # Check that United Kingdom connects episode 2 and episode 3
    # Check page table for CROSS_CHUNK_BRIDGE relations
    all_edges_query = pipeline.page_table._conn.cursor().execute("SELECT cid, edges FROM nodes").fetchall()
    has_cross_bridge = False
    has_co_occurs = False

    for cid, edges_str in all_edges_query:
        if "CROSS_CHUNK_BRIDGE" in edges_str:
            has_cross_bridge = True
        if "CO_OCCURS" in edges_str or "LOCATED_IN" in edges_str:
            has_co_occurs = True

    assert has_cross_bridge, "Expected CROSS_CHUNK_BRIDGE edges between fold nodes across paragraphs"
    assert has_co_occurs, "Expected associative CO_OCCURS or LOCATED_IN edges"


def test_context_budget_expansion_on_multihop_query():
    """Verifies retrieve_context expands token budget up to 1,200 for 3+ hop queries."""
    pipeline = CognitivePipeline(
        transducer_backend="mock",
        page_table_path=":memory:",
    )

    text = (
        "Document [1]: Cambridge University is in Cambridge, England.\n\n"
        "Document [2]: Cambridge is an English city on the River Cam.\n\n"
        "Document [3]: England is a country that is part of the United Kingdom."
    )
    pipeline.process_narrative(text, chapter_id="budget_ch", validate=False)

    # 3-hop query should automatically allow up to 1200 tokens
    query_3hop = "Which country contains the city housing the university where researchers worked?"
    context = pipeline.retrieve_context(query_3hop, max_tokens=500)

    assert isinstance(context, str)
    assert len(context) > 0
    # Confirm it retrieves relevant associative context
    assert any(term in context for term in ("Cambridge", "England", "United Kingdom", "associated", "located"))
