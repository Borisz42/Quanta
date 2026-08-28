"""Unit tests for ASG visualizer, ASCII art generation, and graph exports."""

import pytest
from core.asg import QuantaGraph, QuantaNode
from core.types import QuantaVector, QuaternaryValue
from visualization.asg_visualizer import ASGVisualizer


def test_asg_visualizer_empty_graph():
    graph = QuantaGraph()
    tree = ASGVisualizer.render_ascii_tree(graph)
    assert "[Empty Graph]" in tree
    
    table = ASGVisualizer.render_node_details_table(graph)
    assert "No nodes in graph." in table
    
    mermaid = ASGVisualizer.render_mermaid(graph)
    assert "Empty Graph" in mermaid


def test_asg_visualizer_connected_nodes():
    graph = QuantaGraph()
    
    root_node = QuantaNode(
        vector=QuantaVector({"NSM_DO": QuaternaryValue.TRUE, "VAL_X1_AGENT": QuaternaryValue.TRUE}),
        anchor="wn:chase.v.01",
        literal="chased",
    )
    dog_node = QuantaNode(
        vector=QuantaVector({"TYPE_ANIMATE": QuaternaryValue.TRUE}),
        anchor="wn:dog.n.01",
        literal="dog",
    )
    cat_node = QuantaNode(
        vector=QuantaVector({"TYPE_ANIMATE": QuaternaryValue.TRUE}),
        anchor="wn:cat.n.01",
        literal="cat",
    )
    
    graph.add_node(root_node, set_as_root=True)
    graph.add_edge(root_node, "VAL_X1_AGENT", dog_node)
    graph.add_edge(root_node, "VAL_X2_PATIENT", cat_node)
    
    tree_unicode = ASGVisualizer.render_ascii_tree(graph, style="unicode")
    assert "wn:chase.v.01" in tree_unicode
    assert "(VAL_X1_AGENT)" in tree_unicode
    assert "(VAL_X2_PATIENT)" in tree_unicode
    assert "wn:dog.n.01" in tree_unicode
    assert "wn:cat.n.01" in tree_unicode
    
    tree_ascii = ASGVisualizer.render_ascii_tree(graph, style="ascii")
    assert "wn:chase.v.01" in tree_ascii
    
    table = ASGVisualizer.render_node_details_table(graph)
    assert "wn:chase.v.01" in table
    assert "VAL_X1_AGENT" in table
    assert "TRUE (1)" in table
    
    mermaid = ASGVisualizer.render_mermaid(graph)
    assert "flowchart TD" in mermaid
    assert "-->|VAL_X1_AGENT|" in mermaid
    assert "-->|VAL_X2_PATIENT|" in mermaid
    
    md_block = ASGVisualizer.format_translation_block(
        example_id=1,
        source_text="A dog chased a cat.",
        target_text="A dog chased a cat.",
        source_modality="english",
        target_modality="english",
        graph=graph,
        as_markdown=True,
    )
    assert "### Example 01: ENGLISH → ENGLISH" in md_block
    assert "```mermaid" in md_block
    
    txt_block = ASGVisualizer.format_translation_block(
        example_id=1,
        source_text="A dog chased a cat.",
        target_text="A dog chased a cat.",
        source_modality="english",
        target_modality="english",
        graph=graph,
        as_markdown=False,
    )
    assert "EXAMPLE 01 [ENGLISH -> ENGLISH]" in txt_block
